# -*- coding: utf-8 -*-
"""
Tolerant mapper that guarantees 4 channels (RGB + Thermal) and aligned sizes.

- Reads RGB strictly as 3-channel (drops alpha if present).
- Loads thermal if available; otherwise uses zeros.
- If thermal has multiple planes, reduces to a single plane (averages).
- Resizes thermal to match RGB HxW BEFORE concatenation.
- Concatenates to HxWx4, then runs augs, converts to CHW float32 in [0,1].
- Validates final channel count; prints offending file name and raises if != 4.
- Converts ALL boxes to XYXY_ABS (handles rotated 5-dim boxes) and asserts 4-dim.

Extras in this version:
- Looks up decoded thermal in sibling 'thermal' directory under the same split
  (e.g., data/train/thermal/*.tif or *.tiff), robust to Roboflow ".rf.*" suffix.
- Counts how many samples actually used .tif/.tiff as band-4, logs periodic totals,
  and prints a final summary at process exit.

It also transforms annotations in the standard Detectron2 way.
"""

import os
import glob
import numpy as np
import cv2
import torch
from collections import Counter
import atexit

from typing import Any, Dict, Optional

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import BoxMode
from detectron2.data.detection_utils import (
    transform_instance_annotations,
    annotations_to_instances,
    filter_empty_instances,
)

# Optional: use loguru if available; otherwise fall back to print()
try:
    from loguru import logger as log
except Exception:
    class _P:  # minimal shim
        def info(self, *a, **k): print(*a, flush=True)
        def warning(self, *a, **k): print(*a, flush=True)
        def error(self, *a, **k): print(*a, flush=True)
    log = _P()

# ------------------------
# Thermal usage accounting
# ------------------------
THERM_COUNTS = Counter()  # keys: "tif", "other", "missing"

@atexit.register
def _print_thermal_counts():
    tot = sum(THERM_COUNTS.values())
    try:
        log.info(
            f"UI:INFO:train: Thermal usage summary: "
            f"tif={THERM_COUNTS['tif']} "
            f"other={THERM_COUNTS['other']} "
            f"missing={THERM_COUNTS['missing']} "
            f"total={tot}"
        )
    except Exception:
        print(
            f"UI:INFO:train: Thermal usage summary: "
            f"tif={THERM_COUNTS['tif']} "
            f"other={THERM_COUNTS['other']} "
            f"missing={THERM_COUNTS['missing']} "
            f"total={tot}",
            flush=True,
        )

# ------------------------
# I/O helpers
# ------------------------
def _read_rgb_strict_rgb(path: str) -> np.ndarray:
    """Always return HxWx3 uint8 RGB, dropping alpha or other channels."""
    rgb = utils.read_image(path, format="RGB")  # guarantees 3 channels
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise RuntimeError(f"RGB loader produced shape {rgb.shape} for {path}; expected HxWx3.")
    return rgb


def _read_thermal_best_effort(path: str) -> Optional[np.ndarray]:
    """
    Try to read a thermal image/array. Returns HxW float32 if possible, else None.
    Handles .npy/.npz or typical image formats (.png/.jpg/.tif/.tiff).
    If multi-channel, reduces to 1 plane via mean().
    Normalizes to [0,1] using max value if >1.
    """
    try:
        if path.lower().endswith(".npy"):
            arr = np.load(path)
        elif path.lower().endswith(".npz"):
            with np.load(path) as z:
                first_key = list(z.keys())[0]
                arr = z[first_key]
        else:
            arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if arr is None:
                return None

        arr = np.asarray(arr)
        if arr.ndim > 2:
            if arr.shape[-1] > 1:
                arr = arr.astype("float32").mean(axis=-1)
            else:
                arr = arr[..., 0]
        elif arr.ndim != 2:
            return None

        arr = arr.astype("float32", copy=False)

        vmax = float(arr.max()) if arr.size else 0.0
        if vmax > 1.0:
            denom = vmax if vmax > 0 else 1.0
            arr = arr / denom

        arr = np.clip(arr, 0.0, 1.0)
        return arr
    except Exception as e:
        log.warning(f"UI:WARN:train: Thermal read failed for {path}: {e}")
        return None


def _strip_rf_suffix(stem: str) -> str:
    """
    Remove Roboflow suffix like '.rf.<hash>' from a stem if present.
    Example:
      'DJI_..._T_JPG.rf.38d720eebb53eb37fd541db859b71624' -> 'DJI_..._T_JPG'
    """
    # split once on '.rf.' to keep left portion
    if ".rf." in stem:
        return stem.split(".rf.", 1)[0]
    return stem


def _derive_thermal_from_train_folder(rgb_path: str) -> Optional[str]:
    """
    Given an RGB file path (e.g., /.../data/train/IMG.rf.hash.jpg),
    try to find a matching thermal .tif/.tiff under /.../data/train/thermal/.

    Matching strategy:
      - Use RGB stem without extension, strip Roboflow '.rf.*' tail
      - Look for exact 'stem.tif[f]' in thermal dir (case-insensitive)
      - If not found, try a few common stem variants:
          * replace trailing '_JPG' with nothing
          * replace '_T_JPG' with '_T' (seen in some pipelines)
      - If still not found, do a fallback fuzzy match: any .tif[f] whose stem
        equals the base part before an underscore-count drift, pick the
        longest common prefix match.

    Returns absolute path if found, else None.
    """
    train_dir = os.path.dirname(rgb_path)              # .../data/train
    # Find 'thermal' directory under this split (case-insensitive fallback)
    thermal_dir = os.path.join(train_dir, "thermal")
    if not os.path.isdir(thermal_dir):
        # try case variants
        for cand in ("Thermal", "THERMAL", "ir", "IR"):
            alt = os.path.join(train_dir, cand)
            if os.path.isdir(alt):
                thermal_dir = alt
                break
    if not os.path.isdir(thermal_dir):
        return None

    rgb_stem = os.path.splitext(os.path.basename(rgb_path))[0]  # e.g., DJI_..._T_JPG.rf.hash
    stem = _strip_rf_suffix(rgb_stem)                            # e.g., DJI_..._T_JPG

    candidates = []
    # 1) exact stem .tif/.tiff
    for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
        candidates.append(os.path.join(thermal_dir, stem + ext))

    # 2) common variants
    def add_variant(s: str):
        for ext in (".tif", ".tiff", ".TIF", ".TIFF"):
            candidates.append(os.path.join(thermal_dir, s + ext))

    if stem.endswith("_T_JPG"):
        add_variant(stem[:-4])          # drop '_JPG' -> '_T'
        add_variant(stem.replace("_T_JPG", ""))  # drop entirely
    if stem.endswith("_JPG"):
        add_variant(stem[:-4])
    if stem.endswith("_T"):
        add_variant(stem[:-2])

    # Check direct candidates first
    for p in candidates:
        if os.path.exists(p):
            return p

    # 3) Fallback: scan thermal dir for any tif[f] that "looks like" the stem
    tifs = glob.glob(os.path.join(thermal_dir, "*.tif")) + glob.glob(os.path.join(thermal_dir, "*.tiff"))
    stem_low = stem.lower()
    # Prefer exact (case-insensitive), then substring, then longest common prefix
    exact = [p for p in tifs if os.path.splitext(os.path.basename(p))[0].lower() == stem_low]
    if exact:
        return exact[0]

    subs = [p for p in tifs if stem_low in os.path.splitext(os.path.basename(p))[0].lower()
            or os.path.splitext(os.path.basename(p))[0].lower() in stem_low]
    if subs:
        # pick the one with the longest filename (heuristic: more specific)
        subs.sort(key=lambda p: len(os.path.basename(p)), reverse=True)
        return subs[0]

    return None


def _find_thermal_path(d: Dict[str, Any]) -> Optional[str]:
    """
    Heuristics to discover a thermal path in the dataset dict OR derive it
    from the directory structure (…/train/thermal/*.tif[f]).
    """
    # 1) Check typical dict keys first
    candidate_keys = [
        "thermal_file_name", "thermal_path", "thermal", "t_path", "rjpeg_path",
        "thermal_png_path", "ir_path",
    ]
    for k in candidate_keys:
        p = d.get(k)
        if isinstance(p, str) and p and os.path.exists(p):
            return p

    # 2) Derive by sibling 'thermal' folder rule
    rgb_path = d.get("file_name")
    if isinstance(rgb_path, str) and rgb_path:
        derived = _derive_thermal_from_train_folder(rgb_path)
        if derived and os.path.exists(derived):
            return derived

    return None


class RGBThermalDatasetMapper:
    """
    Compatible with Detectron2's DatasetMapper style. Returns a dict with:
      - "image": torch.FloatTensor CxHxW (C=4), values in [0,1]
      - "instances": training targets (if annotations exist)
      - "height"/"width"
    """

    def __init__(self, cfg, is_train: bool = True, expect_channels: int = 4, log_prefix: str = "UI"):
        self.is_train = is_train
        self.expect_channels = expect_channels
        self.log_prefix = log_prefix

        self.mask_on = cfg.MODEL.MASK_ON
        self.keypoint_on = cfg.MODEL.KEYPOINT_ON

        if is_train:
            # support cfg.INPUT and cfg.Input
            MIN = getattr(cfg, "Input", cfg.INPUT)
            min_sizes = MIN.MIN_SIZE_TRAIN
            max_size = MIN.MAX_SIZE_TRAIN
            sample_style = MIN.MIN_SIZE_TRAIN_SAMPLING

            if sample_style == "range":
                self.augmentations = [
                    T.ResizeShortestEdge(
                        short_edge_length=(min(min_sizes), max(min_sizes)),
                        max_size=max_size,
                        sample_style="range",
                    )
                ]
            else:
                self.augmentations = [
                    T.ResizeShortestEdge(
                        short_edge_length=list(min_sizes),
                        max_size=max_size,
                        sample_style="choice",
                    )
                ]
            flip = MIN.RANDOM_FLIP
            if flip == "horizontal":
                self.augmentations.append(T.RandomFlip(horizontal=True, vertical=False))
            elif flip == "vertical":
                self.augmentations.append(T.RandomFlip(horizontal=False, vertical=True))
        else:
            self.augmentations = [
                T.ResizeShortestEdge(
                    short_edge_length=cfg.INPUT.MIN_SIZE_TEST,
                    max_size=cfg.INPUT.MAX_SIZE_TEST,
                    sample_style="choice",
                )
            ]

        self.augs = T.AugmentationList(self.augmentations)

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        return {
            "cfg": cfg,
            "is_train": is_train,
            "expect_channels": 4,
            "log_prefix": "UI",
        }

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        d = dataset_dict.copy()

        # 1) RGB (HxWx3 uint8)
        rgb_path = d.get("file_name", None)
        if not rgb_path or not os.path.exists(rgb_path):
            raise FileNotFoundError(f"RGB file not found in dataset dict: {rgb_path}")
        rgb = _read_rgb_strict_rgb(rgb_path)  # HxWx3, uint8

        # 2) Thermal best-effort; ensure HxW match BEFORE concat
        H, W = rgb.shape[:2]
        thermal = None

        th_path = _find_thermal_path(d)
        if th_path is not None:
            thermal = _read_thermal_best_effort(th_path)

        # accounting + optional progress logs
        if thermal is not None:
            if th_path and th_path.lower().endswith((".tif", ".tiff")):
                THERM_COUNTS["tif"] += 1
            else:
                THERM_COUNTS["other"] += 1
        else:
            THERM_COUNTS["missing"] += 1
            log.info(f"{self.log_prefix}:INFO:train: No thermal for {os.path.basename(rgb_path)}; using zeros.")

        seen = sum(THERM_COUNTS.values())
        if seen % 50 == 0:
            log.info(
                f"{self.log_prefix}:INFO:train: Thermal usage so far: "
                f"tif={THERM_COUNTS['tif']} other={THERM_COUNTS['other']} "
                f"missing={THERM_COUNTS['missing']} (seen={seen})"
            )

        if thermal is None:
            thermal = np.zeros((H, W), dtype=np.float32)
        else:
            # Optional: log when we *did* find a .tif
            if th_path and th_path.lower().endswith((".tif", ".tiff")):
                log.info(f"{self.log_prefix}:INFO:train: Using thermal tif: {os.path.basename(th_path)}")

        # Ensure HxW 2D
        if thermal.ndim != 2:
            thermal = np.asarray(thermal)
            if thermal.ndim == 3:
                if thermal.shape[2] > 1:
                    log.warning(
                        f"{self.log_prefix}:WARN:train: Thermal has {thermal.shape[2]} channels for "
                        f"{os.path.basename(th_path) if th_path else '<thermal>'}; averaging to 1."
                    )
                    thermal = thermal.astype("float32").mean(axis=2)
                else:
                    thermal = thermal[..., 0]
            else:
                raise RuntimeError(f"Unexpected thermal shape {thermal.shape}; expected HxW.")

        # Resize to RGB HxW if needed
        if thermal.shape[0] != H or thermal.shape[1] != W:
            log.warning(
                f"{self.log_prefix}:WARN:train: Thermal size {thermal.shape[::-1]} != RGB size {(W, H)} for "
                f"{os.path.basename(rgb_path)}; resizing thermal with NEAREST."
            )
            thermal = cv2.resize(thermal, (W, H), interpolation=cv2.INTER_NEAREST)

        # 3) Concat to HxWx4 float32 in [0,1]
        if rgb.dtype != np.uint8:
            rgb = rgb.astype("uint8")
        thermal = np.clip(thermal, 0.0, 1.0).astype("float32")
        rgb_f = (rgb.astype("float32") / 255.0)
        combined = np.concatenate([rgb_f, thermal[..., None]], axis=2)  # HxWx4

        if combined.shape[2] > 4:
            log.warning(
                f"{self.log_prefix}:WARN:train: Combined produced {combined.shape[2]} channels for "
                f"{os.path.basename(rgb_path)}; reducing to RGB + last channel."
            )
            combined = np.concatenate([combined[..., :3], combined[..., -1:]], axis=2)

        # 4) Augment
        aug_input = T.AugInput(combined)
        transforms = self.augs(aug_input)
        image = aug_input.image  # HxWx4 float32

        if image.ndim != 3 or image.shape[2] != self.expect_channels:
            fname = d.get("file_name", "<unknown>")
            print(
                f"{self.log_prefix}:ERR:train: MAPPER produced {image.shape[2]} channels for {fname} "
                f"(expected {self.expect_channels}). Keys: {list(d.keys())}",
                flush=True,
            )
            raise RuntimeError(
                f"Mapper produced {image.shape[2]} channels for {fname}, expected {self.expect_channels}."
            )

        # 5) Annotations (convert ALL to XYXY_ABS; assert 4-dim boxes)
        image_shape_hw = (image.shape[0], image.shape[1])  # H, W

        if "annotations" in d:
            raw_annos = d.pop("annotations")

            annos = [
                transform_instance_annotations(obj, transforms, image_shape_hw)
                for obj in raw_annos
                if obj.get("iscrowd", 0) == 0
            ]

            for a in annos:
                if "bbox" in a and "bbox_mode" in a:
                    try:
                        if a["bbox"] is not None and len(a["bbox"]) == 5:
                            print(
                                f"[dbg][mapper] rotated->xyxy for {os.path.basename(rgb_path)} : {a['bbox']}",
                                flush=True,
                            )
                        a["bbox"] = BoxMode.convert(a["bbox"], a["bbox_mode"], BoxMode.XYXY_ABS)
                        a["bbox_mode"] = BoxMode.XYXY_ABS
                    except Exception as e:
                        print(f"[dbg][mapper] bbox convert failed for {os.path.basename(rgb_path)}: {e}", flush=True)
                        raise

            instances = annotations_to_instances(annos, image_shape_hw, mask_format="polygon")
            instances.gt_boxes.clip(image_shape_hw)
            instances = filter_empty_instances(instances)

            if hasattr(instances, "gt_boxes"):
                t = instances.gt_boxes.tensor
                if t.numel():
                    print(f"[dbg][mapper] {os.path.basename(rgb_path)} gt_box_shape={tuple(t.shape)}", flush=True)
                if t.numel() and t.shape[1] != 4:
                    raise RuntimeError(
                        f"[mapper] non-4D boxes after convert: shape={tuple(t.shape)} "
                        f"file={os.path.basename(rgb_path)}"
                    )

            d["instances"] = instances

        # 6) To CHW float32
        image_chw = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        C = int(image_chw.shape[0])
        if C != self.expect_channels:
            fname = d.get("file_name", "<unknown>")
            print(
                f"{self.log_prefix}:ERR:train: POST-AUG produced {C} channels for {fname} "
                f"(expected {self.expect_channels}).",
                flush=True,
            )
            raise RuntimeError(f"POST-AUG produced {C} channels for {fname}, expected {self.expect_channels}.")

        d["image"] = image_chw
        d["height"] = image_shape_hw[0]
        d["width"] = image_shape_hw[1]

        return d
