from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import lru_cache
import json, cv2, numpy as np, torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .aug_utils import build_geometric_augs, build_rgb_photometric_augs


THERMAL_DIR_NAMES = ("thermal", "ir", "t", "temp")
THERMAL_EXTS = (".tif", ".tiff", ".png")

@lru_cache(maxsize=128)
def _load_pairs_json(images_dir: str) -> Dict[str, str]:
    pj = Path(images_dir) / "thermal" / "pairs.json"
    if pj.exists():
        try:
            j = json.loads(pj.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in j.items()} if isinstance(j, dict) else {}
        except Exception:
            return {}
    return {}

def _guess_thermal_sidecar(images_dir: Path, img_path: Path) -> Optional[Path]:
    for dname in THERMAL_DIR_NAMES:
        tdir = images_dir / dname
        if tdir.exists():
            for ext in THERMAL_EXTS:
                cand = tdir / f"{img_path.stem}{ext}"
                if cand.exists():
                    return cand
    for ext in THERMAL_EXTS:
        cand = images_dir / f"{img_path.stem}{ext}"
        if cand.exists():
            return cand
    return None

def _load_thermal_uint8(path: Path, size_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None: return None
    if arr.ndim == 3: arr = arr[..., 0]
    arr = arr.astype(np.float32)
    p2, p98 = np.percentile(arr, [2.0, 98.0])
    lo, hi = float(p2), float(p98)
    if hi <= lo:
        lo, hi = float(arr.min()), float(max(arr.max(), arr.min() + 1.0))
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    t8 = arr.astype(np.uint8)
    h, w = size_hw
    return cv2.resize(t8, (w, h), interpolation=cv2.INTER_LINEAR) if t8.shape != (h, w) else t8

def _stack_bgrt(bgr: np.ndarray, t8: np.ndarray) -> np.ndarray:
    if bgr.ndim != 3 or bgr.shape[2] != 3: raise ValueError("Expected BGR image with 3 channels")
    if t8.ndim != 2: raise ValueError("Expected thermal as single-band uint8")
    if t8.shape[:2] != bgr.shape[:2]: raise ValueError("Thermal must match RGB size")
    return np.dstack([bgr, t8])

class RGBThermalDatasetMapper:
    """
    Thermal-tolerant mapper:
      - loads RGB, finds/loads thermal sidecar (or neutral channel),
      - stacks to BGRT,
      - applies the same augs to all 4 channels,
      - returns Detectron2-style dict with image tensor and (train) instances.
    """

    def __init__(self, cfg, is_train: bool = True):
        self.is_train = bool(is_train)
        self.image_format = getattr(cfg.INPUT, "FORMAT", "BGR")
        self.mask_format = getattr(cfg.INPUT, "MASK_FORMAT", "polygon")

        # --- normalize sizes from cfg, but FORCE 'choice' to avoid 'range'+[800] crash ---
        def _as_list(x):
            if isinstance(x, (list, tuple)):
                return [int(v) for v in x]
            return [int(x)]

        max_size_train = int(getattr(cfg.INPUT, "MAX_SIZE_TRAIN", 1333))
        max_size_test  = int(getattr(cfg.INPUT, "MAX_SIZE_TEST", 1333))

        if self.is_train:
            # Read whatever is set, but *we* will use 'choice' unconditionally
            min_sizes = _as_list(getattr(cfg.INPUT, "MIN_SIZE_TRAIN", [800])) or [800]

            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge(min_sizes, max_size_train, sample_style="choice"),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ])
        else:
            # Eval/inference: deterministic single-size 'choice'
            min_test = getattr(cfg.INPUT, "MIN_SIZE_TEST", 800)
            if isinstance(min_test, (list, tuple)):
                min_test = int(min_test[0] if min_test else 800)
            else:
                min_test = int(min_test)

            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge([min_test], max_size_test, sample_style="choice")
            ])


    def __call__(self, dataset_dict: Dict) -> Dict:
        d = dataset_dict.copy()
        rgb_path = Path(d["file_name"]); images_dir = rgb_path.parent

        # 1) RGB (BGR) and thermal sidecar (neutral if missing)
        rgb = utils.read_image(str(rgb_path), format=self.image_format)  # HxWx3
        H, W = rgb.shape[:2]

        pairs = _load_pairs_json(str(images_dir))
        if rgb_path.name in pairs:
            cand = Path(pairs[rgb_path.name]); t_path = cand if cand.is_absolute() else (images_dir / cand)
        else:
            t_path = _guess_thermal_sidecar(images_dir, rgb_path)

        th = _load_thermal_uint8(t_path, (H, W)) if (t_path and t_path.exists()) else None
        if th is None:
            th = np.full((H, W), 128, dtype=np.uint8)  # neutral plane

        # 2) Geometric augs: SAME transform for RGB & Thermal
        if getattr(self, "is_train", False):
            geo_list = build_geometric_augs(self.cfg)
        else:
            geo_list = [T.ResizeShortestEdge(self.cfg.INPUT.MIN_SIZE_TEST, self.cfg.INPUT.MAX_SIZE_TEST, "choice")]

        geo = T.AugmentationList(geo_list)
        aug_in = T.AugInput(rgb)
        tfm = geo(aug_in)         # apply to RGB, capture transform
        rgb = aug_in.image
        th  = tfm.apply_image(th) # same transform to thermal

        # 3) Photometric jitter: RGB only (train)
        if getattr(self, "is_train", False):
            rgb = T.AugmentationList(build_rgb_photometric_augs())(T.AugInput(rgb)).image

        # 4) Annotations -> Instances (use ONLY the geometric transform)
        if "annotations" in d:
            annos = [
                utils.transform_instance_annotations(a, tfm, rgb.shape[:2])
                for a in d.pop("annotations")
                if a.get("iscrowd", 0) == 0
            ]
            d["instances"] = utils.filter_empty_instances(
                utils.annotations_to_instances(annos, rgb.shape[:2])
            )

        # 5) Re-stack to 4ch & tensorize
        img4 = _stack_bgrt(rgb, th)  # HxWx4
        d["image"] = torch.as_tensor(img4.transpose(2, 0, 1).copy()).float()
        return d

