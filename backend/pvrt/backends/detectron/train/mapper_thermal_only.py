from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple
from functools import lru_cache
import json, cv2, numpy as np, torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .aug_utils import build_geometric_augs

THERMAL_DIR_NAMES = ("thermal", "ir", "t", "temp")
THERMAL_EXTS = (".tif", ".tiff", ".png")


@lru_cache(maxsize=128)
def _load_pairs_json(images_dir: str) -> Dict[str, str]:
    pj = Path(images_dir) / "thermal" / "pairs.json"
    if pj.exists():
        try:
            j = json.loads(pj.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in j.items()} if isinstance(j, dict) else {}
        except Exception as e:
            logging.getLogger("pvrt").debug("failed to read pairs.json %s: %s", pj, e)
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
    if arr is None:
        return None
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    p2, p98 = np.percentile(arr, [2.0, 98.0])
    lo, hi = float(p2), float(p98)
    if hi <= lo:
        lo, hi = float(arr.min()), float(max(arr.max(), arr.min() + 1.0))
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    t8 = arr.astype(np.uint8)
    h, w = size_hw
    return cv2.resize(t8, (w, h), interpolation=cv2.INTER_LINEAR) if t8.shape != (h, w) else t8


class ThermalOnlyDatasetMapper:
    """
    Mapper for thermal-only training/inference.

    - loads thermal sidecar (or neutral plane if missing),
    - applies geometric transforms,
    - returns Detectron2-style dict where `image` is a single-channel tensor (1,H,W).
    """

    def __init__(self, cfg, is_train: bool = True):
        self.cfg = cfg
        self.is_train = bool(is_train)
        self.image_format = getattr(cfg.INPUT, "FORMAT", "BGR")
        self.mask_format = getattr(cfg.INPUT, "MASK_FORMAT", "polygon")

        max_size_train = int(getattr(cfg.INPUT, "MAX_SIZE_TRAIN", 1333))
        max_size_test = int(getattr(cfg.INPUT, "MAX_SIZE_TEST", 1333))

        if self.is_train:
            min_sizes = getattr(cfg.INPUT, "MIN_SIZE_TRAIN", [800])
            if not isinstance(min_sizes, (list, tuple)):
                min_sizes = [int(min_sizes)]
            self.augmentations = T.AugmentationList([
                T.ResizeShortestEdge(min_sizes, max_size_train, sample_style="choice"),
                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            ])
        else:
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
        # determine thermal path using pairs.json or thermal dir
        rgb_path = Path(d.get("file_name"))
        images_dir = rgb_path.parent

        # load neutral thermal plane if missing
        H, W = 0, 0
        try:
            img = utils.read_image(str(rgb_path), format=self.image_format)
            H, W = img.shape[:2]
        except Exception as e:
            logging.getLogger("pvrt").debug("failed to read image %s: %s", rgb_path, e)
            # as a fallback assume typical size from cfg
            H = int(getattr(self.cfg.INPUT, "MIN_SIZE_TEST", 800))
            W = H

        pairs = _load_pairs_json(str(images_dir))
        if rgb_path.name in pairs:
            cand = Path(pairs[rgb_path.name]); t_path = cand if cand.is_absolute() else (images_dir / cand)
        else:
            t_path = _guess_thermal_sidecar(images_dir, rgb_path)

        th = _load_thermal_uint8(t_path, (H, W)) if (t_path and t_path.exists()) else None
        if th is None:
            th = np.full((H, W), 128, dtype=np.uint8)

        # apply geometric transforms to thermal only
        geo = self.augmentations
        aug_in = T.AugInput(th)
        tfm = geo(aug_in)
        th = aug_in.image

        # annotations -> instances (use geometric transform)
        if "annotations" in d:
            annos = [
                utils.transform_instance_annotations(a, tfm, (th.shape[0], th.shape[1]))
                for a in d.pop("annotations")
                if a.get("iscrowd", 0) == 0
            ]
            d["instances"] = utils.filter_empty_instances(
                utils.annotations_to_instances(annos, th.shape[:2])
            )

        # produce 1xHxW tensor
        img1 = th
        if img1.ndim == 2:
            pass
        elif img1.ndim == 3:
            img1 = img1[..., 0]

        d["image"] = torch.as_tensor(img1.astype('float32')).unsqueeze(0).contiguous()
        return d
