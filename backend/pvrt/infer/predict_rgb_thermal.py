from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

from dji_thermal_sdk.dji_sdk import dji_init
from dji_thermal_sdk.utility import rjpeg_to_heatmap

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_setup

from ..trainops.model_patch import widen_backbone_conv1_to_4ch

log = logging.getLogger("pvrt")

_IMG_EXTS = {".jpg",".jpeg",".png",".tif",".tiff",".JPG",".JPEG",".PNG",".TIF",".TIFF"}
_DJI_INIT = False

def _ensure_dji():
    global _DJI_INIT
    if not _DJI_INIT:
        dji_init("")  # uses env/auto-locate for libdirp.so
        _DJI_INIT = True
        log.info("[infer] DJI Thermal SDK initialized")

def _list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.iterdir()) if p.suffix in _IMG_EXTS and not p.name.endswith("_thermal.tif")]

def _try_thermal(rgb_path: Path) -> np.ndarray | None:
    try:
        _ensure_dji()
        temps = rjpeg_to_heatmap(str(rgb_path), dtype="float32")
        if isinstance(temps, np.ndarray) and temps.ndim == 2:
            return temps
    except Exception:
        pass
    return None

def _maybe_sidecar_tif(rgb_path: Path) -> Path | None:
    cand1 = rgb_path.with_name(f"{rgb_path.stem}_thermal.tif")
    if cand1.exists(): return cand1
    cand2 = rgb_path.parent / "thermal" / f"{rgb_path.stem}_thermal.tif"
    if cand2.exists(): return cand2
    return None

def _build_model(weights_dir: Path) -> Tuple[torch.nn.Module, str, dict]:
    weights = weights_dir / "model_final.pth"
    meta = weights_dir / "model_meta.json"
    if not weights.exists():
        raise FileNotFoundError(f"No weights at {weights}")
    input_mode = "rgb"
    if meta.exists():
        try:
            input_mode = json.loads(meta.read_text()).get("input_mode", "rgb")
        except Exception:
            pass

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.INPUT.FORMAT = "RGB"

    if input_mode == "rgbtherm":
        cfg.MODEL.PIXEL_MEAN = [0.5, 0.5, 0.5, 0.5]
        cfg.MODEL.PIXEL_STD  = [0.5, 0.5, 0.5, 0.5]
        model = build_model(cfg)
        model = widen_backbone_conv1_to_4ch(model)
    else:
        cfg.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]
        cfg.MODEL.PIXEL_STD  = [0.229, 0.224, 0.225]
        model = build_model(cfg)

    DetectionCheckpointer(model).load(str(weights))
    model.eval()
    default_setup(cfg, {})
    return model, input_mode, cfg

def _prep_tensor_rgb(rgb_img: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(rgb_img.transpose(2,0,1).copy())

def _prep_tensor_rgbtherm(rgb_img: np.ndarray, ther_img: np.ndarray | None, t_min=0.0, t_max=100.0) -> torch.Tensor:
    H,W = rgb_img.shape[:2]
    if ther_img is None:
        ther = np.zeros((H,W), dtype="float32")
    else:
        ther = np.clip((ther_img - t_min)/max(1e-6, (t_max - t_min)), 0.0, 1.0)
        if ther.shape != (H,W):
            ther = cv2.resize(ther, (W,H), interpolation=cv2.INTER_CUBIC)
    img4 = np.concatenate([rgb_img, ther[...,None]], axis=2)
    return torch.as_tensor(img4.transpose(2,0,1).copy())

def predict_folder(
    images_dir: str | Path,
    out_dir: str | Path,
    weights_dir: str | Path,
    use_thermal: bool = True,
) -> Path:
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    (out_dir / "preds").mkdir(parents=True, exist_ok=True)

    model, input_mode, cfg = _build_model(Path(weights_dir))
    files = _list_images(images_dir)
    log.info(f"[infer] Images: {len(files)}  use_thermal={use_thermal}  model={input_mode}")

    for i, path in enumerate(files, 1):
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            log.warning(f"[infer] {i}/{len(files)} cannot read {path.name}, skipping")
            continue
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype("float32") / 255.0

        ther_mat = None
        if use_thermal and input_mode == "rgbtherm":
            sidecar = _maybe_sidecar_tif(path)
            if sidecar and sidecar.exists():
                ther_mat = cv2.imread(str(sidecar), cv2.IMREAD_UNCHANGED).astype("float32")
            else:
                ther_mat = _try_thermal(path)
                if ther_mat is None:
                    log.warning(f"[infer] {i}/{len(files)} RJPEG=No → {path.name} | WARNING: No thermal in EXIF; moving ahead with JPG (RGB-only).")

        if input_mode == "rgbtherm":
            tensor = _prep_tensor_rgbtherm(rgb, ther_mat)
        else:
            if use_thermal:
                log.warning(f"[infer] {i}/{len(files)} model is RGB-only; ignoring 'Use thermal band' for {path.name}.")
            tensor = _prep_tensor_rgb(rgb)

        with torch.no_grad():
            outputs = model([{"image": tensor, "height": rgb.shape[0], "width": rgb.shape[1]}])[0]

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist() if "instances" in outputs else []
        scores = outputs["instances"].scores.cpu().numpy().tolist() if "instances" in outputs else []
        classes = outputs["instances"].pred_classes.cpu().numpy().tolist() if "instances" in outputs else []
        out_json = {"file": path.name, "boxes": boxes, "scores": scores, "classes": classes}
        (out_dir / "preds" / f"{path.stem}.json").write_text(json.dumps(out_json, indent=2))
        log.info(f"[infer] {i}/{len(files)} done: {path.name}")

    log.info(f"[infer] Finished. JSONs: {out_dir/'preds'}")
    return out_dir / "preds"