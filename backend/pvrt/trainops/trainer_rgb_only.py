from __future__ import annotations
import json, os
from pathlib import Path
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, default_setup
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from .datasets import register_split_coco
from .mapper_rgb_only import build_rgb_only_mapper
from .helpers import get_num_classes

class RGBOnlyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=build_rgb_only_mapper(cfg.DATASETS.TRAIN[0], is_train=True))
    @classmethod
    def build_test_loader(cls, cfg, name):
        return build_detection_test_loader(cfg, name, mapper=build_rgb_only_mapper(name, is_train=False))

def train_rgb_only(train_dir: str | Path, val_dir: str | Path, out_dir: str | Path) -> Path:
    train_dir, val_dir = Path(train_dir), Path(val_dir)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    register_split_coco("pv_train", train_dir)
    register_split_coco("pv_val",   val_dir)

    ann_train = next((p for p in [train_dir/"_annotations.coco", train_dir/"_annotations.coco.json", train_dir/"train.json", train_dir/"annotations.json"] if p.exists()), None)
    if ann_train is None:
        raise FileNotFoundError(f"COCO JSON not found in {train_dir}")
    num_classes = get_num_classes(ann_train)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("pv_train",)
    cfg.DATASETS.TEST  = ("pv_val",)
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = (6000, 8000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 1001

    cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.PIXEL_MEAN = [0.485, 0.456, 0.406]
    cfg.MODEL.PIXEL_STD  = [0.229, 0.224, 0.225]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    cfg.OUTPUT_DIR = str(out_dir)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    default_setup(cfg, {})

    trainer = RGBOnlyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    evaluator = COCOEvaluator("pv_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = RGBOnlyTrainer.build_test_loader(cfg, "pv_val")
    _ = inference_on_dataset(trainer.model, val_loader, evaluator)

    with (out_dir/"model_meta.json").open("w", encoding="utf-8") as f:
        json.dump({"input_mode": "rgb"}, f, indent=2)

    final = out_dir / "model_final.pth"
    return final if final.exists() else out_dir