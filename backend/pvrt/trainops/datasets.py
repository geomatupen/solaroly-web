from __future__ import annotations
from pathlib import Path
import json
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

def _find_coco_json(split_dir: Path) -> Path:
    for name in ["_annotations.coco.json", "_annotations.coco", "annotations.json", "train.json", "valid.json", "test.json"]:
        p = split_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"COCO JSON not found in {split_dir}")

def register_split_coco(name: str, split_dir: str | Path) -> None:
    split_dir = Path(split_dir)
    anno = _find_coco_json(split_dir)
    register_coco_instances(name, {}, str(anno), str(split_dir))
    # attach thermal pairs if present
    pairs = split_dir / "thermal" / "pairs.json"
    meta = MetadataCatalog.get(name)
    if pairs.exists():
        try:
            meta.thermal_pairs = json.loads(pairs.read_text())
        except Exception:
            meta.thermal_pairs = {}
    else:
        meta.thermal_pairs = {}