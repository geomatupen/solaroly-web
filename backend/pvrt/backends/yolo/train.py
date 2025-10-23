"""Training helper for YOLO using ultralytics.YOLO

Provides a `run_train` function called by the YOLOBackend above. This is a
minimal wrapper around `ultralytics.YOLO(...).train(...)` tuned to the project's
conventions (writes outputs into out_dir/run_name).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import logging
import json

log = logging.getLogger("pvrt")


def _discover_num_classes_from_coco(train_dir: Path) -> int:
    # lightweight: look for _annotations.coco.json or any .json that looks like COCO
    import json
    cand = train_dir / "_annotations.coco.json"
    if cand.exists():
        data = json.loads(cand.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        return len(cats)
    for j in train_dir.glob("*.json"):
        try:
            data = json.loads(j.read_text(encoding="utf-8"))
            if isinstance(data, dict) and all(k in data for k in ("images", "annotations", "categories")):
                cats = data.get("categories", [])
                return len(cats)
        except Exception:
            continue
    return 0


def _discover_class_names_from_coco(train_dir: Path) -> List[str]:
    import json
    cand = train_dir / "_annotations.coco.json"
    if cand.exists():
        data = json.loads(cand.read_text(encoding="utf-8"))
        cats = data.get("categories", [])
        try:
            cats = sorted(cats, key=lambda c: int(c.get("id", 0)))
        except Exception:
            pass
        return [str(c.get("name", f"class_{i}")) for i, c in enumerate(cats)]
    return []


def run_train(train_dir: Path, val_dir: Path, out_dir: Path, use_thermal: bool, max_iter: int, base_lr: float, ims_per_batch: int, run_name: str = "yolo_run", yolo_family: str = "v8", yolo_seg: bool = False) -> Dict[str, Any]:
    """Run a YOLO training job.

    Returns a dict with keys: best_weights, final_weights, model_name, num_classes, class_names
    """
    # Delay import to keep module import cheap when not training
    from ultralytics import YOLO
    import yaml
    from pathlib import Path

    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Attempt to find classes
    class_names = _discover_class_names_from_coco(train_dir)
    num_classes = len(class_names) or _discover_num_classes_from_coco(train_dir)

    # Choose model checkpoint from family + seg flag
    family = (yolo_family or "v8").lower()
    if family in {"v8", "8"}:
        model_weights = "yolov8n-seg.pt" if yolo_seg else "yolov8s.pt"
    elif family in {"v9", "9"}:
        model_weights = "yolov9n-seg.pt" if yolo_seg else "yolov9s.pt"
    elif family in {"v10", "10"}:
        model_weights = "yolov10n-seg.pt" if yolo_seg else "yolov10s.pt"
    else:
        model_weights = "yolov8s.pt"

    # Create a temporary data.yaml compatible with ultralytics
    data_yaml = {
        "path": str(train_dir.parent.resolve()),
        "train": "train",
        "val": "valid",
        "test": "test",
        "names": class_names or {i: f"class_{i}" for i in range(num_classes)}
    }
    yaml_path = run_dir / "data.yaml"
    try:
        yaml.safe_dump(data_yaml, yaml_path)
    except Exception:
        yaml_path.write_text(json.dumps(data_yaml), encoding="utf-8")

    # instantiate model and call train
    model = YOLO(model_weights)

    # Map parameters: convert max_iter -> epochs roughly (heuristic)
    epochs = max(1, int(max_iter // 100)) if max_iter and max_iter > 0 else 50

    # train
    results = model.train(
        data=str(yaml_path),
        imgsz=512,
        epochs=epochs,
        batch=ims_per_batch or 8,
        workers=4,
        device=0,
        optimizer="AdamW",
        lr0=base_lr or 0.001,
        project=str(out_dir),
        name=run_name,
        exist_ok=True,
    )

    # locate weights
    save_dir = Path(results.files[0]).parent if hasattr(results, "files") and results.files else (out_dir / run_name)
    best = save_dir / "weights" / "best.pt"
    final = save_dir / "weights" / "last.pt"

    return {
        "best_weights": str(best) if best.exists() else "",
        "final_weights": str(final) if final.exists() else "",
        "model_name": model_weights.replace('.pt',''),
        "score_thresh_test": 0.25,
        "num_classes": int(num_classes),
        "class_names": class_names,
    }
