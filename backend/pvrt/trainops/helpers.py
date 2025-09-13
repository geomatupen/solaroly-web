from __future__ import annotations
import os, json
from pathlib import Path

def get_num_classes_from_coco(anno_json: str | Path) -> int:
    with open(anno_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    cats = data.get("categories", [])
    return len(cats) if isinstance(cats, list) else 0

def get_num_classes(anno_json: str | Path) -> int:
    mod_path = os.getenv("PVRT_HELPERS_MODULE", "").strip()
    if mod_path:
        try:
            mod = __import__(mod_path, fromlist=["get_num_classes"])
            if hasattr(mod, "get_num_classes"):
                return int(mod.get_num_classes(str(anno_json)))
        except Exception:
            pass
    return get_num_classes_from_coco(anno_json)