# backend/pvrt/backends/detectron/train/datasets.py
from __future__ import annotations

from pathlib import Path
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


# ---------------------------------------------------------------------
# COCO JSON discovery (preserve historical discovery order)
# ---------------------------------------------------------------------

def _find_coco_json(split_dir: Path) -> Path:
    """
    Find a COCO annotations file inside `split_dir`.
    Raises FileNotFoundError if none is found.
    """
    for name in [
        "_annotations.coco.json",
        "_annotations.coco",
        "annotations.json",
        "train.json",
        "valid.json",
        "test.json",
    ]:
        p = split_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"COCO JSON not found in {split_dir}")


# ---------------------------------------------------------------------
# Idempotent (re)registration helpers
# ---------------------------------------------------------------------

def _purge_dataset_name(name: str) -> None:
    """
    Remove any previous registration for `name` from BOTH catalogs.
    Safe to call even if the dataset was never registered.
    Works across Detectron2 versions (public .remove() if present,
    otherwise fall back to private maps).
    """
    # DatasetCatalog: call public API if present; otherwise mutate the private map.
    # Some Detectron2 versions raise KeyError if the name is not registered.
    # Only silence that specific case so other errors still surface.
    if hasattr(DatasetCatalog, "remove"):
        try:
            DatasetCatalog.remove(name)  # type: ignore[attr-defined]
        except KeyError:
            # not registered — that's fine, continue
            pass
    else:
        # Older/newer versions may expose a private registry mapping
        DatasetCatalog._REGISTERED.pop(name, None)  # type: ignore[attr-defined]

    # MetadataCatalog must also be cleared. Otherwise Detectron keeps the
    # previous split's json_file metadata and rejects a subsequent dataset
    # with an assertion such as "old/path.json != new/path.json".
    if hasattr(MetadataCatalog, "remove"):
        try:
            MetadataCatalog.remove(name)  # type: ignore[attr-defined]
        except KeyError:
            pass
    elif hasattr(MetadataCatalog, "_NAME_TO_META"):
        MetadataCatalog._NAME_TO_META.pop(name, None)  # type: ignore[attr-defined]
    elif hasattr(MetadataCatalog, "data"):
        MetadataCatalog.data.pop(name, None)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------
# Public API (same signature/behavior as original, but idempotent)
# ---------------------------------------------------------------------

def register_split_coco(name: str, split_dir: str | Path) -> None:
    """
    Register (or re-register) a COCO dataset split under `name` pointing at `split_dir`.

    - Uses detectron2.datasets.register_coco_instances (simple + battle-tested)
    - Purges any prior registration to allow repeated calls (avoids a 500 on a second run)
    - Attaches `thermal_pairs` metadata if thermal/pairs.json exists
    """
    split_dir = Path(split_dir)

    # 1) Find annotations
    anno = _find_coco_json(split_dir)

    # 1b) Some COCO exports omit the optional top-level 'info' field that
    # pycocotools expects while loading evaluation results. Normalize the
    # selected annotation file in place so detection and segmentation both
    # continue to use the original filename (normally
    # _annotations.coco.json); do not create a second *_pvrt_fixed.json file.
    with open(anno, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'info' not in data:
        data['info'] = {'description': 'pvrt dataset (info added)'}
        temporary = anno.with_name(f".{anno.name}.pvrt.tmp")
        temporary.write_text(json.dumps(data), encoding='utf-8')
        temporary.replace(anno)

    # 2) Purge any prior registration for this name (idempotent re-run)
    _purge_dataset_name(name)

    # 3) Fresh registration
    register_coco_instances(name, {}, str(anno), str(split_dir))

    # 4) Attach thermal pairs to metadata (preserve existing behavior)
    meta = MetadataCatalog.get(name)
    pairs = split_dir / "thermal" / "pairs.json"
    if pairs.exists():
        meta.thermal_pairs = json.loads(pairs.read_text(encoding="utf-8"))
    else:
        meta.thermal_pairs = {}
