# backend/pvrt/backends/detectron/train/datasets.py
from __future__ import annotations

from pathlib import Path
import json

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances


# ---------------------------------------------------------------------
# COCO JSON discovery (kept from your original, with the same order)
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
    # DatasetCatalog
    try:
        DatasetCatalog.remove(name)  # type: ignore[attr-defined]
    except Exception:
        try:
            DatasetCatalog._REGISTERED.pop(name, None)  # type: ignore[attr-defined]
        except Exception:
            pass

    # MetadataCatalog
    try:
        MetadataCatalog._NAME_TO_META.pop(name, None)  # type: ignore[attr-defined]
    except Exception:
        # Some versions lazily create metadata; nothing to purge.
        pass


# ---------------------------------------------------------------------
# Public API (same signature/behavior as your original, but idempotent)
# ---------------------------------------------------------------------

def register_split_coco(name: str, split_dir: str | Path) -> None:
    """
    Register (or re-register) a COCO dataset split under `name` pointing at `split_dir`.

    - Uses detectron2.datasets.register_coco_instances (simple + battle-tested)
    - Purges any prior registration so you can call this repeatedly (fixes 500 on 2nd run)
    - Attaches `thermal_pairs` metadata if thermal/pairs.json exists
    """
    split_dir = Path(split_dir)

    # 1) Find annotations
    anno = _find_coco_json(split_dir)

    # 1b) Defensive: some COCO JSON files omit optional top-level fields like
    # 'info' which pycocotools expects when loading results. Create a small
    # fixed copy with a minimal 'info' section if needed so evaluation does
    # not fail. We write the fixed copy next to the original (idempotent).
    try:
        with open(anno, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'info' not in data:
            fixed = dict(data)
            fixed['info'] = {'description': 'pvrt dataset (info added)'}
            fixed_path = Path(anno).with_name(Path(anno).stem + "_pvrt_fixed.json")
            fixed_path.write_text(json.dumps(fixed), encoding='utf-8')
            anno = fixed_path
    except Exception:
        # If anything goes wrong, fall back to the original anno path and
        # let register_coco_instances raise if it's truly broken.
        pass

    # 2) Purge any prior registration for this name (idempotent re-run)
    _purge_dataset_name(name)

    # 3) Fresh registration
    register_coco_instances(name, {}, str(anno), str(split_dir))

    # 4) Attach thermal pairs to metadata (kept from your original)
    meta = MetadataCatalog.get(name)
    pairs = split_dir / "thermal" / "pairs.json"
    if pairs.exists():
        try:
            meta.thermal_pairs = json.loads(pairs.read_text(encoding="utf-8"))
        except Exception:
            # Keep it predictable; empty dict on any parse issue
            meta.thermal_pairs = {}
    else:
        meta.thermal_pairs = {}
