# backend/pvrt/dataops/scan_decode_split.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..core.io import THERMAL_EXTS
from .thermal_convert import convert_dji_rjpeg

log = logging.getLogger("pvrt")

_IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP", ".WEBP",
}

def scan_split_decode_thermal(images_dir: Path) -> Tuple[Path, Dict[str, int | str | None]]:
    """
    For each RGB image in `images_dir`, ensure a thermal preview exists under
    `images_dir/thermal` and maintain `images_dir/thermal/pairs.json` mapping:
        { "<rgb_filename>": "thermal/<stem>_thermal.<ext>" }

    Rules:
        - If a thermal preview already exists, keep it (idempotent).
        - Else, try to decode a supported DJI DIRP or FLIR FFF RJPEG and write a normalized 3-channel
          JPEG preview (`<stem>_thermal.jpg`). Single-band TIFFs are not written
          for decoded RJPEGs to simplify downstream tooling.
        - Never overwrite existing thermal preview files.
        - Keys in pairs.json are filenames (not absolute paths) to match predictors.

    Returns:
      (pairs_json_path, stats)
      stats = {"ok": int, "fail": int, "total": int, "first_error": Optional[str]}
    """
    log.info(f"UI:INFO:test: thermal decode start: {images_dir}")

    images_dir = Path(images_dir)
    thermal_dir = images_dir / "thermal"
    thermal_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = thermal_dir / "pairs.json"

    # Load existing pairs (preserve prior work)
    pairs: Dict[str, str] = {}
    if pairs_path.exists():
        try:
            pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
            if not isinstance(pairs, dict):
                pairs = {}
        except (OSError, json.JSONDecodeError):
            pairs = {}
    log.info(f"UI:INFO:test: loaded existing pairs: {len(pairs)}")

    ok = fail = reuse = 0
    first_error: Optional[str] = None

    for rgb in sorted(images_dir.iterdir()):
        if not _looks_like_rgb(rgb):
            continue

        stem = rgb.stem
        out_preview = thermal_dir / f"{stem}_thermal.jpg"

        # Already paired correctly?
        already = pairs.get(rgb.name)
        if already and (images_dir / already).exists():
            log.info(f"INFO:prep: reuse pair: {rgb.name} -> {already}")
            reuse += 1
            continue

        # If a preview exists from before, reuse it and set/refresh the pair.
        if out_preview.exists():
            pairs[rgb.name] = str(out_preview.relative_to(images_dir))
            log.info(f"INFO:prep: reuse existing preview for {rgb.name} -> {out_preview.name}")
            reuse += 1
            continue

        # Dispatch to DJI DIRP or FLIR FFF according to the source payload.
        try:
            decoded = convert_dji_rjpeg(rgb, out_preview, quality=100, preserve_metadata=True)
            log.info(
                "INFO:prep: decoded %s - %s via %s | raw range=%.2f..%.2f",
                rgb.name, out_preview.name, decoded["backend"],
                decoded["min_value"], decoded["max_value"],
            )
            pairs[rgb.name] = str(out_preview.relative_to(images_dir))
            ok += 1
        except Exception as e:
            log.warning(f"WARN:prep: failed to decode {rgb.name}: {e}")
            if first_error is None:
                first_error = str(e)
            fail += 1
            continue

    pairs_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    log.info(f"UI:OK:test: pairs.json written: {pairs_path}")
    log.info(f"UI:INFO:test: thermal decode summary -> ok={ok}, fail={fail}, reuse={reuse}, total={ok + fail + reuse}")
    return pairs_path, {"ok": ok, "fail": fail, "total": ok + fail + reuse, "first_error": first_error}


# -----------------------
# Small helpers
# -----------------------

def _looks_like_rgb(p: Path) -> bool:
    # treat any image that is not *our* generated thermal preview as an RGB candidate
    if not (p.is_file() and p.suffix in _IMG_EXTS):
        return False
    # ignore any generated thermal preview sidecars (various extensions)
    for ext in THERMAL_EXTS:
        if p.name.endswith(f"_thermal{ext}"):
            return False
    return True
