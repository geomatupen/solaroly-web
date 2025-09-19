# backend/pvrt/dataops/scan_decode_split.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tifffile

from dji_thermal_sdk.dji_sdk import dji_init
from dji_thermal_sdk.utility import rjpeg_to_heatmap

from ..config import DIRP_LIB, describe_dirp

log = logging.getLogger("pvrt")

_IMG_EXTS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    ".JPG", ".JPEG", ".PNG", ".TIF", ".TIFF", ".BMP", ".WEBP",
}

_DJI_READY = False  # set by ensure_dirp_init()


# -----------------------
# Public, small API
# -----------------------

def ensure_dirp_init() -> None:
    """
    Initialize DJI DIRP once using the absolute shared library path (DIRP_LIB).
    Raises a clear error if the library is missing or invalid.
    """
    global _DJI_READY
    if _DJI_READY:
        return

    lib = Path(DIRP_LIB) if DIRP_LIB else None
    if not lib or not lib.exists():
        raise FileNotFoundError(
            f"DJI DIRP library not found. {describe_dirp()} "
            f"Set PVRT_DIRP_LIB to the absolute path of your libdirp (.so/.dll)."
        )

    # Initialize the SDK with the exact .so/.dll path.
    dji_init(str(lib))
    log.info(f"[DJI] DIRP initialized: {lib}")
    _DJI_READY = True


def scan_split_decode_thermal(images_dir: Path) -> Tuple[Path, Dict[str, int | str | None]]:
    """
    For each RGB image in `images_dir`, ensure a thermal TIFF exists in `images_dir/thermal`
    and maintain `images_dir/thermal/pairs.json` with:
        { "<rgb_filename>": "thermal/<stem>_thermal.tif" }

    Rules:
      - If a thermal TIFF already exists, keep it (idempotent).
      - Else, try to decode RJPEG via DJI SDK and write a float32 single-band TIFF.
      - Never overwrite existing thermal files.
      - Keys in pairs.json are *filenames* (not absolute paths) to match predictors.

    Returns:
      (pairs_json_path, stats)
      stats = {"ok": int, "fail": int, "total": int, "first_error": Optional[str]}
    """
    # [LOG] starting scan
    log.info(f"UI:INFO:test: thermal decode start: {images_dir}")

    images_dir = Path(images_dir)
    thermal_dir = images_dir / "thermal"
    thermal_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = thermal_dir / "pairs.json"

    # Load existing pairs (don’t lose prior work)
    try:
        pairs: Dict[str, str] = json.loads(pairs_path.read_text(encoding="utf-8"))
        if not isinstance(pairs, dict):
            pairs = {}
    except Exception:
        pairs = {}
    # [LOG] pairs loaded
    log.info(f"UI:INFO:test: loaded existing pairs: {len(pairs)}")

    ok = fail = reuse = 0
    first_error: Optional[str] = None

    # Try once at the start; if it fails, we skip decoding and keep pairs as-is.
    try:
        ensure_dirp_init()
        # [LOG] dirp ready
        log.info("UI:INFO:test: DIRP SDK initialized")
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        log.error(f"UI:ERR:test: DIRP init failed: {msg}. Details: {describe_dirp()}")
        pairs_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
        # [LOG] summary (early exit)
        log.info("UI:INFO:test: thermal decode summary -> ok=0, fail=0, total=0")
        return pairs_path, {"ok": 0, "fail": 0, "total": 0, "first_error": msg}

    for rgb in sorted(images_dir.iterdir()):
        if not _looks_like_rgb(rgb):
            continue

        stem = rgb.stem
        out_tif = thermal_dir / f"{stem}_thermal.tif"

        # Already paired correctly?
        already = pairs.get(rgb.name)
        if already and (images_dir / already).exists():
            # [LOG] already paired; skip
            log.info(f"UI:INFO:prep: reuse pair: {rgb.name} -> {already}")
            reuse +=1
            continue

        # If a TIFF exists from before, reuse it and set/refresh the pair.
        if out_tif.exists():
            pairs[rgb.name] = str(out_tif.relative_to(images_dir))
            # [LOG] reuse existing tiff
            log.info(f"UI:INFO:prep: reuse existing TIFF for {rgb.name} -> {out_tif.name}")
            reuse += 1
            continue

        # Decode RJPEG → float32 map → TIFF
        try:
            temps = rjpeg_to_heatmap(str(rgb), dtype=np.float32)  # HxW float32
            if not isinstance(temps, np.ndarray) or temps.ndim != 2:
                raise ValueError("Invalid thermal plane read from RJPEG.")

            # [LOG] decoded range preview
            try:
                tmin = float(np.nanmin(temps))
                tmax = float(np.nanmax(temps))
                log.info(f"UI:INFO:prep: decoded {rgb.name} → {out_tif.name} | range={tmin:.2f}..{tmax:.2f}°C")
            except Exception:
                log.info(f"UI:INFO:prep: decoded {rgb.name} → {out_tif.name}")

            # Write single-band float32 TIFF
            tifffile.imwrite(str(out_tif), temps.astype(np.float32))
            pairs[rgb.name] = str(out_tif.relative_to(images_dir))
            ok += 1
        except Exception as e:
            # Record only the first error for UI, continue processing others
            if first_error is None:
                first_error = f"{type(e).__name__}: {e}"
            fail += 1
            # [LOG] per-file failure
            log.warning(f"UI:WARN:prep: failed {rgb.name}: {type(e).__name__}: {e}")

    pairs_path.write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    # [LOG] summary
    log.info(f"UI:OK:test: pairs.json written: {pairs_path}")
    log.info(f"UI:INFO:test: thermal decode summary -> ok={ok}, fail={fail}, reuse={reuse}, total={ok + fail + reuse}")
    return pairs_path, {"ok": ok, "fail": fail, "total": ok + fail, "first_error": first_error}


# -----------------------
# Small helpers
# -----------------------

def _looks_like_rgb(p: Path) -> bool:
    # treat any image that is not *our* generated thermal tif as an RGB candidate
    return p.is_file() and p.suffix in _IMG_EXTS and not p.name.endswith("_thermal.tif")
