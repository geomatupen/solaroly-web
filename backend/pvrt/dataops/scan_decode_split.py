# backend/pvrt/dataops/scan_decode_split.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import tifffile
from ..core.thermal import normalize_thermal

from dji_thermal_sdk.dji_sdk import dji_init
from dji_thermal_sdk.utility import rjpeg_to_heatmap

from ..config import DIRP_LIB, describe_dirp
from ..core.io import THERMAL_EXTS

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
    For each RGB image in `images_dir`, ensure a thermal preview exists under
    `images_dir/thermal` and maintain `images_dir/thermal/pairs.json` mapping:
        { "<rgb_filename>": "thermal/<stem>_thermal.<ext>" }

    Rules:
        - If a thermal preview already exists, keep it (idempotent).
        - Else, try to decode RJPEG via DJI SDK and write a normalized 3-channel
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

    # Ensure DIRP is initialized once; let initialization errors propagate.
    ensure_dirp_init()
    log.info("UI:INFO:test: DIRP SDK initialized")

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

        # Attempt decode using DJI SDK
        try:
            temps = rjpeg_to_heatmap(str(rgb), dtype=np.float32)  # HxW float32
            if not isinstance(temps, np.ndarray) or temps.ndim != 2:
                raise ValueError("Invalid thermal plane read from RJPEG.")

            # Log decoded range when possible
            try:
                tmin = float(np.nanmin(temps))
                tmax = float(np.nanmax(temps))
                log.info(f"INFO:prep: decoded {rgb.name} - {out_preview.name} | range={tmin:.2f}..{tmax:.2f}°C")
            except (TypeError, ValueError):
                log.info(f"INFO:prep: decoded {rgb.name} - {out_preview.name}")

            # Normalize to canonical uint8 using the shared helper. This
            # guarantees the decoder writes exactly the same 8-bit preview
            # that training/inference expect (2..98 percentile stretch for
            # numeric arrays; uint8 arrays are returned unchanged).
            try:
                g8 = normalize_thermal(temps)
            except Exception:
                # Fallback to the original inline normalization if anything
                # unexpected happens in normalize_thermal.
                vals = temps.ravel()
                p2 = float(np.percentile(vals, 2)) if vals.size else 0.0
                p98 = float(np.percentile(vals, 98)) if vals.size else 1.0
                g = (np.clip(temps, p2, p98) - p2) / max(1e-12, (p98 - p2))
                g8 = (np.nan_to_num(g) * 255.0).astype(np.uint8)

            # make 3-channel RGB by stacking the single-channel uint8 preview
            if g8.ndim == 2:
                rgb_arr = np.stack([g8, g8, g8], axis=2)
            else:
                rgb_arr = g8[..., :3]

            from PIL import Image
            Image.fromarray(rgb_arr).save(str(out_preview), format="JPEG", quality=90)
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
