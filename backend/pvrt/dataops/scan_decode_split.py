# backend/pvrt/dataops/scan_decode_split.py
from __future__ import annotations
import json, traceback, logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import tifffile

from dji_thermal_sdk.dji_sdk import dji_init
from dji_thermal_sdk.utility import rjpeg_to_heatmap

from ..config import DIRP_LIB, describe_dirp

_IMG_EXTS = {".jpg",".jpeg",".JPG",".JPEG",".png",".PNG",".tif",".tiff",".TIF",".TIFF"}
_DJI_INIT = False
log = logging.getLogger("pvrt")

def _ensure_dji():
    """
    Initialize DJI DIRP with an explicit, correct library path.
    Raises with a clear message if the lib is missing or wrong.
    """
    global _DJI_INIT
    if _DJI_INIT:
        return
    if not DIRP_LIB or not DIRP_LIB.exists():
        raise FileNotFoundError(
            f"DJI DIRP library not found. {describe_dirp()}. "
            f"Set PVRT_DIRP_LIB to the absolute path of your libdirp.so/.dll."
        )
    # This mirrors your previously working code: pass the exact .so/.dll path.
    dji_init(str(DIRP_LIB))
    log.info(f"[dji] Initialized DIRP using: {DIRP_LIB}")
    _DJI_INIT = True

def _list_rgb_images(split_dir: Path):
    for p in sorted(split_dir.iterdir()):
        if p.suffix in _IMG_EXTS and not p.name.endswith("_thermal.tif"):
            yield p

def scan_and_decode_split(split_dir: str | Path) -> Tuple[Path, Dict]:
    """
    Try to decode thermal from every image in `split_dir`.
    Writes thermal tiffs into split_dir/thermal/, plus pairs.json mapping:
        { "<abs/rgb.jpg>": "<abs/thermal.tif>", ... }

    Returns: (pairs_json_path, stats_dict)
    stats_dict = {"ok": int, "fail": int, "total": int, "first_error": Optional[str]}
    """
    split_dir = Path(split_dir)
    out_dir = split_dir / "thermal"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = {}
    ok = fail = 0
    first_error: Optional[str] = None

    try:
        _ensure_dji()
    except Exception as e:
        # If DJI init fails, record once and skip decoding all (so caller can decide fallback)
        tb = "".join(traceback.format_exception_only(type(e), e)).strip()
        first_error = tb
        # Log a very explicit diagnostic:
        log.error(f"[dji] INIT FAILED: {tb}. Details: {describe_dirp()}")
        return (out_dir / "pairs.json"), {"ok": 0, "fail": 0, "total": 0, "first_error": first_error}

    for rgb in _list_rgb_images(split_dir):
        try:
            # Prefer a dtype object, but the wrapper also accepts "float32"
            temps = rjpeg_to_heatmap(str(rgb), dtype=np.float32)
            if not isinstance(temps, np.ndarray) or temps.ndim != 2:
                raise ValueError("Thermal plane missing or invalid shape.")
            tpath = out_dir / f"{rgb.stem}_thermal.tif"
            tifffile.imwrite(str(tpath), temps.astype("float32"))
            pairs[str(rgb.resolve())] = str(tpath.resolve())
            ok += 1
        except Exception as e:
            if first_error is None:
                # Keep the first meaningful error to bubble up to UI
                first_error = "".join(traceback.format_exception_only(type(e), e)).strip()
            fail += 1

    pairs_path = out_dir / "pairs.json"
    pairs_path.write_text(json.dumps(pairs, indent=2))
    return pairs_path, {"ok": ok, "fail": fail, "total": ok + fail, "first_error": first_error}
