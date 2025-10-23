"""Simple preprocessor to merge RGB images with thermal band into 4-channel PNGs.

This writes RGBA PNGs where the alpha channel contains the (rescaled) thermal band.
This is a convenience helper; most training frameworks (including ultralytics) do
not automatically treat alpha as a model input channel — you'll need a custom
dataloader to actually load 4-channel inputs. The function is provided to make
it easy to generate merged images if you decide to implement a custom loader.
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image


def merge_rgb_with_thermal(images_dir: Path, out_dir: Path) -> int:
    """Scan `images_dir` for RGB files and a `thermal/` subdir with matching basenames.

    For each RGB image X and thermal image Y with same stem, create an RGBA PNG
    where RGB channels are from X and the A channel is a uint8 rescaled Y.
    Returns number of merged images written.
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    therm_dir = images_dir / "thermal"
    if not therm_dir.exists() or not therm_dir.is_dir():
        return 0

    count = 0
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue
        stem = p.stem
        t = therm_dir / f"{stem}.png"
        if not t.exists():
            t = therm_dir / f"{stem}.tif"
        if not t.exists():
            continue

        try:
            img = Image.open(p).convert("RGB")
            therm = Image.open(t).convert("L")
            # rescale thermal to 0..255
            a = np.array(therm).astype(np.float32)
            lo, hi = np.percentile(a, 2), np.percentile(a, 98)
            if hi <= lo: hi = lo + 1.0
            a = np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
            alpha = Image.fromarray(a, mode="L")
            rgba = Image.merge("RGBA", (*img.split(), alpha))
            out_path = out_dir / f"{stem}.png"
            rgba.save(out_path)
            count += 1
        except Exception:
            continue

    return count
