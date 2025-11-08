"""Shared thermal normalization helpers.

Provide a single canonical function `normalize_thermal(source)` which accepts
either a Path to a thermal file (TIFF/JPEG/PNG) or a numpy array and returns
an 8-bit (uint8) 2D numpy array suitable for stacking or visualization.

        Normalization policy:
         - If input is a uint8 image (previews produced by the decoder) return as-is.
         - Otherwise apply a 2..98 percentile stretch to 0..255 with a guard when
             p98 <= p2. Numeric TIFFs (float / >8-bit) are read with tifffile when
             available to preserve dtype and then normalized by percentile stretch.
 - If a multi-band TIFF is provided, prefer the first channel.
 - tifffile is used when available to preserve original numeric types.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from typing import Union

try:
    import tifffile
except Exception:
    tifffile = None


def _normalize_numeric_array(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    if a.size == 0:
        return np.zeros(a.shape[:2], dtype=np.uint8)
    # if multi-band, take first band
    if a.ndim == 3:
        a = a[..., 0]
    # Use a robust 2..98 percentile stretch for numeric arrays. This matches
    # the decoder/preview generation behavior which produces visually
    # consistent previews for both RJPEG-derived arrays and TIFFs. Percentile
    # stretching adapts to the distribution of values and ensures parity
    # between training-side previews and test/predict previews.
    vals = a.ravel()
    p2 = float(np.percentile(vals, 2)) if vals.size else 0.0
    p98 = float(np.percentile(vals, 98)) if vals.size else (p2 + 1.0)
    if p98 <= p2:
        p98 = p2 + 1.0
    scaled = np.clip((a - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    return scaled


def normalize_thermal(source: Union[Path, str, np.ndarray]) -> np.ndarray:
    """Return a uint8 2D numpy array normalized from `source`.

    `source` may be a Path/str pointing to a file (TIFF, PNG, JPG) or a
    numpy array already loaded. The function prefers numeric TIFF reads via
    tifffile when available, and treats uint8 arrays as already-stretched
    previews (returned unchanged).
    """
    # If given a numpy array, normalize/type-check directly
    if isinstance(source, np.ndarray):
        if source.dtype == np.uint8:
            arr = source
            if arr.ndim == 3:
                # collapse to single channel by taking first band
                arr = arr[..., 0]
            return arr.astype(np.uint8)
        return _normalize_numeric_array(source)

    # treat as path
    p = Path(str(source))
    ext = p.suffix.lower()
    # Prefer tifffile for TIFFs to preserve numeric dtype
    if ext in (".tif", ".tiff") and tifffile is not None:
        try:
            arr = tifffile.imread(str(p))
        except Exception:
            # fallback to PIL
            from PIL import Image
            arr = np.array(Image.open(p).convert("L"))
    else:
        # non-TIFF images are assumed to be previews (uint8). Use PIL to read
        try:
            from PIL import Image
            img = Image.open(p)
            arr = np.array(img.convert("L"))
        except Exception:
            # last resort: try numpy load
            from PIL import Image
            arr = np.array(Image.open(p).convert("L"))

    if arr.dtype == np.uint8:
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.uint8)
    return _normalize_numeric_array(arr)


def enhance_preview_for_display(u8: np.ndarray, contrast: float = 1.3, gamma: float = 1.2) -> np.ndarray:
    """Return a small, display-focused enhancement of a uint8 single-channel image.

    This is intentionally separate from `normalize_thermal` and MUST NOT be used
    for training data. It performs a gentle contrast stretch around mid-gray
    and a slight gamma correction so thumbnails/overlays appear more visually
    readable in the UI. Operates on 2D uint8 arrays and returns a 2D uint8.

    Parameters:
      contrast: multiplicative contrast factor around 128 (1.0 = no change)
      gamma:     gamma correction to apply (<=1 brightens, >1 darkens)
    """
    if not isinstance(u8, np.ndarray):
        raise TypeError("enhance_preview_for_display expects a numpy ndarray")
    if u8.dtype != np.uint8:
        # coerce but don't perform numeric normalization
        u8 = np.clip(u8, 0, 255).astype(np.uint8)
    # center contrast around mid-gray (128)
    arr = u8.astype(np.float32)
    arr = (arr - 128.0) * float(contrast) + 128.0
    arr = np.clip(arr, 0.0, 255.0)
    # optional gamma: map [0,255] -> [0,1] -> pow -> [0,255]
    if gamma is not None and gamma > 0 and abs(gamma - 1.0) > 1e-6:
        arr = 255.0 * np.power(arr / 255.0, float(gamma))
    out = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    return out
