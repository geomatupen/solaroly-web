"""Compare thermal normalization between Detectron and YOLO pipelines.

Usage:
    python3 scripts/thermal_compare.py /path/to/images_dir /path/to/out_dir

This writes per-image stats to stdout and two grayscale previews per image:
  <stem>_detectron.png  (Detectron normalization)
  <stem>_yolo.png      (YOLO normalization)

The script implements the same normalization logic used by the code in this
repo so you can quickly eyeball differences.
"""
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
from PIL import Image

try:
    import tifffile
except Exception:
    tifffile = None

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
THERM_PREV_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


def find_thermal(images_dir: Path, rgb_path: Path) -> Path | None:
    tdir = images_dir / "thermal"
    pjson = tdir / "pairs.json"
    if pjson.exists():
        try:
            pairs = json.loads(pjson.read_text(encoding="utf-8"))
            target = pairs.get(rgb_path.name)
            if target:
                cand = (rgb_path.parent / target).resolve()
                if cand.exists():
                    return cand
        except Exception:
            pass
    # prefer previews
    for e in THERM_PREV_EXTS:
        cand = tdir / f"{rgb_path.stem}_thermal{e}"
        if cand.exists():
            return cand
        cand2 = tdir / f"{rgb_path.stem}{e}"
        if cand2.exists():
            return cand2
    # sidecar next to rgb
    for e in THERM_PREV_EXTS:
        cand = rgb_path.with_name(f"{rgb_path.stem}_thermal{e}")
        if cand.exists():
            return cand
    return None


# Detectron normalization: take a numeric array (float or int). If max>1.5 we
# assume Celsius and divide by 100; else use 2-98 percentile stretch.

def detectron_normalize(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    vmax = float(np.nanmax(a)) if a.size else 0.0
    if vmax > 1.5:
        a = np.clip(a, 0.0, 100.0) / 100.0
        out = (np.clip(a * 255.0, 0, 255)).astype(np.uint8)
        return out
    # else percentile stretch 2..98
    vals = a.ravel()
    if vals.size == 0:
        return np.zeros_like(a, dtype=np.uint8)
    p2 = float(np.percentile(vals, 2))
    p98 = float(np.percentile(vals, 98))
    if p98 <= p2:
        p98 = p2 + 1.0
    scaled = np.clip((a - p2) * (255.0 / (p98 - p2)), 0, 255).astype(np.uint8)
    return scaled


# YOLO normalization (repo change): if preview is JPG/PNG (uint8) use as-is;
# if TIFF numeric, apply same numeric normalization as Detectron.

def yolo_normalize(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext in (".png", ".jpg", ".jpeg"):
        try:
            return np.array(Image.open(path).convert("L"))
        except Exception:
            pass
    # else read numeric TIFF or fallback to PIL
    if ext in (".tif", ".tiff") and tifffile is not None:
        arr = tifffile.imread(str(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return detectron_normalize(arr)
    # generic fallback
    arr = np.array(Image.open(path).convert("L"))
    if arr.dtype == np.uint8:
        return arr
    return detectron_normalize(arr)


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/thermal_compare.py /path/to/images_dir /path/to/out_dir")
        return
    images_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    # load pairs.json if present
    for p in sorted(images_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        # skip generated thermal previews
        if p.name.endswith("_thermal.jpg") or p.name.endswith("_thermal.png"):
            continue
        t = find_thermal(images_dir, p)
        print("---\n", p.name, "->", t)
        if t is None or not t.exists():
            print(' no thermal preview found')
            continue
        # read using tifffile if possible
        ext = t.suffix.lower()
        if ext in (".tif", ".tiff") and tifffile is not None:
            arr = tifffile.imread(str(t))
            if arr.ndim == 3:
                arr = arr[..., 0]
            print(' TIFF numeric:', arr.dtype, 'min,max=', np.nanmin(arr), np.nanmax(arr))
        else:
            img = Image.open(t)
            arr = np.array(img.convert("L"))
            print(' Preview uint8:', arr.dtype, 'min,max=', arr.min(), arr.max())

        det = detectron_normalize(arr)
        ylo = yolo_normalize(t)

        # save side-by-side
        Image.fromarray(det).save(out_dir / f"{p.stem}_detectron.png")
        Image.fromarray(ylo).save(out_dir / f"{p.stem}_yolo.png")
        # print percentiles
        vals = arr.ravel()
        if vals.size:
            p2 = float(np.percentile(vals, 2))
            p98 = float(np.percentile(vals, 98))
        else:
            p2 = p98 = 0.0
        print(f" stats: p2={p2:.3f}, p98={p98:.3f}")


if __name__ == '__main__':
    main()
