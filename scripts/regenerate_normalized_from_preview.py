#!/usr/bin/env python3
"""
Normalize a folder of preview images by locating raw .tif sources (data/**/<stem>_thermal.tif) and applying normalize_thermal; writes to out_dir with same filenames (png/jpeg) but normalized.
Usage:
  python3 scripts/regenerate_normalized_from_preview.py <preview_dir> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    from backend.pvrt.core.thermal import normalize_thermal
except Exception as e:
    print('ERROR: could not import normalize_thermal:', e)
    sys.exit(2)


def find_raw_tif_for_stem(stem):
    root = Path('data')
    pattern = f"**/{stem}_thermal.tif"
    matches = list(root.glob(pattern))
    return matches[0] if matches else None


def force_normalize_from_uint8(path_or_arr):
    if isinstance(path_or_arr, (str, Path)):
        img = Image.open(str(path_or_arr)).convert('L')
        a = np.array(img).astype(np.float32)
    else:
        a = np.array(path_or_arr).astype(np.float32)
    p2 = np.percentile(a, 2)
    p98 = np.percentile(a, 98)
    if p98 <= p2:
        p2, p98 = a.min(), a.max()
    out = (a - p2) / max(1e-6, (p98 - p2))
    out = np.clip(out, 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def main():
    if len(sys.argv) < 3:
        print('Usage: regenerate_normalized_from_preview.py <preview_dir> <out_dir>')
        sys.exit(1)
    preview_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(preview_dir.iterdir()):
        if p.is_dir():
            continue
        stem = p.stem
        raw = find_raw_tif_for_stem(stem)
        if raw:
            try:
                norm = normalize_thermal(str(raw))
            except Exception:
                norm = force_normalize_from_uint8(raw)
        else:
            norm = force_normalize_from_uint8(p)
        outp = out_dir / p.name
        Image.fromarray(norm).save(outp)
        print('Wrote', outp, 'from raw=', raw)

if __name__ == '__main__':
    main()
