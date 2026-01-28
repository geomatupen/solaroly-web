#!/usr/bin/env python3
"""
Compare thermal images between two folders and non-destructively regenerate normalized copies using the canonical normalize_thermal().
Usage:
  python3 scripts/compare_and_regen_normalized.py <dir_a> <dir_b> <out_dir>

It will look for matching stems (filename without extension, with optional trailing '_thermal' stripped) and write normalized PNGs to out_dir/<a|b>/<stem>_norm.png
Prints a summary of comparisons and top diffs.
"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# ensure repository root on path to import normalize_thermal
repo_root = Path(__file__).resolve().parents[1]
import sys
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from backend.pvrt.core.thermal import normalize_thermal
except Exception as e:
    print("ERROR: could not import normalize_thermal:", e)
    sys.exit(2)


def list_image_files(d):
    files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        files.extend(Path(d).rglob(ext))
    return files


def canonical_stem(p: Path):
    s = p.stem
    if s.endswith('_thermal'):
        return s[:-8]
    return s


def to_gray_uint8(arr):
    # arr may be PIL Image, numpy array, or path
    if isinstance(arr, (str, Path)):
        arr = Image.open(str(arr)).convert('L')
        a = np.array(arr)
    elif isinstance(arr, Image.Image):
        a = np.array(arr.convert('L'))
    else:
        a = np.array(arr)
        if a.ndim == 3:
            # convert to luminance-like
            a = a[..., 0]
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a


def stats(a):
    return {
        'min': int(np.min(a)),
        'max': int(np.max(a)),
        'mean': float(np.mean(a)),
        'p2': float(np.percentile(a, 2)),
        'p98': float(np.percentile(a, 98)),
    }


def main():
    if len(sys.argv) < 4:
        print('Usage: compare_and_regen_normalized.py <dir_a> <dir_b> <out_dir>')
        sys.exit(1)
    dir_a = Path(sys.argv[1])
    dir_b = Path(sys.argv[2])
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    files_a = list_image_files(dir_a)
    files_b = list_image_files(dir_b)

    map_a = {canonical_stem(p): p for p in files_a}
    map_b = {canonical_stem(p): p for p in files_b}

    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    print(f'Found {len(files_a)} images in A ({dir_a}), {len(files_b)} in B ({dir_b}), {len(common)} common stems')

    results = []
    for stem in common:
        pa = map_a[stem]
        pb = map_b[stem]
        try:
            na = normalize_thermal(str(pa))
        except Exception:
            # fallback to opening as image
            na = to_gray_uint8(pa)
        try:
            nb = normalize_thermal(str(pb))
        except Exception:
            nb = to_gray_uint8(pb)

        # ensure shapes match; if not, resize nb to na shape using PIL
        if na.shape != nb.shape:
            # resize NB to NA
            nb_img = Image.fromarray(to_gray_uint8(nb))
            nb_img = nb_img.resize((na.shape[1], na.shape[0]), Image.BILINEAR)
            nb = np.array(nb_img).astype(np.uint8)

        out_a_dir = out_dir / 'A'
        out_b_dir = out_dir / 'B'
        out_a_dir.mkdir(parents=True, exist_ok=True)
        out_b_dir.mkdir(parents=True, exist_ok=True)

        out_a = out_a_dir / f'{stem}_A_norm.png'
        out_b = out_b_dir / f'{stem}_B_norm.png'
        Image.fromarray(np.clip(na,0,255).astype(np.uint8)).save(out_a)
        Image.fromarray(np.clip(nb,0,255).astype(np.uint8)).save(out_b)

        diff = np.abs(na.astype(np.int16) - nb.astype(np.int16))
        res = {
            'stem': stem,
            'a': str(pa),
            'b': str(pb),
            'stats_a': stats(na),
            'stats_b': stats(nb),
            'mean_abs_diff': float(np.mean(diff)),
            'max_abs_diff': int(np.max(diff)),
            'out_a': str(out_a),
            'out_b': str(out_b),
        }
        results.append(res)

    if not results:
        print('No matching images to compare.')
        return

    # summary
    diffs = sorted(results, key=lambda r: r['mean_abs_diff'], reverse=True)
    print('\nTop differences (by mean abs diff):')
    for r in diffs[:10]:
        print(f"{r['stem']}: mean_diff={r['mean_abs_diff']:.3f} max_diff={r['max_abs_diff']} meanA={r['stats_a']['mean']:.2f} meanB={r['stats_b']['mean']:.2f} -> saved {r['out_a']} and {r['out_b']}")

    eq_count = sum(1 for r in results if r['max_abs_diff'] == 0)
    within1 = sum(1 for r in results if r['mean_abs_diff'] < 1.0)
    print(f"\nCompared {len(results)} images: exactly_equal={eq_count}, mean_diff<1.0: {within1}")

if __name__ == '__main__':
    main()
