#!/usr/bin/env python3
"""Compare thermal/predict images in two folders and print per-file stats.

Usage: python3 scripts/analyze_thermal_folders.py <folder_a> <folder_b>

Prints counts, number of matching basenames, and stats for up to 20 matching files.
"""
import sys
from pathlib import Path
import numpy as np

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import tifffile
except Exception:
    tifffile = None


def load_image(path: Path):
    ext = path.suffix.lower()
    if ext in ('.tif', '.tiff') and tifffile:
        arr = tifffile.imread(str(path))
        return np.asarray(arr)
    if Image:
        with Image.open(path) as im:
            return np.asarray(im)
    # fallback: try numpy.load for .npy
    if ext == '.npy':
        return np.load(path)
    # last resort: try tifffile even if import failed earlier
    try:
        import imageio
        arr = imageio.v3.imread(str(path))
        return np.asarray(arr)
    except Exception:
        raise RuntimeError(f"Unable to read image: {path}")


def stats_for(arr: np.ndarray):
    out = {}
    out['shape'] = tuple(arr.shape)
    out['dtype'] = str(arr.dtype)
    a = np.asarray(arr)
    if a.size == 0:
        out.update(min=None, max=None, mean=None, p2=None, p98=None)
        return out
    # if multi-channel, compute stats on grayscale conversion
    if a.ndim == 3 and a.shape[2] >= 3:
        # simple luminance
        a2 = a[..., :3].astype(np.float32)
        g = (0.2989*a2[...,0] + 0.5870*a2[...,1] + 0.1140*a2[...,2])
    elif a.ndim == 3 and a.shape[2] == 1:
        g = a[...,0].astype(float)
    else:
        g = a.astype(float)
    out['min'] = float(np.nanmin(g))
    out['max'] = float(np.nanmax(g))
    out['mean'] = float(np.nanmean(g))
    try:
        p2 = float(np.nanpercentile(g, 2))
        p98 = float(np.nanpercentile(g, 98))
    except Exception:
        p2, p98 = None, None
    out['p2'] = p2
    out['p98'] = p98
    return out


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    fa = Path(sys.argv[1])
    fb = Path(sys.argv[2])
    if not fa.exists() or not fb.exists():
        print('Folder missing:', fa, fb)
        sys.exit(2)
    files_a = sorted([p for p in fa.iterdir() if p.is_file()])
    files_b = sorted([p for p in fb.iterdir() if p.is_file()])
    print(f'Folder A: {fa}  files={len(files_a)}')
    print(f'Folder B: {fb}  files={len(files_b)}')

    basenames_a = {p.stem: p for p in files_a}
    basenames_b = {p.stem: p for p in files_b}
    common = sorted(set(basenames_a.keys()) & set(basenames_b.keys()))
    # If no exact stem matches, try matching by trailing sequence number like '_0001_T' or ending digits before _T
    def seq_key(stem: str):
        # try to find a trailing '_<digits>_T' or ending '_<digits>' pattern
        import re
        m = re.search(r"_(\d{3,6})_T$", stem)
        if m:
            return m.group(1)
        m2 = re.search(r"_(\d{3,6})$", stem)
        if m2:
            return m2.group(1)
        return None

    if not common:
        keys_a = {}
        keys_b = {}
        for s, p in basenames_a.items():
            k = seq_key(s)
            if k:
                keys_a.setdefault(k, []).append((s, p))
        for s, p in basenames_b.items():
            k = seq_key(s)
            if k:
                keys_b.setdefault(k, []).append((s, p))
        seq_common = sorted(set(keys_a.keys()) & set(keys_b.keys()))
        if seq_common:
            print(f'No exact stem matches, but found {len(seq_common)} matches by trailing sequence id. Showing first 20 of these matches.')
            # flatten into common-like list using composite key 'seq:<id>:<stemA>:<stemB>' but we'll handle printing below
            common_by_seq = []
            for k in seq_common[:20]:
                a_list = keys_a[k]
                b_list = keys_b[k]
                # take first from each list
                sa, pa = a_list[0]
                sb, pb = b_list[0]
                common_by_seq.append((k, sa, pa, sb, pb))
            # print these matches
            for i, (k, sa, pa, sb, pb) in enumerate(common_by_seq, 1):
                try:
                    arr_a = load_image(pa)
                    stats_a = stats_for(arr_a)
                except Exception as e:
                    stats_a = {'error': str(e)}
                try:
                    arr_b = load_image(pb)
                    stats_b = stats_for(arr_b)
                except Exception as e:
                    stats_b = {'error': str(e)}
                print(f'{i}. seq={k}')
                print('  A:', pa.name, '|', stats_a)
                print('  B:', pb.name, '|', stats_b)
                print('')
            return
    print(f'matching basenames: {len(common)}')
    if len(common) == 0:
        print('\nNo matching basenames found. Showing first 10 files in each folder:')
        print('\nFolder A sample:')
        for p in files_a[:10]:
            print(' ', p.name)
        print('\nFolder B sample:')
        for p in files_b[:10]:
            print(' ', p.name)
        return

    # analyze up to 20 matches
    N = min(20, len(common))
    print(f'Analyzing first {N} matching files (basename -> A_path | B_path)\n')
    for i, stem in enumerate(common[:N], 1):
        pa = basenames_a[stem]
        pb = basenames_b[stem]
        try:
            arr_a = load_image(pa)
            stats_a = stats_for(arr_a)
        except Exception as e:
            stats_a = {'error': str(e)}
        try:
            arr_b = load_image(pb)
            stats_b = stats_for(arr_b)
        except Exception as e:
            stats_b = {'error': str(e)}
        print(f'{i}. {stem}')
        print('  A:', pa.name, '|', stats_a)
        print('  B:', pb.name, '|', stats_b)
        print('')

if __name__ == '__main__':
    main()
