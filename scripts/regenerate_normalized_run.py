#!/usr/bin/env python3
"""
Regenerate normalized thermal previews and 3-channel thermal-as-RGB images for a merged_thermal_3ch run non-destructively.
Usage:
  python3 scripts/regenerate_normalized_run.py <merged_run_dir> <out_root>

It will look for raw .tif sources under data/**/thermal matching each stem and use normalize_thermal on them; if no raw TIFF found, it will try to normalize the existing run preview by forcing percentile stretch.
"""
import sys
from pathlib import Path
import shutil
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
    # search data/**/thermal/<stem>_thermal.tif
    root = Path('data')
    pattern = f"**/{stem}_thermal.tif"
    matches = list(root.glob(pattern))
    return matches[0] if matches else None


def force_normalize_from_uint8(path_or_arr):
    # fallback when only a uint8 preview exists: perform 2-98 percentile stretch to 0..255
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


def make_3ch(img8):
    if img8.ndim == 2:
        return np.stack([img8, img8, img8], axis=-1)
    if img8.ndim == 3 and img8.shape[2] == 3:
        return img8
    # otherwise collapse channels
    return np.stack([img8[...,0]]*3, axis=-1)


def main():
    if len(sys.argv) < 3:
        print('Usage: regenerate_normalized_run.py <merged_run_dir> <out_root>')
        sys.exit(1)
    merged_run_dir = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    if not merged_run_dir.exists():
        print('merged_run_dir does not exist:', merged_run_dir)
        sys.exit(2)

    # process train/ and valid/ subdirs if present
    for split in ['train', 'valid']:
        src_split = merged_run_dir / split
        if not src_split.exists():
            continue
        dst_split = out_root / split
        dst_split.mkdir(parents=True, exist_ok=True)
        dst_thermal = dst_split / 'thermal'
        dst_thermal.mkdir(parents=True, exist_ok=True)

        # iterate PNG/JPEG files in src_split (ignore _thermal images in subfolder)
        for p in sorted(src_split.iterdir()):
            if p.is_dir():
                continue
            if not p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                continue
            stem = p.stem
            # strip trailing _thermal if present
            if stem.endswith('_thermal'):
                stem_base = stem[:-8]
            else:
                # if name ends with _T or similar, keep as-is
                stem_base = stem
            # look for raw tif
            raw = find_raw_tif_for_stem(stem_base)
            if raw:
                try:
                    norm = normalize_thermal(str(raw))
                except Exception:
                    norm = force_normalize_from_uint8(raw)
            else:
                # fallback: normalize from the existing preview
                norm = force_normalize_from_uint8(p)

            # save thermal preview
            out_therm = dst_thermal / f'{stem_base}_T_thermal.png'
            Image.fromarray(norm).save(out_therm)

            # make 3-channel thermal-as-RGB and save to dst_split with same name as original if original ends with _T
            out_3ch = dst_split / f'{stem_base}_T.png'
            img3 = make_3ch(norm)
            Image.fromarray(img3).save(out_3ch)

            print(f'Wrote normalized for {stem_base}: raw={raw} -> {out_3ch}')

if __name__ == '__main__':
    main()
