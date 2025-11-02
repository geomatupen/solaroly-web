# backend/pvrt/core/io.py
"""
I/O helpers used by all backends.

Keeps file-handling and small, re-usable utilities in one place so
individual backends (Detectron, YOLO, ...) stay lean.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable
import json


# ---------- JSON helpers ----------

def read_json_safe(path: Path) -> Dict[str, Any]:
    """
    Read a JSON file if it exists, otherwise return {}.
    Never raises on read/parse errors; returns {} instead.
    """
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_json_safe(path: Path, obj: Dict[str, Any]) -> None:
    """
    Write JSON atomically (best-effort): write to a temp file then replace.
    Falls back to direct write if replace fails (e.g., cross-device).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    tmp.write_text(data, encoding="utf-8")
    try:
        tmp.replace(path)
    except Exception:
        # Fall back to direct write
        path.write_text(data, encoding="utf-8")
        try:
            tmp.unlink(missing_ok=True)  # py>=3.8
        except Exception:
            pass


# ---------- Model meta helpers ----------

def load_model_meta(run_or_weights_dir: Path) -> Dict[str, Any]:
    """
    Load `model_meta.json` from a trained run directory.
    Returns {} if not found or invalid.
    """
    meta_path = Path(run_or_weights_dir) / "model_meta.json"
    return read_json_safe(meta_path)


def save_model_meta(run_dir: Path, meta: Dict[str, Any]) -> None:
    """
    Save `model_meta.json` to a trained run directory.
    """
    meta_path = Path(run_dir) / "model_meta.json"
    write_json_safe(meta_path, meta)


def input_mode_from_meta(meta: Dict[str, Any], default: str = "rgb") -> str:
    """
    Normalize the model's input mode from metadata.
    Expected values: "rgb" (3-band), "rgbt" (RGB+Thermal).
    """
    val = (meta.get("input_mode") or default).strip().lower()
    if val in {"rgbt", "rgb+t", "rgb_thermal", "thermal_rgb"}:
        return "rgbt"
    return "rgb"


def backend_name_from_meta(meta: Dict[str, Any], default: str = "detectron") -> str:
    """
    Extract which backend trained the model (e.g., 'detectron', 'yolo').
    """
    name = (meta.get("backend") or default).strip().lower()
    return name or default


# ---------- Thermal availability helpers ----------

THERMAL_DIR_CANDIDATES: Iterable[str] = ("thermal", "ir", "t", "temp")

def has_thermal_for_images(images_dir: Path) -> bool:
    """
    Heuristic to decide if thermal data is available for a set of images.
    Rules (in order):
      1) If a `thermal/pairs.json` exists - True.
      2) If any known thermal subdir contains files - True.
      3) Otherwise - False.
    """
    d = Path(images_dir)

    # 1) Explicit pairing file
    pairs = d / "thermal" / "pairs.json"
    if pairs.exists():
        try:
            j = read_json_safe(pairs)
            if j:  # any content signals presence
                return True
        except Exception:
            # ignore parse errors, fall through to scan
            pass

    # 2) Subdir scan
    for name in THERMAL_DIR_CANDIDATES:
        td = d / name
        if td.exists() and td.is_dir():
            # any file in subdir is considered a positive signal
            for _ in td.iterdir():
                return True

    return False


def images_are_single_channel(images_dir: Path, max_samples: int = 50) -> bool:
    """
    Determine whether the dataset's image files are single-channel (one band) by
    sampling up to `max_samples` image files and checking their band count via PIL.

    Returns True only if at least one image file is found and all sampled images
    have exactly one band. Returns False otherwise (including when no images
    could be read).
    """
    try:
        from PIL import Image
    except Exception:
        return False

    d = Path(images_dir)
    if not d.exists() or not d.is_dir():
        return False

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    count = 0
    any_found = False
    for p in sorted(d.rglob('*')):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        any_found = True
        try:
            with Image.open(p) as im:
                bands = im.getbands() or ()
                if len(bands) != 1:
                    return False
        except Exception:
            # unreadable file -> treat conservatively as multi-channel
            return False
        count += 1
        if count >= max_samples:
            break

    return any_found and count > 0


def prepare_dataset_for_run(src_train: Path, src_valid: Path, dest_run: Path, selected_bands: list | None, channel_count: int = 3) -> dict:
    """
    Prepare a per-run dataset directory under dest_run/prepared with consistent channels.

    - selected_bands: list of band identifiers, e.g. ['rgb','thermal'] or None => auto
    - channel_count: 1,3,4 desired. For channel_count==1 we will still provide 3-channel images
      to keep compatibility unless downstream explicitly supports 1-channel.

    Returns a dict with keys: train_dir, valid_dir, channel_count, selected_bands
    """
    from shutil import copy2
    from PIL import Image
    import os

    prepared_root = Path(dest_run) / "prepared"
    train_out = prepared_root / "train"
    valid_out = prepared_root / "valid"
    train_out.mkdir(parents=True, exist_ok=True)
    valid_out.mkdir(parents=True, exist_ok=True)

    # Simple band detection: assume 'rgb' images are top-level and thermal in subdir 'thermal'
    def _find_bands(src_dir: Path):
        bands = []
        if (src_dir / "thermal").exists():
            bands.append("thermal")
        # always include rgb if there are RGB files
        for p in src_dir.iterdir():
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                bands.append("rgb")
                break
        return bands

    sel = list(selected_bands) if selected_bands else _find_bands(src_train)
    # default channel_count sanity
    if channel_count not in (1,3,4):
        channel_count = 3

    # If the source dataset already carries thermal data and the user requested
    # thermal-aware channels (1 or 4) we can avoid creating a per-run copy and
    # use the existing folders in-place. This keeps your workspace small and
    # avoids duplicating large images. Backends are expected to handle paired
    # thermal/ files when present.
    try:
        # If a caller explicitly requests a 1-channel (thermal-only) run and the
        # source dataset contains thermal files, prepare a per-run folder where
        # thermal files are resampled/duplicated to exact image sizes so downstream
        # training/inference can rely on pixel alignment. This avoids mutating the
        # original dataset and keeps behavior explicit.
        if channel_count == 1 and has_thermal_for_images(src_train):
            # Build prepared output dirs
            from PIL import Image
            prepared_root = Path(dest_run) / "prepared"
            train_out = prepared_root / "train"
            valid_out = prepared_root / "valid"
            train_out.mkdir(parents=True, exist_ok=True)
            valid_out.mkdir(parents=True, exist_ok=True)

            def _find_thermal_candidate_for(stem: str, base: Path):
                # look for common thermal naming patterns
                exts = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
                tdir = base / 'thermal'
                # pairs.json mapping (best-effort)
                pj = tdir / 'pairs.json'
                if pj.exists():
                    try:
                        j = read_json_safe(pj)
                        rel = j.get(stem)
                        if rel:
                            cand = base / rel
                            if cand.exists():
                                return cand
                    except Exception:
                        pass
                # common candidates
                for e in exts:
                    c1 = tdir / f"{stem}{e}"
                    if c1.exists():
                        return c1
                    c2 = tdir / f"{stem}_thermal{e}"
                    if c2.exists():
                        return c2
                # sidecar next to image
                for e in exts:
                    cand = base / f"{stem}_thermal{e}"
                    if cand.exists():
                        return cand
                return None

            def _resample_and_write(thermal_p: Path, rgb_size: tuple[int, int], out_p: Path):
                """
                Conservative resample: try rasterio.reproject if available to preserve
                geotransform when source is GeoTIFF; otherwise fallback to PIL resize.
                Save as single-band PNG (L mode).
                """
                out_p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    try:
                        import rasterio
                        from rasterio.enums import Resampling
                        import numpy as np
                        with rasterio.open(thermal_p) as src:
                            # read first band, resample to target shape
                            data = src.read(1, out_shape=(rgb_size[1], rgb_size[0]), resampling=Resampling.bilinear)
                            # normalize to uint8 if needed
                            if data.dtype != 'uint8':
                                mn = float(data.min())
                                mx = float(data.max()) if float(data.max()) > mn else mn + 1.0
                                arr = ((data - mn) / (mx - mn) * 255.0).astype('uint8')
                            else:
                                arr = data.astype('uint8')
                            from PIL import Image
                            im = Image.fromarray(arr)
                            im = im.convert('L')
                            im.save(out_p)
                            return
                    except Exception:
                        # rasterio not available or failed - fallback to PIL
                        from PIL import Image
                        im = Image.open(thermal_p).convert('L')
                        im = im.resize(rgb_size, resample=Image.BILINEAR)
                        im.save(out_p)
                        return
                except Exception:
                    # final fallback: copy as-is
                    try:
                        from shutil import copy2
                        copy2(thermal_p, out_p)
                    except Exception:
                        pass

            # iterate over RGB-like files (prefer top-level images) and build thermal-only prepared images
            rgb_candidates = [p for p in sorted(src_train.iterdir()) if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
            if not rgb_candidates:
                # no RGB examples - try to use thermal files directly (copy into prepared)
                for t in sorted((src_train / 'thermal').iterdir() if (src_train / 'thermal').exists() else []):
                    if not t.is_file():
                        continue
                    outp = train_out / t.name
                    try:
                        from shutil import copy2
                        copy2(t, outp)
                    except Exception:
                        try: outp.write_bytes(t.read_bytes())
                        except Exception: pass
                # do the same for valid if available
                for t in sorted((src_valid / 'thermal').iterdir() if (src_valid / 'thermal').exists() else []):
                    if not t.is_file(): continue
                    outp = valid_out / t.name
                    try:
                        from shutil import copy2
                        copy2(t, outp)
                    except Exception:
                        try: outp.write_bytes(t.read_bytes())
                        except Exception: pass
            else:
                # For each RGB entry, try to find a thermal counterpart and resample to RGB size. If not found,
                # fallback to converting the RGB to grayscale (least-preferred).
                from PIL import Image
                for p in rgb_candidates:
                    stem = p.stem
                    rgb_size = (0, 0)
                    try:
                        with Image.open(p) as _im:
                            rgb_size = _im.size  # (w, h)
                    except Exception:
                        rgb_size = (512, 512)

                    therm = _find_thermal_candidate_for(stem, src_train)
                    outp = train_out / (stem + '.png')
                    if therm is not None and therm.exists():
                        _resample_and_write(therm, rgb_size, outp)
                    else:
                        # fallback: convert RGB to grayscale
                        try:
                            with Image.open(p) as _im:
                                gray = _im.convert('L')
                                gray = gray.resize(rgb_size, resample=Image.BILINEAR)
                                gray.save(outp)
                        except Exception:
                            try:
                                from shutil import copy2
                                copy2(p, outp)
                            except Exception:
                                try: outp.write_bytes(p.read_bytes())
                                except Exception: pass

                # Mirror the same logic for validation set
                if src_valid and src_valid.exists():
                    val_rgb = [p for p in sorted(src_valid.iterdir()) if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}]
                    for p in val_rgb:
                        stem = p.stem
                        try:
                            with Image.open(p) as _im:
                                rgb_size = _im.size
                        except Exception:
                            rgb_size = (512, 512)
                        therm = _find_thermal_candidate_for(stem, src_valid)
                        outp = valid_out / (stem + '.png')
                        if therm is not None and therm.exists():
                            _resample_and_write(therm, rgb_size, outp)
                        else:
                            try:
                                with Image.open(p) as _im:
                                    gray = _im.convert('L')
                                    gray = gray.resize(rgb_size, resample=Image.BILINEAR)
                                    gray.save(outp)
                            except Exception:
                                try:
                                    from shutil import copy2
                                    copy2(p, outp)
                                except Exception:
                                    try: outp.write_bytes(p.read_bytes())
                                    except Exception: pass

            return {"train_dir": str(train_out), "valid_dir": str(valid_out), "selected_bands": ['thermal'], "channel_count": 1}

        # For other cases (e.g., channel_count==4) keep the previous fast-return behavior
        if channel_count in (4,) and has_thermal_for_images(src_train):
            return {"train_dir": str(src_train), "valid_dir": str(src_valid), "selected_bands": sel, "channel_count": channel_count}
    except Exception:
        # if any check fails, fall through to perform prepare work
        pass

    def _copy_and_map(src_dir: Path, dst_dir: Path):
        # copy images by mapping selected bands -> output channels
        # For now: support combinations of 'rgb' and 'thermal'
        allowed_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        for p in sorted(src_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
            rel = p.name
            try:
                # thermal files are usually in a 'thermal' subdir
                is_thermal = ("/thermal/" in str(p).lower()) or (p.parent.name.lower() == "thermal")

                if is_thermal:
                    im = Image.open(p).convert("L")
                    if channel_count in (1, 3):
                        # duplicate thermal into RGB-compatible PNG
                        rgb = Image.merge("RGB", (im, im, im))
                        outp = dst_dir / (p.stem + ".png")
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        rgb.save(outp)
                    elif channel_count == 4:
                        # thermal-only -> duplicate into RGBA (thermal into RGB+alpha)
                        rgba = Image.merge("RGBA", (im, im, im, im))
                        outp = dst_dir / (p.stem + ".png")
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        rgba.save(outp)
                else:
                    # RGB or other visual file
                    if channel_count == 1:
                        im = Image.open(p).convert("RGB")
                        r = im.split()[0]
                        rgb = Image.merge("RGB", (r, r, r))
                        outp = dst_dir / (p.stem + ".png")
                        outp.parent.mkdir(parents=True, exist_ok=True)
                        rgb.save(outp)
                    elif channel_count == 3:
                        # prefer symlink to avoid copying large files
                        dst = dst_dir / rel
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            relpath = os.path.relpath(str(p), start=str(dst.parent))
                            os.symlink(relpath, dst)
                        except Exception:
                            copy2(p, dst)
                    elif channel_count == 4:
                        # try to find thermal partner next to RGB file
                        therm = p.parent / "thermal" / (p.stem + ".png")
                        if therm.exists():
                            t = Image.open(therm).convert("L")
                            r,g,b = Image.open(p).convert("RGB").split()
                            rgba = Image.merge("RGBA", (r, g, b, t))
                            outp = dst_dir / (p.stem + ".png")
                            outp.parent.mkdir(parents=True, exist_ok=True)
                            rgba.save(outp)
                        else:
                            # fallback: add opaque alpha channel
                            im = Image.open(p).convert("RGB")
                            r,g,b = im.split()
                            import numpy as np
                            a = Image.fromarray(np.full((im.size[1], im.size[0]), 255, dtype='uint8'))
                            rgba = Image.merge("RGBA", (r, g, b, a))
                            outp = dst_dir / (p.stem + ".png")
                            outp.parent.mkdir(parents=True, exist_ok=True)
                            rgba.save(outp)
            except Exception:
                # best-effort fallback: copy raw file
                try:
                    dst = dst_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    copy2(p, dst)
                except Exception:
                    pass

    _copy_and_map(src_train, train_out)
    _copy_and_map(src_valid, valid_out)

    return {"train_dir": str(train_out), "valid_dir": str(valid_out), "selected_bands": sel, "channel_count": channel_count}
