"""Simple preprocessor to merge RGB images with thermal band into 4-channel PNGs.

This writes RGBA PNGs where the alpha channel contains the (rescaled) thermal band.
This is a convenience helper; most training frameworks (including ultralytics) do
not automatically treat alpha as a model input channel — you'll need a custom
dataloader to actually load 4-channel inputs. The function is provided to make
it easy to generate merged images if you decide to implement a custom loader.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image


def merge_rgb_with_thermal(
    images_dir: Path,
    out_dir: Path,
    requested_channels: int = 3,
    use_thermal: bool = False,
    symlink: bool = False,
) -> int:
    """Prepare a YOLO-friendly dataset targeted at `requested_channels`.

        Behavior:
        - requested_channels == 3: include only RGB images (skip single-channel thermals).
        - requested_channels == 4 and use_thermal=True: include only images that have a
            decoded thermal sidecar; output RGBA images where A is the rescaled thermal band.
        - any other request (including 1) is treated as 3-channel RGB (thermal-as-RGB when
            where appropriate is handled by writing 3-channel grayscale images elsewhere).

    The function writes outputs preserving subdirectory structure relative to
    `images_dir` into `out_dir`. Label files with the same stem (``.txt``) are
    copied alongside images when present. Returns the number of output images
    written.
    """
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    therm_dir = images_dir / "thermal"
    written = 0

    # helper to find thermal sidecar for an image path
    def find_thermal(p: Path):
        # pairs.json
        pj = therm_dir / "pairs.json"
        if pj.exists():
            import json as _json
            try:
                pairs = _json.loads(pj.read_text(encoding="utf-8"))
                rel = pairs.get(p.name) if isinstance(pairs, dict) else None
                if rel:
                    cand = images_dir / rel
                    if cand.exists():
                        return cand
            except (_json.JSONDecodeError, OSError) as e:
                # malformed pairs.json or IO issue -> ignore mapping and fall back
                import logging
                logging.getLogger("pvrt").warning("malformed thermal/pairs.json ignored: %s", e)

        # common names in thermal dir
        # prefer image previews (PNG/JPG). We no longer look for single-band
        # TIFFs in the thermal/ folder — thermal previews are stored as JPG/PNG.
        for ext in (".png", ".jpg", ".jpeg"):
            cand = therm_dir / f"{p.stem}_thermal{ext}"
            if cand.exists():
                return cand
        for ext in (".png", ".jpg", ".jpeg"):
            cand = therm_dir / f"{p.stem}{ext}"
            if cand.exists():
                return cand

        # sidecar next to image
        for ext in (".png", ".jpg", ".jpeg"):
            cand = p.with_name(f"{p.stem}_thermal{ext}")
            if cand.exists():
                return cand

        # legacy fallback: prefer common image preview suffixes (do not
        # search for single-band TIFFs anymore)
        for ext in (".png", ".jpg", ".jpeg"):
            cand = p.with_name(f"{p.stem}_thermal{ext}")
            if cand.exists():
                return cand
        return None

    for src in images_dir.rglob("*"):
        if not src.is_file():
            continue
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            continue

        # preserve relative path
        rel = src.relative_to(images_dir)
        out_path = out_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(src) as im:
                mode = im.mode
                # determine if this is a single-channel thermal image
                is_single = mode in ("L", "I;16", "I")

                if requested_channels == 3:
                    # include only RGB-capable images
                    if is_single:
                        # skip single-channel thermal-only files
                        continue
                    # prefer to create a symlink to the original RGB file to
                    # avoid duplicating image bytes. If symlink is False or
                    # the filesystem doesn't support symlinks, fall back to
                    # writing a converted PNG.
                    target = out_path.with_suffix(src.suffix)
                    try:
                        if symlink:
                            # remove existing target if any
                            if target.exists() or target.is_symlink():
                                try:
                                    target.unlink()
                                except Exception as e:
                                    logging.getLogger("pvrt").debug("ignored preproc error: %s", e)
                            target.symlink_to(src.resolve())
                        else:
                            rgb = im.convert("RGB")
                            target = out_path.with_suffix(".png")
                            rgb.save(target)
                    except Exception:
                        # fallback: write converted PNG
                        try:
                            rgb = im.convert("RGB")
                            target = out_path.with_suffix(".png")
                            rgb.save(target)
                        except Exception:
                            continue
                    # copy label if exists
                    lbl = src.with_suffix(".txt")
                    if lbl.exists():
                        try:
                            (out_path.with_suffix(".txt")).write_bytes(lbl.read_bytes())
                        except Exception as e:
                            logging.getLogger("pvrt").debug("ignored preproc error: %s", e)
                    written += 1

                elif requested_channels == 4 and use_thermal:
                    # require thermal sidecar
                    t = find_thermal(src)
                    if t is None or not t.exists():
                        continue
                    # 4-channel output requires composing RGB+thermal into an
                    # RGBA file; symlinking is not possible because the file
                    # doesn't exist beforehand. Always write the RGBA PNG for
                    # requested_channels==4.
                    try:
                        rgb = im.convert("RGB")
                        with Image.open(t) as therm:
                            therm_l = therm.convert("L")
                            a = np.array(therm_l).astype(np.float32)
                            lo, hi = np.percentile(a, 2), np.percentile(a, 98)
                            if hi <= lo:
                                hi = lo + 1.0
                            a = np.clip((a - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
                            alpha = Image.fromarray(a, mode="L")
                            rgba = Image.merge("RGBA", (*rgb.split(), alpha))
                            target = out_path.with_suffix(".png")
                            # save RGBA
                            rgba.save(target)
                            lbl = src.with_suffix(".txt")
                            if lbl.exists():
                                try:
                                    (out_path.with_suffix(".txt")).write_bytes(lbl.read_bytes())
                                except Exception as e:
                                    logging.getLogger("pvrt").debug("ignored preproc error: %s", e)
                            written += 1
                    except Exception:
                        continue
                else:
                    # any other combination: treat as 3-channel RGB behaviour
                    # (covers requested==1 coerced to 3 or other unsupported values)
                    # try to symlink/convert as RGB as above
                    target = out_path.with_suffix(src.suffix)
                    try:
                        if symlink:
                            if target.exists() or target.is_symlink():
                                try:
                                    target.unlink()
                                except Exception as e:
                                    logging.getLogger("pvrt").debug("ignored preproc error: %s", e)
                            target.symlink_to(src.resolve())
                        else:
                            rgb = im.convert("RGB")
                            target = out_path.with_suffix(".png")
                            rgb.save(target)
                    except Exception:
                        try:
                            rgb = im.convert("RGB")
                            target = out_path.with_suffix(".png")
                            rgb.save(target)
                        except Exception:
                            continue
                    lbl = src.with_suffix(".txt")
                    if lbl.exists():
                        try:
                            (out_path.with_suffix(".txt")).write_bytes(lbl.read_bytes())
                        except Exception as e:
                            logging.getLogger("pvrt").debug("ignored preproc error: %s", e)
                    written += 1

        except Exception:
            # ignore problematic files
            continue

    return written
