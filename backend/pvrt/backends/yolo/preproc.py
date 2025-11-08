"""Simple preprocessor to merge RGB images with thermal band into 4-channel PNGs.

This writes RGBA PNGs where the alpha channel contains the (rescaled) thermal band.
This is a convenience helper; most training frameworks (including ultralytics)
do not automatically treat alpha as a model input channel — a custom
dataloader is required to load 4-channel inputs. The function makes it easy
to generate merged images if a custom loader is implemented.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Iterable
import numpy as np
from PIL import Image
import json
from ...core.thermal import normalize_thermal


def merge_rgb_with_thermal(
    images_dir: Path,
    out_dir: Path,
    requested_channels: int = 3,
    use_thermal: bool = False,
    symlink: bool = False,
    thermal_as_rgb: bool = False,
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
            try:
                pairs = json.loads(pj.read_text(encoding="utf-8"))
                rel = pairs.get(p.name) if isinstance(pairs, dict) else None
                if rel:
                    cand = images_dir / rel
                    if cand.exists():
                        return cand
            except (json.JSONDecodeError, OSError) as e:
                # malformed pairs.json or IO issue -> warn and fall back
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

        with Image.open(src) as im:
                mode = im.mode
                # determine if this is a single-channel thermal image
                is_single = mode in ("L", "I;16", "I")

                if requested_channels == 3:
                    # Two modes for 3-channel output:
                    # - thermal_as_rgb == False: include only RGB-capable images
                    # - thermal_as_rgb == True: produce 3-channel grayscale
                    #   images from thermal previews when available; skip
                    #   images without thermal previews so the trainer sees a
                    #   consistent thermal-only dataset.
                    if not thermal_as_rgb:
                        # include only RGB-capable images
                        if is_single:
                            # skip single-channel thermal-only files
                            continue
                        # prefer to create a symlink to the original RGB file to
                        # avoid duplicating image bytes. If symlink is False or
                        # the filesystem doesn't support symlinks, fall back to
                        # writing a converted PNG.
                        target = out_path.with_suffix(src.suffix)
                        if symlink:
                            # remove existing target if any
                            if target.exists() or target.is_symlink():
                                target.unlink()
                            target.symlink_to(src.resolve())
                        else:
                            rgb = im.convert("RGB")
                            target = out_path.with_suffix(".png")
                            rgb.save(target)
                        # copy label if exists
                        lbl = src.with_suffix(".txt")
                        if lbl.exists():
                            (out_path.with_suffix(".txt")).write_bytes(lbl.read_bytes())
                        written += 1
                    else:
                        # thermal_as_rgb: produce 3-channel grayscale images
                        # for images that have thermal previews; skip others.
                        t = find_thermal(src)
                        if t is None or not t.exists():
                            continue
                        # If caller asked for symlinks and the thermal preview is
                        # already a 3-channel image we can symlink directly to
                        # avoid duplicating bytes. Otherwise, compose a 3-channel
                        # grayscale image using the canonical normalizer and write
                        # the output into the destination.
                        try:
                            with Image.open(t) as ti:
                                bands = ti.getbands()
                                is_rgb_preview = bands and len(bands) >= 3
                        except Exception:
                            is_rgb_preview = False

                        if symlink and is_rgb_preview:
                            # create parent dirs then symlink the preview file
                            target = out_path.with_suffix(t.suffix)
                            if target.exists() or target.is_symlink():
                                target.unlink()
                            try:
                                target.symlink_to(t.resolve())
                            except Exception:
                                # fallback to copying if symlink creation fails
                                with Image.open(t) as ti:
                                    ti.convert("RGB").save(target.with_suffix('.png'))
                        else:
                            # compose grayscale 3-channel image from thermal preview
                            # Use shared normalization helper which handles TIFF numeric
                            # arrays (via tifffile when available) and uint8 previews.
                            try:
                                a8 = normalize_thermal(t)
                            except Exception:
                                a8 = np.array(Image.open(t).convert("L"))
                            gray = Image.fromarray(a8, mode="L")
                            rgb_out = Image.merge("RGB", (gray, gray, gray))
                            target = out_path.with_suffix(".png")
                            rgb_out.save(target)
                        lbl = src.with_suffix(".txt")
                        if lbl.exists():
                            (out_path.with_suffix(".txt")).write_bytes(lbl.read_bytes())
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
                    rgb = im.convert("RGB")
                    # Similar normalization for RGBA alpha channel: prefer
                    # reading TIFFs as numeric arrays and normalizing; for
                    # JPG/PNG previews use the existing 8-bit values.
                    try:
                        a8 = normalize_thermal(t)
                    except Exception:
                        a8 = np.array(Image.open(t).convert("L"))
                    alpha = Image.fromarray(a8, mode="L")
                    rgba = Image.merge("RGBA", (*rgb.split(), alpha))
                    target = out_path.with_suffix(".png")
                    # save RGBA
                    rgba.save(target)
                    lbl = src.with_suffix(".txt")
                    if lbl.exists():
                        (out_path.with_suffix(".txt")).write_bytes(lbl.read_bytes())
                    written += 1
                else:
                    # any other combination: treat as 3-channel RGB behaviour
                    # (covers requested==1 coerced to 3 or other unsupported values)
                    # try to symlink/convert as RGB as above
                    target = out_path.with_suffix(src.suffix)
                    if symlink:
                        if target.exists() or target.is_symlink():
                            target.unlink()
                        target.symlink_to(src.resolve())
                    else:
                        rgb = im.convert("RGB")
                        target = out_path.with_suffix(".png")
                        rgb.save(target)
                    lbl = src.with_suffix(".txt")
                    if lbl.exists():
                        (out_path.with_suffix(".txt")).write_bytes(lbl.read_bytes())
                    written += 1

        # NOTE: let exceptions propagate for problematic files so callers
        # can see and handle failures instead of silently skipping them.

    return written
