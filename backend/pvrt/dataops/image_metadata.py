"""Metadata-preserving image encoding shared by conversion and correction workflows."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict

from PIL import Image, PngImagePlugin


def read_transferable_metadata(source: Path) -> Dict[str, object]:
    """Read standard metadata Pillow can safely carry into newly encoded pixels."""
    with Image.open(source) as original:
        info = original.info.copy()
        exif = info.get("exif")
        if not exif:
            try:
                source_exif = original.getexif()
                exif = source_exif.tobytes() if source_exif else None
            except (AttributeError, OSError, ValueError):
                exif = None
        return {
            "exif": exif,
            "icc_profile": info.get("icc_profile"),
            "dpi": info.get("dpi"),
            "comment": info.get("comment"),
            "xmp": info.get("xmp") or info.get("XML:com.adobe.xmp"),
        }


def save_with_metadata(image: Image.Image, source: Path, output: Path, quality: int = 100) -> None:
    """Encode pixels while retaining transferable camera metadata and file timestamps."""
    metadata = read_transferable_metadata(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    common = {
        key: value
        for key, value in metadata.items()
        if key in {"exif", "icc_profile", "dpi"} and value is not None
    }

    if suffix in {".jpg", ".jpeg"}:
        jpeg_image = image.convert("RGB") if image.mode not in {"RGB", "L", "CMYK"} else image
        jpeg_args = {
            **common,
            "format": "JPEG",
            "quality": max(1, min(100, int(quality))),
            "subsampling": 0,
        }
        if metadata.get("comment") is not None:
            jpeg_args["comment"] = metadata["comment"]
        if metadata.get("xmp") is not None:
            jpeg_args["xmp"] = metadata["xmp"]
        jpeg_image.save(output, **jpeg_args)
    elif suffix == ".png":
        png_info = PngImagePlugin.PngInfo()
        if metadata.get("xmp") is not None:
            xmp = metadata["xmp"]
            if isinstance(xmp, bytes):
                xmp = xmp.decode("utf-8", errors="replace")
            png_info.add_itxt("XML:com.adobe.xmp", str(xmp))
        if metadata.get("comment") is not None:
            comment = metadata["comment"]
            if isinstance(comment, bytes):
                comment = comment.decode("utf-8", errors="replace")
            png_info.add_text("Comment", str(comment))
        image.save(output, format="PNG", pnginfo=png_info, **common)
    else:
        raise ValueError("Output format must be JPG or PNG.")

    shutil.copystat(source, output)
