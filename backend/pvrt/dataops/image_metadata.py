"""Metadata-preserving image encoding shared by conversion and correction workflows."""

from __future__ import annotations

import os
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Dict, Optional

import piexif
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


def _decimal_degrees_to_exif(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    total_seconds = Fraction(str(abs(float(value)))) * 3600
    degrees = int(total_seconds // 3600)
    remaining = total_seconds - degrees * 3600
    minutes = int(remaining // 60)
    seconds = (remaining - minutes * 60).limit_denominator(1_000_000)
    return ((degrees, 1), (minutes, 1), (seconds.numerator, seconds.denominator))


def _corrected_exif_bytes(
    source: Path,
    *,
    latitude: float,
    longitude: float,
    absolute_altitude: Optional[float] = None,
) -> bytes:
    with Image.open(source) as image:
        existing = image.info.get("exif")
        if not existing:
            image_exif = image.getexif()
            existing = image_exif.tobytes() if image_exif else None
    try:
        exif = piexif.load(existing) if existing else {
            "0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None,
        }
    except (ValueError, TypeError, piexif.InvalidImageDataError):
        exif = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}

    gps = exif.setdefault("GPS", {})
    gps[piexif.GPSIFD.GPSLatitudeRef] = b"N" if latitude >= 0 else b"S"
    gps[piexif.GPSIFD.GPSLatitude] = _decimal_degrees_to_exif(latitude)
    gps[piexif.GPSIFD.GPSLongitudeRef] = b"E" if longitude >= 0 else b"W"
    gps[piexif.GPSIFD.GPSLongitude] = _decimal_degrees_to_exif(longitude)

    # The alignment is horizontal only. Preserve the camera's existing absolute
    # GPS altitude; only restore it from parsed metadata when the tag is absent.
    if piexif.GPSIFD.GPSAltitude not in gps and absolute_altitude is not None:
        altitude = Fraction(abs(float(absolute_altitude))).limit_denominator(1000)
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if absolute_altitude >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = (altitude.numerator, altitude.denominator)
    return piexif.dump(exif)


def write_corrected_gps(
    image_path: Path,
    *,
    latitude: float,
    longitude: float,
    absolute_altitude: Optional[float] = None,
) -> None:
    """Update horizontal EXIF GPS on an exported JPG/PNG without touching source images."""
    image_path = Path(image_path)
    exif_bytes = _corrected_exif_bytes(
        image_path,
        latitude=latitude,
        longitude=longitude,
        absolute_altitude=absolute_altitude,
    )
    stat = image_path.stat()
    suffix = image_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        piexif.insert(exif_bytes, str(image_path))
        os.utime(image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        return
    if suffix != ".png":
        raise ValueError("Corrected GPS can only be embedded in JPG or PNG exports.")

    metadata = read_transferable_metadata(image_path)
    with Image.open(image_path) as source_image:
        image = source_image.copy()
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
    save_args = {"format": "PNG", "pnginfo": png_info, "exif": exif_bytes}
    for key in ("icc_profile", "dpi"):
        if metadata.get(key) is not None:
            save_args[key] = metadata[key]
    fd, temporary_name = tempfile.mkstemp(prefix=f".{image_path.stem}_gps_", suffix=".png", dir=image_path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, **save_args)
        os.replace(temporary, image_path)
        os.utime(image_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    finally:
        temporary.unlink(missing_ok=True)
