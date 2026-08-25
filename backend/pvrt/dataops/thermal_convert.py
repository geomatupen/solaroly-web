"""Shared radiometric and standard image-to-grayscale conversion utilities."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
from PIL import Image

from ..config import DIRP_LIB, describe_dirp
from ..core.thermal import normalize_thermal
from .flir_rjpeg import extract_flir_raw, has_flir_fff
from .image_metadata import save_with_metadata

log = logging.getLogger("pvrt")

RADIOMETRIC_SOURCE_EXTS = {".jpg", ".jpeg"}
STANDARD_SOURCE_EXTS = {".jpg", ".jpeg", ".png"}
CONVERSION_TYPES = {"radiometric", "standard"}
SUPPORTED_CAMERA_INFO = (
    "DJI DIRP radiometric JPEG: M3T/M3TD and compatible DJI thermal models; "
    "FLIR FFF radiometric JPEG: DJI Zenmuse XT2 and compatible FLIR cameras."
)
_DJI_READY = False


def ensure_dirp_init() -> None:
    """Initialize the DJI DIRP SDK once for this process."""
    global _DJI_READY
    if _DJI_READY:
        return
    lib = Path(DIRP_LIB) if DIRP_LIB else None
    if not lib or not lib.exists():
        raise FileNotFoundError(
            f"DJI DIRP library not found. {describe_dirp()} "
            "Set PVRT_DIRP_LIB to the absolute path of your libdirp (.so/.dll)."
        )
    from dji_thermal_sdk.dji_sdk import dji_init

    dji_init(str(lib))
    log.info("[DJI] DIRP initialized: %s", lib)
    _DJI_READY = True


def inspect_thermal_source(source: Path) -> Dict[str, object]:
    """Classify one source without invoking either native thermal decoder."""
    source = Path(source)
    result: Dict[str, object] = {
        "file": source.name,
        "supported": False,
        "backend": None,
        "camera_model": None,
        "reason": None,
    }
    if not source.is_file() or source.suffix.lower() not in RADIOMETRIC_SOURCE_EXTS:
        result["reason"] = "not a JPG/JPEG source"
        return result
    try:
        with Image.open(source) as image:
            exif = image.getexif()
            make = str(exif.get(0x010F) or "").strip().rstrip("\x00")
            model = str(exif.get(0x0110) or "").strip().rstrip("\x00")
            description = str(exif.get(0x010E) or "").strip().rstrip("\x00")
        result["camera_model"] = model or make or "Unknown"
        if has_flir_fff(source):
            result.update(supported=True, backend="flir_fff")
            return result

        data = source.read_bytes()
        jpeg_end = data.find(b"\xff\xd9")
        trailing_bytes = len(data) - jpeg_end - 2 if jpeg_end >= 0 else 0
        model_upper = model.upper()
        dji_models = (
            "M3T", "M3TD", "MAVIC3", "M30T", "H20T", "H20N",
            "ZH20T", "ZH20N", "MAVIC2-ENTERPRISE-ADVANCED",
        )
        looks_radiometric = trailing_bytes > 65536 or description.lower() in {
            "ironred", "whitehot", "blackhot", "rainbow", "hotspot",
        }
        if make.upper() == "DJI" and any(token in model_upper for token in dji_models) and looks_radiometric:
            result.update(supported=True, backend="dji_dirp")
            return result
        result["reason"] = f"unsupported or non-radiometric camera payload ({model or make or 'unknown'})"
    except Exception as exc:
        result["reason"] = f"unreadable image metadata: {exc}"
    return result


def inspect_standard_source(source: Path, *, include_radiometric: bool = False) -> Dict[str, object]:
    """Check whether a JPG/JPEG/PNG can be decoded as a standard image."""
    source = Path(source)
    result: Dict[str, object] = {
        "file": source.name,
        "supported": False,
        "backend": "visible_grayscale",
        "camera_model": None,
        "reason": None,
        "excluded": None,
    }
    if not source.is_file() or source.suffix.lower() not in STANDARD_SOURCE_EXTS:
        result["reason"] = "not a JPG/JPEG/PNG source"
        return result
    try:
        if not include_radiometric and source.suffix.lower() in RADIOMETRIC_SOURCE_EXTS:
            thermal = inspect_thermal_source(source)
            if thermal["supported"]:
                result.update(
                    camera_model=thermal.get("camera_model"),
                    reason="radiometric JPEG; use Radiometric thermal JPEG mode",
                    excluded="radiometric",
                )
                return result
        with Image.open(source) as image:
            image.verify()
        result["supported"] = True
    except Exception as exc:
        result["reason"] = f"unreadable image: {exc}"
    return result


def scan_conversion_folder(
    input_dir: Path,
    *,
    conversion_type: str = "radiometric",
    include_radiometric: bool = False,
) -> Dict[str, object]:
    """Return mode-specific conversion eligibility for direct folder children."""
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input folder not found: {input_dir}")
    mode = str(conversion_type).strip().lower()
    if mode not in CONVERSION_TYPES:
        raise ValueError("Conversion type must be radiometric or standard.")
    source_exts = RADIOMETRIC_SOURCE_EXTS if mode == "radiometric" else STANDARD_SOURCE_EXTS
    candidates = [
        path for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in source_exts
    ]
    if mode == "radiometric":
        inspected = [inspect_thermal_source(path) for path in candidates]
    else:
        inspected = [
            inspect_standard_source(path, include_radiometric=include_radiometric)
            for path in candidates
        ]
    supported = [item for item in inspected if item["supported"]]
    excluded_radiometric = [item for item in inspected if item.get("excluded") == "radiometric"]
    unsupported = [
        item for item in inspected
        if not item["supported"] and item.get("excluded") != "radiometric"
    ]
    cameras: Dict[str, Dict[str, object]] = {}
    if mode == "radiometric":
        for item in supported:
            key = str(item.get("camera_model") or "Unknown")
            entry = cameras.setdefault(key, {"model": key, "backend": item["backend"], "count": 0})
            entry["count"] = int(entry["count"]) + 1
    file_types: Dict[str, int] = {}
    supported_names = {str(item["file"]) for item in supported}
    for path in candidates:
        if path.name not in supported_names:
            continue
        label = "JPG" if path.suffix.lower() in {".jpg", ".jpeg"} else "PNG"
        file_types[label] = file_types.get(label, 0) + 1
    known_image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    ignored_images = sum(
        1 for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in known_image_exts - source_exts
    )
    return {
        "input_dir": str(input_dir),
        "conversion_type": mode,
        "include_radiometric": bool(include_radiometric) if mode == "standard" else False,
        "candidate_total": len(candidates),
        "supported": len(supported),
        "unsupported": len(unsupported),
        "excluded_radiometric": len(excluded_radiometric),
        "ignored_images": ignored_images,
        "cameras": sorted(cameras.values(), key=lambda item: str(item["model"])),
        "file_types": file_types,
        "unsupported_samples": unsupported[:10],
        "excluded_radiometric_samples": excluded_radiometric[:10],
        "supported_files": [item["file"] for item in supported],
        "support_info": (
            SUPPORTED_CAMERA_INFO if mode == "radiometric"
            else "Standard visible-pixel conversion: readable JPG, JPEG, and PNG images."
        ),
    }


def scan_thermal_folder(input_dir: Path) -> Dict[str, object]:
    """Backward-compatible radiometric folder scan."""
    return scan_conversion_folder(input_dir, conversion_type="radiometric")


def convert_thermal_rjpeg(
    source: Path,
    output: Path,
    *,
    quality: int = 100,
    preserve_metadata: bool = True,
) -> Dict[str, float | str]:
    """Decode one supported DJI DIRP or FLIR FFF radiometric JPEG."""
    source = Path(source)
    output = Path(output)
    inspection = inspect_thermal_source(source)
    if not inspection["supported"]:
        raise ValueError(str(inspection["reason"] or "Unsupported thermal image."))
    if inspection["backend"] == "flir_fff":
        thermal_values = extract_flir_raw(source)
    else:
        ensure_dirp_init()
        from dji_thermal_sdk.utility import rjpeg_to_heatmap

        thermal_values = rjpeg_to_heatmap(str(source), dtype=np.float32)
    if not isinstance(thermal_values, np.ndarray) or thermal_values.ndim != 2:
        raise ValueError("Invalid thermal plane read from RJPEG.")
    gray = normalize_thermal(thermal_values)
    gray_image = Image.fromarray(gray, mode="L")
    with Image.open(source) as original:
        source_size = original.size
    if gray_image.size != source_size:
        raise ValueError(
            "Decoded thermal dimensions do not match the source image: "
            f"thermal={gray_image.size[0]}x{gray_image.size[1]}, "
            f"source={source_size[0]}x{source_size[1]}. "
            "Conversion was skipped to protect annotation alignment."
        )
    rgb_gray = Image.merge("RGB", (gray_image, gray_image, gray_image))
    if preserve_metadata:
        save_with_metadata(rgb_gray, source, output, max(1, min(100, int(quality))))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        rgb_gray.save(output, quality=max(1, min(100, int(quality))))
    return {
        "source": str(source),
        "output": str(output),
        "backend": str(inspection["backend"]),
        "camera_model": str(inspection["camera_model"] or "Unknown"),
        "min_value": float(np.nanmin(thermal_values)),
        "max_value": float(np.nanmax(thermal_values)),
    }


# Backward-compatible name used by existing training preparation code.
convert_dji_rjpeg = convert_thermal_rjpeg


def convert_standard_image(
    source: Path,
    output: Path,
    *,
    quality: int = 100,
    preserve_metadata: bool = True,
    include_radiometric: bool = False,
) -> Dict[str, float | str]:
    """Convert the visible pixels of one JPG/JPEG/PNG to RGB grayscale."""
    source = Path(source)
    output = Path(output)
    inspection = inspect_standard_source(source, include_radiometric=include_radiometric)
    if not inspection["supported"]:
        raise ValueError(str(inspection["reason"] or "Unsupported standard image."))
    with Image.open(source) as original:
        source_size = original.size
        gray_image = original.convert("L")
    if gray_image.size != source_size:
        raise ValueError("Converted dimensions do not match the source image.")
    rgb_gray = Image.merge("RGB", (gray_image, gray_image, gray_image))
    if preserve_metadata:
        save_with_metadata(rgb_gray, source, output, max(1, min(100, int(quality))))
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        rgb_gray.save(output, quality=max(1, min(100, int(quality))))
    return {
        "source": str(source),
        "output": str(output),
        "backend": "visible_grayscale",
        "camera_model": "Standard image",
        "min_value": 0.0,
        "max_value": 255.0,
    }


def convert_thermal_folder(
    input_dir: Path,
    output_dir: Path,
    *,
    output_format: str = "jpg",
    conversion_type: str = "radiometric",
    include_radiometric: bool = False,
    quality: int = 100,
    overwrite: bool = False,
    progress: Optional[Callable[[Dict[str, object]], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
) -> Dict[str, object]:
    """Convert supported direct-child images using the selected conversion mode."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    fmt = output_format.lower().lstrip(".")
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"jpg", "png"}:
        raise ValueError("Output format must be JPG or PNG.")
    mode = str(conversion_type).strip().lower()
    scan = scan_conversion_folder(
        input_dir,
        conversion_type=mode,
        include_radiometric=include_radiometric,
    )
    sources = [input_dir / name for name in scan["supported_files"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, object] = {
        "total": len(sources), "completed": 0, "converted": 0,
        "skipped": 0, "failed": 0, "first_error": None,
        "unsupported": scan["unsupported"], "ignored_images": scan["ignored_images"],
        "excluded_radiometric": scan["excluded_radiometric"],
        "cameras": scan["cameras"],
    }
    if progress:
        progress({**result, "current_file": None})

    for source in sources:
        if should_cancel and should_cancel():
            result["cancelled"] = True
            break
        destination = output_dir / f"{source.stem}.{fmt}"
        try:
            if destination.exists() and not overwrite:
                result["skipped"] = int(result["skipped"]) + 1
            else:
                if mode == "radiometric":
                    convert_thermal_rjpeg(source, destination, quality=quality, preserve_metadata=True)
                else:
                    convert_standard_image(
                        source,
                        destination,
                        quality=quality,
                        preserve_metadata=True,
                        include_radiometric=include_radiometric,
                    )
                result["converted"] = int(result["converted"]) + 1
        except Exception as exc:
            result["failed"] = int(result["failed"]) + 1
            if result["first_error"] is None:
                result["first_error"] = str(exc)
            log.warning("Thermal conversion failed for %s: %s", source.name, exc)
        result["completed"] = int(result["completed"]) + 1
        if progress:
            progress({**result, "current_file": source.name})
    return result


# Backward-compatible name for callers created before FLIR support was added.
convert_dji_folder = convert_thermal_folder


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert supported radiometric JPEGs to grayscale images.")
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg", dest="output_format")
    parser.add_argument("--type", choices=("radiometric", "standard"), default="radiometric", dest="conversion_type")
    parser.add_argument("--include-radiometric", action="store_true")
    parser.add_argument("--quality", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    stats = convert_thermal_folder(
        args.input_dir, args.output_dir, output_format=args.output_format,
        conversion_type=args.conversion_type,
        include_radiometric=args.include_radiometric,
        quality=args.quality, overwrite=args.overwrite,
        progress=lambda state: print(
            f"{state['completed']}/{state['total']} {state.get('current_file') or ''}", flush=True
        ),
    )
    print(stats)
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
