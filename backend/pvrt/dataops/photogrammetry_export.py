"""Finalize metadata-preserving undistorted exports for photogrammetry software."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Optional

from .image_metadata import write_corrected_gps


SUPPORTED_EXPORT_IMAGES = {".jpg", ".jpeg", ".png"}


def _find_camera_entry(camera_meta: dict[str, Any], image_name: str) -> Optional[dict[str, Any]]:
    name = Path(image_name).name.casefold()
    stem = Path(image_name).stem.casefold()
    for key, value in camera_meta.items():
        if str(key).startswith("__") or not isinstance(value, dict):
            continue
        key_path = Path(str(key))
        if key_path.name.casefold() == name or key_path.stem.casefold() == stem:
            return value
    return None


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def finalize_photogrammetry_export(
    *,
    export_dir: Path,
    camera_meta: dict[str, Any],
    alignment_report: dict[str, Any],
    camera_meta_path: Optional[Path] = None,
    alignment_report_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Embed corrected horizontal poses and write a portable ``geo.txt`` sidecar."""
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        raise FileNotFoundError("The undistorted photogrammetry export was not created.")

    alignment_images = alignment_report.get("images") or {}
    geo_lines = ["EPSG:4326"]
    embedded = 0
    aligned = 0
    retained = 0
    skipped: list[dict[str, str]] = []
    for image_path in sorted(export_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in SUPPORTED_EXPORT_IMAGES:
            continue
        entry = _find_camera_entry(camera_meta, image_path.name)
        if entry is None:
            skipped.append({"image": image_path.name, "reason": "camera metadata not found"})
            continue
        latitude = _finite_number(entry.get("lat"))
        longitude = _finite_number(entry.get("lon"))
        altitude = _finite_number(entry.get("absolute_altitude"))
        if latitude is None or longitude is None:
            skipped.append({"image": image_path.name, "reason": "GPS coordinates not found"})
            continue

        record = None
        for report_name, report_value in alignment_images.items():
            if Path(str(report_name)).stem.casefold() == image_path.stem.casefold():
                record = report_value if isinstance(report_value, dict) else None
                break
        is_aligned = bool(record and record.get("status") == "aligned")
        if is_aligned:
            write_corrected_gps(
                image_path,
                latitude=latitude,
                longitude=longitude,
                absolute_altitude=altitude,
            )
            embedded += 1
            aligned += 1
        else:
            retained += 1

        geo_row = f"{image_path.name} {longitude:.10f} {latitude:.10f}"
        if altitude is not None:
            geo_row += f" {altitude:.3f}"
        geo_lines.append(geo_row)

    geo_path = export_dir / "geo.txt"
    geo_path.write_text("\n".join(geo_lines) + "\n", encoding="utf-8")
    if camera_meta_path and Path(camera_meta_path).is_file():
        shutil.copy2(camera_meta_path, export_dir / "camera_meta.json")
    if alignment_report_path and Path(alignment_report_path).is_file():
        shutil.copy2(alignment_report_path, export_dir / "image_alignment.json")

    summary = {
        "version": 1,
        "coordinate_system": "EPSG:4326",
        "image_count": embedded + retained,
        "corrected_gps_embedded": embedded,
        "aligned_images": aligned,
        "original_gps_retained": retained,
        "skipped": skipped,
        "files": {
            "geo": geo_path.name,
            "camera_metadata": "camera_meta.json",
            "alignment_report": "image_alignment.json",
            "summary": "photogrammetry_export.json",
        },
        "notes": [
            "LightGlue corrections update horizontal latitude/longitude only.",
            "Absolute GPS altitude and all non-position camera metadata remain unchanged.",
            "Original dataset images were not modified.",
        ],
    }
    (export_dir / "photogrammetry_export.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
