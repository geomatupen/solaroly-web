"""Align prepared north-up thermal images to mapped solar-row geometry.

The input images are the exact post-undistortion, post-rotation files used by
inference.  Alignment does not resample them again.  Instead it refines each
image centre and records a small residual map rotation that downstream
pixel-to-world projection applies to both the image footprint and predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image


ANALYSIS_MAX_DIMENSION = 384
MIN_ROW_PIXELS = 45
MIN_ALIGNMENT_SCORE = 0.28
MIN_CORRECTION_IMPROVEMENT = 0.018
MIN_AMBIGUITY_GAP = 0.008


@dataclass(frozen=True)
class RowAlignmentOptions:
    maximum_position_correction_m: float = 8.0
    maximum_rotation_correction_deg: float = 10.0


def _iter_coordinate_lines(geometry: dict[str, Any]) -> Iterable[list[list[float]]]:
    kind = str(geometry.get("type") or "")
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return
    if kind == "LineString":
        yield coordinates
    elif kind == "MultiLineString":
        yield from coordinates
    elif kind == "Polygon":
        yield from coordinates
    elif kind == "MultiPolygon":
        for polygon in coordinates:
            yield from polygon


def _load_row_lines(path: Path) -> list[np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") != "FeatureCollection":
        raise ValueError("Rows GeoJSON must be a FeatureCollection.")
    lines: list[np.ndarray] = []
    for feature in payload.get("features") or []:
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(geometry, dict):
            continue
        for coordinates in _iter_coordinate_lines(geometry):
            try:
                line = np.asarray(coordinates, dtype=np.float64)[:, :2]
            except (TypeError, ValueError, IndexError):
                continue
            finite = np.isfinite(line).all(axis=1)
            line = line[finite]
            if len(line) >= 2:
                lines.append(line)
    if not lines:
        raise ValueError("The selected Rows GeoJSON contains no usable line or polygon boundaries.")
    return lines


def _entry_for_image(camera_meta: dict[str, Any], image_name: str) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    stem = Path(image_name).stem.casefold()
    for key, value in camera_meta.items():
        if str(key).startswith("__") or not isinstance(value, dict):
            continue
        if Path(str(key)).stem.casefold() == stem:
            return str(key), value
    return None, None


def _thermal_edge_proximity(path: Path) -> tuple[np.ndarray, float, tuple[int, int]]:
    with Image.open(path) as source:
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    height, width = rgba.shape[:2]
    scale = min(1.0, ANALYSIS_MAX_DIMENSION / max(height, width))
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    if scale < 1.0:
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    valid = alpha > 32
    valid_values = gray[valid]
    median = float(np.median(valid_values)) if valid_values.size else 96.0
    lower = int(max(12.0, 0.55 * median))
    upper = int(min(245.0, max(lower + 24.0, 1.45 * median)))
    edges = cv2.Canny(gray, lower, upper)
    edges[~valid] = 0
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    distance = cv2.distanceTransform(np.where(edges > 0, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
    proximity = np.exp(-distance / 2.75).astype(np.float32)
    proximity[~valid] = 0.0
    return proximity, scale, (width, height)


def _local_lines(
    lines: list[np.ndarray], lat: float, lon: float, radius_m: float,
) -> list[np.ndarray]:
    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = max(1.0, metres_per_degree_lat * math.cos(math.radians(lat)))
    selected: list[np.ndarray] = []
    for line in lines:
        east = (line[:, 0] - lon) * metres_per_degree_lon
        north = (line[:, 1] - lat) * metres_per_degree_lat
        if (
            float(np.max(east)) < -radius_m or float(np.min(east)) > radius_m
            or float(np.max(north)) < -radius_m or float(np.min(north)) > radius_m
        ):
            continue
        selected.append(np.column_stack((east, north)))
    return selected


def _render_rows(
    local_lines: list[np.ndarray], shape: tuple[int, int], analysis_mpp: float, rotation_deg: float,
) -> np.ndarray:
    height, width = shape
    centre_x, centre_y = width / 2.0, height / 2.0
    angle = math.radians(rotation_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    canvas = np.zeros((height, width), dtype=np.uint8)
    for line in local_lines:
        east, north = line[:, 0], line[:, 1]
        pixel_x = centre_x + (cosine * east - sine * north) / analysis_mpp
        pixel_y = centre_y + (-sine * east - cosine * north) / analysis_mpp
        points = np.rint(np.column_stack((pixel_x, pixel_y))).astype(np.int32)
        cv2.polylines(canvas, [points], False, 1, thickness=2, lineType=cv2.LINE_AA)
    return canvas


def _second_peak(response: np.ndarray, best_x: int, best_y: int, exclusion_px: int) -> float:
    copy = response.copy()
    x0, x1 = max(0, best_x - exclusion_px), min(copy.shape[1], best_x + exclusion_px + 1)
    y0, y1 = max(0, best_y - exclusion_px), min(copy.shape[0], best_y + exclusion_px + 1)
    copy[y0:y1, x0:x1] = -np.inf
    finite = copy[np.isfinite(copy)]
    return float(np.max(finite)) if finite.size else 0.0


def _align_one(
    image_path: Path,
    entry: dict[str, Any],
    row_lines: list[np.ndarray],
    options: RowAlignmentOptions,
) -> dict[str, Any]:
    try:
        lat = float(entry["lat"])
        lon = float(entry["lon"])
        metres_per_pixel = float(entry["meters_per_pixel"])
    except (KeyError, TypeError, ValueError):
        return {"status": "skipped_metadata", "reason": "GPS or GSD metadata is unavailable."}
    if not math.isfinite(metres_per_pixel) or metres_per_pixel <= 0:
        return {"status": "skipped_metadata", "reason": "A valid image GSD is unavailable."}

    proximity, scale, original_size = _thermal_edge_proximity(image_path)
    analysis_mpp = metres_per_pixel / scale
    height, width = proximity.shape
    half_diagonal_m = 0.5 * math.hypot(width * analysis_mpp, height * analysis_mpp)
    local = _local_lines(
        row_lines, lat, lon,
        half_diagonal_m + options.maximum_position_correction_m + 3.0,
    )
    if not local:
        return {"status": "skipped_no_rows", "reason": "No mapped rows intersect the GPS search area."}

    padding = max(1, int(math.ceil(options.maximum_position_correction_m / analysis_mpp)))
    padded = cv2.copyMakeBorder(proximity, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    max_rotation = float(options.maximum_rotation_correction_deg)
    coarse_angles = np.arange(-max_rotation, max_rotation + 0.001, 2.0)
    candidates: list[tuple[float, float, int, int, float, float]] = []

    def evaluate(angle: float) -> None:
        row_mask = _render_rows(local, (height, width), analysis_mpp, float(angle))
        row_pixels = int(np.count_nonzero(row_mask))
        if row_pixels < MIN_ROW_PIXELS:
            return
        response = cv2.matchTemplate(padded, row_mask.astype(np.float32), cv2.TM_CCORR)
        response /= float(row_pixels)
        _minimum, maximum, _minimum_at, maximum_at = cv2.minMaxLoc(response)
        best_x, best_y = maximum_at
        second = _second_peak(response, best_x, best_y, max(2, int(round(1.0 / analysis_mpp))))
        initial = float(response[padding, padding])
        candidates.append((float(maximum), float(angle), best_x, best_y, initial, second))

    for angle in coarse_angles:
        evaluate(float(angle))
    if not candidates:
        return {"status": "skipped_no_rows", "reason": "Too little row geometry is visible in the image footprint."}
    coarse_best = max(candidates, key=lambda item: item[0])
    refinement_start = max(-max_rotation, coarse_best[1] - 1.5)
    refinement_end = min(max_rotation, coarse_best[1] + 1.5)
    for angle in np.arange(refinement_start, refinement_end + 0.001, 0.5):
        if not any(abs(existing[1] - float(angle)) < 1e-6 for existing in candidates):
            evaluate(float(angle))

    best_score, angle, best_x, best_y, initial_score, second_score = max(candidates, key=lambda item: item[0])
    delta_x = float(best_x - padding)
    delta_y = float(best_y - padding)
    angle_rad = math.radians(angle)
    east_correction = -analysis_mpp * (delta_x * math.cos(angle_rad) - delta_y * math.sin(angle_rad))
    north_correction = analysis_mpp * (delta_x * math.sin(angle_rad) + delta_y * math.cos(angle_rad))
    correction_distance = float(math.hypot(east_correction, north_correction))
    improvement = float(best_score - initial_score)
    ambiguity_gap = float(best_score - second_score)
    near_original = correction_distance <= max(0.6, analysis_mpp * 3.0) and abs(angle) <= 1.0
    accepted = (
        best_score >= MIN_ALIGNMENT_SCORE
        and (near_original or improvement >= MIN_CORRECTION_IMPROVEMENT)
        and (near_original or ambiguity_gap >= MIN_AMBIGUITY_GAP)
    )
    confidence = float(np.clip(
        0.55 * best_score
        + 0.25 * min(1.0, max(0.0, improvement) / 0.12)
        + 0.20 * min(1.0, max(0.0, ambiguity_gap) / 0.05),
        0.0, 1.0,
    ))
    record: dict[str, Any] = {
        "status": "aligned" if accepted else "retained_original",
        "score": round(best_score, 6),
        "initial_score": round(initial_score, 6),
        "ambiguity_gap": round(ambiguity_gap, 6),
        "confidence": round(confidence, 6),
        "east_correction_m": round(east_correction, 6),
        "north_correction_m": round(north_correction, 6),
        "position_correction_m": round(correction_distance, 6),
        "rotation_correction_deg": round(angle, 6),
        "prepared_image_size": [int(original_size[0]), int(original_size[1])],
        "nearby_row_count": len(local),
    }
    if not accepted:
        record["reason"] = "The row match was weak or ambiguous; original metadata was retained."
        return record

    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = max(1.0, metres_per_degree_lat * math.cos(math.radians(lat)))
    final_lat = lat + north_correction / metres_per_degree_lat
    final_lon = lon + east_correction / metres_per_degree_lon
    record["original_lat_lon"] = [lat, lon]
    record["final_lat_lon"] = [final_lat, final_lon]
    entry["lat"] = float(final_lat)
    entry["lon"] = float(final_lon)
    entry["row_alignment_rotation_deg"] = float(angle)
    entry["row_alignment"] = record.copy()
    entry["location_source"] = "gps_refined_with_solar_rows"
    return record


def align_rotated_images_to_rows(
    *,
    images_dir: Path,
    camera_meta: dict[str, Any],
    rows_geojson: Path,
    report_path: Path,
    source: dict[str, Any],
    options: RowAlignmentOptions | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    options = options or RowAlignmentOptions()
    row_lines = _load_row_lines(rows_geojson)
    image_paths = sorted(
        path for path in Path(images_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    )
    records: dict[str, Any] = {}
    for index, image_path in enumerate(image_paths, start=1):
        metadata_key, entry = _entry_for_image(camera_meta, image_path.name)
        if entry is None:
            record = {"status": "skipped_metadata", "reason": "No matching camera metadata entry was found."}
        else:
            record = _align_one(image_path, entry, row_lines, options)
            record["camera_metadata_key"] = metadata_key
            entry["row_alignment"] = record.copy()
        records[image_path.name] = record
        if progress:
            progress(index, len(image_paths), image_path.name)

    counts: dict[str, int] = {}
    for record in records.values():
        status = str(record.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    report = {
        "version": 1,
        "source": source,
        "options": {
            "maximum_position_correction_m": options.maximum_position_correction_m,
            "maximum_rotation_correction_deg": options.maximum_rotation_correction_deg,
        },
        "row_line_count": len(row_lines),
        "image_count": len(image_paths),
        "counts": counts,
        "images": records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
