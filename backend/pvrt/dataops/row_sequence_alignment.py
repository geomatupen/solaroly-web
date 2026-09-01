"""Sequence- and overlap-aware constraints for thermal image alignment.

The solar layout is repetitive, so an independently strongest map correlation
can be displaced by one panel or one row.  This module matches overlapping
prepared images to each other, verifies those matches against the GPS prior,
and solves the per-image corrections together.  Images are never resampled or
updated sequentially; all corrections are committed only after the graph solve.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import threading
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


PAIR_ANALYSIS_MAX_DIMENSION = 768
PAIR_MAX_FEATURES = 1024
PAIR_MIN_MATCHES = 8
PAIR_MAX_TEMPORAL_GPS_ERROR_M = 1.5
PAIR_MAX_LATERAL_GPS_ERROR_M = 2.0
LIGHTGLUE_FILTER_THRESHOLD = 0.15


_LIGHTGLUE_RUNTIME: dict[str, Any] | None = None
_LIGHTGLUE_RUNTIME_LOCK = threading.Lock()


@dataclass
class _FrameFeatures:
    points: np.ndarray
    features: dict[str, Any] | None
    width: int
    height: int
    resize_scale: float
    analysis_mpp: float


def _empty_features() -> _FrameFeatures:
    return _FrameFeatures(np.empty((0, 2), np.float32), None, 0, 0, 1.0, 0.0)


def _lightglue_runtime() -> dict[str, Any]:
    """Load the official SIFT+LightGlue models once per backend process."""
    global _LIGHTGLUE_RUNTIME
    if _LIGHTGLUE_RUNTIME is not None:
        return _LIGHTGLUE_RUNTIME
    with _LIGHTGLUE_RUNTIME_LOCK:
        if _LIGHTGLUE_RUNTIME is not None:
            return _LIGHTGLUE_RUNTIME
        try:
            import torch
            from lightglue import LightGlue, SIFT
        except ImportError as exc:
            raise RuntimeError(
                "Image alignment requires the LightGlue dependency. Reinstall the backend requirements."
            ) from exc
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _LIGHTGLUE_RUNTIME = {
            "torch": torch,
            "device": device,
            "extractor": SIFT(max_num_keypoints=PAIR_MAX_FEATURES, backend="opencv").eval(),
            "matcher": LightGlue(
                features="sift",
                depth_confidence=0.9,
                width_confidence=0.95,
                filter_threshold=LIGHTGLUE_FILTER_THRESHOLD,
            ).eval().to(device),
        }
    return _LIGHTGLUE_RUNTIME


def _lat_lon(entry: dict[str, Any]) -> tuple[float, float] | None:
    try:
        latitude = float(entry["lat"])
        longitude = float(entry["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(latitude) or not math.isfinite(longitude):
        return None
    return latitude, longitude


def _distance_and_offset(first: dict[str, Any], second: dict[str, Any]) -> tuple[float, float, float]:
    first_position = _lat_lon(first)
    second_position = _lat_lon(second)
    if first_position is None or second_position is None:
        return math.inf, 0.0, 0.0
    first_latitude, first_longitude = first_position
    second_latitude, second_longitude = second_position
    mean_latitude = (first_latitude + second_latitude) * 0.5
    east = (second_longitude - first_longitude) * 111_320.0 * math.cos(math.radians(mean_latitude))
    north = (second_latitude - first_latitude) * 111_320.0
    return float(math.hypot(east, north)), float(east), float(north)


def _timestamp(entry: dict[str, Any]) -> float:
    try:
        return float(entry.get("timestamp") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _load_features(path: Path, entry: dict[str, Any]) -> _FrameFeatures:
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        return _empty_features()
    if source.ndim == 2:
        gray = source
        valid = np.full(source.shape, 255, dtype=np.uint8)
    else:
        gray = cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2GRAY)
        valid = source[:, :, 3] if source.shape[2] >= 4 else np.full(gray.shape, 255, dtype=np.uint8)
    original_height, original_width = gray.shape
    resize_scale = min(1.0, PAIR_ANALYSIS_MAX_DIMENSION / max(original_height, original_width))
    if resize_scale < 1.0:
        size = (
            max(1, int(round(original_width * resize_scale))),
            max(1, int(round(original_height * resize_scale))),
        )
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
        valid = cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.erode((valid > 32).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gradient_x = cv2.Sobel(enhanced, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(enhanced, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)
    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Keep most descriptor information in the normalized thermal texture. A
    # gradient-dominant descriptor would make every repeated panel seam look
    # distinctive and could verify the same one-row jump we are trying to stop.
    feature_image = cv2.addWeighted(enhanced, 0.75, gradient, 0.25, 0.0)
    runtime = _lightglue_runtime()
    torch = runtime["torch"]
    image_tensor = torch.from_numpy(feature_image).float()[None] / 255.0
    with _LIGHTGLUE_RUNTIME_LOCK, torch.inference_mode():
        extracted = runtime["extractor"].extract(image_tensor, resize=None)
    points = extracted["keypoints"][0].detach().cpu().numpy().astype(np.float32, copy=False)
    if len(points):
        rounded = np.rint(points).astype(np.int32)
        rounded[:, 0] = np.clip(rounded[:, 0], 0, mask.shape[1] - 1)
        rounded[:, 1] = np.clip(rounded[:, 1], 0, mask.shape[0] - 1)
        keep = torch.from_numpy(mask[rounded[:, 1], rounded[:, 0]] > 0)
        feature_count = extracted["keypoints"].shape[1]
        extracted = {
            key: value[:, keep] if hasattr(value, "shape") and len(value.shape) >= 2 and value.shape[1] == feature_count else value
            for key, value in extracted.items()
        }
        points = points[keep.numpy()]
    extracted = {key: value.detach().cpu() for key, value in extracted.items()}
    try:
        metres_per_pixel = float(entry["meters_per_pixel"])
    except (KeyError, TypeError, ValueError):
        metres_per_pixel = 0.0
    return _FrameFeatures(
        points=points,
        features=extracted,
        width=gray.shape[1],
        height=gray.shape[0],
        resize_scale=resize_scale,
        analysis_mpp=metres_per_pixel / max(resize_scale, 1e-9),
    )


def _candidate_pairs(
    names: list[str], entries: dict[str, dict[str, Any]], records: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str]]:
    ordered = sorted(names, key=lambda name: (_timestamp(entries[name]), name))
    order_index = {name: index for index, name in enumerate(ordered)}
    pairs: dict[tuple[str, str], str] = {}

    # Along-track neighbors: consecutive captures normally have the strongest
    # overlap and preserve the flight sequence.
    for index, first in enumerate(ordered):
        first_time = _timestamp(entries[first])
        for second in ordered[index + 1:index + 4]:
            second_time = _timestamp(entries[second])
            if first_time and second_time and second_time - first_time > 8.0:
                break
            pairs[(first, second)] = "temporal"

    # Cross-track/lateral neighbors: add the closest GPS-overlapping frames even
    # when they are far apart in capture order (adjacent flight strips).
    for first in ordered:
        first_record = records.get(first) or {}
        first_size = first_record.get("prepared_image_size") or [0, 0]
        try:
            first_radius = 0.5 * math.hypot(*map(float, first_size)) * float(entries[first]["meters_per_pixel"])
        except (KeyError, TypeError, ValueError):
            first_radius = 10.0
        nearby: list[tuple[float, str]] = []
        for second in ordered:
            if second == first or abs(order_index[second] - order_index[first]) <= 3:
                continue
            distance, _east, _north = _distance_and_offset(entries[first], entries[second])
            second_record = records.get(second) or {}
            second_size = second_record.get("prepared_image_size") or [0, 0]
            try:
                second_radius = 0.5 * math.hypot(*map(float, second_size)) * float(entries[second]["meters_per_pixel"])
            except (KeyError, TypeError, ValueError):
                second_radius = 10.0
            if distance <= min(30.0, 0.85 * (first_radius + second_radius)):
                nearby.append((distance, second))
        for _distance, second in sorted(nearby)[:4]:
            key = tuple(sorted((first, second)))
            pairs.setdefault(key, "lateral")
    return [(first, second, kind) for (first, second), kind in pairs.items()]


def _lightglue_matches(first: _FrameFeatures, second: _FrameFeatures) -> tuple[np.ndarray, np.ndarray]:
    if first.features is None or second.features is None or len(first.points) < 4 or len(second.points) < 4:
        return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    runtime = _lightglue_runtime()
    device = runtime["device"]
    image0 = {key: value.to(device) for key, value in first.features.items()}
    image1 = {key: value.to(device) for key, value in second.features.items()}
    with _LIGHTGLUE_RUNTIME_LOCK, runtime["torch"].inference_mode():
        result = runtime["matcher"]({"image0": image0, "image1": image1})
    pairs = result["matches"][0].detach().cpu().numpy().astype(np.int32, copy=False)
    scores = result["scores"][0].detach().cpu().numpy().astype(np.float32, copy=False)
    return pairs, scores


def _match_pair(
    first_name: str,
    second_name: str,
    kind: str,
    first: _FrameFeatures,
    second: _FrameFeatures,
    first_entry: dict[str, Any],
    second_entry: dict[str, Any],
    maximum_pair_position_error_m: float | None = None,
) -> dict[str, Any] | None:
    if first.analysis_mpp <= 0 or second.analysis_mpp <= 0:
        return None
    matches, match_scores = _lightglue_matches(first, second)
    if len(matches) < PAIR_MIN_MATCHES:
        return None
    _distance, gps_east, gps_north = _distance_and_offset(first_entry, second_entry)
    maximum_relative_error_m = (
        float(maximum_pair_position_error_m)
        if maximum_pair_position_error_m is not None
        else PAIR_MAX_TEMPORAL_GPS_ERROR_M if kind == "temporal" else PAIR_MAX_LATERAL_GPS_ERROR_M
    )
    expected_scale = second.analysis_mpp / first.analysis_mpp
    first_centre = np.array([first.width * 0.5, first.height * 0.5], dtype=np.float32)
    second_centre = np.array([second.width * 0.5, second.height * 0.5], dtype=np.float32)
    expected_second_centre = first_centre + np.array(
        [gps_east / first.analysis_mpp, -gps_north / first.analysis_mpp], dtype=np.float32,
    )
    expected_translation = expected_second_centre - expected_scale * second_centre
    maximum_prior_error_px = maximum_relative_error_m / first.analysis_mpp + 6.0
    filtered_indices: list[int] = []
    for index, (first_index, second_index) in enumerate(matches):
        source = second.points[second_index]
        destination = first.points[first_index]
        predicted = expected_scale * source + expected_translation
        if float(np.linalg.norm(destination - predicted)) <= maximum_prior_error_px:
            filtered_indices.append(index)
    if len(filtered_indices) < PAIR_MIN_MATCHES:
        return None
    filtered_matches = matches[filtered_indices]
    filtered_scores = match_scores[filtered_indices]
    source_points = second.points[filtered_matches[:, 1]]
    destination_points = first.points[filtered_matches[:, 0]]
    transform, inlier_mask = cv2.estimateAffinePartial2D(
        source_points,
        destination_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=3000,
        confidence=0.995,
        refineIters=20,
    )
    if transform is None or inlier_mask is None:
        return None
    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / max(1, len(filtered_matches))
    if inlier_count < PAIR_MIN_MATCHES or inlier_ratio < 0.35:
        return None
    affine_scale = math.hypot(float(transform[0, 0]), float(transform[1, 0]))
    if not 0.92 * expected_scale <= affine_scale <= 1.08 * expected_scale:
        return None
    affine_rotation = math.degrees(math.atan2(float(transform[1, 0]), float(transform[0, 0])))
    if abs(affine_rotation) > 6.0:
        return None
    mapped_centre = transform[:, :2] @ second_centre + transform[:, 2]
    visual_east = float(mapped_centre[0] - first_centre[0]) * first.analysis_mpp
    visual_north = -float(mapped_centre[1] - first_centre[1]) * first.analysis_mpp
    correction_east = visual_east - gps_east
    correction_north = visual_north - gps_north
    correction_distance = math.hypot(correction_east, correction_north)
    if correction_distance > maximum_relative_error_m:
        return None
    inlier_scores = filtered_scores[inlier_mask.ravel().astype(bool)]
    learned_confidence = float(np.mean(inlier_scores)) if len(inlier_scores) else 0.0
    confidence = float(np.clip(
        0.45 * inlier_ratio + 0.30 * min(1.0, inlier_count / 30.0) + 0.25 * learned_confidence,
        0.0,
        1.0,
    ))
    return {
        "first": first_name,
        "second": second_name,
        "kind": kind,
        "matcher": "lightglue_sift",
        "match_count": len(filtered_matches),
        "inlier_count": inlier_count,
        "inlier_ratio": round(inlier_ratio, 6),
        "learned_match_confidence": round(learned_confidence, 6),
        "confidence": round(confidence, 6),
        "east_correction_delta_m": round(correction_east, 6),
        "north_correction_delta_m": round(correction_north, 6),
        # Mapping image 2 into image 1 rotates by the inverse of image 2's
        # residual map rotation relative to image 1.
        "rotation_correction_delta_deg": round(-affine_rotation, 6),
    }


def build_visual_constraints(
    *,
    image_paths: list[Path],
    entries: dict[str, dict[str, Any]],
    records: dict[str, dict[str, Any]],
    progress: Callable[[int, int, str], None] | None = None,
    maximum_pair_position_error_m: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    names = [path.name for path in image_paths if path.name in entries]
    path_index = {path.name: path for path in image_paths}
    features: dict[str, _FrameFeatures] = {}
    for index, name in enumerate(names, start=1):
        features[name] = _load_features(path_index[name], entries[name])
        if progress and (index == 1 or index == len(names) or index % max(1, len(names) // 10) == 0):
            progress(index, len(names), f"Extracting LightGlue features: {name}")
    pairs = _candidate_pairs(names, entries, records)
    constraints: list[dict[str, Any]] = []
    attempted = {"temporal": 0, "lateral": 0}
    accepted = {"temporal": 0, "lateral": 0}
    for index, (first, second, kind) in enumerate(pairs, start=1):
        attempted[kind] += 1
        constraint = _match_pair(
            first, second, kind, features[first], features[second], entries[first], entries[second],
            maximum_pair_position_error_m,
        )
        if constraint is not None:
            constraints.append(constraint)
            accepted[kind] += 1
        if progress and (index == 1 or index == len(pairs) or index % max(1, len(pairs) // 10) == 0):
            progress(index, len(pairs), f"Verifying {kind} image overlaps")
    return constraints, {
        "matcher": "lightglue_sift",
        "device": str(_lightglue_runtime()["device"]),
        "maximum_pair_position_error_m": maximum_pair_position_error_m,
        "candidate_pairs": len(pairs),
        "verified_pairs": len(constraints),
        "attempted_by_kind": attempted,
        "verified_by_kind": accepted,
    }


def _solve_weighted(
    node_index: dict[str, int],
    equations: list[tuple[dict[str, float], float, float]],
    huber_delta: float,
) -> np.ndarray:
    matrix = np.zeros((len(equations), len(node_index)), dtype=np.float64)
    targets = np.zeros(len(equations), dtype=np.float64)
    base_weights = np.zeros(len(equations), dtype=np.float64)
    for row, (coefficients, target, weight) in enumerate(equations):
        for name, coefficient in coefficients.items():
            matrix[row, node_index[name]] = coefficient
        targets[row] = target
        base_weights[row] = weight
    robust_weights = np.ones(len(equations), dtype=np.float64)
    solution = np.zeros(len(node_index), dtype=np.float64)
    for _iteration in range(5):
        weights = np.sqrt(np.maximum(1e-8, base_weights * robust_weights))
        solution = np.linalg.lstsq(matrix * weights[:, None], targets * weights, rcond=None)[0]
        residuals = np.abs(matrix @ solution - targets)
        robust_weights = np.where(residuals <= huber_delta, 1.0, huber_delta / np.maximum(residuals, 1e-9))
    return solution


def apply_visual_pose_graph(
    *,
    constraints: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    entries: dict[str, dict[str, Any]],
    maximum_position_correction_m: float,
    maximum_rotation_correction_deg: float,
) -> dict[str, Any]:
    if not constraints:
        return {"status": "skipped", "reason": "No image pairs passed geometric verification."}
    names = sorted(entries, key=lambda name: (_timestamp(entries[name]), name))
    node_index = {name: index for index, name in enumerate(names)}
    adjacency: dict[str, set[str]] = {name: set() for name in names}
    position_east: list[tuple[dict[str, float], float, float]] = []
    position_north: list[tuple[dict[str, float], float, float]] = []
    rotations: list[tuple[dict[str, float], float, float]] = []
    support: dict[str, dict[str, int]] = {
        name: {"temporal": 0, "lateral": 0, "sequence": 0} for name in names
    }
    visually_matched_nodes: set[str] = set()
    for constraint in constraints:
        first, second = constraint["first"], constraint["second"]
        if first not in node_index or second not in node_index:
            continue
        kind = str(constraint["kind"])
        adjacency[first].add(second)
        adjacency[second].add(first)
        visually_matched_nodes.update((first, second))
        support[first][kind] += 1
        support[second][kind] += 1
        weight = 3.5 + 3.0 * float(constraint["confidence"])
        coefficients = {first: -1.0, second: 1.0}
        position_east.append((coefficients, float(constraint["east_correction_delta_m"]), weight))
        position_north.append((coefficients, float(constraint["north_correction_delta_m"]), weight))
        rotations.append((coefficients, float(constraint["rotation_correction_delta_deg"]), weight * 0.55))

    # GPS remains a weak absolute prior. It prevents an entire connected graph
    # from drifting while relative image-overlap constraints are optimized.
    for name in names:
        position_east.append(({name: 1.0}, 0.0, 0.22))
        position_north.append(({name: 1.0}, 0.0, 0.22))
        rotations.append(({name: 1.0}, 0.0, 0.18))

    # Corrections, unlike camera positions, should vary smoothly along a flight
    # strip. Add a second-difference constraint only across close timestamps and
    # near-straight GPS motion, so turns do not join separate strips.
    smooth_equations = 0
    for first, middle, last in zip(names, names[1:], names[2:]):
        first_gap = _timestamp(entries[middle]) - _timestamp(entries[first])
        second_gap = _timestamp(entries[last]) - _timestamp(entries[middle])
        if first_gap < 0 or second_gap < 0 or first_gap > 5.0 or second_gap > 5.0:
            continue
        _d1, east1, north1 = _distance_and_offset(entries[first], entries[middle])
        _d2, east2, north2 = _distance_and_offset(entries[middle], entries[last])
        if math.hypot(east1, north1) < 0.3 or math.hypot(east2, north2) < 0.3:
            continue
        direction_change = abs((math.degrees(math.atan2(east2, north2)) - math.degrees(math.atan2(east1, north1)) + 180.0) % 360.0 - 180.0)
        if direction_change > 30.0:
            continue
        coefficients = {first: 1.0, middle: -2.0, last: 1.0}
        position_east.append((coefficients, 0.0, 0.65))
        position_north.append((coefficients, 0.0, 0.65))
        rotations.append((coefficients, 0.0, 0.45))
        # Consecutive DJI GPS errors are largely common-mode. Penalizing a
        # sudden change in the correction itself is what prevents a verified
        # frame from stepping sideways by one repeated panel pitch.
        for earlier, later in ((first, middle), (middle, last)):
            velocity_coefficients = {earlier: -1.0, later: 1.0}
            position_east.append((velocity_coefficients, 0.0, 0.75))
            position_north.append((velocity_coefficients, 0.0, 0.75))
            rotations.append((velocity_coefficients, 0.0, 0.55))
        # These are flight-strip continuity edges, not visual matches. They let
        # a weak frame between verified neighbors inherit a smooth correction
        # without snapping independently to another repeated solar row.
        adjacency[first].add(middle)
        adjacency[middle].update((first, last))
        adjacency[last].add(middle)
        support[first]["sequence"] += 1
        support[middle]["sequence"] += 2
        support[last]["sequence"] += 1
        smooth_equations += 1

    east_solution = _solve_weighted(node_index, position_east, huber_delta=0.8)
    north_solution = _solve_weighted(node_index, position_north, huber_delta=0.8)
    rotation_solution = _solve_weighted(node_index, rotations, huber_delta=1.5)

    # Only components connected to verified visual matches are updated.
    # Isolated images retain their original metadata.
    anchored_nodes: set[str] = set()
    queue = list(visually_matched_nodes)
    while queue:
        current = queue.pop()
        if current in anchored_nodes:
            continue
        anchored_nodes.add(current)
        queue.extend(adjacency[current] - anchored_nodes)

    updated = 0
    recovered = 0
    rejected_limits = 0
    for name in names:
        if name not in anchored_nodes or not adjacency[name]:
            continue
        index = node_index[name]
        east = float(east_solution[index])
        north = float(north_solution[index])
        rotation = float(rotation_solution[index])
        if math.hypot(east, north) > maximum_position_correction_m or abs(rotation) > maximum_rotation_correction_deg:
            rejected_limits += 1
            continue
        record = records[name]
        was_aligned = record.get("status") == "aligned"
        record["pre_pose_graph_alignment"] = {
            "status": record.get("status"),
            "alignment_method": record.get("alignment_method"),
            "east_correction_m": record.get("east_correction_m"),
            "north_correction_m": record.get("north_correction_m"),
            "rotation_correction_deg": record.get("rotation_correction_deg"),
        }
        record["east_correction_m"] = round(east, 6)
        record["north_correction_m"] = round(north, 6)
        record["position_correction_m"] = round(math.hypot(east, north), 6)
        record["rotation_correction_deg"] = round(rotation, 6)
        record["status"] = "aligned"
        record["alignment_method"] = "lightglue_pose_graph"
        record["visual_temporal_support"] = support[name]["temporal"]
        record["visual_lateral_support"] = support[name]["lateral"]
        record["flight_sequence_support"] = support[name]["sequence"]
        record.pop("reason", None)
        updated += 1
        if not was_aligned:
            recovered += 1
    return {
        "status": "applied",
        "absolute_anchor": "gps_prior",
        "anchored_image_count": len(anchored_nodes),
        "updated_image_count": updated,
        "recovered_image_count": recovered,
        "rejected_by_correction_limits": rejected_limits,
        "smooth_sequence_constraints": smooth_equations,
    }


def find_camera_entry(camera_meta: dict[str, Any], image_name: str) -> tuple[str | None, dict[str, Any] | None]:
    stem = Path(image_name).stem.casefold()
    for key, value in camera_meta.items():
        if str(key).startswith("__") or not isinstance(value, dict):
            continue
        key_path = Path(str(key))
        if key_path.name.casefold() == image_name.casefold() or key_path.stem.casefold() == stem:
            return str(key), value
    return None, None


def _commit_lightglue_record(entry: dict[str, Any], record: dict[str, Any]) -> None:
    original = record.get("original_lat_lon")
    if record.get("status") != "aligned" or not isinstance(original, list) or len(original) < 2:
        entry.pop("row_alignment_rotation_deg", None)
        entry["row_alignment"] = record.copy()
        return
    latitude, longitude = map(float, original[:2])
    metres_per_degree_lon = max(1.0, 111_320.0 * math.cos(math.radians(latitude)))
    final_latitude = latitude + float(record["north_correction_m"]) / 111_320.0
    final_longitude = longitude + float(record["east_correction_m"]) / metres_per_degree_lon
    record["final_lat_lon"] = [final_latitude, final_longitude]
    entry["lat"] = final_latitude
    entry["lon"] = final_longitude
    entry["row_alignment_rotation_deg"] = float(record["rotation_correction_deg"])
    entry["row_alignment"] = record.copy()
    entry["location_source"] = "gps_refined_with_lightglue_image_matches"


def align_images_with_lightglue(
    *,
    images_dir: Path,
    camera_meta: dict[str, Any],
    report_path: Path,
    maximum_position_correction_m: float,
    maximum_rotation_correction_deg: float,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Refine individual image poses from overlaps without using reference GeoJSON."""
    image_paths = sorted(
        path for path in Path(images_dir).iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
    )
    records: dict[str, dict[str, Any]] = {}
    entries: dict[str, dict[str, Any]] = {}
    for image_path in image_paths:
        metadata_key, entry = find_camera_entry(camera_meta, image_path.name)
        if entry is None:
            records[image_path.name] = {
                "status": "skipped_metadata",
                "reason": "No matching camera metadata entry was found.",
            }
            continue
        position = _lat_lon(entry)
        try:
            metres_per_pixel = float(entry["meters_per_pixel"])
        except (KeyError, TypeError, ValueError):
            metres_per_pixel = 0.0
        if position is None or not math.isfinite(metres_per_pixel) or metres_per_pixel <= 0:
            records[image_path.name] = {
                "status": "skipped_metadata",
                "reason": "GPS or GSD metadata is unavailable.",
                "camera_metadata_key": metadata_key,
            }
            continue
        try:
            with Image.open(image_path) as image:
                prepared_size = [int(image.width), int(image.height)]
        except OSError:
            prepared_size = [0, 0]
        latitude, longitude = position
        entries[image_path.name] = entry
        records[image_path.name] = {
            "status": "retained_original",
            "reason": "No verified LightGlue-connected correction was available; original metadata was retained.",
            "alignment_method": None,
            "camera_metadata_key": metadata_key,
            "original_lat_lon": [latitude, longitude],
            "east_correction_m": 0.0,
            "north_correction_m": 0.0,
            "position_correction_m": 0.0,
            "rotation_correction_deg": 0.0,
            "prepared_image_size": prepared_size,
        }

    constraints, visual_matching = build_visual_constraints(
        image_paths=image_paths,
        entries=entries,
        records=records,
        progress=progress,
        maximum_pair_position_error_m=min(float(maximum_position_correction_m), 12.0),
    )
    pose_graph = apply_visual_pose_graph(
        constraints=constraints,
        records=records,
        entries=entries,
        maximum_position_correction_m=maximum_position_correction_m,
        maximum_rotation_correction_deg=maximum_rotation_correction_deg,
    )
    for image_name, entry in entries.items():
        _commit_lightglue_record(entry, records[image_name])

    counts: dict[str, int] = {}
    for record in records.values():
        status = str(record.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    report = {
        "version": 1,
        "mode": "lightglue",
        "source": {"type": "overlapping_images", "uses_reference_geojson": False},
        "options": {
            "maximum_position_correction_m": maximum_position_correction_m,
            "maximum_rotation_correction_deg": maximum_rotation_correction_deg,
        },
        "visual_matching": visual_matching,
        "visual_pose_graph": pose_graph,
        "image_count": len(image_paths),
        "counts": counts,
        "images": records,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
