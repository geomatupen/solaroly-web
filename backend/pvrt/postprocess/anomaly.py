"""Deduplicate overlapping-image anomalies and associate them with panel IDs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from shapely.geometry import Point, Polygon, mapping, shape
from shapely.strtree import STRtree

from .common import class_key, feature, infer_metric_crs, load_polygon_features, project_geometry, score, transformer, write_feature_collection


ProgressCallback = Callable[[int, str], None]
DEFAULT_DUPLICATE_WEIGHTS = {
    "appearance": 0.45,
    "context": 0.20,
    "shape": 0.10,
    "size": 0.10,
    "orientation": 0.10,
    "proximity": 0.05,
}
DEFAULT_REPRESENTATIVE_WEIGHTS = {
    "image_center": 0.40,
    "spatial_centrality": 0.35,
    "model_confidence": 0.25,
}


def _notify(callback: ProgressCallback | None, progress: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, int(progress))), message)


def _intersection_over_union(first: Any, second: Any) -> float:
    intersection = first.intersection(second).area
    union = first.area + second.area - intersection
    return float(intersection / union) if union > 0 else 0.0


def _anomaly_id(properties: dict[str, Any], source_index: int) -> str:
    for key in ("anomaly_id", "detection_id", "prediction_id"):
        value = str(properties.get(key) or "").strip()
        if value:
            if value.upper().startswith("ANOM-") and value[5:].isdigit():
                return str(int(value[5:]))
            return value
    return str(source_index + 1)


def _image_name(properties: dict[str, Any]) -> str:
    return str(
        properties.get("image")
        or properties.get("tile")
        or properties.get("src")
        or properties.get("source_image")
        or ""
    ).strip()


def _image_catalog(result_dir: Path) -> dict[str, dict[str, Any]]:
    path = result_dir / "images.geojson"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    catalog: dict[str, dict[str, Any]] = {}
    for item in payload.get("features") or []:
        properties = dict(item.get("properties") or {})
        name = _image_name(properties)
        corners = properties.get("corners")
        if not name:
            continue
        centre = None
        geometry = item.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") == "Point" and isinstance(coordinates, list) and len(coordinates) >= 2:
            centre = [float(coordinates[0]), float(coordinates[1])]
        if isinstance(corners, list) and len(corners) == 4:
            centre = centre or [
                sum(float(corner[0]) for corner in corners) / 4.0,
                sum(float(corner[1]) for corner in corners) / 4.0,
            ]
        entry = {"name": name, "corners": corners if isinstance(corners, list) and len(corners) == 4 else None, "centre": centre}
        for key in {name, Path(name).name, Path(name).stem}:
            catalog[key.lower()] = entry
    return catalog


def find_review_image(result_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    candidates = [
        result_dir / "overlays" / f"{stem}.png",
        result_dir / "overlays" / f"{stem}.jpg",
        result_dir / "overlays" / f"{stem}.jpeg",
    ]
    for directory in (result_dir / "rotated_images", result_dir / "images"):
        if directory.is_dir():
            candidates.extend(path for path in directory.glob(f"{stem}.*") if path.is_file())
    return next((path for path in candidates if path.is_file()), None)


def _catalog_entry(catalog: dict[str, dict[str, Any]], image_name: str) -> dict[str, Any] | None:
    for key in (image_name, Path(image_name).name, Path(image_name).stem):
        if key.lower() in catalog:
            return catalog[key.lower()]
    return None


def _anomaly_crop(
    geometry: Any,
    image_path: Path,
    corners: list[Any],
    output_path: Path,
) -> tuple[Any, float, list[float]] | None:
    import cv2
    import numpy as np

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    try:
        source = np.asarray(corners, dtype=np.float64)
        centre = source.mean(axis=0)
        scale = max(float(np.ptp(source[:, 0])), float(np.ptp(source[:, 1])), 1e-12)
        source = ((source - centre) / scale).astype(np.float32)
        destination = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source, destination)
        coordinates: list[tuple[float, float]] = []
        geometries = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in geometries:
            coordinates.extend((float(x), float(y)) for x, y in polygon.exterior.coords)
        points = ((np.asarray(coordinates, dtype=np.float64) - centre) / scale).astype(np.float32)
        pixels = cv2.perspectiveTransform(points.reshape(-1, 1, 2), transform).reshape(-1, 2)
    except (ValueError, TypeError, cv2.error, AttributeError):
        return None
    x1, y1 = np.floor(pixels.min(axis=0)).astype(int)
    x2, y2 = np.ceil(pixels.max(axis=0)).astype(int)
    focus_box = [
        round(max(0.0, min(1.0, float(x1) / max(width, 1))), 6),
        round(max(0.0, min(1.0, float(y1) / max(height, 1))), 6),
        round(max(0.0, min(1.0, float(x2) / max(width, 1))), 6),
        round(max(0.0, min(1.0, float(y2) / max(height, 1))), 6),
    ]
    anomaly_center_x = (x1 + x2) / 2.0
    anomaly_center_y = (y1 + y2) / 2.0
    normalized_distance = (
        ((anomaly_center_x - width / 2.0) / max(width / 2.0, 1.0)) ** 2
        + ((anomaly_center_y - height / 2.0) / max(height / 2.0, 1.0)) ** 2
    ) ** 0.5 / 2 ** 0.5
    image_center_proximity = round(max(0.0, min(1.0, 1.0 - normalized_distance)), 4)
    padding = max(12, int(max(x2 - x1, y2 - y1) * 1.25))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)
    return crop, image_center_proximity, focus_box


def _prediction_crop(
    result_dir: Path,
    image_path: Path,
    image_name: str,
    prediction_index: int,
    output_path: Path,
) -> tuple[Any, float, list[float]] | None:
    import cv2
    import numpy as np

    prediction_path = result_dir / "preds" / f"{Path(image_name).stem}.json"
    if not prediction_path.is_file():
        return None
    try:
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        polygons = prediction.get("polygons") or []
        boxes = prediction.get("boxes") or []
        coordinates = np.asarray(polygons[prediction_index], dtype=np.float32) if prediction_index < len(polygons) else None
        if coordinates is not None and coordinates.size >= 6:
            x1, y1 = np.floor(coordinates.min(axis=0)).astype(int)
            x2, y2 = np.ceil(coordinates.max(axis=0)).astype(int)
        elif prediction_index < len(boxes) and len(boxes[prediction_index]) == 4:
            x1, y1, x2, y2 = (int(round(float(value))) for value in boxes[prediction_index])
        else:
            return None
    except (OSError, ValueError, TypeError, IndexError, json.JSONDecodeError):
        return None
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return None
    height, width = image.shape[:2]
    focus_box = [
        round(max(0.0, min(1.0, float(x1) / max(width, 1))), 6),
        round(max(0.0, min(1.0, float(y1) / max(height, 1))), 6),
        round(max(0.0, min(1.0, float(x2) / max(width, 1))), 6),
        round(max(0.0, min(1.0, float(y2) / max(height, 1))), 6),
    ]
    anomaly_center_x = (x1 + x2) / 2.0
    anomaly_center_y = (y1 + y2) / 2.0
    normalized_distance = (
        ((anomaly_center_x - width / 2.0) / max(width / 2.0, 1.0)) ** 2
        + ((anomaly_center_y - height / 2.0) / max(height / 2.0, 1.0)) ** 2
    ) ** 0.5 / 2 ** 0.5
    image_center_proximity = round(max(0.0, min(1.0, 1.0 - normalized_distance)), 4)
    padding = max(12, int(max(x2 - x1, y2 - y1) * 1.25))
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    crop = image[y1:y2, x1:x2].copy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), crop)
    return crop, image_center_proximity, focus_box


def _visual_similarity(first: Any, second: Any) -> float:
    """Return a conservative 0..1 similarity after suppressing red annotations."""
    import cv2
    import numpy as np

    prepared: list[Any] = []
    for image in (first, second):
        resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
        blue, green, red = cv2.split(resized)
        annotation = ((red > 145) & (red > green * 1.35) & (red > blue * 1.35)).astype(np.uint8) * 255
        annotation = cv2.dilate(annotation, np.ones((3, 3), np.uint8), iterations=1)
        if np.any(annotation):
            resized = cv2.inpaint(resized, annotation, 3, cv2.INPAINT_TELEA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        prepared.append(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray))
    correlation = float(cv2.matchTemplate(prepared[0], prepared[1], cv2.TM_CCOEFF_NORMED)[0, 0])
    correlation_score = max(0.0, min(1.0, (correlation + 1.0) / 2.0))
    hashes = [cv2.resize(item, (16, 16), interpolation=cv2.INTER_AREA) > float(item.mean()) for item in prepared]
    hash_score = 1.0 - float(np.count_nonzero(hashes[0] != hashes[1])) / hashes[0].size
    return round(max(0.0, min(1.0, correlation_score * 0.75 + hash_score * 0.25)), 4)


def _centre_crop(image: Any, fraction: float = 0.48) -> Any:
    height, width = image.shape[:2]
    crop_width = max(8, int(width * fraction))
    crop_height = max(8, int(height * fraction))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image[top:top + crop_height, left:left + crop_width]


def _shape_similarity(first: Any, second: Any) -> float:
    def descriptors(geometry: Any) -> tuple[float, float]:
        rectangle = geometry.minimum_rotated_rectangle
        coordinates = list(rectangle.exterior.coords)
        sides = [Point(coordinates[index]).distance(Point(coordinates[index + 1])) for index in range(4)]
        longest = max(sides) if sides else 0.0
        shortest = min(sides) if sides else 0.0
        aspect = shortest / longest if longest > 0 else 0.0
        compactness = geometry.area / rectangle.area if rectangle.area > 0 else 0.0
        return aspect, compactness

    first_aspect, first_compactness = descriptors(first)
    second_aspect, second_compactness = descriptors(second)
    aspect_score = min(first_aspect, second_aspect) / max(first_aspect, second_aspect) if max(first_aspect, second_aspect) > 0 else 0.0
    compactness_score = min(first_compactness, second_compactness) / max(first_compactness, second_compactness) if max(first_compactness, second_compactness) > 0 else 0.0
    return round((aspect_score + compactness_score) / 2.0, 4)


def _orientation_similarity(first: Any, second: Any) -> float:
    """Compare the longest-axis direction, treating 0° and 180° as equivalent."""
    def orientation(geometry: Any) -> float | None:
        rectangle = geometry.minimum_rotated_rectangle
        coordinates = list(rectangle.exterior.coords)
        edges = [
            (
                Point(coordinates[index]).distance(Point(coordinates[index + 1])),
                coordinates[index],
                coordinates[index + 1],
            )
            for index in range(min(4, max(0, len(coordinates) - 1)))
        ]
        if not edges:
            return None
        length, start, end = max(edges, key=lambda edge: edge[0])
        if length <= 0:
            return None
        return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180.0

    first_orientation = orientation(first)
    second_orientation = orientation(second)
    if first_orientation is None or second_orientation is None:
        return 0.0
    difference = abs(first_orientation - second_orientation) % 180.0
    difference = min(difference, 180.0 - difference)
    return round(max(0.0, 1.0 - difference / 90.0), 4)


def _pair_orientation_similarity(pair: dict[str, Any]) -> float:
    value = pair.get("orientation_similarity")
    if value is not None:
        return float(value)
    try:
        return _orientation_similarity(shape(pair["first_geometry"]), shape(pair["second_geometry"]))
    except (KeyError, TypeError, ValueError):
        return 0.0


def _duplicate_score(pair: dict[str, Any], weights: dict[str, float] | None = None) -> float | None:
    if pair.get("appearance_similarity") is None or pair.get("context_similarity") is None:
        return None
    selected = weights or DEFAULT_DUPLICATE_WEIGHTS
    total_weight = sum(max(0.0, float(value)) for value in selected.values())
    if total_weight <= 0:
        return None
    value = sum(
        (_pair_orientation_similarity(pair) if name == "orientation" else float(pair.get(f"{name}_similarity") or 0.0))
        * max(0.0, float(weight))
        for name, weight in selected.items()
    ) / total_weight
    return round(max(0.0, min(1.0, value)), 4)


def _image_locations(
    projected: list[tuple[int, Any, Any, dict[str, Any]]],
    catalog: dict[str, dict[str, Any]],
    metric_crs: Any,
) -> dict[str, Any]:
    grouped: dict[str, list[Any]] = {}
    for _, metric_geometry, _, properties in projected:
        name = _image_name(properties)
        if name:
            grouped.setdefault(name, []).append(metric_geometry)
    locations: dict[str, Any] = {}
    for name, geometries in grouped.items():
        entry = _catalog_entry(catalog, name) or {}
        corners = entry.get("corners")
        centre = entry.get("centre")
        if corners:
            footprint = project_geometry(Polygon(corners), "EPSG:4326", metric_crs)
            locations[name] = footprint.centroid
        elif centre:
            locations[name] = project_geometry(Point(centre), "EPSG:4326", metric_crs)
        else:
            locations[name] = Point(
                sum(geometry.centroid.x for geometry in geometries) / len(geometries),
                sum(geometry.centroid.y for geometry in geometries) / len(geometries),
            )
    return locations


def _neighbor_images(locations: dict[str, Any], radius_m: float) -> dict[str, set[str]]:
    names = list(locations)
    points = [locations[name] for name in names]
    tree = STRtree(points)
    neighbors: dict[str, set[str]] = {name: set() for name in names}
    for position, (name, point) in enumerate(zip(names, points)):
        for candidate_value in tree.query(point.buffer(radius_m)):
            candidate_position = int(candidate_value)
            if candidate_position == position:
                continue
            if point.distance(points[candidate_position]) <= radius_m:
                neighbors[name].add(names[candidate_position])
    return neighbors


def image_neighbor_statistics(
    input_path: Path,
    result_dir: Path,
    neighbor_image_radius_m: float,
) -> dict[str, Any]:
    """Return exact neighbor counts using image metadata only; no image pixels are opened."""
    _, records, _ = load_polygon_features(input_path)
    if not records:
        return {
            "image_count": 0,
            "average_neighbors": 0.0,
            "minimum_neighbors": 0,
            "maximum_neighbors": 0,
            "isolated_images": 0,
            "neighbor_image_radius_m": neighbor_image_radius_m,
        }
    metric_crs = infer_metric_crs(record[1] for record in records)
    projected = [
        (index, project_geometry(geometry, "EPSG:4326", metric_crs), geometry, properties)
        for index, geometry, properties in records
    ]
    locations = _image_locations(projected, _image_catalog(result_dir), metric_crs)
    neighbors = _neighbor_images(locations, neighbor_image_radius_m)
    counts = [len(items) for items in neighbors.values()]
    return {
        "image_count": len(counts),
        "average_neighbors": round(sum(counts) / len(counts), 2) if counts else 0.0,
        "minimum_neighbors": min(counts) if counts else 0,
        "maximum_neighbors": max(counts) if counts else 0,
        "isolated_images": sum(count == 0 for count in counts),
        "neighbor_image_radius_m": neighbor_image_radius_m,
    }


def analyze_visual_duplicates(
    input_path: Path,
    review_path: Path,
    review_images_dir: Path,
    result_dir: Path,
    *,
    maximum_center_distance_m: float = 5.0,
    neighbor_image_radius_m: float = 25.0,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Measure image similarity for spatially plausible duplicate pairs."""
    _notify(callback, 2, "Reading anomaly predictions and image footprints…")
    payload, records, invalid = load_polygon_features(input_path)
    metric_crs = infer_metric_crs(record[1] for record in records)
    projected = [
        (index, project_geometry(geometry, "EPSG:4326", metric_crs), geometry, properties)
        for index, geometry, properties in records
    ]
    tree = STRtree([item[1] for item in projected])
    catalog = _image_catalog(result_dir)
    locations = _image_locations(projected, catalog, metric_crs)
    neighbors = _neighbor_images(locations, neighbor_image_radius_m)
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    crop_cache: dict[int, tuple[Any, str, float, list[float]] | None] = {}
    representative_components: dict[str, dict[str, float]] = {}
    image_ordinals: dict[int, int] = {}
    image_counts: dict[str, int] = {}
    for source_index, _, _, properties in projected:
        image_name = _image_name(properties)
        ordinal = image_counts.get(image_name, 0)
        image_ordinals[source_index] = ordinal
        image_counts[image_name] = ordinal + 1

    def crop_for(source_index: int, geometry: Any, properties: dict[str, Any]) -> tuple[Any, str, float, list[float]] | None:
        if source_index in crop_cache:
            return crop_cache[source_index]
        image_name = _image_name(properties)
        entry = _catalog_entry(catalog, image_name) if image_name else None
        image_path = find_review_image(result_dir, image_name) if image_name else None
        crop_output = review_images_dir / f"anomaly_{source_index + 1}.jpg"
        relative = crop_output.relative_to(review_path.parent).as_posix()
        prediction_index = int(properties.get("prediction_index", image_ordinals.get(source_index, 0)))
        crop_result = _prediction_crop(
            result_dir,
            image_path,
            image_name,
            prediction_index,
            crop_output,
        ) if image_path else None
        if crop_result is None and entry and entry.get("corners") and image_path:
            crop_result = _anomaly_crop(geometry, image_path, entry["corners"], crop_output)
        confidence = score(properties)
        confidence = confidence / 100.0 if confidence > 1.0 else confidence
        image_center_proximity = crop_result[1] if crop_result is not None else 0.5
        representative_components[str(source_index)] = {
            "model_confidence": round(max(0.0, min(1.0, confidence)), 4),
            "image_center_proximity": image_center_proximity,
        }
        crop_cache[source_index] = (
            crop_result[0], relative, image_center_proximity, crop_result[2]
        ) if crop_result is not None else None
        return crop_cache[source_index]

    total = max(1, len(projected))
    for position, (source_index, metric_geometry, wgs_geometry, properties) in enumerate(projected):
        source_class = class_key(properties)
        source_image = _image_name(properties)
        candidates: list[tuple[float, int, Any, Any, dict[str, Any], float, float]] = []
        for candidate_position_value in tree.query(metric_geometry.buffer(maximum_center_distance_m)):
            candidate_index, candidate_metric, candidate_wgs, candidate_properties = projected[int(candidate_position_value)]
            key = tuple(sorted((source_index, candidate_index)))
            candidate_image = _image_name(candidate_properties)
            image_neighbors = (
                candidate_image in neighbors.get(source_image, set())
                and source_image in neighbors.get(candidate_image, set())
            )
            if (
                source_index == candidate_index
                or key in seen
                or not source_image
                or not candidate_image
                or source_image == candidate_image
                or not image_neighbors
                or class_key(candidate_properties) != source_class
            ):
                continue
            intersection = metric_geometry.intersection(candidate_metric).area
            smaller_area = min(metric_geometry.area, candidate_metric.area)
            smaller_overlap = intersection / smaller_area if smaller_area > 0 else 0.0
            iou = _intersection_over_union(metric_geometry, candidate_metric)
            distance = metric_geometry.centroid.distance(candidate_metric.centroid)
            if distance > maximum_center_distance_m:
                continue
            candidates.append((distance, candidate_index, candidate_metric, candidate_wgs, candidate_properties, iou, smaller_overlap))
        for distance, candidate_index, candidate_metric, candidate_wgs, candidate_properties, iou, smaller_overlap in sorted(candidates):
            seen.add(tuple(sorted((source_index, candidate_index))))
            first_crop = crop_for(source_index, wgs_geometry, properties)
            second_crop = crop_for(candidate_index, candidate_wgs, candidate_properties)
            appearance = _visual_similarity(_centre_crop(first_crop[0]), _centre_crop(second_crop[0])) if first_crop and second_crop else None
            context = _visual_similarity(first_crop[0], second_crop[0]) if first_crop and second_crop else None
            size = min(metric_geometry.area, candidate_metric.area) / max(metric_geometry.area, candidate_metric.area) if max(metric_geometry.area, candidate_metric.area) > 0 else 0.0
            pair = {
                "first_index": source_index,
                "second_index": candidate_index,
                "first_anomaly_id": _anomaly_id(properties, source_index),
                "second_anomaly_id": _anomaly_id(candidate_properties, candidate_index),
                "first_geometry": mapping(wgs_geometry),
                "second_geometry": mapping(candidate_wgs),
                "first_image": _image_name(properties),
                "second_image": _image_name(candidate_properties),
                "first_crop_path": first_crop[1] if first_crop else None,
                "second_crop_path": second_crop[1] if second_crop else None,
                "first_focus_box": first_crop[3] if first_crop else None,
                "second_focus_box": second_crop[3] if second_crop else None,
                "iou": round(iou, 4),
                "smaller_overlap": round(smaller_overlap, 4),
                "center_distance_m": round(distance, 4),
                "appearance_similarity": appearance,
                "context_similarity": context,
                "shape_similarity": _shape_similarity(metric_geometry, candidate_metric),
                "size_similarity": round(size, 4),
                "orientation_similarity": _orientation_similarity(metric_geometry, candidate_metric),
                "proximity_similarity": round(max(0.0, 1.0 - distance / maximum_center_distance_m), 4),
                "visual_status": "compared" if appearance is not None else "unavailable",
            }
            pair["visual_similarity"] = appearance
            pair["duplicate_score"] = _duplicate_score(pair)
            pairs.append(pair)
        if position % 100 == 0:
            _notify(callback, 10 + int(85 * position / total), "Comparing candidate anomalies in adjacent images…")
    review = {
        "input_path": str(input_path),
        "metric_crs": metric_crs.to_string(),
        "pairs": pairs,
        "representative_components": representative_components,
    }
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review, indent=2), encoding="utf-8")
    compared = sum(pair["visual_similarity"] is not None for pair in pairs)
    _notify(callback, 100, "Visual duplicate analysis is ready for review.")
    return {
        "input_features": len(payload["features"]),
        "invalid_input_features": invalid,
        "spatial_candidate_pairs": len(pairs),
        "visually_compared_pairs": compared,
        "missing_image_pairs": len(pairs) - compared,
        "suggested_duplicates_at_80_percent": sum(
            (pair.get("duplicate_score") or 0) >= 0.80
            for pair in pairs
        ),
        "neighbor_image_radius_m": neighbor_image_radius_m,
        "maximum_location_shift_m": maximum_center_distance_m,
        "metric_crs": metric_crs.to_string(),
        "review_path": str(review_path),
    }


def apply_visual_deduplication(
    input_path: Path,
    review_path: Path,
    output_path: Path,
    *,
    duplicate_score_threshold: float = 0.80,
    weights: dict[str, float] | None = None,
    representative_weights: dict[str, float] | None = None,
    manual_decisions: list[dict[str, int]] | None = None,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Remove only spatial candidates whose image similarity clears the chosen threshold."""
    _notify(
        callback,
        5,
        "Applying accepted manual duplicate selections…"
        if manual_decisions is not None
        else "Applying the selected visual similarity threshold…",
    )
    payload, records, invalid = load_polygon_features(input_path)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    anomaly_ids_by_index = {
        source_index: _anomaly_id(properties, source_index)
        for source_index, _, properties in records
    }
    qualifying_pairs = {}
    available_pairs = {}
    for pair in review.get("pairs") or []:
        composite = _duplicate_score(pair, weights)
        pair["duplicate_score"] = composite
        edge = tuple(sorted((int(pair["first_index"]), int(pair["second_index"]))))
        available_pairs[edge] = pair
        if (
            manual_decisions is None
            and
            composite is not None
            and composite >= duplicate_score_threshold
        ):
            qualifying_pairs[edge] = pair
    if manual_decisions is not None:
        for decision in manual_decisions:
            edge = tuple(sorted((int(decision["first_index"]), int(decision["second_index"]))))
            if edge not in available_pairs:
                pair_label = "/".join(anomaly_ids_by_index.get(index, str(index + 1)) for index in edge)
                raise ValueError(f"Manual duplicate pair with anomaly IDs {pair_label} is not part of this review.")
            keep_index = int(decision["keep_index"])
            if keep_index not in edge:
                raise ValueError("Manual keep selection must belong to its duplicate pair.")
            qualifying_pairs[edge] = available_pairs[edge]
    selected_representative_weights = representative_weights or DEFAULT_REPRESENTATIVE_WEIGHTS
    representative_weight_total = sum(max(0.0, float(value)) for value in selected_representative_weights.values())
    if representative_weight_total <= 0:
        raise ValueError("Representative-selection weights must have a positive total.")
    metric_crs = infer_metric_crs(record[1] for record in records)
    metric_geometries = {
        source_index: project_geometry(geometry, "EPSG:4326", metric_crs)
        for source_index, geometry, _ in records
    }
    stored_components = review.get("representative_components") or {}
    components_by_index: dict[int, dict[str, float]] = {}
    for source_index, _, properties in records:
        stored = stored_components.get(str(source_index)) or {}
        confidence = score(properties)
        confidence = confidence / 100.0 if confidence > 1.0 else confidence
        components_by_index[source_index] = {
            "image_center": float(stored.get("image_center_proximity", 0.5)),
            "model_confidence": max(0.0, min(1.0, float(stored.get("model_confidence", confidence)))),
            "spatial_centrality": 0.0,
        }

    removed: set[int] = set()
    duplicate_groups: dict[int, list[int]] = {}
    duplicate_details: dict[int, list[dict[str, Any]]] = {}
    representative_scores: dict[int, float] = {}
    remaining_edges = set(qualifying_pairs)
    if manual_decisions is not None:
        kept_indices = {int(decision["keep_index"]) for decision in manual_decisions}
        removed_indices = {
            next(source_index for source_index in (int(decision["first_index"]), int(decision["second_index"])) if source_index != int(decision["keep_index"]))
            for decision in manual_decisions
        }
        conflicts = kept_indices & removed_indices
        if conflicts:
            labels = ", ".join(
                anomaly_ids_by_index.get(source_index, str(source_index + 1))
                for source_index in sorted(conflicts)
            )
            raise ValueError(
                f"Manual selections conflict for anomaly ID {labels}. "
                "Choose one consistent representative per linked group."
            )
        removed.update(removed_indices)
        for decision in manual_decisions:
            keep_index = int(decision["keep_index"])
            edge = tuple(sorted((int(decision["first_index"]), int(decision["second_index"]))))
            remove_index = next(source_index for source_index in edge if source_index != keep_index)
            duplicate_groups.setdefault(keep_index, []).append(remove_index)
            duplicate_details.setdefault(keep_index, []).append(qualifying_pairs[edge])
            representative_scores[keep_index] = 1.0
        remaining_edges.clear()
    while remaining_edges:
        seed = next(iter(remaining_edges))[0]
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            connected = {
                second if first == current else first
                for first, second in remaining_edges
                if first == current or second == current
            }
            new_items = connected - component
            component.update(new_items)
            frontier.extend(new_items)
        maximum_group_distance = max(
            (
                metric_geometries[first].centroid.distance(metric_geometries[second].centroid)
                for first in component
                for second in component
                if first < second
            ),
            default=0.0,
        )
        for source_index in component:
            other_indices = component - {source_index}
            mean_distance = (
                sum(
                    metric_geometries[source_index].centroid.distance(metric_geometries[other].centroid)
                    for other in other_indices
                ) / len(other_indices)
                if other_indices else 0.0
            )
            components_by_index[source_index]["spatial_centrality"] = (
                max(0.0, 1.0 - mean_distance / maximum_group_distance)
                if maximum_group_distance > 0 else 1.0
            )
            representative_scores[source_index] = sum(
                components_by_index[source_index].get(name, 0.0) * max(0.0, float(weight))
                for name, weight in selected_representative_weights.items()
            ) / representative_weight_total
        representative = max(
            component,
            key=lambda source_index: (
                representative_scores[source_index],
                components_by_index[source_index]["model_confidence"],
                -source_index,
            ),
        )
        duplicates = {
            second if first == representative else first
            for first, second in remaining_edges
            if first == representative or second == representative
        }
        removed.update(duplicates)
        duplicate_groups[representative] = sorted(duplicates)
        duplicate_details[representative] = [
            qualifying_pairs[tuple(sorted((representative, duplicate)))]
            for duplicate in sorted(duplicates)
        ]
        discarded = duplicates | {representative}
        remaining_edges = {
            edge for edge in remaining_edges
            if not any(source_index in discarded for source_index in edge)
        }
    output_features = []
    for source_index, geometry, properties in records:
        if source_index in removed:
            continue
        duplicates = duplicate_groups.get(source_index, [])
        details = duplicate_details.get(source_index, [])
        source_images = [_image_name(properties)]
        source_images.extend(
            pair.get("second_image") if int(pair["first_index"]) == source_index else pair.get("first_image")
            for pair in details
        )
        output_properties = dict(properties)
        output_properties.pop("source_anomaly_index", None)
        output_properties.pop("duplicate_source_indices", None)
        output_properties.update({
            "anomaly_id": _anomaly_id(output_properties, source_index),
            "postprocess_stage": "deduplicated_anomalies",
            "duplicate_count": len(duplicates),
            "duplicate_anomaly_ids": [
                anomaly_ids_by_index.get(index, str(index + 1))
                for index in duplicates
            ],
            "duplicate_scores": [pair.get("duplicate_score") for pair in details],
            "duplicate_appearance_similarities": [pair.get("appearance_similarity") for pair in details],
            "source_images": [name for name in source_images if name],
            "duplicate_score_threshold": duplicate_score_threshold,
            "representative_score": round(representative_scores.get(source_index, 1.0), 4),
            "representative_components": {
                name: round(value, 4)
                for name, value in components_by_index.get(source_index, {}).items()
            },
            "representative_weights": selected_representative_weights,
            "deduplication_method": "manual" if manual_decisions is not None else "threshold",
        })
        output_features.append({
            "type": "Feature",
            "geometry": mapping(geometry),
            "properties": output_properties,
        })
    write_feature_collection(output_path, output_features, source=str(input_path), duplicate_score_threshold=duplicate_score_threshold)
    _notify(callback, 100, "Visual anomaly deduplication is complete.")
    return {
        "input_features": len(payload["features"]),
        "invalid_input_features": invalid,
        "output_features": len(output_features),
        "duplicates_removed": len(removed),
        "duplicate_score_threshold": duplicate_score_threshold,
        "weights": weights or DEFAULT_DUPLICATE_WEIGHTS,
        "representative_weights": selected_representative_weights,
        "deduplication_method": "manual" if manual_decisions is not None else "threshold",
        "output_path": str(output_path),
    }


def deduplicate_anomalies(
    input_path: Path,
    output_path: Path,
    *,
    minimum_iou: float = 0.35,
    maximum_center_distance_m: float = 0.35,
    minimum_smaller_overlap: float = 0.55,
    overlap_only: bool = False,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Keep the strongest of same-class predictions representing one anomaly."""
    _notify(callback, 2, "Reading anomaly predictions…")
    payload, records, invalid = load_polygon_features(input_path)
    anomaly_ids_by_index = {
        source_index: _anomaly_id(properties, source_index)
        for source_index, _, properties in records
    }
    metric_crs = infer_metric_crs(record[1] for record in records)
    to_wgs84 = transformer(metric_crs, "EPSG:4326")
    projected = [
        (index, project_geometry(geometry, "EPSG:4326", metric_crs), properties)
        for index, geometry, properties in records
    ]
    ordered = sorted(projected, key=lambda item: (-score(item[2]), item[0]))
    rank = {source_index: position for position, (source_index, _, _) in enumerate(ordered)}
    tree = STRtree([item[1] for item in projected])
    removed: set[int] = set()
    kept: list[tuple[int, Any, dict[str, Any], list[int]]] = []
    total = max(1, len(ordered))
    _notify(callback, 12, "Comparing overlapping predictions from source images…")
    for position, (source_index, geometry, properties) in enumerate(ordered):
        if source_index in removed:
            continue
        duplicate_indices: list[int] = []
        source_class = class_key(properties)
        nearby = tree.query(geometry.buffer(maximum_center_distance_m))
        for candidate_position_value in nearby:
            candidate_source_position = int(candidate_position_value)
            candidate_index, candidate, candidate_properties = projected[candidate_source_position]
            if rank[candidate_index] <= position:
                continue
            if candidate_index in removed or class_key(candidate_properties) != source_class:
                continue
            if geometry.distance(candidate) > maximum_center_distance_m and not geometry.intersects(candidate):
                continue
            intersection = geometry.intersection(candidate).area
            smaller_overlap = intersection / min(geometry.area, candidate.area) if min(geometry.area, candidate.area) > 0 else 0.0
            centres_close = geometry.centroid.distance(candidate.centroid) <= maximum_center_distance_m
            if (
                smaller_overlap >= minimum_smaller_overlap
                if overlap_only
                else _intersection_over_union(geometry, candidate) >= minimum_iou
                or (centres_close and smaller_overlap >= minimum_smaller_overlap)
            ):
                removed.add(candidate_index)
                duplicate_indices.append(candidate_index)
        kept.append((source_index, geometry, properties, duplicate_indices))
        if position % 250 == 0:
            _notify(callback, 12 + int(72 * position / total), "Removing duplicate anomaly predictions…")
    output_features = []
    for source_index, geometry, properties, duplicates in kept:
        output_properties = dict(properties)
        output_properties.pop("source_anomaly_index", None)
        output_properties.pop("duplicate_source_indices", None)
        output_properties.update({
            "anomaly_id": _anomaly_id(output_properties, source_index),
            "postprocess_stage": "deduplicated_anomalies",
            "duplicate_count": len(duplicates),
            "duplicate_anomaly_ids": [
                anomaly_ids_by_index.get(index, str(index + 1))
                for index in duplicates
            ],
        })
        output_features.append(feature(geometry, output_properties, to_wgs84))
    write_feature_collection(output_path, output_features, source=str(input_path), metric_crs=metric_crs.to_string())
    _notify(callback, 100, "Anomaly deduplication is complete.")
    return {
        "input_features": len(payload["features"]),
        "invalid_input_features": invalid,
        "output_features": len(output_features),
        "duplicates_removed": len(removed),
        "minimum_overlap": minimum_smaller_overlap if overlap_only else None,
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path),
    }


def associate_anomalies(
    anomaly_path: Path,
    panel_path: Path,
    output_path: Path,
    panel_output_path: Path | None = None,
    row_path: Path | None = None,
    row_output_path: Path | None = None,
    *,
    minimum_overlap: float = 0.20,
    maximum_distance_m: float = 0.50,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Attach panel and row IDs to anomalies without changing their geometry."""
    _notify(callback, 2, "Reading anomaly and identified-panel layers…")
    anomaly_payload, anomalies, invalid_anomalies = load_polygon_features(anomaly_path)
    panel_payload, loaded_panels, invalid_panels = load_polygon_features(panel_path)
    row_payload = None
    invalid_rows = 0
    if row_path is not None:
        row_payload, _, invalid_rows = load_polygon_features(row_path)
    panels = [record for record in loaded_panels if record[2].get("panel_id")]
    if not panels:
        raise ValueError("The selected hierarchy layer does not contain identified panel features.")
    metric_crs = infer_metric_crs([item[1] for item in anomalies] + [item[1] for item in panels])
    to_wgs84 = transformer(metric_crs, "EPSG:4326")
    projected_panels = [
        (project_geometry(geometry, "EPSG:4326", metric_crs), properties)
        for _, geometry, properties in panels
    ]
    panel_tree = STRtree([item[0] for item in projected_panels])
    output_features = []
    assigned = 0
    nearest_assigned = 0
    unassigned = 0
    panel_anomaly_ids: dict[str, list[str]] = {}
    row_anomaly_ids: dict[str, list[str]] = {}
    row_panel_ids: dict[str, list[str]] = {}
    total = max(1, len(anomalies))
    for position, (source_index, geometry, properties) in enumerate(anomalies):
        projected = project_geometry(geometry, "EPSG:4326", metric_crs)
        best_overlap = 0.0
        best_distance = float("inf")
        best_panel: tuple[Any, dict[str, Any]] | None = None
        method = "unassigned"
        candidate_indices = panel_tree.query(projected.buffer(maximum_distance_m))
        for panel_index_value in candidate_indices:
            panel_geometry, panel_properties = projected_panels[int(panel_index_value)]
            intersection = projected.intersection(panel_geometry).area
            overlap = intersection / projected.area if projected.area > 0 else 0.0
            distance = projected.distance(panel_geometry)
            if overlap > best_overlap or (overlap == best_overlap and distance < best_distance):
                best_overlap, best_distance = overlap, distance
                best_panel = (panel_geometry, panel_properties)
        if best_panel and best_overlap >= minimum_overlap:
            method = "overlap"
            assigned += 1
        elif best_panel and best_distance <= maximum_distance_m:
            method = "nearest"
            assigned += 1
            nearest_assigned += 1
        else:
            best_panel = None
            unassigned += 1
        output_properties = dict(properties)
        output_properties.pop("source_anomaly_index", None)
        output_properties.pop("duplicate_source_indices", None)
        anomaly_id = _anomaly_id(output_properties, source_index)
        output_properties.update({
            "anomaly_id": anomaly_id,
            "postprocess_stage": "associated_anomalies",
            "association_method": method,
            "panel_overlap_fraction": round(best_overlap, 6),
            "panel_distance_m": round(best_distance, 4) if best_distance != float("inf") else None,
            "panel_id": best_panel[1].get("panel_id") if best_panel else None,
            "row_id": best_panel[1].get("row_id") if best_panel else None,
            "review_required": method != "overlap",
        })
        if best_panel:
            panel_id = str(best_panel[1].get("panel_id") or "")
            row_id = str(best_panel[1].get("row_id") or "")
            if panel_id:
                panel_anomaly_ids.setdefault(panel_id, []).append(anomaly_id)
            if row_id:
                row_anomaly_ids.setdefault(row_id, []).append(anomaly_id)
                if panel_id and panel_id not in row_panel_ids.setdefault(row_id, []):
                    row_panel_ids[row_id].append(panel_id)
        output_features.append(feature(projected, output_properties, to_wgs84))
        if position % 250 == 0:
            _notify(callback, 12 + int(78 * position / total), "Assigning anomalies to panels and rows…")
    write_feature_collection(
        output_path,
        output_features,
        anomaly_source=str(anomaly_path),
        panel_source=str(panel_path),
        metric_crs=metric_crs.to_string(),
    )
    panels_with_anomalies = len(panel_anomaly_ids)
    if panel_output_path is not None:
        updated_panels = []
        for panel_feature in panel_payload["features"]:
            updated = dict(panel_feature) if isinstance(panel_feature, dict) else panel_feature
            if isinstance(updated, dict):
                properties = dict(updated.get("properties") or {})
                panel_id = str(properties.get("panel_id") or "")
                anomaly_ids = panel_anomaly_ids.get(panel_id, [])
                properties["anomaly_count"] = len(anomaly_ids)
                properties["anomaly_ids"] = anomaly_ids
                updated["properties"] = properties
            updated_panels.append(updated)
        panel_metadata = {
            key: value for key, value in panel_payload.items()
            if key not in {"type", "features"}
        }
        write_feature_collection(panel_output_path, updated_panels, **panel_metadata)
    rows_with_anomalies = len(row_anomaly_ids)
    if row_payload is not None and row_output_path is not None:
        updated_rows = []
        for row_feature in row_payload["features"]:
            updated = dict(row_feature) if isinstance(row_feature, dict) else row_feature
            if isinstance(updated, dict):
                properties = dict(updated.get("properties") or {})
                row_id = str(properties.get("row_id") or "")
                anomaly_ids = row_anomaly_ids.get(row_id, [])
                properties["anomaly_count"] = len(anomaly_ids)
                properties["anomaly_ids"] = anomaly_ids
                properties["anomaly_panel_ids"] = row_panel_ids.get(row_id, [])
                updated["properties"] = properties
            updated_rows.append(updated)
        row_metadata = {
            key: value for key, value in row_payload.items()
            if key not in {"type", "features"}
        }
        write_feature_collection(row_output_path, updated_rows, **row_metadata)
    _notify(callback, 100, "Anomaly-to-panel association is complete.")
    return {
        "input_features": len(anomaly_payload["features"]),
        "invalid_anomaly_features": invalid_anomalies,
        "invalid_panel_features": invalid_panels,
        "invalid_row_features": invalid_rows,
        "output_features": len(output_features),
        "assigned": assigned,
        "assigned_by_nearest": nearest_assigned,
        "unassigned": unassigned,
        "panels_with_anomalies": panels_with_anomalies,
        "rows_with_anomalies": rows_with_anomalies,
        "panel_features_updated": len(panel_payload["features"]) if panel_output_path is not None else 0,
        "row_features_updated": len(row_payload["features"]) if row_payload is not None and row_output_path is not None else 0,
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path),
    }
