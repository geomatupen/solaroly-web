"""Deduplicate overlapping-image anomalies and associate them with panel IDs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from shapely.strtree import STRtree

from .common import class_key, feature, infer_metric_crs, load_polygon_features, project_geometry, score, transformer, write_feature_collection


ProgressCallback = Callable[[int, str], None]


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
            return value
    return f"ANOM-{source_index + 1:06d}"


def deduplicate_anomalies(
    input_path: Path,
    output_path: Path,
    *,
    minimum_iou: float = 0.35,
    maximum_center_distance_m: float = 0.35,
    minimum_smaller_overlap: float = 0.55,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Keep the strongest of same-class predictions representing one anomaly."""
    _notify(callback, 2, "Reading anomaly predictions…")
    payload, records, invalid = load_polygon_features(input_path)
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
            if _intersection_over_union(geometry, candidate) >= minimum_iou or (centres_close and smaller_overlap >= minimum_smaller_overlap):
                removed.add(candidate_index)
                duplicate_indices.append(candidate_index)
        kept.append((source_index, geometry, properties, duplicate_indices))
        if position % 250 == 0:
            _notify(callback, 12 + int(72 * position / total), "Removing duplicate anomaly predictions…")
    output_features = []
    for source_index, geometry, properties, duplicates in kept:
        output_properties = dict(properties)
        output_properties.update({
            "anomaly_id": _anomaly_id(output_properties, source_index),
            "postprocess_stage": "deduplicated_anomalies",
            "source_anomaly_index": source_index,
            "duplicate_count": len(duplicates),
            "duplicate_source_indices": duplicates,
        })
        output_features.append(feature(geometry, output_properties, to_wgs84))
    write_feature_collection(output_path, output_features, source=str(input_path), metric_crs=metric_crs.to_string())
    _notify(callback, 100, "Anomaly deduplication is complete.")
    return {
        "input_features": len(payload["features"]),
        "invalid_input_features": invalid,
        "output_features": len(output_features),
        "duplicates_removed": len(removed),
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path),
    }


def associate_anomalies(
    anomaly_path: Path,
    panel_path: Path,
    output_path: Path,
    panel_output_path: Path | None = None,
    *,
    minimum_overlap: float = 0.20,
    maximum_distance_m: float = 0.50,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Attach panel and row IDs to anomalies without changing their geometry."""
    _notify(callback, 2, "Reading anomaly and identified-panel layers…")
    anomaly_payload, anomalies, invalid_anomalies = load_polygon_features(anomaly_path)
    panel_payload, loaded_panels, invalid_panels = load_polygon_features(panel_path)
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
        anomaly_id = _anomaly_id(output_properties, source_index)
        output_properties.update({
            "anomaly_id": anomaly_id,
            "postprocess_stage": "associated_anomalies",
            "source_anomaly_index": source_index,
            "association_method": method,
            "panel_overlap_fraction": round(best_overlap, 6),
            "panel_distance_m": round(best_distance, 4) if best_distance != float("inf") else None,
            "panel_id": best_panel[1].get("panel_id") if best_panel else None,
            "row_id": best_panel[1].get("row_id") if best_panel else None,
            "review_required": method != "overlap",
        })
        if best_panel:
            panel_id = str(best_panel[1].get("panel_id") or "")
            if panel_id:
                panel_anomaly_ids.setdefault(panel_id, []).append(anomaly_id)
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
    _notify(callback, 100, "Anomaly-to-panel association is complete.")
    return {
        "input_features": len(anomaly_payload["features"]),
        "invalid_anomaly_features": invalid_anomalies,
        "invalid_panel_features": invalid_panels,
        "output_features": len(output_features),
        "assigned": assigned,
        "assigned_by_nearest": nearest_assigned,
        "unassigned": unassigned,
        "panels_with_anomalies": panels_with_anomalies,
        "panel_features_updated": len(panel_payload["features"]) if panel_output_path is not None else 0,
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path),
    }
