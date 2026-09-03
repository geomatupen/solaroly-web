"""Create deterministic solar-panel and parent-row hierarchy layers."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.strtree import STRtree

from .common import class_key, feature, infer_metric_crs, load_polygon_features, project_geometry, transformer, write_feature_collection


ProgressCallback = Callable[[int, str], None]


@dataclass
class Panel:
    source_index: int
    geometry: Any
    properties: dict[str, Any]
    centre_x: float
    centre_y: float
    short_m: float
    long_m: float
    angle_deg: float
    class_key: str


class _Groups:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        first_root, second_root = self.find(first), self.find(second)
        if first_root != second_root:
            self.parent[second_root] = first_root


def _notify(callback: ProgressCallback | None, progress: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, int(progress))), message)


def _rectangle_measurements(geometry: Any) -> tuple[float, float, float]:
    rectangle = geometry.minimum_rotated_rectangle
    coordinates = list(rectangle.exterior.coords)
    edges = []
    for first, second in zip(coordinates, coordinates[1:]):
        dx, dy = second[0] - first[0], second[1] - first[1]
        edges.append((math.hypot(dx, dy), math.degrees(math.atan2(dy, dx)) % 180.0))
    length, angle = max(edges, key=lambda item: item[0])
    short = min(item[0] for item in edges)
    return short, length, angle


def _angle_difference(first: float, second: float) -> float:
    difference = abs(first - second) % 180.0
    return min(difference, 180.0 - difference)


def _average_orientation(panels: list[Panel]) -> float:
    sine = sum(math.sin(math.radians(panel.angle_deg * 2.0)) for panel in panels)
    cosine = sum(math.cos(math.radians(panel.angle_deg * 2.0)) for panel in panels)
    return (math.degrees(math.atan2(sine, cosine)) / 2.0) % 180.0


def _letter_label(index: int) -> str:
    """Return spreadsheet-style labels: A..Z, AA..AZ, BA..."""
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _projection_interval(geometry: Any, axis_x: float, axis_y: float) -> tuple[float, float]:
    rectangle = geometry.minimum_rotated_rectangle
    values = [x * axis_x + y * axis_y for x, y in rectangle.exterior.coords]
    return min(values), max(values)


def _map_reading_order(
    arrays: list[list[int]],
    inner_row_geometries: list[Any],
) -> list[list[int]]:
    """Order arrays like text: horizontal bands top-to-bottom, then left-to-right."""
    records = []
    short_dimensions = []
    for indices in arrays:
        geometry = unary_union([inner_row_geometries[index] for index in indices]).convex_hull.minimum_rotated_rectangle
        centre = geometry.centroid
        short, _, _ = _rectangle_measurements(geometry)
        short_dimensions.append(short)
        records.append({"indices": indices, "x": centre.x, "y": centre.y})
    band_tolerance = max(0.05, statistics.median(short_dimensions) * 0.75)
    remaining = sorted(records, key=lambda record: (-record["y"], record["x"]))
    ordered: list[list[int]] = []
    while remaining:
        anchor = remaining[0]
        band = [record for record in remaining if abs(record["y"] - anchor["y"]) <= band_tolerance]
        band_ids = {id(record) for record in band}
        remaining = [record for record in remaining if id(record) not in band_ids]
        band.sort(key=lambda record: (record["x"], -record["y"]))
        ordered.extend(record["indices"] for record in band)
    return ordered


def _group_inner_rows(
    panels: list[Panel],
    max_orientation_difference_deg: float,
    max_lateral_distance_factor: float,
    max_along_gap_factor: float,
    callback: ProgressCallback | None = None,
) -> tuple[list[list[Panel]], float, float]:
    median_short = statistics.median(panel.short_m for panel in panels)
    median_long = statistics.median(panel.long_m for panel in panels)
    groups = _Groups(len(panels))
    maximum_search = median_short * (max_along_gap_factor + 1.0)
    centre_points = [Point(panel.centre_x, panel.centre_y) for panel in panels]
    centre_tree = STRtree(centre_points)
    for first_index, first in enumerate(panels):
        angle = math.radians((first.angle_deg + 90.0) % 180.0)
        along_x, along_y = math.cos(angle), math.sin(angle)
        nearby_indices = centre_tree.query(centre_points[first_index].buffer(maximum_search))
        for second_index_value in nearby_indices:
            second_index = int(second_index_value)
            if second_index <= first_index:
                continue
            second = panels[second_index]
            if second.class_key != first.class_key:
                continue
            dx, dy = second.centre_x - first.centre_x, second.centre_y - first.centre_y
            if math.hypot(dx, dy) > maximum_search:
                continue
            if _angle_difference(first.angle_deg, second.angle_deg) > max_orientation_difference_deg:
                continue
            along = abs(dx * along_x + dy * along_y)
            lateral = abs(-dx * along_y + dy * along_x)
            lateral_limit = max(median_short, (first.short_m + second.short_m) / 2.0) * max_lateral_distance_factor
            along_limit = ((first.short_m + second.short_m) / 2.0) * max_along_gap_factor
            if lateral <= lateral_limit and along <= along_limit:
                groups.union(first_index, second_index)
        if callback and first_index % 250 == 0:
            _notify(callback, 18 + int(35 * first_index / max(1, len(panels))), "Grouping adjacent panels into rows…")
    components: dict[int, list[Panel]] = {}
    for index, panel in enumerate(panels):
        components.setdefault(groups.find(index), []).append(panel)
    return list(components.values()), median_short, median_long


def build_panel_hierarchy(
    input_path: Path,
    output_path: Path | None,
    *,
    rows_output_path: Path | None = None,
    panels_output_path: Path | None = None,
    max_orientation_difference_deg: float = 15.0,
    max_lateral_distance_factor: float = 1.5,
    max_along_gap_factor: float = 1.5,
    max_inner_row_gap_factor: float = 0.8,
    min_row_overlap_percent: float = 20.0,
    assign_ids: bool = True,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Group panel rectangles into rows and assign stable row/panel identifiers."""
    _notify(callback, 2, "Reading regularized panel polygons…")
    payload, records, invalid = load_polygon_features(input_path)
    metric_crs = infer_metric_crs(record[1] for record in records)
    to_metric = transformer("EPSG:4326", metric_crs)
    to_wgs84 = transformer(metric_crs, "EPSG:4326")
    panels: list[Panel] = []
    for source_index, geometry, properties in records:
        projected = project_geometry(geometry, "EPSG:4326", metric_crs)
        short, long, angle = _rectangle_measurements(projected)
        centre = projected.centroid
        panels.append(Panel(source_index, projected, properties, centre.x, centre.y, short, long, angle, class_key(properties)))
    _notify(callback, 18, "Finding neighbouring panels with compatible orientation…")
    inner_rows, median_short, median_long = _group_inner_rows(
        panels,
        max_orientation_difference_deg,
        max_lateral_distance_factor,
        max_along_gap_factor,
        callback,
    )
    inner_row_geometries = [
        unary_union([panel.geometry for panel in component]).convex_hull.minimum_rotated_rectangle
        for component in inner_rows
    ]
    array_groups = _Groups(len(inner_rows))
    _notify(callback, 53, "Grouping adjacent inner rows into solar arrays…")
    for first_index, first_component in enumerate(inner_rows):
        first_geometry = inner_row_geometries[first_index]
        first_orientation = _average_orientation(first_component)
        row_angle = math.radians((first_orientation + 90.0) % 180.0)
        row_axis_x, row_axis_y = math.cos(row_angle), math.sin(row_angle)
        first_interval = _projection_interval(first_geometry, row_axis_x, row_axis_y)
        for second_index in range(first_index + 1, len(inner_rows)):
            second_component = inner_rows[second_index]
            if first_component[0].class_key != second_component[0].class_key:
                continue
            if _angle_difference(first_orientation, _average_orientation(second_component)) > max_orientation_difference_deg:
                continue
            second_geometry = inner_row_geometries[second_index]
            if first_geometry.distance(second_geometry) > median_short * max_inner_row_gap_factor:
                continue
            second_interval = _projection_interval(second_geometry, row_axis_x, row_axis_y)
            overlap = max(0.0, min(first_interval[1], second_interval[1]) - max(first_interval[0], second_interval[0]))
            smaller_span = min(first_interval[1] - first_interval[0], second_interval[1] - second_interval[0])
            if smaller_span > 0 and overlap / smaller_span >= 0.35:
                array_groups.union(first_index, second_index)

    arrays: dict[int, list[int]] = {}
    for index in range(len(inner_rows)):
        arrays.setdefault(array_groups.find(index), []).append(index)
    array_indices = list(arrays.values())
    array_geometries = [
        unary_union([inner_row_geometries[index] for index in indices])
        .convex_hull
        .minimum_rotated_rectangle
        for indices in array_indices
    ]
    containment_groups = _Groups(len(array_indices))
    for first_index, first_geometry in enumerate(array_geometries):
        for second_index in range(first_index + 1, len(array_geometries)):
            second_geometry = array_geometries[second_index]
            smaller_area = min(first_geometry.area, second_geometry.area)
            overlap_area = first_geometry.intersection(second_geometry).area
            overlap_percent = 100.0 * overlap_area / smaller_area if smaller_area > 0 else 0.0
            if overlap_area > 0 and overlap_percent >= min_row_overlap_percent:
                containment_groups.union(first_index, second_index)
    merged_arrays: dict[int, list[int]] = {}
    for index, inner_row_indices in enumerate(array_indices):
        merged_arrays.setdefault(containment_groups.find(index), []).extend(inner_row_indices)
    ordered_arrays = _map_reading_order(
        list(merged_arrays.values()), inner_row_geometries
    )
    row_features: list[dict[str, Any]] = []
    panel_features: list[dict[str, Any]] = []
    _notify(callback, 62, "Assigning row and panel identifiers…")
    inner_row_total = 0
    for array_number, inner_row_indices in enumerate(ordered_arrays, start=1000):
        row_id = str(array_number)
        array_panels = [panel for index in inner_row_indices for panel in inner_rows[index]]
        orientation = _average_orientation(array_panels)
        row_geometry = (
            unary_union([panel.geometry for panel in array_panels])
            .convex_hull
            .minimum_rotated_rectangle
        )
        minx, miny, maxx, maxy = row_geometry.bounds
        array_is_horizontal = (maxx - minx) >= (maxy - miny)
        ordered_inner_rows = sorted(
            (inner_rows[index] for index in inner_row_indices),
            key=(
                (lambda component: (
                    -statistics.mean(panel.centre_y for panel in component),
                    statistics.mean(panel.centre_x for panel in component),
                ))
                if array_is_horizontal
                else (lambda component: (
                    statistics.mean(panel.centre_x for panel in component),
                    -statistics.mean(panel.centre_y for panel in component),
                ))
            ),
        )
        row_properties = {
            "postprocess_stage": "panel_rows",
            "panel_count": len(array_panels),
            "inner_row_count": len(ordered_inner_rows),
            "orientation_deg": round(orientation, 4),
            "area_m2": float(row_geometry.area),
        }
        if assign_ids:
            row_properties["row_id"] = row_id
        row_features.append(feature(row_geometry, row_properties, to_wgs84))
        inner_row_total += len(ordered_inner_rows)
        if not assign_ids:
            continue
        for inner_row_index, component in enumerate(ordered_inner_rows):
            inner_row_label = _letter_label(inner_row_index)
            component_geometry = unary_union([panel.geometry for panel in component]).bounds
            component_is_horizontal = (
                component_geometry[2] - component_geometry[0]
                >= component_geometry[3] - component_geometry[1]
            )
            component.sort(
                key=(
                    (lambda panel: (panel.centre_x, -panel.centre_y))
                    if component_is_horizontal
                    else (lambda panel: (-panel.centre_y, panel.centre_x))
                )
            )
            for panel_number, panel in enumerate(component, start=1):
                properties = dict(panel.properties)
                properties.update({
                    "postprocess_stage": "identified_panels",
                    "row_id": row_id,
                    "inner_row": inner_row_label,
                    "panel_number": panel_number,
                    "panel_id": f"{row_id}-{inner_row_label}{panel_number}",
                    "row_panel_count": len(array_panels),
                    "inner_row_panel_count": len(component),
                    "source_panel_index": panel.source_index,
                })
                panel_features.append(feature(panel.geometry, properties, to_wgs84))
    metadata = {"source": str(input_path), "metric_crs": metric_crs.to_string()}
    hierarchy_features = row_features + panel_features
    if output_path is not None:
        write_feature_collection(output_path, hierarchy_features, **metadata)
    if rows_output_path is not None:
        write_feature_collection(rows_output_path, row_features, **metadata)
    if panels_output_path is not None and assign_ids:
        write_feature_collection(panels_output_path, panel_features, **metadata)
    _notify(callback, 100, "Panel hierarchy is ready." if assign_ids else "Rows are ready for editing.")
    return {
        "input_features": len(payload["features"]),
        "invalid_input_features": invalid,
        "panel_count": len(panel_features) if assign_ids else len(panels),
        "row_count": len(row_features),
        "output_features": len(hierarchy_features),
        "inner_row_count": inner_row_total,
        "singleton_rows": sum(1 for indices in ordered_arrays if sum(len(inner_rows[index]) for index in indices) == 1),
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path) if output_path else None,
        "rows_output_path": str(rows_output_path) if rows_output_path else None,
        "panels_output_path": str(panels_output_path) if panels_output_path else None,
    }


_PANEL_ID_KEYS = {
    "row_id", "inner_row", "panel_number", "panel_id", "row_panel_count",
    "inner_row_panel_count", "source_panel_index",
}


def clear_panel_ids(path: Path) -> None:
    """Remove hierarchy identifiers from a GeoJSON layer in place."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features")
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise ValueError("The selected file is not a GeoJSON FeatureCollection.")
    cleaned_features = []
    for original in features:
        cleaned = dict(original)
        cleaned["properties"] = {
            key: value for key, value in (original.get("properties") or {}).items()
            if key not in _PANEL_ID_KEYS
        }
        cleaned_features.append(cleaned)
    metadata = {key: value for key, value in payload.items() if key not in {"type", "features"}}
    write_feature_collection(path, cleaned_features, **metadata)


def assign_panel_ids(
    panels_path: Path,
    rows_path: Path,
    *,
    max_orientation_difference_deg: float = 15.0,
    max_lateral_distance_factor: float = 1.5,
    max_along_gap_factor: float = 1.5,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Assign IDs to panels and edited rows without changing row geometry."""
    panel_payload, panel_records, invalid_panels = load_polygon_features(panels_path)
    row_payload, row_records, invalid_rows = load_polygon_features(rows_path)
    metric_crs = infer_metric_crs(record[1] for record in panel_records)
    panels: list[Panel] = []
    for source_index, geometry, properties in panel_records:
        projected = project_geometry(geometry, "EPSG:4326", metric_crs)
        short, long, angle = _rectangle_measurements(projected)
        centre = projected.centroid
        panels.append(Panel(source_index, projected, properties, centre.x, centre.y, short, long, angle, class_key(properties)))
    projected_rows = [
        (source_index, project_geometry(geometry, "EPSG:4326", metric_crs), properties)
        for source_index, geometry, properties in row_records
    ]
    ordered_indices = [
        indices[0]
        for indices in _map_reading_order(
            [[index] for index in range(len(projected_rows))],
            [record[1] for record in projected_rows],
        )
    ]
    panels_by_row: dict[int, list[Panel]] = {index: [] for index in range(len(projected_rows))}
    _notify(callback, 20, "Matching regularized panels to edited rows…")
    for panel_index, panel in enumerate(panels):
        best_row_index = None
        best_overlap = 0.0
        for row_index, (_, row_geometry, _) in enumerate(projected_rows):
            overlap = panel.geometry.intersection(row_geometry).area
            if overlap > best_overlap:
                best_overlap = overlap
                best_row_index = row_index
        if best_row_index is not None and best_overlap > 0:
            panels_by_row[best_row_index].append(panel)
        if panel_index % 250 == 0:
            _notify(callback, 20 + int(35 * panel_index / max(1, len(panels))), "Matching regularized panels to edited rows…")

    panel_updates: dict[int, dict[str, Any]] = {}
    row_updates: dict[int, dict[str, Any]] = {}
    inner_row_total = 0
    assigned_panel_count = 0
    _notify(callback, 58, "Assigning row and panel identifiers…")
    for row_number, row_index in enumerate(ordered_indices, start=1000):
        source_index, row_geometry, row_properties = projected_rows[row_index]
        row_id = str(row_number)
        row_panels = panels_by_row[row_index]
        inner_rows = []
        if row_panels:
            inner_rows, _, _ = _group_inner_rows(
                row_panels,
                max_orientation_difference_deg,
                max_lateral_distance_factor,
                max_along_gap_factor,
            )
        minx, miny, maxx, maxy = row_geometry.bounds
        row_is_horizontal = (maxx - minx) >= (maxy - miny)
        inner_rows.sort(
            key=(
                (lambda component: (-statistics.mean(panel.centre_y for panel in component), statistics.mean(panel.centre_x for panel in component)))
                if row_is_horizontal
                else (lambda component: (statistics.mean(panel.centre_x for panel in component), -statistics.mean(panel.centre_y for panel in component)))
            )
        )
        updated_row_properties = {key: value for key, value in row_properties.items() if key not in _PANEL_ID_KEYS}
        updated_row_properties.update({
            "postprocess_stage": "panel_rows",
            "row_id": row_id,
            "panel_count": len(row_panels),
            "inner_row_count": len(inner_rows),
            "area_m2": float(row_geometry.area),
        })
        updated_row = dict(row_payload["features"][source_index])
        updated_row["properties"] = updated_row_properties
        row_updates[source_index] = updated_row
        inner_row_total += len(inner_rows)
        assigned_panel_count += len(row_panels)
        for inner_row_index, component in enumerate(inner_rows):
            inner_row_label = _letter_label(inner_row_index)
            component_bounds = unary_union([panel.geometry for panel in component]).bounds
            component_is_horizontal = component_bounds[2] - component_bounds[0] >= component_bounds[3] - component_bounds[1]
            component.sort(
                key=(
                    (lambda panel: (panel.centre_x, -panel.centre_y))
                    if component_is_horizontal
                    else (lambda panel: (-panel.centre_y, panel.centre_x))
                )
            )
            for panel_number, panel in enumerate(component, start=1):
                properties = {key: value for key, value in panel.properties.items() if key not in _PANEL_ID_KEYS}
                properties.update({
                    "postprocess_stage": "identified_panels",
                    "row_id": row_id,
                    "inner_row": inner_row_label,
                    "panel_number": panel_number,
                    "panel_id": f"{row_id}-{inner_row_label}{panel_number}",
                    "row_panel_count": len(row_panels),
                    "inner_row_panel_count": len(component),
                    "source_panel_index": panel.source_index,
                })
                updated_panel = dict(panel_payload["features"][panel.source_index])
                updated_panel["properties"] = properties
                panel_updates[panel.source_index] = updated_panel

    def updated_features(payload: dict[str, Any], updates: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for index, original in enumerate(payload["features"]):
            if index in updates:
                result.append(updates[index])
                continue
            cleaned = dict(original)
            cleaned["properties"] = {
                key: value for key, value in (original.get("properties") or {}).items()
                if key not in _PANEL_ID_KEYS
            }
            result.append(cleaned)
        return result

    panel_metadata = {key: value for key, value in panel_payload.items() if key not in {"type", "features"}}
    row_metadata = {key: value for key, value in row_payload.items() if key not in {"type", "features"}}
    write_feature_collection(panels_path, updated_features(panel_payload, panel_updates), **panel_metadata)
    write_feature_collection(rows_path, updated_features(row_payload, row_updates), **row_metadata)
    _notify(callback, 100, "Row and panel IDs are ready.")
    return {
        "panel_count": len(panels),
        "assigned_panel_count": assigned_panel_count,
        "unassigned_panel_count": len(panels) - assigned_panel_count,
        "invalid_panel_count": invalid_panels,
        "row_count": len(projected_rows),
        "invalid_row_count": invalid_rows,
        "inner_row_count": inner_row_total,
        "metric_crs": metric_crs.to_string(),
    }
