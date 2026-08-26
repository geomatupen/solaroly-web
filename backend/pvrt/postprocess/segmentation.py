"""Create deterministic solar-panel and parent-row hierarchy layers."""

from __future__ import annotations

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


def build_panel_hierarchy(
    input_path: Path,
    output_path: Path,
    *,
    rows_output_path: Path | None = None,
    panels_output_path: Path | None = None,
    max_orientation_difference_deg: float = 12.0,
    max_lateral_distance_factor: float = 1.5,
    max_along_gap_factor: float = 2.5,
    max_inner_row_gap_factor: float = 1.0,
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
    median_short = statistics.median(panel.short_m for panel in panels)
    median_long = statistics.median(panel.long_m for panel in panels)
    groups = _Groups(len(panels))
    # A physical panel row normally progresses across the panels' short axis:
    # portrait panels sit side-by-side while their long axes remain aligned.
    maximum_search = median_short * (max_along_gap_factor + 1.0)
    centre_points = [Point(panel.centre_x, panel.centre_y) for panel in panels]
    centre_tree = STRtree(centre_points)
    _notify(callback, 18, "Finding neighbouring panels with compatible orientation…")
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
        if first_index % 250 == 0:
            _notify(callback, 18 + int(35 * first_index / max(1, len(panels))), "Grouping adjacent panels into rows…")

    components: dict[int, list[Panel]] = {}
    for index, panel in enumerate(panels):
        components.setdefault(groups.find(index), []).append(panel)
    inner_rows = list(components.values())
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
    ordered_arrays = sorted(
        arrays.values(),
        key=lambda indices: (
            -statistics.mean(panel.centre_y for index in indices for panel in inner_rows[index]),
            statistics.mean(panel.centre_x for index in indices for panel in inner_rows[index]),
        ),
    )
    row_features: list[dict[str, Any]] = []
    panel_features: list[dict[str, Any]] = []
    _notify(callback, 62, "Assigning row and panel identifiers…")
    inner_row_total = 0
    for array_number, inner_row_indices in enumerate(ordered_arrays, start=1000):
        row_id = str(array_number)
        array_panels = [panel for index in inner_row_indices for panel in inner_rows[index]]
        orientation = _average_orientation(array_panels)
        row_geometry = unary_union([panel.geometry for panel in array_panels]).convex_hull.minimum_rotated_rectangle
        ordered_inner_rows = sorted(
            (inner_rows[index] for index in inner_row_indices),
            key=lambda component: (
                -statistics.mean(panel.centre_y for panel in component),
                statistics.mean(panel.centre_x for panel in component),
            ),
        )
        row_properties = {
            "postprocess_stage": "panel_rows",
            "row_id": row_id,
            "panel_count": len(array_panels),
            "inner_row_count": len(ordered_inner_rows),
            "orientation_deg": round(orientation, 4),
            "area_m2": float(row_geometry.area),
        }
        row_features.append(feature(row_geometry, row_properties, to_wgs84))
        for inner_row_index, component in enumerate(ordered_inner_rows):
            inner_row_total += 1
            inner_row_label = _letter_label(inner_row_index)
            inner_orientation = _average_orientation(component)
            radians = math.radians((inner_orientation + 90.0) % 180.0)
            axis_x, axis_y = math.cos(radians), math.sin(radians)
            if axis_x < -1e-9 or (abs(axis_x) <= 1e-9 and axis_y < 0):
                axis_x, axis_y = -axis_x, -axis_y
            component.sort(key=lambda panel: panel.centre_x * axis_x + panel.centre_y * axis_y)
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
    write_feature_collection(output_path, hierarchy_features, **metadata)
    if rows_output_path is not None:
        write_feature_collection(rows_output_path, row_features, **metadata)
    if panels_output_path is not None:
        write_feature_collection(panels_output_path, panel_features, **metadata)
    _notify(callback, 100, "Panel hierarchy is ready.")
    return {
        "input_features": len(payload["features"]),
        "invalid_input_features": invalid,
        "panel_count": len(panel_features),
        "row_count": len(row_features),
        "output_features": len(hierarchy_features),
        "inner_row_count": inner_row_total,
        "singleton_rows": sum(1 for indices in ordered_arrays if sum(len(inner_rows[index]) for index in indices) == 1),
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path),
        "rows_output_path": str(rows_output_path) if rows_output_path else None,
        "panels_output_path": str(panels_output_path) if panels_output_path else None,
    }
