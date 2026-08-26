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


def build_panel_hierarchy(
    input_path: Path,
    rows_output_path: Path,
    panels_output_path: Path,
    *,
    max_orientation_difference_deg: float = 12.0,
    max_lateral_distance_factor: float = 1.5,
    max_along_gap_factor: float = 2.5,
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
    ordered_components = sorted(
        components.values(),
        key=lambda component: (-statistics.mean(panel.centre_y for panel in component), statistics.mean(panel.centre_x for panel in component)),
    )
    row_features: list[dict[str, Any]] = []
    panel_features: list[dict[str, Any]] = []
    _notify(callback, 62, "Assigning row and panel identifiers…")
    for row_number, component in enumerate(ordered_components, start=1):
        row_id = f"ROW-{row_number:04d}"
        orientation = _average_orientation(component)
        radians = math.radians((orientation + 90.0) % 180.0)
        component.sort(key=lambda panel: panel.centre_x * math.cos(radians) + panel.centre_y * math.sin(radians))
        row_geometry = unary_union([panel.geometry for panel in component]).convex_hull.minimum_rotated_rectangle
        row_properties = {
            "postprocess_stage": "panel_rows",
            "row_id": row_id,
            "panel_count": len(component),
            "orientation_deg": round(orientation, 4),
            "area_m2": float(row_geometry.area),
        }
        row_features.append(feature(row_geometry, row_properties, to_wgs84))
        for panel_number, panel in enumerate(component, start=1):
            properties = dict(panel.properties)
            properties.update({
                "postprocess_stage": "identified_panels",
                "row_id": row_id,
                "panel_id": f"{row_id}-PANEL-{panel_number:04d}",
                "panel_order": panel_number,
                "row_panel_count": len(component),
                "source_panel_index": panel.source_index,
            })
            panel_features.append(feature(panel.geometry, properties, to_wgs84))
    metadata = {"source": str(input_path), "metric_crs": metric_crs.to_string()}
    write_feature_collection(rows_output_path, row_features, **metadata)
    write_feature_collection(panels_output_path, panel_features, **metadata)
    _notify(callback, 100, "Panel hierarchy is ready.")
    return {
        "input_features": len(payload["features"]),
        "invalid_input_features": invalid,
        "panel_count": len(panel_features),
        "row_count": len(row_features),
        "singleton_rows": sum(1 for component in ordered_components if len(component) == 1),
        "metric_crs": metric_crs.to_string(),
        "rows_output_path": str(rows_output_path),
        "panels_output_path": str(panels_output_path),
    }
