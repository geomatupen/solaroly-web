from __future__ import annotations

import json
import math
import numbers
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, mapping, shape
from shapely.ops import transform as transform_geometry
from shapely.ops import unary_union
from shapely.strtree import STRtree

try:
    from shapely import make_valid
except ImportError:  # pragma: no cover - Shapely < 2 fallback
    make_valid = None


ProgressCallback = Callable[[int, str], None]
_TILE_NAME_RE = re.compile(r"^(?P<prefix>.+)_(?P<row>\d+)_(?P<col>\d+)$")


@dataclass(frozen=True)
class TileInfo:
    name: str
    path: Path
    prefix: str
    row: int
    col: int
    width: int
    height: int
    crs: CRS
    transform: Any


@dataclass
class FeatureRecord:
    index: int
    geometry_wgs84: Any
    geometry_metric: Any
    properties: dict[str, Any]
    class_key: str
    tile: TileInfo | None
    pixel_bounds: tuple[float, float, float, float] | None
    boundary_sides: frozenset[str]
    pixel_size_m: float | None


def _notify(callback: ProgressCallback | None, progress: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, int(progress))), message)


def _load_feature_collection(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read GeoJSON: {exc}") from exc
    if data.get("type") != "FeatureCollection" or not isinstance(data.get("features"), list):
        raise ValueError("The selected file is not a GeoJSON FeatureCollection.")
    return data


def _polygonal(geometry: Any) -> Any:
    if geometry is None or geometry.is_empty:
        return None
    if not geometry.is_valid:
        try:
            geometry = make_valid(geometry) if make_valid else geometry.buffer(0)
        except Exception:
            geometry = geometry.buffer(0)
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        polygons = [part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        return unary_union(polygons) if polygons else None
    return None


def _feature_geometry(feature: dict[str, Any]) -> Any:
    try:
        return _polygonal(shape(feature.get("geometry") or {}))
    except Exception:
        return None


def _iter_xy(geometry: Any) -> Iterable[tuple[float, float]]:
    if geometry.geom_type == "Polygon":
        yield from geometry.exterior.coords
    elif geometry.geom_type == "MultiPolygon":
        for polygon in geometry.geoms:
            yield from polygon.exterior.coords


def _infer_metric_crs(geometries: list[Any]) -> CRS:
    if not geometries:
        raise ValueError("The selected GeoJSON does not contain polygon features.")
    minx = min(geometry.bounds[0] for geometry in geometries)
    miny = min(geometry.bounds[1] for geometry in geometries)
    maxx = max(geometry.bounds[2] for geometry in geometries)
    maxy = max(geometry.bounds[3] for geometry in geometries)
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0
    if not (-180 <= lon <= 180 and -90 <= lat <= 90):
        raise ValueError("GeoJSON coordinates must be WGS84 longitude/latitude.")
    zone = max(1, min(60, int((lon + 180) // 6) + 1))
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _transformer(source: CRS | str, destination: CRS | str) -> Callable[..., Any]:
    return Transformer.from_crs(source, destination, always_xy=True).transform


def _tile_name(value: Any) -> str:
    return Path(str(value or "")).stem


def _read_tile_info(result_dir: Path, tile_name: str) -> TileInfo | None:
    match = _TILE_NAME_RE.match(tile_name)
    if not match:
        return None
    tile_path = next(
        (
            candidate
            for extension in (".tif", ".tiff")
            if (candidate := result_dir / "tiles" / f"{tile_name}{extension}").is_file()
        ),
        None,
    )
    if tile_path is None:
        return None
    try:
        with rasterio.open(tile_path) as dataset:
            if dataset.crs is None:
                return None
            return TileInfo(
                name=tile_name,
                path=tile_path,
                prefix=match.group("prefix"),
                row=int(match.group("row")),
                col=int(match.group("col")),
                width=int(dataset.width),
                height=int(dataset.height),
                crs=CRS.from_user_input(dataset.crs),
                transform=dataset.transform,
            )
    except (OSError, rasterio.errors.RasterioError):
        return None


def _pixel_geometry(geometry_wgs84: Any, tile: TileInfo) -> Any:
    native = transform_geometry(_transformer("EPSG:4326", tile.crs), geometry_wgs84)
    inverse = ~tile.transform

    def to_pixel(x: Any, y: Any, z: Any = None) -> tuple[Any, Any]:
        col = inverse.a * x + inverse.b * y + inverse.c
        row = inverse.d * x + inverse.e * y + inverse.f
        return col, row

    return transform_geometry(to_pixel, native)


def _pixel_size_metres(tile: TileInfo, metric_crs: CRS) -> float:
    to_wgs = _transformer(tile.crs, "EPSG:4326")
    to_metric = Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True)
    center_col = tile.width / 2.0
    center_row = tile.height / 2.0
    samples = [
        tile.transform * (center_col, center_row),
        tile.transform * (center_col + 1.0, center_row),
        tile.transform * (center_col, center_row + 1.0),
    ]
    lon, lat = to_wgs([point[0] for point in samples], [point[1] for point in samples])
    x, y = to_metric.transform(lon, lat)
    horizontal = math.hypot(x[1] - x[0], y[1] - y[0])
    vertical = math.hypot(x[2] - x[0], y[2] - y[0])
    return max(1e-9, (horizontal + vertical) / 2.0)


def _class_key(properties: dict[str, Any]) -> str:
    class_id = properties.get("class_id", properties.get("class"))
    class_name = properties.get("class_name", properties.get("classname", "unknown"))
    return f"{class_id}:{class_name}"


def _prepare_records(
    data: dict[str, Any],
    result_dir: Path,
    edge_tolerance_px: float,
    callback: ProgressCallback | None = None,
) -> tuple[list[FeatureRecord], CRS, dict[str, TileInfo], int]:
    features = data.get("features", [])
    parsed: list[tuple[int, Any, dict[str, Any]]] = []
    invalid_count = 0
    for index, feature in enumerate(features):
        geometry = _feature_geometry(feature)
        if geometry is None or geometry.is_empty or geometry.area <= 0:
            invalid_count += 1
            continue
        parsed.append((index, geometry, dict(feature.get("properties") or {})))
    metric_crs = _infer_metric_crs([item[1] for item in parsed])
    to_metric = _transformer("EPSG:4326", metric_crs)
    tile_cache: dict[str, TileInfo] = {}
    pixel_sizes: dict[str, float] = {}
    records: list[FeatureRecord] = []
    total = max(1, len(parsed))
    for position, (index, geometry, properties) in enumerate(parsed):
        tile_name = _tile_name(properties.get("tile"))
        tile = None
        pixel_bounds = None
        sides: set[str] = set()
        pixel_size = None
        if tile_name:
            if tile_name not in tile_cache:
                tile_info = _read_tile_info(result_dir, tile_name)
                if tile_info:
                    tile_cache[tile_name] = tile_info
            tile = tile_cache.get(tile_name)
        if tile:
            try:
                pixels = _pixel_geometry(geometry, tile)
                pixel_bounds = tuple(float(value) for value in pixels.bounds)
                minx, miny, maxx, maxy = pixel_bounds
                if minx <= edge_tolerance_px:
                    sides.add("left")
                if maxx >= tile.width - edge_tolerance_px:
                    sides.add("right")
                if miny <= edge_tolerance_px:
                    sides.add("top")
                if maxy >= tile.height - edge_tolerance_px:
                    sides.add("bottom")
                if tile.name not in pixel_sizes:
                    pixel_sizes[tile.name] = _pixel_size_metres(tile, metric_crs)
                pixel_size = pixel_sizes[tile.name]
            except Exception:
                tile = None
                pixel_bounds = None
                sides.clear()
        records.append(
            FeatureRecord(
                index=index,
                geometry_wgs84=geometry,
                geometry_metric=transform_geometry(to_metric, geometry),
                properties=properties,
                class_key=_class_key(properties),
                tile=tile,
                pixel_bounds=pixel_bounds,
                boundary_sides=frozenset(sides),
                pixel_size_m=pixel_size,
            )
        )
        if position % 500 == 0:
            _notify(callback, 5 + int(20 * position / total), "Inspecting polygon and tile boundaries…")
    return records, metric_crs, tile_cache, invalid_count


def _rectangle_dimensions(geometry: Any) -> tuple[float, float, float]:
    rectangle = geometry.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.geom_type != "Polygon":
        return 0.0, 0.0, 0.0
    coordinates = list(rectangle.exterior.coords)[:4]
    lengths = [
        math.dist(coordinates[index], coordinates[(index + 1) % 4]) for index in range(4)
    ]
    return min(lengths), max(lengths), float(rectangle.area)


def _median_or_none(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _templates(records: list[FeatureRecord]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[FeatureRecord]] = {}
    for record in records:
        if not record.boundary_sides:
            grouped.setdefault(record.class_key, []).append(record)
    templates: dict[str, dict[str, float]] = {}
    for class_key, class_records in grouped.items():
        dimensions = [_rectangle_dimensions(record.geometry_metric) for record in class_records]
        dimensions = [item for item in dimensions if item[0] > 0 and item[1] > 0]
        if not dimensions:
            continue
        templates[class_key] = {
            "area": float(statistics.median(record.geometry_metric.area for record in class_records)),
            "short": float(statistics.median(item[0] for item in dimensions)),
            "long": float(statistics.median(item[1] for item in dimensions)),
            "sample_count": float(len(dimensions)),
        }
    return templates


def analyze_geojson(
    input_path: Path,
    result_dir: Path,
    *,
    edge_tolerance_px: float = 7.0,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _notify(callback, 1, "Reading GeoJSON…")
    data = _load_feature_collection(input_path)
    records, metric_crs, tiles, invalid_count = _prepare_records(
        data, result_dir, edge_tolerance_px, callback
    )
    geometry_types: dict[str, int] = {}
    for feature in data.get("features", []):
        geometry_type = str((feature.get("geometry") or {}).get("type") or "Unknown")
        geometry_types[geometry_type] = geometry_types.get(geometry_type, 0) + 1
    edge_records = [record for record in records if record.boundary_sides]
    templates = _templates(records)
    pixel_sizes = [record.pixel_size_m for record in records if record.pixel_size_m]
    _notify(callback, 100, "Analysis complete.")
    return {
        "feature_count": len(data.get("features", [])),
        "valid_polygon_count": len(records),
        "invalid_feature_count": invalid_count,
        "geometry_types": geometry_types,
        "features_on_tile_edges": len(edge_records),
        "features_away_from_tile_edges": len(records) - len(edge_records),
        "tile_count": len(tiles),
        "tile_metadata_available": bool(tiles),
        "metric_crs": metric_crs.to_string(),
        "median_pixel_size_m": _median_or_none(pixel_sizes),
        "templates": templates,
    }


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.members = {index: {index} for index in range(size)}

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: int, second: int) -> int:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return root_first
        if len(self.members[root_first]) < len(self.members[root_second]):
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        self.members[root_first].update(self.members.pop(root_second))
        return root_first


def _interval_overlap(
    first: tuple[float, float], second: tuple[float, float]
) -> tuple[float, float]:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    smaller = max(1e-9, min(first[1] - first[0], second[1] - second[0]))
    return overlap, overlap / smaller


def _neighbor_pairs(
    records: list[FeatureRecord], min_boundary_overlap: float
) -> list[tuple[float, int, int]]:
    tile_lookup = {
        (record.tile.prefix, record.tile.row, record.tile.col): record.tile
        for record in records
        if record.tile
    }
    side_index: dict[tuple[str, str], list[int]] = {}
    for index, record in enumerate(records):
        if not record.tile:
            continue
        for side in record.boundary_sides:
            side_index.setdefault((record.tile.name, side), []).append(index)

    candidates: list[tuple[float, int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, first in enumerate(records):
        tile = first.tile
        bounds = first.pixel_bounds
        if not tile or not bounds:
            continue
        minx, miny, maxx, maxy = bounds
        requests = []
        if "right" in first.boundary_sides:
            requests.append((tile.row, tile.col + tile.width, "left", (miny, maxy), "vertical"))
        if "bottom" in first.boundary_sides:
            requests.append((tile.row + tile.height, tile.col, "top", (minx, maxx), "horizontal"))
        for row, col, opposite_side, first_interval, axis in requests:
            neighbor = tile_lookup.get((tile.prefix, row, col))
            if not neighbor:
                continue
            for other_index in side_index.get((neighbor.name, opposite_side), []):
                if other_index == index:
                    continue
                pair = (min(index, other_index), max(index, other_index))
                if pair in seen:
                    continue
                seen.add(pair)
                second = records[other_index]
                if first.class_key != second.class_key or not second.pixel_bounds:
                    continue
                ominx, ominy, omaxx, omaxy = second.pixel_bounds
                second_interval = (ominy, omaxy) if axis == "vertical" else (ominx, omaxx)
                overlap, ratio = _interval_overlap(first_interval, second_interval)
                if overlap <= 0 or ratio < min_boundary_overlap:
                    continue
                candidates.append((-ratio, index, other_index))
    candidates.sort()
    return candidates


def _plausible_component(
    geometry: Any,
    template: dict[str, float] | None,
    max_dimension_factor: float,
    max_area_factor: float,
) -> bool:
    if not template:
        return True
    short, long, _ = _rectangle_dimensions(geometry)
    return (
        geometry.area <= template["area"] * max_area_factor
        and short <= template["short"] * max_dimension_factor
        and long <= template["long"] * max_dimension_factor
    )


def _aggregate_properties(group: list[FeatureRecord]) -> dict[str, Any]:
    scored = []
    for record in group:
        try:
            scored.append((float(record.properties.get("score", 0.0)), record))
        except (TypeError, ValueError):
            scored.append((0.0, record))
    scored.sort(key=lambda item: item[0], reverse=True)
    properties = dict(scored[0][1].properties if scored else group[0].properties)
    scores = [item[0] for item in scored]
    tiles = sorted({record.tile.name for record in group if record.tile})
    properties.update(
        {
            "score": max(scores) if scores else 0.0,
            "score_mean": sum(scores) / len(scores) if scores else 0.0,
            "source_feature_count": len(group),
            "source_feature_indices": [record.index for record in group],
            "source_tiles": tiles,
            "postprocess_stage": "combined",
        }
    )
    return properties


def _write_feature_collection(path: Path, features: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_contained_components(
    components: list[tuple[Any, dict[str, Any], str]],
    coverage_threshold: float = 0.98,
) -> tuple[list[tuple[Any, dict[str, Any], str]], int]:
    """Remove same-class shapes almost entirely covered by a preferred outer shape."""
    if len(components) < 2:
        return components, 0
    geometries = [item[0] for item in components]
    tree = STRtree(geometries)
    geometry_indexes = {id(geometry): index for index, geometry in enumerate(geometries)}
    removed: set[int] = set()

    def score(index: int) -> float:
        try:
            return float(components[index][1].get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    for index, inner in enumerate(geometries):
        if index in removed or inner.is_empty or inner.area <= 0:
            continue
        for candidate in tree.query(inner):
            other_index = (
                int(candidate)
                if isinstance(candidate, numbers.Integral)
                else geometry_indexes.get(id(candidate), -1)
            )
            if other_index < 0 or other_index == index or other_index in removed:
                continue
            outer = geometries[other_index]
            if components[index][2] != components[other_index][2]:
                continue
            area_tolerance = max(inner.area, outer.area) * 1e-6
            outer_is_preferred = outer.area > inner.area + area_tolerance
            if abs(outer.area - inner.area) <= area_tolerance:
                outer_is_preferred = (score(other_index), -other_index) > (score(index), -index)
            if not outer_is_preferred:
                continue
            try:
                coverage = inner.intersection(outer).area / inner.area
            except Exception:
                continue
            if coverage >= coverage_threshold:
                removed.add(index)
                break
    return [item for index, item in enumerate(components) if index not in removed], len(removed)


def combine_tile_fragments(
    input_path: Path,
    output_path: Path,
    result_dir: Path,
    *,
    edge_tolerance_px: float = 7.0,
    gap_tolerance_px: float = 10.0,
    min_boundary_overlap: float = 0.20,
    max_dimension_factor: float = 1.65,
    max_area_factor: float = 1.75,
    remove_contained_polygons: bool = True,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _notify(callback, 1, "Reading and validating polygons…")
    data = _load_feature_collection(input_path)
    records, metric_crs, tiles, invalid_count = _prepare_records(
        data, result_dir, edge_tolerance_px, callback
    )
    if not tiles:
        raise ValueError(
            "This GeoJSON has no usable result-tile metadata. Grid-aware combining requires the saved tiles."
        )
    templates = _templates(records)
    _notify(callback, 30, "Finding matching fragments on adjacent tile edges…")
    candidates = _neighbor_pairs(records, min_boundary_overlap)
    groups = _DisjointSet(len(records))
    accepted = 0
    rejected_distance = 0
    rejected_size = 0
    total_candidates = max(1, len(candidates))
    for position, (_, first_index, second_index) in enumerate(candidates):
        first_root = groups.find(first_index)
        second_root = groups.find(second_index)
        if first_root == second_root:
            continue
        first = records[first_index]
        second = records[second_index]
        pixel_size = statistics.median(
            value
            for value in (first.pixel_size_m, second.pixel_size_m)
            if value is not None
        )
        maximum_gap = gap_tolerance_px * pixel_size
        first_geometry = unary_union(
            [records[index].geometry_metric for index in groups.members[first_root]]
        )
        second_geometry = unary_union(
            [records[index].geometry_metric for index in groups.members[second_root]]
        )
        if first_geometry.distance(second_geometry) > maximum_gap:
            rejected_distance += 1
            continue
        combined = unary_union([first_geometry, second_geometry])
        template = templates.get(first.class_key)
        if not _plausible_component(
            combined, template, max_dimension_factor, max_area_factor
        ):
            rejected_size += 1
            continue
        groups.union(first_root, second_root)
        accepted += 1
        if position % 100 == 0:
            _notify(
                callback,
                35 + int(40 * position / total_candidates),
                f"Combining adjacent fragments… {position}/{len(candidates)}",
            )

    _notify(callback, 78, "Building combined GeoJSON…")
    to_wgs84 = _transformer(metric_crs, "EPSG:4326")
    components: list[tuple[Any, dict[str, Any], str]] = []
    merged_components = 0
    for member_indices in groups.members.values():
        group = [records[index] for index in sorted(member_indices)]
        merged_metric = unary_union([record.geometry_metric for record in group])
        if len(group) > 1:
            pixel_sizes = [record.pixel_size_m for record in group if record.pixel_size_m]
            if pixel_sizes and merged_metric.geom_type == "MultiPolygon":
                bridge = gap_tolerance_px * statistics.median(pixel_sizes) / 2.0
                closed = merged_metric.buffer(bridge, join_style=2).buffer(-bridge, join_style=2)
                if not closed.is_empty:
                    merged_metric = closed
            merged_components += 1
        components.append((merged_metric, _aggregate_properties(group), group[0].class_key))
    contained_removed = 0
    if remove_contained_polygons:
        _notify(callback, 88, "Removing polygons contained inside larger polygons…")
        components, contained_removed = _remove_contained_components(components)
    output_features = [
        {
            "type": "Feature",
            "geometry": mapping(transform_geometry(to_wgs84, geometry)),
            "properties": properties,
        }
        for geometry, properties, _ in components
    ]
    _write_feature_collection(output_path, output_features)
    _notify(callback, 100, "Fragment combining complete.")
    return {
        "input_features": len(data.get("features", [])),
        "valid_input_features": len(records),
        "invalid_input_features": invalid_count,
        "candidate_pairs": len(candidates),
        "accepted_links": accepted,
        "rejected_by_distance": rejected_distance,
        "rejected_by_panel_size": rejected_size,
        "merged_components": merged_components,
        "contained_polygons_removed": contained_removed,
        "remove_contained_polygons": remove_contained_polygons,
        "output_features": len(output_features),
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path),
    }


def regularize_polygons(
    input_path: Path,
    output_path: Path,
    *,
    max_area_change_percent: float = 35.0,
    callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    _notify(callback, 1, "Reading combined polygons…")
    data = _load_feature_collection(input_path)
    parsed = []
    invalid_count = 0
    for index, feature in enumerate(data.get("features", [])):
        geometry = _feature_geometry(feature)
        if geometry is None or geometry.is_empty:
            invalid_count += 1
            continue
        parsed.append((index, geometry, dict(feature.get("properties") or {})))
    metric_crs = _infer_metric_crs([item[1] for item in parsed])
    to_metric = _transformer("EPSG:4326", metric_crs)
    to_wgs84 = _transformer(metric_crs, "EPSG:4326")
    output_features = []
    flagged = 0
    total = max(1, len(parsed))
    for position, (index, geometry, properties) in enumerate(parsed):
        metric_geometry = transform_geometry(to_metric, geometry)
        rectangle = metric_geometry.minimum_rotated_rectangle
        if rectangle.is_empty or rectangle.geom_type != "Polygon":
            invalid_count += 1
            continue
        area_before = float(metric_geometry.area)
        area_after = float(rectangle.area)
        area_change = (
            abs(area_after - area_before) / area_before * 100.0 if area_before > 0 else 0.0
        )
        review_required = area_change > max_area_change_percent
        if review_required:
            flagged += 1
        short, long, _ = _rectangle_dimensions(rectangle)
        regularized_properties = dict(properties)
        regularized_properties.update(
            {
                "postprocess_stage": "regularized",
                "regularization": "minimum_rotated_rectangle",
                "source_combined_index": index,
                "area_before_m2": area_before,
                "area_after_m2": area_after,
                "area_change_percent": area_change,
                "rectangle_short_m": short,
                "rectangle_long_m": long,
                "review_required": review_required,
            }
        )
        output_features.append(
            {
                "type": "Feature",
                "geometry": mapping(transform_geometry(to_wgs84, rectangle)),
                "properties": regularized_properties,
            }
        )
        if position % 500 == 0:
            _notify(
                callback,
                10 + int(80 * position / total),
                f"Fitting oriented rectangles… {position}/{len(parsed)}",
            )
    _write_feature_collection(output_path, output_features)
    _notify(callback, 100, "Polygon regularization complete.")
    return {
        "input_features": len(data.get("features", [])),
        "invalid_input_features": invalid_count,
        "output_features": len(output_features),
        "review_required": flagged,
        "metric_crs": metric_crs.to_string(),
        "output_path": str(output_path),
    }
