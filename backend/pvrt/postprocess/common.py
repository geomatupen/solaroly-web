"""Shared GeoJSON and projected-geometry helpers for post-processing stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pyproj import CRS, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as transform_geometry

try:
    from shapely import make_valid
except ImportError:  # pragma: no cover - Shapely < 2 fallback
    make_valid = None


def polygonal_geometry(feature: dict[str, Any]) -> Any:
    """Return a valid polygonal geometry for a GeoJSON feature, if possible."""
    try:
        geometry = shape(feature.get("geometry") or {})
    except Exception:
        return None
    if geometry.is_empty:
        return None
    if not geometry.is_valid:
        try:
            geometry = make_valid(geometry) if make_valid else geometry.buffer(0)
        except Exception:
            geometry = geometry.buffer(0)
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    if geometry.geom_type == "GeometryCollection":
        parts = [part for part in geometry.geoms if part.geom_type in {"Polygon", "MultiPolygon"}]
        if parts:
            from shapely.ops import unary_union

            return unary_union(parts)
    return None


def load_polygon_features(path: Path) -> tuple[dict[str, Any], list[tuple[int, Any, dict[str, Any]]], int]:
    """Load a FeatureCollection and return valid polygon records plus invalid count."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read GeoJSON: {exc}") from exc
    if payload.get("type") != "FeatureCollection" or not isinstance(payload.get("features"), list):
        raise ValueError("The selected file is not a GeoJSON FeatureCollection.")
    records: list[tuple[int, Any, dict[str, Any]]] = []
    invalid = 0
    for index, feature in enumerate(payload["features"]):
        geometry = polygonal_geometry(feature) if isinstance(feature, dict) else None
        if geometry is None or geometry.is_empty or geometry.area <= 0:
            invalid += 1
            continue
        records.append((index, geometry, dict(feature.get("properties") or {})))
    if not records:
        raise ValueError("The selected GeoJSON does not contain valid polygon features.")
    return payload, records, invalid


def infer_metric_crs(geometries: Iterable[Any]) -> CRS:
    items = list(geometries)
    if not items:
        raise ValueError("The selected GeoJSON does not contain polygon features.")
    minx = min(item.bounds[0] for item in items)
    miny = min(item.bounds[1] for item in items)
    maxx = max(item.bounds[2] for item in items)
    maxy = max(item.bounds[3] for item in items)
    longitude = (minx + maxx) / 2.0
    latitude = (miny + maxy) / 2.0
    if not (-180 <= longitude <= 180 and -90 <= latitude <= 90):
        raise ValueError("GeoJSON coordinates must be WGS84 longitude/latitude.")
    zone = max(1, min(60, int((longitude + 180) // 6) + 1))
    return CRS.from_epsg((32600 if latitude >= 0 else 32700) + zone)


def transformer(source: CRS | str, destination: CRS | str):
    return Transformer.from_crs(source, destination, always_xy=True).transform


def project_geometry(geometry: Any, source: CRS | str, destination: CRS | str) -> Any:
    return transform_geometry(transformer(source, destination), geometry)


def write_feature_collection(path: Path, features: list[dict[str, Any]], **metadata: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": features}
    payload.update(metadata)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def feature(geometry: Any, properties: dict[str, Any], to_wgs84) -> dict[str, Any]:
    return {
        "type": "Feature",
        "geometry": mapping(transform_geometry(to_wgs84, geometry)),
        "properties": properties,
    }


def class_key(properties: dict[str, Any]) -> str:
    class_id = properties.get("class_id", properties.get("class"))
    class_name = properties.get("class_name", properties.get("classname", properties.get("label", "unknown")))
    return f"{class_id}:{class_name}"


def score(properties: dict[str, Any]) -> float:
    for key in ("score", "confidence", "conf", "probability"):
        try:
            return float(properties[key])
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0
