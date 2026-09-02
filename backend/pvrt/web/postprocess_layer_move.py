"""Whole-layer movement for post-processing workspaces.

This module owns the coordinate translation, dependent-output archival, and API
route.  The main web application only supplies its project path and JSON-write
helpers, which keeps this feature isolated from the rest of the application.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Request


MOVE_DEPENDENCIES: dict[str, set[str]] = {
    "source": {"combined", "regularized", "solar_rows", "overlap_deduplicated", "deduplicated", "associated"},
    "combined": {"regularized", "solar_rows"},
    "regularized": {"solar_rows"},
    "solar_rows": set(),
    "overlap_deduplicated": {"deduplicated", "associated"},
    "deduplicated": {"associated"},
    "associated": set(),
}


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()[:128]).strip("-._")


def translate_geojson_payload(payload: dict[str, Any], east_m: float, north_m: float) -> dict[str, Any]:
    """Translate WGS84 GeoJSON coordinates by a local metric offset."""
    from pyproj import CRS, Transformer

    features = payload.get("features") if isinstance(payload, dict) else None
    if payload.get("type") != "FeatureCollection" or not isinstance(features, list):
        raise HTTPException(status_code=400, detail="The selected layer is not a GeoJSON FeatureCollection.")
    positions: list[tuple[float, float]] = []

    def collect(value: Any) -> None:
        if not isinstance(value, list):
            return
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            positions.append((float(value[0]), float(value[1])))
            return
        for child in value:
            collect(child)

    for feature in features:
        collect((feature.get("geometry") or {}).get("coordinates") if isinstance(feature, dict) else None)
    if not positions:
        raise HTTPException(status_code=400, detail="The selected layer has no coordinates to move.")
    center_lon = (min(point[0] for point in positions) + max(point[0] for point in positions)) / 2.0
    center_lat = (min(point[1] for point in positions) + max(point[1] for point in positions)) / 2.0
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.12f} +lon_0={center_lon:.12f} +datum=WGS84 +units=m +no_defs"
    )
    forward = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
    reverse = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)

    def shifted(value: Any) -> Any:
        if not isinstance(value, list):
            return value
        if len(value) >= 2 and all(isinstance(item, (int, float)) for item in value[:2]):
            x, y = forward.transform(float(value[0]), float(value[1]))
            lon, lat = reverse.transform(x + east_m, y + north_m)
            return [lon, lat, *value[2:]]
        return [shifted(child) for child in value]

    translated = copy.deepcopy(payload)
    translated.pop("bbox", None)
    for feature in translated["features"]:
        if not isinstance(feature, dict):
            continue
        feature.pop("bbox", None)
        geometry = feature.get("geometry") or {}
        if "coordinates" in geometry:
            geometry["coordinates"] = shifted(geometry["coordinates"])
    return translated


def archive_postprocess_outputs(
    job_dir: Path,
    workspace: Path,
    workflow_id: str,
    stages: set[str],
    revision_dir: Path,
    write_json: Callable[[Path, dict[str, Any]], None],
) -> list[dict[str, str]]:
    """Move invalidated derived files into a recoverable job revision."""
    if not workflow_id or not stages:
        return []
    workflow_dir = (workspace / "postprocess" / _safe_name(workflow_id)).resolve()
    if workflow_dir.parent != (workspace / "postprocess").resolve():
        return []
    status_path = workflow_dir / "status.json"
    if not status_path.is_file():
        return []
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    outputs = dict(status.get("outputs") or {})
    archived: list[dict[str, str]] = []
    for stage in sorted(stages):
        output = outputs.pop(stage, None)
        relative = str((output or {}).get("path") or "")
        if not relative:
            continue
        source = (workspace / relative).resolve()
        try:
            source.relative_to(workspace.resolve())
        except ValueError:
            continue
        if not source.is_file():
            continue
        destination = revision_dir / workspace.name / workflow_dir.name / stage / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        archived.append({
            "stage": stage,
            "original_path": source.relative_to(job_dir).as_posix(),
            "archived_path": destination.relative_to(job_dir).as_posix(),
        })
    if "deduplicated" in stages:
        review_path = workflow_dir / "visual_review.json"
        if review_path.is_file():
            destination = revision_dir / workspace.name / workflow_dir.name / "visual_review.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(review_path), str(destination))
            archived.append({
                "stage": "visual_review",
                "original_path": review_path.relative_to(job_dir).as_posix(),
                "archived_path": destination.relative_to(job_dir).as_posix(),
            })
        for key in ("visual_review_path", "visual_review_preview", "visual_review_available"):
            status.pop(key, None)
    status.update({
        "outputs": outputs,
        "status": "complete",
        "stage": "layer_move",
        "progress": 100,
        "message": "Dependent outputs were archived after a whole-layer movement.",
        "updated_at": datetime.now().isoformat(),
    })
    write_json(status_path, status)
    meta_path = workflow_dir / "postprocess_meta.json"
    if meta_path.is_file():
        write_json(meta_path, status)
    return archived


def raster_shift_in_mercator(path: Path, east_m: float, north_m: float) -> tuple[float, float]:
    """Convert a local ground-metre offset at a raster centre to EPSG:3857."""
    if not east_m and not north_m:
        return 0.0, 0.0
    import rasterio
    from pyproj import CRS, Transformer

    with rasterio.open(path) as dataset:
        left, bottom, right, top = rasterio.warp.transform_bounds(
            dataset.crs, CRS.from_epsg(4326), *dataset.bounds, densify_pts=21,
        )
    center_lon = (left + right) / 2.0
    center_lat = (bottom + top) / 2.0
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.12f} +lon_0={center_lon:.12f} +datum=WGS84 +units=m +no_defs"
    )
    local_to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    wgs84_to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    shifted_lon, shifted_lat = local_to_wgs84.transform(east_m, north_m)
    start_x, start_y = wgs84_to_mercator.transform(center_lon, center_lat)
    shifted_x, shifted_y = wgs84_to_mercator.transform(shifted_lon, shifted_lat)
    return shifted_x - start_x, shifted_y - start_y


def _raster_crs_offset(path: Path, east_m: float, north_m: float) -> tuple[float, float]:
    """Convert local ground metres at the raster centre into its native CRS."""
    import rasterio
    from pyproj import CRS, Transformer

    with rasterio.open(path) as dataset:
        raster_crs = dataset.crs
        left, bottom, right, top = rasterio.warp.transform_bounds(
            raster_crs, CRS.from_epsg(4326), *dataset.bounds, densify_pts=21,
        )
    center_lon = (left + right) / 2.0
    center_lat = (bottom + top) / 2.0
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat:.12f} +lon_0={center_lon:.12f} +datum=WGS84 +units=m +no_defs"
    )
    local_to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
    wgs84_to_raster = Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    shifted_lon, shifted_lat = local_to_wgs84.transform(east_m, north_m)
    start_x, start_y = wgs84_to_raster.transform(center_lon, center_lat)
    shifted_x, shifted_y = wgs84_to_raster.transform(shifted_lon, shifted_lat)
    return shifted_x - start_x, shifted_y - start_y


def create_or_update_raster_working_copy(
    job_dir: Path,
    kind: str,
    source_paths: list[Path],
    east_m: float,
    north_m: float,
) -> list[Path]:
    """Atomically create/update the job's sole portable raster working copy."""
    import rasterio

    if not source_paths:
        raise HTTPException(status_code=404, detail="No orthophoto or mosaic GeoTIFF was found for this source.")
    working_dir = job_dir / "rasters" / kind
    working_dir.mkdir(parents=True, exist_ok=True)
    destinations: list[Path] = []
    temporary_files: list[Path] = []
    used_names: set[str] = set()
    try:
        for index, source in enumerate(source_paths):
            if not source.is_file() or source.suffix.lower() not in {".tif", ".tiff"}:
                raise HTTPException(status_code=404, detail="A configured raster source is no longer available.")
            name = source.name
            if name.casefold() in used_names:
                name = f"{source.stem}_{index + 1}{source.suffix}"
            used_names.add(name.casefold())
            destination = working_dir / name
            temporary = working_dir / f".{name}.{uuid.uuid4().hex}.tmp{source.suffix}"
            shutil.copy2(source, temporary)
            delta_x, delta_y = _raster_crs_offset(temporary, east_m, north_m)
            with rasterio.open(temporary, "r+") as dataset:
                transform = dataset.transform
                dataset.transform = type(transform)(
                    transform.a, transform.b, transform.c + delta_x,
                    transform.d, transform.e, transform.f + delta_y,
                )
            destinations.append(destination)
            temporary_files.append(temporary)
        for temporary, destination in zip(temporary_files, destinations):
            temporary.replace(destination)
        keep = {path.resolve() for path in destinations}
        for stale in working_dir.iterdir():
            if stale.is_file() and stale.suffix.lower() in {".tif", ".tiff"} and stale.resolve() not in keep:
                stale.unlink()
        return destinations
    finally:
        for temporary in temporary_files:
            if temporary.exists():
                temporary.unlink()


def create_postprocess_layer_move_router(
    read_job: Callable[[str], tuple[Path, dict[str, Any]]],
    resolve_workspace: Callable[[str], Path],
    write_json: Callable[[Path, dict[str, Any]], None],
    resolve_rasters: Callable[[str], list[Path]],
) -> APIRouter:
    router = APIRouter()

    @router.post("/api/postprocess-jobs/{job_id}/move-layer")
    async def move_postprocess_job_layer(job_id: str, request: Request):
        directory, metadata = read_job(job_id)
        payload = await request.json()
        if payload.get("confirm_move") is not True:
            raise HTTPException(status_code=400, detail="Confirm the whole-layer movement before saving.")
        kind = str(payload.get("kind") or "").strip()
        layer_type = str(payload.get("layer_type") or "").strip()
        stage = str(payload.get("stage") or "").strip()
        if kind not in {"segmentation", "anomaly"}:
            raise HTTPException(status_code=400, detail="Layer source must be segmentation or anomaly.")
        if layer_type not in {"geojson", "raster"}:
            raise HTTPException(status_code=400, detail="Only GeoJSON and orthophoto/mosaic layers can be moved.")
        try:
            east_m = float(payload.get("east_m") or 0.0)
            north_m = float(payload.get("north_m") or 0.0)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail="Layer movement must be expressed in metres.") from exc
        if not all(math.isfinite(value) and abs(value) <= 10_000 for value in (east_m, north_m)):
            raise HTTPException(status_code=400, detail="Layer movement must be finite and within 10 km.")
        if math.hypot(east_m, north_m) < 0.001:
            raise HTTPException(status_code=400, detail="Move the layer before saving.")

        source = (metadata.get("sources") or {}).get(kind) or {}
        workspace = resolve_workspace(str(source.get("workspace_result_id") or ""))
        now = datetime.now().isoformat()
        revision_id = f"move_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        revision_dir = directory / "outdated" / revision_id
        archived: list[dict[str, str]] = []

        if layer_type == "raster":
            if stage != "orthophoto":
                raise HTTPException(status_code=400, detail="Individual images and non-raster references cannot be moved.")
            raster_copies = dict(metadata.get("raster_copies") or {})
            previous_copy = raster_copies.get(kind) or {}
            existing_paths: list[Path] = []
            for relative in previous_copy.get("paths") or []:
                candidate = (directory / str(relative)).resolve()
                try:
                    candidate.relative_to(directory.resolve())
                except ValueError:
                    continue
                if candidate.is_file():
                    existing_paths.append(candidate)
            source_paths = existing_paths or resolve_rasters(str(source.get("result_id") or ""))
            legacy_shift = ((metadata.get("raster_shifts") or {}).get(kind) or {}) if not existing_paths else {}
            applied_east = east_m + float(legacy_shift.get("east_m") or 0.0)
            applied_north = north_m + float(legacy_shift.get("north_m") or 0.0)
            destinations = await asyncio.to_thread(
                create_or_update_raster_working_copy,
                directory,
                kind,
                source_paths,
                applied_east,
                applied_north,
            )
            previous_east = float(previous_copy.get("east_m") or legacy_shift.get("east_m") or 0.0)
            previous_north = float(previous_copy.get("north_m") or legacy_shift.get("north_m") or 0.0)
            raster_copies[kind] = {
                "paths": [path.relative_to(directory).as_posix() for path in destinations],
                "east_m": previous_east + east_m,
                "north_m": previous_north + north_m,
                "updated_at": now,
                "revision": revision_id,
            }
            metadata["raster_copies"] = raster_copies
            metadata.get("raster_shifts", {}).pop(kind, None)
        else:
            if stage not in MOVE_DEPENDENCIES:
                raise HTTPException(status_code=400, detail="This GeoJSON layer cannot be moved.")
            workflow_id = str(((metadata.get("workflows") or {}).get(kind) or {}).get("workflow_id") or "")
            workflow_dir = workspace / "postprocess" / _safe_name(workflow_id)
            if stage == "source":
                output_path = workspace / "source.geojson"
            else:
                if not workflow_id:
                    raise HTTPException(status_code=404, detail="The selected processing workflow was not found.")
                try:
                    status = json.loads((workflow_dir / "status.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise HTTPException(status_code=404, detail="The selected processing workflow was not found.") from exc
                output_path = (workspace / str(((status.get("outputs") or {}).get(stage) or {}).get("path") or "")).resolve()
                try:
                    output_path.relative_to(workflow_dir.resolve())
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="Invalid processing layer path.") from exc
            if not output_path.is_file():
                raise HTTPException(status_code=404, detail="The selected GeoJSON layer was not found.")
            try:
                geojson = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=409, detail="The selected GeoJSON layer could not be read.") from exc
            translated = translate_geojson_payload(geojson, east_m, north_m)
            history = list(translated.get("postprocess_layer_movements") or [])
            history.append({"east_m": east_m, "north_m": north_m, "updated_at": now})
            translated["postprocess_layer_movements"] = history[-20:]
            write_json(output_path, translated)

            dependent_stages = set(MOVE_DEPENDENCIES[stage])
            dependent_stages &= ({"combined", "regularized", "solar_rows"} if kind == "segmentation" else {"overlap_deduplicated", "deduplicated", "associated"})
            archived.extend(archive_postprocess_outputs(
                directory, workspace, workflow_id, dependent_stages, revision_dir, write_json,
            ))
            if kind == "segmentation" and stage in {"source", "combined", "regularized"}:
                anomaly_source = (metadata.get("sources") or {}).get("anomaly") or {}
                anomaly_workflow_id = str(((metadata.get("workflows") or {}).get("anomaly") or {}).get("workflow_id") or "")
                anomaly_workspace_id = str(anomaly_source.get("workspace_result_id") or "")
                if anomaly_workspace_id and anomaly_workflow_id:
                    archived.extend(archive_postprocess_outputs(
                        directory,
                        resolve_workspace(anomaly_workspace_id),
                        anomaly_workflow_id,
                        {"associated"},
                        revision_dir,
                        write_json,
                    ))
            if stage == "source":
                stat = output_path.stat()
                source["workspace_mtime"] = stat.st_mtime_ns
                source["fingerprint"] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                metadata["sources"][kind] = source

        if archived:
            write_json(revision_dir / "revision.json", {
                "id": revision_id,
                "created_at": now,
                "reason": "whole_layer_move",
                "moved_layer": {"kind": kind, "type": layer_type, "stage": stage},
                "movement": {"east_m": east_m, "north_m": north_m},
                "files": archived,
            })
            revisions = list(metadata.get("outdated_revisions") or [])
            revisions.append({
                "id": revision_id,
                "created_at": now,
                "reason": "whole_layer_move",
                "file_count": len(archived),
            })
            metadata["outdated_revisions"] = revisions[-50:]
        metadata["updated_at"] = now
        write_json(directory / "job.json", metadata)
        return {
            "ok": True,
            "job": {**metadata, "id": directory.name},
            "movement": {"east_m": east_m, "north_m": north_m},
            "archived": archived,
        }

    return router
