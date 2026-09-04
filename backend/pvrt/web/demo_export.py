from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any


EXPORT_FILENAME = "solar_demo_export.zip"


def delete_solar_demo_export(job_dir: Path) -> dict[str, Any]:
    export_path = Path(job_dir).resolve() / EXPORT_FILENAME
    if not export_path.is_file():
        raise FileNotFoundError(export_path)
    size = export_path.stat().st_size
    export_path.unlink()
    return {"path": str(export_path), "deleted_size": size}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object.")
    return payload


def _workspace_output(workspace: Path, status: dict[str, Any], stage: str) -> Path:
    raw_path = str(((status.get("outputs") or {}).get(stage) or {}).get("path") or "")
    if not raw_path:
        raise ValueError(f"The completed {stage.replace('_', ' ')} output is unavailable.")
    output = (workspace / raw_path).resolve()
    try:
        output.relative_to(workspace.resolve())
    except ValueError as exc:
        raise ValueError(f"The {stage.replace('_', ' ')} output path is invalid.") from exc
    if not output.is_file():
        raise ValueError(f"The {stage.replace('_', ' ')} output file was not found.")
    return output


def _bound_workflow(job_dir: Path, job: dict[str, Any], kind: str) -> tuple[Path, dict[str, Any]]:
    binding = (job.get("workflows") or {}).get(kind) or {}
    workflow_id = str(binding.get("workflow_id") or "")
    if not workflow_id:
        raise ValueError(f"Complete the {kind} post-processing workflow before exporting.")
    workspace = (job_dir / "snapshots" / kind).resolve()
    workflow_dir = (workspace / "postprocess" / workflow_id).resolve()
    try:
        workflow_dir.relative_to((workspace / "postprocess").resolve())
    except ValueError as exc:
        raise ValueError(f"The bound {kind} workflow path is invalid.") from exc
    status = _read_json(workflow_dir / "status.json")
    if status.get("status") != "complete":
        raise ValueError(f"The {kind} post-processing workflow is not complete.")
    return workspace, status


def _validate_image_coverage(anomalies_path: Path, images_path: Path, image_dir: Path) -> tuple[int, int]:
    anomalies = _read_json(anomalies_path).get("features")
    images = _read_json(images_path).get("features")
    if not isinstance(anomalies, list) or not isinstance(images, list):
        raise ValueError("Anomalies and images must be GeoJSON FeatureCollections.")
    image_stems = {
        Path(str((feature.get("properties") or {}).get("image") or "")).stem
        for feature in images
        if isinstance(feature, dict)
    }
    png_stems = {path.stem for path in image_dir.glob("*.png") if path.is_file()}
    referenced = {
        Path(str((feature.get("properties") or {}).get("image") or "")).stem
        for feature in anomalies
        if isinstance(feature, dict) and (feature.get("properties") or {}).get("image")
    }
    missing_metadata = sorted(referenced - image_stems)
    missing_images = sorted(referenced - png_stems)
    if missing_metadata:
        raise ValueError(f"images.geojson is missing {len(missing_metadata)} referenced image records.")
    if missing_images:
        raise ValueError(f"The image folder is missing {len(missing_images)} referenced PNG files.")
    return len(anomalies), len(png_stems)


def create_solar_demo_export(job_dir: Path, sessions_dir: Path, *, replace: bool = False) -> dict[str, Any]:
    job_dir = Path(job_dir).resolve()
    sessions_dir = Path(sessions_dir).resolve()
    job = _read_json(job_dir / "job.json")
    segmentation_workspace, segmentation_status = _bound_workflow(job_dir, job, "segmentation")
    anomaly_workspace, anomaly_status = _bound_workflow(job_dir, job, "anomaly")

    panels_path = _workspace_output(segmentation_workspace, segmentation_status, "regularized")
    rows_path = _workspace_output(segmentation_workspace, segmentation_status, "solar_rows")
    anomalies_path = _workspace_output(anomaly_workspace, anomaly_status, "associated")

    anomaly_result_id = str(((job.get("sources") or {}).get("anomaly") or {}).get("result_id") or "")
    anomaly_result = (sessions_dir / anomaly_result_id).resolve()
    if not anomaly_result_id or anomaly_result.parent != sessions_dir or not anomaly_result.is_dir():
        raise ValueError("The bound anomaly test result is unavailable.")
    images_path = anomaly_result / "images.geojson"
    image_dir = anomaly_result / "rotated_images"
    if not images_path.is_file() or not image_dir.is_dir():
        raise ValueError("The bound anomaly result has no images.geojson or rotated_images folder.")

    anomaly_count, image_count = _validate_image_coverage(
        anomalies_path,
        images_path,
        image_dir,
    )
    export_path = job_dir / EXPORT_FILENAME
    if export_path.exists() and not replace:
        raise FileExistsError(export_path)
    temporary = job_dir / f".{EXPORT_FILENAME}.tmp"
    if temporary.exists():
        temporary.unlink()
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            archive.write(panels_path, "vector/solar_panels.geojson", compress_type=zipfile.ZIP_DEFLATED)
            archive.write(rows_path, "vector/solar_rows.geojson", compress_type=zipfile.ZIP_DEFLATED)
            archive.write(anomalies_path, "vector/anomalies.geojson", compress_type=zipfile.ZIP_DEFLATED)
            archive.write(images_path, "anomaly_overlays/images.geojson", compress_type=zipfile.ZIP_DEFLATED)
            for image_path in sorted(image_dir.glob("*.png")):
                if image_path.is_file():
                    archive.write(image_path, f"anomaly_overlays/{image_path.name}", compress_type=zipfile.ZIP_STORED)
        temporary.replace(export_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    stat = export_path.stat()
    return {
        "path": str(export_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "anomaly_count": anomaly_count,
        "image_count": image_count,
    }
