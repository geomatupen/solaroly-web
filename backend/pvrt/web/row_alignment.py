"""Rows-alignment source discovery and safe post-process output resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter


def _safe_job_dir(sessions_dir: Path, job_id: str) -> Path:
    jobs_root = (Path(sessions_dir).resolve() / ".postprocess_jobs").resolve()
    job_dir = (jobs_root / str(job_id or "")).resolve()
    if not job_id or job_dir.parent != jobs_root or not job_dir.is_dir():
        raise ValueError("The selected post-processing job was not found.")
    return job_dir


def resolve_rows_source(sessions_dir: Path, job_id: str, relative_path: str) -> Path:
    job_dir = _safe_job_dir(sessions_dir, job_id)
    snapshot = (job_dir / "snapshots" / "segmentation").resolve()
    candidate = (snapshot / str(relative_path or "").replace("\\", "/")).resolve()
    try:
        candidate.relative_to(snapshot)
    except ValueError as exc:
        raise ValueError("Rows GeoJSON must belong to the selected post-processing job.") from exc
    if candidate.suffix.lower() not in {".geojson", ".json"} or not candidate.is_file():
        raise ValueError("The selected Rows GeoJSON was not found.")
    return candidate


def list_rows_sources(sessions_dir: Path) -> list[dict[str, Any]]:
    jobs_root = Path(sessions_dir).resolve() / ".postprocess_jobs"
    if not jobs_root.is_dir():
        return []
    sources: list[dict[str, Any]] = []
    for job_dir in sorted(jobs_root.iterdir()):
        if not job_dir.is_dir():
            continue
        try:
            metadata = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segmentation = (metadata.get("workflows") or {}).get("segmentation") or {}
        workflow_id = str(segmentation.get("workflow_id") or "")
        snapshot = job_dir / "snapshots" / "segmentation"
        workflow_dir = snapshot / "postprocess" / workflow_id
        try:
            status = json.loads((workflow_dir / "status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        output = (status.get("outputs") or {}).get("solar_rows") or {}
        relative_path = str(output.get("path") or "")
        if not relative_path:
            continue
        try:
            path = resolve_rows_source(Path(sessions_dir), job_dir.name, relative_path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            feature_count = len(payload.get("features") or [])
        except (ValueError, OSError, json.JSONDecodeError):
            continue
        sources.append({
            "job_id": job_dir.name,
            "job_name": str(metadata.get("name") or job_dir.name),
            "workflow_id": workflow_id,
            "path": relative_path,
            "name": path.name,
            "feature_count": feature_count,
            "updated_at": metadata.get("updated_at"),
        })
    sources.sort(key=lambda item: (str(item.get("job_name") or "").casefold(), str(item.get("name") or "").casefold()))
    return sources


def count_postprocess_jobs(sessions_dir: Path) -> int:
    jobs_root = Path(sessions_dir).resolve() / ".postprocess_jobs"
    if not jobs_root.is_dir():
        return 0
    count = 0
    for job_dir in jobs_root.iterdir():
        if not job_dir.is_dir():
            continue
        try:
            payload = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            count += 1
    return count


def create_row_alignment_router(get_sessions_dir: Callable[[], Path]) -> APIRouter:
    router = APIRouter(prefix="/api/row-alignment", tags=["row-alignment"])

    @router.get("/sources")
    async def sources() -> dict[str, Any]:
        sessions_dir = Path(get_sessions_dir())
        return {
            "ok": True,
            "sources": list_rows_sources(sessions_dir),
            "postprocess_job_count": count_postprocess_jobs(sessions_dir),
        }

    return router
