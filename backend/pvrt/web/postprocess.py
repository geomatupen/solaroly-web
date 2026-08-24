from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..postprocess import analyze_geojson, combine_tile_fragments, regularize_polygons


_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="geojson-postprocess")
_STATUS_LOCK = threading.Lock()
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class AnalyzeRequest(BaseModel):
    input_path: str
    edge_tolerance_px: float = Field(default=7.0, ge=0.0, le=50.0)


class CombineRequest(BaseModel):
    input_path: str
    output_name: str = "panel_postprocess"
    edge_tolerance_px: float = Field(default=7.0, ge=0.0, le=50.0)
    gap_tolerance_px: float = Field(default=10.0, ge=0.0, le=100.0)
    min_boundary_overlap: float = Field(default=0.20, ge=0.0, le=1.0)
    max_dimension_factor: float = Field(default=1.65, ge=1.0, le=5.0)
    max_area_factor: float = Field(default=1.75, ge=1.0, le=10.0)
    remove_contained_polygons: bool = True


class RegularizeRequest(BaseModel):
    max_area_change_percent: float = Field(default=35.0, ge=0.0, le=1000.0)


class RenameWorkflowRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


def _safe_name(value: str, fallback: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", str(value or "").strip()).strip("._-")
    return (safe or fallback)[:80]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def create_postprocess_router(
    get_sessions_dir: Callable[[], Path],
    media_url: Callable[[Path], str],
    logger: logging.Logger | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/results", tags=["postprocess"])
    log = logger or logging.getLogger("pvrt.postprocess")

    def resolve_result(result_id: str) -> Path:
        safe_id = _safe_name(result_id, "")
        root = Path(get_sessions_dir()).resolve()
        result = (root / safe_id).resolve()
        if not safe_id or result.parent != root or not result.is_dir():
            raise HTTPException(status_code=404, detail="Result not found.")
        return result

    def resolve_input(result_dir: Path, relative_path: str) -> Path:
        raw = str(relative_path or "").strip().replace("\\", "/")
        candidate = (result_dir / raw).resolve()
        try:
            candidate.relative_to(result_dir.resolve())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="GeoJSON must belong to the selected result.") from exc
        if candidate.suffix.lower() != ".geojson" or not candidate.is_file():
            raise HTTPException(status_code=404, detail="Selected GeoJSON was not found.")
        return candidate

    def resolve_workflow(result_dir: Path, workflow_id: str) -> Path:
        safe_workflow = _safe_name(workflow_id, "")
        workflow_dir = (result_dir / "postprocess" / safe_workflow).resolve()
        if (
            not safe_workflow
            or workflow_dir.parent != (result_dir / "postprocess").resolve()
            or not workflow_dir.is_dir()
        ):
            raise HTTPException(status_code=404, detail="Post-processing workflow not found.")
        return workflow_dir

    def read_status(workflow_dir: Path) -> dict[str, Any]:
        status_path = workflow_dir / "status.json"
        if not status_path.is_file():
            raise HTTPException(status_code=404, detail="Post-processing workflow not found.")
        payload: dict[str, Any] | None = None
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                # Status is updated frequently by a worker thread. Serializing
                # access avoids transient read/replace failures on WSL-mounted drives.
                with _STATUS_LOCK:
                    payload = json.loads(status_path.read_text(encoding="utf-8"))
                break
            except (OSError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(0.05 * (attempt + 1))
        if payload is None:
            log.warning("Could not read post-processing status %s: %s", status_path, last_error)
            raise HTTPException(
                status_code=503,
                detail="Workflow status is temporarily unavailable; retrying is safe.",
            ) from last_error
        outputs = payload.get("outputs") or {}
        for output in outputs.values():
            if isinstance(output, dict) and output.get("path"):
                output_path = (workflow_dir.parent.parent / output["path"]).resolve()
                if output_path.is_file():
                    output["url"] = media_url(output_path)
        log_path = workflow_dir / "postprocess.log"
        if log_path.is_file():
            try:
                payload["log"] = log_path.read_text(encoding="utf-8").splitlines()[-200:]
            except OSError:
                payload["log"] = []
        return payload

    def update_status(workflow_dir: Path, **updates: Any) -> dict[str, Any]:
        with _STATUS_LOCK:
            status_path = workflow_dir / "status.json"
            current: dict[str, Any] = {}
            if status_path.is_file():
                try:
                    current = json.loads(status_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    current = {}
            current.update(updates)
            current["updated_at"] = datetime.now().isoformat()
            _atomic_json(status_path, current)
            _atomic_json(workflow_dir / "postprocess_meta.json", current)
            return current

    def append_log(workflow_dir: Path, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        with _STATUS_LOCK:
            with (workflow_dir / "postprocess.log").open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] {message}\n")

    def progress_callback(workflow_dir: Path, stage: str) -> Callable[[int, str], None]:
        def callback(progress: int, message: str) -> None:
            append_log(workflow_dir, message)
            update_status(
                workflow_dir,
                status="running",
                stage=stage,
                progress=progress,
                message=message,
            )
            log.info("UI:INFO:postprocess: %s (%s%%)", message, progress)

        return callback

    @router.get("/{result_id}/postprocess/geojsons")
    async def list_geojsons(result_id: str) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        files = list(result_dir.glob("*.geojson"))
        files.extend((result_dir / "postprocess").glob("*/*.geojson"))
        items = []
        for path in sorted(set(files), key=lambda item: (item.stat().st_mtime, item.name), reverse=True):
            relative = path.relative_to(result_dir).as_posix()
            stage = "source"
            if path.name == "combined.geojson":
                stage = "combined"
            elif path.name == "regularized.geojson":
                stage = "regularized"
            items.append(
                {
                    "name": path.name,
                    "path": relative,
                    "stage": stage,
                    "size": path.stat().st_size,
                    "mtime": int(path.stat().st_mtime),
                    "url": media_url(path),
                }
            )
        return {"ok": True, "files": items}

    @router.post("/{result_id}/postprocess/analyze")
    async def analyze(result_id: str, request: AnalyzeRequest) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        input_path = resolve_input(result_dir, request.input_path)
        try:
            summary = await asyncio.to_thread(
                analyze_geojson,
                input_path,
                result_dir,
                edge_tolerance_px=request.edge_tolerance_px,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "input_path": request.input_path, "summary": summary}

    @router.post("/{result_id}/postprocess/combine")
    async def combine(result_id: str, request: CombineRequest) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        input_path = resolve_input(result_dir, request.input_path)
        prefix = _safe_name(request.output_name, "panel_postprocess")
        workflow_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        workflow_dir = result_dir / "postprocess" / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=False)
        input_relative = input_path.relative_to(result_dir).as_posix()
        initial = {
            "ok": True,
            "id": workflow_id,
            "display_name": request.output_name.strip() or prefix,
            "result_id": result_dir.name,
            "status": "queued",
            "stage": "combine",
            "progress": 0,
            "message": "Queued fragment combining.",
            "created_at": datetime.now().isoformat(),
            "input_path": input_relative,
            "parameters": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "outputs": {},
        }
        _atomic_json(workflow_dir / "status.json", initial)
        _atomic_json(workflow_dir / "postprocess_meta.json", initial)

        def run() -> None:
            output_path = workflow_dir / "combined.geojson"
            try:
                stats = combine_tile_fragments(
                    input_path,
                    output_path,
                    result_dir,
                    edge_tolerance_px=request.edge_tolerance_px,
                    gap_tolerance_px=request.gap_tolerance_px,
                    min_boundary_overlap=request.min_boundary_overlap,
                    max_dimension_factor=request.max_dimension_factor,
                    max_area_factor=request.max_area_factor,
                    remove_contained_polygons=request.remove_contained_polygons,
                    callback=progress_callback(workflow_dir, "combine"),
                )
                relative = output_path.relative_to(result_dir).as_posix()
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="combine",
                    progress=100,
                    message="Combined GeoJSON is ready.",
                    combine_stats=stats,
                    outputs={"combined": {"path": relative}},
                )
                log.info("UI:OK:postprocess: Combined GeoJSON ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(
                    workflow_dir,
                    status="failed",
                    stage="combine",
                    message=str(exc),
                    error=str(exc),
                )
                log.exception("Post-processing combine failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return initial

    @router.post("/{result_id}/postprocess/{workflow_id}/regularize")
    async def regularize(
        result_id: str, workflow_id: str, request: RegularizeRequest
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        status = read_status(workflow_dir)
        if status.get("status") == "running":
            raise HTTPException(status_code=409, detail="This workflow is already running.")
        combined_path = workflow_dir / "combined.geojson"
        if not combined_path.is_file():
            raise HTTPException(status_code=409, detail="Combine fragments before regularizing.")
        update_status(
            workflow_dir,
            status="queued",
            stage="regularize",
            progress=0,
            message="Queued polygon regularization.",
            regularize_parameters=(
                request.model_dump() if hasattr(request, "model_dump") else request.dict()
            ),
        )

        def run() -> None:
            output_path = workflow_dir / "regularized.geojson"
            try:
                stats = regularize_polygons(
                    combined_path,
                    output_path,
                    max_area_change_percent=request.max_area_change_percent,
                    callback=progress_callback(workflow_dir, "regularize"),
                )
                relative = output_path.relative_to(result_dir).as_posix()
                latest = read_status(workflow_dir)
                outputs = dict(latest.get("outputs") or {})
                outputs["regularized"] = {"path": relative}
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="regularize",
                    progress=100,
                    message="Regularized GeoJSON is ready.",
                    regularize_stats=stats,
                    outputs=outputs,
                )
                log.info("UI:OK:postprocess: Regularized GeoJSON ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(
                    workflow_dir,
                    status="failed",
                    stage="regularize",
                    message=str(exc),
                    error=str(exc),
                )
                log.exception("Post-processing regularization failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return {"ok": True, "id": workflow_id, "status": "queued", "stage": "regularize"}

    @router.get("/{result_id}/postprocess/{workflow_id}")
    async def workflow_status(result_id: str, workflow_id: str) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        return read_status(workflow_dir)

    @router.patch("/{result_id}/postprocess/{workflow_id}")
    async def rename_workflow(
        result_id: str, workflow_id: str, request: RenameWorkflowRequest
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        display_name = request.name.strip()
        if not display_name:
            raise HTTPException(status_code=400, detail="Output name cannot be empty.")
        status = update_status(workflow_dir, display_name=display_name)
        return {"ok": True, "workflow": read_status(workflow_dir), "status": status.get("status")}

    @router.delete("/{result_id}/postprocess/{workflow_id}")
    async def delete_workflow(result_id: str, workflow_id: str) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        status = read_status(workflow_dir)
        if status.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="A running workflow cannot be deleted.")
        shutil.rmtree(workflow_dir)
        log.info("UI:OK:postprocess: Deleted workflow %s for %s", workflow_id, result_id)
        return {"ok": True, "deleted": workflow_id}

    @router.get("/{result_id}/postprocess")
    async def list_workflows(result_id: str) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        root = result_dir / "postprocess"
        workflows = []
        if root.is_dir():
            for workflow_dir in sorted(
                (path for path in root.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            ):
                try:
                    workflows.append(read_status(workflow_dir))
                except HTTPException:
                    continue
        return {"ok": True, "workflows": workflows}

    return router
