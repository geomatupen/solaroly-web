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

from ..postprocess import (
    analyze_geojson,
    associate_anomalies,
    build_panel_hierarchy,
    combine_tile_fragments,
    deduplicate_anomalies,
    regularize_polygons,
)


_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="geojson-postprocess")
_STATUS_LOCK = threading.Lock()
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class AnalyzeRequest(BaseModel):
    input_path: str
    edge_tolerance_px: float = Field(default=7.0, ge=0.0, le=50.0)


class CombineRequest(BaseModel):
    input_path: str
    workflow_id: str | None = None
    output_name: str = "panel_postprocess"
    edge_tolerance_px: float = Field(default=7.0, ge=0.0, le=50.0)
    gap_tolerance_px: float = Field(default=10.0, ge=0.0, le=100.0)
    min_boundary_overlap: float = Field(default=0.20, ge=0.0, le=1.0)
    max_dimension_factor: float = Field(default=1.65, ge=1.0, le=5.0)
    max_area_factor: float = Field(default=1.75, ge=1.0, le=10.0)
    remove_contained_polygons: bool = True


class RegularizeRequest(BaseModel):
    max_area_change_percent: float = Field(default=35.0, ge=0.0, le=1000.0)


class HierarchyRequest(BaseModel):
    input_path: str
    max_orientation_difference_deg: float = Field(default=12.0, ge=0.0, le=45.0)
    max_lateral_distance_factor: float = Field(default=1.5, ge=0.25, le=10.0)
    max_along_gap_factor: float = Field(default=2.5, ge=0.25, le=20.0)
    max_inner_row_gap_factor: float = Field(default=1.0, ge=0.0, le=10.0)


class DeduplicateAnomaliesRequest(BaseModel):
    input_path: str
    output_name: str = "anomaly_postprocess"
    minimum_iou: float = Field(default=0.35, ge=0.0, le=1.0)
    maximum_center_distance_m: float = Field(default=0.35, ge=0.0, le=100.0)
    minimum_smaller_overlap: float = Field(default=0.55, ge=0.0, le=1.0)


class AssociateAnomaliesRequest(BaseModel):
    panel_path: str
    panel_result_id: str | None = None
    minimum_overlap: float = Field(default=0.20, ge=0.0, le=1.0)
    maximum_distance_m: float = Field(default=0.50, ge=0.0, le=100.0)


class RenameWorkflowRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ShareLayerRequest(BaseModel):
    name: str | None = Field(default=None, max_length=80)


class EditLayerRequest(BaseModel):
    geojson: dict[str, Any]
    name: str | None = Field(default=None, max_length=80)


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
    get_overlays_dir: Callable[[], Path],
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

    def remove_stage_outputs(
        result_dir: Path,
        workflow_dir: Path,
        status: dict[str, Any],
        stages: set[str],
    ) -> dict[str, Any]:
        """Remove replaceable base outputs while leaving edited snapshots intact."""
        outputs = dict(status.get("outputs") or {})
        for stage in stages:
            output = outputs.pop(stage, None) or {}
            raw_path = str(output.get("path") or "")
            candidate = (result_dir / raw_path).resolve() if raw_path else workflow_dir / f"{stage}.geojson"
            try:
                candidate.relative_to(workflow_dir)
            except ValueError:
                continue
            if candidate.is_file():
                candidate.unlink()
        return outputs

    @router.get("/postprocess/panel-layers")
    async def list_identified_panel_layers() -> dict[str, Any]:
        sessions_root = Path(get_sessions_dir()).resolve()
        items: list[dict[str, Any]] = []
        if not sessions_root.is_dir():
            return {"ok": True, "layers": items}
        for result_dir in (path for path in sessions_root.iterdir() if path.is_dir()):
            workflow_root = result_dir / "postprocess"
            if not workflow_root.is_dir():
                continue
            for workflow_dir in (path for path in workflow_root.iterdir() if path.is_dir()):
                try:
                    status = read_status(workflow_dir)
                except HTTPException:
                    continue
                workflow_outputs = status.get("outputs") or {}
                candidates = [
                    (
                        "solar_panels" if workflow_outputs.get("solar_panels") else "panel_hierarchy",
                        workflow_outputs.get("solar_panels") or workflow_outputs.get("panel_hierarchy"),
                    )
                ]
                revision = (status.get("manual_revisions") or [{}])[-1]
                if revision.get("source_stage") in {"panel_hierarchy", "identified_panels"}:
                    candidates.append(("panel_hierarchy_edited", (status.get("outputs") or {}).get("edited")))
                for stage, output in candidates:
                    if not output or not output.get("path"):
                        continue
                    source = (result_dir / output["path"]).resolve()
                    try:
                        source.relative_to(result_dir)
                    except ValueError:
                        continue
                    if not source.is_file():
                        continue
                    items.append({
                        "result_id": result_dir.name,
                        "workflow_id": workflow_dir.name,
                        "workflow_name": status.get("display_name") or workflow_dir.name,
                        "stage": stage,
                        "path": source.relative_to(result_dir).as_posix(),
                        "url": media_url(source),
                        "mtime": int(source.stat().st_mtime),
                    })
        items.sort(key=lambda item: item["mtime"], reverse=True)
        return {"ok": True, "layers": items}

    @router.get("/{result_id}/postprocess/geojsons")
    async def list_geojsons(result_id: str) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        files = list(result_dir.glob("*.geojson"))
        files.extend(
            path
            for path in (result_dir / "postprocess").glob("*/*.geojson")
            if not path.name.startswith("edited_") and path.name != "edited.geojson"
        )
        items = []
        for path in sorted(set(files), key=lambda item: (item.stat().st_mtime, item.name), reverse=True):
            relative = path.relative_to(result_dir).as_posix()
            stage = "source"
            stage_names = {
                "combined.geojson": "combined",
                "regularized.geojson": "regularized",
                "panel_rows.geojson": "panel_rows",
                "identified_panels.geojson": "identified_panels",
                "panel_hierarchy.geojson": "panel_hierarchy",
                "solar_panels.geojson": "solar_panels",
                "solar_rows.geojson": "solar_rows",
                "deduplicated.geojson": "deduplicated",
                "associated.geojson": "associated",
            }
            stage = stage_names.get(path.name, stage)
            if path.name.endswith("_edited.geojson"):
                stage = "edited"
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
        input_relative = input_path.relative_to(result_dir).as_posix()
        existing: dict[str, Any] = {}
        if request.workflow_id:
            workflow_dir = resolve_workflow(result_dir, request.workflow_id)
            existing = read_status(workflow_dir)
            if existing.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="This workflow is already running.")
            if str(existing.get("input_path") or "") != input_relative:
                raise HTTPException(status_code=400, detail="The replacement workflow belongs to a different source GeoJSON.")
            workflow_id = workflow_dir.name
            outputs = remove_stage_outputs(
                result_dir,
                workflow_dir,
                existing,
                {"combined", "regularized", "panel_hierarchy", "solar_panels", "solar_rows", "panel_rows", "identified_panels"},
            )
        else:
            workflow_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            workflow_dir = result_dir / "postprocess" / workflow_id
            workflow_dir.mkdir(parents=True, exist_ok=False)
            outputs = {}
        initial = {
            "ok": True,
            "id": workflow_id,
            "display_name": request.output_name.strip() or prefix,
            "result_id": result_dir.name,
            "status": "queued",
            "stage": "combine",
            "progress": 0,
            "message": "Queued fragment combining.",
            "created_at": existing.get("created_at") or datetime.now().isoformat(),
            "input_path": input_relative,
            "parameters": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "outputs": outputs,
        }
        if existing.get("manual_revisions"):
            initial["manual_revisions"] = existing["manual_revisions"]
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
                    outputs={**outputs, "combined": {"path": relative}},
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
        if status.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="This workflow is already running.")
        combined_path = workflow_dir / "combined.geojson"
        if not combined_path.is_file():
            raise HTTPException(status_code=409, detail="Combine fragments before regularizing.")
        outputs = remove_stage_outputs(
            result_dir,
            workflow_dir,
            status,
            {"regularized", "panel_hierarchy", "solar_panels", "solar_rows", "panel_rows", "identified_panels"},
        )
        update_status(
            workflow_dir,
            status="queued",
            stage="regularize",
            progress=0,
            message="Queued polygon regularization.",
            regularize_parameters=(
                request.model_dump() if hasattr(request, "model_dump") else request.dict()
            ),
            outputs=outputs,
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
                current_outputs = dict(latest.get("outputs") or {})
                current_outputs["regularized"] = {"path": relative}
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="regularize",
                    progress=100,
                    message="Regularized GeoJSON is ready.",
                    regularize_stats=stats,
                    outputs=current_outputs,
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

    @router.post("/{result_id}/postprocess/{workflow_id}/hierarchy")
    async def hierarchy(
        result_id: str, workflow_id: str, request: HierarchyRequest
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        status = read_status(workflow_dir)
        if status.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="This workflow is already running.")
        input_path = resolve_input(result_dir, request.input_path)
        try:
            input_path.relative_to(workflow_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Select an output from this segmentation workflow.") from exc
        outputs = remove_stage_outputs(
            result_dir,
            workflow_dir,
            status,
            {"panel_hierarchy", "solar_panels", "solar_rows", "panel_rows", "identified_panels"},
        )
        update_status(
            workflow_dir,
            status="queued",
            stage="hierarchy",
            progress=0,
            message="Queued panel-row hierarchy generation.",
            hierarchy_parameters=request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            outputs=outputs,
        )

        def run() -> None:
            hierarchy_path = workflow_dir / "panel_hierarchy.geojson"
            panels_path = workflow_dir / "solar_panels.geojson"
            rows_path = workflow_dir / "solar_rows.geojson"
            try:
                stats = build_panel_hierarchy(
                    input_path,
                    hierarchy_path,
                    rows_output_path=rows_path,
                    panels_output_path=panels_path,
                    max_orientation_difference_deg=request.max_orientation_difference_deg,
                    max_lateral_distance_factor=request.max_lateral_distance_factor,
                    max_along_gap_factor=request.max_along_gap_factor,
                    max_inner_row_gap_factor=request.max_inner_row_gap_factor,
                    callback=progress_callback(workflow_dir, "hierarchy"),
                )
                latest = read_status(workflow_dir)
                current_outputs = dict(latest.get("outputs") or {})
                current_outputs["panel_hierarchy"] = {
                    "path": hierarchy_path.relative_to(result_dir).as_posix()
                }
                current_outputs["solar_panels"] = {
                    "path": panels_path.relative_to(result_dir).as_posix()
                }
                current_outputs["solar_rows"] = {
                    "path": rows_path.relative_to(result_dir).as_posix()
                }
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="hierarchy",
                    progress=100,
                    message="Panel rows and IDs are ready.",
                    hierarchy_stats=stats,
                    outputs=current_outputs,
                )
                log.info("UI:OK:postprocess: Panel hierarchy ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(workflow_dir, status="failed", stage="hierarchy", message=str(exc), error=str(exc))
                log.exception("Panel hierarchy generation failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return {"ok": True, "id": workflow_id, "status": "queued", "stage": "hierarchy"}

    @router.post("/{result_id}/postprocess/anomalies/deduplicate")
    async def deduplicate(result_id: str, request: DeduplicateAnomaliesRequest) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        input_path = resolve_input(result_dir, request.input_path)
        prefix = _safe_name(request.output_name, "anomaly_postprocess")
        workflow_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        workflow_dir = result_dir / "postprocess" / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=False)
        initial = {
            "ok": True,
            "id": workflow_id,
            "workflow_kind": "anomaly",
            "display_name": request.output_name.strip() or prefix,
            "result_id": result_dir.name,
            "status": "queued",
            "stage": "deduplicate",
            "progress": 0,
            "message": "Queued anomaly deduplication.",
            "created_at": datetime.now().isoformat(),
            "input_path": input_path.relative_to(result_dir).as_posix(),
            "parameters": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "outputs": {},
        }
        _atomic_json(workflow_dir / "status.json", initial)
        _atomic_json(workflow_dir / "postprocess_meta.json", initial)

        def run() -> None:
            output_path = workflow_dir / "deduplicated.geojson"
            try:
                stats = deduplicate_anomalies(
                    input_path,
                    output_path,
                    minimum_iou=request.minimum_iou,
                    maximum_center_distance_m=request.maximum_center_distance_m,
                    minimum_smaller_overlap=request.minimum_smaller_overlap,
                    callback=progress_callback(workflow_dir, "deduplicate"),
                )
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="deduplicate",
                    progress=100,
                    message="Deduplicated anomaly GeoJSON is ready.",
                    deduplicate_stats=stats,
                    outputs={"deduplicated": {"path": output_path.relative_to(result_dir).as_posix()}},
                )
                log.info("UI:OK:postprocess: Anomaly deduplication ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(workflow_dir, status="failed", stage="deduplicate", message=str(exc), error=str(exc))
                log.exception("Anomaly deduplication failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return initial

    @router.post("/{result_id}/postprocess/{workflow_id}/associate")
    async def associate(
        result_id: str, workflow_id: str, request: AssociateAnomaliesRequest
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        status = read_status(workflow_dir)
        if status.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="This workflow is already running.")
        anomaly_output = (status.get("outputs") or {}).get("edited") or (status.get("outputs") or {}).get("deduplicated") or {}
        anomaly_path = resolve_input(result_dir, str(anomaly_output.get("path") or ""))
        panel_result_dir = resolve_result(request.panel_result_id or result_id)
        panel_path = resolve_input(panel_result_dir, request.panel_path)
        update_status(
            workflow_dir,
            status="queued",
            stage="associate",
            progress=0,
            message="Queued anomaly-to-panel association.",
            association_parameters=request.model_dump() if hasattr(request, "model_dump") else request.dict(),
        )

        def run() -> None:
            output_path = workflow_dir / "associated.geojson"
            try:
                stats = associate_anomalies(
                    anomaly_path,
                    panel_path,
                    output_path,
                    minimum_overlap=request.minimum_overlap,
                    maximum_distance_m=request.maximum_distance_m,
                    callback=progress_callback(workflow_dir, "associate"),
                )
                latest = read_status(workflow_dir)
                outputs = dict(latest.get("outputs") or {})
                outputs["associated"] = {"path": output_path.relative_to(result_dir).as_posix()}
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="associate",
                    progress=100,
                    message="Anomalies are associated with panel and row IDs.",
                    association_stats=stats,
                    outputs=outputs,
                )
                log.info("UI:OK:postprocess: Anomaly association ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(workflow_dir, status="failed", stage="associate", message=str(exc), error=str(exc))
                log.exception("Anomaly association failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return {"ok": True, "id": workflow_id, "status": "queued", "stage": "associate"}

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
        overlays_root = Path(get_overlays_dir()).resolve()
        if overlays_root.is_dir():
            for overlay_dir in overlays_root.iterdir():
                if not overlay_dir.is_dir():
                    continue
                try:
                    metadata = json.loads((overlay_dir / ".overlay_meta.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if (
                    metadata.get("reference_kind") == "postprocess"
                    and metadata.get("source_result") == result_dir.name
                    and metadata.get("workflow_id") == workflow_dir.name
                ):
                    shutil.rmtree(overlay_dir)
        log.info("UI:OK:postprocess: Deleted workflow %s for %s", workflow_id, result_id)
        return {"ok": True, "deleted": workflow_id}

    @router.post("/{result_id}/postprocess/{workflow_id}/{stage}/share")
    async def share_workflow_layer(
        result_id: str,
        workflow_id: str,
        stage: str,
        request: ShareLayerRequest,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        allowed_stages = {"combined", "regularized", "panel_hierarchy", "solar_panels", "solar_rows", "panel_rows", "identified_panels", "deduplicated", "associated", "edited"}
        if stage not in allowed_stages:
            raise HTTPException(status_code=400, detail="Only generated post-processing outputs can be sent to Map.")
        status = read_status(workflow_dir)
        output = (status.get("outputs") or {}).get(stage) or {}
        source = (
            (result_dir / str(output.get("path") or "")).resolve()
            if output.get("path")
            else workflow_dir / f"{stage}.geojson"
        )
        try:
            source.relative_to(workflow_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid workflow output path.") from exc
        if not source.is_file():
            raise HTTPException(status_code=404, detail=f"The {stage} GeoJSON is not available.")
        base_name = request.name or status.get("display_name") or status.get("parameters", {}).get("output_name") or workflow_id
        stage_label = stage.title()
        if stage == "edited":
            edited_records = status.get("manual_revisions") or []
            edited_source = str((edited_records[-1] if edited_records else {}).get("source_stage") or "combined").lower()
            if edited_source not in allowed_stages - {"edited"}:
                edited_source = "combined"
            stage_label = f"{edited_source}_edited"
        display_name = f"{str(base_name).strip()} — {stage_label}"
        overlay_id = _safe_name(f"postprocess-{workflow_id[:54]}-{stage}", f"postprocess-{uuid.uuid4().hex[:10]}")
        overlay_dir = (Path(get_overlays_dir()).resolve() / overlay_id).resolve()
        if overlay_dir.parent != Path(get_overlays_dir()).resolve():
            raise HTTPException(status_code=400, detail="Invalid overlay destination.")
        overlay_dir.mkdir(parents=True, exist_ok=True)
        legacy_copy = overlay_dir / "layer.geojson"
        if legacy_copy.is_file():
            legacy_copy.unlink()
        _atomic_json(
            overlay_dir / ".overlay_meta.json",
            {
                "display_name": display_name,
                "reference_kind": "postprocess",
                "source_result": result_dir.name,
                "workflow_id": workflow_id,
                "stage": stage,
            },
        )
        log.info("UI:OK:postprocess: Linked %s/%s to Map overlays", workflow_id, stage)
        return {
            "ok": True,
            "overlay": {
                "type": "geojson",
                "name": display_name,
                "overlay_id": overlay_id,
                "path": media_url(source),
                "reference": True,
            },
        }

    @router.post("/{result_id}/postprocess/{workflow_id}/{stage}/revisions")
    async def save_edited_revision(
        result_id: str,
        workflow_id: str,
        stage: str,
        request: EditLayerRequest,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        editable_stages = {"combined", "regularized", "panel_hierarchy", "panel_rows", "identified_panels", "deduplicated", "associated", "edited"}
        if stage not in editable_stages:
            raise HTTPException(status_code=400, detail="This layer cannot be edited.")
        geojson = request.geojson
        features = geojson.get("features") if isinstance(geojson, dict) else None
        if geojson.get("type") != "FeatureCollection" or not isinstance(features, list):
            raise HTTPException(status_code=400, detail="Edited data must be a GeoJSON FeatureCollection.")
        if len(features) > 500_000:
            raise HTTPException(status_code=400, detail="Edited GeoJSON contains too many features.")
        current = read_status(workflow_dir)
        previous_edits = current.get("manual_revisions") or []
        source_stage = stage
        if stage == "edited":
            source_stage = next(
                (
                    str(record.get("source_stage") or "").lower()
                    for record in reversed(previous_edits)
                    if str(record.get("source_stage") or "").lower() in editable_stages - {"edited"}
                ),
                "",
            )
            if source_stage not in editable_stages - {"edited"}:
                previous_path = str((current.get("outputs") or {}).get("edited", {}).get("path") or "")
                source_stage = Path(previous_path).name.removesuffix("_edited.geojson") or "combined"
        cleaned_features = []
        for index, feature in enumerate(features):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise HTTPException(status_code=400, detail=f"Feature {index} is invalid.")
            geometry = feature.get("geometry") or {}
            if geometry.get("type") not in {"Polygon", "MultiPolygon"} or not geometry.get("coordinates"):
                raise HTTPException(status_code=400, detail=f"Feature {index} is not a polygon.")
            cleaned = dict(feature)
            properties = dict(cleaned.get("properties") or {})
            properties["manual_edit_source_stage"] = source_stage
            cleaned["properties"] = properties
            cleaned_features.append(cleaned)
        output_path = workflow_dir / f"{source_stage}_edited.geojson"
        _atomic_json(output_path, {"type": "FeatureCollection", "features": cleaned_features})
        superseded_paths = list(workflow_dir.glob("edited_*.geojson"))
        superseded_paths.extend(workflow_dir / name for name in (
            "edited.geojson",
            "combined_edited.geojson",
            "regularized_edited.geojson",
            "panel_rows_edited.geojson",
            "identified_panels_edited.geojson",
            "panel_hierarchy_edited.geojson",
            "deduplicated_edited.geojson",
            "associated_edited.geojson",
        ))
        for legacy_path in set(superseded_paths):
            if legacy_path == output_path or not legacy_path.is_file():
                continue
            try:
                legacy_path.unlink()
            except OSError as exc:
                log.warning("Could not remove superseded edited GeoJSON %s: %s", legacy_path, exc)
        relative = output_path.relative_to(result_dir).as_posix()
        outputs = dict(current.get("outputs") or {})
        outputs["edited"] = {"path": relative}
        revision_name = request.name or f"{current.get('display_name') or workflow_id} — {source_stage}_edited"
        edited_record = {
            "id": "edited",
            "name": revision_name,
            "source_stage": source_stage,
            "path": relative,
            "feature_count": len(cleaned_features),
            "updated_at": datetime.now().isoformat(),
        }
        update_status(
            workflow_dir,
            status="complete",
            stage="manual_edit",
            progress=100,
            message="Edited GeoJSON is ready.",
            outputs=outputs,
            manual_revisions=[edited_record],
        )
        return read_status(workflow_dir)

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
