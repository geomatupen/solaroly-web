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
from typing import Any, Callable, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..postprocess import (
    analyze_geojson,
    analyze_visual_duplicates,
    apply_visual_deduplication,
    associate_anomalies,
    assign_panel_ids,
    build_panel_hierarchy,
    clear_panel_ids,
    combine_tile_fragments,
    deduplicate_anomalies,
    find_review_image,
    image_neighbor_statistics,
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
    max_orientation_difference_deg: float = Field(default=15.0, ge=0.0, le=45.0)
    max_lateral_distance_factor: float = Field(default=1.5, ge=0.25, le=10.0)
    max_along_gap_factor: float = Field(default=1.5, ge=0.25, le=20.0)
    max_inner_row_gap_factor: float = Field(default=0.8, ge=0.0, le=10.0)
    min_row_overlap_percent: float = Field(default=20.0, ge=0.0, le=100.0)


class DeduplicateAnomaliesRequest(BaseModel):
    input_path: str
    output_name: str = "anomaly_postprocess"
    workflow_id: str | None = None
    maximum_center_distance_m: float = Field(default=5.0, gt=0.0, le=100.0)
    neighbor_image_radius_m: float = Field(default=25.0, gt=0.0, le=10_000.0)


class OverlapDeduplicateAnomaliesRequest(BaseModel):
    input_path: str
    output_name: str = "anomaly_postprocess"
    workflow_id: str | None = None
    minimum_overlap_percent: float = Field(default=60.0, gt=0.0, le=100.0)


class NeighborImageStatsRequest(BaseModel):
    input_path: str
    neighbor_image_radius_m: float = Field(default=25.0, gt=0.0, le=10_000.0)


class ManualDuplicateDecision(BaseModel):
    first_index: int = Field(ge=0)
    second_index: int = Field(ge=0)
    keep_index: int = Field(ge=0)


class VisualReviewDecisionRequest(BaseModel):
    first_index: int = Field(ge=0)
    second_index: int = Field(ge=0)
    status: Literal["accepted", "rejected", "unreviewed"]
    keep_index: int | None = Field(default=None, ge=0)


class ApplyVisualDeduplicationRequest(BaseModel):
    deduplication_mode: Literal["threshold", "manual"] = "threshold"
    manual_decisions: list[ManualDuplicateDecision] = Field(default_factory=list)
    duplicate_score_percent: float = Field(default=80.0, ge=0.0, le=100.0)
    appearance_weight_percent: float = Field(default=45.0, ge=0.0, le=100.0)
    context_weight_percent: float = Field(default=20.0, ge=0.0, le=100.0)
    shape_weight_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    size_weight_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    orientation_weight_percent: float = Field(default=10.0, ge=0.0, le=100.0)
    proximity_weight_percent: float = Field(default=5.0, ge=0.0, le=100.0)
    representative_image_center_weight_percent: float = Field(default=40.0, ge=0.0, le=100.0)
    representative_spatial_centrality_weight_percent: float = Field(default=35.0, ge=0.0, le=100.0)
    representative_confidence_weight_percent: float = Field(default=25.0, ge=0.0, le=100.0)


class AssociateAnomaliesRequest(BaseModel):
    panel_path: str
    row_path: str | None = None
    panel_result_id: str | None = None
    panel_workflow_id: str | None = None
    minimum_overlap: float = Field(default=0.20, ge=0.0, le=1.0)
    maximum_distance_m: float = Field(default=1.50, ge=0.0, le=100.0)


class RenameWorkflowRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class ShareLayerRequest(BaseModel):
    name: str | None = Field(default=None, max_length=80)


class EditLayerRequest(BaseModel):
    geojson: dict[str, Any]
    name: str | None = Field(default=None, max_length=80)


class EditSourceRequest(BaseModel):
    input_path: str
    output_name: str = Field(default="edited_source", max_length=80)
    geojson: dict[str, Any]


def _safe_name(value: str, fallback: str) -> str:
    safe = _SAFE_NAME_RE.sub("_", str(value or "").strip()).strip("._-")
    return (safe or fallback)[:80]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _visual_review_decision_summary(
    pairs: list[dict[str, Any]],
) -> tuple[dict[str, int], list[int], list[str]]:
    counts = {"accepted": 0, "rejected": 0, "unreviewed": 0}
    kept_indices: set[int] = set()
    removed_indices: set[int] = set()
    labels_by_index: dict[int, str] = {}
    for pair in pairs:
        first_index = int(pair["first_index"])
        second_index = int(pair["second_index"])
        labels_by_index[first_index] = str(pair.get("first_anomaly_id") or first_index + 1)
        labels_by_index[second_index] = str(pair.get("second_anomaly_id") or second_index + 1)
        status = pair.get("manual_review_status")
        key = status if status in {"accepted", "rejected"} else "unreviewed"
        counts[key] += 1
        if status != "accepted":
            continue
        edge = (first_index, second_index)
        keep_index = int(pair.get("manual_keep_index", edge[0]))
        kept_indices.add(keep_index)
        removed_indices.update(index for index in edge if index != keep_index)
    conflict_indices = sorted(kept_indices & removed_indices)
    conflict_ids = [labels_by_index.get(index, str(index + 1)) for index in conflict_indices]
    return counts, conflict_indices, conflict_ids


def create_postprocess_router(
    get_sessions_dir: Callable[[], Path],
    get_overlays_dir: Callable[[], Path],
    media_url: Callable[[Path], str],
    logger: logging.Logger | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api/results", tags=["postprocess"])
    log = logger or logging.getLogger("pvrt.postprocess")
    neighbor_stats_cache: dict[tuple[str, int, int, float], dict[str, Any]] = {}

    def resolve_result(result_id: str) -> Path:
        workspace_match = re.fullmatch(r"ppjob__(.+)__(segmentation|anomaly)", str(result_id or ""))
        if workspace_match:
            job_id, kind = workspace_match.groups()
            jobs_root = (Path(get_sessions_dir()).resolve() / ".postprocess_jobs").resolve()
            job_dir = (jobs_root / _safe_name(job_id, "")).resolve()
            result = (job_dir / "snapshots" / kind).resolve()
            if (
                not job_id
                or job_dir.parent != jobs_root
                or result.parent != (job_dir / "snapshots").resolve()
                or not result.is_dir()
            ):
                raise HTTPException(status_code=404, detail="Post-processing job snapshot not found.")
            return result
        safe_id = _safe_name(result_id, "")
        root = Path(get_sessions_dir()).resolve()
        result = (root / safe_id).resolve()
        if not safe_id or result.parent != root or not result.is_dir():
            raise HTTPException(status_code=404, detail="Result not found.")
        return result

    def resolve_assets(result_dir: Path) -> Path:
        snapshot_path = result_dir / "snapshot.json"
        if not snapshot_path.is_file():
            return result_dir
        try:
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            original_result_id = _safe_name(str(snapshot.get("original_result_id") or ""), "")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            raise HTTPException(status_code=409, detail="Job snapshot dependency metadata is unavailable.") from exc
        sessions_root = Path(get_sessions_dir()).resolve()
        assets = (sessions_root / original_result_id).resolve()
        if not original_result_id or assets.parent != sessions_root or not assets.is_dir():
            raise HTTPException(
                status_code=409,
                detail="The test result referenced by this post-processing job is unavailable.",
            )
        return assets

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

    def source_fingerprint(path: Path) -> dict[str, int]:
        """Small, read-only source signature used to detect later replacement."""
        stat = path.stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

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
        scoring_parameters = payload.get("scoring_parameters") or {}
        deduplication_method = (
            (payload.get("deduplicate_stats") or {}).get("deduplication_method")
            or scoring_parameters.get("deduplication_mode")
        )
        if (
            payload.get("status") == "complete"
            and payload.get("stage") == "deduplicate_apply"
            and deduplication_method == "manual"
        ):
            accepted_count = len(scoring_parameters.get("manual_decisions") or [])
            payload["message"] = (
                "Manual selections applied. Deduplicated anomaly GeoJSON is ready"
                f" from {accepted_count} accepted pair{'s' if accepted_count != 1 else ''}."
            )
        outputs = dict(payload.get("outputs") or {})
        # Edited copies were used by an older workflow. Current edits update the
        # selected processing layer directly, so legacy copies stay hidden.
        outputs.pop("edited", None)
        payload["outputs"] = outputs
        payload.pop("manual_revisions", None)
        source_path = str(payload.get("input_path") or "")
        expected_fingerprint = payload.get("source_fingerprint")
        if source_path and isinstance(expected_fingerprint, dict):
            candidate = (workflow_dir.parent.parent / source_path).resolve()
            try:
                candidate.relative_to(workflow_dir.parent.parent.resolve())
                current_fingerprint = source_fingerprint(candidate)
                payload["source_changed"] = current_fingerprint != expected_fingerprint
                payload["source_current_fingerprint"] = current_fingerprint
            except (OSError, ValueError):
                payload["source_changed"] = True
                payload["source_current_fingerprint"] = None
        for output in outputs.values():
            if isinstance(output, dict) and output.get("path"):
                output_path = (workflow_dir.parent.parent / output["path"]).resolve()
                if output_path.is_file():
                    output["url"] = media_url(output_path)
                    output["mtime"] = int(output_path.stat().st_mtime_ns)
        review_path_value = str(payload.get("visual_review_path") or "")
        if not review_path_value and (workflow_dir / "visual_review.json").is_file():
            review_path_value = (workflow_dir / "visual_review.json").relative_to(workflow_dir.parent.parent).as_posix()
            payload["visual_review_path"] = review_path_value
        if review_path_value:
            review_path = (workflow_dir.parent.parent / review_path_value).resolve()
            try:
                review_path.relative_to(workflow_dir)
                review = json.loads(review_path.read_text(encoding="utf-8"))
                all_review_pairs = review.get("pairs") or []
                stored_preview = payload.get("visual_review_preview")
                if isinstance(stored_preview, dict):
                    review_pairs_source = stored_preview.get("pairs") or []
                    review_total = int(stored_preview.get("total_pairs") or 0)
                else:
                    review_pairs_source = all_review_pairs
                    review_total = len(review_pairs_source)
                decision_counts, conflict_indices, conflict_ids = _visual_review_decision_summary(all_review_pairs)
                payload["visual_review_decision_counts"] = decision_counts
                payload["visual_review_conflict_indices"] = conflict_indices
                payload["visual_review_conflict_ids"] = conflict_ids
                review_pairs = []
                # Keep workflow-list payloads small; the UI intentionally shows
                # a representative review grid rather than every candidate.
                for pair in review_pairs_source[:12]:
                    item = dict(pair)
                    for key in ("first_crop_path", "second_crop_path"):
                        relative = str(item.get(key) or "")
                        crop_path = (workflow_dir / relative).resolve() if relative else None
                        if crop_path and crop_path.is_file() and crop_path.is_relative_to(workflow_dir):
                            item[key.replace("_path", "_url")] = media_url(crop_path)
                    review_pairs.append(item)
                payload["visual_review"] = {
                    "pairs": review_pairs,
                    "total_pairs": review_total,
                    "displayed_pairs": len(review_pairs),
                }
                payload["visual_review_available"] = True
                payload["visual_review_total_pairs"] = review_total
                if not isinstance(payload.get("visual_analysis_stats"), dict):
                    parameters = payload.get("parameters") or {}
                    statistics_pairs = review_pairs_source
                    if isinstance(stored_preview, dict):
                        statistics_pairs = all_review_pairs
                    payload["visual_analysis_stats"] = {
                        "spatial_candidate_pairs": review_total,
                        "visually_compared_pairs": sum(
                            pair.get("appearance_similarity") is not None for pair in statistics_pairs
                        ),
                        "missing_image_pairs": sum(
                            pair.get("appearance_similarity") is None for pair in statistics_pairs
                        ),
                        "neighbor_image_radius_m": parameters.get("neighbor_image_radius_m", 0),
                        "maximum_location_shift_m": parameters.get("maximum_center_distance_m", 0),
                        "recovered_from_saved_review": True,
                    }
            except (OSError, ValueError, json.JSONDecodeError):
                payload["visual_review"] = {"pairs": [], "total_pairs": 0, "displayed_pairs": 0}
            payload.pop("visual_review_preview", None)
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

    def remove_linked_overlays(result_id: str, workflow_id: str, stage: str | None = None) -> None:
        overlays_root = Path(get_overlays_dir()).resolve()
        if not overlays_root.is_dir():
            return
        for overlay_dir in overlays_root.iterdir():
            if not overlay_dir.is_dir():
                continue
            try:
                metadata = json.loads((overlay_dir / ".overlay_meta.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                metadata.get("reference_kind") == "postprocess"
                and metadata.get("source_result") == result_id
                and metadata.get("workflow_id") == workflow_id
                and (stage is None or metadata.get("stage") == stage)
            ):
                shutil.rmtree(overlay_dir)

    def clear_panel_anomaly_assignments(
        anomaly_workflow_dir: Path,
        status: dict[str, Any],
    ) -> None:
        """Clear panel attributes derived from an associated output being invalidated."""
        parameters = status.get("association_parameters") or {}
        panel_path_value = str(parameters.get("panel_path") or "")
        if not panel_path_value:
            return
        try:
            panel_result_dir = resolve_result(str(parameters.get("panel_result_id") or status.get("result_id") or ""))
            panel_path = resolve_input(panel_result_dir, panel_path_value)
            payload = json.loads(panel_path.read_text(encoding="utf-8"))
            for panel in payload.get("features") or []:
                if not isinstance(panel, dict):
                    continue
                properties = dict(panel.get("properties") or {})
                properties["anomaly_count"] = 0
                properties["anomaly_ids"] = []
                panel["properties"] = properties
            _atomic_json(panel_path, payload)
            panel_workflow_id = str(parameters.get("panel_workflow_id") or "")
            if panel_workflow_id:
                panel_workflow_dir = resolve_workflow(panel_result_dir, panel_workflow_id)
                panel_status = read_status(panel_workflow_dir)
                update_status(
                    panel_workflow_dir,
                    anomaly_association=None,
                    outputs=panel_status.get("outputs") or {},
                )
        except (HTTPException, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            append_log(anomaly_workflow_dir, f"WARNING: Could not clear stale panel anomaly attributes: {exc}")

    @router.get("/{result_id}/postprocess/geojsons")
    async def list_geojsons(result_id: str) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        files = list(result_dir.glob("*.geojson"))
        files.extend(
            path
            for path in (result_dir / "postprocess").glob("*/*.geojson")
            if not path.name.startswith("edited_")
            and not path.name.endswith("_edited.geojson")
            and path.name not in {
                "edited.geojson", "panel_hierarchy.geojson", "solar_panels.geojson",
                "panel_rows.geojson", "identified_panels.geojson",
            }
        )
        items = []
        for path in sorted(set(files), key=lambda item: (item.stat().st_mtime, item.name), reverse=True):
            relative = path.relative_to(result_dir).as_posix()
            stage = "source"
            stage_names = {
                "combined.geojson": "combined",
                "regularized.geojson": "regularized",
                "solar_rows.geojson": "solar_rows",
                "overlap_deduplicated.geojson": "overlap_deduplicated",
                "deduplicated.geojson": "deduplicated",
                "associated.geojson": "associated",
            }
            stage = stage_names.get(path.name, stage)
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
        assets_dir = resolve_assets(result_dir)
        input_path = resolve_input(result_dir, request.input_path)
        try:
            summary = await asyncio.to_thread(
                analyze_geojson,
                input_path,
                assets_dir,
                edge_tolerance_px=request.edge_tolerance_px,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"ok": True, "input_path": request.input_path, "summary": summary}

    @router.post("/{result_id}/postprocess/source-edits")
    async def save_source_working_copy(result_id: str, request: EditSourceRequest) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        original = resolve_input(result_dir, request.input_path)
        geojson = request.geojson
        features = geojson.get("features") if isinstance(geojson, dict) else None
        if geojson.get("type") != "FeatureCollection" or not isinstance(features, list):
            raise HTTPException(status_code=400, detail="Edited data must be a GeoJSON FeatureCollection.")
        if len(features) > 500_000:
            raise HTTPException(status_code=400, detail="Edited GeoJSON contains too many features.")
        cleaned_features = []
        for index, item in enumerate(features):
            if not isinstance(item, dict) or item.get("type") != "Feature":
                raise HTTPException(status_code=400, detail=f"Feature {index} is invalid.")
            geometry = item.get("geometry") or {}
            if geometry.get("type") not in {"Polygon", "MultiPolygon"} or not geometry.get("coordinates"):
                raise HTTPException(status_code=400, detail=f"Feature {index} is not a polygon.")
            cleaned = dict(item)
            properties = dict(cleaned.get("properties") or {})
            properties["manually_edited"] = True
            cleaned["properties"] = properties
            cleaned_features.append(cleaned)
        prefix = _safe_name(request.output_name, "edited_source")
        workflow_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        workflow_dir = result_dir / "postprocess" / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=False)
        output_path = workflow_dir / "source.geojson"
        saved_payload = {
            key: value for key, value in geojson.items()
            if key not in {"type", "features"}
        }
        saved_payload.update({"type": "FeatureCollection", "features": cleaned_features})
        _atomic_json(output_path, saved_payload)
        relative = output_path.relative_to(result_dir).as_posix()
        original_relative = original.relative_to(result_dir).as_posix()
        now = datetime.now().isoformat()
        status = {
            "ok": True,
            "id": workflow_id,
            "display_name": request.output_name.strip() or prefix,
            "result_id": result_dir.name,
            "status": "complete",
            "stage": "manual_source_edit",
            "progress": 100,
            "message": "Editable source working copy saved.",
            "created_at": now,
            "updated_at": now,
            "input_path": relative,
            "original_input_path": original_relative,
            "source_fingerprint": source_fingerprint(output_path),
            "original_source_fingerprint": source_fingerprint(original),
            "outputs": {"source": {"path": relative}},
            "manual_edits": {"source": {"feature_count": len(cleaned_features), "updated_at": now}},
        }
        _atomic_json(workflow_dir / "status.json", status)
        _atomic_json(workflow_dir / "postprocess_meta.json", status)
        return read_status(workflow_dir)

    @router.post("/{result_id}/postprocess/combine")
    async def combine(result_id: str, request: CombineRequest) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        assets_dir = resolve_assets(result_dir)
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
            "source_fingerprint": source_fingerprint(input_path),
            "parameters": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "outputs": outputs,
        }
        _atomic_json(workflow_dir / "status.json", initial)
        _atomic_json(workflow_dir / "postprocess_meta.json", initial)

        def run() -> None:
            output_path = workflow_dir / "combined.geojson"
            try:
                stats = combine_tile_fragments(
                    input_path,
                    output_path,
                    assets_dir,
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
        manual_edits = {
            stage: details
            for stage, details in (status.get("manual_edits") or {}).items()
            if stage != "solar_rows"
        }
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
            manual_edits=manual_edits,
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
        manual_edits = {
            stage: details
            for stage, details in (status.get("manual_edits") or {}).items()
            if stage not in {"regularized", "solar_rows"}
        }
        update_status(
            workflow_dir,
            status="queued",
            stage="hierarchy",
            progress=0,
            message="Queued row generation.",
            hierarchy_parameters=request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            assignment_stats=None,
            outputs=outputs,
            manual_edits=manual_edits,
        )

        def run() -> None:
            rows_path = workflow_dir / "solar_rows.geojson"
            try:
                stats = build_panel_hierarchy(
                    input_path,
                    None,
                    rows_output_path=rows_path,
                    panels_output_path=input_path,
                    max_orientation_difference_deg=request.max_orientation_difference_deg,
                    max_lateral_distance_factor=request.max_lateral_distance_factor,
                    max_along_gap_factor=request.max_along_gap_factor,
                    max_inner_row_gap_factor=request.max_inner_row_gap_factor,
                    min_row_overlap_percent=request.min_row_overlap_percent,
                    assign_ids=False,
                    callback=progress_callback(workflow_dir, "hierarchy"),
                )
                clear_panel_ids(input_path)
                latest = read_status(workflow_dir)
                current_outputs = dict(latest.get("outputs") or {})
                current_outputs["regularized"] = {
                    "path": input_path.relative_to(result_dir).as_posix()
                }
                current_outputs["solar_rows"] = {
                    "path": rows_path.relative_to(result_dir).as_posix()
                }
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="hierarchy",
                    progress=100,
                    message="Rows are ready for editing.",
                    hierarchy_stats=stats,
                    outputs=current_outputs,
                )
                log.info("UI:OK:postprocess: Rows ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(workflow_dir, status="failed", stage="hierarchy", message=str(exc), error=str(exc))
                log.exception("Row generation failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return {"ok": True, "id": workflow_id, "status": "queued", "stage": "hierarchy"}

    @router.post("/{result_id}/postprocess/{workflow_id}/assign-ids")
    async def assign_ids(result_id: str, workflow_id: str) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        status = read_status(workflow_dir)
        if status.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="This workflow is already running.")
        outputs = status.get("outputs") or {}
        panel_value = str((outputs.get("regularized") or {}).get("path") or "")
        row_value = str((outputs.get("solar_rows") or {}).get("path") or "")
        if not panel_value or not row_value:
            raise HTTPException(status_code=409, detail="Build and edit Rows before assigning IDs.")
        panels_path = resolve_input(result_dir, panel_value)
        rows_path = resolve_input(result_dir, row_value)
        try:
            panels_path.relative_to(workflow_dir)
            rows_path.relative_to(workflow_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Select outputs from this segmentation workflow.") from exc
        parameters = status.get("hierarchy_parameters") or {}
        update_status(
            workflow_dir,
            status="queued",
            stage="assign_ids",
            progress=0,
            message="Queued row and panel ID assignment.",
            outputs=outputs,
        )

        def run() -> None:
            try:
                stats = assign_panel_ids(
                    panels_path,
                    rows_path,
                    max_orientation_difference_deg=float(parameters.get("max_orientation_difference_deg", 15.0)),
                    max_lateral_distance_factor=float(parameters.get("max_lateral_distance_factor", 1.5)),
                    max_along_gap_factor=float(parameters.get("max_along_gap_factor", 1.5)),
                    callback=progress_callback(workflow_dir, "assign_ids"),
                )
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="assign_ids",
                    progress=100,
                    message="Row and panel IDs are ready.",
                    assignment_stats=stats,
                    outputs=outputs,
                )
                log.info("UI:OK:postprocess: Row and panel IDs ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(workflow_dir, status="failed", stage="assign_ids", message=str(exc), error=str(exc))
                log.exception("Row and panel ID assignment failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return {"ok": True, "id": workflow_id, "status": "queued", "stage": "assign_ids"}

    @router.post("/{result_id}/postprocess/anomalies/overlap-deduplicate")
    async def overlap_deduplicate(
        result_id: str,
        request: OverlapDeduplicateAnomaliesRequest,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        input_path = resolve_input(result_dir, request.input_path)
        prefix = _safe_name(request.output_name, "anomaly_postprocess")
        if request.workflow_id:
            workflow_dir = resolve_workflow(result_dir, request.workflow_id)
            existing = read_status(workflow_dir)
            if existing.get("status") in {"queued", "running"}:
                raise HTTPException(status_code=409, detail="This workflow is already running.")
            workflow_id = request.workflow_id
            had_associated_output = bool((existing.get("outputs") or {}).get("associated"))
            outputs = remove_stage_outputs(
                result_dir,
                workflow_dir,
                existing,
                {"overlap_deduplicated", "deduplicated", "associated"},
            )
            if had_associated_output:
                clear_panel_anomaly_assignments(workflow_dir, existing)
            review_path = workflow_dir / "visual_review.json"
            review_images_dir = workflow_dir / "review_images"
            if review_path.is_file():
                review_path.unlink()
            if review_images_dir.is_dir():
                shutil.rmtree(review_images_dir)
        else:
            workflow_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            workflow_dir = result_dir / "postprocess" / workflow_id
            workflow_dir.mkdir(parents=True, exist_ok=False)
            existing = {}
            outputs = {}
        initial = {
            "ok": True,
            "id": workflow_id,
            "workflow_kind": "anomaly",
            "display_name": request.output_name.strip() or prefix,
            "result_id": result_dir.name,
            "status": "queued",
            "stage": "overlap_deduplicate",
            "progress": 0,
            "message": "Queued overlapping-polygon removal.",
            "created_at": existing.get("created_at") or datetime.now().isoformat(),
            "input_path": input_path.relative_to(result_dir).as_posix(),
            "source_fingerprint": source_fingerprint(input_path),
            "parameters": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "outputs": outputs,
            "overlap_deduplicate_stats": None,
            "deduplicate_stats": None,
            "association_stats": None,
            "association_parameters": None,
            "visual_review_available": False,
            "visual_review_total_pairs": 0,
            "visual_review_path": "",
            "visual_review_preview": {"pairs": [], "total_pairs": 0},
        }
        _atomic_json(workflow_dir / "status.json", initial)
        _atomic_json(workflow_dir / "postprocess_meta.json", initial)

        def run_overlap() -> None:
            output_path = workflow_dir / "overlap_deduplicated.geojson"
            try:
                stats = deduplicate_anomalies(
                    input_path,
                    output_path,
                    minimum_smaller_overlap=request.minimum_overlap_percent / 100.0,
                    overlap_only=True,
                    callback=progress_callback(workflow_dir, "overlap_deduplicate"),
                )
                output = {"path": output_path.relative_to(result_dir).as_posix()}
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="overlap_deduplicate",
                    progress=100,
                    message=f"Overlapping polygons at {request.minimum_overlap_percent:g}% or more were removed.",
                    overlap_deduplicate_stats=stats,
                    overlap_input_path=output_path.relative_to(result_dir).as_posix(),
                    outputs={"overlap_deduplicated": output},
                )
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(workflow_dir, status="failed", stage="overlap_deduplicate", message=str(exc), error=str(exc))
                log.exception("Overlapping anomaly removal failed for %s", result_dir.name)

        _EXECUTOR.submit(run_overlap)
        return initial

    @router.post("/{result_id}/postprocess/anomalies/deduplicate")
    async def deduplicate(result_id: str, request: DeduplicateAnomaliesRequest) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        assets_dir = resolve_assets(result_dir)
        input_path = resolve_input(result_dir, request.input_path)
        prefix = _safe_name(request.output_name, "anomaly_postprocess")
        if request.workflow_id:
            workflow_dir = resolve_workflow(result_dir, request.workflow_id)
            existing = read_status(workflow_dir)
            workflow_id = request.workflow_id
        else:
            workflow_id = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            workflow_dir = result_dir / "postprocess" / workflow_id
            workflow_dir.mkdir(parents=True, exist_ok=False)
            existing = {}
        initial = {**existing,
            "ok": True,
            "id": workflow_id,
            "workflow_kind": "anomaly",
            "display_name": request.output_name.strip() or prefix,
            "result_id": result_dir.name,
            "status": "queued",
            "stage": "deduplicate",
            "progress": 0,
            "message": "Queued visual duplicate analysis.",
            "created_at": existing.get("created_at") or datetime.now().isoformat(),
            "input_path": input_path.relative_to(result_dir).as_posix(),
            "source_fingerprint": source_fingerprint(input_path),
            "parameters": request.model_dump() if hasattr(request, "model_dump") else request.dict(),
            "outputs": dict(existing.get("outputs") or {}),
        }
        previous_review_available = bool(
            existing.get("visual_review_available")
            or existing.get("visual_review_path")
            or existing.get("visual_review")
        )
        previous_review_total = int(
            existing.get("visual_review_total_pairs")
            or (existing.get("visual_review") or {}).get("total_pairs")
            or 0
        )
        for transient_key in (
            "visual_review", "visual_review_path", "visual_review_preview", "visual_analysis_stats",
        ):
            initial.pop(transient_key, None)
        initial["visual_review_available"] = previous_review_available
        initial["visual_review_total_pairs"] = previous_review_total
        _atomic_json(workflow_dir / "status.json", initial)
        _atomic_json(workflow_dir / "postprocess_meta.json", initial)

        def run() -> None:
            review_path = workflow_dir / "visual_review.json"
            review_images_dir = workflow_dir / "review_images"
            try:
                if review_images_dir.is_dir():
                    shutil.rmtree(review_images_dir)
                if review_path.is_file():
                    review_path.unlink()
                stats = analyze_visual_duplicates(
                    input_path,
                    review_path,
                    review_images_dir,
                    assets_dir,
                    maximum_center_distance_m=request.maximum_center_distance_m,
                    neighbor_image_radius_m=request.neighbor_image_radius_m,
                    callback=progress_callback(workflow_dir, "deduplicate"),
                )
                review_payload = json.loads(review_path.read_text(encoding="utf-8"))
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="deduplicate",
                    progress=100,
                    message="Visual duplicate candidates are ready. Choose a similarity threshold to apply.",
                    visual_analysis_stats=stats,
                    visual_review_path=review_path.relative_to(result_dir).as_posix(),
                    visual_review_preview={
                        "pairs": (review_payload.get("pairs") or [])[:12],
                        "total_pairs": len(review_payload.get("pairs") or []),
                    },
                    visual_review_available=True,
                    visual_review_total_pairs=len(review_payload.get("pairs") or []),
                    outputs=dict(existing.get("outputs") or {}),
                )
                log.info("UI:OK:postprocess: Visual anomaly review ready for %s", result_dir.name)
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(
                    workflow_dir,
                    status="failed",
                    stage="deduplicate",
                    message=str(exc),
                    error=str(exc),
                    visual_review_available=False,
                    visual_review_total_pairs=0,
                    visual_review_path="",
                    visual_review_preview={"pairs": [], "total_pairs": 0},
                )
                log.exception("Anomaly deduplication failed for %s", result_dir.name)

        _EXECUTOR.submit(run)
        return initial

    @router.post("/{result_id}/postprocess/anomalies/neighbor-stats")
    async def anomaly_neighbor_stats(
        result_id: str,
        request: NeighborImageStatsRequest,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        assets_dir = resolve_assets(result_dir)
        input_path = resolve_input(result_dir, request.input_path)
        fingerprint = source_fingerprint(input_path)
        key = (
            str(input_path),
            fingerprint["size"],
            fingerprint["mtime_ns"],
            round(request.neighbor_image_radius_m, 3),
        )
        cached = neighbor_stats_cache.get(key)
        if cached is None:
            cached = await asyncio.to_thread(
                image_neighbor_statistics,
                input_path,
                assets_dir,
                request.neighbor_image_radius_m,
            )
            if len(neighbor_stats_cache) >= 64:
                neighbor_stats_cache.pop(next(iter(neighbor_stats_cache)))
            neighbor_stats_cache[key] = cached
        return {"ok": True, **cached}

    @router.post("/{result_id}/postprocess/{workflow_id}/deduplicate/apply")
    async def apply_deduplication(
        result_id: str,
        workflow_id: str,
        request: ApplyVisualDeduplicationRequest,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        status = read_status(workflow_dir)
        input_path = resolve_input(result_dir, str(status.get("input_path") or ""))
        review_path = workflow_dir / "visual_review.json"
        if not review_path.is_file():
            raise HTTPException(status_code=409, detail="Run visual duplicate analysis before applying a threshold.")
        manual_decisions = [
            decision.model_dump() if hasattr(decision, "model_dump") else decision.dict()
            for decision in request.manual_decisions
        ]
        if request.deduplication_mode == "manual":
            try:
                review = json.loads(review_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HTTPException(status_code=500, detail="Visual review data could not be read.") from exc
            saved_by_edge = {
                tuple(sorted((int(pair["first_index"]), int(pair["second_index"])))): {
                    "first_index": int(pair["first_index"]),
                    "second_index": int(pair["second_index"]),
                    "keep_index": int(pair.get("manual_keep_index", pair["first_index"])),
                }
                for pair in (review.get("pairs") or [])
                if pair.get("manual_review_status") == "accepted"
            }
            saved_by_edge.update({
                tuple(sorted((int(decision["first_index"]), int(decision["second_index"])))): decision
                for decision in manual_decisions
            })
            manual_decisions = list(saved_by_edge.values())
        if request.deduplication_mode == "manual" and not manual_decisions:
            raise HTTPException(status_code=400, detail="Mark at least one image pair as a duplicate before applying manual review.")
        supplied_fields = getattr(request, "model_fields_set", None)
        if supplied_fields is None:
            supplied_fields = getattr(request, "__fields_set__", set())
        legacy_weight_total = sum((
            request.appearance_weight_percent,
            request.context_weight_percent,
            request.shape_weight_percent,
            request.size_weight_percent,
            request.proximity_weight_percent,
        ))
        orientation_weight = request.orientation_weight_percent
        if "orientation_weight_percent" not in supplied_fields:
            orientation_weight = max(0.0, 100.0 - legacy_weight_total)
        weight_values = {
            "appearance": request.appearance_weight_percent,
            "context": request.context_weight_percent,
            "shape": request.shape_weight_percent,
            "size": request.size_weight_percent,
            "orientation": orientation_weight,
            "proximity": request.proximity_weight_percent,
        }
        if request.deduplication_mode == "threshold" and abs(sum(weight_values.values()) - 100.0) > 0.01:
            raise HTTPException(status_code=400, detail="Duplicate score weights must total 100%.")
        representative_weight_values = {
            "image_center": request.representative_image_center_weight_percent,
            "spatial_centrality": request.representative_spatial_centrality_weight_percent,
            "model_confidence": request.representative_confidence_weight_percent,
        }
        if request.deduplication_mode == "threshold" and abs(sum(representative_weight_values.values()) - 100.0) > 0.01:
            raise HTTPException(status_code=400, detail="Representative-selection weights must total 100%.")
        threshold = request.duplicate_score_percent / 100.0
        normalized_weights = {name: value / 100.0 for name, value in weight_values.items()}
        normalized_representative_weights = {
            name: value / 100.0 for name, value in representative_weight_values.items()
        }
        queued_message = (
            f"Queued {len(manual_decisions)} accepted manual duplicate "
            f"selection{'s' if len(manual_decisions) != 1 else ''}."
            if request.deduplication_mode == "manual"
            else f"Queued deduplication at {request.duplicate_score_percent:g}% duplicate score."
        )
        update_status(
            workflow_dir,
            status="queued",
            stage="deduplicate_apply",
            progress=0,
            message=queued_message,
            scoring_parameters={
                **(request.model_dump() if hasattr(request, "model_dump") else request.dict()),
                "orientation_weight_percent": orientation_weight,
                "manual_decisions": manual_decisions if request.deduplication_mode == "manual" else [],
            },
        )

        def run_apply() -> None:
            output_path = workflow_dir / "deduplicated.geojson"
            try:
                stats = apply_visual_deduplication(
                    input_path,
                    review_path,
                    output_path,
                    duplicate_score_threshold=threshold,
                    weights=normalized_weights,
                    representative_weights=normalized_representative_weights,
                    manual_decisions=(
                        manual_decisions
                        if request.deduplication_mode == "manual" else None
                    ),
                    callback=progress_callback(workflow_dir, "deduplicate_apply"),
                )
                latest = read_status(workflow_dir)
                had_associated_output = bool((latest.get("outputs") or {}).get("associated"))
                current_outputs = remove_stage_outputs(
                    result_dir,
                    workflow_dir,
                    latest,
                    {"associated"},
                )
                if had_associated_output:
                    clear_panel_anomaly_assignments(workflow_dir, latest)
                current_outputs["deduplicated"] = {"path": output_path.relative_to(result_dir).as_posix()}
                completion_message = (
                    f"Manual selections applied. Deduplicated anomaly GeoJSON is ready from "
                    f"{len(manual_decisions)} accepted pair{'s' if len(manual_decisions) != 1 else ''}."
                    if request.deduplication_mode == "manual"
                    else f"Deduplicated anomaly GeoJSON is ready at {request.duplicate_score_percent:g}% duplicate score."
                )
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="deduplicate_apply",
                    progress=100,
                    message=completion_message,
                    deduplicate_stats=stats,
                    association_stats=None,
                    association_parameters=None,
                    outputs=current_outputs,
                )
            except Exception as exc:
                append_log(workflow_dir, f"ERROR: {exc}")
                update_status(workflow_dir, status="failed", stage="deduplicate_apply", message=str(exc), error=str(exc))
                log.exception("Visual anomaly deduplication application failed for %s", result_dir.name)

        _EXECUTOR.submit(run_apply)
        return read_status(workflow_dir)

    @router.get("/{result_id}/postprocess/{workflow_id}/visual-review")
    async def visual_review_page(
        result_id: str,
        workflow_id: str,
        offset: int = 0,
        limit: int = 48,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        assets_dir = resolve_assets(result_dir)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        review_path = workflow_dir / "visual_review.json"
        if not review_path.is_file():
            raise HTTPException(status_code=404, detail="Visual review is not available for this workflow.")
        offset = max(0, offset)
        limit = max(1, min(100, limit))
        try:
            review = await asyncio.to_thread(
                lambda: json.loads(review_path.read_text(encoding="utf-8"))
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=500, detail="Visual review data could not be read.") from exc
        all_pairs = review.get("pairs") or []
        decision_counts, conflict_indices, conflict_ids = _visual_review_decision_summary(all_pairs)
        pairs = []
        for pair in all_pairs[offset:offset + limit]:
            item = dict(pair)
            for key in ("first_crop_path", "second_crop_path"):
                relative = str(item.get(key) or "")
                crop_path = (workflow_dir / relative).resolve() if relative else None
                if crop_path and crop_path.is_file() and crop_path.is_relative_to(workflow_dir):
                    item[key.replace("_path", "_url")] = media_url(crop_path)
            for prefix in ("first", "second"):
                image_path = find_review_image(assets_dir, str(item.get(f"{prefix}_image") or ""))
                if image_path and image_path.is_file() and image_path.is_relative_to(assets_dir):
                    item[f"{prefix}_image_url"] = media_url(image_path)
            pairs.append(item)
        return {
            "ok": True,
            "pairs": pairs,
            "offset": offset,
            "limit": limit,
            "total_pairs": len(all_pairs),
            "decision_counts": decision_counts,
            "conflict_indices": conflict_indices,
            "conflict_ids": conflict_ids,
            "has_more": offset + len(pairs) < len(all_pairs),
        }

    @router.patch("/{result_id}/postprocess/{workflow_id}/visual-review/decision")
    async def update_visual_review_decision(
        result_id: str,
        workflow_id: str,
        request: VisualReviewDecisionRequest,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        review_path = workflow_dir / "visual_review.json"
        if not review_path.is_file():
            raise HTTPException(status_code=404, detail="Visual review is not available for this workflow.")
        edge = tuple(sorted((request.first_index, request.second_index)))
        if edge[0] == edge[1]:
            raise HTTPException(status_code=400, detail="A comparison must contain two different anomalies.")
        if request.status == "accepted" and request.keep_index not in edge:
            raise HTTPException(status_code=400, detail="Choose one anomaly from the accepted pair to keep.")
        try:
            with _STATUS_LOCK:
                review = json.loads(review_path.read_text(encoding="utf-8"))
                labels_by_index: dict[int, str] = {}
                for review_pair in review.get("pairs") or []:
                    first_index = int(review_pair["first_index"])
                    second_index = int(review_pair["second_index"])
                    labels_by_index[first_index] = str(review_pair.get("first_anomaly_id") or first_index + 1)
                    labels_by_index[second_index] = str(review_pair.get("second_anomaly_id") or second_index + 1)

                def anomaly_label(index: int) -> str:
                    return labels_by_index.get(index, str(index + 1))

                matched = None
                for pair in review.get("pairs") or []:
                    pair_edge = tuple(sorted((int(pair["first_index"]), int(pair["second_index"]))))
                    if pair_edge != edge:
                        continue
                    matched = pair
                    break
                if matched is None:
                    raise HTTPException(status_code=404, detail="The comparison pair is not part of this review.")
                if request.status == "accepted":
                    requested_keep = int(request.keep_index)
                    requested_remove = next(index for index in edge if index != requested_keep)
                    for pair in review.get("pairs") or []:
                        pair_edge = tuple(sorted((int(pair["first_index"]), int(pair["second_index"]))))
                        if pair_edge == edge or pair.get("manual_review_status") != "accepted":
                            continue
                        existing_keep = int(pair.get("manual_keep_index", pair_edge[0]))
                        existing_remove = next(index for index in pair_edge if index != existing_keep)
                        existing_label = f"{anomaly_label(pair_edge[0])}–{anomaly_label(pair_edge[1])}"
                        if existing_keep == requested_remove:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"Cannot accept pair {anomaly_label(edge[0])}–{anomaly_label(edge[1])} while keeping anomaly "
                                    f"{anomaly_label(requested_keep)}. Anomaly {anomaly_label(requested_remove)} is already kept by "
                                    f"accepted pair {existing_label}. Reject this pair or undo that accepted pair first."
                                ),
                            )
                        if existing_remove == requested_keep:
                            raise HTTPException(
                                status_code=409,
                                detail=(
                                    f"Cannot keep anomaly {anomaly_label(requested_keep)}. Accepted pair {existing_label} "
                                    f"already removes it and keeps anomaly {anomaly_label(existing_keep)}. Reject this pair "
                                    "or undo that accepted pair first."
                                ),
                            )
                if request.status == "unreviewed":
                    matched.pop("manual_review_status", None)
                    matched.pop("manual_keep_index", None)
                else:
                    matched["manual_review_status"] = request.status
                    if request.status == "accepted":
                        matched["manual_keep_index"] = request.keep_index
                    else:
                        matched.pop("manual_keep_index", None)
                _atomic_json(review_path, review)
        except HTTPException:
            raise
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Visual review decision could not be saved.") from exc
        decision_counts, conflict_indices, conflict_ids = _visual_review_decision_summary(review.get("pairs") or [])
        return {
            "ok": True,
            "first_index": request.first_index,
            "second_index": request.second_index,
            "status": request.status,
            "keep_index": request.keep_index if request.status == "accepted" else None,
            "decision_counts": decision_counts,
            "conflict_indices": conflict_indices,
            "conflict_ids": conflict_ids,
        }

    @router.post("/{result_id}/postprocess/{workflow_id}/associate")
    async def associate(
        result_id: str, workflow_id: str, request: AssociateAnomaliesRequest
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        status = read_status(workflow_dir)
        if status.get("status") in {"queued", "running"}:
            raise HTTPException(status_code=409, detail="This workflow is already running.")
        outputs = status.get("outputs") or {}
        anomaly_output = outputs.get("deduplicated") or outputs.get("overlap_deduplicated") or {}
        anomaly_path = resolve_input(result_dir, str(anomaly_output.get("path") or ""))
        panel_result_dir = resolve_result(request.panel_result_id or result_id)
        panel_path = resolve_input(panel_result_dir, request.panel_path)
        row_path = resolve_input(panel_result_dir, request.row_path) if request.row_path else None
        panel_workflow_dir = (
            resolve_workflow(panel_result_dir, request.panel_workflow_id)
            if request.panel_workflow_id
            else None
        )
        if panel_workflow_dir is not None:
            try:
                panel_path.relative_to(panel_workflow_dir)
                if row_path is not None:
                    row_path.relative_to(panel_workflow_dir)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Selected panels or rows do not belong to the supplied segmentation workflow.") from exc
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
                    panel_output_path=panel_path,
                    row_path=row_path,
                    row_output_path=row_path,
                    minimum_overlap=request.minimum_overlap,
                    maximum_distance_m=request.maximum_distance_m,
                    callback=progress_callback(workflow_dir, "associate"),
                )
                stats["panel_updated_mtime"] = int(panel_path.stat().st_mtime_ns)
                stats["row_updated_mtime"] = int(row_path.stat().st_mtime_ns) if row_path is not None else None
                if panel_workflow_dir is not None:
                    panel_status = read_status(panel_workflow_dir)
                    update_status(
                        panel_workflow_dir,
                        anomaly_association={
                            "anomaly_result_id": result_dir.name,
                            "anomaly_workflow_id": workflow_id,
                            "associated_anomalies": stats["assigned"],
                            "panels_with_anomalies": stats["panels_with_anomalies"],
                            "rows_with_anomalies": stats["rows_with_anomalies"],
                            "updated_at": datetime.now().isoformat(),
                        },
                        outputs=panel_status.get("outputs") or {},
                    )
                latest = read_status(workflow_dir)
                outputs = dict(latest.get("outputs") or {})
                outputs["associated"] = {"path": output_path.relative_to(result_dir).as_posix()}
                update_status(
                    workflow_dir,
                    status="complete",
                    stage="associate",
                    progress=100,
                    message="Final anomalies are ready with panel and row IDs.",
                    association_stats=stats,
                    outputs=outputs,
                )
                log.info("UI:OK:postprocess: Final anomalies ready for %s", result_dir.name)
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
        remove_linked_overlays(result_dir.name, workflow_dir.name)
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
        allowed_stages = {"combined", "regularized", "solar_rows", "overlap_deduplicated", "deduplicated", "associated"}
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
        stage_labels = {
            "solar_rows": "Rows",
            "overlap_deduplicated": "Overlap-filtered anomalies",
            "deduplicated": "Visually deduplicated anomalies",
        }
        stage_label = stage_labels.get(stage, stage.title())
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

    @router.post("/{result_id}/postprocess/{workflow_id}/{stage}/edits")
    async def save_layer_edits(
        result_id: str,
        workflow_id: str,
        stage: str,
        request: EditLayerRequest,
    ) -> dict[str, Any]:
        result_dir = resolve_result(result_id)
        workflow_dir = resolve_workflow(result_dir, workflow_id)
        editable_stages = {"combined", "regularized", "solar_rows", "overlap_deduplicated", "deduplicated", "associated"}
        if stage not in editable_stages:
            raise HTTPException(status_code=400, detail="This layer cannot be edited.")
        geojson = request.geojson
        features = geojson.get("features") if isinstance(geojson, dict) else None
        if geojson.get("type") != "FeatureCollection" or not isinstance(features, list):
            raise HTTPException(status_code=400, detail="Edited data must be a GeoJSON FeatureCollection.")
        if len(features) > 500_000:
            raise HTTPException(status_code=400, detail="Edited GeoJSON contains too many features.")
        current = read_status(workflow_dir)
        output = (current.get("outputs") or {}).get(stage) or {}
        raw_path = str(output.get("path") or "")
        if not raw_path:
            raise HTTPException(status_code=404, detail="The selected layer is not available.")
        output_path = (result_dir / raw_path).resolve()
        try:
            output_path.relative_to(workflow_dir)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid editable layer path.") from exc
        if not output_path.is_file():
            raise HTTPException(status_code=404, detail="The selected layer file was not found.")
        cleaned_features = []
        for index, feature in enumerate(features):
            if not isinstance(feature, dict) or feature.get("type") != "Feature":
                raise HTTPException(status_code=400, detail=f"Feature {index} is invalid.")
            geometry = feature.get("geometry") or {}
            if geometry.get("type") not in {"Polygon", "MultiPolygon"} or not geometry.get("coordinates"):
                raise HTTPException(status_code=400, detail=f"Feature {index} is not a polygon.")
            cleaned = dict(feature)
            properties = dict(cleaned.get("properties") or {})
            if stage == "solar_rows":
                for key in {"row_id", "inner_row", "panel_number", "panel_id", "row_panel_count", "inner_row_panel_count", "source_panel_index"}:
                    properties.pop(key, None)
            properties["manually_edited"] = True
            cleaned["properties"] = properties
            cleaned_features.append(cleaned)
        try:
            existing_payload = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_payload = {}
        saved_payload = {
            key: value for key, value in existing_payload.items()
            if key not in {"type", "features"}
        }
        saved_payload.update({"type": "FeatureCollection", "features": cleaned_features})
        _atomic_json(output_path, saved_payload)
        manual_edits = dict(current.get("manual_edits") or {})
        manual_edits[stage] = {
            "feature_count": len(cleaned_features),
            "updated_at": datetime.now().isoformat(),
        }
        if stage == "solar_rows":
            regularized = (current.get("outputs") or {}).get("regularized") or {}
            regularized_path = str(regularized.get("path") or "")
            if regularized_path:
                clear_panel_ids((result_dir / regularized_path).resolve())
        update_status(
            workflow_dir,
            status="complete",
            stage="manual_edit",
            progress=100,
            message=f"{stage.replace('_', ' ').title()} GeoJSON updated.",
            manual_edits=manual_edits,
            assignment_stats=None if stage == "solar_rows" else current.get("assignment_stats"),
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
