from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Form, HTTPException

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent

router = APIRouter()

COLMAP_JOBS: Dict[str, Dict[str, Any]] = {}
COLMAP_MAX_LOGS = 400
COLMAP_PROGRESS_RE = re.compile(r"Processed file \[(\d+)/(\d+)\]")
COLMAP_STAGE_WEIGHTS: List[Tuple[str, float]] = [
    ("feature_extractor", 0.45),
    ("matcher", 0.2),
    ("mapper", 0.3),
    ("model_converter", 0.05),
]
COLMAP_STAGE_WEIGHT_LOOKUP = {label: weight for label, weight in COLMAP_STAGE_WEIGHTS}

_dep_get_project_test_dir: Optional[Callable[..., Path]] = None
_dep_get_project_colmap_dir: Optional[Callable[..., Path]] = None
_dep_now_stamp: Optional[Callable[[], str]] = None
_dep_is_image: Optional[Callable[[Path], bool]] = None
_dep_build_camera_meta_from_exif: Optional[Callable[[Path], Dict[str, Dict[str, Any]]]] = None
_dep_scan_image_sizes: Optional[Callable[[Path], Dict[str, Tuple[int, int]]]] = None
_dep_lookup_camera_meta_entry: Optional[Callable[[Dict[str, Dict[str, Any]], str], Optional[Dict[str, Any]]]] = None


def configure_colmap_dependencies(*,
    get_project_test_dir: Callable[..., Path],
    get_project_colmap_dir: Callable[..., Path],
    now_stamp: Callable[[], str],
    is_image: Callable[[Path], bool],
    build_camera_meta_from_exif: Callable[[Path], Dict[str, Dict[str, Any]]],
    scan_image_sizes: Callable[[Path], Dict[str, Tuple[int, int]]],
    lookup_camera_meta_entry: Callable[[Dict[str, Dict[str, Any]], str], Optional[Dict[str, Any]]],
) -> None:
    """Inject project-aware helpers so this module stays decoupled from app.py."""
    global _dep_get_project_test_dir, _dep_get_project_colmap_dir
    global _dep_now_stamp, _dep_is_image
    global _dep_build_camera_meta_from_exif, _dep_scan_image_sizes, _dep_lookup_camera_meta_entry
    _dep_get_project_test_dir = get_project_test_dir
    _dep_get_project_colmap_dir = get_project_colmap_dir
    _dep_now_stamp = now_stamp
    _dep_is_image = is_image
    _dep_build_camera_meta_from_exif = build_camera_meta_from_exif
    _dep_scan_image_sizes = scan_image_sizes
    _dep_lookup_camera_meta_entry = lookup_camera_meta_entry


def _require_dep(dep, name: str):
    if dep is None:
        raise RuntimeError(f"COLMAP dependency '{name}' is not configured. Call configure_colmap_dependencies() first.")
    return dep


def _project_test_dir(*args, **kwargs) -> Path:
    func = _require_dep(_dep_get_project_test_dir, "get_project_test_dir")
    return func(*args, **kwargs)


def _project_colmap_dir(*args, **kwargs) -> Path:
    func = _require_dep(_dep_get_project_colmap_dir, "get_project_colmap_dir")
    return func(*args, **kwargs)


def _now_stamp_fn() -> str:
    func = _require_dep(_dep_now_stamp, "_now_stamp")
    return func()


def _is_image_file(path: Path) -> bool:
    func = _require_dep(_dep_is_image, "_is_image")
    return func(path)


def _build_camera_meta_from_exif_dep(images_dir: Path) -> Dict[str, Dict[str, Any]]:
    func = _require_dep(_dep_build_camera_meta_from_exif, "_build_camera_meta_from_exif")
    return func(images_dir)


def _scan_image_sizes_dep(images_dir: Path) -> Dict[str, Tuple[int, int]]:
    func = _require_dep(_dep_scan_image_sizes, "_scan_image_sizes")
    return func(images_dir)


def _lookup_camera_meta_entry_dep(camera_meta: Dict[str, Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    func = _require_dep(_dep_lookup_camera_meta_entry, "_lookup_camera_meta_entry")
    return func(camera_meta, name)

# --- Accurate location (COLMAP) helpers ---

def _colmap_dataset_dir(dataset: str) -> Path:
    test_dir = _project_test_dir()
    ds_dir = test_dir / dataset
    if not ds_dir.exists() or not ds_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset}' not found.")
    return ds_dir


def _colmap_meta_path(dataset: str) -> Path:
    return _project_colmap_dir() / dataset / "colmap_meta.json"


def _colmap_ready_path(dataset: str) -> Path:
    return _project_colmap_dir() / dataset / "ready.json"


def _load_colmap_meta(dataset: str) -> Dict[str, Any]:
    mp = _colmap_meta_path(dataset)
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _merge_optical_metadata(
    base_thermal_meta: Dict[str, Any],
    optimization_project: str,
) -> Dict[str, Any]:
    """
    Merge optical project geometry (rotation, gimbal_yaw, heading, lat, lon)
    into thermal metadata by matching image names (strip _T/_V suffixes).
    """
    logger = logging.getLogger("pvrt.test")
    merged = dict(base_thermal_meta)
    
    optical_meta = _load_colmap_meta(optimization_project)
    if not optical_meta:
        logger.warning(f"UI:WARN:test: optimization_project '{optimization_project}' has no colmap_meta.json")
        return merged
    
    def normalize_basename(fname: str) -> str:
        """Remove _T, _V, -T, -V suffixes from filename"""
        base = fname.rsplit(".", 1)[0] if "." in fname else fname
        for suffix in ["_T", "_V", "-T", "-V"]:
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        return base
    
    optical_index = {}
    for fname, entry in optical_meta.items():
        if fname.startswith("__") or not isinstance(entry, dict):
            continue
        optical_index[normalize_basename(fname)] = entry
    
    matched_count = 0
    total_thermal = sum(1 for k in merged.keys() if not k.startswith("__"))
    
    for fname in list(merged.keys()):
        if fname.startswith("__"):
            continue
        norm = normalize_basename(fname)
        if norm in optical_index:
            optical_entry = optical_index[norm]
            thermal_entry = merged[fname]
            for field in ["rotation", "gimbal_yaw", "heading", "lat", "lon", "latitude", "longitude"]:
                if field in optical_entry:
                    thermal_entry[field] = optical_entry[field]
            matched_count += 1
    
    match_pct = (matched_count * 100 // total_thermal) if total_thermal > 0 else 0
    
    logger.info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
    logger.info(f"UI:INFO:test: Optical/Thermal Image Matching Summary:")
    logger.info(f"UI:INFO:test:   - Thermal images: {total_thermal}")
    logger.info(f"UI:INFO:test:   - Optical project: {optimization_project}")
    logger.info(f"UI:INFO:test:   - Matched images: {matched_count}/{total_thermal} ({match_pct}%)")
    
    if match_pct < 80:
        logger.warning(f"UI:WARN:test: Match percentage ({match_pct}%) is below 80% threshold")
        logger.warning(f"UI:WARN:test: Reverting to standard EXIF metadata (non-accurate poses)")
        logger.info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
        raise ValueError(f"Insufficient match rate: only {matched_count}/{total_thermal} images ({match_pct}%) matched. Need ≥80% for optical sync.")
    
    logger.info(f"UI:INFO:test: ✓ Using accurate poses from optical project '{optimization_project}'")
    logger.info(f"UI:INFO:test: ══════════════════════════════════════════════════════")
    
    meta_info = merged.setdefault("__meta__", {})
    if isinstance(meta_info, dict):
        meta_info["source"] = "thermal_exif+optical_geometry"
        meta_info["optimization_project"] = optimization_project
        meta_info["match_percent"] = match_pct
    
    return merged


def _colmap_ready(dataset: str) -> bool:
    rp = _colmap_ready_path(dataset)
    if not rp.exists():
        return False
    try:
        data = json.loads(rp.read_text(encoding="utf-8"))
        return bool(data.get("finalized"))
    except Exception:
        return False


def _set_colmap_ready(dataset: str, job_id: str):
    rp = _colmap_ready_path(dataset)
    rp.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "finalized": True,
        "job_id": job_id,
        "finished_at": _now_stamp_fn(),
    }
    rp.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_colmap_ready(dataset: str):
    rp = _colmap_ready_path(dataset)
    try:
        if rp.exists():
            rp.unlink()
    except Exception:
        pass


def _dataset_image_list(dataset: str) -> List[Path]:
    return [path for path, _ in _colmap_image_sources(dataset)]


def _colmap_image_sources(dataset: str) -> List[Tuple[Path, str]]:
    ds_dir = _colmap_dataset_dir(dataset)
    records: List[Tuple[Path, str]] = []
    for entry in sorted(ds_dir.iterdir()):
        if not entry.is_file() or not _is_image_file(entry):
            continue
        rel = entry.relative_to(ds_dir).as_posix()
        records.append((entry, rel))
    records.sort(key=lambda pair: pair[1])
    return records


def _gather_colmap_cameras(dataset: str) -> Dict[str, Dict[str, Any]]:
    ds_dir = _colmap_dataset_dir(dataset)
    try:
        camera_meta = _build_camera_meta_from_exif_dep(ds_dir)
    except Exception:
        camera_meta = {}
    sizes_index = _scan_image_sizes_dep(ds_dir)
    meta_lookup = _load_colmap_meta(dataset)
    ready_flag = _colmap_ready(dataset)

    cameras: Dict[str, Dict[str, Any]] = {}
    for img_path in _dataset_image_list(dataset):
        key = img_path.name
        entry = _lookup_camera_meta_entry_dep(camera_meta, key) if camera_meta else None
        lat = float(entry.get("lat")) if entry and entry.get("lat") is not None else None
        lon = float(entry.get("lon")) if entry and entry.get("lon") is not None else None
        alt = float(entry.get("alt")) if entry and entry.get("alt") is not None else None
        w_px = int(entry.get("w_px")) if entry and entry.get("w_px") is not None else None
        h_px = int(entry.get("h_px")) if entry and entry.get("h_px") is not None else None
        mpp = float(entry.get("meters_per_pixel")) if entry and entry.get("meters_per_pixel") is not None else None

        size = sizes_index.get(key)
        if size:
            w_px = w_px or int(size[0])
            h_px = h_px or int(size[1])

        cameras[key] = {
            "file": key,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "w": w_px,
            "h": h_px,
            "meters_per_pixel": mpp,
            "rotation": entry.get("rotation") if entry else None,
            "rotation_gimbal": entry.get("rotation_gimbal") if entry else None,
            "rotation_aircraft": entry.get("rotation_aircraft") if entry else None,
            "rotation_source": entry.get("rotation_source") if entry else None,
            "status": "pending",
            "calibrated": False,
            "has_gps": lat is not None and lon is not None,
            "optimized": None,
        }

        meta_entry = meta_lookup.get(key) or meta_lookup.get(Path(key).name, None)
        if isinstance(meta_entry, dict):
            cameras[key]["optimized"] = {
                "lat": meta_entry.get("lat"),
                "lon": meta_entry.get("lon"),
                "alt": meta_entry.get("alt"),
                "rotation": meta_entry.get("rotation"),
                "heading": meta_entry.get("rotation"),
            }
            cameras[key]["status"] = "calibrated" if ready_flag else "optimized"
            cameras[key]["calibrated"] = True if ready_flag else False

    return cameras


def _colmap_job_summary(job: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not job:
        return None
    return {
        "id": job.get("id"),
        "status": job.get("status"),
        "dataset": job.get("dataset"),
        "current_step": job.get("current_step"),
        "error": job.get("error"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "params": job.get("params"),
        "progress": job.get("progress"),
        "progress_detail": job.get("progress_detail"),
        "total_images": job.get("total_images"),
    }


def _begin_colmap_stage(job: Dict[str, Any], label: str, offset: float = 0.0):
    weight = COLMAP_STAGE_WEIGHT_LOOKUP.get(label, 0.0)
    job["current_step"] = label
    job["progress_offset"] = offset
    job["progress_stage_weight"] = weight
    job["progress"] = offset
    if label != "feature_extractor":
        job.pop("progress_detail", None)


def _finish_colmap_stage(job: Dict[str, Any]):
    offset = job.get("progress_offset", 0.0)
    weight = job.get("progress_stage_weight", 0.0)
    job["progress"] = max(0.0, min(1.0, offset + weight))
    if job.get("current_step") != "feature_extractor":
        job.pop("progress_detail", None)


def _append_colmap_log(job: Dict[str, Any], line: str):
    line = line.strip()
    if not line:
        return
    logs = job.setdefault("logs", [])
    logs.append(line)
    if len(logs) > COLMAP_MAX_LOGS:
        del logs[: len(logs) - COLMAP_MAX_LOGS]
    logging.getLogger("pvrt.colmap").info(line)

    match = COLMAP_PROGRESS_RE.search(line)
    if match:
        done = int(match.group(1))
        total = int(match.group(2)) or 1
        offset = job.get("progress_offset", 0.0)
        weight = job.get("progress_stage_weight", 0.0) or 0.0
        job["progress"] = max(0.0, min(1.0, offset + weight * (done / total)))
        job["progress_detail"] = {"done": done, "total": total}


def _colmap_state(dataset: str) -> Dict[str, Any]:
    cameras = _gather_colmap_cameras(dataset)
    job = COLMAP_JOBS.get(dataset)
    if job:
        cam_state = job.get("cameras", {})
        for key, state in cam_state.items():
            base = cameras.setdefault(key, {"file": key})
            for field, value in state.items():
                base[field] = value
    ready_flag = _colmap_ready(dataset)
    meta_path = _colmap_meta_path(dataset)
    meta_ref = str(meta_path) if meta_path.exists() else None
    return {
        "dataset": dataset,
        "ready": bool(ready_flag),
        "meta_path": meta_ref,
        "cameras": list(cameras.values()),
        "job": _colmap_job_summary(job),
        "logs": (job or {}).get("logs", [])[-100:],
    }


def _colmap_binary() -> str:
    candidates: List[str] = []
    env_bin = os.environ.get("COLMAP_BIN")
    if env_bin:
        candidates.append(env_bin)
    env_home = os.environ.get("COLMAP_HOME")
    if env_home:
        candidates.extend([
            env_home,
            str(Path(env_home) / "colmap"),
            str(Path(env_home) / "bin" / "colmap"),
        ])
    repo_colmap = PROJECT_ROOT / "colmap"
    candidates.extend([
        shutil.which("colmap"),
        str(repo_colmap),
        str(repo_colmap / "colmap"),
        str(repo_colmap / "bin" / "colmap"),
    ])

    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        if p.is_dir():
            p = p / "colmap"
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    raise HTTPException(status_code=400, detail="COLMAP binary not found. Set COLMAP_BIN or install COLMAP to use accurate locations.")


async def _run_colmap_command(job: Dict[str, Any], label: str, args: List[str], cwd: Optional[Path] = None) -> None:
    job["current_step"] = label
    job.setdefault("logs", [])
    display_cmd = " ".join(shlex.quote(str(a)) for a in args)
    _append_colmap_log(job, f"[{label}] $ {display_cmd}")

    def _runner():
        env = os.environ.copy()
        # Force Qt to run headless so COLMAP binaries do not require an X server.
        env.setdefault("QT_QPA_PLATFORM", "offscreen")
        proc = subprocess.Popen(
            [str(a) for a in args],
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                _append_colmap_log(job, line)
        finally:
            proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"{label} failed with exit code {proc.returncode}")

    await asyncio.to_thread(_runner)


async def _colmap_pipeline(job: Dict[str, Any]) -> None:
    dataset = job.get("dataset")
    if not dataset:
        job["status"] = "failed"
        job["error"] = "Dataset missing"
        return

    job["status"] = "running"
    job.setdefault("logs", [])
    job["started_at"] = job.get("started_at") or _now_stamp_fn()
    job["progress"] = job.get("progress", 0.0)
    job["progress_offset"] = 0.0
    job["progress_stage_weight"] = 0.0

    ds_dir = _colmap_dataset_dir(dataset)
    base_dir = _project_colmap_dir() / dataset
    job_dir = base_dir / job["id"]
    job_dir.mkdir(parents=True, exist_ok=True)
    job["work_dir"] = str(job_dir)
    database_path = job_dir / "database.db"
    sparse_dir = job_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    text_dir = job_dir / "model-text"
    text_dir.mkdir(parents=True, exist_ok=True)

    params = job.get("params") or {}

    def _as_int(val: Any) -> Optional[int]:
        try:
            return int(val)
        except (TypeError, ValueError):
            return None

    def _as_float(val: Any) -> Optional[float]:
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def _as_bool(val: Any) -> Optional[bool]:
        if isinstance(val, str):
            return val.lower() in {"1", "true", "yes", "on"}
        if val is None:
            return None
        try:
            return bool(val)
        except Exception:
            return None

    matcher = str(params.get("matcher", "exhaustive")).lower()
    min_tri = params.get("min_triangulation_angle")
    try:
        min_tri = float(min_tri) if min_tri is not None else None
    except Exception:
        min_tri = None
    seq_overlap = params.get("seq_overlap")
    try:
        seq_overlap = int(seq_overlap) if seq_overlap is not None else None
    except Exception:
        seq_overlap = None
    max_image_size = params.get("max_image_size")
    try:
        max_image_size = int(max_image_size) if max_image_size is not None else None
        if max_image_size is not None and max_image_size <= 0:
            max_image_size = None
    except Exception:
        max_image_size = None
    camera_model = params.get("camera_model")
    if isinstance(camera_model, str):
        camera_model = camera_model.strip() or None
    else:
        camera_model = None
    peak_threshold = _as_float(params.get("peak_threshold"))
    edge_threshold = _as_float(params.get("edge_threshold"))
    max_num_features = _as_int(params.get("max_num_features"))
    num_threads = _as_int(params.get("num_threads"))
    exhaustive_block_size = _as_int(params.get("exhaustive_block_size"))
    ba_refine_focal_length = _as_bool(params.get("ba_refine_focal_length"))
    ba_refine_principal_point = _as_bool(params.get("ba_refine_principal_point"))
    ba_refine_extra_params = _as_bool(params.get("ba_refine_extra_params"))
    min_num_matches = _as_int(params.get("min_num_matches"))
    init_min_inliers = _as_int(params.get("init_min_num_inliers"))
    abs_pose_min_inliers = _as_int(params.get("abs_pose_min_num_inliers"))
    max_model_overlap = _as_int(params.get("max_model_overlap"))
    max_num_models = _as_int(params.get("max_num_models"))
    use_gpu = params.get("use_gpu")
    if isinstance(use_gpu, str):
        use_gpu = use_gpu.lower() in {"1", "true", "yes", "on"}
    elif use_gpu is not None:
        use_gpu = bool(use_gpu)

    image_sources = _colmap_image_sources(dataset)
    if not image_sources:
        raise RuntimeError("Dataset does not contain any supported image files for COLMAP.")
    image_list_path = job_dir / "colmap_images.txt"
    image_list_path.write_text("\n".join(rel for _, rel in image_sources), encoding="utf-8")
    job["total_images"] = len(image_sources)

    _clear_colmap_ready(dataset)

    try:
        colmap_bin = _colmap_binary()
        progress_offset = 0.0

        def _advance_stage(label: str):
            nonlocal progress_offset
            _begin_colmap_stage(job, label, progress_offset)

        def _complete_stage():
            nonlocal progress_offset
            _finish_colmap_stage(job)
            progress_offset = job.get("progress", progress_offset)

        _advance_stage("feature_extractor")
        feature_args = [
            colmap_bin,
            "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(ds_dir),
            "--image_list_path", str(image_list_path),
        ]
        if max_image_size is not None:
            feature_args.extend(["--SiftExtraction.max_image_size", str(max_image_size)])
        if peak_threshold is not None:
            feature_args.extend(["--SiftExtraction.peak_threshold", f"{peak_threshold}"])
        if edge_threshold is not None:
            feature_args.extend(["--SiftExtraction.edge_threshold", f"{edge_threshold}"])
        if max_num_features is not None:
            feature_args.extend(["--SiftExtraction.max_num_features", str(max_num_features)])
        if num_threads is not None and num_threads > 0:
            feature_args.extend(["--SiftExtraction.num_threads", str(num_threads)])
        if use_gpu is not None:
            feature_args.extend(["--SiftExtraction.use_gpu", "1" if use_gpu else "0"])
        if camera_model:
            feature_args.extend(["--ImageReader.camera_model", camera_model])
        await _run_colmap_command(job, "feature_extractor", feature_args)
        _complete_stage()

        matcher_cmd = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
        matcher_args = [
            colmap_bin,
            matcher_cmd,
            "--database_path", str(database_path),
        ]
        if matcher == "sequential" and seq_overlap is not None:
            matcher_args.extend(["--SequentialMatching.overlap", str(seq_overlap)])
        if matcher == "exhaustive" and exhaustive_block_size is not None:
            matcher_args.extend(["--ExhaustiveMatching.block_size", str(exhaustive_block_size)])

        _advance_stage("matcher")
        await _run_colmap_command(job, matcher_cmd, matcher_args)
        _complete_stage()

        _advance_stage("mapper")
        mapper_args = [
            colmap_bin,
            "mapper",
            "--database_path", str(database_path),
            "--image_path", str(ds_dir),
            "--output_path", str(sparse_dir),
        ]
        if min_tri is not None:
            mapper_args.extend(["--Mapper.filter_min_tri_angle", str(min_tri)])
        min_model_size = params.get("min_model_size")
        if min_model_size is not None:
            mapper_args.extend(["--Mapper.min_model_size", str(min_model_size)])
        if min_num_matches is not None:
            mapper_args.extend(["--Mapper.min_num_matches", str(min_num_matches)])
        if init_min_inliers is not None:
            mapper_args.extend(["--Mapper.init_min_num_inliers", str(init_min_inliers)])
        if abs_pose_min_inliers is not None:
            mapper_args.extend(["--Mapper.abs_pose_min_num_inliers", str(abs_pose_min_inliers)])
        if max_model_overlap is not None:
            mapper_args.extend(["--Mapper.max_model_overlap", str(max_model_overlap)])
        if max_num_models is not None:
            mapper_args.extend(["--Mapper.max_num_models", str(max_num_models)])
        if ba_refine_focal_length is not None:
            mapper_args.extend(["--Mapper.ba_refine_focal_length", "1" if ba_refine_focal_length else "0"])
        if ba_refine_principal_point is not None:
            mapper_args.extend(["--Mapper.ba_refine_principal_point", "1" if ba_refine_principal_point else "0"])
        if ba_refine_extra_params is not None:
            mapper_args.extend(["--Mapper.ba_refine_extra_params", "1" if ba_refine_extra_params else "0"])
        
        # Optional: relaxed filtering for thermal imagery
        max_reproj_error = _as_float(params.get("max_reproj_error"))
        if max_reproj_error is not None:
            mapper_args.extend(["--Mapper.filter_max_reproj_error", str(max_reproj_error)])
        
        ba_global_max_iter = _as_int(params.get("ba_global_max_iterations"))
        if ba_global_max_iter is not None:
            mapper_args.extend(["--Mapper.ba_global_max_num_iterations", str(ba_global_max_iter)])
        
        ba_local_max_iter = _as_int(params.get("ba_local_max_iterations"))
        if ba_local_max_iter is not None:
            mapper_args.extend(["--Mapper.ba_local_max_num_iterations", str(ba_local_max_iter)])

        await _run_colmap_command(job, "mapper", mapper_args)
        _complete_stage()

        model_dirs = [p for p in sparse_dir.iterdir() if p.is_dir()]
        if not model_dirs:
            raise RuntimeError("COLMAP mapper produced no sparse models.")
        
        # Pick the largest model (most registered images)
        def _model_size(model_path: Path) -> int:
            images_bin = model_path / "images.bin"
            if not images_bin.exists():
                return 0
            # Rough estimate: file size / typical bytes per image entry
            return images_bin.stat().st_size
        
        model_dirs.sort(key=_model_size, reverse=True)
        model_dir = model_dirs[0]

        _advance_stage("model_converter")
        converter_args = [
            colmap_bin,
            "model_converter",
            "--input_path", str(model_dir),
            "--output_path", str(text_dir),
            "--output_type", "TXT",
        ]
        await _run_colmap_command(job, "model_converter", converter_args)
        _complete_stage()

        images_txt = text_dir / "images.txt"
        solution = _parse_colmap_images_txt(images_txt)
        if not solution:
            raise RuntimeError("COLMAP images.txt empty; no calibrated cameras recorded.")

        alignment = _align_colmap_solution(dataset, solution)
        meta_path = _write_colmap_meta(dataset, solution, alignment)
        job["meta_path"] = str(meta_path)

        # Read back the written meta to get calibrated rotation values
        colmap_meta = _load_colmap_meta(dataset)
        
        job_cams: Dict[str, Dict[str, Any]] = {}
        for img in _dataset_image_list(dataset):
            name = img.name
            aligned = alignment.get(name)
            meta_entry = colmap_meta.get(name)
            optimized = None
            if aligned and meta_entry:
                optimized = {
                    "lat": meta_entry.get("lat"),
                    "lon": meta_entry.get("lon"),
                    "alt": meta_entry.get("alt"),
                    "rotation": meta_entry.get("rotation"),  # Use calibrated rotation from meta
                    "aligned_center": aligned.get("aligned_center"),
                    "scale": aligned.get("scale"),
                }
            job_cams[name] = {
                "file": name,
                "status": "calibrated" if aligned else "pending",
                "calibrated": bool(aligned),
                "optimized": optimized,
            }
        job["cameras"] = job_cams

        job["status"] = "awaiting_finish"
        job["current_step"] = None
        job["progress"] = 1.0
        job["progress_detail"] = {
            "done": job.get("total_images", 0),
            "total": job.get("total_images", 0),
        }
        job["finished_at"] = _now_stamp_fn()
        _append_colmap_log(job, "COLMAP optimization complete. Click Finish to accept the results.")
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["error"] = "cancelled"
        job["finished_at"] = _now_stamp_fn()
        _append_colmap_log(job, "COLMAP optimization cancelled.")
        raise
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        job["finished_at"] = _now_stamp_fn()
        _append_colmap_log(job, f"ERROR: {exc}")
        return


# ================== Accurate location API ==================


def _parse_colmap_params(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid params JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="params must be a JSON object")
    return data


def _should_poll(job: Optional[Dict[str, Any]]) -> bool:
    if not job:
        return False
    return job.get("status") in {"queued", "running", "awaiting_finish"}


@router.get("/api/colmap/state")
async def api_colmap_state(dataset: str):
    _colmap_dataset_dir(dataset)
    state = _colmap_state(dataset)
    return {"ok": True, "state": state}


@router.get("/api/colmap/cameras")
async def api_colmap_cameras(dataset: str, page: int = 0, limit: int = 50):
    """Paginated cameras list for dataset."""
    _colmap_dataset_dir(dataset)
    cameras = _gather_colmap_cameras(dataset)
    job = COLMAP_JOBS.get(dataset)
    if job:
        cam_state = job.get("cameras", {})
        for key, state in cam_state.items():
            base = cameras.setdefault(key, {"file": key})
            # Don't overwrite optimized.rotation - use the corrected gimbal+aircraft from metadata
            if "optimized" in state and state["optimized"] is not None:
                # Only merge status/calibrated from job, keep metadata rotation
                if "optimized" not in base:
                    base["optimized"] = {}
                if isinstance(base["optimized"], dict):
                    base["optimized"].update({k: v for k, v in state["optimized"].items() if k != "rotation"})
                base["status"] = state.get("status")
                base["calibrated"] = state.get("calibrated")
            else:
                for field, value in state.items():
                    base[field] = value
    
    all_cameras = list(cameras.values())
    total = len(all_cameras)
    
    # Pagination
    start = max(0, page * limit)
    end = start + limit
    paginated = all_cameras[start:end]
    
    return {
        "ok": True,
        "total": total,
        "page": page,
        "limit": limit,
        "cameras": paginated,
        "has_more": end < total,
    }


@router.post("/api/colmap/start")
async def api_colmap_start(dataset: str = Form(...), params: str = Form(default=""), confirm_reset: bool = Form(default=False)):
    _colmap_dataset_dir(dataset)
    existing = COLMAP_JOBS.get(dataset)
    if existing and _should_poll(existing):
        raise HTTPException(status_code=409, detail="COLMAP optimization is already running for this dataset.")

    # Always clear previous results/cached state on each start; frontend handles user confirmation.
    base_dir = _project_colmap_dir() / dataset
    if base_dir.exists():
        try:
            shutil.rmtree(base_dir)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to clear previous COLMAP results: {exc}")
    if dataset in COLMAP_JOBS:
        COLMAP_JOBS.pop(dataset, None)

    job_id = uuid.uuid4().hex[:12]
    job: Dict[str, Any] = {
        "id": job_id,
        "dataset": dataset,
        "status": "queued",
        "params": _parse_colmap_params(params),
        "created_at": _now_stamp_fn(),
        "started_at": None,
        "finished_at": None,
        "logs": [],
        "current_step": None,
        "progress": 0.0,
        "progress_detail": None,
        "progress_offset": 0.0,
        "progress_stage_weight": 0.0,
        "total_images": 0,
    }
    COLMAP_JOBS[dataset] = job

    async def _runner():
        try:
            await _colmap_pipeline(job)
        finally:
            job.pop("task", None)

    task = asyncio.create_task(_runner())
    job["task"] = task

    state = _colmap_state(dataset)
    return {"ok": True, "job": _colmap_job_summary(job), "state": state}


@router.post("/api/colmap/finish")
async def api_colmap_finish(dataset: str = Form(...), job_id: str = Form(...)):
    _colmap_dataset_dir(dataset)
    job = COLMAP_JOBS.get(dataset)
    meta_path = _colmap_meta_path(dataset)

    # If backend was reloaded, the in-memory job is gone; allow finalization if metadata exists
    if not meta_path.exists():
        raise HTTPException(status_code=400, detail="COLMAP metadata missing; rerun optimization.")

    if job:
        if job.get("id") != job_id:
            raise HTTPException(status_code=404, detail="COLMAP job not found for dataset.")
        if job.get("status") != "awaiting_finish":
            raise HTTPException(status_code=400, detail="Job is not ready to finalize.")
    else:
        # Recreate minimal job to log and return state
        job = {
            "id": job_id,
            "dataset": dataset,
            "status": "awaiting_finish",
            "logs": [],
            "meta_path": str(meta_path),
        }
        COLMAP_JOBS[dataset] = job

    _set_colmap_ready(dataset, job_id)
    job["status"] = "finalized"
    job["current_step"] = None
    job["finalized_at"] = _now_stamp_fn()
    _append_colmap_log(job, "Dataset marked ready for accurate locations.")
    state = _colmap_state(dataset)
    return {"ok": True, "state": state}


def _quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    # Normalize quaternion so qw >= 0 (removes 180° ambiguity in COLMAP output).
    if qw < 0:
        qw, qx, qy, qz = -qw, -qx, -qy, -qz
    
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-8:
        return np.eye(3)
    s = 2.0 / n
    x, y, z = qx, qy, qz
    w = qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - s * (yy + zz),     s * (xy - wz),     s * (xz + wy)],
        [    s * (xy + wz), 1 - s * (xx + zz),     s * (yz - wx)],
        [    s * (xz - wy),     s * (yz + wx), 1 - s * (xx + yy)],
    ], dtype=float)


def _rotation_matrix_to_heading(rot: np.ndarray) -> float:
    try:
        # Extract yaw (heading) from rotation matrix using ZYX Euler angle convention.
        # For a rotation matrix R from ZYX Euler angles:
        # yaw = atan2(R[1,0], R[0,0])
        heading = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
        while heading > 180.0:
            heading -= 360.0
        while heading < -180.0:
            heading += 360.0
        return heading
    except Exception:
        return 0.0


def _latlon_to_local(lat: float, lon: float, alt: float, origin_lat: float, origin_lon: float, origin_alt: float) -> np.ndarray:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))
    east = (lon - origin_lon) * meters_per_deg_lon
    north = (lat - origin_lat) * meters_per_deg_lat
    up = (alt - origin_alt)
    return np.array([east, north, up], dtype=float)


def _local_to_latlon(east: float, north: float, origin_lat: float, origin_lon: float) -> Tuple[float, float]:
    meters_per_deg_lat = 111_320.0
    meters_per_deg_lon = 111_320.0 * math.cos(math.radians(origin_lat))
    lat = origin_lat + (north / meters_per_deg_lat)
    lon = origin_lon + (east / meters_per_deg_lon)
    return lat, lon


def _umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    n = src.shape[0]
    mean_src = src.mean(axis=0)
    mean_dst = dst.mean(axis=0)
    src_demean = src - mean_src
    dst_demean = dst - mean_dst
    cov = (src_demean.T @ dst_demean) / float(n)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_src = np.sum(src_demean ** 2) / float(n)
    scale = np.trace(np.diag(D) @ S) / var_src if var_src > 0 else 1.0
    t = mean_dst - scale * R @ mean_src
    return scale, R, t


def _parse_colmap_images_txt(images_txt: Path) -> Dict[str, Dict[str, Any]]:
    data: Dict[str, Dict[str, Any]] = {}
    if not images_txt.exists():
        return data
    with images_txt.open("r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    idx = 0
    ln = len(lines)
    while idx < ln:
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 10:
            continue
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        rot = _quaternion_to_matrix(qw, qx, qy, qz)
        center = (-rot.T @ np.array([tx, ty, tz])).tolist()
        data[name] = {
            "id": image_id,
            "camera_id": cam_id,
            "name": name,
            "qvec": (qw, qx, qy, qz),
            "tvec": (tx, ty, tz),
            "rot": rot,
            "center": center,
        }
        # Skip the following line containing 2D points (if present)
        if idx < ln:
            next_line = lines[idx].strip()
            if next_line and not next_line.startswith("#"):
                idx += 1
    return data


def _align_colmap_solution(dataset: str, solution: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    ds_dir = _colmap_dataset_dir(dataset)
    try:
        camera_meta = _build_camera_meta_from_exif_dep(ds_dir)
    except Exception as exc:
        raise RuntimeError(f"Failed to read EXIF metadata for alignment: {exc}")

    pairs = []
    for name, pose in solution.items():
        entry = _lookup_camera_meta_entry_dep(camera_meta, name)
        if not entry:
            continue
        lat = entry.get("lat")
        lon = entry.get("lon")
        if lat is None or lon is None:
            continue
        alt = float(entry.get("alt") or 0.0)
        pairs.append((name, pose, float(lat), float(lon), alt, entry))

    if len(pairs) < 3:
        raise RuntimeError("Need at least 3 GPS-tagged images for COLMAP alignment.")

    origin_lat = sum(p[2] for p in pairs) / len(pairs)
    origin_lon = sum(p[3] for p in pairs) / len(pairs)
    origin_alt = sum(p[4] for p in pairs) / len(pairs)

    gps_points = []
    colmap_points = []
    for _, pose, lat, lon, alt, _ in pairs:
        gps_points.append(_latlon_to_local(lat, lon, alt, origin_lat, origin_lon, origin_alt))
        colmap_points.append(np.array(pose["center"], dtype=float))

    src = np.stack(colmap_points)
    dst = np.stack(gps_points)
    scale, rot_align, trans = _umeyama(src, dst)

    results: Dict[str, Dict[str, Any]] = {}
    for name, pose in solution.items():
        center = np.array(pose["center"], dtype=float)
        aligned = scale * (rot_align @ center) + trans
        lat, lon = _local_to_latlon(aligned[0], aligned[1], origin_lat, origin_lon)
        alt = origin_alt + aligned[2]
        # Compute camera heading (yaw) using the provided quaternion→matrix convention
        # and maintain world→camera orientation while aligning to the GPS/local frame.
        # For alignment: R_wc_local = R_wc @ rot_align.T
        R_wc = np.array(pose["rot"], dtype=float)
        R_wc_local = R_wc @ rot_align.T
        heading = _rotation_matrix_to_heading(R_wc_local)
        results[name] = {
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "rotation": heading,
            "rotation_source": "colmap_aligned",
            "aligned_center": aligned.tolist(),
            "scale": scale,
        }
    return results


def _write_colmap_meta(dataset: str, solution: Dict[str, Dict[str, Any]], alignment: Dict[str, Dict[str, Any]]):
    ds_dir = _colmap_dataset_dir(dataset)
    try:
        camera_meta = _build_camera_meta_from_exif_dep(ds_dir)
    except Exception:
        camera_meta = {}
    sizes_index = _scan_image_sizes_dep(ds_dir)
    # --- Optional yaw calibration: compute dataset-level offset between aligned COLMAP yaw and EXIF yaw ---
    def _norm_deg(a: float) -> float:
        while a > 180.0:
            a -= 360.0
        while a < -180.0:
            a += 360.0
        return a

    diffs_rad = []
    for name, pose in solution.items():
        aligned = alignment.get(name) or {}
        colmap_yaw = aligned.get("rotation")
        base = _lookup_camera_meta_entry_dep(camera_meta, name) or {}
        exif_yaw = base.get("rotation_gimbal")
        if exif_yaw is None:
            exif_yaw = base.get("rotation_aircraft")
        if colmap_yaw is None or exif_yaw is None:
            continue
        diff = _norm_deg(float(colmap_yaw) - float(exif_yaw))
        diffs_rad.append(math.radians(diff))

    yaw_delta: Optional[float] = None
    if len(diffs_rad) >= 5:
        # circular mean of differences
        s = sum(math.sin(d) for d in diffs_rad)
        c = sum(math.cos(d) for d in diffs_rad)
        yaw_delta = _norm_deg(math.degrees(math.atan2(s, c)))

    meta_out: Dict[str, Dict[str, Any]] = {}
    for name, pose in solution.items():
        base = _lookup_camera_meta_entry_dep(camera_meta, name) or {}
        aligned = alignment.get(name)
        if not aligned:
            continue
        lat = aligned.get("lat")
        lon = aligned.get("lon")
        alt = aligned.get("alt")
        # Prefer the aligned COLMAP yaw; optionally calibrate to EXIF frame via dataset-level offset.
        heading = aligned.get("rotation")
        heading_source = aligned.get("rotation_source") or ("colmap_aligned" if heading is not None else None)
        if heading is not None and yaw_delta is not None:
            heading = _norm_deg(float(heading) - yaw_delta)
            heading_source = "colmap_aligned_calibrated"
        
        # Fallback to EXIF gimbal/aircraft yaw if aligned yaw not available
        if heading is None:
            gimbal_yaw = base.get("rotation_gimbal")
            if gimbal_yaw is not None:
                heading = float(gimbal_yaw)
                heading_source = "gimbal_yaw"
            else:
                aircraft_yaw = base.get("rotation_aircraft")
                if aircraft_yaw is not None:
                    heading = float(aircraft_yaw)
                    heading_source = "aircraft_yaw"
        
        entry = {
            "file": name,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "rotation": heading,
            "rotation_source": heading_source,
            "meters_per_pixel": base.get("meters_per_pixel") or 0.05,
            "w_px": base.get("w_px") or (sizes_index.get(name, (None, None))[0]),
            "h_px": base.get("h_px") or (sizes_index.get(name, (None, None))[1]),
            "source": "colmap",
            "qw": pose["qvec"][0],
            "qx": pose["qvec"][1],
            "qy": pose["qvec"][2],
            "qz": pose["qvec"][3],
            "tx": pose["tvec"][0],
            "ty": pose["tvec"][1],
            "tz": pose["tvec"][2],
        }
        meta_out[name] = entry

    if not meta_out:
        raise RuntimeError("COLMAP alignment produced no entries.")

    # Note: Rotation now prefers the aligned COLMAP yaw (rotation_source='colmap_aligned').
    # If unavailable, it falls back to EXIF gimbal/aircraft yaw.

    meta_path = _colmap_meta_path(dataset)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta_out, indent=2), encoding="utf-8")
    return meta_path


async def _exec_colmap_step(job: Dict[str, Any], step: str, cmd: List[str]):
    job["current_step"] = step
    _append_colmap_log(job, f"[STEP] {step}")
    # Force Qt to run headless so COLMAP binaries do not require an X server.
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        while True:
            if proc.stdout is None:
                break
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                decoded = line.decode("utf-8", errors="ignore").strip()
            except Exception:
                decoded = line.decode(errors="ignore").strip()
            _append_colmap_log(job, decoded)
        code = await proc.wait()
        if code != 0:
            raise RuntimeError(f"COLMAP step '{step}' failed with exit code {code}")
    finally:
        # StreamReader doesn't provide close(); rely on process lifecycle.
        pass


async def _run_colmap_job(job: Dict[str, Any]):
    dataset = job.get("dataset")
    job["status"] = "running"
    job["error"] = None
    job["finished_at"] = None
    try:
        colmap_bin = _colmap_binary()
        ds_dir = _colmap_dataset_dir(dataset)
        work_base = _project_colmap_dir() / dataset / job["id"]
        work_base.mkdir(parents=True, exist_ok=True)
        db_path = work_base / "database.db"
        sparse_dir = work_base / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        model_text_dir = work_base / "model_text"
        model_text_dir.mkdir(parents=True, exist_ok=True)
        job["workspace"] = str(work_base)

        params = job.get("params", {})
        camera_model = params.get("camera_model", "SIMPLE_RADIAL")
        max_image_size = params.get("max_image_size")
        use_gpu = bool(params.get("use_gpu", True))

        feature_cmd = [
            colmap_bin,
            "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(ds_dir),
            "--ImageReader.camera_model", str(camera_model),
            "--ImageReader.single_camera", "1",
        ]
        if max_image_size:
            feature_cmd += ["--SiftExtraction.max_image_size", str(max_image_size)]
        if use_gpu:
            feature_cmd += ["--SiftExtraction.use_gpu", "1"]
        await _exec_colmap_step(job, "feature_extractor", feature_cmd)

        matcher = str(params.get("matcher", "sequential")).lower()
        if matcher not in {"sequential", "exhaustive", "spatial"}:
            matcher = "sequential"
        matcher_cmd = [colmap_bin, f"{matcher}_matcher", "--database_path", str(db_path)]
        if matcher == "sequential":
            matcher_cmd += ["--SequentialMatching.overlap", str(params.get("seq_overlap", 3))]
        if matcher == "spatial":
            matcher_cmd += ["--SpatialMatching.max_num_neighbors", str(params.get("spatial_neighbors", 20))]
        await _exec_colmap_step(job, f"{matcher}_matcher", matcher_cmd)

        mapper_cmd = [
            colmap_bin,
            "mapper",
            "--database_path", str(db_path),
            "--image_path", str(ds_dir),
            "--output_path", str(sparse_dir),
            "--Mapper.min_num_matches", str(params.get("min_matches", 15)),
        ]
        if params.get("ba_iterations"):
            mapper_cmd += ["--Mapper.ba_global_max_num_iterations", str(params.get("ba_iterations"))]
        await _exec_colmap_step(job, "mapper", mapper_cmd)

        recon_dirs = sorted([p for p in sparse_dir.iterdir() if p.is_dir()])
        if not recon_dirs:
            raise RuntimeError("COLMAP mapper produced no reconstructions.")
        convert_cmd = [
            colmap_bin,
            "model_converter",
            "--input_path", str(recon_dirs[0]),
            "--output_path", str(model_text_dir),
            "--output_type", "TXT",
        ]
        await _exec_colmap_step(job, "model_converter", convert_cmd)

        images_txt = model_text_dir / "images.txt"
        solution = _parse_colmap_images_txt(images_txt)
        if not solution:
            raise RuntimeError("COLMAP produced no registered camera poses.")
        alignment = _align_colmap_solution(dataset, solution)
        meta_path = _write_colmap_meta(dataset, solution, alignment)
        job["meta_path"] = str(meta_path)
        
        # Read back the written meta to get calibrated rotation values
        colmap_meta = _load_colmap_meta(dataset)
        
        job.setdefault("cameras", {})
        for name, aligned in alignment.items():
            meta_entry = colmap_meta.get(name)
            cam = job["cameras"].setdefault(name, {"file": name})
            cam["optimized"] = {
                "lat": meta_entry.get("lat") if meta_entry else aligned.get("lat"),
                "lon": meta_entry.get("lon") if meta_entry else aligned.get("lon"),
                "alt": meta_entry.get("alt") if meta_entry else aligned.get("alt"),
                "rotation": meta_entry.get("rotation") if meta_entry else aligned.get("rotation"),  # Use calibrated rotation
            }
            cam["status"] = "optimized"
            cam["calibrated"] = False
        job["status"] = "complete"
        job["finished_at"] = _now_stamp_fn()
        _append_colmap_log(job, f"COLMAP job {job['id']} completed. Metadata written to {meta_path}.")
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["finished_at"] = _now_stamp_fn()
        _append_colmap_log(job, f"Error: {exc}")


__all__ = [
    "router",
    "configure_colmap_dependencies",
    "_colmap_ready",
    "_load_colmap_meta",
    "_merge_optical_metadata",
]
