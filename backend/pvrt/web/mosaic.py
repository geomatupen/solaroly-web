from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..dataops.mosaic_from_colmap import create_mosaic_from_rotated_images

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
ROTATION_SCRIPT = PROJECT_ROOT / "backend" / "pvrt" / "dataops" / "regenerate_geojson_from_preds.py"


@dataclass
class RotationResult:
    input_type: str
    run_images_dir: Path
    tiles_dir: Optional[Path]
    tif_src: Optional[Path]


def prepare_rotation_and_mosaic(
    *,
    input_type: str,
    session_dir: Path,
    out_root: Path,
    camera_meta: Dict[str, Any],
    mosaic_enabled: bool,
    ds_dir: Path,
    model_is_thermal: bool,
    undistort_thermal: bool,
    tile_tif_func: Callable[[Path, Path, int, Optional[int]], None],
    run_images_dir: Path,
    tiles_dir: Optional[Path],
    tif_src: Optional[Path],
    tile_size: int = 1024,
    tile_stride: Optional[int] = 1024,
) -> RotationResult:
    """Rotate images via regenerate script and optionally create/Tile a mosaic."""
    log = logging.getLogger("pvrt.test")
    log.info(
        "UI:INFO:test: Rotation check - mosaic_enabled=%s, input_type=%s, camera_meta_count=%s",
        mosaic_enabled,
        input_type,
        len(camera_meta) if camera_meta else 0,
    )

    if input_type != "images" or not camera_meta:
        return RotationResult(
            input_type=input_type,
            run_images_dir=run_images_dir,
            tiles_dir=tiles_dir,
            tif_src=tif_src,
        )

    try:
        log.info("UI:INFO:test: \u2713 Conditions met: Starting image rotation...")
        log.info("UI:INFO:test: session_dir=%s, out_root=%s", session_dir, out_root)

        cm_path = session_dir / "camera_meta.json"
        if not cm_path.exists():
            raise RuntimeError(f"camera_meta.json not found at {cm_path}")
        cm_size = cm_path.stat().st_size
        try:
            cm_json = json.loads(cm_path.read_text(encoding="utf-8"))
            cm_count = len([k for k in cm_json.keys() if not k.startswith("__")])
            log.info(
                "UI:INFO:test: \u2713 camera_meta.json exists (size=%s bytes, entries=%s)",
                cm_size,
                cm_count,
            )
        except Exception as exc:
            raise RuntimeError(f"camera_meta.json invalid: {exc}") from exc

        session_contents_before = sorted(x.name for x in session_dir.glob("*"))
        log.info("UI:INFO:test: session_dir before rotation: %s", session_contents_before)

        if ROTATION_SCRIPT.exists():
            log.info(
                "UI:INFO:test: Running regenerate script for rotation (session_dir=%s, src_images=%s, thermal=%s)",
                session_dir,
                ds_dir,
                model_is_thermal,
            )
            cmd = [sys.executable, str(ROTATION_SCRIPT), str(session_dir), str(ds_dir)]
            if model_is_thermal:
                cmd.append("--use-thermal")
            if undistort_thermal:
                cmd.append("--correct-lens-distortion")
                log.info("UI:INFO:test: Checking lens distortion before north-up rotation…")
                log.info(
                    "UI:INFO:test: Lens-correction rules: skip images already marked corrected; "
                    "skip calibrated displacement below 2.00 px; correct displacement at or above "
                    "2.00 px; stop if correction status or calibration cannot be determined."
                )

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=300,
            )
            log.info("UI:INFO:test: Rotation completed (exit=%s)", proc.returncode)
            if proc.stdout:
                correction_summary = None
                for line in proc.stdout.splitlines():
                    if not line.startswith("[undistort]"):
                        continue
                    decision = line.removeprefix("[undistort]").strip()
                    # Every per-image decision remains available in the full Logs tab.
                    log.info("Lens-correction decision: %s", decision)
                    if decision.startswith("Summary:"):
                        correction_summary = decision.removeprefix("Summary:").strip()
                # Keep the Test mini-log useful without flooding it with one row per image.
                if correction_summary:
                    log.info("UI:OK:test: Lens-correction decisions: %s", correction_summary)
            if proc.stderr:
                for line in proc.stderr.splitlines()[-10:]:
                    # The endpoint emits a concise UI error below; retain subprocess detail
                    # in the full log for diagnosis without flooding the mini-log.
                    log.warning("[rotation script] %s", line)
            if proc.returncode != 0:
                error_lines = [line.strip() for line in proc.stderr.splitlines() if line.strip()]
                helpful = next((line for line in error_lines if "Cannot correct lens distortion" in line), None)
                raise RuntimeError(helpful or (error_lines[-1] if error_lines else "Image preparation failed."))
            time.sleep(0.3)
        else:
            log.warning("UI:INFO:test: Script not found at %s", ROTATION_SCRIPT)

        rotated_images_dir = Path(session_dir / "rotated_images")
        rotated_files = list(rotated_images_dir.glob("*")) if rotated_images_dir.exists() else []
        log.info(
            "UI:INFO:test: rotated_images_dir=%s, exists=%s, file_count=%s",
            rotated_images_dir,
            rotated_images_dir.exists(),
            len(rotated_files),
        )

        session_contents_after = sorted(x.name for x in session_dir.glob("*"))
        log.info("UI:INFO:test: session_dir after rotation: %s", session_contents_after)

        if rotated_images_dir.exists() and rotated_files:
            if mosaic_enabled:
                mosaic_path = out_root / "mosaic.tif"
                log.info(
                    "UI:INFO:test: Creating mosaic from %s rotated images...",
                    len(rotated_files),
                )
                create_mosaic_from_rotated_images(
                    rotated_images_dir=rotated_images_dir,
                    out_mosaic_path=mosaic_path,
                    plane_z=0.0,
                    resolution=0.1,
                    camera_meta=camera_meta,
                )
                log.info("UI:INFO:test: \u2713 Mosaic created: %s", mosaic_path)
                tiles_dir = out_root / "tiles"
                log.info("UI:INFO:test: Tiling mosaic from %s to %s", mosaic_path, tiles_dir)
                tile_tif_func(mosaic_path, tiles_dir, tile_size=tile_size, stride=tile_stride)
                run_images_dir = tiles_dir
                input_type = "tif"
                tif_src = mosaic_path
                tile_count = len(list(tiles_dir.glob("*")))
                log.info(
                    "UI:INFO:test: \u2713 Mosaic tiled; running orthophoto pipeline on %s tiles",
                    tile_count,
                )
            else:
                run_images_dir = rotated_images_dir
                log.info(
                    "UI:INFO:test: \u2713 Using rotated images for per-image inference: %s",
                    run_images_dir,
                )
        else:
            log.warning(
                "\u2717 Rotated images not found or empty; proceeding with original images (alignment may be incorrect)"
            )
    except Exception as exc:
        log.warning("\u2717 Failed to generate rotation/mosaic: %s", exc)
        log.warning("Traceback: %s", traceback.format_exc())
        if undistort_thermal:
            raise

    return RotationResult(
        input_type=input_type,
        run_images_dir=run_images_dir,
        tiles_dir=tiles_dir,
        tif_src=tif_src,
    )
