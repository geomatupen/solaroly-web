# SolarOly Project

SolarOly turns DJI thermal flights and orthophotos into actionable solar anomaly reports. A FastAPI backend orchestrates thermal decoding, Detectron2/YOLO inference, COLMAP-assisted alignment, and data prep utilities, while a lightweight JS frontend handles uploads, training orchestration, and geospatial visualization.

---

## 1. Project Snapshot
- Multi-project workspace rooted at `backend/projects/<project_id>` with isolated train/test data, outputs, overlays, and logs per session.
- Dual inference engines (Detectron2 and YOLOv8) plus optional COLMAP pose refinement exposed through feature flags so heavy integrations stay opt-in.
- Server-Sent Events stream live logs for training, testing, mosaicking, and ZIP uploads directly into the browser.
- DJI Thermal SDK integration decodes 16-bit radiometric frames, preserves metadata, and falls back to RGB thermal renders when raw extraction is unavailable.
- Built-in tools regenerate GeoJSON, rotate/align frames, tile orthophotos, and rebuild thermal mosaics directly from COLMAP camera poses.

---

## 2. Installation Options

### 2.1 Local Virtual Environment (recommended for development)
1. Create and activate an isolated environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   python -m pip install --upgrade pip wheel
   ```
2. Decide which heavy integrations you need (Detectron2, or YOLO, COLMAP helpers). Use Section 3 to comment blocks in `requirements.txt` **before** installing.
3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Install native extras (COLMAP, GDAL, GPU drivers) if your workflow requires them.
5. Complete the DJI Thermal SDK steps in Section 4 so thermal decoding works end-to-end.

### 2.2 Docker Container (deployment / reproducibility)
1. Review the `Dockerfile` at the repository root. It installs the base requirements, Torch/Detectron2/YOLO, and exports the `PVRT_ENABLE_*` flags. Swap the base image (e.g., `nvidia/cuda`) or add apt packages if you need GPU builds or COLMAP inside the container.
2. Build and run:
   ```bash
   docker build -t solaroly-web .
   docker run --rm -p 8001:8001 \
     -e PVRT_ENABLE_DETECTRON=1 \
     -e PVRT_ENABLE_YOLO=1 \
     -e PVRT_ENABLE_COLMAP=0 \
     solaroly-web
   ```
3. Mount external datasets/models with `-v /host/path:/app/backend/projects` and override feature flags per run as needed.
4. Thermal SDK paths inside Docker must still point to `third_party/utility/bin/linux/release_x64/libdirp.so`. Export `DIRP_SDK_PATH` and `LD_LIBRARY_PATH` accordingly or bake them into the image.

---

## 3. Optional Integrations & Dependency Matrix
The committed `requirements.txt` lists *all* integrations. Comment out the lines you do not need **before** running `pip install -r requirements.txt`. Pair dependency changes with the matching feature flag so the UI hides unsupported workflows.

| Feature | Keep these requirements | Comment when unused | Runtime toggle |
| --- | --- | --- | --- |
| Shared PyTorch runtime (Detectron2 & YOLO) | `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121`, `triton==3.1.0` (swap in CPU wheels if you do not ship CUDA) | Only comment if you disable **both** backends entirely. | Required whenever either `PVRT_ENABLE_DETECTRON` or `PVRT_ENABLE_YOLO` is 1 |
| Detectron2-based training/testing | `detectron2 @ git+...`, `fvcore`, `iopath`, `hydra-core`, `omegaconf`, `yacs`, `pycocotools`, `tensorboard`, `tabulate`, `matplotlib` | Comment the entire block if you only plan to run YOLO. | `PVRT_ENABLE_DETECTRON=0/1` |
| YOLOv8 pipeline | `ultralytics>=8.3.0,<9.0.0` plus the shared OpenCV/numpy stack already present | Comment `ultralytics` if Detectron2 is the only backend you ship. | `PVRT_ENABLE_YOLO=0/1` |
| CUDA accelerators | All `nvidia-*` wheels plus CUDA-specific Torch builds | Comment GPU wheels and install CPU Torch (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`) when no CUDA-capable device exists. | Not flag-controlled; match your hardware |
| DJI Thermal SDK (optional) | `dji-thermal-sdk==0.0.2`, `opencv-python-headless`, `tifffile`, `piexif`, `exif` | Comment this block if you never ingest DJI R-JPEG/radiometric frames. Keep it to unlock grayscale thermal extraction, temperature metadata, and the COLMAP-driven thermal mosaic pipeline. | Export `DIRP_SDK_PATH`, `LD_LIBRARY_PATH` whenever installed |
| COLMAP-assisted alignment | Installed separately via `conda install -c conda-forge colmap` or system packages | Skip entirely if you only need per-frame inference. | `PVRT_ENABLE_COLMAP=0/1` |

**Comment example:**
```text
# --- Detectron2 stack (comment out to ship YOLO-only; keep the shared PyTorch runtime above) ---
# detectron2 @ git+https://github.com/facebookresearch/detectron2.git@a1ce2f9
# fvcore==0.1.5.post20221221
# iopath==0.1.9
# hydra-core==1.3.2
# omegaconf==2.3.0
# yacs==0.1.8
# pycocotools==2.0.10
# tensorboard==2.20.0
# tabulate==0.9.0
# matplotlib==3.10.6
# --- end Detectron2 stack ---
```
Match the flag to the dependency list—if `PVRT_ENABLE_DETECTRON=0`, the UI hides the Detectron forms and the backend never imports those packages.

---

## 4. DJI Thermal SDK Setup (only for DJI radiometric workflows)
1. **Native binaries:** The repo already vendors DJI Thermal SDK under `third_party/`. If you download a newer SDK, unzip it into the same folder so `third_party/tsdk-core/lib/linux/release_x64/libdirp.so` exists.
2. **Environment activation:** Append to `venv/bin/activate` (or export manually) so runtime paths are set every time the environment loads:
   ```bash
   # --- DJI Thermal SDK ---
   export DIRP_SDK_PATH="$PWD/third_party/utility/bin/linux/release_x64/libdirp.so"
   export LD_LIBRARY_PATH="$(dirname "$DIRP_SDK_PATH"):${LD_LIBRARY_PATH:-}"
   # --- end DJI Thermal SDK ---
   ```
3. **Python wrapper:**
   ```bash
   python -m pip install --upgrade dji-thermal-sdk
   python -m pip show dji-thermal-sdk  # verify 0.0.2+
   ```
4. **Smoke test:**
   ```python
   python - <<'PY'
   from dji_thermal_sdk.dji_sdk import dji_init
   from dji_thermal_sdk.utility import rjpeg_to_heatmap
   print("DJI SDK ready")
   PY
   ```
5. Reactivate the environment or restart Docker containers whenever you change the library paths.

> **When you can skip this:** If your datasets only contain RGB imagery or thermal renders from non-DJI drones (i.e., no DJI R-JPEG radiometric frames), you may comment the DJI SDK dependencies, skip this section, and simply leave the "Use thermal" checkbox unchecked in the UI. The backend will operate on RGB or externally supplied thermal PNG/JPEG files but will not attempt grayscale extraction or temperature calculations.

---

## 5. Configuration & Data Layout
- **Project tree:**
  ```
  backend/projects/<project_id>/
  ├── train/
  │   ├── data/
  │   │   ├── train/{images,thermal,_annotations.coco.json,labels}
  │   │   └── valid/{images,thermal,_annotations.coco.json,labels}
  │   └── outputs/<timestamped_session>/
  ├── test/
  │   ├── data/uploads/            # raw ZIPs, single images, or GeoTIFFs
  │   └── outputs/<session>/
  ├── overlays/overlay-*/          # saved GeoJSON or GeoTIFF overlays
  └── logs/
  ```
- **Configuration file:** Each project writes an `outputs/config.yaml` capturing the last-used backend, weights folder, score thresholds, and thermal extraction preferences. The UI overwrites this file whenever you submit the Training or Testing forms.
- **Feature flags:** `backend/pvrt/web/settings.py` reads `PVRT_ENABLE_DETECTRON`, `PVRT_ENABLE_YOLO`, and `PVRT_ENABLE_COLMAP`. Export them before launching uvicorn or set them inside Docker. Disabled integrations disappear from the UI and the backend guards routes accordingly.
- **Camera metadata:** `camera_meta.json` stores alignment hints (heading offsets, reference elevations) and is used by the rotation/mosaic pipeline plus the orthophoto tiler.

---

## 6. Running the Stack
1. Activate your environment and export feature flags:
   ```bash
   source venv/bin/activate
   export PVRT_ENABLE_DETECTRON=1
   export PVRT_ENABLE_YOLO=1
   export PVRT_ENABLE_COLMAP=0
   ```
2. Start FastAPI:
   ```bash
   uvicorn backend.pvrt.web.app:app --host 0.0.0.0 --port 8001 --reload
   ```
3. Visit `http://localhost:8001` for the frontend. The SPA serves test uploads, project selectors, training forms, SSE log panes, maps, overlays, and downloads directly from the backend.
4. Capture backend logs for debugging:
   ```bash
   uvicorn backend.pvrt.web.app:app --log-level debug > fastapi.log 2>&1
   tail -f fastapi.log
   ```

---

## 7. Processing Pipelines & Feature Details

### 7.1 Single DJI Frame Inference
1. **Upload** JPG/R-JPEG bundles under *Test → Uploads*. The server unpacks them into `test/data/uploads` for the active project.
2. **Thermal extraction / fallback:** With the DJI SDK installed, the platform can decode DJI R-JPEG files into 16-bit grayscale rasters and expose temperature metadata. If you skipped the SDK (non-DJI thermal cameras or RGB-only workflows), keep the "Use thermal" checkbox disabled in frontend when you train —the system will operate on RGB imagery or any pre-rendered thermal PNG/JPEG files you provide, but thermal data extraction, create grayscale and temperature readouts will be unavailable.
3. **Rotation & normalization:** Metadata-driven heading correction rotates each frame north-up. If rotation scripts fail or metadata is missing, the system continues with the original orientation but flags the session in the log.
4. **Inference:**
   - Detectron2: loads the configured `model_final.pth` under `train/outputs/<run>/weights/` and respects batch size/thresholds from the UI.
   - YOLOv8: uses Ultralytics models; per-image overlays are rendered with CV2, stored in `test/outputs/<session>/overlays`, and summarized via `raw_results_summary.json` and `metrics.json` (see `backend/pvrt/backends/yolo/infer.py`).
5. **Geo outputs:** Every run writes `preds/*.json`, `rotated_images/`, `images.geojson`, `anomalies.geojson`, overlays, and `test.log`. The frontend map tab consumes the GeoJSON directly and renders anomalies + camera markers.

### 7.2 Orthophoto Pipeline
1. Upload or reference a georeferenced orthophoto (single GeoTIFF) through *Test → Uploads*. The backend stores it under `test/data/uploads` for the active project.
2. The orthophoto worker (`backend/pvrt/web/mosaic.py`) splits the GeoTIFF into patches/tiles sized per the UI form and writes them to `test/outputs/<session>/tiles`.
3. Each tile runs through whichever backend you enabled (Detectron2 or YOLO). Because the tiles are already north-up and geo-aligned, **no COLMAP step is required**.
4. Post-processing reprojects detections into the orthophoto CRS, merges them into `images.geojson`/`anomalies.geojson`, and exposes a ready-to-download GeoTIFF via `/api/download_ortho`.
5. Overlay upload/download endpoints let you bring the orthophoto detections back into the map tab alongside manual overlays.

### 7.3 Training Pipeline
1. Organize data under `train/data/{train,valid}` in either COCO or YOLO format (both are supported simultaneously).
2. Choose backend + dataset + augmentation flags in the UI. Detectron2 jobs respect iteration counts, base LR, `num_classes`, and dataset names; YOLO jobs expose epochs, batch sizes, and image resolution.
3. Logs stream through SSE to the browser (`UI:INFO:train` messages) and persist into `train/outputs/<run>/train.log`. Cancel buttons terminate the worker gracefully.
4. Completed runs hold `model_meta.json`, weights, metrics, and preview overlays. Switching to Testing allows you to pick any run folder as the active weights directory.

### 7.4 COLMAP Workflows

#### 7.4.1 Pose Optimization for Single Images
- `PVRT_ENABLE_COLMAP=1` unlocks pose uploads inside the single-image testing form. COLMAP-derived intrinsics/extrinsics refine each camera’s latitude/longitude/heading before inference so detections line up with site plans even when EXIF yaw is noisy.
- Optimized poses live beside the session under `test/outputs/<session>/colmap_pose.json` and are referenced whenever rotated imagery, GeoJSON footprints, or overlays are generated.
- If COLMAP stays disabled, the backend simply relies on EXIF yaw plus `camera_meta.json`, so location/orientation updates are skipped but inference still runs.

#### 7.4.2 Thermal Mosaic Builder
- `backend/pvrt/dataops/mosaic_from_colmap.py` can optionally re-project raw thermal frames onto a flat plane using the COLMAP poses you uploaded, producing radiometrically faithful mosaics (`mosaic.tif`).
- The resulting mosaic tiles plus `images.geojson` entries allow you to review COLMAP-derived mosaics separately from the orthophoto pipeline (which never calls COLMAP). This workflow assumes you have the DJI SDK installed so radiometric frames can be decoded.
- If you only need per-frame inference, skip this workflow entirely—nothing else in the system depends on the mosaic artifacts.

### 7.5 Frontend Highlights
- **Results tab:** lists sessions with confidence histograms, download buttons (GeoJSON, overlays, logs), and backend metadata.
- **Map tab:** Leaflet map layers for imagery, anomalies, user overlays, and orthophoto tiles; filter chips toggle anomaly types and confidence thresholds.
- **Lightbox:** Keyboard/pointer navigation across rotated images with bounding-box callouts kept in sync with map selections.

---

## 8. Debugging & Troubleshooting
- **Custom weights placement:** Detectron2 loads `model_final.pth` or `model_best.pth` from `backend/projects/<project_id>/train/outputs/<run_name>/`. YOLO searches the same run folders for `model_best.pt`, `model_final.pt`, or `weights/best.pt`. Drop external checkpoints into those directories (or select a specific file in the UI) before starting inference to avoid "weights not found" errors.
- **YOLO weights missing:** Pre-download with `python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"` if your environment blocks outbound downloads.
- **Thermal SDK errors:** Verify `DIRP_SDK_PATH` and `LD_LIBRARY_PATH` echo correctly. If empty, re-activate the venv or ensure Docker `ENV` entries exist.
- **Map zoom glitches or blank overlays:** Hard-refresh (`Ctrl+Shift+R`). Confirm `camera_meta.json` contains lat/lon for every rotated image and that `images.geojson` is present in the session outputs.
- **Mosaic failures:** Require `rotated_images/` plus valid COLMAP metadata. Install `rasterio` for tiling and ensure `camera_meta.json` includes altitude / rotation data.
- **FastAPI errors:** Run uvicorn with `--log-level debug` and capture `fastapi.log`, then share the traceback when asking for help.

---

## 9. Data Preparation Tools
- **Labelme ➜ COCO:** This is just for information if you have labelme data and need to convert to coco json. Not needed if you already have coco json for detectron. 
  ```bash
  pip install labelme2coco
  labelme2coco data/train/labelme annotations/train_coco.json
  sed -i 's|/full/path/to/images/||g' annotations/train_coco.json
  ```
- **GeoJSON regeneration:**
  ```bash
  python backend/pvrt/dataops/regenerate_geojson_from_preds.py <session_id> <source_images_dir>
  ```
  Produces `rotated_images/`, `images.geojson`, and `anomalies.geojson` for a past run.
- **Thermal mosaic creation:**
  ```python
  from backend.pvrt.dataops.mosaic_from_colmap import create_mosaic_from_rotated_images
  from pathlib import Path

  create_mosaic_from_rotated_images(
      rotated_images_dir=Path("backend/projects/<project_id>/test/outputs/<session>/rotated_images"),
      out_mosaic_path=Path("backend/projects/<project_id>/test/outputs/<session>/mosaic.tif"),
      plane_z=0.0,
      resolution=0.1,
      camera_meta={...}
  )
  ```

With these steps documented in one place, you can choose the minimal dependency set, enable only the needed feature flags, and understand how each pipeline (single-frame, orthophoto, training, and COLMAP-assisted mosaics) flows through the system. Fire up uvicorn or Docker, point your browser to the frontend, and start exploring new solar anomaly datasets.
