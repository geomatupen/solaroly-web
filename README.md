# SolarOly Project

SolarOly is an end-to-end inspection console for Solar PV. It ingests DJI thermal flights, orthophotos, then turns them into actionable anomaly reports that can be reviewed, exported, or pushed into downstream GIS tools.

This repository packages the FastAPI backend that drives thermal decoding, Detectron2/YOLO inference, COLMAP-assisted alignment, and batch utilities, plus the single-page frontend that manages uploads, project workspaces, training knobs, and map reviews. Feature flags (`PVRT_ENABLE_*`) keep heavy integrations opt-in so the same codebase runs on a laptop or a GPU workstation without code changes.

**Use SolarOly to:**
- Create isolated projects for different sites and keep train/test data, outputs, overlays, and logs neatly partitioned.
- Decode DJI radiometric frames, fine-tune Detectron2 or YOLO models, and launch inference or orthophoto tiling jobs from the browser.
- Align imagery with COLMAP or metadata-based rotations, regenerate GeoJSON overlays, and stream results into Leaflet maps for QA before exporting.
- Monitor every job via Server-Sent Events, download artifacts (weights, logs, overlays), and hand off the deliverables to ops teams without leaving the app.

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
2. Decide which heavy integrations you need (Detectron2 or YOLO, COLMAP helpers). Use Section 3 to comment blocks in `requirements.txt` **before** installing.
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
| Detectron2-based training/testing | `detectron2 @ git+...`, `fvcore`, `iopath`, `hydra-core`, `omegaconf`, `yacs`, `pycocotools`, `tensorboard`, `tabulate`, `matplotlib` | Comment the entire block if you only plan to run YOLO. | `PVRT_ENABLE_DETECTRON=0/1` |
| YOLOv8 pipeline | `ultralytics>=8.3.0,<9.0.0` plus the shared OpenCV/numpy stack already present | Comment `ultralytics` if Detectron2 is the only backend you ship. | `PVRT_ENABLE_YOLO=0/1` |
| CUDA accelerators | All `nvidia-*` wheels plus CUDA-specific Torch builds | Comment GPU wheels and install CPU Torch (`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`) when no CUDA-capable device exists. | Not flag-controlled; match your hardware |
| DJI Thermal SDK (optional) | `dji-thermal-sdk==0.0.2`, `opencv-python-headless`, `tifffile`, `piexif`, `exif` | Comment this block if you never ingest DJI R-JPEG/radiometric frames. Keep it to unlock grayscale thermal extraction, temperature metadata, and the COLMAP-driven thermal mosaic pipeline. | `PVRT_ENABLE_THERMAL=0/1` (drives the shared `thermal_data_extraction` flag; when 0 the UI hides "Use thermal" toggles and the backend rejects decode/train/test requests that require DJI payloads) + export `DIRP_SDK_PATH`, `LD_LIBRARY_PATH` when enabled |
| COLMAP-assisted alignment | Installed separately via `conda install -c conda-forge colmap` or system packages | Skip entirely if you only need per-frame inference. | `PVRT_ENABLE_COLMAP=0/1` |

**Base requirement:** PyTorch (`torch`, `torchvision`, `torchaudio`, `triton`) is always needed because both Detectron2 and YOLO sit on top of it. Install the CUDA wheels shown in `requirements.txt` or swap in the CPU-only wheels before enabling either backend.

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

> **When you can skip this:** If your datasets only contain RGB imagery or thermal renders from non-DJI drones (i.e., no DJI R-JPEG radiometric frames), you may comment the DJI SDK dependencies, skip this section, **and set `PVRT_ENABLE_THERMAL=0`**. Leave the "Use thermal" checkbox unchecked in the UI. The backend will operate on RGB or externally supplied thermal PNG/JPEG files but will not attempt grayscale extraction or temperature calculations.

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
- **Feature flags:** `backend/pvrt/web/settings.py` reads `PVRT_ENABLE_DETECTRON`, `PVRT_ENABLE_YOLO`, `PVRT_ENABLE_COLMAP`, and `PVRT_ENABLE_THERMAL`. Export them before launching uvicorn or set them inside Docker. Disabled integrations disappear from the UI and the backend guards routes accordingly.
   - The thermal flag surfaces to the SPA as `thermal`/`thermal_data_extraction`. When off, the frontend disables "Use thermal" checkboxes automatically and the API refuses dataset decodes, thermal-aware training, and thermal-only inference to prevent accidental DJI SDK calls.
- **Camera metadata:** `camera_meta.json` stores alignment hints (heading offsets, reference elevations) and is used by the rotation/mosaic pipeline plus the orthophoto tiler.

---

## 6. Running the Stack
1. Activate your environment and export feature flags:
   ```bash
   source venv/bin/activate
   export PVRT_ENABLE_DETECTRON=1
   export PVRT_ENABLE_YOLO=1
   export PVRT_ENABLE_COLMAP=0
   export PVRT_ENABLE_THERMAL=1
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
2. **Thermal extraction / fallback:** With the DJI SDK installed and `PVRT_ENABLE_THERMAL=1`, the platform decodes DJI R-JPEG files into 16-bit grayscale rasters and exposes temperature metadata. If you skipped the SDK (non-DJI thermal cameras or RGB-only workflows) and set `PVRT_ENABLE_THERMAL=0`, the UI auto-disables the "Use thermal" checkbox and the API blocks decode/train/test calls that would require DJI libraries. RGB imagery or pre-rendered thermal PNG/JPEG files still run normally, but grayscale extraction and temperature readouts remain unavailable.
3. **Rotation & normalization:** Metadata-driven heading correction rotates each frame north-up. If rotation scripts fail or metadata is missing, the system continues with the original orientation but flags the session in the log.
4. **Inference:**
   - Detectron2: loads the configured `model_final.pth` under `train/outputs/<run>/weights/` and respects batch size/thresholds from the UI.
   - YOLOv8: uses Ultralytics models; per-image overlays are rendered with CV2, stored in `test/outputs/<session>/overlays`, and summarized via `raw_results_summary.json` and `metrics.json` (see `backend/pvrt/backends/yolo/infer.py`).
5. **Geo outputs:** Every run writes `preds/*.json`, `rotated_images/`, `images.geojson`, `anomalies.geojson`, overlays, and `test.log`. The frontend map tab consumes the GeoJSON directly and renders anomalies + camera markers.

#### Optional runtime lens correction

Enable **Test → Advanced → Correct lens distortion automatically** to estimate radial correction before rotation and inference. The option is off by default and orthophotos are skipped. For each camera/sensor/resolution group, the runtime checks up to eight evenly spaced files from the sorted folder, rejects structurally weak samples, and jointly fits one shared model from the best three using repeated long-line evidence. Held-out traces must improve consistently and the remap must pass coverage, monotonicity, and displacement checks. If exactly one of the best three disagrees, it is first replaced by the fourth-ranked usable image and all three are validated again. If that still fails, the two originally consistent samples are refitted once; every image in the accepted attempt must pass or preparation stops without corrected images. Overlap is not required, dimensions are preserved, and fitted values plus decisions—including failed calibration diagnostics—are saved in `preprocessing.json`. The implementation uses the existing OpenCV and NumPy dependencies only.

When lens correction is enabled, an additional default-off **Export undistorted originals for photogrammetry** option appears. Enabling both options writes `test/outputs/<result>/undistorted_images/` before north-up rotation. These files retain the original filename, JPG/PNG format, dimensions, transferable EXIF/GPS, XMP, ICC profile, comments, DPI and filesystem timestamps for WebODM testing. Inference continues to use its separate prepared/rotated files. Proprietary radiometric payloads and MPO secondary frames are intentionally not copied because they describe the original encoded sensor product rather than the remapped pixels.

#### Approximate image mosaic without COLMAP

Enable **Test → Create approximate mosaic** for a folder containing at least two geotagged images. Its inference-source and alignment options remain hidden until mosaicing is enabled. The runtime first performs any selected lens correction and rotates each complete image north-up from camera/gimbal heading. It derives each full image footprint from that image’s metadata GSD, places it on the selected mosaic-resolution canvas without pre-cropping, and checks at most eight plausible overlapping neighbours. Hybrid matching combines ORB features with descriptor points along long structures, rejects pixel-identical frames, validates constraints with GPS-gated partial-affine RANSAC, and only moves components supported by at least three connected images. The fitted pair angle validates the trusted heading but does not replace it. Images without validated overlap retain GPS placement. Images are blended first; only the completed mosaic is cut into inference tiles. Detailed pair decisions, structural-point counts and final corrections are saved in `mosaic_alignment.json`.

#### Segmentation and anomaly post-processing

The **Post-process** page has separate Segmentation and Anomalies workflows backed by one reusable map/editor. The shared Test result selector sits above those workflow tabs. Segmentation keeps three replaceable processing layers: Combined, Regularized, and one hierarchy GeoJSON containing both merged array/row polygons and identified panels. Adjacent inner rows are grouped into merged rectangles and receive deterministic IDs such as `1000-A1` (`1000` array, inner row `A`, panel `1`). Row IDs are displayed persistently on the map and panel IDs appear on hover whenever those properties exist. The hierarchy also exports exactly two Saved outputs—`solar_panels.geojson` and `solar_rows.geojson`—without adding extra processing layers. Re-running a base stage replaces it after confirmation and invalidates only dependent base outputs; edited snapshots are retained. The numbered workflow cards can be minimized. Anomaly processing can remove duplicate same-class predictions from overlapping source images and associate each retained anomaly with an identified panel and row, including panel layers from a different test result. Generated layers remain GeoJSON files inside the source result, may be edited with vertex/move/rotate/delete tools and undo/redo, and can be linked to Map without copying the data.

The revealed inference-source choices are exclusive: **Individual images only** creates the mosaic for map review but predicts every prepared source frame, while **Approximate mosaic only** predicts only tiles cut from the finished mosaic. The Map image list shows both the mosaic layer and its prepared source frames; mosaic-only source frames are marked as review sources and are not inferred a second time. The backend rejects concurrent runs writing to the same result folder and refuses silent reuse of a non-empty result. This is a quick visual/inference mosaic rather than a survey orthophoto; GPS, heading, altitude, perspective, and terrain errors can still leave seams or offsets.

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

The Train tab's **Training data** asset panel has a dedicated uploader separate from test-data intake. It accepts one ZIP archive or a browser-selected folder using `train/` and `valid/` splits containing images plus COCO annotation JSON, matching YOLO `.txt` sidecars, or both. New uploads are stored automatically under `<project>/train/datasets/<dataset-name>` and registered when at least one supported annotation format validates. Recorded backend compatibility determines whether a dataset appears for Detectron, YOLO, or both in the Train selector. Uploaded datasets receive a stable ID and can be opened from the asset list for a validation summary. The selected dataset is revalidated immediately before each run, and its ID, name, path, and format are recorded in model metadata. The Trained models list also shows interrupted or failed run folders as **Incomplete**, allowing their artifacts to be deleted; only **Complete** models appear in Testing. Clicking a model card opens the same flattened **Model Meta** list used in Results alongside the latest recorded Detectron or YOLO losses; this also works for incomplete runs that produced metrics before stopping.

The Test tab has a matching right-side asset library for **Test datasets** and saved **Model results**. Both lists show separate display names and stable folder IDs, support display-name changes without moving their folders, and provide scoped delete actions. Model-result cards show **Complete** or **Incomplete** using explicit status files for new inference runs and required-artifact checks for legacy runs. Opening a result switches to its full Results view, while selecting a test-dataset card makes it the active inference input.

The standalone **Image to grayscale** button on the Projects page works independently of any project. Choose **Radiometric thermal JPEG** to extract DJI DIRP data (including M3T/M3TD) or FLIR FFF data (including DJI Zenmuse XT2). Choose **Standard JPG/PNG** to convert the visible pixels of ordinary JPG, JPEG, and PNG inputs. Standard mode skips radiometric JPEGs by default; explicitly select **Include radiometric JPEGs using their visible pixels** to include them without extracting their sensor data. Enter exact absolute input and output folder paths and scan the input before starting. The preflight warns about files unsupported by the selected mode and prevents conversion when none are supported. Conversion skips unsupported files, reports per-file progress, produces grayscale JPG or PNG at the exact source dimensions, and preserves transferable EXIF/GPS/camera metadata, ICC/XMP data, and filesystem timestamps. The same converter is available from the command line; add `--type standard` for ordinary images and `--include-radiometric` when desired:

```bash
PYTHONPATH=backend python -m pvrt.dataops.thermal_convert INPUT_DIR OUTPUT_DIR --format jpg
```

### 7.4 COLMAP Workflows

#### 7.4.1 Pose Optimization for Single Images
- `PVRT_ENABLE_COLMAP=1` unlocks pose uploads inside the single-image testing form. COLMAP-derived intrinsics/extrinsics refine each camera’s latitude/longitude/heading before inference so detections line up with site plans even when EXIF yaw is noisy.
- Optimized poses live beside the session under `test/outputs/<session>/colmap_pose.json` and are referenced whenever rotated imagery, GeoJSON footprints, or overlays are generated.
- If COLMAP stays disabled, the backend simply relies on EXIF yaw plus `camera_meta.json`, so location/orientation updates are skipped but inference still runs.

#### 7.4.2 Thermal Mosaic Builder
- `backend/pvrt/dataops/mosaic_from_colmap.py` can optionally re-project raw thermal frames onto a flat plane using the COLMAP poses you uploaded, producing radiometrically faithful mosaics (`mosaic.tif`).
- The resulting mosaic tiles plus `images.geojson` entries allow you to review COLMAP-derived mosaics separately from the orthophoto pipeline (which never calls COLMAP). This workflow assumes you have the DJI SDK installed and `PVRT_ENABLE_THERMAL=1` so radiometric frames can be decoded.
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
- **Mosaic failures:** Approximate mosaics require `rotated_images/` and at least two images with readable EXIF GPS coordinates; heading and altitude metadata improve placement. Install `rasterio` for GeoTIFF writing and tiling. COLMAP is required only when you explicitly request optimized camera poses.
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
