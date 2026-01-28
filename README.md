# SolarOly Project

This project processes drone imagery using PyTorch, Detectron2, and DJI Thermal SDK.  
It provides a web interface with FastAPI for training, inference, and visualization, including thermal image processing.

---

# 1. Prerequisites

- Python 3.10 or 3.11 (or 3.12 if compatible)  
- CUDA 12.1 for GPU usage (adjust if using another CUDA version)  
- Git  

---

# 2. Create & Activate Virtual Environment

```bash
cd /path/to/project
python3 -m venv venv       # or python3.11
source venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install -U pip setuptools wheel
```

---

# 3. Install PyTorch

### GPU (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CPU-only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Adjust the URL if you use a different CUDA version.

---

# 4. Install Detectron2

For Torch 2.3 + CUDA 12.1:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

---

# 5. Install Other Python Dependencies

Install from requirements.txt:
```bash
pip install -r requirements.txt
```
OR

Create `requirements.txt` (excluding torch + detectron2):

```text
fastapi==0.115.*
uvicorn[standard]==0.30.*
starlette==0.38.*
pydantic==2.*
opencv-python-headless==4.10.*
tifffile==2024.*
piexif==1.1.*
numpy==1.26.*
pycocotools==2.0.*
python-multipart==0.0.20
```


---

# 6. DJI Thermal SDK Setup

### A. Native Library
The third_party folder already contains the DJI thermal SDK, skip this step if you dont want to upgrade the version of this sdk.

OR if you want to upgrade,
Download DJI thermal SDK from official DJI website and extract the zip in the folder third_party

After downloading the official DJI Thermal SDK, the native library should located at:

```bash
third_party/tsdk-core/lib/linux/release_x64/libdirp.so
```

### B. Configure venv Activation
Add the following lines at the end of the file: `venv/bin/activate` so the paths are set automatically:

```bash
# --- DJI Thermal SDK config ---
export DIRP_SDK_PATH="$PWD/third_party/utility/bin/linux/release_x64/libdirp.so"
export LD_LIBRARY_PATH="$(dirname "$DIRP_SDK_PATH"):${LD_LIBRARY_PATH:-}"
# --- end DJI Thermal SDK config ---

### C. Install Python Wrapper
```bash
python -m pip install --upgrade pip wheel
python -m pip install --upgrade dji-thermal-sdk
```

Check it’s installed:
```bash
python -m pip show dji-thermal-sdk
```

Expected output:
```text
Name: dji-thermal-sdk
Version: 0.0.2
Location: /path/to/project/venv/lib/python3.12/site-packages
```


### D. Reactivate Environment
```

Reactivate your virtual environment:
```bash
source venv/bin/activate
```

---

# 7. Verify Installation

Run Python and check imports:

```python
python3
from dji_thermal_sdk.dji_sdk import dji_init
from dji_thermal_sdk.utility import rjpeg_to_heatmap
```

If no errors, the SDK is ready. you can type exit() to exit python cell.

---

# 7.5 Install YOLO (Optional - for YOLO backend)

If you plan to use YOLO for object detection instead of Detectron2:

```bash
pip install ultralytics==8.0.*
```

Verify installation:
```python
python3
from ultralytics import YOLO
print(YOLO)
```

YOLO models will be auto-downloaded on first use, or you can pre-download:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"  # nano model
```

Available YOLO sizes: `yolov8n` (nano), `yolov8s` (small), `yolov8m` (medium), `yolov8l` (large), `yolov8x` (xlarge)

---

# 7.6 Install Additional Dependencies (Optional Features)

### For GeoTIFF/Mosaic Support
If you need to create thermal mosaics and GeoTIFF tiling:
```bash
pip install rasterio==1.3.*
```

### For COLMAP Integration (Advanced)
For camera pose optimization and 3D reconstruction:
- Install COLMAP from [colmap.github.io](https://colmap.github.io)
- Or use conda: `conda install -c conda-forge colmap`

---

# 8. Configuration

### Backend & Model Selection

Configuration is managed in `output/config.yaml` or set via the web UI:

```yaml
backend: "yolo"              # Options: "detectron" or "yolo"
model_folder: "models/yolo"  # Directory containing model weights
channel_config: "rgb+thermal" # Options: "rgb", "thermal", "rgb+thermal"
```

### Datasets Structure

Organize training/validation data:
```
data/
├── train/
│   ├── images/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── _annotations.coco.json
│   └── labels/ (for YOLO format)
└── valid/
    ├── images/
    ├── _annotations.coco.json
    └── labels/ (for YOLO format)
```

Thermal images expected at: `data/*/thermal/<stem>_thermal.tif`

---

# 9. Run Backend + Frontend

From the project root:

```bash
uvicorn backend.pvrt.web.app:app --reload --port 8001
OR
uvicorn backend.pvrt.web.app:app --workers 1 --port 8001  #change workers number based on the hardware you have.

```

Open in browser:  
[http://localhost:8001/](http://localhost:8001/)


## Debugging FastAPI/uvicorn errors

To capture detailed logs (including tracebacks) when running the backend, use:

```bash
uvicorn backend.pvrt.web.app:app --reload --port 8001 > fastapi.log 2>&1
```

This will save all output (including errors) to `fastapi.log`. If you encounter an error, open this file and share the relevant traceback for debugging.

Alternatively, you can run with more verbose logging:

```bash
uvicorn backend.pvrt.web.app:app --reload --port 8001 --log-level debug #> fastapi.log 2>&1
```

For real-time viewing, use:

```bash
tail -f fastapi.log
```

This helps diagnose issues that occur during training, evaluation, or API calls.

---

# 10. Key Features

## Web Interface
- **Dataset Management**: Upload and organize drone imagery
- **Model Training**: Train on COCO-annotated datasets (Detectron2 or YOLO)
- **Inference/Testing**: Run detection on new images and generate results
- **Results Viewer**: 
  - Results tab: View predictions with detection badges and filtering
  - Map tab: Geo-referenced visualization with anomaly filtering
  - Lightbox: Pan/zoom image gallery with keyboard navigation
- **Session Management**: Save, load, and compare inference sessions
- **GeoJSON Export**: Download detection results as geo-referenced GeoJSON

## Detection Features
- Real-time detection count badges on image thumbnails
- Filter images by detection presence ("Show only detected")
- Filter anomalies by active camera locations
- Geo-referenced anomaly visualization on map

## Thermal Processing
- Automatic thermal image rotation based on camera heading
- Normalized thermal preview generation
- Single-plane mosaic creation from rotated images (via COLMAP poses)
- GeoTIFF export with proper georeferencing

## Supported Backends
- **Detectron2**: Mask R-CNN and other instance segmentation models
- **YOLO**: YOLOv8 object detection models
- **Thermal SDK**: DJI Thermal SDK for radiometric processing

---

# 11. Troubleshooting

### YOLO Model Not Found
If YOLO fails to download models automatically, download manually:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### Thermal Image Processing Errors
Verify DJI SDK paths are set:
```bash
echo $DIRP_SDK_PATH
echo $LD_LIBRARY_PATH
```

Should show paths to the thermal SDK library. If empty, re-activate venv:
```bash
source venv/bin/activate
```

### Map Tab Zoom Issues
Hard refresh browser: `Ctrl+Shift+R`  
Camera locations must have valid lat/lon in `camera_meta.json` and `manifest.json`

### Mosaic Generation Failures
Ensure:
- `rotated_images/` folder exists in session directory
- `camera_meta.json` has lat/lon coordinates for all images
- Rasterio is installed: `pip install rasterio`

---

# 12. Data Preparation Tools

## Converting Labelme Annotations to COCO

Use labelme2coco to convert Labelme JSON annotations to COCO format:

```bash
pip install labelme2coco
labelme2coco <annotations_folder> <output_coco_json>
```

Example:
```bash
labelme2coco data/train/labelme annotations/train_coco.json
```

Then update paths in the generated JSON (remove full paths, keep only filenames):
```bash
sed -i 's|/full/path/to/images/||g' annotations/train_coco.json
```

## Regenerating GeoJSON from Predictions

After inference, regenerate GeoJSON and rotated images:

```bash
python backend/pvrt/dataops/regenerate_geojson_from_preds.py <session_id> <source_images_dir>
```

This creates:
- `media/sessions/<session_id>/rotated_images/` - north-up image copies
- `media/sessions/<session_id>/images.geojson` - image footprints with lat/lon
- `media/sessions/<session_id>/anomalies.geojson` - detection boxes as geo-polygons

## Creating Thermal Mosaics

Generate a georeferenced thermal mosaic from rotated images (requires COLMAP poses):

```python
from backend.pvrt.dataops.mosaic_from_colmap import create_mosaic_from_rotated_images
from pathlib import Path

create_mosaic_from_rotated_images(
    rotated_images_dir=Path("media/sessions/session_id/rotated_images"),
    out_mosaic_path=Path("output/session_id/mosaic.tif"),
    plane_z=0.0,
    resolution=0.1,  # meters per pixel
    camera_meta={...}  # from camera_meta.json
)
```

---

# 13. Project Structure

```
solaroly-web/
├── backend/
│   └── pvrt/
│       ├── web/
│       │   └── app.py              # FastAPI application
│       ├── core/
│       │   ├── registry.py         # Backend registration
│       │   ├── io.py               # File I/O utilities
│       │   └── thermal.py          # Thermal processing
│       ├── dataops/
│       │   ├── mosaic_from_colmap.py       # Mosaic generation
│       │   ├── regenerate_geojson_from_preds.py  # GeoJSON generation
│       │   └── scan_decode_split.py        # Data splitting
│       ├── backends/
│       │   ├── detectron/
│       │   └── yolo/                # YOLO backend
│       └── config.py
├── frontend/
│   ├── app.js                       # Main JavaScript
│   ├── index.html                   # Web interface
│   └── styles.css                   # Styling
├── data/                            # Training/validation datasets
├── media/sessions/                  # Session outputs
├── output/                          # Inference results
├── models/                          # Model weights
└── third_party/                     # DJI Thermal SDK
```

---

# 14. Converting Labelme Annotations to COCO


https://github.com/mcp?utm_source=vscode-website&utm_campaign=mcp-registry-server-launch-2025 