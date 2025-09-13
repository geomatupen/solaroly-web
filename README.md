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
python3.10 -m venv .venv       # or python3.11
source .venv/bin/activate      # Windows: .venv\Scripts\activate
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
```

Install them:
```bash
pip install -r requirements.txt
```

---

# 6. DJI Thermal SDK Setup

### A. Native Library
After downloading the official DJI Thermal SDK, the native library is located at:

```bash
third_party/tsdk-core/lib/linux/release_x64/libdirp.so
```

### B. Configure venv Activation
Append the following lines to `.venv/bin/activate` so the paths are set automatically:

```bash
# --- DJI Thermal SDK config ---
export DIRP_SDK_PATH="$PWD/third_party/tsdk-core/lib/linux/release_x64/libdirp.so"
export LD_LIBRARY_PATH="$(dirname "$DIRP_SDK_PATH"):${LD_LIBRARY_PATH:-}"
# --- end DJI Thermal SDK config ---
```

Reactivate your virtual environment:
```bash
source .venv/bin/activate
```

### C. Install Python Wrapper
Before activating your venv, run:
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

---

# 7. Verify Installation

Run Python and check imports:

```python
from dji_thermal_sdk.dji_sdk import dji_init
from dji_thermal_sdk.utility import rjpeg_to_heatmap
```

If no errors, the SDK is ready.

---

# 8. Test Thermal Image Processing

Example test script:

```python
from dji_thermal_sdk.dji_sdk import dji_init
from dji_thermal_sdk.utility import rjpeg_to_heatmap
from pathlib import Path

# Initialize the SDK (no path needed if DIRP_SDK_PATH is set)
dji_init()

# Pick a sample R-JPEG image
img = Path("~/Pictures/DJI_0001.JPG").expanduser()

# Convert to temperature array
temps = rjpeg_to_heatmap(str(img), dtype="float32")

# Print min and max temperatures
print("min/max =", temps.min(), temps.max(), "°C")
```

---

# 9. Run Backend + Frontend

From the project root:

```bash
uvicorn backend.pvrt.web.app:app --reload --port 8001
```

Open in browser:  
[http://localhost:8001/](http://localhost:8001/)

---
