# SolarOly

SolarOly is an open-source solar PV inspection platform for preparing imagery,
training and running segmentation or anomaly-detection models, reviewing
GeoJSON results on a map, and producing isolated post-processing outputs.

The FastAPI backend and browser frontend support DJI radiometric thermal
imagery, orthophotos, Detectron2, Ultralytics YOLO, SIFT + LightGlue image
alignment, approximate mosaics, and panel/anomaly post-processing.

> SolarOly is under active development. Back up project folders before an
> upgrade and validate generated geospatial results before operational use.

## Features

- Project workspaces for training data, test data, results, overlays and logs
- Detectron2 and Ultralytics YOLO training and inference
- DJI R-JPEG thermal decoding through the DJI Thermal SDK
- Individual-image, approximate-mosaic and orthophoto workflows
- Optional LightGlue alignment before inference
- Segmentation regularization, rows and panel IDs
- Overlap and visual anomaly deduplication
- Panel/row association and final anomaly output
- Leaflet review maps, layer editing and measurement tools
- Server-Sent Events for long-running job progress

## Requirements

- Linux or WSL2 (recommended)
- Python 3.11
- Git, a C/C++ build toolchain, and Ninja
- An NVIDIA CUDA-capable GPU for the supported high-performance installation
- A compatible [NVIDIA driver](https://www.nvidia.com/download/index.aspx)
- [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive),
  including `nvcc`, to build Detectron2 against the pinned CUDA 12.1 PyTorch
  runtime
- Internet access while installing pinned packages and when LightGlue first
  downloads its pretrained matcher checkpoint

SolarOly supports native installation. GPU and native-library combinations are
easier to diagnose and keep consistent in a local Python environment. A
CPU-only installation is documented below for development and limited testing,
but training and production inference are not recommended on CPU.

Storage requirements depend on imagery and generated model artifacts. Keep
project data on a disk with enough capacity for uploads, prepared images,
overlays, model checkpoints and post-processing snapshots.

### License-sensitive optional prerequisite

If you need radiometric DJI R-JPEG decoding, download the native runtime from
DJI's official [Thermal SDK download page](https://www.dji.com/downloads/softwares/dji-thermal-sdk)
before installing SolarOly. The SDK is not open source and is not included in
this repository. Review its packaged license and DJI's
[SDK EULA](https://developer.dji.com/policies/eula/) yourself. The setup under
[DJI Thermal SDK](#dji-thermal-sdk) explains how to point SolarOly at your
local copy.

No trained detection or segmentation weights are included. A model is not
required to install or open SolarOly, but training or inference requires a
compatible user-created or separately licensed checkpoint.

## Native installation

Install Python 3.11, Git, a C/C++ compiler and Ninja using your operating
system's package manager. For NVIDIA acceleration, install a driver and CUDA
12.1 first, then confirm both the GPU and compiler are visible:

```bash
nvidia-smi
nvcc --version
```

Clone SolarOly and create an isolated environment:

```bash
git clone https://github.com/geomatupen/solaroly-web.git
cd solaroly-web
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel ninja
```

Install the pinned CUDA build of PyTorch before the remaining requirements.
Detectron2 is then compiled against that exact PyTorch/CUDA environment:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install --no-build-isolation -r requirements.txt
```

The pinned combinations are intentional. Consult the official
[PyTorch previous-version matrix](https://pytorch.org/get-started/previous-versions/) and
[Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
before changing PyTorch, torchvision, CUDA or the compiler toolchain.

Verify the GPU stack before processing data:

```bash
python -c "import torch; print(torch.cuda.get_device_name()); assert torch.cuda.is_available()"
python -m detectron2.utils.collect_env
```

The Detectron2 report should show compatible values for PyTorch CUDA,
`CUDA_HOME`, the Detectron2 CUDA compiler and the GPU architecture. If PyTorch
or CUDA is changed later, reinstall Detectron2 so its compiled extension is not
left linked to the old runtime.

### CPU-only installation

CPU-only users must install SolarOly manually. Use the same virtual-environment
steps above, but install the CPU PyTorch build before `requirements.txt`:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-build-isolation -r requirements.txt
```

This path is useful for setup checks and light post-processing, but model
training, inference and image alignment can be prohibitively slow. Do not
install a CUDA PyTorch build over the CPU environment later; recreate the
virtual environment instead.

### Model weights

SolarOly does not bundle trained Detectron2 or Ultralytics model weights.
Create your own model in **Train**, or obtain a compatible checkpoint from a
source whose model and dataset licenses permit your intended use. Store model
weights in a project training output and select them in the UI. Some upstream
model tools can download pretrained weights on first use.

Start SolarOly:

```bash
export PVRT_ENABLE_DETECTRON=1
export PVRT_ENABLE_YOLO=1
export PVRT_ENABLE_THERMAL=0
uvicorn backend.pvrt.web.app:app --host 127.0.0.1 --port 8001
```

Open <http://localhost:8001>. A healthy backend returns `{"ok": true}` from
<http://localhost:8001/api/health>.

The development server has no authentication or TLS. Do not expose port 8001
directly to the public internet; use an authenticated HTTPS reverse proxy for
shared or remote deployments.

The default map uses third-party CDN assets and public OpenStreetMap/Esri tile
services. These require network access and remain subject to each provider's
availability and usage terms. For sustained, commercial or offline use,
configure an appropriate tile provider or self-hosted service; do not bulk
download from the public OpenStreetMap tile servers.

## DJI Thermal SDK

DJI's native Thermal SDK is proprietary and is **not distributed with
SolarOly**. It is only required to decode radiometric DJI R-JPEG files. Download
the appropriate Windows or Linux archive from the official
[DJI Thermal SDK download page](https://www.dji.com/downloads/softwares/dji-thermal-sdk),
review and accept DJI's included license and
[SDK EULA](https://developer.dji.com/policies/eula/), then extract it outside
the SolarOly repository.

The Python `dji-thermal-sdk` wrapper is installed by `requirements.txt`, but it
does not replace DJI's native `libdirp` runtime. For a Linux x86-64 SDK unpacked
at `/opt/dji-tsdk`:

```bash
export DIRP_SDK_PATH="/opt/dji-tsdk/utility/bin/linux/release_x64/libdirp.so"
export LD_LIBRARY_PATH="$(dirname "$DIRP_SDK_PATH"):${LD_LIBRARY_PATH:-}"
export PVRT_ENABLE_THERMAL=1
```

Keep every companion library from the selected DJI `release_x64` directory
together; `libdirp.so` is not a standalone file. Set `PVRT_ENABLE_THERMAL=0`
when DJI decoding is not required. Ordinary RGB imagery including DJI R-JPEG (if you want to use them as JPEG and dont need decoding), orthophotos and
already-rendered thermal PNG/JPEG files remain usable without the SDK, but DJI
radiometric extraction and temperature values are unavailable.

## Typical workflow

1. Create a project.
2. Import training data or select existing model weights.
3. Upload individual images or an orthophoto in **Test**.
4. Configure optional preprocessing, LightGlue alignment or an approximate
   mosaic.
5. Run inference and inspect `predictions.geojson` in **Results** or **Map**.
6. Create a post-processing job. Its selected GeoJSON inputs are copied into an
   isolated job snapshot so the original test result remains unchanged.
7. Complete the segmentation and/or anomaly steps and link final outputs to the
   map when needed.

LightGlue alignment refines horizontal position and residual in-plane
orientation from verified image overlaps. It is a planar placement refinement,
not a photogrammetric reconstruction: it does not solve terrain, altitude,
pitch, roll or a full calibrated 3D camera pose. Unmatched images retain their
metadata-derived pose. Corrected placement metadata is recorded in
`camera_meta.json` and `image_alignment.json`.

## Project data

Default projects are stored under:

```text
backend/projects/
├── projects.json
└── <project_id>/
    ├── train/
    │   ├── data/
    │   └── outputs/
    ├── test/
    │   ├── data/
    │   └── outputs/
    │       └── .postprocess_jobs/
    ├── overlays/
    └── logs/
```

Project folders contain source imagery and generated artifacts and are not part
of the application source. Back up project roots before upgrades. Do not commit
datasets, credentials, customer imagery or model weights to the repository.

## Troubleshooting

- **Backend does not start:** activate the intended virtual environment, check
  the traceback, and verify `/api/health` after starting Uvicorn.
- **CUDA is unavailable:** run `nvidia-smi`, inspect
  `python -m detectron2.utils.collect_env`, and confirm PyTorch, the CUDA
  Toolkit and the compiled Detectron2 extension use compatible CUDA versions.
- **Model weights are missing:** place compatible `.pth` or `.pt` files in a
  project training output and select them in the UI.
- **LightGlue cannot load:** the first use may need network access to download
  its pretrained SIFT matcher checkpoint.
- **DJI thermal decoding fails:** verify `DIRP_SDK_PATH`,
  `LD_LIBRARY_PATH`, host architecture, vendor runtime companion libraries and
  that `PVRT_ENABLE_THERMAL=1` was set before startup.
- **Map imagery is blank:** verify browser connectivity to the configured tile
  provider and inspect `images.geojson` and `camera_meta.json`.

## License and source

SolarOly is licensed under the
[GNU Affero General Public License v3.0](LICENSE.txt). If you modify SolarOly
and make the modified application available to users over a network, those
users must be offered the corresponding source as required by the AGPL.

Source for this version is available at
<https://github.com/geomatupen/solaroly-web>.

Third-party copyrights, license summaries and citations are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and datasets
may carry separate terms.

## Acknowledgements

SolarOly uses [LightGlue](https://github.com/cvg/LightGlue) by Philipp
Lindenberger, Paul-Edouard Sarlin and Marc Pollefeys for local feature
matching. If you use SolarOly's alignment workflow in research, please cite
*LightGlue: Local Feature Matching at Light Speed*, ICCV 2023. The complete
BibTeX entry is provided in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Special thanks to Termatics, Austria, for providing the opportunity and support
to develop this project.
