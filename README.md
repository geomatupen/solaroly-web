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

### Docker installation

- Docker Engine 24+ or Docker Desktop
- Docker Compose v2
- An x86-64 host when DJI thermal decoding is enabled
- Internet access during the first build for Python packages and pinned source
  archives
- Internet access the first time LightGlue weights are used, unless its
  checkpoint is already cached

The supplied image is CPU-based for portability. Inference and especially
training are considerably faster with a CUDA-capable GPU, but a CUDA Docker
image is not currently supplied. Use a local CUDA environment or derive a GPU
image from an NVIDIA/PyTorch development image and keep the documented package
versions consistent.

### Local installation

- Linux or WSL2 recommended
- Python 3.11
- Git and a C/C++ build toolchain
- CUDA-compatible drivers only when using CUDA
- DJI's compatible native library only when decoding DJI radiometric JPEGs

Storage requirements depend on imagery and generated model artifacts. Keep
project data on a disk with enough capacity for uploads, prepared images,
overlays, model checkpoints and post-processing snapshots.

## Quick start with Docker

Clone the repository and start the application:

```bash
git clone https://github.com/geomatupen/solaroly-web.git
cd solaroly-web
docker compose up --build
```

Open <http://localhost:8001>. Check the backend directly with:

```bash
curl http://localhost:8001/api/health
```

A healthy response contains `"ok": true`. Stop the application with:

```bash
docker compose down
```

The Compose configuration stores the complete `backend/projects` directory in
the named `solaroly-projects` volume, including the project registry and
default project folders. `docker compose down` preserves it;
`docker compose down --volumes` permanently removes it.

To use projects at another host location, add a bind mount to
`compose.yaml`, for example:

```yaml
services:
  solaroly:
    volumes:
      - solaroly-projects:/app/backend/projects
      - /absolute/host/project-data:/project-data
```

Create or register those projects using the corresponding container path, such
as `/project-data/site-a`. A container cannot access an unmounted host path.

The Docker image includes the pinned Detectron2, Ultralytics and LightGlue
integrations. Model weights are not bundled and may be mounted or selected from
a persistent project folder.

### Docker feature flags

All integrations are enabled by default in `compose.yaml`. Disable an unused
UI/backend integration without rebuilding:

```yaml
environment:
  PVRT_ENABLE_DETECTRON: "1"
  PVRT_ENABLE_YOLO: "1"
  PVRT_ENABLE_THERMAL: "1"
```

A disabled flag hides and guards that integration, but does not remove its
package from the image.

### Docker limitations

- The included DJI runtime is Linux x86-64. Disable thermal support on an
  unsupported architecture.
- The development server has no built-in authentication or TLS. Do not expose
  port 8001 directly to the public internet. Put it behind an authenticated
  HTTPS reverse proxy for shared or remote deployments.
- The default image is CPU-only.
- OpenStreetMap tiles and CDN-hosted map libraries require browser network
  access.

## Local installation

Create a virtual environment and install PyTorch first. For CPU:

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.1, replace the PyTorch command with:

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install `requirements.txt`. Do not install a second PyTorch build over
the selected runtime.

Start SolarOly:

```bash
export PVRT_ENABLE_DETECTRON=1
export PVRT_ENABLE_YOLO=1
export PVRT_ENABLE_THERMAL=1
uvicorn backend.pvrt.web.app:app --host 127.0.0.1 --port 8001
```

Open <http://localhost:8001>.

## DJI Thermal SDK

SolarOly retains the selected DJI runtime files under
`third_party/utility/bin/`. On Linux x86-64:

```bash
export DIRP_SDK_PATH="$PWD/third_party/utility/bin/linux/release_x64/libdirp.so"
export LD_LIBRARY_PATH="$(dirname "$DIRP_SDK_PATH"):${LD_LIBRARY_PATH:-}"
```

Set `PVRT_ENABLE_THERMAL=0` if DJI radiometric extraction is not required.
Ordinary RGB images and externally rendered thermal PNG/JPEG files can still be
processed, but radiometric measurements are unavailable.

The vendor license and EULA notice are retained in
[`third_party/License.txt`](third_party/License.txt). Verify that your DJI
camera and operating system are supported before processing a large dataset.

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
of the application source. Back up the persistent Docker volume or external
project roots before upgrades. Do not commit datasets, credentials, customer
imagery or model weights to the repository.

## Troubleshooting

- **Docker build context is unexpectedly large:** confirm that
  `.dockerignore` exists and that datasets are under ignored or external
  project folders.
- **Backend is unhealthy:** run `docker compose logs -f solaroly` and request
  `/api/health`.
- **Model weights are missing:** place compatible `.pth` or `.pt` files in a
  persistent project training output and select them in the UI.
- **LightGlue cannot load:** the first use may need network access to download
  its pretrained SIFT matcher checkpoint.
- **DJI thermal decoding fails:** verify `DIRP_SDK_PATH`,
  `LD_LIBRARY_PATH`, host architecture and the vendor runtime files.
- **An external project is not found in Docker:** mount its host parent and use
  the mounted container path when registering the project.
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
