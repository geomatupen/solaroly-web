# Third-party notices

SolarOly is distributed under the GNU Affero General Public License v3.0. It
uses third-party software and data whose copyrights and licenses remain with
their respective owners. This summary is provided for attribution and does not
replace the license text shipped by each dependency.

## Ultralytics YOLO

SolarOly includes an integration with [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
Ultralytics software and models are offered under the GNU Affero General Public
License v3.0, with a separate enterprise license available from Ultralytics.
The Docker image pins Ultralytics 8.3.220.

## LightGlue

SolarOly uses the official [LightGlue](https://github.com/cvg/LightGlue)
implementation and its pretrained SIFT matcher weights, released under the
Apache License 2.0.

Please cite:

```bibtex
@inproceedings{lindenberger2023lightglue,
  author    = {Philipp Lindenberger and Paul-Edouard Sarlin and Marc Pollefeys},
  title     = {LightGlue: Local Feature Matching at Light Speed},
  booktitle = {ICCV},
  year      = {2023}
}
```

## Detectron2

SolarOly includes [Detectron2](https://github.com/facebookresearch/detectron2),
released by Facebook AI Research under the Apache License 2.0.

Recommended citation:

```bibtex
@misc{wu2019detectron2,
  author       = {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title        = {Detectron2},
  howpublished = {https://github.com/facebookresearch/detectron2},
  year         = {2019}
}
```

## DJI Thermal SDK

SolarOly vendors selected DJI Thermal SDK runtime files for radiometric DJI
imagery. Copyright belongs to DJI. Some files are MIT-licensed and other SDK
portions are governed by DJI's SDK End User License Agreement. The original
notices and terms are retained in [`third_party/License.txt`](third_party/License.txt),
with vendor documentation in [`third_party/Readme.md`](third_party/Readme.md).

## Web mapping

- [Leaflet](https://github.com/Leaflet/Leaflet) is distributed under the BSD
  2-Clause License.
- [Leaflet-Geoman Free](https://github.com/geoman-io/leaflet-geoman) is
  distributed under the MIT License.
- Map data and tiles are provided by
  [OpenStreetMap contributors](https://www.openstreetmap.org/copyright) under
  the Open Database License. Required attribution remains visible on maps.

## Other dependencies

SolarOly also depends on projects including PyTorch, torchvision, OpenCV,
Kornia, FastAPI, Rasterio, GDAL, PROJ, Shapely, NumPy and Pillow. Their package
metadata and upstream distributions contain the applicable copyright and
license notices. Models, checkpoints and datasets supplied by users may have
separate terms; users are responsible for confirming that their use and
redistribution are permitted.
