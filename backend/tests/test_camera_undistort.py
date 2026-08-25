from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from pvrt.dataops.camera_undistort import (
    CameraIdentity,
    UndistortionError,
    _parse_dewarp_data,
    inspect_camera,
    resolve_calibration,
    undistort_pil_image,
)
from pvrt.web.mosaic import prepare_rotation_and_mosaic


class CameraUndistortionTests(unittest.TestCase):
    def _camera_image(self, directory: Path, *, dewarp_flag: str = "") -> Path:
        path = directory / "thermal.jpg"
        exif = Image.Exif()
        exif[271] = "DJI"
        exif[272] = "M3TD"
        options = {"exif": exif, "quality": 100}
        if dewarp_flag:
            options["xmp"] = (
                '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
                '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                '<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
                f'drone-dji:DewarpFlag="{dewarp_flag}"/></rdf:RDF></x:xmpmeta>'
            ).encode("utf-8")
        Image.new("L", (8, 6), 127).save(path, **options)
        return path

    def test_embedded_dji_dewarp_uses_center_offsets(self) -> None:
        identity = CameraIdentity("DJI", "M3TD", "InfraredCamera", 640, 512, "serial")
        calibration = _parse_dewarp_data(
            "2026-01-01;500,501,2,-3,-0.1,0.02,0.001,-0.002,0.003",
            identity,
        )
        self.assertEqual(calibration.camera_matrix[0][2], 322.0)
        self.assertEqual(calibration.camera_matrix[1][2], 253.0)
        self.assertEqual(len(calibration.distortion_coefficients), 5)

    def test_profile_is_detected_and_image_size_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._camera_image(root)
            profile_dir = root / "profiles"
            profile_dir.mkdir()
            (profile_dir / "m3td.json").write_text(json.dumps({
                "profile_id": "test-m3td",
                "make": "DJI",
                "model": "M3TD",
                "width": 8,
                "height": 6,
                "camera_matrix": [[10, 0, 4], [0, 10, 3], [0, 0, 1]],
                "distortion_coefficients": [0, 0, 0, 0, 0],
            }), encoding="utf-8")
            with Image.open(source) as image:
                corrected, record = undistort_pil_image(image.convert("RGB"), source, profile_dir)
            self.assertEqual(corrected.size, (8, 6))
            self.assertEqual(record["calibration"]["profile_id"], "test-m3td")
            self.assertEqual(record["status"], "skipped_below_threshold")
            self.assertEqual(np.asarray(corrected).shape, (6, 8, 3))

            with Image.open(source) as image:
                _, forced_record = undistort_pil_image(
                    image.convert("RGB"), source, profile_dir, minimum_displacement_px=0,
                )
            self.assertEqual(forced_record["status"], "corrected")

    def test_dewarp_flag_skips_without_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._camera_image(root, dewarp_flag="1")
            with Image.open(source) as image:
                corrected, record = undistort_pil_image(image.convert("RGB"), source, root / "empty")
            self.assertEqual(corrected.size, (8, 6))
            self.assertEqual(record["status"], "skipped_already_corrected")
            self.assertIsNone(record["calibration"])

    def test_missing_calibration_has_actionable_strict_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._camera_image(root)
            identity, xmp = inspect_camera(source)
            with self.assertRaisesRegex(UndistortionError, "Disable .Correct lens distortion automatically."):
                resolve_calibration(identity, xmp, root / "empty")

    def test_strict_rotation_propagates_undistortion_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            session = root / "session"
            source.mkdir()
            session.mkdir()
            (session / "camera_meta.json").write_text(
                json.dumps({"thermal.jpg": {"lat": 1, "lon": 1}}),
                encoding="utf-8",
            )
            failed = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="Cannot correct lens distortion for DJI M3TD. Disable ‘Correct lens distortion automatically’ and run again.",
            )
            with patch("pvrt.web.mosaic.subprocess.run", return_value=failed):
                with self.assertRaisesRegex(RuntimeError, "Cannot correct lens distortion for DJI M3TD"):
                    prepare_rotation_and_mosaic(
                        input_type="images",
                        session_dir=session,
                        out_root=session,
                        camera_meta={"thermal.jpg": {"lat": 1, "lon": 1}},
                        mosaic_enabled=False,
                        ds_dir=source,
                        model_is_thermal=True,
                        undistort_thermal=True,
                        tile_tif_func=lambda *_args, **_kwargs: None,
                        run_images_dir=source,
                        tiles_dir=None,
                        tif_src=None,
                    )
            with patch("pvrt.web.mosaic.subprocess.run", return_value=failed):
                result = prepare_rotation_and_mosaic(
                    input_type="images",
                    session_dir=session,
                    out_root=session,
                    camera_meta={"thermal.jpg": {"lat": 1, "lon": 1}},
                    mosaic_enabled=False,
                    ds_dir=source,
                    model_is_thermal=True,
                    undistort_thermal=False,
                    tile_tif_func=lambda *_args, **_kwargs: None,
                    run_images_dir=source,
                    tiles_dir=None,
                    tif_src=None,
                )
            self.assertEqual(result.run_images_dir, source)


if __name__ == "__main__":
    unittest.main()
