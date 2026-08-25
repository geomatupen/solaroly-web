from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from pvrt.dataops.lens_distortion import (
    CameraGroup,
    RuntimeCalibration,
    correct_pil_image,
    inspect_camera_group,
)
from pvrt.dataops.plumb_line_calibration import estimate_runtime_calibration
from pvrt.dataops.plumb_line_calibration import CurveTrace
from pvrt.web.mosaic import prepare_rotation_and_mosaic


class RuntimeLensCalibrationTests(unittest.TestCase):
    def test_camera_group_uses_sensor_identity_and_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "thermal.jpg"
            exif = Image.Exif()
            exif[271] = "DJI"
            exif[272] = "M3TD"
            Image.new("L", (80, 64), 127).save(source, exif=exif)
            group = inspect_camera_group(source)
            self.assertEqual(group.make, "DJI")
            self.assertEqual(group.model, "M3TD")
            self.assertEqual((group.width, group.height), (80, 64))

    def test_runtime_mapping_preserves_dimensions(self) -> None:
        group = CameraGroup("Test", "Camera", "", 80, 64)
        calibration = RuntimeCalibration(
            [[80.0, 0.0, 40.0], [0.0, 80.0, 32.0], [0.0, 0.0, 1.0]],
            [-0.08, 0.01, 0.0, 0.0, 0.0],
        )
        corrected, record = correct_pil_image(Image.new("RGB", (80, 64), "white"), group, calibration)
        self.assertEqual(corrected.size, (80, 64))
        self.assertEqual(record["status"], "corrected")
        self.assertGreater(record["maximum_displacement_px"], 0)

    def test_blank_image_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "blank.png"
            Image.new("L", (160, 120), 64).save(source)
            group = CameraGroup("Test", "Blank", "", 160, 120)
            calibration, record = estimate_runtime_calibration([source], group)
            self.assertIsNone(calibration)
            self.assertEqual(record["status"], "rejected_insufficient_structure")

    def test_single_curved_grid_can_produce_a_validated_model(self) -> None:
        width, height = 320, 240
        matrix = np.asarray([[320.0, 0.0, width / 2], [0.0, 320.0, height / 2], [0.0, 0.0, 1.0]])
        grid = np.zeros((height, width), dtype=np.uint8)
        for x in range(12, width, 24):
            cv2.line(grid, (x, 0), (x, height - 1), 255, 2)
        for y in range(10, height, 22):
            cv2.line(grid, (0, y), (width - 1, y), 255, 2)
        curved = cv2.undistort(grid, matrix, np.asarray([0.32, -0.05, 0.0, 0.0, 0.0]), None, matrix)
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "grid.png"
            Image.fromarray(curved).save(source)
            group = CameraGroup("Test", "Grid", "", width, height)
            calibration, record = estimate_runtime_calibration([source], group)
            self.assertIsNotNone(calibration, record)
            self.assertEqual(record["status"], "accepted")
            self.assertGreater(record["candidate"]["validation_improvement"], 0.15)

    def test_one_disagreeing_sample_tries_fourth_then_consistent_two(self) -> None:
        group = CameraGroup("Test", "Camera", "", 160, 120)
        paths = [Path(f"sample_{index}.png") for index in range(4)]
        accepted = RuntimeCalibration(
            [[160.0, 0.0, 80.0], [0.0, 160.0, 60.0], [0.0, 0.0, 1.0]],
            [-0.1, 0.0, 0.0, 0.0, 0.0],
        )

        def fake_traces(_gray: np.ndarray, image_index: int = 0) -> list[CurveTrace]:
            return [
                CurveTrace(
                    points=np.asarray([[0.0, float(offset)], [159.0, float(offset)]], dtype=np.float64),
                    span=159.0,
                    support=159.0,
                    family=0,
                    image_index=image_index,
                )
                for offset in range(4)
            ]

        first_attempt = {
            "status": "rejected_cross_image_disagreement",
            "reason": "The fitted model was not independently supported by every selected image.",
            "selected_images": [path.name for path in paths],
            "disagreeing_images": [paths[0].name],
        }
        second_attempt = {
            "status": "rejected_cross_image_disagreement",
            "selected_images": [paths[1].name, paths[2].name, paths[3].name],
            "disagreeing_images": [paths[3].name],
        }
        third_attempt = {
            "status": "accepted",
            "selected_images": [paths[1].name, paths[2].name],
            "candidate": {"validation_improvement": 0.3},
        }
        with (
            patch(
                "pvrt.dataops.plumb_line_calibration._read_gray",
                return_value=(np.zeros((120, 160), dtype=np.uint8), 1.0),
            ),
            patch("pvrt.dataops.plumb_line_calibration.trace_straight_structures", side_effect=fake_traces),
            patch(
                "pvrt.dataops.plumb_line_calibration._fit_selected_images",
                side_effect=[
                    (None, first_attempt),
                    (None, second_attempt),
                    (accepted, third_attempt),
                ],
            ) as fit_mock,
        ):
            calibration, record = estimate_runtime_calibration(paths, group)
        self.assertEqual(calibration, accepted)
        self.assertTrue(record["fallback_used"])
        self.assertEqual(record["fallback_stage"], "two_image_consensus")
        self.assertEqual(record["fallback_removed_images"], [paths[0].name])
        self.assertEqual(record["fallback_replacement_image"], paths[3].name)
        self.assertEqual(fit_mock.call_count, 3)

    def test_rotation_passes_optional_runtime_flag_and_propagates_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            session = root / "session"
            source.mkdir()
            session.mkdir()
            (session / "camera_meta.json").write_text(
                json.dumps({"thermal.jpg": {"lat": 1, "lon": 1}}), encoding="utf-8"
            )
            failed = SimpleNamespace(
                returncode=1,
                stdout="",
                stderr=(
                    "Automatic lens correction was not applied. No images were modified. "
                    "Expand ‘Advanced’ above and untick the option."
                ),
            )
            with patch("pvrt.web.mosaic.subprocess.run", return_value=failed) as run_mock:
                with self.assertRaisesRegex(RuntimeError, "Automatic lens correction was not applied"):
                    prepare_rotation_and_mosaic(
                        input_type="images",
                        session_dir=session,
                        out_root=session,
                        camera_meta={"thermal.jpg": {"lat": 1, "lon": 1}},
                        mosaic_enabled=False,
                        ds_dir=source,
                        model_is_thermal=False,
                        undistort_thermal=True,
                        export_undistorted_images=True,
                        tile_tif_func=lambda *_args, **_kwargs: None,
                        run_images_dir=source,
                        tiles_dir=None,
                        tif_src=None,
                    )
                command = run_mock.call_args.args[0]
                self.assertIn("--correct-lens-distortion", command)
                self.assertIn("--export-undistorted-images", command)

    def test_orthophoto_bypasses_runtime_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch("pvrt.web.mosaic.subprocess.run") as run_mock:
                result = prepare_rotation_and_mosaic(
                    input_type="tif",
                    session_dir=root,
                    out_root=root,
                    camera_meta={},
                    mosaic_enabled=False,
                    ds_dir=root,
                    model_is_thermal=False,
                    undistort_thermal=True,
                    tile_tif_func=lambda *_args, **_kwargs: None,
                    run_images_dir=root,
                    tiles_dir=root / "tiles",
                    tif_src=root / "orthophoto.tif",
                )
            run_mock.assert_not_called()
            self.assertEqual(result.input_type, "tif")

    def test_approximate_mosaic_requires_two_geolocated_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(RuntimeError, "at least two images with readable GPS"):
                prepare_rotation_and_mosaic(
                    input_type="images",
                    session_dir=root,
                    out_root=root,
                    camera_meta={"one.jpg": {"lat": 1.0, "lon": 2.0}},
                    mosaic_enabled=True,
                    ds_dir=root,
                    model_is_thermal=False,
                    undistort_thermal=False,
                    tile_tif_func=lambda *_args, **_kwargs: None,
                    run_images_dir=root,
                    tiles_dir=None,
                    tif_src=None,
                )

    def test_mosaic_can_be_created_while_individual_images_remain_inference_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()
            rotated = root / "rotated_images"
            rotated.mkdir()
            for name in ("one.png", "two.png"):
                Image.new("RGB", (32, 24), "white").save(rotated / name)
            camera_meta = {
                "one.png": {"lat": 1.0, "lon": 2.0},
                "two.png": {"lat": 1.0001, "lon": 2.0001},
            }
            (root / "camera_meta.json").write_text(json.dumps(camera_meta), encoding="utf-8")

            def create_mosaic(*_args, **kwargs):
                output = Path(kwargs["out_mosaic_path"])
                output.touch()
                return output

            def create_tiles(_source, destination, **_kwargs):
                destination.mkdir(parents=True, exist_ok=True)
                (destination / "tile.png").touch()

            completed = SimpleNamespace(returncode=0, stdout="", stderr="")
            with patch("pvrt.web.mosaic.subprocess.run", return_value=completed), patch(
                "pvrt.web.mosaic.create_mosaic_from_rotated_images", side_effect=create_mosaic
            ):
                result = prepare_rotation_and_mosaic(
                    input_type="images",
                    session_dir=root,
                    out_root=root,
                    camera_meta=camera_meta,
                    mosaic_enabled=True,
                    inference_source="individual",
                    ds_dir=source,
                    model_is_thermal=False,
                    undistort_thermal=False,
                    tile_tif_func=create_tiles,
                    run_images_dir=source,
                    tiles_dir=None,
                    tif_src=None,
                )

            self.assertEqual(result.input_type, "images")
            self.assertEqual(result.run_images_dir, rotated)
            self.assertEqual(result.tif_src, root / "mosaic.tif")


if __name__ == "__main__":
    unittest.main()
