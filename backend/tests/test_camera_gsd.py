from __future__ import annotations

import math
import unittest

from pvrt.dataops.camera_geometry import compute_meters_per_pixel


class CameraGsdTests(unittest.TestCase):
    def test_35mm_equivalent_uses_image_and_full_frame_diagonals(self) -> None:
        altitude_m = 18.715
        expected = (
            altitude_m
            * math.hypot(36.0, 24.0)
            / (40.0 * math.hypot(640.0, 512.0))
        )

        actual = compute_meters_per_pixel(
            altitude_m,
            640,
            9.1,
            40.0,
            None,
            None,
            512,
        )

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(actual, 0.024699085, places=9)

    def test_real_focal_length_and_pixel_pitch_take_priority(self) -> None:
        altitude_m = 18.715
        pixel_pitch_mm = 0.012
        pixels_per_mm = 1.0 / pixel_pitch_mm
        expected = altitude_m * pixel_pitch_mm / 9.1

        actual = compute_meters_per_pixel(
            altitude_m,
            640,
            9.1,
            10.0,
            pixels_per_mm,
            4,
            512,
        )

        self.assertAlmostEqual(actual, expected, places=12)
        self.assertAlmostEqual(actual, 0.024679121, places=9)

    def test_micrometer_resolution_unit_is_supported(self) -> None:
        altitude_m = 20.0
        pixel_pitch_mm = 0.0024
        pixels_per_micrometer = 1.0 / (pixel_pitch_mm * 1000.0)

        actual = compute_meters_per_pixel(
            altitude_m,
            4000,
            8.8,
            None,
            pixels_per_micrometer,
            5,
            3000,
        )

        self.assertAlmostEqual(actual, altitude_m * pixel_pitch_mm / 8.8, places=12)

    def test_absolute_altitude_cannot_be_used_without_ground_elevation(self) -> None:
        actual = compute_meters_per_pixel(
            None,
            640,
            9.1,
            40.0,
            None,
            None,
            512,
        )

        self.assertIsNone(actual)


if __name__ == "__main__":
    unittest.main()
