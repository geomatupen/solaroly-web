from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from pvrt.dataops.fast_mosaic_alignment import ImagePlacement, refine_mosaic_placements


class FastMosaicAlignmentTests(unittest.TestCase):
    def test_overlapping_images_refine_their_relative_centres(self) -> None:
        rng = np.random.default_rng(73)
        scene = rng.integers(0, 70, size=(320, 720), dtype=np.uint8)
        for index in range(45):
            x = int(rng.integers(15, 705))
            y = int(rng.integers(15, 305))
            cv2.circle(scene, (x, y), int(rng.integers(3, 10)), int(rng.integers(130, 255)), -1)
        for x in range(20, 700, 55):
            cv2.line(scene, (x, 10), (x + 25, 310), 210, 2)

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first_path = root / "first.png"
            second_path = root / "second.png"
            third_path = root / "third.png"
            Image.fromarray(scene[:, :420]).save(first_path)
            Image.fromarray(scene[:, 180:600]).save(second_path)
            Image.fromarray(scene[:, 300:720]).save(third_path)
            placements = [
                ImagePlacement("first.png", first_path, 220.0, 180.0, 420, 320),
                # GPS starts 24 px short of the true 180 px centre displacement.
                ImagePlacement("second.png", second_path, 376.0, 180.0, 420, 320),
                ImagePlacement("third.png", third_path, 510.0, 180.0, 420, 320),
            ]
            refined, report = refine_mosaic_placements(placements)

        self.assertEqual(report["status"], "refined")
        self.assertGreaterEqual(report["accepted_pair_count"], 1)
        relative_x = refined["second.png"][0] - refined["first.png"][0]
        self.assertAlmostEqual(relative_x, 180.0, delta=3.0)
        self.assertAlmostEqual(
            refined["second.png"][1] - refined["first.png"][1],
            0.0,
            delta=2.0,
        )

    def test_pixel_identical_images_are_not_alignment_constraints(self) -> None:
        rng = np.random.default_rng(9)
        image = rng.integers(0, 255, size=(180, 240), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / f"duplicate_{index}.png" for index in range(3)]
            for path in paths:
                Image.fromarray(image).save(path)
            placements = [
                ImagePlacement(path.name, path, 100.0 + index * 60.0, 100.0, 240, 180)
                for index, path in enumerate(paths)
            ]
            refined, report = refine_mosaic_placements(placements)

        self.assertEqual(report["status"], "gps_only")
        self.assertTrue(report["pairs"])
        self.assertTrue(all(pair["status"] == "rejected_duplicate" for pair in report["pairs"]))
        for placement in placements:
            self.assertEqual(refined[placement.name], (placement.center_x, placement.center_y))

    def test_untextured_images_keep_gps_positions(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / "first.png", root / "second.png"]
            for path in paths:
                Image.new("L", (200, 160), 80).save(path)
            placements = [
                ImagePlacement(paths[0].name, paths[0], 100.0, 100.0, 200, 160),
                ImagePlacement(paths[1].name, paths[1], 180.0, 100.0, 200, 160),
            ]
            refined, report = refine_mosaic_placements(placements)

        self.assertEqual(report["status"], "gps_only")
        self.assertEqual(refined[paths[0].name], (100.0, 100.0))
        self.assertEqual(refined[paths[1].name], (180.0, 100.0))


if __name__ == "__main__":
    unittest.main()
