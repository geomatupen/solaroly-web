from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pvrt.web.app import (
    _append_mosaic_source_images_geojson,
    _camera_heading_from_entry,
)


class MosaicSourceCatalogTests(unittest.TestCase):
    def test_camera_heading_keeps_gimbal_selected_rotation(self) -> None:
        entry = {
            "rotation": 105.5,
            "rotation_gimbal": 105.5,
            "rotation_aircraft": -75.0,
        }
        self.assertEqual(_camera_heading_from_entry(entry, {}), 105.5)

    def test_mosaic_catalog_references_prepared_source_without_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp)
            rotated = session / "rotated_images"
            rotated.mkdir()
            prepared = rotated / "frame_001.png"
            Image.new("RGBA", (120, 80), (50, 60, 70, 255)).save(prepared)
            (session / "images.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": []}),
                encoding="utf-8",
            )
            camera_meta = {
                "frame_001.jpeg": {
                    "lat": 47.5,
                    "lon": 17.25,
                    "meters_per_pixel": 0.1,
                    "row_alignment_rotation_deg": 2.0,
                    "row_alignment": {"status": "aligned"},
                }
            }

            output = _append_mosaic_source_images_geojson(session, camera_meta)
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["features"]), 1)
        feature = payload["features"][0]
        self.assertEqual(feature["geometry"]["coordinates"], [17.25, 47.5])
        props = feature["properties"]
        self.assertEqual(props["source_role"], "mosaic_input")
        self.assertFalse(props["inference_performed"])
        self.assertEqual((props["w"], props["h"]), (120, 80))
        self.assertEqual(props["meters_per_pixel"], 0.1)
        self.assertEqual(props["rotation"], 2.0)
        self.assertEqual(props["alignment_status"], "aligned")
        self.assertIn("rotated_images", props["prepared_image"])


if __name__ == "__main__":
    unittest.main()
