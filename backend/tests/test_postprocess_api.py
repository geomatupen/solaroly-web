import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pvrt.web.postprocess import create_postprocess_router
from pvrt.web.postprocess import EditLayerRequest


class PostprocessApiTests(unittest.TestCase):
    def test_layer_edits_update_selected_geojson_without_creating_a_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            result_dir = sessions / "test-result"
            workflow_dir = result_dir / "postprocess" / "solar-panels"
            workflow_dir.mkdir(parents=True)
            overlays.mkdir()

            combined_path = workflow_dir / "combined.geojson"
            combined_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            status = {
                "id": "solar-panels",
                "status": "complete",
                "outputs": {
                    "combined": {"path": "postprocess/solar-panels/combined.geojson"},
                },
            }
            (workflow_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item
                for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/{workflow_id}/{stage}/edits"
                and "POST" in item.methods
            )
            edited_geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                    },
                    "properties": {"row_id": "1000"},
                }],
            }
            payload = asyncio.run(route.endpoint(
                "test-result",
                "solar-panels",
                "combined",
                EditLayerRequest(geojson=edited_geojson),
            ))

            self.assertTrue(combined_path.is_file())
            saved = json.loads(combined_path.read_text(encoding="utf-8"))
            self.assertEqual(len(saved["features"]), 1)
            self.assertTrue(saved["features"][0]["properties"]["manually_edited"])
            self.assertEqual(payload["manual_edits"]["combined"]["feature_count"], 1)
            self.assertFalse((workflow_dir / "combined_edited.geojson").exists())


if __name__ == "__main__":
    unittest.main()
