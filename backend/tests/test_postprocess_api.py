import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pvrt.web.postprocess import create_postprocess_router


class PostprocessApiTests(unittest.TestCase):
    def test_delete_edited_layer_preserves_base_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            result_dir = sessions / "test-result"
            workflow_dir = result_dir / "postprocess" / "solar-panels"
            workflow_dir.mkdir(parents=True)
            overlays.mkdir()

            combined_path = workflow_dir / "combined.geojson"
            edited_path = workflow_dir / "combined_edited.geojson"
            combined_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            edited_path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            status = {
                "id": "solar-panels",
                "status": "complete",
                "outputs": {
                    "combined": {"path": "postprocess/solar-panels/combined.geojson"},
                    "edited": {"path": "postprocess/solar-panels/combined_edited.geojson"},
                },
                "manual_revisions": [{"id": "edited", "source_stage": "combined"}],
            }
            (workflow_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")

            overlay_dir = overlays / "postprocess-solar-panels-edited"
            overlay_dir.mkdir()
            (overlay_dir / ".overlay_meta.json").write_text(json.dumps({
                "reference_kind": "postprocess",
                "source_result": "test-result",
                "workflow_id": "solar-panels",
                "stage": "edited",
            }), encoding="utf-8")

            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item
                for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/{workflow_id}/edited"
                and "DELETE" in item.methods
            )
            payload = asyncio.run(route.endpoint("test-result", "solar-panels"))

            self.assertTrue(combined_path.is_file())
            self.assertFalse(edited_path.exists())
            self.assertFalse(overlay_dir.exists())
            self.assertIn("combined", payload["outputs"])
            self.assertNotIn("edited", payload["outputs"])
            self.assertEqual(payload["manual_revisions"], [])


if __name__ == "__main__":
    unittest.main()
