import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from pvrt.web.postprocess import create_postprocess_router
from pvrt.web.postprocess import EditLayerRequest
from pvrt.web.postprocess import EditSourceRequest


class PostprocessApiTests(unittest.TestCase):
    def test_saved_visual_review_is_recovered_after_status_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            workflow_dir = sessions / "test-result" / "postprocess" / "anomaly-review"
            workflow_dir.mkdir(parents=True)
            overlays.mkdir()
            source = sessions / "test-result" / "source.geojson"
            source.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            (workflow_dir / "visual_review.json").write_text(json.dumps({
                "pairs": [{
                    "first_index": 0,
                    "second_index": 1,
                    "appearance_similarity": 0.9,
                }],
            }), encoding="utf-8")
            (workflow_dir / "status.json").write_text(json.dumps({
                "id": "anomaly-review",
                "workflow_kind": "anomaly",
                "status": "complete",
                "input_path": "source.geojson",
                "parameters": {
                    "neighbor_image_radius_m": 25,
                    "maximum_center_distance_m": 5,
                },
                "outputs": {},
            }), encoding="utf-8")
            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/{workflow_id}"
                and "GET" in item.methods
            )
            payload = asyncio.run(route.endpoint("test-result", "anomaly-review"))
            self.assertEqual(payload["visual_review"]["total_pairs"], 1)
            self.assertTrue(payload["visual_review_available"])
            self.assertEqual(payload["visual_review_total_pairs"], 1)
            self.assertEqual(payload["visual_analysis_stats"]["visually_compared_pairs"], 1)
            self.assertTrue(payload["visual_analysis_stats"]["recovered_from_saved_review"])

    def test_job_snapshot_workflow_resolves_without_original_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            workflow_dir = (
                sessions / ".postprocess_jobs" / "self_contained_job"
                / "snapshots" / "segmentation" / "postprocess" / "panels"
            )
            workflow_dir.mkdir(parents=True)
            overlays.mkdir()
            source = workflow_dir.parent.parent / "source.geojson"
            source.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            output = workflow_dir / "regularized.geojson"
            output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            (workflow_dir / "status.json").write_text(json.dumps({
                "id": "panels",
                "status": "complete",
                "input_path": "source.geojson",
                "outputs": {"regularized": {"path": "postprocess/panels/regularized.geojson"}},
            }), encoding="utf-8")
            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/{workflow_id}"
                and "GET" in item.methods
            )
            payload = asyncio.run(route.endpoint("ppjob__self_contained_job__segmentation", "panels"))
            self.assertEqual(payload["id"], "panels")
            self.assertEqual(payload["outputs"]["regularized"]["url"], "/media/regularized.geojson")

    def test_source_edits_create_working_copy_and_preserve_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            result_dir = sessions / "test-result"
            result_dir.mkdir(parents=True)
            overlays.mkdir()
            source = result_dir / "predictions.geojson"
            original = {"type": "FeatureCollection", "features": []}
            source.write_text(json.dumps(original), encoding="utf-8")
            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/source-edits"
                and "POST" in item.methods
            )
            edited = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                    "properties": {},
                }],
            }
            payload = asyncio.run(route.endpoint(
                "test-result",
                EditSourceRequest(input_path="predictions.geojson", geojson=edited),
            ))
            copied = result_dir / payload["outputs"]["source"]["path"]
            self.assertTrue(copied.is_file())
            self.assertEqual(json.loads(source.read_text(encoding="utf-8")), original)
            self.assertTrue(json.loads(copied.read_text(encoding="utf-8"))["features"][0]["properties"]["manually_edited"])

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
