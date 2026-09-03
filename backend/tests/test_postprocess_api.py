import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

from fastapi import HTTPException

from pvrt.web.postprocess import create_postprocess_router
from pvrt.web.postprocess import EditLayerRequest
from pvrt.web.postprocess import EditSourceRequest
from pvrt.web.postprocess import OverlapDeduplicateAnomaliesRequest
from pvrt.web.postprocess import UploadPanelReferenceRequest
from pvrt.web.postprocess import VisualReviewDecisionRequest


class PostprocessApiTests(unittest.TestCase):
    def test_panel_reference_upload_requires_unique_selected_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            workflow_dir = sessions / "test-result" / "postprocess" / "anomalies"
            workflow_dir.mkdir(parents=True)
            overlays.mkdir()
            (workflow_dir / "status.json").write_text(json.dumps({
                "id": "anomalies",
                "status": "complete",
                "workflow_kind": "anomaly",
                "outputs": {},
            }), encoding="utf-8")
            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/{workflow_id}/panel-reference"
                and "POST" in item.methods
            )
            polygon = {
                "type": "Polygon",
                "coordinates": [[[8.0, 49.0], [8.00001, 49.0], [8.00001, 49.00001], [8.0, 49.0]]],
            }
            uploaded = asyncio.run(route.endpoint(
                "test-result",
                "anomalies",
                UploadPanelReferenceRequest(geojson={
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "geometry": polygon, "properties": {"asset_code": "P-1"}},
                        {"type": "Feature", "geometry": polygon, "properties": {"asset_code": "P-2"}},
                    ],
                }, id_field="asset_code"),
            ))
            output = workflow_dir / "uploaded_panels.geojson"
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual([item["properties"]["panel_id"] for item in saved["features"]], ["P-1", "P-2"])
            self.assertEqual(uploaded["uploaded_panel_reference"]["feature_count"], 2)
            self.assertEqual(uploaded["outputs"]["uploaded_panels"]["url"], "/media/uploaded_panels.geojson")

            with self.assertRaises(HTTPException) as duplicate:
                asyncio.run(route.endpoint(
                    "test-result",
                    "anomalies",
                    UploadPanelReferenceRequest(geojson={
                        "type": "FeatureCollection",
                        "features": [
                            {"type": "Feature", "geometry": polygon, "properties": {"asset_code": "P-1"}},
                            {"type": "Feature", "geometry": polygon, "properties": {"asset_code": "P-1"}},
                        ],
                    }, id_field="asset_code"),
                ))
            self.assertEqual(duplicate.exception.status_code, 400)
            self.assertIn("duplicated", duplicate.exception.detail)

    def test_overlap_filter_has_its_own_replaceable_output_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            result_dir = sessions / "test-result"
            result_dir.mkdir(parents=True)
            overlays.mkdir()
            source = result_dir / "predictions.geojson"
            source.write_text(json.dumps({
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [[[8.0, 49.0], [8.00001, 49.0], [8.00001, 49.00001], [8.0, 49.00001], [8.0, 49.0]]]},
                        "properties": {"score": 0.9},
                    },
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [[[8.000002, 49.0], [8.000012, 49.0], [8.000012, 49.00001], [8.000002, 49.00001], [8.000002, 49.0]]]},
                        "properties": {"score": 0.8},
                    },
                ],
            }), encoding="utf-8")
            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/anomalies/overlap-deduplicate"
                and "POST" in item.methods
            )
            first = asyncio.run(route.endpoint(
                "test-result",
                OverlapDeduplicateAnomaliesRequest(input_path="predictions.geojson"),
            ))
            workflow_dir = result_dir / "postprocess" / first["id"]

            def completed_status():
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    status = json.loads((workflow_dir / "status.json").read_text(encoding="utf-8"))
                    if status.get("status") in {"complete", "failed"}:
                        return status
                    time.sleep(0.02)
                self.fail("Overlap filtering did not finish in time.")

            first_status = completed_status()
            self.assertEqual(first_status["status"], "complete")
            self.assertIn("overlap_deduplicated", first_status["outputs"])
            self.assertNotIn("deduplicated", first_status["outputs"])

            second = asyncio.run(route.endpoint(
                "test-result",
                OverlapDeduplicateAnomaliesRequest(
                    input_path="predictions.geojson",
                    workflow_id=first["id"],
                    minimum_overlap_percent=70,
                ),
            ))
            self.assertEqual(second["id"], first["id"])
            second_status = completed_status()
            self.assertEqual(second_status["status"], "complete")
            self.assertIn("overlap_deduplicated", second_status["outputs"])
            self.assertEqual(len(list((result_dir / "postprocess").iterdir())), 1)

    def test_visual_review_decisions_are_persisted_and_restorable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            workflow_dir = sessions / "test-result" / "postprocess" / "anomaly-review"
            workflow_dir.mkdir(parents=True)
            overlays.mkdir()
            review_path = workflow_dir / "visual_review.json"
            review_path.write_text(json.dumps({
                "pairs": [
                    {
                        "first_index": 2,
                        "second_index": 5,
                        "first_anomaly_id": "A-203",
                        "second_anomaly_id": "A-506",
                    },
                    {
                        "first_index": 5,
                        "second_index": 8,
                        "first_anomaly_id": "A-506",
                        "second_anomaly_id": "A-809",
                    },
                ],
            }), encoding="utf-8")
            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/{workflow_id}/visual-review/decision"
                and "PATCH" in item.methods
            )
            asyncio.run(route.endpoint(
                "test-result",
                "anomaly-review",
                VisualReviewDecisionRequest(
                    first_index=2,
                    second_index=5,
                    status="rejected",
                ),
            ))
            rejected = json.loads(review_path.read_text(encoding="utf-8"))["pairs"][0]
            self.assertEqual(rejected["manual_review_status"], "rejected")

            asyncio.run(route.endpoint(
                "test-result",
                "anomaly-review",
                VisualReviewDecisionRequest(
                    first_index=2,
                    second_index=5,
                    status="accepted",
                    keep_index=5,
                ),
            ))
            accepted = json.loads(review_path.read_text(encoding="utf-8"))["pairs"][0]
            self.assertEqual(accepted["manual_review_status"], "accepted")
            self.assertEqual(accepted["manual_keep_index"], 5)

            with self.assertRaises(HTTPException) as conflict:
                asyncio.run(route.endpoint(
                    "test-result",
                    "anomaly-review",
                    VisualReviewDecisionRequest(
                        first_index=5,
                        second_index=8,
                        status="accepted",
                        keep_index=8,
                    ),
                ))
            self.assertEqual(conflict.exception.status_code, 409)
            self.assertIn("Anomaly A-506 is already kept by accepted pair A-203–A-506", conflict.exception.detail)
            conflicting_pair = json.loads(review_path.read_text(encoding="utf-8"))["pairs"][1]
            self.assertNotIn("manual_review_status", conflicting_pair)

            asyncio.run(route.endpoint(
                "test-result",
                "anomaly-review",
                VisualReviewDecisionRequest(
                    first_index=2,
                    second_index=5,
                    status="unreviewed",
                ),
            ))
            restored = json.loads(review_path.read_text(encoding="utf-8"))["pairs"][0]
            self.assertNotIn("manual_review_status", restored)
            self.assertNotIn("manual_keep_index", restored)

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
                    "manual_review_status": "accepted",
                    "manual_keep_index": 0,
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
            self.assertEqual(payload["visual_review_decision_counts"]["accepted"], 1)
            self.assertEqual(payload["visual_review_conflict_indices"], [])
            self.assertEqual(payload["visual_review_conflict_ids"], [])
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

    def test_row_edits_clear_previous_assignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "sessions"
            overlays = root / "overlays"
            result_dir = sessions / "test-result"
            workflow_dir = result_dir / "postprocess" / "solar-panels"
            workflow_dir.mkdir(parents=True)
            overlays.mkdir()
            regularized_path = workflow_dir / "regularized.geojson"
            rows_path = workflow_dir / "solar_rows.geojson"
            feature = {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                "properties": {"row_id": "1000", "panel_id": "1000-A1"},
            }
            regularized_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
            rows_path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}), encoding="utf-8")
            (workflow_dir / "status.json").write_text(json.dumps({
                "id": "solar-panels",
                "status": "complete",
                "assignment_stats": {"assigned_panel_count": 1},
                "outputs": {
                    "regularized": {"path": "postprocess/solar-panels/regularized.geojson"},
                    "solar_rows": {"path": "postprocess/solar-panels/solar_rows.geojson"},
                },
            }), encoding="utf-8")
            router = create_postprocess_router(
                lambda: sessions,
                lambda: overlays,
                lambda path: f"/media/{path.name}",
            )
            route = next(
                item for item in router.routes
                if item.path == "/api/results/{result_id}/postprocess/{workflow_id}/{stage}/edits"
                and "POST" in item.methods
            )
            edited = {
                "type": "FeatureCollection",
                "features": [{
                    **feature,
                    "properties": {"row_id": "1000", "panel_count": 1},
                }],
            }

            payload = asyncio.run(route.endpoint(
                "test-result", "solar-panels", "solar_rows", EditLayerRequest(geojson=edited),
            ))

            saved_row = json.loads(rows_path.read_text(encoding="utf-8"))["features"][0]
            saved_panel = json.loads(regularized_path.read_text(encoding="utf-8"))["features"][0]
            self.assertNotIn("row_id", saved_row["properties"])
            self.assertNotIn("row_id", saved_panel["properties"])
            self.assertNotIn("panel_id", saved_panel["properties"])
            self.assertIsNone(payload["assignment_stats"])


if __name__ == "__main__":
    unittest.main()
