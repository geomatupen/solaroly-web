import json
import asyncio
import tempfile
import unittest
from pathlib import Path

import rasterio
from pyproj import Geod
from rasterio.crs import CRS
from rasterio.transform import from_origin

from pvrt.web.app import _build_tile_layer_defs
from pvrt.web.postprocess_layer_move import (
    archive_postprocess_outputs,
    create_postprocess_layer_move_router,
    translate_geojson_payload,
)


class JsonRequest:
    def __init__(self, payload):
        self.payload = payload

    async def json(self):
        return self.payload


class PostprocessLayerMoveTests(unittest.TestCase):
    def test_move_endpoint_updates_layer_and_archives_downstream_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job"
            workspace = job / "snapshots" / "segmentation"
            workflow = workspace / "postprocess" / "workflow"
            workflow.mkdir(parents=True)
            regularized = workflow / "regularized.geojson"
            rows = workflow / "rows.geojson"
            base = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [8.0, 49.0]},
                    "properties": {},
                }],
            }
            regularized.write_text(json.dumps(base), encoding="utf-8")
            rows.write_text(json.dumps(base), encoding="utf-8")
            status = {
                "outputs": {
                    "regularized": {"path": "postprocess/workflow/regularized.geojson"},
                    "solar_rows": {"path": "postprocess/workflow/rows.geojson"},
                },
                "status": "complete",
            }
            (workflow / "status.json").write_text(json.dumps(status), encoding="utf-8")
            metadata = {
                "sources": {"segmentation": {"workspace_result_id": "workspace-seg"}},
                "workflows": {"segmentation": {"workflow_id": "workflow"}},
            }
            (job / "job.json").write_text(json.dumps(metadata), encoding="utf-8")
            request = JsonRequest({
                "kind": "segmentation",
                "layer_type": "geojson",
                "stage": "regularized",
                "east_m": 4,
                "north_m": 0,
                "confirm_move": True,
            })
            def write_json(path, payload):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(payload), encoding="utf-8")

            router = create_postprocess_layer_move_router(
                lambda _job_id: (job, metadata),
                lambda _workspace_id: workspace,
                write_json,
            )
            endpoint = next(route.endpoint for route in router.routes if route.path.endswith("/move-layer"))
            response = asyncio.run(endpoint("job", request))
            moved = json.loads(regularized.read_text(encoding="utf-8"))
            self.assertGreater(moved["features"][0]["geometry"]["coordinates"][0], 8.0)
            self.assertFalse(rows.exists())
            self.assertEqual(response["archived"][0]["stage"], "solar_rows")
            self.assertIn("outdated_revisions", response["job"])

    def test_geojson_translation_uses_ground_metres(self):
        payload = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[8.0, 49.0], [8.00001, 49.0], [8.00001, 49.00001], [8.0, 49.0]]],
                },
                "properties": {"panel_id": "1"},
            }],
        }
        moved = translate_geojson_payload(payload, 5.0, 3.0)
        original = payload["features"][0]["geometry"]["coordinates"][0][0]
        translated = moved["features"][0]["geometry"]["coordinates"][0][0]
        azimuth, _, distance = Geod(ellps="WGS84").inv(*original, *translated)
        self.assertAlmostEqual(distance, (5.0 ** 2 + 3.0 ** 2) ** 0.5, delta=0.03)
        self.assertAlmostEqual(azimuth, 59.04, delta=0.3)
        self.assertEqual(payload["features"][0]["geometry"]["coordinates"][0][0], [8.0, 49.0])

    def test_raster_layer_definition_reports_shifted_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mosaic.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                width=10,
                height=10,
                count=1,
                dtype="uint8",
                crs=CRS.from_epsg(4326),
                transform=from_origin(8.0, 49.001, 0.0001, 0.0001),
            ) as dataset:
                dataset.write_band(1, __import__("numpy").ones((10, 10), dtype="uint8"))
            original = _build_tile_layer_defs("move-test-original", [path])[0]
            shifted = _build_tile_layer_defs(
                "move-test-shifted", [path], {"east_m": 5.0, "north_m": 3.0},
            )[0]
            original_center = (
                (original["bounds"][0][1] + original["bounds"][1][1]) / 2,
                (original["bounds"][0][0] + original["bounds"][1][0]) / 2,
            )
            shifted_center = (
                (shifted["bounds"][0][1] + shifted["bounds"][1][1]) / 2,
                (shifted["bounds"][0][0] + shifted["bounds"][1][0]) / 2,
            )
            _, _, distance = Geod(ellps="WGS84").inv(*original_center, *shifted_center)
            self.assertAlmostEqual(distance, (5.0 ** 2 + 3.0 ** 2) ** 0.5, delta=0.05)
            self.assertEqual(shifted["movement"], {"east_m": 5.0, "north_m": 3.0})

    def test_dependent_output_is_archived_and_removed_from_active_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "job"
            workspace = job / "snapshots" / "segmentation"
            workflow = workspace / "postprocess" / "workflow"
            workflow.mkdir(parents=True)
            output = workflow / "rows.geojson"
            output.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")
            status = {
                "outputs": {"solar_rows": {"path": "postprocess/workflow/rows.geojson"}},
                "status": "complete",
            }
            (workflow / "status.json").write_text(json.dumps(status), encoding="utf-8")
            revision = job / "outdated" / "revision"
            archived = archive_postprocess_outputs(
                job,
                workspace,
                "workflow",
                {"solar_rows"},
                revision,
                lambda path, payload: path.write_text(json.dumps(payload), encoding="utf-8"),
            )
            current = json.loads((workflow / "status.json").read_text(encoding="utf-8"))
            self.assertNotIn("solar_rows", current["outputs"])
            self.assertFalse(output.exists())
            self.assertEqual(len(archived), 1)
            self.assertTrue((job / archived[0]["archived_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
