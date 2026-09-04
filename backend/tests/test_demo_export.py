import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from pvrt.web.demo_export import create_solar_demo_export, delete_solar_demo_export


class DemoExportTests(unittest.TestCase):
    def test_creates_expected_archive_and_requires_replace_confirmation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sessions = root / "outputs"
            job_dir = sessions / ".postprocess_jobs" / "demo-job"
            segmentation_workspace = job_dir / "snapshots" / "segmentation"
            anomaly_workspace = job_dir / "snapshots" / "anomaly"
            segmentation_workflow = segmentation_workspace / "postprocess" / "panels"
            anomaly_workflow = anomaly_workspace / "postprocess" / "anomalies"
            anomaly_result = sessions / "anomaly-result"
            rotated_images = anomaly_result / "rotated_images"
            for directory in (segmentation_workflow, anomaly_workflow, rotated_images):
                directory.mkdir(parents=True)

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[8.0, 49.0], [8.1, 49.0], [8.1, 49.1], [8.0, 49.0]]],
                },
                "properties": {"image": "panel_1"},
            }
            collection = {"type": "FeatureCollection", "features": [feature]}
            panels = segmentation_workflow / "regularized.geojson"
            rows = segmentation_workflow / "solar_rows.geojson"
            anomalies = anomaly_workflow / "associated.geojson"
            images = anomaly_result / "images.geojson"
            for path in (panels, rows, anomalies, images):
                path.write_text(json.dumps(collection), encoding="utf-8")
            (rotated_images / "panel_1.png").write_bytes(b"png-data")

            (job_dir / "job.json").write_text(json.dumps({
                "sources": {"anomaly": {"result_id": "anomaly-result"}},
                "workflows": {
                    "segmentation": {"workflow_id": "panels"},
                    "anomaly": {"workflow_id": "anomalies"},
                },
            }), encoding="utf-8")
            (segmentation_workflow / "status.json").write_text(json.dumps({
                "status": "complete",
                "outputs": {
                    "regularized": {"path": "postprocess/panels/regularized.geojson"},
                    "solar_rows": {"path": "postprocess/panels/solar_rows.geojson"},
                },
            }), encoding="utf-8")
            (anomaly_workflow / "status.json").write_text(json.dumps({
                "status": "complete",
                "outputs": {"associated": {"path": "postprocess/anomalies/associated.geojson"}},
            }), encoding="utf-8")

            result = create_solar_demo_export(job_dir, sessions)

            archive_path = Path(result["path"])
            self.assertTrue(archive_path.is_file())
            self.assertEqual(result["anomaly_count"], 1)
            self.assertEqual(result["image_count"], 1)
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), {
                    "vector/solar_panels.geojson",
                    "vector/solar_rows.geojson",
                    "vector/anomalies.geojson",
                    "anomaly_overlays/images.geojson",
                    "anomaly_overlays/panel_1.png",
                })
                self.assertEqual(archive.read("anomaly_overlays/panel_1.png"), b"png-data")

            with self.assertRaises(FileExistsError):
                create_solar_demo_export(job_dir, sessions)
            replaced = create_solar_demo_export(job_dir, sessions, replace=True)
            self.assertEqual(replaced["path"], str(archive_path))
            deleted = delete_solar_demo_export(job_dir)
            self.assertEqual(deleted["path"], str(archive_path))
            self.assertGreater(deleted["deleted_size"], 0)
            self.assertFalse(archive_path.exists())
            with self.assertRaises(FileNotFoundError):
                delete_solar_demo_export(job_dir)


if __name__ == "__main__":
    unittest.main()
