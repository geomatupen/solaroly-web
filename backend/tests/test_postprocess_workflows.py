import json
import tempfile
import unittest
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform as transform_geometry

from pvrt.postprocess import associate_anomalies, build_panel_hierarchy, deduplicate_anomalies


TO_WGS84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform


def _feature(geometry, **properties):
    return {
        "type": "Feature",
        "geometry": mapping(transform_geometry(TO_WGS84, geometry)),
        "properties": properties,
    }


def _write(path: Path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")


class PostprocessWorkflowTests(unittest.TestCase):
    def test_builds_deterministic_panel_and_row_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "regularized.geojson"
            panels = [
                _feature(box(500000 + column * 1.15, 5500000 - row * 2, 500001 + column * 1.15, 5500001.8 - row * 2), class_name="panel")
                for row in range(2)
                for column in range(3)
            ]
            _write(source, panels)
            rows_path, panels_path = root / "rows.geojson", root / "panels.geojson"
            stats = build_panel_hierarchy(source, rows_path, panels_path)
            self.assertEqual(stats["row_count"], 1)
            self.assertEqual(stats["inner_row_count"], 2)
            identified = json.loads(panels_path.read_text(encoding="utf-8"))["features"]
            self.assertEqual(len(identified), 6)
            self.assertEqual({item["properties"]["row_id"] for item in identified}, {"1000"})
            self.assertEqual(len({item["properties"]["panel_id"] for item in identified}), 6)
            self.assertEqual(
                {item["properties"]["panel_id"] for item in identified},
                {"1000-A1", "1000-A2", "1000-A3", "1000-B1", "1000-B2", "1000-B3"},
            )

    def test_deduplicates_then_associates_anomalies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            anomaly_source = root / "anomalies.geojson"
            _write(anomaly_source, [
                _feature(box(500000.4, 5500000.2, 500000.8, 5500000.6), class_name="hotspot", score=0.95),
                _feature(box(500000.42, 5500000.22, 500000.82, 5500000.62), class_name="hotspot", score=0.70),
                _feature(box(500020, 5500020, 500020.4, 5500020.4), class_name="hotspot", score=0.80),
            ])
            deduplicated = root / "deduplicated.geojson"
            stats = deduplicate_anomalies(anomaly_source, deduplicated)
            self.assertEqual(stats["duplicates_removed"], 1)
            self.assertEqual(stats["output_features"], 2)

            panel_source = root / "panels.geojson"
            _write(panel_source, [
                _feature(box(500000, 5500000, 500002, 5500001), panel_id="ROW-0001-PANEL-0001", row_id="ROW-0001")
            ])
            associated = root / "associated.geojson"
            association = associate_anomalies(deduplicated, panel_source, associated)
            self.assertEqual(association["assigned"], 1)
            self.assertEqual(association["unassigned"], 1)
            output = json.loads(associated.read_text(encoding="utf-8"))["features"]
            assigned = next(item for item in output if item["properties"]["panel_id"])
            self.assertEqual(assigned["properties"]["row_id"], "ROW-0001")


if __name__ == "__main__":
    unittest.main()
