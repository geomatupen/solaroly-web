import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from pyproj import Transformer
from shapely.affinity import rotate
from shapely.geometry import box, mapping
from shapely.ops import transform as transform_geometry

from pvrt.postprocess import (
    analyze_visual_duplicates,
    apply_visual_deduplication,
    associate_anomalies,
    build_panel_hierarchy,
    deduplicate_anomalies,
    image_neighbor_statistics,
)
from pvrt.postprocess.anomaly import _orientation_similarity


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
    def test_orientation_similarity_treats_parallel_axes_as_matching(self):
        horizontal = box(0, 0, 4, 1)
        self.assertEqual(_orientation_similarity(horizontal, rotate(horizontal, 180)), 1.0)
        self.assertAlmostEqual(_orientation_similarity(horizontal, rotate(horizontal, 10)), 0.8889, places=4)
        self.assertEqual(_orientation_similarity(horizontal, rotate(horizontal, 90)), 0.0)

    def test_representative_weights_choose_which_duplicate_polygon_is_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.geojson"
            _write(source, [
                _feature(box(500000, 5500000, 500000.4, 5500000.4), anomaly_id="hot-edge", class_name="hotspot", score=0.95, image="edge.jpg"),
                _feature(box(500001, 5500000, 500001.4, 5500000.4), anomaly_id="hot-center", class_name="hotspot", score=0.50, image="center.jpg"),
            ])
            review = root / "visual_review.json"
            review.write_text(json.dumps({
                "pairs": [{
                    "first_index": 0,
                    "second_index": 1,
                    "first_image": "edge.jpg",
                    "second_image": "center.jpg",
                    "appearance_similarity": 0.95,
                    "context_similarity": 0.95,
                    "shape_similarity": 1.0,
                    "size_similarity": 1.0,
                    "proximity_similarity": 0.80,
                }],
                "representative_components": {
                    "0": {"image_center_proximity": 0.10, "model_confidence": 0.95},
                    "1": {"image_center_proximity": 0.90, "model_confidence": 0.50},
                },
            }), encoding="utf-8")
            center_output = root / "center-weighted.geojson"
            apply_visual_deduplication(
                source,
                review,
                center_output,
                representative_weights={"image_center": 1.0, "spatial_centrality": 0.0, "model_confidence": 0.0},
            )
            center_kept = json.loads(center_output.read_text(encoding="utf-8"))["features"]
            self.assertEqual(center_kept[0]["properties"]["anomaly_id"], "hot-center")
            self.assertNotIn("source_anomaly_index", center_kept[0]["properties"])

            confidence_output = root / "confidence-weighted.geojson"
            apply_visual_deduplication(
                source,
                review,
                confidence_output,
                representative_weights={"image_center": 0.0, "spatial_centrality": 0.0, "model_confidence": 1.0},
            )
            confidence_kept = json.loads(confidence_output.read_text(encoding="utf-8"))["features"]
            self.assertEqual(confidence_kept[0]["properties"]["anomaly_id"], "hot-edge")

            manual_output = root / "manual.geojson"
            apply_visual_deduplication(
                source,
                review,
                manual_output,
                manual_decisions=[{"first_index": 0, "second_index": 1, "keep_index": 1}],
            )
            manual_kept = json.loads(manual_output.read_text(encoding="utf-8"))["features"]
            self.assertEqual(manual_kept[0]["properties"]["anomaly_id"], "hot-center")
            self.assertEqual(manual_kept[0]["properties"]["deduplication_method"], "manual")

    def test_image_neighbor_statistics_use_center_radius(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.geojson"
            _write(source, [
                _feature(box(500000, 5500000, 500000.4, 5500000.4), image="first.jpg"),
                _feature(box(500010, 5500000, 500010.4, 5500000.4), image="second.jpg"),
                _feature(box(500030, 5500000, 500030.4, 5500000.4), image="third.jpg"),
            ])
            stats = image_neighbor_statistics(source, root, 15.0)
            self.assertEqual(stats["image_count"], 3)
            self.assertAlmostEqual(stats["average_neighbors"], 0.67, places=2)
            self.assertEqual(stats["minimum_neighbors"], 0)
            self.assertEqual(stats["maximum_neighbors"], 1)
            self.assertEqual(stats["isolated_images"], 1)

    def test_visual_deduplication_applies_configurable_component_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.geojson"
            _write(source, [
                _feature(box(500000, 5500000, 500000.4, 5500000.4), class_name="hotspot", score=0.95, image="first.jpg"),
                _feature(box(500001, 5500000, 500001.4, 5500000.4), class_name="hotspot", score=0.85, image="second.jpg"),
            ])
            review = root / "visual_review.json"
            review.write_text(json.dumps({"pairs": [{
                "first_index": 0,
                "second_index": 1,
                "first_image": "first.jpg",
                "second_image": "second.jpg",
                "appearance_similarity": 0.90,
                "context_similarity": 0.90,
                "shape_similarity": 0.10,
                "size_similarity": 0.10,
                "proximity_similarity": 0.10,
            }]}), encoding="utf-8")
            appearance_output = root / "appearance.geojson"
            appearance_result = apply_visual_deduplication(
                source,
                review,
                appearance_output,
                weights={"appearance": 1.0, "context": 0.0, "shape": 0.0, "size": 0.0, "proximity": 0.0},
            )
            self.assertEqual(appearance_result["duplicates_removed"], 1)
            shape_output = root / "shape.geojson"
            shape_result = apply_visual_deduplication(
                source,
                review,
                shape_output,
                weights={"appearance": 0.0, "context": 0.0, "shape": 1.0, "size": 0.0, "proximity": 0.0},
            )
            self.assertEqual(shape_result["duplicates_removed"], 0)

    def test_visual_deduplication_uses_only_the_weighted_score_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.geojson"
            _write(source, [
                _feature(box(500000, 5500000, 500000.4, 5500000.4), class_name="hotspot", score=0.95),
                _feature(box(500001, 5500000, 500001.4, 5500000.4), class_name="hotspot", score=0.85),
            ])
            review = root / "visual_review.json"
            review.write_text(json.dumps({"pairs": [{
                "first_index": 0,
                "second_index": 1,
                "appearance_similarity": 0.10,
                "context_similarity": 0.10,
                "shape_similarity": 0.90,
                "size_similarity": 0.10,
                "proximity_similarity": 0.10,
            }]}), encoding="utf-8")
            result = apply_visual_deduplication(
                source,
                review,
                root / "weighted.geojson",
                duplicate_score_threshold=0.80,
                weights={"appearance": 0.0, "context": 0.0, "shape": 1.0, "size": 0.0, "proximity": 0.0},
            )
            self.assertEqual(result["duplicates_removed"], 1)

    def test_overlap_only_deduplication_obeys_the_configured_percentage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.geojson"
            _write(source, [
                _feature(box(500000, 5500000, 500001, 5500001), class_name="hotspot", score=0.95),
                _feature(box(500000.4, 5500000, 500001.4, 5500001), class_name="hotspot", score=0.85),
            ])
            strict = deduplicate_anomalies(
                source,
                root / "strict.geojson",
                minimum_smaller_overlap=0.70,
                overlap_only=True,
            )
            permissive = deduplicate_anomalies(
                source,
                root / "permissive.geojson",
                minimum_smaller_overlap=0.50,
                overlap_only=True,
            )
            self.assertEqual(strict["duplicates_removed"], 0)
            self.assertEqual(permissive["duplicates_removed"], 1)

    def test_visual_deduplication_keeps_candidates_when_images_are_unavailable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "predictions.geojson"
            _write(source, [
                _feature(box(500000.4, 5500000.2, 500000.8, 5500000.6), class_name="hotspot", score=0.95, image="first.jpg"),
                _feature(box(500002.0, 5500000.2, 500002.4, 5500000.6), class_name="hotspot", score=0.85, image="second.jpg"),
            ])
            workflow_dir = root / "postprocess" / "visual"
            review_path = workflow_dir / "visual_review.json"
            stats = analyze_visual_duplicates(source, review_path, workflow_dir / "review_images", root)
            self.assertEqual(stats["spatial_candidate_pairs"], 1)
            self.assertEqual(stats["missing_image_pairs"], 1)
            pair = json.loads(review_path.read_text(encoding="utf-8"))["pairs"][0]
            self.assertEqual(pair["iou"], 0.0)
            self.assertGreater(pair["center_distance_m"], 1.0)
            output = workflow_dir / "deduplicated.geojson"
            applied = apply_visual_deduplication(source, review_path, output)
            self.assertEqual(applied["duplicates_removed"], 0)
            self.assertEqual(applied["output_features"], 2)

    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV is not installed in this test environment")
    def test_visual_duplicate_review_precedes_threshold_application(self):
        import cv2
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_dir = root / "result"
            overlays = result_dir / "overlays"
            overlays.mkdir(parents=True)
            first_geometry = box(500000.4, 5500000.2, 500000.8, 5500000.6)
            second_geometry = box(500000.42, 5500000.22, 500000.82, 5500000.62)
            anomaly_source = result_dir / "predictions.geojson"
            _write(anomaly_source, [
                _feature(first_geometry, class_name="hotspot", score=0.95, image="first.jpg"),
                _feature(second_geometry, class_name="hotspot", score=0.85, image="second.jpg"),
            ])
            footprint = transform_geometry(TO_WGS84, box(499998, 5499998, 500003, 5500003))
            min_x, min_y, max_x, max_y = footprint.bounds
            corners = [[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y]]
            images_payload = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "geometry": None, "properties": {"image": name, "corners": corners}}
                    for name in ("first.jpg", "second.jpg")
                ],
            }
            (result_dir / "images.geojson").write_text(json.dumps(images_payload), encoding="utf-8")
            image = np.zeros((240, 240, 3), dtype=np.uint8)
            cv2.rectangle(image, (75, 75), (165, 165), (180, 180, 180), -1)
            cv2.circle(image, (120, 120), 18, (245, 245, 245), -1)
            cv2.imwrite(str(overlays / "first.png"), image)
            cv2.imwrite(str(overlays / "second.png"), image)

            workflow_dir = result_dir / "postprocess" / "visual"
            review_path = workflow_dir / "visual_review.json"
            stats = analyze_visual_duplicates(
                anomaly_source,
                review_path,
                workflow_dir / "review_images",
                result_dir,
            )
            self.assertEqual(stats["spatial_candidate_pairs"], 1)
            self.assertEqual(stats["visually_compared_pairs"], 1)
            pair = json.loads(review_path.read_text(encoding="utf-8"))["pairs"][0]
            self.assertGreaterEqual(pair["visual_similarity"], 0.80)
            self.assertGreaterEqual(pair["duplicate_score"], 0.80)
            self.assertIn("context_similarity", pair)
            self.assertIn("shape_similarity", pair)
            self.assertIn("orientation_similarity", pair)

            output = workflow_dir / "deduplicated.geojson"
            applied = apply_visual_deduplication(
                anomaly_source,
                review_path,
                output,
                duplicate_score_threshold=0.80,
            )
            self.assertEqual(applied["duplicates_removed"], 1)
            self.assertEqual(applied["output_features"], 1)

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
            hierarchy_path = root / "panel_hierarchy.geojson"
            rows_path = root / "solar_rows.geojson"
            panels_path = root / "solar_panels.geojson"
            stats = build_panel_hierarchy(
                source,
                hierarchy_path,
                rows_output_path=rows_path,
                panels_output_path=panels_path,
            )
            self.assertEqual(stats["row_count"], 1)
            self.assertEqual(stats["inner_row_count"], 2)
            hierarchy = json.loads(hierarchy_path.read_text(encoding="utf-8"))["features"]
            identified = [item for item in hierarchy if item["properties"].get("panel_id")]
            rows = [item for item in hierarchy if item["properties"].get("postprocess_stage") == "panel_rows"]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["geometry"]["type"], "Polygon")
            self.assertEqual(len(rows[0]["geometry"]["coordinates"]), 1)
            self.assertEqual(len(rows[0]["geometry"]["coordinates"][0]), 5)
            self.assertEqual(len(json.loads(rows_path.read_text(encoding="utf-8"))["features"]), 1)
            self.assertEqual(len(json.loads(panels_path.read_text(encoding="utf-8"))["features"]), 6)
            self.assertEqual(len(identified), 6)
            self.assertEqual({item["properties"]["row_id"] for item in identified}, {"1000"})
            self.assertEqual(len({item["properties"]["panel_id"] for item in identified}), 6)
            self.assertEqual(
                {item["properties"]["panel_id"] for item in identified},
                {"1000-A1", "1000-A2", "1000-A3", "1000-B1", "1000-B2", "1000-B3"},
            )

    def test_row_ids_follow_map_reading_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "regularized.geojson"
            locations = [
                ("top_left", 500000.0, 5500100.0),
                ("top_right", 500010.0, 5500099.7),
                ("bottom_left", 500000.0, 5500090.0),
                ("bottom_right", 500010.0, 5500090.2),
            ]
            _write(source, [
                _feature(box(x, y, x + 1.0, y + 1.8), class_name="panel", marker=marker)
                for marker, x, y in locations
            ])
            hierarchy_path = root / "panel_hierarchy.geojson"
            build_panel_hierarchy(source, hierarchy_path)
            features = json.loads(hierarchy_path.read_text(encoding="utf-8"))["features"]
            panels = [item["properties"] for item in features if item["properties"].get("panel_id")]
            row_by_marker = {item["marker"]: item["row_id"] for item in panels}
            self.assertEqual(row_by_marker, {
                "top_left": "1000",
                "top_right": "1001",
                "bottom_left": "1002",
                "bottom_right": "1003",
            })

    def test_assigns_panel_ids_in_place_and_writes_rows_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regularized_path = root / "regularized.geojson"
            rows_path = root / "solar_rows.geojson"
            _write(regularized_path, [
                _feature(box(500000 + column * 1.15, 5500000, 500001 + column * 1.15, 5500001.8), class_name="panel")
                for column in range(3)
            ])

            build_panel_hierarchy(
                regularized_path,
                None,
                rows_output_path=rows_path,
                panels_output_path=regularized_path,
            )

            panels = json.loads(regularized_path.read_text(encoding="utf-8"))["features"]
            rows = json.loads(rows_path.read_text(encoding="utf-8"))["features"]
            self.assertEqual(len(panels), 3)
            self.assertTrue(all(item["properties"].get("panel_id") for item in panels))
            self.assertEqual({item["properties"]["row_id"] for item in panels}, {"1000"})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["properties"]["row_id"], "1000")
            self.assertFalse((root / "panel_hierarchy.geojson").exists())

    def test_merges_candidate_row_contained_inside_outer_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regularized_path = root / "regularized.geojson"
            rows_path = root / "solar_rows.geojson"
            panels = [
                _feature(box(500000 + column * 1.15, 5500000, 500001 + column * 1.15, 5500001.8), class_name="panel")
                for column in range(3)
            ]
            panels.append(_feature(
                rotate(box(500001.3, 5500000.7, 500001.7, 5500001.1), 45),
                class_name="panel",
            ))
            _write(regularized_path, panels)

            stats = build_panel_hierarchy(
                regularized_path,
                None,
                rows_output_path=rows_path,
                panels_output_path=regularized_path,
            )

            rows = json.loads(rows_path.read_text(encoding="utf-8"))["features"]
            identified = json.loads(regularized_path.read_text(encoding="utf-8"))["features"]
            self.assertEqual(stats["row_count"], 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]["geometry"]["coordinates"]), 1)
            self.assertEqual({item["properties"]["row_id"] for item in identified}, {"1000"})

    def test_absorbs_candidate_rows_above_overlap_threshold(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "regularized.geojson"
            _write(source, [
                _feature(box(500000, 5500000, 500001, 5500001.8), class_name="first"),
                _feature(box(500000.75, 5500000, 500001.75, 5500001.8), class_name="second"),
            ])

            default_stats = build_panel_hierarchy(source, root / "default.geojson")
            strict_stats = build_panel_hierarchy(
                source,
                root / "strict.geojson",
                min_row_overlap_percent=30.0,
            )

            self.assertEqual(default_stats["row_count"], 1)
            self.assertEqual(strict_stats["row_count"], 2)

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
            row_source = root / "rows.geojson"
            _write(row_source, [
                _feature(box(500000, 5500000, 500002, 5500001), row_id="ROW-0001")
            ])
            associated = root / "associated.geojson"
            association = associate_anomalies(
                deduplicated,
                panel_source,
                associated,
                panel_output_path=panel_source,
                row_path=row_source,
                row_output_path=row_source,
            )
            self.assertEqual(association["assigned"], 1)
            self.assertEqual(association["unassigned"], 1)
            output = json.loads(associated.read_text(encoding="utf-8"))["features"]
            assigned = next(item for item in output if item["properties"]["panel_id"])
            self.assertEqual(assigned["properties"]["row_id"], "ROW-0001")
            self.assertTrue(assigned["properties"]["anomaly_id"].isdigit())
            updated_panel = json.loads(panel_source.read_text(encoding="utf-8"))["features"][0]
            self.assertEqual(updated_panel["properties"]["anomaly_count"], 1)
            self.assertEqual(
                updated_panel["properties"]["anomaly_ids"],
                [assigned["properties"]["anomaly_id"]],
            )
            updated_row = json.loads(row_source.read_text(encoding="utf-8"))["features"][0]
            self.assertEqual(updated_row["properties"]["anomaly_count"], 1)
            self.assertEqual(updated_row["properties"]["anomaly_ids"], [assigned["properties"]["anomaly_id"]])
            self.assertEqual(updated_row["properties"]["anomaly_panel_ids"], ["ROW-0001-PANEL-0001"])


if __name__ == "__main__":
    unittest.main()
