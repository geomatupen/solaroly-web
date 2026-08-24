import json
import tempfile
import unittest
from pathlib import Path

import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import transform as transform_geometry

from pvrt.postprocess import analyze_geojson, combine_tile_fragments, regularize_polygons


def _write_tile(path: Path, x_origin: float, y_origin: float, pixel_size: float) -> None:
    import numpy as np

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=100,
        height=100,
        count=1,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(x_origin, y_origin, pixel_size, pixel_size),
    ) as dataset:
        dataset.write(np.zeros((1, 100, 100), dtype="uint8"))


def _feature(polygon: Polygon, tile: str) -> dict:
    to_wgs84 = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform
    return {
        "type": "Feature",
        "geometry": mapping(transform_geometry(to_wgs84, polygon)),
        "properties": {"class_id": 0, "class_name": "panel", "score": 0.95, "tile": tile},
    }


class GeojsonPostprocessTests(unittest.TestCase):
    def test_combines_fragments_using_each_results_actual_resolution(self):
        for pixel_size in (0.02, 0.05):
            with self.subTest(pixel_size=pixel_size), tempfile.TemporaryDirectory() as temporary:
                self._run_resolution_case(Path(temporary), pixel_size)

    def _run_resolution_case(self, tmp_path: Path, pixel_size: float) -> None:
        result_dir = tmp_path / "result"
        tiles_dir = result_dir / "tiles"
        tiles_dir.mkdir(parents=True)
        origin_x, origin_y = 500000.0, 5500000.0
        boundary_x = origin_x + 100 * pixel_size
        _write_tile(tiles_dir / "scene_0_0.tif", origin_x, origin_y, pixel_size)
        _write_tile(tiles_dir / "scene_0_100.tif", boundary_x, origin_y, pixel_size)

        # One intact panel provides the expected-size template. Two other features
        # are halves of one panel on opposite sides of the adjoining tile edge.
        y_top = origin_y - 10 * pixel_size
        intact = Polygon([
            (origin_x + 10 * pixel_size, y_top),
            (origin_x + 10 * pixel_size + 1.2, y_top),
            (origin_x + 10 * pixel_size + 1.2, y_top - 1.0),
            (origin_x + 10 * pixel_size, y_top - 1.0),
        ])
        inner = Polygon([
            (origin_x + 10 * pixel_size + 0.2, y_top - 0.2),
            (origin_x + 10 * pixel_size + 0.6, y_top - 0.2),
            (origin_x + 10 * pixel_size + 0.6, y_top - 0.6),
            (origin_x + 10 * pixel_size + 0.2, y_top - 0.6),
        ])
        intact_right = Polygon([
            (boundary_x + 10 * pixel_size, y_top),
            (boundary_x + 10 * pixel_size + 1.2, y_top),
            (boundary_x + 10 * pixel_size + 1.2, y_top - 1.0),
            (boundary_x + 10 * pixel_size, y_top - 1.0),
        ])
        split_top = origin_y - 65 * pixel_size
        left = Polygon([
            (boundary_x - 0.4, split_top), (boundary_x, split_top),
            (boundary_x, split_top - 1.0), (boundary_x - 0.4, split_top - 1.0),
        ])
        right = Polygon([
            (boundary_x, split_top), (boundary_x + 0.8, split_top),
            (boundary_x + 0.8, split_top - 1.0), (boundary_x, split_top - 1.0),
        ])
        source = result_dir / "predictions.geojson"
        source.write_text(json.dumps({
            "type": "FeatureCollection",
            "features": [
                _feature(intact, "scene_0_0"),
                _feature(inner, "scene_0_0"),
                _feature(intact_right, "scene_0_100"),
                _feature(left, "scene_0_0"),
                _feature(right, "scene_0_100"),
            ],
        }), encoding="utf-8")

        analysis = analyze_geojson(source, result_dir)
        self.assertTrue(analysis["tile_metadata_available"])
        self.assertEqual(analysis["features_on_tile_edges"], 2)
        self.assertAlmostEqual(analysis["median_pixel_size_m"], pixel_size, delta=pixel_size * 0.02)

        combined = result_dir / "postprocess" / "job" / "combined.geojson"
        stats = combine_tile_fragments(source, combined, result_dir)
        self.assertEqual(stats["accepted_links"], 1)
        self.assertEqual(stats["contained_polygons_removed"], 1)
        self.assertEqual(stats["output_features"], 3)

        payload = json.loads(combined.read_text(encoding="utf-8"))
        merged = [item for item in payload["features"] if item["properties"]["source_feature_count"] == 2]
        self.assertEqual(len(merged), 1)
        self.assertTrue(shape(merged[0]["geometry"]).is_valid)

        regularized = combined.with_name("regularized.geojson")
        regularize_stats = regularize_polygons(combined, regularized)
        self.assertEqual(regularize_stats["output_features"], 3)
        self.assertTrue(regularized.is_file())


if __name__ == "__main__":
    unittest.main()
