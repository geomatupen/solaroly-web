import json

from PIL import Image

from backend.pvrt.web.app import _canonical_image_candidate_names, _preds_to_geojson


def test_canonical_image_candidates_prefer_original_manifest_name():
    candidates = _canonical_image_candidate_names(
        {"DJI_0001_T.png", "DJI_0002_T.png"},
        {"DJI_0001_T.jpeg"},
        {"DJI_0001_T.jpeg", "DJI_0002_T.jpeg"},
        {"DJI_0001_T.jpeg", "DJI_0002_T.jpeg"},
    )

    assert candidates == {"DJI_0001_T.jpeg", "DJI_0002_T.jpeg"}


def test_canonical_image_candidates_keep_generated_name_without_source_record():
    candidates = _canonical_image_candidate_names(
        {"prepared_only.png"},
        set(),
        set(),
        set(),
    )

    assert candidates == {"prepared_only.png"}


def test_prediction_geojson_uses_row_aligned_result_metadata(tmp_path):
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    images_dir = result_dir / "rotated_images"
    images_dir.mkdir()
    Image.new("RGB", (100, 100), (20, 30, 40)).save(images_dir / "frame_001.png")
    predictions_dir = tmp_path / "prediction_data"
    (predictions_dir / "preds").mkdir(parents=True)
    (predictions_dir / "preds" / "frame_001.json").write_text(json.dumps({
        "file": "frame_001.png",
        "boxes": [[40, 40, 60, 60]],
        "scores": [0.95],
        "classes": [0],
    }), encoding="utf-8")
    camera_meta = {
        "frame_001.jpeg": {
            "lat": 47.1234,
            "lon": 16.9876,
            "w_px": 100,
            "h_px": 100,
            "meters_per_pixel": 0.1,
            "row_alignment_rotation_deg": 4.0,
            "row_alignment": {"status": "aligned"},
        }
    }

    predictions_path, images_path = _preds_to_geojson(
        images_dir=images_dir,
        preds_dir=predictions_dir,
        out_session=result_dir,
        class_names=["anomaly"],
        exif_index={"frame_001.png": (46.0, 15.0)},
        camera_meta=camera_meta,
    )

    images = json.loads(images_path.read_text(encoding="utf-8"))
    assert images["features"][0]["geometry"]["coordinates"] == [16.9876, 47.1234]
    assert images["features"][0]["properties"]["rotation"] == 4.0
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    polygon = predictions["features"][0]["geometry"]["coordinates"][0]
    mean_lon = sum(point[0] for point in polygon[:-1]) / (len(polygon) - 1)
    mean_lat = sum(point[1] for point in polygon[:-1]) / (len(polygon) - 1)
    assert abs(mean_lon - 16.9876) < 1e-8
    assert abs(mean_lat - 47.1234) < 1e-8
