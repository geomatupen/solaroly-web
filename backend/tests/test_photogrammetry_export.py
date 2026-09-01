import json
import tempfile
import unittest
from pathlib import Path

import piexif
from PIL import Image

from pvrt.dataops.photogrammetry_export import finalize_webodm_export


def _gps_decimal(gps, coordinate_key, ref_key):
    values = gps[coordinate_key]
    degrees = values[0][0] / values[0][1]
    minutes = values[1][0] / values[1][1]
    seconds = values[2][0] / values[2][1]
    value = degrees + minutes / 60.0 + seconds / 3600.0
    if gps[ref_key] in {b"S", b"W"}:
        value *= -1
    return value


class PhotogrammetryExportTests(unittest.TestCase):
    def test_embeds_aligned_horizontal_gps_and_preserves_altitude(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            export_dir = root / "undistorted_images"
            export_dir.mkdir()
            image_path = export_dir / "DJI_0001.jpg"
            gps = {
                piexif.GPSIFD.GPSLatitudeRef: b"N",
                piexif.GPSIFD.GPSLatitude: ((50, 1), (0, 1), (0, 1)),
                piexif.GPSIFD.GPSLongitudeRef: b"E",
                piexif.GPSIFD.GPSLongitude: ((8, 1), (0, 1), (0, 1)),
                piexif.GPSIFD.GPSAltitudeRef: 0,
                piexif.GPSIFD.GPSAltitude: (12345, 100),
            }
            Image.new("RGB", (12, 8), "white").save(
                image_path,
                format="JPEG",
                exif=piexif.dump({"0th": {}, "Exif": {}, "GPS": gps, "Interop": {}, "1st": {}, "thumbnail": None}),
            )
            camera_meta_path = root / "camera_meta.json"
            alignment_path = root / "image_alignment.json"
            camera_meta = {
                "DJI_0001.jpg": {
                    "lat": 50.123456,
                    "lon": 8.654321,
                    "absolute_altitude": 123.45,
                }
            }
            report = {"images": {"DJI_0001.jpg": {"status": "aligned"}}}
            camera_meta_path.write_text(json.dumps(camera_meta), encoding="utf-8")
            alignment_path.write_text(json.dumps(report), encoding="utf-8")

            summary = finalize_webodm_export(
                export_dir=export_dir,
                camera_meta=camera_meta,
                alignment_report=report,
                camera_meta_path=camera_meta_path,
                alignment_report_path=alignment_path,
            )

            updated_gps = piexif.load(str(image_path))["GPS"]
            self.assertAlmostEqual(
                _gps_decimal(updated_gps, piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLatitudeRef),
                50.123456,
                places=6,
            )
            self.assertAlmostEqual(
                _gps_decimal(updated_gps, piexif.GPSIFD.GPSLongitude, piexif.GPSIFD.GPSLongitudeRef),
                8.654321,
                places=6,
            )
            self.assertEqual(updated_gps[piexif.GPSIFD.GPSAltitude], (12345, 100))
            self.assertEqual(summary["corrected_gps_embedded"], 1)
            self.assertEqual(
                (export_dir / "geo.txt").read_text(encoding="utf-8").splitlines(),
                ["EPSG:4326", "DJI_0001.jpg 8.6543210000 50.1234560000 123.450"],
            )
            self.assertTrue((export_dir / "camera_meta.json").is_file())
            self.assertTrue((export_dir / "image_alignment.json").is_file())

    def test_retained_image_is_listed_without_rewriting_gps(self):
        with tempfile.TemporaryDirectory() as temporary:
            export_dir = Path(temporary)
            image_path = export_dir / "DJI_0002.jpg"
            Image.new("RGB", (8, 8), "black").save(image_path, format="JPEG")
            original = image_path.read_bytes()
            camera_meta = {"DJI_0002.jpg": {"lat": 51.0, "lon": 9.0, "absolute_altitude": 100.0}}

            summary = finalize_webodm_export(
                export_dir=export_dir,
                camera_meta=camera_meta,
                alignment_report={"images": {"DJI_0002.jpg": {"status": "retained_original"}}},
            )

            self.assertEqual(image_path.read_bytes(), original)
            self.assertEqual(summary["corrected_gps_embedded"], 0)
            self.assertEqual(summary["original_gps_retained"], 1)


if __name__ == "__main__":
    unittest.main()
