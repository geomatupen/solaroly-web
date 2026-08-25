from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from pvrt.dataops.image_metadata import save_with_metadata


class ImageMetadataTests(unittest.TestCase):
    def test_jpeg_export_preserves_exif_xmp_dimensions_and_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.jpeg"
            output = root / "output.jpeg"
            exif = Image.Exif()
            exif[0x010F] = "DJI"
            exif[0x0110] = "M3TD"
            exif[0x9003] = "2026:08:25 12:34:56"
            xmp = b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><test>gps-xmp</test></x:xmpmeta>'
            Image.new("RGB", (80, 64), (20, 30, 40)).save(
                source, format="JPEG", quality=95, exif=exif, xmp=xmp
            )
            timestamp_ns = 1_700_000_000_123_456_789
            os.utime(source, ns=(timestamp_ns, timestamp_ns))

            save_with_metadata(Image.new("RGB", (80, 64), (50, 60, 70)), source, output)

            with Image.open(source) as original, Image.open(output) as exported:
                self.assertEqual(exported.size, original.size)
                self.assertEqual(exported.getexif().get(0x010F), "DJI")
                self.assertEqual(exported.getexif().get(0x0110), "M3TD")
                self.assertEqual(exported.getexif().get(0x9003), "2026:08:25 12:34:56")
                self.assertEqual(exported.info.get("xmp"), original.info.get("xmp"))
            self.assertEqual(output.stat().st_mtime_ns, source.stat().st_mtime_ns)


if __name__ == "__main__":
    unittest.main()
