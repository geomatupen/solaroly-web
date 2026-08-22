"""Minimal FLIR FFF extractor for radiometric JPEG grayscale conversion.

This reads the FLIR APP1 chunk layout used by cameras such as DJI's Zenmuse
XT2. It extracts raw sensor values only; temperature calibration is not needed
for the normalized grayscale output used by this application.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


FLIR_MAGIC = b"FLIR\x00"


def has_flir_fff(path: Path) -> bool:
    """Return whether a JPEG contains FLIR APP1/FFF metadata."""
    try:
        return FLIR_MAGIC in Path(path).read_bytes()
    except OSError:
        return False


def _collect_flir_chunks(jpeg: bytes) -> bytes:
    chunks: dict[int, bytes] = {}
    expected_last: int | None = None
    cursor = 2
    while cursor + 4 <= len(jpeg):
        marker_pos = jpeg.find(b"\xff\xe1", cursor)
        if marker_pos < 0 or marker_pos + 12 > len(jpeg):
            break
        segment_length = int.from_bytes(jpeg[marker_pos + 2:marker_pos + 4], "big")
        segment_end = marker_pos + 2 + segment_length
        if segment_length < 10 or segment_end > len(jpeg):
            cursor = marker_pos + 2
            continue
        payload = jpeg[marker_pos + 4:segment_end]
        if payload.startswith(FLIR_MAGIC) and len(payload) >= 8:
            chunk_number = payload[6]
            chunk_last = payload[7]
            if expected_last is None:
                expected_last = chunk_last
            elif expected_last != chunk_last:
                raise ValueError("Inconsistent FLIR FFF chunk count.")
            if chunk_number in chunks:
                raise ValueError("Duplicate FLIR FFF chunk.")
            chunks[chunk_number] = payload[8:]
        cursor = segment_end

    if expected_last is None:
        raise ValueError("No FLIR FFF payload found.")
    missing = [number for number in range(expected_last + 1) if number not in chunks]
    if missing:
        raise ValueError(f"Incomplete FLIR FFF payload; missing chunks: {missing}")
    return b"".join(chunks[number] for number in range(expected_last + 1))


def _raw_record(fff: bytes) -> tuple[int, bytes]:
    if fff[:4] not in {b"FFF\x00", b"AFF\x00"} or len(fff) < 64:
        raise ValueError("Invalid FLIR FFF header.")
    directory_offset = int.from_bytes(fff[24:28], "big")
    entry_count = int.from_bytes(fff[28:32], "big")
    if entry_count <= 0 or entry_count > 1024:
        raise ValueError("Invalid FLIR FFF record directory.")
    for index in range(entry_count):
        start = directory_offset + index * 32
        entry = fff[start:start + 32]
        if len(entry) != 32:
            break
        record_type = int.from_bytes(entry[0:2], "big")
        subtype = int.from_bytes(entry[2:4], "big")
        offset = int.from_bytes(entry[12:16], "big")
        length = int.from_bytes(entry[16:20], "big")
        if record_type == 1 and offset >= 0 and length > 32 and offset + length <= len(fff):
            return subtype, fff[offset:offset + length]
    raise ValueError("FLIR FFF raw thermal record not found.")


def extract_flir_raw(path: Path) -> np.ndarray:
    """Extract a 2-D numeric raw thermal plane from a FLIR radiometric JPEG."""
    fff = _collect_flir_chunks(Path(path).read_bytes())
    subtype, record = _raw_record(fff)
    width = int.from_bytes(record[2:4], "little")
    height = int.from_bytes(record[4:6], "little")
    if width <= 0 or height <= 0:
        raise ValueError("Invalid FLIR thermal dimensions.")
    payload = record[32:]
    expected_bytes = width * height * 2

    if len(payload) >= expected_bytes and subtype in {1, 2}:
        dtype = ">u2" if subtype == 1 else "<u2"
        return np.frombuffer(payload[:expected_bytes], dtype=dtype).reshape(height, width).copy()

    try:
        with Image.open(BytesIO(payload)) as embedded:
            raw = np.asarray(embedded)
    except Exception as exc:
        raise ValueError("Unsupported FLIR raw thermal record encoding.") from exc
    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.shape != (height, width):
        raise ValueError(
            f"FLIR raw dimensions differ from metadata: raw={raw.shape}, expected={(height, width)}"
        )
    if subtype == 3 and raw.dtype.itemsize == 2:
        raw = raw.byteswap()
    return raw.copy()
