"""Shared types and image remapping for runtime lens calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import ExifTags, Image


class LensCalibrationError(ValueError):
    """Raised when runtime calibration cannot be estimated or applied safely."""


@dataclass(frozen=True)
class CameraGroup:
    make: str
    model: str
    image_source: str
    width: int
    height: int
    serial_number: str = ""
    # Focal values constrain fitting but do not split an otherwise identical
    # camera group when a subset of files omits optional EXIF fields.
    focal_length_mm: float = field(default=0.0, compare=False)
    focal_length_35mm: float = field(default=0.0, compare=False)

    @property
    def label(self) -> str:
        camera = " ".join(value for value in (self.make, self.model) if value).strip() or "Unknown camera"
        source = f" {self.image_source}" if self.image_source else ""
        return f"{camera}{source} ({self.width}×{self.height})"


@dataclass(frozen=True)
class RuntimeCalibration:
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    source: str = "runtime multi-line calibration"
    profile_id: str = "runtime-plumb-lines"


def _local_name(value: str) -> str:
    return str(value).rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _xmp_values(image: Image.Image) -> dict[str, str]:
    blob = image.info.get("XML:com.adobe.xmp") or image.info.get("xmp") or image.info.get("XMP")
    if not blob:
        return {}
    text = blob.decode("utf-8", errors="ignore") if isinstance(blob, bytes) else str(blob)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    values: dict[str, str] = {}
    for element in root.iter():
        content = (element.text or "").strip()
        if content:
            values[_local_name(element.tag)] = content
        for key, value in element.attrib.items():
            values[_local_name(key)] = str(value).strip()
    return values


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0


def inspect_camera_group(path: Path) -> CameraGroup:
    """Read only identity fields used to prevent mixing different optical systems."""
    try:
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
            named = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
            try:
                exif_ifd = exif.get_ifd(34665)
                named.update({ExifTags.TAGS.get(key, str(key)): value for key, value in exif_ifd.items()})
            except Exception:
                pass
            xmp = _xmp_values(image)
    except Exception as exc:
        raise LensCalibrationError(f"Could not read {path.name}: {exc}") from exc
    return CameraGroup(
        make=str(xmp.get("Make") or named.get("Make") or "").strip(),
        model=str(xmp.get("Model") or xmp.get("DroneModel") or named.get("Model") or "").strip(),
        image_source=str(xmp.get("ImageSource") or "").strip(),
        width=int(width),
        height=int(height),
        serial_number=str(
            xmp.get("CameraSerialNumber")
            or xmp.get("SerialNumber")
            or named.get("CameraSerialNumber")
            or named.get("BodySerialNumber")
            or ""
        ).strip(),
        focal_length_mm=_number(named.get("FocalLength")),
        focal_length_35mm=_number(named.get("FocalLengthIn35mmFilm")),
    )


def focal_pixels(group: CameraGroup) -> float:
    """Conservative focal initialization; line fitting estimates distortion, not focal length."""
    diagonal = float(np.hypot(group.width, group.height))
    if group.focal_length_35mm > 0:
        # 43.266 mm is the full-frame diagonal used by 35 mm equivalence.
        estimate = group.focal_length_35mm * diagonal / 43.266
        if np.isfinite(estimate) and diagonal * 0.35 <= estimate <= diagonal * 4.0:
            return float(estimate)
    return float(max(group.width, group.height))


def maximum_displacement_px(group: CameraGroup, calibration: RuntimeCalibration) -> float:
    width = float(group.width - 1)
    height = float(group.height - 1)
    positions = np.linspace(0.0, 1.0, 41, dtype=np.float64)
    boundary: list[tuple[float, float]] = []
    for position in positions:
        boundary.extend(
            (
                (position * width, 0.0),
                (position * width, height),
                (0.0, position * height),
                (width, position * height),
            )
        )
    source = np.asarray(boundary, dtype=np.float64).reshape(-1, 1, 2)
    matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    coefficients = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    corrected = cv2.undistortPoints(source, matrix, coefficients, P=matrix).reshape(-1, 2)
    displacement = np.linalg.norm(corrected - source.reshape(-1, 2), axis=1)
    maximum = float(np.max(displacement)) if displacement.size else 0.0
    if not np.isfinite(maximum):
        raise LensCalibrationError(f"Runtime calibration for {group.label} produced invalid coordinates.")
    return maximum


def validate_mapping(group: CameraGroup, calibration: RuntimeCalibration) -> float:
    """Reject folding, extreme, or numerically invalid mappings before touching images."""
    matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    coefficients = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    map_x, map_y = cv2.initUndistortRectifyMap(
        matrix,
        coefficients,
        None,
        matrix,
        (group.width, group.height),
        cv2.CV_32FC1,
    )
    if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
        raise LensCalibrationError(f"Runtime calibration for {group.label} generated invalid map values.")
    # A valid radial remap must remain locally monotonic; negative Jacobians fold the image.
    dx_x = np.gradient(map_x, axis=1)
    dx_y = np.gradient(map_x, axis=0)
    dy_x = np.gradient(map_y, axis=1)
    dy_y = np.gradient(map_y, axis=0)
    determinant = dx_x * dy_y - dx_y * dy_x
    if float(np.nanpercentile(determinant, 1.0)) <= 0.05:
        raise LensCalibrationError(f"Runtime calibration for {group.label} would fold part of the image.")
    maximum = maximum_displacement_px(group, calibration)
    if maximum > float(np.hypot(group.width, group.height)) * 0.18:
        raise LensCalibrationError(
            f"Runtime calibration for {group.label} is too strong ({maximum:.1f}px maximum displacement)."
        )
    return maximum


def correct_pil_image(
    image: Image.Image,
    group: CameraGroup,
    calibration: RuntimeCalibration,
) -> tuple[Image.Image, dict[str, Any]]:
    if image.size != (group.width, group.height):
        raise LensCalibrationError(
            f"Cannot apply {group.label} calibration to {image.width}×{image.height} pixels."
        )
    maximum = validate_mapping(group, calibration)
    source_mode = image.mode
    source = np.asarray(image)
    matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    coefficients = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    corrected = cv2.undistort(source, matrix, coefficients, None, matrix)
    output = Image.fromarray(corrected)
    if output.mode != source_mode:
        output = output.convert(source_mode)
    return output, {
        "camera_group": asdict(group),
        "calibration": asdict(calibration),
        "input_size": [group.width, group.height],
        "output_size": [output.width, output.height],
        "maximum_displacement_px": round(maximum, 6),
        "status": "corrected",
    }
