from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from xml.etree import ElementTree as ET

import cv2
import numpy as np
from PIL import ExifTags, Image


PROFILE_DIRECTORY = Path(__file__).with_name("camera_profiles")
DEFAULT_MINIMUM_DISPLACEMENT_PX = 2.0


class UndistortionError(ValueError):
    """Raised when strict lens correction cannot be performed safely."""


@dataclass(frozen=True)
class CameraIdentity:
    make: str
    model: str
    image_source: str
    width: int
    height: int
    serial_number: str = ""

    @property
    def label(self) -> str:
        name = " ".join(part for part in (self.make, self.model) if part).strip() or "Unknown camera"
        source = f" {self.image_source}" if self.image_source else ""
        return f"{name}{source} ({self.width}×{self.height})"


@dataclass(frozen=True)
class Calibration:
    camera_matrix: list[list[float]]
    distortion_coefficients: list[float]
    source: str
    profile_id: str


def _local_name(value: str) -> str:
    return str(value).rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _xmp_values(image: Image.Image) -> dict[str, str]:
    blob = image.info.get("XML:com.adobe.xmp") or image.info.get("xmp") or image.info.get("XMP")
    if not blob:
        return {}
    if isinstance(blob, bytes):
        text = blob.decode("utf-8", errors="ignore")
    else:
        text = str(blob)
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    values: dict[str, str] = {}
    for element in root.iter():
        name = _local_name(element.tag)
        content = (element.text or "").strip()
        if content:
            values[name] = content
        for key, value in element.attrib.items():
            values[_local_name(key)] = str(value).strip()
    return values


def inspect_camera(path: Path) -> tuple[CameraIdentity, dict[str, str]]:
    try:
        with Image.open(path) as image:
            width, height = image.size
            exif = image.getexif()
            named_exif = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
            xmp = _xmp_values(image)
    except Exception as exc:
        raise UndistortionError(f"Could not read camera metadata from {path.name}: {exc}") from exc
    make = str(xmp.get("Make") or named_exif.get("Make") or "").strip()
    model = str(xmp.get("Model") or xmp.get("DroneModel") or named_exif.get("Model") or "").strip()
    identity = CameraIdentity(
        make=make,
        model=model,
        image_source=str(xmp.get("ImageSource") or "").strip(),
        width=int(width),
        height=int(height),
        serial_number=str(xmp.get("CameraSerialNumber") or "").strip(),
    )
    return identity, xmp


def _parse_dewarp_data(value: str, identity: CameraIdentity) -> Calibration:
    # DJI stores an optional date before the semicolon, followed by
    # fx, fy, cx, cy, k1, k2, p1, p2, k3. cx/cy are offsets from image centre.
    payload = str(value).rsplit(";", 1)[-1]
    numbers = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", payload)]
    if len(numbers) != 9:
        raise UndistortionError(
            f"{identity.label} contains DewarpData, but it does not contain the expected nine calibration values."
        )
    fx, fy, cx_offset, cy_offset, k1, k2, p1, p2, k3 = numbers
    if fx <= 0 or fy <= 0:
        raise UndistortionError(f"{identity.label} contains invalid calibrated focal lengths.")
    cx = identity.width / 2.0 + cx_offset
    cy = identity.height / 2.0 + cy_offset
    if not (-identity.width <= cx <= identity.width * 2 and -identity.height <= cy <= identity.height * 2):
        raise UndistortionError(f"{identity.label} contains an invalid calibrated optical centre.")
    return Calibration(
        camera_matrix=[[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        distortion_coefficients=[k1, k2, p1, p2, k3],
        source="embedded DewarpData",
        profile_id=f"embedded:{identity.serial_number or identity.model or 'camera'}",
    )


def _profile_matches(profile: dict[str, Any], identity: CameraIdentity) -> bool:
    expected = {
        "make": identity.make,
        "model": identity.model,
        "image_source": identity.image_source,
        "serial_number": identity.serial_number,
    }
    for key, actual in expected.items():
        wanted = str(profile.get(key) or "").strip()
        if wanted and wanted.casefold() != str(actual).strip().casefold():
            return False
    for key, actual in (("width", identity.width), ("height", identity.height)):
        wanted = profile.get(key)
        if wanted is None or int(wanted) != actual:
            return False
    return bool(profile.get("model") or profile.get("serial_number"))


def _load_profiles(directory: Path = PROFILE_DIRECTORY) -> Iterable[dict[str, Any]]:
    if not directory.is_dir():
        return []
    profiles = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload["__path"] = str(path)
                profiles.append(payload)
        except (OSError, ValueError, TypeError):
            continue
    return profiles


def _calibration_from_profile(profile: dict[str, Any], identity: CameraIdentity) -> Calibration:
    matrix = profile.get("camera_matrix")
    coefficients = profile.get("distortion_coefficients")
    try:
        matrix_array = np.asarray(matrix, dtype=np.float64)
        coefficient_array = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    except Exception as exc:
        raise UndistortionError(f"Calibration profile for {identity.label} is malformed: {exc}") from exc
    if matrix_array.shape != (3, 3) or coefficient_array.size not in {4, 5, 8, 12, 14}:
        raise UndistortionError(f"Calibration profile for {identity.label} has invalid matrix dimensions.")
    if matrix_array[0, 0] <= 0 or matrix_array[1, 1] <= 0:
        raise UndistortionError(f"Calibration profile for {identity.label} has invalid focal lengths.")
    return Calibration(
        camera_matrix=matrix_array.tolist(),
        distortion_coefficients=coefficient_array.tolist(),
        source="saved camera profile",
        profile_id=str(profile.get("profile_id") or Path(str(profile.get("__path") or "profile")).stem),
    )


def resolve_calibration(
    identity: CameraIdentity,
    xmp: dict[str, str],
    profile_directory: Path = PROFILE_DIRECTORY,
) -> Calibration:
    dewarp_data = xmp.get("DewarpData")
    if dewarp_data:
        return _parse_dewarp_data(dewarp_data, identity)

    matches = [profile for profile in _load_profiles(profile_directory) if _profile_matches(profile, identity)]
    if matches:
        def specificity(profile: dict[str, Any]) -> int:
            return sum(bool(profile.get(key)) for key in ("serial_number", "image_source", "make", "model"))

        best_score = max(specificity(profile) for profile in matches)
        best = [profile for profile in matches if specificity(profile) == best_score]
        if len(best) > 1:
            names = ", ".join(str(profile.get("profile_id") or Path(profile["__path"]).stem) for profile in best)
            raise UndistortionError(f"Multiple equally specific calibration profiles match {identity.label}: {names}.")
        return _calibration_from_profile(best[0], identity)

    detected = identity.label
    serial = f" Serial: {identity.serial_number}." if identity.serial_number else ""
    raise UndistortionError(
        f"Cannot correct lens distortion for {detected}.{serial} The image has no usable DewarpData and no matching saved "
        "camera calibration profile. Disable ‘Correct lens distortion automatically’ and run again, "
        "or add a calibrated profile."
    )


def dewarp_state(xmp: dict[str, str]) -> Optional[bool]:
    """Return True when already corrected, False when explicitly raw, else None."""
    raw = str(xmp.get("DewarpFlag") or "").strip().casefold()
    try:
        numeric = float(raw)
        if numeric == 1.0:
            return True
        if numeric == 0.0:
            return False
    except ValueError:
        pass
    if raw in {"1", "true", "yes", "dewarped", "corrected", "applied"}:
        return True
    if raw in {"0", "false", "no", "not_dewarped", "uncorrected", "not_applied"}:
        return False
    return None


def maximum_displacement_px(identity: CameraIdentity, calibration: Calibration) -> float:
    """Measure the largest calibrated displacement along the image boundary."""
    width = float(identity.width - 1)
    height = float(identity.height - 1)
    samples = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    boundary = []
    for position in samples:
        boundary.extend((
            (position * width, 0.0),
            (position * width, height),
            (0.0, position * height),
            (width, position * height),
        ))
    points = np.asarray(boundary, dtype=np.float64).reshape(-1, 1, 2)
    matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    coefficients = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    corrected = cv2.undistortPoints(points, matrix, coefficients, P=matrix).reshape(-1, 2)
    original = points.reshape(-1, 2)
    displacement = np.linalg.norm(corrected - original, axis=1)
    maximum = float(np.max(displacement)) if displacement.size else 0.0
    if not np.isfinite(maximum):
        raise UndistortionError(f"Calibration for {identity.label} produced an invalid distortion displacement.")
    return maximum


def undistort_pil_image(
    image: Image.Image,
    metadata_path: Path,
    profile_directory: Path = PROFILE_DIRECTORY,
    minimum_displacement_px: float = DEFAULT_MINIMUM_DISPLACEMENT_PX,
) -> tuple[Image.Image, dict[str, Any]]:
    identity, xmp = inspect_camera(metadata_path)
    if image.size != (identity.width, identity.height):
        raise UndistortionError(
            f"Cannot correct lens distortion for {metadata_path.name}: decoded image size {image.width}×{image.height} does not "
            f"match its calibration source size {identity.width}×{identity.height}."
        )
    threshold = max(0.0, float(minimum_displacement_px))
    applied = dewarp_state(xmp)
    if applied is True:
        return image.copy(), {
            "camera": asdict(identity),
            "calibration": None,
            "input_size": [identity.width, identity.height],
            "output_size": [image.width, image.height],
            "minimum_displacement_px": threshold,
            "status": "skipped_already_corrected",
        }

    calibration = resolve_calibration(identity, xmp, profile_directory)
    maximum = maximum_displacement_px(identity, calibration)
    base_record = {
        "camera": asdict(identity),
        "calibration": asdict(calibration),
        "input_size": [identity.width, identity.height],
        "output_size": [image.width, image.height],
        "maximum_displacement_px": round(maximum, 6),
        "minimum_displacement_px": threshold,
        "dewarp_flag": applied,
    }
    if maximum < threshold:
        return image.copy(), {**base_record, "status": "skipped_below_threshold"}

    source_mode = image.mode
    array = np.asarray(image)
    matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
    coefficients = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    corrected = cv2.undistort(array, matrix, coefficients, None, matrix)
    output = Image.fromarray(corrected)
    if output.mode != source_mode:
        output = output.convert(source_mode)
    record = {**base_record, "output_size": [output.width, output.height], "status": "corrected"}
    return output, record
