"""Fail-closed runtime radial calibration from repeated straight structures.

The detector traces families of locally straight edge segments into long curves.
It never needs to name the object: unrelated natural curves are outliers unless
one shared radial model straightens several independent traces and also wins on
traces withheld from fitting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import cv2
import numpy as np
from PIL import Image

try:
    from .lens_distortion import (
        CameraGroup,
        LensCalibrationError,
        RuntimeCalibration,
        focal_pixels,
        validate_mapping,
    )
except ImportError:  # Direct execution by the existing preparation subprocess.
    from lens_distortion import CameraGroup, LensCalibrationError, RuntimeCalibration, focal_pixels, validate_mapping


MAX_CANDIDATE_IMAGES = 8
MAX_FIT_IMAGES = 3
ANALYSIS_MAX_DIMENSION = 900
MIN_TRACES = 4
MIN_TRACES_PER_IMAGE = 4
MIN_VALIDATION_IMPROVEMENT = 0.15


@dataclass(frozen=True)
class CurveTrace:
    points: np.ndarray
    span: float
    support: float
    family: int
    image_index: int

    @property
    def weight(self) -> float:
        return float(max(self.span, 1.0) * max(self.support, 1.0))


def _even_sample(paths: list[Path], limit: int) -> list[Path]:
    if len(paths) <= limit:
        return paths
    indexes = np.linspace(0, len(paths) - 1, limit, dtype=int)
    return [paths[int(index)] for index in indexes]


def _read_gray(path: Path, size: tuple[int, int]) -> Optional[tuple[np.ndarray, float]]:
    try:
        with Image.open(path) as image:
            if image.size != size:
                return None
            gray = np.asarray(image.convert("L"), dtype=np.uint8)
            scale = min(1.0, ANALYSIS_MAX_DIMENSION / max(gray.shape))
            if scale < 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            return gray, float(scale)
    except Exception:
        return None


def _angular_distance(first: np.ndarray | float, second: float) -> np.ndarray:
    return np.abs((np.asarray(first) - second + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _dominant_families(angles: np.ndarray, lengths: np.ndarray) -> list[float]:
    bins = 36
    histogram = np.zeros(bins, dtype=np.float64)
    indexes = np.floor((angles % np.pi) / np.pi * bins).astype(int) % bins
    for index, length in zip(indexes, lengths):
        histogram[index] += float(length)
    if not np.any(histogram):
        return []
    first = int(np.argmax(histogram))
    family_angles = [(first + 0.5) * np.pi / bins]
    candidates = np.arange(bins)
    candidate_angles = (candidates + 0.5) * np.pi / bins
    separation = _angular_distance(candidate_angles, family_angles[0])
    eligible = (separation >= np.deg2rad(55.0)) & (separation <= np.deg2rad(125.0))
    if np.any(eligible):
        scores = np.where(eligible, histogram, -1.0)
        second = int(np.argmax(scores))
        if histogram[second] >= histogram[first] * 0.12:
            family_angles.append((second + 0.5) * np.pi / bins)
    return family_angles


def _fit_curve_family(
    segments: np.ndarray,
    lengths: np.ndarray,
    angles: np.ndarray,
    family_angle: float,
    family_index: int,
    image_index: int,
    diagonal: float,
) -> list[CurveTrace]:
    selected = np.flatnonzero(_angular_distance(angles, family_angle) <= np.deg2rad(24.0))
    if selected.size < 3:
        return []
    direction = np.asarray([np.cos(family_angle), np.sin(family_angle)], dtype=np.float64)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    midpoints = (segments[:, :2] + segments[:, 2:]) / 2.0
    u = midpoints @ direction
    v = midpoints @ normal
    local = list(map(int, selected))
    traces: list[CurveTrace] = []
    rng = np.random.default_rng(7103 + image_index * 31 + family_index)
    distance_limit = max(2.0, diagonal * 0.0045)
    angle_limit = np.deg2rad(13.0)

    for _curve_index in range(16):
        if len(local) < 3:
            break
        pool = np.asarray(local, dtype=int)
        best: np.ndarray | None = None
        best_score = 0.0
        trials = min(450, max(100, len(local) * 8))
        for _ in range(trials):
            sample = rng.choice(pool, size=3, replace=False)
            if np.ptp(u[sample]) < diagonal * 0.10:
                continue
            try:
                coefficients = np.polyfit(u[sample], v[sample], 2)
            except np.linalg.LinAlgError:
                continue
            predicted = np.polyval(coefficients, u[pool])
            slope = 2.0 * coefficients[0] * u[pool] + coefficients[1]
            predicted_angle = family_angle + np.arctan(slope)
            distance_ok = np.abs(v[pool] - predicted) <= distance_limit
            angle_ok = _angular_distance(angles[pool], predicted_angle) <= angle_limit
            inliers = pool[distance_ok & angle_ok]
            if inliers.size < 3:
                continue
            span = float(np.ptp(u[inliers]))
            support = float(np.sum(lengths[inliers]))
            score = support + span * 0.5
            if span >= diagonal * 0.18 and support >= diagonal * 0.14 and score > best_score:
                best = inliers
                best_score = score
        if best is None:
            break
        curve_points = np.concatenate(
            (segments[best, :2], segments[best, 2:], midpoints[best]), axis=0
        ).astype(np.float64)
        projected = curve_points @ direction
        order = np.argsort(projected)
        curve_points = curve_points[order]
        traces.append(
            CurveTrace(
                points=curve_points,
                span=float(np.ptp(projected)),
                support=float(np.sum(lengths[best])),
                family=family_index,
                image_index=image_index,
            )
        )
        remove = set(map(int, best))
        local = [index for index in local if index not in remove]
    return traces


def trace_straight_structures(gray: np.ndarray, image_index: int = 0) -> list[CurveTrace]:
    """Link local edge segments into long smooth curves from up to two directions."""
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(enhanced)[0]
    if detected is None:
        return []
    segments = detected[:, 0, :].astype(np.float64)
    delta = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(delta, axis=1)
    diagonal = float(np.hypot(gray.shape[1], gray.shape[0]))
    keep = lengths >= max(10.0, diagonal * 0.022)
    segments = segments[keep]
    lengths = lengths[keep]
    if segments.shape[0] < 6:
        return []
    delta = segments[:, 2:] - segments[:, :2]
    angles = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), np.pi)
    families = _dominant_families(angles, lengths)
    traces: list[CurveTrace] = []
    for family_index, family_angle in enumerate(families):
        traces.extend(
            _fit_curve_family(
                segments,
                lengths,
                angles,
                family_angle,
                family_index,
                image_index,
                diagonal,
            )
        )
    return traces


def _restore_trace_scale(traces: list[CurveTrace], scale: float) -> list[CurveTrace]:
    if scale >= 1.0:
        return traces
    return [
        CurveTrace(
            points=trace.points / scale,
            span=trace.span / scale,
            support=trace.support / scale,
            family=trace.family,
            image_index=trace.image_index,
        )
        for trace in traces
    ]


def _calibration(group: CameraGroup, focal: float, parameters: np.ndarray) -> RuntimeCalibration:
    k1, k2, offset_x, offset_y = map(float, parameters)
    cx = group.width * (0.5 + offset_x)
    cy = group.height * (0.5 + offset_y)
    return RuntimeCalibration(
        camera_matrix=[[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        distortion_coefficients=[k1, k2, 0.0, 0.0, 0.0],
    )


def _line_rms(points: np.ndarray) -> float:
    centred = points - np.mean(points, axis=0, keepdims=True)
    if centred.shape[0] < 3:
        return float("inf")
    covariance = centred.T @ centred / max(1, centred.shape[0] - 1)
    values = np.linalg.eigvalsh(covariance)
    return float(np.sqrt(max(0.0, values[0])))


def _trace_errors(
    traces: list[CurveTrace],
    calibration: Optional[RuntimeCalibration],
) -> np.ndarray:
    errors = []
    if calibration is not None:
        matrix = np.asarray(calibration.camera_matrix, dtype=np.float64)
        coefficients = np.asarray(calibration.distortion_coefficients, dtype=np.float64)
    for trace in traces:
        points = trace.points
        if calibration is not None:
            points = cv2.undistortPoints(
                points.reshape(-1, 1, 2), matrix, coefficients, P=matrix
            ).reshape(-1, 2)
        errors.append(_line_rms(points))
    return np.asarray(errors, dtype=np.float64)


def _weighted_error(traces: list[CurveTrace], calibration: Optional[RuntimeCalibration]) -> float:
    errors = _trace_errors(traces, calibration)
    if not errors.size or not np.isfinite(errors).all():
        return 1e9
    weights = np.asarray([trace.weight for trace in traces], dtype=np.float64)
    weights /= max(float(np.sum(weights)), 1.0)
    return float(np.sum(np.minimum(errors, 20.0) * weights))


def _coverage(traces: list[CurveTrace], group: CameraGroup) -> dict[str, Any]:
    points = np.concatenate([trace.points for trace in traces], axis=0)
    width_coverage = float(np.ptp(points[:, 0]) / max(1.0, group.width - 1))
    height_coverage = float(np.ptp(points[:, 1]) / max(1.0, group.height - 1))
    quadrants = set()
    for x, y in points:
        quadrants.add((int(x >= group.width / 2), int(y >= group.height / 2)))
    return {
        "width": round(width_coverage, 6),
        "height": round(height_coverage, 6),
        "quadrants": len(quadrants),
        "families": len({trace.family for trace in traces}),
    }


def _fit_parameters(
    traces: list[CurveTrace],
    group: CameraGroup,
    focal: float,
    *,
    conservative: bool,
) -> np.ndarray:
    if conservative:
        # A single view cannot reliably separate principal-point movement, k1,
        # and k2. Fit only k1 and keep the optical centre at the image centre.
        candidates = np.linspace(-1.0, 1.0, 161, dtype=np.float64)
        scores = [
            _weighted_error(
                traces,
                _calibration(group, focal, np.asarray([k1, 0.0, 0.0, 0.0])),
            )
            for k1 in candidates
        ]
        best = float(candidates[int(np.argmin(scores))])
        step = 0.0125
        for _ in range(16):
            options = np.clip(np.asarray([best - step, best, best + step]), -1.0, 1.0)
            values = [
                _weighted_error(
                    traces,
                    _calibration(group, focal, np.asarray([k1, 0.0, 0.0, 0.0])),
                )
                for k1 in options
            ]
            best = float(options[int(np.argmin(values))])
            step *= 0.5
        return np.asarray([best, 0.0, 0.0, 0.0], dtype=np.float64)

    def objective(parameters: np.ndarray) -> float:
        calibration = _calibration(group, focal, parameters)
        error = _weighted_error(traces, calibration)
        # Keep the principal point near the sensor centre unless the evidence is strong.
        return error + 0.35 * float(parameters[2] ** 2 + parameters[3] ** 2)

    lower = np.asarray([-1.0, -0.8, -0.08, -0.08], dtype=np.float64)
    upper = np.asarray([1.0, 0.8, 0.08, 0.08], dtype=np.float64)
    rng = np.random.default_rng(8431)
    seeds = [np.zeros(4, dtype=np.float64)]
    seeds.extend(rng.uniform(lower, upper, size=(320, 4)))
    best = min(seeds, key=objective).copy()
    best_score = objective(best)
    step = (upper - lower) / 5.0
    # Deterministic bounded coordinate refinement avoids introducing a new
    # optimization dependency into the image-preparation runtime.
    for _ in range(18):
        improved = False
        for dimension in range(4):
            for direction in (-1.0, 1.0):
                candidate = best.copy()
                candidate[dimension] = np.clip(
                    candidate[dimension] + direction * step[dimension],
                    lower[dimension],
                    upper[dimension],
                )
                score = objective(candidate)
                if score < best_score:
                    best = candidate
                    best_score = score
                    improved = True
        step *= 0.62 if improved else 0.45
    return best


def estimate_runtime_calibration(
    image_paths: Iterable[Path],
    group: CameraGroup,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[Optional[RuntimeCalibration], dict[str, Any]]:
    """Inspect eight evenly spaced candidates and fit one shared model from the best three."""
    emit = log or (lambda _message: None)
    unique_paths = sorted({Path(path) for path in image_paths})
    candidates = _even_sample(unique_paths, MAX_CANDIDATE_IMAGES)
    analysed: list[tuple[Path, list[CurveTrace]]] = []
    for index, path in enumerate(candidates):
        loaded = _read_gray(path, (group.width, group.height))
        if loaded is None:
            continue
        gray, scale = loaded
        traces = _restore_trace_scale(trace_straight_structures(gray, image_index=index), scale)
        if len(traces) >= MIN_TRACES_PER_IMAGE:
            analysed.append((path, traces))
            emit(f"Analysed {path.name}: {len(traces)} long curve trace(s)")
        elif traces:
            emit(
                f"Excluded {path.name}: only {len(traces)} reliable long trace(s); "
                f"at least {MIN_TRACES_PER_IMAGE} are required per sample"
            )
    analysed.sort(key=lambda item: sum(trace.weight for trace in item[1]), reverse=True)
    selected = analysed[:MAX_FIT_IMAGES]
    traces = [trace for _path, image_traces in selected for trace in image_traces]
    record: dict[str, Any] = {
        "method": "runtime multi-image plumb-line calibration",
        "camera_group": group.label,
        "candidate_image_count": len(candidates),
        "selected_images": [path.name for path, _traces in selected],
        "trace_count": len(traces),
        "maximum_fit_images": MAX_FIT_IMAGES,
        "minimum_validation_improvement": MIN_VALIDATION_IMPROVEMENT,
    }
    if len(traces) < MIN_TRACES:
        record.update(
            status="rejected_insufficient_structure",
            reason=f"Only {len(traces)} reliable long traces were found; at least {MIN_TRACES} are required.",
        )
        return None, record
    coverage = _coverage(traces, group)
    record["coverage"] = coverage
    if coverage["quadrants"] < 3 or max(coverage["width"], coverage["height"]) < 0.65:
        record.update(
            status="rejected_insufficient_coverage",
            reason="Straight structures do not cover enough of the frame to distinguish radial distortion safely.",
        )
        return None, record

    ordered = sorted(traces, key=lambda trace: (trace.image_index, trace.family, -trace.weight))
    fit_traces = ordered[::2]
    validation_traces = ordered[1::2]
    if len(validation_traces) < 2:
        validation_traces = fit_traces
    focal = focal_pixels(group)
    baseline_fit = _weighted_error(fit_traces, None)
    baseline_validation = _weighted_error(validation_traces, None)
    if baseline_validation < 0.35:
        record.update(
            status="rejected_no_measurable_distortion",
            reason="Detected structures are already too straight to justify remapping.",
            baseline_validation_rms=round(baseline_validation, 6),
        )
        return None, record
    conservative = len(selected) == 1
    parameters = _fit_parameters(fit_traces, group, focal, conservative=conservative)
    calibration = _calibration(group, focal, parameters)
    corrected_fit = _weighted_error(fit_traces, calibration)
    corrected_validation = _weighted_error(validation_traces, calibration)
    improvement = 1.0 - corrected_validation / max(baseline_validation, 1e-9)
    baseline_each = _trace_errors(validation_traces, None)
    corrected_each = _trace_errors(validation_traces, calibration)
    wins = int(np.sum(corrected_each < baseline_each * 0.95))
    required_wins = max(2, int(np.ceil(len(validation_traces) * 0.60)))
    try:
        maximum = validate_mapping(group, calibration)
    except LensCalibrationError as exc:
        record.update(status="rejected_unsafe_mapping", reason=str(exc))
        return None, record
    record["candidate"] = {
        "focal_px": round(focal, 6),
        "principal_point": [round(float(calibration.camera_matrix[0][2]), 6), round(float(calibration.camera_matrix[1][2]), 6)],
        "k1": round(float(parameters[0]), 9),
        "k2": round(float(parameters[1]), 9),
        "fit_rms_before": round(baseline_fit, 6),
        "fit_rms_after": round(corrected_fit, 6),
        "validation_rms_before": round(baseline_validation, 6),
        "validation_rms_after": round(corrected_validation, 6),
        "validation_improvement": round(improvement, 6),
        "validation_wins": wins,
        "validation_trace_count": len(validation_traces),
        "maximum_displacement_px": round(maximum, 6),
        "model_complexity": "k1 with centred optical axis" if conservative else "k1, k2 and constrained optical centre",
    }
    emit(
        f"Validation: curvature RMS {baseline_validation:.3f}px → {corrected_validation:.3f}px "
        f"({improvement * 100:.1f}% improvement), wins={wins}/{len(validation_traces)}, "
        f"maximum displacement={maximum:.2f}px"
    )
    if improvement < MIN_VALIDATION_IMPROVEMENT or wins < required_wins or maximum < 1.5:
        record.update(
            status="rejected_not_proven",
            reason="The fitted model did not improve enough independent traces to justify changing the images.",
        )
        return None, record
    # A two-image fit must help traces from every selected image; this rejects scene-specific curves.
    for image_index in {trace.image_index for trace in validation_traces}:
        image_traces = [trace for trace in validation_traces if trace.image_index == image_index]
        if image_traces and _weighted_error(image_traces, calibration) >= _weighted_error(image_traces, None) * 0.95:
            record.update(
                status="rejected_cross_image_disagreement",
                reason="The fitted model was not independently supported by every selected image.",
            )
            return None, record
    record.update(status="accepted")
    return calibration, record
