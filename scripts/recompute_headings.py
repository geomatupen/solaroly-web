#!/usr/bin/env python3
"""
Recompute camera headings in media/colmap/<dataset>/colmap_meta.json
from stored quaternions (qw,qx,qy,qz) using updated orientation logic.

Usage:
  python scripts/recompute_headings.py <dataset>
Example:
  python scripts/recompute_headings.py optical_jan
"""
import json
import sys
from pathlib import Path
import math
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEDIA_COLMAP = PROJECT_ROOT / "media" / "colmap"


def _quaternion_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    n = qw*qw + qx*qx + qy*qy + qz*qz
    if n < 1e-8:
        return np.eye(3)
    s = 2.0 / n
    x, y, z = qx, qy, qz
    w = qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - s * (yy + zz),     s * (xy - wz),     s * (xz + wy)],
        [    s * (xy + wz), 1 - s * (xx + zz),     s * (yz - wx)],
        [    s * (xz - wy),     s * (yz + wx), 1 - s * (xx + yy)],
    ], dtype=float)


def _rotation_matrix_to_heading(rot: np.ndarray) -> float:
    # Convert world-to-camera R to camera-to-world.
    Rcw = rot.T
    # Camera axes in world frame.
    z_w = Rcw @ np.array([0.0, 0.0, 1.0])   # optical axis
    y_w = Rcw @ np.array([0.0, 1.0, 0.0])   # image down
    up_w = -y_w                              # image up
    # Choose axis: use image up when nadir, optical axis otherwise.
    z_norm = float(np.linalg.norm(z_w)) or 1.0
    pitch_cos = abs(z_w[2]) / z_norm
    if pitch_cos > 0.7:
        east, north = up_w[0], up_w[1]
    else:
        east, north = z_w[0], z_w[1]
    if abs(east) < 1e-8 and abs(north) < 1e-8:
        x_w = Rcw @ np.array([1.0, 0.0, 0.0])
        east, north = x_w[0], x_w[1]
    # 0=N, +CW convention
    heading = -math.degrees(math.atan2(east, north))
    while heading > 180.0:
        heading -= 360.0
    while heading < -180.0:
        heading += 360.0
    return float(heading)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/recompute_headings.py <dataset>")
        sys.exit(1)
    dataset = sys.argv[1]
    meta_path = MEDIA_COLMAP / dataset / "colmap_meta.json"
    if not meta_path.exists():
        print(f"Missing metadata: {meta_path}")
        sys.exit(2)
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    updated = 0
    for name, entry in data.items():
        try:
            qw = float(entry.get("qw"))
            qx = float(entry.get("qx"))
            qy = float(entry.get("qy"))
            qz = float(entry.get("qz"))
        except Exception:
            continue
        R = _quaternion_to_matrix(qw, qx, qy, qz)
        heading = _rotation_matrix_to_heading(R)
        entry["rotation"] = heading
        # Ensure source reflects recomputation from COLMAP orientation
        entry["rotation_source"] = "colmap"
        # Remove any prior offset applied
        if "rotation_offset_deg" in entry:
            entry.pop("rotation_offset_deg", None)
        updated += 1
    backup = meta_path.with_suffix(".json.bak")
    backup.write_text(meta_path.read_text(encoding="utf-8"), encoding="utf-8")
    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Updated {updated} entries in {meta_path}")


if __name__ == "__main__":
    main()
