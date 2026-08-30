from __future__ import annotations

import math
from typing import Optional


def compute_meters_per_pixel(
    altitude_m: Optional[float],
    width_px: Optional[int],
    focal_length_mm: Optional[float],
    focal_length_35mm: Optional[float],
    focal_plane_x_res: Optional[float],
    focal_plane_res_unit: Optional[int],
    height_px: Optional[int] = None,
) -> Optional[float]:
    """Estimate nadir, flat-plane GSD from image-specific camera metadata."""
    if not altitude_m or altitude_m <= 0 or not width_px or width_px <= 0:
        return None

    # Prefer real focal length and physical pixel pitch when available. This
    # avoids the rounding and aspect-ratio ambiguity of 35 mm equivalence.
    if focal_length_mm and focal_plane_x_res and focal_plane_x_res > 0 and focal_plane_res_unit:
        unit = int(focal_plane_res_unit)
        per_mm = None
        if unit == 2:      # inches
            per_mm = float(focal_plane_x_res) / 25.4
        elif unit == 3:    # centimeters
            per_mm = float(focal_plane_x_res) / 10.0
        elif unit == 4:    # millimeters
            per_mm = float(focal_plane_x_res)
        elif unit == 5:    # micrometers
            per_mm = float(focal_plane_x_res) * 1000.0
        if per_mm and per_mm > 0:
            pixel_size_mm = 1.0 / per_mm
            try:
                # Both optical values are in millimetres. Their ratio is
                # dimensionless, leaving the altitude result in metres/pixel.
                return (float(altitude_m) * pixel_size_mm) / float(focal_length_mm)
            except Exception:
                return None

    if focal_length_35mm and focal_length_35mm > 1e-6 and height_px and height_px > 0:
        try:
            # 35 mm-equivalent focal length is defined against the frame
            # diagonal, not the 36 mm horizontal edge.
            image_diagonal_px = math.hypot(float(width_px), float(height_px))
            full_frame_diagonal_mm = math.hypot(36.0, 24.0)
            focal_length_px = (
                float(focal_length_35mm)
                * image_diagonal_px
                / full_frame_diagonal_mm
            )
            if focal_length_px > 1e-6:
                return float(altitude_m) / focal_length_px
        except Exception:
            return None

    return None
