"""Toggle web features by reading environment variables.

Example: set ``PVRT_ENABLE_YOLO=1`` before starting the server to show YOLO UI
and endpoints. Leave it unset (or 0) to hide those pieces.
This enables users to avoid installing heavy dependencies like Detectron2, yolo, colmap if they don't need it for now. eg. just use detectron and dont install yolo and colmap (for location and orientation optimization for individual images)
"""

from __future__ import annotations

import os
from typing import List


def _env_flag(name: str, default: bool) -> bool:
    """Read a bool flag from environment variables."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class WebSettings:
    """Central place to toggle heavy integrations. True if the feature is needed and installed already. if it is not installed keep false."""

    def __init__(self) -> None:
        self.enable_detectron = _env_flag("PVRT_ENABLE_DETECTRON", True)  # Change this to turn on or off Detectron
        self.enable_yolo = _env_flag("PVRT_ENABLE_YOLO", True)  # Change this to turn on or off YOLO
        self.enable_colmap = _env_flag("PVRT_ENABLE_COLMAP", True) # Change this to turn on or off COLMAP (for location and orientation optimization for individual images)
        self.enable_thermal_data_extraction = _env_flag(
            "PVRT_ENABLE_THERMAL",
            True,
        )  # Change this to disable DJI thermal SDK dependent features

    @property
    def enabled_backends(self) -> List[str]:
        backends: List[str] = []
        if self.enable_detectron:
            backends.append("detectron")
        if self.enable_yolo:
            backends.append("yolo")
        return backends

    def as_feature_payload(self) -> dict:
        return {
            "colmap": self.enable_colmap,
            "detectron": self.enable_detectron,
            "yolo": self.enable_yolo,
            "thermal": self.enable_thermal_data_extraction,
            "thermal_data_extraction": self.enable_thermal_data_extraction,
            "backends": self.enabled_backends,
        }


settings = WebSettings()
