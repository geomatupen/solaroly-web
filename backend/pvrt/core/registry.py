# backend/pvrt/core/registry.py
"""
Backend registry
----------------
A minimal, typed interface for ML backends (Detectron, YOLO, etc.).
The web layer calls these methods without caring which library is used.

Add a backend by calling `register_backend("detectron", DetectronBackend)`.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Protocol, runtime_checkable


# ---------- config objects passed into backends ----------

@dataclass
class TrainConfig:
    train_dir: Path
    val_dir: Path
    out_dir: Path
    use_thermal: bool
    max_iter: int
    base_lr: float
    ims_per_batch: int
    # optional name to store alongside artifacts
    run_name: str = ""
    model_type: str = "maskrcnn"


@dataclass
class PredictConfig:
    images_dir: Path
    out_dir: Path
    weights_dir: Path
    # user request: prefer thermal if model supports it and thermal exists
    use_thermal: bool
    score_thresh: Optional[float] = None


# ---------- what a backend must implement ----------

@runtime_checkable
class Backend(Protocol):
    """Minimal interface the HTTP layer relies on."""
    def train(self, cfg: TrainConfig) -> Path:
        """Run training. Return path to the final weights file or run folder."""
        ...

    def predict(self, cfg: PredictConfig) -> Path:
        """Run inference. Return path that contains per-image *.json results."""
        ...

    def read_meta(self, weights_dir: Path) -> dict:
        """
        Read metadata stored with a trained run (e.g., input_mode, classes).
        Should return {} if unavailable.
        """
        ...


# ---------- a simple registry ----------

_FABRICS: Dict[str, Callable[[], Backend]] = {}


def register_backend(name: str, factory: Callable[[], Backend]) -> None:
    """
    Register a backend by name.
    Example:
        register_backend("detectron", lambda: DetectronBackend())
    """
    key = name.strip().lower()
    if not key:
        raise ValueError("backend name cannot be empty")
    _FABRICS[key] = factory


def get_backend(name: str) -> Backend:
    """
    Construct a backend by name. Raises KeyError if unknown.
    """
    key = name.strip().lower()
    if key not in _FABRICS:
        raise KeyError(f"Unknown backend '{name}'. Available: {', '.join(sorted(_FABRICS)) or 'none'}")
    return _FABRICS[key]()


def list_backends() -> list[str]:
    """For UI/debug: which backends are available?"""
    return sorted(_FABRICS.keys())
