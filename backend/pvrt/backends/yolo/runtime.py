"""Runtime selection shared by YOLO training and inference."""

from __future__ import annotations

from typing import Any


def resolve_yolo_device(torch_module: Any | None = None) -> str | int:
    """Choose CPU for a CPU-only PyTorch build and CUDA device 0 otherwise.

    A CUDA-enabled build that cannot access CUDA is treated as a broken GPU
    installation. Silently falling back in that case could turn a planned GPU
    training job into an unexpectedly long CPU job.
    """
    if torch_module is None:
        import torch as torch_module

    version = getattr(torch_module, "version", None)
    cuda_build = getattr(version, "cuda", None)
    if not cuda_build:
        return "cpu"

    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None and cuda.is_available():
        return 0

    raise RuntimeError(
        "YOLO cannot access CUDA. This installation uses a CUDA-enabled PyTorch "
        "build, but no working CUDA GPU is available. Fix the NVIDIA driver/CUDA "
        "runtime, or install the CPU-only PyTorch build to run YOLO on CPU."
    )
