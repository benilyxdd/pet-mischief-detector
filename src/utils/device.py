"""Device selection that works across CUDA (Linux/WSL), MPS (Apple Silicon), and CPU.

Ultralytics accepts device strings such as "0" (cuda index), "mps", or "cpu".
"""
from __future__ import annotations


def select_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"

    return "cpu"
