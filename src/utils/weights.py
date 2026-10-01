"""Helpers for locating trained YOLO weights on disk."""
from __future__ import annotations

from pathlib import Path

from src.constants.paths import RUNS_DIR, WEIGHTS_DIR


def find_weights() -> Path:
    """Return the path to the most recently saved best-performing weights.

    Looks first at the hand-curated copy under ``outputs/weights/best.pt`` and
    falls back to scanning ``outputs/runs/`` for any ``best.pt`` left behind by
    ultralytics.
    """
    direct = WEIGHTS_DIR / "best.pt"
    if direct.exists():
        return direct

    candidates = sorted(
        RUNS_DIR.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "No trained weights found. Expected "
        f"{WEIGHTS_DIR / 'best.pt'} or a best.pt under {RUNS_DIR}. "
        "Run option 2 'Train model' first."
    )
