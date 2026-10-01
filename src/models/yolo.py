"""Thin wrappers around ``ultralytics.YOLO`` so other modules don't import it directly."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.device import select_device
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_model(weights: str | Path):
    """Load a YOLO model from a checkpoint (``.pt``)."""
    from ultralytics import YOLO

    return YOLO(str(weights))


def train_model(cfg: dict[str, Any]):
    """Fine-tune a YOLO model.

    ``cfg`` is a mutable copy of ``configs/train.yaml``; ``model`` and
    ``device`` are consumed here, everything else is forwarded as-is to
    ``ultralytics.YOLO.train()``.
    """
    from ultralytics import YOLO

    model_name = cfg.pop("model", "yolov8n.pt")
    device = cfg.pop("device", None) or select_device()
    logger.info("Loading base model '%s'. Training on device: %s", model_name, device)

    model = YOLO(model_name)
    results = model.train(device=device, **cfg)
    return model, results


def predict_image(
    model,
    image_path: str | Path,
    conf: float = 0.25,
    iou: float = 0.45,
):
    """Run inference on a single image and return the first ``Results`` object."""
    results = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
    )
    return results[0] if results else None
