"""Orchestrates a YOLOv8 fine-tuning run from ``configs/train.yaml``.

Resuming
--------
If a training run was interrupted (crash, OOM, Ctrl+C) the Ultralytics
trainer drops a ``last.pt`` checkpoint + an ``args.yaml`` next to it.
:func:`resume_training` picks that up and continues from the exact step
with the original hyperparameters. Menu option "R" wires this up for you.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable

from src.config.loader import load_train_config
from src.constants.paths import PROJECT_ROOT, RUNS_DIR, WEIGHTS_DIR, YOLO_DATASET_YAML
from src.models.yolo import load_model, train_model
from src.utils.device import select_device
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def run_training():
    """Fine-tune YOLO using ``configs/train.yaml``."""
    cfg = dict(load_train_config())
    seed_everything(cfg.get("seed", 42))

    if not YOLO_DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset yaml not found at {YOLO_DATASET_YAML}. "
            "Run option 1 'Prepare dataset' first."
        )

    cfg.setdefault("data", str(YOLO_DATASET_YAML))
    cfg.setdefault("project", str(RUNS_DIR))
    ensure_dir(RUNS_DIR)
    ensure_dir(WEIGHTS_DIR)

    loggable = {k: v for k, v in cfg.items() if k != "data"}
    logger.info("Starting YOLO training with config: %s", loggable)

    model, results = train_model(cfg)

    _copy_best_weights(model)
    return results


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------


# Where an interrupted Ultralytics run would have dropped `last.pt`. Ordered
# by preference: the canonical run folder, then the legacy location used by
# the earlier training run.
_DEFAULT_RESUME_SEARCH_PATHS: tuple[Path, ...] = (
    RUNS_DIR / "pet_mischief" / "weights" / "last.pt",
    PROJECT_ROOT / "runs" / "detect" / "outputs" / "runs"
    / "pet_mischief" / "weights" / "last.pt",
)


def _find_last_checkpoint(extra_paths: Iterable[Path] = ()) -> Path | None:
    """Return the first ``last.pt`` that exists on disk, or ``None``."""
    for candidate in tuple(extra_paths) + _DEFAULT_RESUME_SEARCH_PATHS:
        if candidate.exists():
            return candidate
    return None


def resume_training(checkpoint: str | Path | None = None):
    """Resume an interrupted training run from ``last.pt``.

    Ultralytics handles state restoration: ``YOLO(last.pt).train(resume=True)``
    reads ``args.yaml`` next to the checkpoint, restores the optimiser
    state + LR schedule position + current epoch, and continues.

    Parameters
    ----------
    checkpoint
        Explicit path to a ``last.pt`` file. If ``None``, we search the
        standard run folders.
    """
    ckpt = Path(checkpoint) if checkpoint else _find_last_checkpoint()
    if ckpt is None or not ckpt.exists():
        raise FileNotFoundError(
            "No `last.pt` checkpoint found. Expected one of:\n  - "
            + "\n  - ".join(str(p) for p in _DEFAULT_RESUME_SEARCH_PATHS)
            + "\nStart a fresh training run instead (menu option 2)."
        )

    args_yaml = ckpt.parent.parent / "args.yaml"
    if not args_yaml.exists():
        logger.warning(
            "args.yaml not found next to %s -- Ultralytics may fall back to "
            "defaults for missing hyperparameters.",
            ckpt,
        )

    device = select_device()
    logger.info("Resuming training from %s on device=%s", ckpt, device)

    model = load_model(ckpt)
    results = model.train(resume=True, device=device)

    _copy_best_weights(model)
    return results


def _copy_best_weights(model) -> None:
    """Copy the trainer's ``best.pt`` to ``outputs/weights/best.pt`` if present."""
    try:
        trainer = getattr(model, "trainer", None)
        best_attr = getattr(trainer, "best", None) if trainer is not None else None
        best_path = Path(best_attr) if best_attr else None
        if best_path and best_path.exists():
            ensure_dir(WEIGHTS_DIR)
            dest = WEIGHTS_DIR / "best.pt"
            shutil.copy2(best_path, dest)
            logger.info("Copied best weights -> %s", dest)
    except Exception as e:  # noqa: BLE001 - logging is enough here
        logger.warning("Could not copy best weights: %s", e)
