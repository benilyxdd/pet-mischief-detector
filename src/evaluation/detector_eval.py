"""Run ultralytics' validator on the test split and persist the metrics.

Evaluation hyperparameters (imgsz, NMS IoU, TTA, FP16) are loaded from
``configs/eval.yaml`` so they can be tuned without code changes. Any
explicit kwargs passed to :func:`evaluate_detector` override the YAML.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.config.loader import load_eval_config
from src.constants.classes import CLASSES
from src.constants.paths import EVAL_DIR, YOLO_DATASET_YAML
from src.models.yolo import load_model
from src.utils.device import select_device
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir
from src.utils.weights import find_weights

logger = get_logger(__name__)


# Keys from configs/eval.yaml that we forward verbatim to YOLO.val().
_FORWARDED_EVAL_KEYS: tuple[str, ...] = (
    "imgsz",
    "iou",
    "conf",
    "augment",
    "half",
    "max_det",
    "rect",
    "plots",
    "verbose",
    "batch",
    "workers",
)


def evaluate_detector(
    weights: str | Path | None = None,
    data_yaml: str | Path = YOLO_DATASET_YAML,
    split: str = "test",
    imgsz: int | None = None,
    **overrides: Any,
):
    """Run ``YOLO.val`` on ``split`` and dump a JSON metrics report.

    Parameters
    ----------
    weights
        Path to a ``.pt`` checkpoint. Auto-discovered via
        :func:`src.utils.weights.find_weights` if not provided.
    data_yaml
        Dataset YAML (defaults to the curated ``data/yolo/dataset.yaml``).
    split
        Which split in ``data_yaml`` to evaluate on -- defaults to ``test``.
    imgsz
        Optional override for the eval image size. If ``None``, the value
        from ``configs/eval.yaml`` (or Ultralytics' default) is used. Kept
        as an explicit kwarg for backwards compatibility with older callers.
    **overrides
        Any other key that ``YOLO.val`` accepts will override the value from
        ``configs/eval.yaml``.
    """
    weights_path = Path(weights) if weights else find_weights()
    logger.info("Evaluating weights=%s on split='%s'", weights_path, split)

    model = load_model(weights_path)
    device = select_device()

    out_dir = ensure_dir(EVAL_DIR / split)

    eval_cfg = dict(load_eval_config())
    if imgsz is not None:
        eval_cfg["imgsz"] = imgsz
    eval_cfg.update(overrides)

    val_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "split": split,
        "device": device,
        "project": str(out_dir.parent),
        "name": out_dir.name,
        "exist_ok": True,
    }
    for key in _FORWARDED_EVAL_KEYS:
        if key in eval_cfg and eval_cfg[key] is not None:
            val_kwargs[key] = eval_cfg[key]
    val_kwargs.setdefault("plots", True)
    val_kwargs.setdefault("verbose", False)

    loggable = {k: v for k, v in val_kwargs.items() if k != "data"}
    logger.info("model.val() kwargs: %s", loggable)

    metrics = model.val(**val_kwargs)

    box = metrics.box
    per_class_maps = list(box.maps) if hasattr(box, "maps") else []

    # Mean precision / recall across classes (scalars for the report summary).
    mean_precision = float(box.mp) if hasattr(box, "mp") else None
    mean_recall = float(box.mr) if hasattr(box, "mr") else None

    # Per-class precision / recall vectors (present on the box object for
    # recent ultralytics versions; fall back gracefully if unavailable).
    per_class_p = _as_list(getattr(box, "p", None))
    per_class_r = _as_list(getattr(box, "r", None))
    per_class_f1 = _as_list(getattr(box, "f1", None))

    confusion_matrix_path = out_dir / "confusion_matrix_normalized.png"
    confusion_matrix_csv = _dump_confusion_matrix_csv(metrics, out_dir)

    # Per-image latency. Ultralytics fills `metrics.speed` with the
    # preprocess / inference / postprocess milliseconds, averaged over the
    # whole eval set. We persist all three plus a convenience scalar so
    # the report can quote a single "ms / image" number directly.
    speed = getattr(metrics, "speed", {}) or {}
    speed_ms = {k: float(v) for k, v in speed.items() if v is not None}
    inference_ms = float(speed_ms.get("inference", 0.0))
    total_ms = float(sum(speed_ms.values())) if speed_ms else 0.0
    fps = (1000.0 / total_ms) if total_ms > 0 else None

    report: dict[str, Any] = {
        "split": split,
        "weights": str(weights_path),
        "device": device,
        "mAP50-95": float(box.map),
        "mAP50": float(box.map50),
        "mAP75": float(box.map75),
        "precision_mean": mean_precision,
        "recall_mean": mean_recall,
        "speed_ms": speed_ms,
        "inference_ms_per_image": inference_ms,
        "total_ms_per_image": total_ms,
        "fps": fps,
        "eval_config": {k: val_kwargs.get(k) for k in _FORWARDED_EVAL_KEYS if k in val_kwargs},
        "confusion_matrix_png": str(confusion_matrix_path)
        if confusion_matrix_path.exists()
        else None,
        "confusion_matrix_csv": str(confusion_matrix_csv) if confusion_matrix_csv else None,
        "per_class": {
            CLASSES[i]: {
                "AP50-95": float(per_class_maps[i]) if i < len(per_class_maps) else None,
                "precision": float(per_class_p[i]) if i < len(per_class_p) else None,
                "recall": float(per_class_r[i]) if i < len(per_class_r) else None,
                "f1": float(per_class_f1[i]) if i < len(per_class_f1) else None,
            }
            for i in range(len(CLASSES))
        },
    }

    logger.info(
        "Overall mAP@0.5 = %.4f | mAP@[0.5:0.95] = %.4f | mAP@0.75 = %.4f",
        report["mAP50"],
        report["mAP50-95"],
        report["mAP75"],
    )
    if total_ms > 0:
        logger.info(
            "Speed (device=%s): inference=%.2f ms | total=%.2f ms/image | %.1f FPS",
            device,
            inference_ms,
            total_ms,
            fps if fps is not None else 0.0,
        )
    for cls, vals in report["per_class"].items():
        logger.info("  %-14s AP@[0.5:0.95] = %s", cls, vals["AP50-95"])

    save_json(report, out_dir / "metrics.json")
    logger.info("Saved metrics to %s", out_dir / "metrics.json")
    return report


def _as_list(x: Any) -> list[float]:
    """Best-effort conversion of a numpy array / tensor to a ``list[float]``."""
    if x is None:
        return []
    try:
        return [float(v) for v in x]
    except TypeError:
        return []


def _dump_confusion_matrix_csv(metrics: Any, out_dir: Path) -> Path | None:
    """Dump the Ultralytics confusion matrix as a CSV for offline analysis.

    Ultralytics builds an ``(nc+1) x (nc+1)`` matrix where the extra row /
    column is the "background" (false positive / false negative) bucket.
    Returns the written path, or ``None`` if the matrix wasn't available.
    """
    cm = getattr(metrics, "confusion_matrix", None)
    raw = getattr(cm, "matrix", None) if cm is not None else None
    if raw is None:
        return None

    try:
        import numpy as np
    except ImportError:
        return None

    matrix = np.asarray(raw)
    labels = list(CLASSES) + ["background"]
    csv_path = out_dir / "confusion_matrix.csv"

    with open(csv_path, "w") as f:
        f.write("," + ",".join(labels) + "\n")
        for i, row_name in enumerate(labels):
            if i >= matrix.shape[0]:
                break
            row = matrix[i].tolist()
            f.write(row_name + "," + ",".join(f"{float(v):.6g}" for v in row) + "\n")

    logger.info("Saved confusion matrix CSV to %s", csv_path)
    return csv_path
