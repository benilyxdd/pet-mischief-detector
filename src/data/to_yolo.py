"""Export FiftyOne splits to YOLOv5 format and emit a clean ``dataset.yaml``.

Layout produced under ``output_dir``::

    images/train/...     labels/train/...
    images/val/...       labels/val/...
    images/test/...      labels/test/...
    dataset.yaml
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from src.constants.classes import CLASSES
from src.constants.paths import YOLO_DIR
from src.utils.io import save_yaml
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def _write_dataset_yaml(output_dir: Path, classes: Sequence[str]) -> Path:
    """Replace FiftyOne's auto-generated yaml with a ``nc`` + full-splits yaml."""
    yaml_path = output_dir / "dataset.yaml"
    data_yaml = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(classes),
        "names": {i: c for i, c in enumerate(classes)},
    }
    save_yaml(data_yaml, yaml_path)
    logger.info("Wrote %s", yaml_path)
    return yaml_path


def export_splits_to_yolo(
    splits: Mapping[str, object],
    output_dir: Path | str = YOLO_DIR,
    classes: Sequence[str] = CLASSES,
    label_field: str = "ground_truth",
) -> Path:
    """Export each split view to ``output_dir`` as a YOLOv5 dataset."""
    import fiftyone as fo

    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    for name, view in splits.items():
        logger.info(
            "Exporting split '%s' (%d samples) to YOLOv5 at %s",
            name,
            len(view),
            output_dir,
        )
        view.export(
            export_dir=str(output_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field=label_field,
            classes=list(classes),
            split=name,
        )

    return _write_dataset_yaml(output_dir, classes)
