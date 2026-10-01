"""Pulls COCO 2017 images/annotations for our target classes via FiftyOne.

FiftyOne's zoo loader does a class-filtered download out of the box: only
images containing at least one annotation from ``classes`` are fetched, so the
on-disk footprint is a small fraction of the full COCO 2017 (~18 GB).

The downloaded images and the companion Mongo DB are both kept inside the
repo (under ``data/raw/fiftyone/``) so deleting the project folder removes
all dataset data from disk.
"""
from __future__ import annotations

import os
from typing import Sequence

from src.constants.classes import CLASSES
from src.constants.paths import FIFTYONE_DATABASE_DIR, FIFTYONE_ZOO_DIR
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir

logger = get_logger(__name__)


def _configure_fiftyone_storage() -> None:
    """Point FiftyOne's Mongo DB + zoo dir at the in-repo locations."""
    ensure_dir(FIFTYONE_DATABASE_DIR)
    ensure_dir(FIFTYONE_ZOO_DIR)
    os.environ["FIFTYONE_DATABASE_DIR"] = str(FIFTYONE_DATABASE_DIR)
    os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(FIFTYONE_ZOO_DIR)


def download_coco_filtered(
    classes: Sequence[str] = CLASSES,
    splits: Sequence[str] = ("train", "validation"),
    max_samples: int | None = None,
    dataset_name: str = "coco2017_mischief_raw",
):
    """Download COCO-2017 samples that contain any of ``classes``.

    Returns a FiftyOne ``Dataset``. The dataset is persisted so re-running this
    function after a successful download is a no-op (instant load).
    """
    _configure_fiftyone_storage()

    import fiftyone as fo
    import fiftyone.zoo as foz

    if fo.dataset_exists(dataset_name):
        ds = fo.load_dataset(dataset_name)
        logger.info(
            "Using cached FiftyOne dataset '%s' (%d samples)", dataset_name, len(ds)
        )
        return ds

    logger.info(
        "Downloading COCO-2017 splits=%s filtered to %d classes: %s",
        list(splits),
        len(classes),
        list(classes),
    )
    ds = foz.load_zoo_dataset(
        "coco-2017",
        splits=list(splits),
        label_types=["detections"],
        classes=list(classes),
        max_samples=max_samples,
        dataset_name=dataset_name,
    )
    ds.persistent = True
    logger.info("Downloaded %d samples into FiftyOne dataset '%s'", len(ds), dataset_name)
    return ds
