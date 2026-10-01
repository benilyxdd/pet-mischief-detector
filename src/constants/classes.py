"""Canonical class definitions for the Pet Mischief Detector.

All class names are taken verbatim from the COCO 2017 label set so they can be
passed directly to FiftyOne / ultralytics without any remapping.
"""
from __future__ import annotations


PETS: tuple[str, ...] = (
    "cat",
    "dog",
)

CLASSES: tuple[str, ...] = (
    "cat",
    "dog",
    "cup",
    "bottle",
    "wine glass",
    "vase",
    "potted plant",
    "laptop",
    "couch",
    "bed",
    "chair",
    "dining table",
    "tv",
)
NUM_CLASSES: int = len(CLASSES)

CLASS_TO_IDX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CLASSES)}
