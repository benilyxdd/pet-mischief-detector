"""Pure-geometry helpers used by the mischief scorer.

All coordinates are in the image pixel frame (top-left origin), matching what
ultralytics returns via ``Results.boxes.xyxy``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def iou(a: BBox, b: BBox) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def min_edge_distance(a: BBox, b: BBox) -> float:
    """Pixel distance between the two boxes' closest edges (0 if they overlap)."""
    dx = max(0.0, max(a.x1 - b.x2, b.x1 - a.x2))
    dy = max(0.0, max(a.y1 - b.y2, b.y1 - a.y2))
    return math.hypot(dx, dy)


def image_diagonal(width: int, height: int) -> float:
    return math.hypot(width, height)


def normalized_proximity(a: BBox, b: BBox, img_w: int, img_h: int) -> float:
    """Proximity score in [0, 1]. 1 = touching/overlapping, 0 = diagonal apart.

    Uses a linear decay with image diagonal so the score is scale-invariant.
    """
    diag = image_diagonal(img_w, img_h)
    if diag <= 0:
        return 0.0
    dist = min_edge_distance(a, b)
    return max(0.0, 1.0 - dist / diag)
