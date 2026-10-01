"""Core mischief scorer: detections -> ranked list of (pet, object) risk pairs."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.constants.classes import PETS
from src.mischief.geometry import BBox, iou, normalized_proximity
from src.mischief.messages import format_message
from src.mischief.rules import RiskTier, tier_for_object


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: BBox
    confidence: float


@dataclass(frozen=True)
class MischiefPair:
    pet: Detection
    obj: Detection
    iou: float
    proximity: float
    raw_score: float      # IoU/proximity combination before tier weighting
    score: float          # tier-weighted, in [0, 1]
    tier_name: str        # "high" / "medium" / "low"
    risk_level: str       # final bucket after threshold comparison
    message: str


def _risk_from_score(score: float, thresholds: dict) -> str:
    if score >= float(thresholds.get("high", 0.66)):
        return "high"
    if score >= float(thresholds.get("medium", 0.33)):
        return "medium"
    return "low"


def score_detections(
    detections: Sequence[Detection],
    image_wh: tuple[int, int],
    tiers: Sequence[RiskTier],
    thresholds: dict,
    weights: dict,
    messages_cfg: dict,
    ignore_distance: float = 0.35,
) -> list[MischiefPair]:
    """Return mischief pairs for the image, sorted high-score-first."""
    img_w, img_h = image_wh
    pets = [d for d in detections if d.label in PETS]
    objects = [d for d in detections if d.label not in PETS]

    w_iou = float(weights.get("iou", 0.5))
    w_prox = float(weights.get("proximity", 0.5))
    total_w = (w_iou + w_prox) or 1.0

    pairs: list[MischiefPair] = []
    for pet in pets:
        for obj in objects:
            tier = tier_for_object(tiers, obj.label)
            if tier is None:
                continue

            prox = normalized_proximity(pet.bbox, obj.bbox, img_w, img_h)
            # Skip pairs whose normalized distance exceeds `ignore_distance`.
            # prox = 1 - dist/diag, so dist/diag > ignore_distance  <=>  prox < 1 - ignore_distance.
            if prox < (1.0 - ignore_distance):
                continue

            pair_iou = iou(pet.bbox, obj.bbox)
            raw = (w_iou * pair_iou + w_prox * prox) / total_w
            score = raw * tier.weight

            risk = _risk_from_score(score, thresholds)
            # Hard cap: low-tier pairs (couch/bed/...) can never be "high".
            if tier.name == "low" and risk == "high":
                risk = "medium" if score >= float(thresholds.get("medium", 0.33)) else "low"

            message = format_message(messages_cfg, risk, pet.label, obj.label)
            pairs.append(
                MischiefPair(
                    pet=pet,
                    obj=obj,
                    iou=pair_iou,
                    proximity=prox,
                    raw_score=raw,
                    score=score,
                    tier_name=tier.name,
                    risk_level=risk,
                    message=message,
                )
            )

    pairs.sort(key=lambda p: p.score, reverse=True)
    return pairs
