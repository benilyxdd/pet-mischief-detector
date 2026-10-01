"""Risk-tier definitions built from ``configs/mischief.yaml``."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RiskTier:
    name: str                    # "high" / "medium" / "low"
    objects: frozenset[str]      # object class names belonging to this tier
    weight: float                # multiplier in [0, 1] applied to raw score


def build_tiers(rules_cfg: Mapping[str, Mapping]) -> list[RiskTier]:
    """Convert the dict under ``rules:`` in the YAML into ``RiskTier`` objects."""
    tiers: list[RiskTier] = []
    for tier_key, spec in rules_cfg.items():
        name = tier_key.replace("_risk", "")
        tiers.append(
            RiskTier(
                name=name,
                objects=frozenset(spec.get("objects", []) or []),
                weight=float(spec.get("weight", 0.0)),
            )
        )
    return tiers


def tier_for_object(tiers: Sequence[RiskTier], object_label: str) -> RiskTier | None:
    for tier in tiers:
        if object_label in tier.objects:
            return tier
    return None
