"""Reproducible 80 / 10 / 10 split using FiftyOne's random-split utility."""
from __future__ import annotations

from typing import Mapping

from src.utils.logging import get_logger

logger = get_logger(__name__)


SPLIT_TAGS = ("train", "val", "test")


def split_dataset(
    dataset,
    ratios: Mapping[str, float] | None = None,
    seed: int = 42,
):
    """Tag each sample in ``dataset`` with its split name and return per-split views.

    ``dataset`` must be a mutable FiftyOne ``Dataset`` (not a view) because
    ``random_split`` mutates the underlying sample tags in-place.
    """
    ratios = dict(ratios or {"train": 0.8, "val": 0.1, "test": 0.1})

    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0 (got {total}): {ratios}")

    import fiftyone.utils.random as four

    # Clear any previous split tags so repeated runs are deterministic.
    for tag in SPLIT_TAGS:
        try:
            dataset.untag_samples(tag)
        except Exception:
            pass

    four.random_split(dataset, ratios, seed=seed)

    splits = {name: dataset.match_tags(name) for name in ratios}
    for name, view in splits.items():
        logger.info("Split '%s': %d samples", name, len(view))
    return splits
