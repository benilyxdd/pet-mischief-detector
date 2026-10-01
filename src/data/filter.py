"""Drop non-target annotations and empty samples from a FiftyOne dataset."""
from __future__ import annotations

from typing import Sequence

from src.constants.classes import CLASSES
from src.utils.logging import get_logger

logger = get_logger(__name__)


def filter_to_target_classes(
    dataset,
    classes: Sequence[str] = CLASSES,
    label_field: str = "ground_truth",
):
    """Return a view containing only the annotations for ``classes``.

    COCO images usually contain several classes we don't care about (people,
    bananas, trucks, ...). ``filter_labels`` removes those detections from the
    view, and the subsequent ``match`` drops any sample that ends up with zero
    target detections.
    """
    from fiftyone import ViewField as F

    view = dataset.filter_labels(label_field, F("label").is_in(list(classes)))
    view = view.match(F(f"{label_field}.detections").length() > 0)
    logger.info(
        "Filtered '%s': %d samples -> %d samples containing target classes",
        dataset.name,
        len(dataset),
        len(view),
    )
    return view
