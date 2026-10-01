"""Annotate a detection image with mischief pairs + warning banner."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.constants.classes import PETS
from src.mischief.scorer import Detection, MischiefPair


# BGR because OpenCV.
RISK_COLORS = {
    "high": (0, 0, 255),      # red
    "medium": (0, 165, 255),  # orange
    "low": (0, 200, 0),       # green
}
PET_COLOR = (255, 200, 0)     # cyan-ish
OBJ_COLOR = (200, 200, 200)   # light grey
TEXT_BG = (0, 0, 0)
TEXT_FG = (255, 255, 255)


def _draw_box(img, det: Detection, color: tuple[int, int, int]) -> None:
    import cv2

    x1, y1, x2, y2 = (int(det.bbox.x1), int(det.bbox.y1), int(det.bbox.x2), int(det.bbox.y2))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{det.label} {det.confidence:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        img,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def _draw_banner(img, text: str, color: tuple[int, int, int]):
    import cv2

    banner_h = 56
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (img.shape[1], banner_h), color, -1)
    img = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
    cv2.putText(
        img,
        text,
        (12, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_FG,
        2,
        cv2.LINE_AA,
    )
    return img


def annotate_image(
    image_path: str | Path,
    detections: Sequence[Detection],
    pairs: Sequence[MischiefPair],
    output_path: str | Path,
) -> Path:
    import cv2

    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image {image_path}")

    # All raw detections first.
    for det in detections:
        color = PET_COLOR if det.label in PETS else OBJ_COLOR
        _draw_box(img, det, color)

    # Connecting lines for each scored pair.
    for pair in pairs:
        color = RISK_COLORS.get(pair.risk_level, (255, 255, 255))
        pcx, pcy = map(int, pair.pet.bbox.center)
        ocx, ocy = map(int, pair.obj.bbox.center)
        cv2.line(img, (pcx, pcy), (ocx, ocy), color, 2)

    if pairs:
        top = pairs[0]
        banner_color = RISK_COLORS.get(top.risk_level, (255, 255, 255))
        banner_text = (
            f"[{top.risk_level.upper()}] {top.message}  (score={top.score:.2f})"
        )
        img = _draw_banner(img, banner_text, banner_color)
    else:
        img = _draw_banner(img, "No pet-object pairs within risk radius.", (80, 80, 80))

    cv2.imwrite(str(output_path), img)
    return output_path
