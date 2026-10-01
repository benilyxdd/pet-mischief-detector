"""Infer: image -> detections -> mischief pairs -> annotated image."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.config.loader import load_mischief_config
from src.constants.paths import PREDICTIONS_DIR
from src.mischief.geometry import BBox
from src.mischief.rules import build_tiers
from src.mischief.scorer import Detection, MischiefPair, score_detections
from src.models.yolo import load_model, predict_image
from src.utils.io import save_json
from src.utils.logging import get_logger
from src.utils.paths import ensure_dir
from src.utils.weights import find_weights
from src.inference.draw import annotate_image

logger = get_logger(__name__)


def _extract_detections(result) -> tuple[list[Detection], tuple[int, int]]:
    """Convert an ultralytics ``Results`` object to our internal types."""
    h, w = result.orig_shape[:2]

    detections: list[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return detections, (w, h)

    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    for (x1, y1, x2, y2), cf, ci in zip(xyxy, confs, clses):
        label = names.get(int(ci), str(int(ci)))
        detections.append(
            Detection(
                label=label,
                bbox=BBox(float(x1), float(y1), float(x2), float(y2)),
                confidence=float(cf),
            )
        )
    return detections, (w, h)


def run_inference(
    image_path: str | Path,
    output_path: str | Path | None = None,
    weights: str | Path | None = None,
    conf: float = 0.15,
    iou: float = 0.60,
) -> Path | None:
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    weights_path = Path(weights) if weights else find_weights()
    logger.info("Loading weights from %s", weights_path)
    model = load_model(weights_path)

    logger.info("Running inference on %s", image_path)
    result = predict_image(model, image_path, conf=conf, iou=iou)
    if result is None:
        logger.warning("No results returned from model.")
        return None

    detections, (w, h) = _extract_detections(result)
    logger.info("Detected %d objects", len(detections))

    cfg = load_mischief_config()
    tiers = build_tiers(cfg.get("rules", {}))
    pairs = score_detections(
        detections=detections,
        image_wh=(w, h),
        tiers=tiers,
        thresholds=cfg.get("thresholds", {}),
        weights=cfg.get("weights", {}),
        messages_cfg=cfg.get("messages", {}),
        ignore_distance=float(cfg.get("ignore_distance", 0.35)),
    )

    for p in pairs:
        logger.info(
            "  %s x %s -> %s (score=%.3f) %s",
            p.pet.label,
            p.obj.label,
            p.risk_level,
            p.score,
            p.message,
        )

    if output_path is None:
        output_path = PREDICTIONS_DIR / f"{image_path.stem}_annotated.jpg"
    else:
        output_path = Path(output_path)

    ensure_dir(output_path.parent)
    annotate_image(image_path, detections, pairs, output_path)
    logger.info("Annotated image saved to %s", output_path)

    # Persist the structured scoring breakdown next to the annotated image
    # so the report can quote exact numbers (IoU / proximity / score / tier
    # / risk / message) per pair without re-scraping logs. The verdict.txt
    # is the single banner string the visualisation drew.
    pairs_path = output_path.with_name(f"{output_path.stem}_pairs.json")
    verdict_path = output_path.with_name(f"{output_path.stem}_verdict.txt")
    _save_inference_artifacts(
        image_path=image_path,
        image_wh=(w, h),
        detections=detections,
        pairs=pairs,
        annotated_path=output_path,
        pairs_path=pairs_path,
        verdict_path=verdict_path,
        weights_path=weights_path,
        conf=conf,
        iou=iou,
    )
    logger.info("Pair breakdown saved to %s", pairs_path)
    logger.info("Verdict saved to %s", verdict_path)
    return output_path


def _save_inference_artifacts(
    image_path: Path,
    image_wh: tuple[int, int],
    detections: Sequence[Detection],
    pairs: Sequence[MischiefPair],
    annotated_path: Path,
    pairs_path: Path,
    verdict_path: Path,
    weights_path: Path,
    conf: float,
    iou: float,
) -> None:
    w, h = image_wh
    payload = {
        "image": str(image_path),
        "annotated_image": str(annotated_path),
        "weights": str(weights_path),
        "image_wh": [w, h],
        "inference_params": {"conf": conf, "iou": iou},
        "detections": [
            {
                "label": d.label,
                "confidence": d.confidence,
                "bbox_xyxy": [d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2],
            }
            for d in detections
        ],
        "pairs": [
            {
                "pet": p.pet.label,
                "object": p.obj.label,
                "pet_confidence": p.pet.confidence,
                "object_confidence": p.obj.confidence,
                "iou": p.iou,
                "proximity": p.proximity,
                "raw_score": p.raw_score,
                "score": p.score,
                "tier": p.tier_name,
                "risk_level": p.risk_level,
                "message": p.message,
            }
            for p in pairs
        ],
    }

    if pairs:
        top = pairs[0]
        payload["top_verdict"] = {
            "risk_level": top.risk_level,
            "score": top.score,
            "message": top.message,
        }
        verdict_text = (
            f"[{top.risk_level.upper()}] {top.message} "
            f"(score={top.score:.2f}, iou={top.iou:.2f}, "
            f"prox={top.proximity:.2f})"
        )
    else:
        payload["top_verdict"] = None
        verdict_text = "No pet-object pairs within risk radius."

    save_json(payload, pairs_path)
    verdict_path.write_text(verdict_text + "\n")
