"""End-to-end data preparation: download -> filter -> split -> export."""
from __future__ import annotations

from pathlib import Path

from src.config.loader import load_data_config
from src.constants.classes import CLASSES
from src.constants.paths import FIGURES_DIR, YOLO_DIR
from src.data.download import download_coco_filtered
from src.data.filter import filter_to_target_classes
from src.data.split import split_dataset
from src.data.stats import (
    compute_class_counts,
    log_per_split_stats,
    plot_class_distribution,
    save_dataset_stats,
)
from src.data.to_yolo import export_splits_to_yolo
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

logger = get_logger(__name__)


def prepare_dataset() -> Path:
    """Run the full data pipeline. Returns the path to the YOLO ``dataset.yaml``.

    Idempotent: the FiftyOne datasets are persistent, so re-running skips the
    download step. The YOLO export directory is re-created fresh each run.
    """
    cfg = load_data_config()
    seed = cfg["split"]["seed"]
    seed_everything(seed)

    import fiftyone as fo

    label_field = cfg["fiftyone"]["label_field"]

    logger.info("Step 1/5: downloading COCO 2017 filtered to target classes")
    raw_name = f"{cfg['fiftyone']['dataset_name']}_raw"
    raw = download_coco_filtered(
        classes=CLASSES,
        splits=tuple(cfg["fiftyone"]["splits_to_download"]),
        max_samples=cfg["fiftyone"].get("max_samples_per_split"),
        dataset_name=raw_name,
    )

    logger.info("Step 2/5: filtering annotations to target classes")
    filtered_name = cfg["fiftyone"]["dataset_name"]
    if fo.dataset_exists(filtered_name):
        fo.delete_dataset(filtered_name)
    view = filter_to_target_classes(raw, classes=CLASSES, label_field=label_field)
    filtered = view.clone(name=filtered_name)
    filtered.persistent = True
    logger.info("Filtered dataset: %d samples", len(filtered))

    logger.info("Step 3/5: computing class distribution for merged dataset")
    counts = compute_class_counts(filtered, label_field=label_field)
    total_instances = sum(counts.values())
    plot_class_distribution(
        counts,
        output_path=FIGURES_DIR / "class_distribution.png",
        title=f"Class distribution (n={total_instances} instances)",
    )

    logger.info("Step 4/5: splitting dataset 80/10/10 (seed=%d)", seed)
    splits = split_dataset(filtered, ratios=cfg["split"]["ratios"], seed=seed)
    per_split_counts = log_per_split_stats(splits, label_field=label_field)
    save_dataset_stats(
        splits=splits,
        per_split_counts=per_split_counts,
        output_dir=FIGURES_DIR,
        label_field=label_field,
    )

    logger.info("Step 5/5: exporting YOLO format to %s", YOLO_DIR)
    # Clear any previous export so we don't mix up stale labels.
    if YOLO_DIR.exists() and cfg.get("export", {}).get("overwrite", True):
        import shutil

        shutil.rmtree(YOLO_DIR)
        logger.info("Removed previous export at %s", YOLO_DIR)

    yaml_path = export_splits_to_yolo(
        splits,
        output_dir=YOLO_DIR,
        classes=CLASSES,
        label_field=label_field,
    )
    logger.info("Done. YOLO dataset yaml at %s", yaml_path)
    return yaml_path
