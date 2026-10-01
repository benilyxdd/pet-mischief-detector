"""Centralised filesystem paths.

Every other module imports the directories it needs from here so that moving
the project around only requires touching this one file.
"""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Source + configs
SRC_DIR: Path = PROJECT_ROOT / "src"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"

# Dataset directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
YOLO_DIR: Path = DATA_DIR / "yolo"
EXTERNAL_DIR: Path = DATA_DIR / "external"

YOLO_DATASET_YAML: Path = YOLO_DIR / "dataset.yaml"

# We keep the actual COCO images (and FiftyOne's tightly-coupled Mongo DB)
# inside the repo so deleting the project folder removes all downloaded
# image data. Other third-party caches (Ultralytics settings, torch hub,
# matplotlib fonts, HF) are left at their default user-home locations.
FIFTYONE_DIR: Path = RAW_DIR / "fiftyone"
FIFTYONE_DATABASE_DIR: Path = FIFTYONE_DIR / "db"
FIFTYONE_ZOO_DIR: Path = FIFTYONE_DIR / "zoo"

# Training / evaluation outputs
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
RUNS_DIR: Path = OUTPUTS_DIR / "runs"
WEIGHTS_DIR: Path = OUTPUTS_DIR / "weights"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
EVAL_DIR: Path = OUTPUTS_DIR / "eval"
