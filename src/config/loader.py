"""Thin wrappers around the YAML configs in ``configs/``."""
from __future__ import annotations

from typing import Any

from src.constants.paths import CONFIGS_DIR
from src.utils.io import load_yaml


def load_data_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "data.yaml")


def load_train_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "train.yaml")


def load_eval_config() -> dict[str, Any]:
    """Load ``configs/eval.yaml`` if present, otherwise an empty dict.

    Keeping this optional lets evaluation work in a fresh checkout without
    the file; missing keys fall back to Ultralytics' ``model.val()`` defaults.
    """
    eval_path = CONFIGS_DIR / "eval.yaml"
    if not eval_path.exists():
        return {}
    return load_yaml(eval_path)


def load_mischief_config() -> dict[str, Any]:
    return load_yaml(CONFIGS_DIR / "mischief.yaml")
