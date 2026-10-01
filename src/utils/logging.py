"""Shared logger factory so every module logs with the same format."""
from __future__ import annotations

import logging


_DEFAULT_FMT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"


def get_logger(name: str = "pet-mischief-detector", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt=_DEFAULT_FMT, datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger
