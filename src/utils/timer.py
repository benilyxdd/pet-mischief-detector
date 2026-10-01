"""Tiny context-manager timer for ad-hoc profiling."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def timer(name: str = "block") -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        print(f"[timer] {name}: {elapsed:.2f}s")
