"""
Memory measurement utilities.
"""

from __future__ import annotations

import gc
import os
import tracemalloc
from typing import Callable, TypeVar

import psutil

T = TypeVar("T")


def tracemalloc_peak(fn: Callable[[], T]) -> tuple[T, int]:
    """Run `fn`, returning (result, peak_traced_bytes) during its
    execution. `fn` should take no arguments (use a closure/lambda)."""
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, peak


def process_rss_delta(fn: Callable[[], T]) -> tuple[T, int]:
    """Run `fn`, returning (result, rss_delta_bytes) measured from the
    OS process's resident set size before/after. Noisier than
    tracemalloc but reflects true process memory use, including
    C-extension buffers such as bitarray's."""
    proc = psutil.Process(os.getpid())
    gc.collect()
    rss_before = proc.memory_info().rss
    result = fn()
    gc.collect()
    rss_after = proc.memory_info().rss
    return result, max(0, rss_after - rss_before)
