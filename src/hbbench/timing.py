"""
Latency measurement utilities.
"""

from __future__ import annotations

import time
from statistics import mean, median, stdev
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def time_bulk_insert(insert_fn: Callable[[str], object], items: Iterable[str]) -> float:
    """Time inserting every item in `items` one at a time. Returns total
    wall-clock seconds."""
    items = list(items)
    start = time.perf_counter()
    for item in items:
        insert_fn(item)
    return time.perf_counter() - start


def time_bulk_query(query_fn: Callable[[str], object], items: Iterable[str]) -> float:
    """Time querying every item in `items` one at a time. Returns total
    wall-clock seconds."""
    items = list(items)
    start = time.perf_counter()
    for item in items:
        query_fn(item)
    return time.perf_counter() - start


def repeated_trials(fn: Callable[[], float], n_trials: int = 10) -> dict:
    """Run a zero-arg timing function `n_trials` times and summarize.

    `fn` should return a duration in seconds (e.g. a call to
    `time_bulk_insert`/`time_bulk_query` wrapped in a lambda).
    """
    durations = [fn() for _ in range(n_trials)]
    return {
        "trials": durations,
        "mean": mean(durations),
        "median": median(durations),
        "stdev": stdev(durations) if len(durations) > 1 else 0.0,
        "min": min(durations),
        "max": max(durations),
    }
