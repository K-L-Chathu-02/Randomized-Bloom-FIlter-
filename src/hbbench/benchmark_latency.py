

from __future__ import annotations

import pandas as pd

from .bloomfilter import BloomFilter
from .data_loader import generate_disjoint_absent_keys
from .hashset import HashSet
from .timing import time_bulk_insert, time_bulk_query


def run_latency_benchmark(
    sizes: list[int],
    key_generator,
    *,
    target_fpr: float = 0.01,
    n_query_sample: int = 5_000,
    n_trials: int = 10,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    sizes:
        n values to test.
    key_generator:
        Callable(n) -> list[str] of n unique keys.
    n_query_sample:
        Number of present / absent keys to query per trial (capped to n
        for the "present" side so we never sample more present keys
        than exist).
    n_trials:
        Repeated timing trials per (structure, n, operation).
    """
    rows = []
    for n in sizes:
        keys = key_generator(n)
        present_sample = keys[: min(n_query_sample, n)]
        absent_sample = generate_disjoint_absent_keys(keys, min(n_query_sample, n))

        # ---------------- V1: HashSet ----------------
        # Rebuild fresh each trial so later trials aren't measuring
        # inserts into an already-populated (and possibly resized) table.
        insert_durations = []
        for _ in range(n_trials):
            hs = HashSet()
            insert_durations.append(time_bulk_insert(hs.insert, keys))
        insert_stats = _summarize(insert_durations)

        hs = HashSet()
        hs.insert_many(keys)
        present_query_stats = _summarize(
            [time_bulk_query(hs.query, present_sample) for _ in range(n_trials)]
        )
        absent_query_stats = _summarize(
            [time_bulk_query(hs.query, absent_sample) for _ in range(n_trials)]
        )

        rows.append(_row("HashSet (V1)", n, "insert", len(keys), insert_stats))
        rows.append(_row("HashSet (V1)", n, "query_present", len(present_sample), present_query_stats))
        rows.append(_row("HashSet (V1)", n, "query_absent", len(absent_sample), absent_query_stats))

        # ---------------- V2: BloomFilter ----------------
        insert_durations = []
        for _ in range(n_trials):
            bf = BloomFilter.from_target_fpr(n_expected=n, target_fpr=target_fpr)
            insert_durations.append(time_bulk_insert(bf.insert, keys))
        insert_stats = _summarize(insert_durations)

        bf = BloomFilter.from_target_fpr(n_expected=n, target_fpr=target_fpr)
        bf.insert_many(keys)
        present_query_stats = _summarize(
            [time_bulk_query(bf.query, present_sample) for _ in range(n_trials)]
        )
        absent_query_stats = _summarize(
            [time_bulk_query(bf.query, absent_sample) for _ in range(n_trials)]
        )

        rows.append(_row("BloomFilter (V2)", n, "insert", len(keys), insert_stats))
        rows.append(_row("BloomFilter (V2)", n, "query_present", len(present_sample), present_query_stats))
        rows.append(_row("BloomFilter (V2)", n, "query_absent", len(absent_sample), absent_query_stats))

    return pd.DataFrame(rows)


def _summarize(durations: list[float]) -> dict:
    from statistics import mean, median, stdev

    return {
        "mean_s": mean(durations),
        "median_s": median(durations),
        "stdev_s": stdev(durations) if len(durations) > 1 else 0.0,
        "min_s": min(durations),
        "max_s": max(durations),
    }


def _row(structure: str, n: int, operation: str, n_ops: int, stats: dict) -> dict:
    return {
        "structure": structure,
        "n": n,
        "operation": operation,
        "n_ops": n_ops,
        "mean_s": stats["mean_s"],
        "median_s": stats["median_s"],
        "stdev_s": stats["stdev_s"],
        "mean_us_per_op": (stats["mean_s"] / n_ops) * 1e6 if n_ops else float("nan"),
    }
