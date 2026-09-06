"""
Memory Footprint benchmark.
"""

from __future__ import annotations

import pandas as pd

from .bloomfilter import BloomFilter
from .hashset import HashSet
from .memory_utils import process_rss_delta, tracemalloc_peak


def run_memory_benchmark(
    sizes: list[int],
    key_generator,
    *,
    target_fpr: float = 0.01,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    sizes:
        List of n values to test (e.g. [1_000, 10_000, ..., 1_000_000]).
    key_generator:
        Callable(n) -> list[str] of n unique keys (e.g.
        `data_loader.generate_synthetic_keys`, or a slice of a real
        dataset's URL list).
    target_fpr:
        False-positive rate the Bloom filter is sized for at each n via
        `BloomFilter.from_target_fpr`.
    """
    rows = []
    for n in sizes:
        keys = key_generator(n)

        # --- V1: HashSet ---
        hs = HashSet()

        def build_hashset():
            for item in keys:
                hs.insert(item)
            return hs

        _, hs_tracemalloc_peak = tracemalloc_peak(build_hashset)
        # Rebuild fresh for the RSS measurement (tracemalloc run already
        # populated `hs`; use a new instance for a clean delta).
        hs2 = HashSet()

        def build_hashset2():
            for item in keys:
                hs2.insert(item)
            return hs2

        _, hs_rss_delta = process_rss_delta(build_hashset2)

        rows.append(
            {
                "structure": "HashSet (V1)",
                "n": n,
                "tracemalloc_peak_bytes": hs_tracemalloc_peak,
                "process_rss_delta_bytes": hs_rss_delta,
                "exact_bytes": None,  # no closed form; depends on string sizes
                "primary_bytes": hs_tracemalloc_peak,
                "m": None,
                "k": None,
            }
        )

        # --- V2: BloomFilter ---
        bf = BloomFilter.from_target_fpr(n_expected=n, target_fpr=target_fpr)

        def build_bloom():
            for item in keys:
                bf.insert(item)
            return bf

        _, bf_tracemalloc_peak = tracemalloc_peak(build_bloom)
        bf2 = BloomFilter(m=bf.m, k=bf.k)

        def build_bloom2():
            for item in keys:
                bf2.insert(item)
            return bf2

        _, bf_rss_delta = process_rss_delta(build_bloom2)
        exact_bytes = bf.size_in_bytes()

        rows.append(
            {
                "structure": "BloomFilter (V2)",
                "n": n,
                "tracemalloc_peak_bytes": bf_tracemalloc_peak,
                "process_rss_delta_bytes": bf_rss_delta,
                "exact_bytes": exact_bytes,  # closed form: m/8
                "primary_bytes": exact_bytes,
                "m": bf.m,
                "k": bf.k,
            }
        )

    return pd.DataFrame(rows)
