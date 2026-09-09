from __future__ import annotations

import pandas as pd

from .bloomfilter import BloomFilter
from .data_loader import generate_disjoint_absent_keys


def run_fpr_benchmark(
    m: int,
    k: int,
    sizes: list[int],
    key_generator,
    *,
    n_absent_queries: int = 20_000,
    seed: int = 1337,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    m, k:
        Fixed Bloom filter parameters (kept constant across `sizes` so
        the fill ratio / FPR curve as n grows is visible).
    sizes:
        Increasing element counts to insert (cumulative — each size in
        `sizes` should be the total inserted so far, e.g.
        [1000, 5000, 10000, ...]).
    key_generator:
        Callable(n) -> list[str] of n unique keys.
    n_absent_queries:
        Size of the disjoint absent-key set queried at each checkpoint.
    """
    bf = BloomFilter(m=m, k=k)
    all_keys = key_generator(max(sizes))
    absent_keys = generate_disjoint_absent_keys(all_keys, n_absent_queries, seed=seed)

    rows = []
    inserted_so_far = 0
    for n in sorted(sizes):
        batch = all_keys[inserted_so_far:n]
        for key in batch:
            bf.insert(key)
        inserted_so_far = n

        # Zero-false-negative check: every inserted key must still query True.
        present_check = all(bf.query(key) for key in all_keys[:n])

        false_positives = sum(bf.query(key) for key in absent_keys)
        empirical_fpr = false_positives / len(absent_keys)
        theoretical_fpr = bf.theoretical_fpr(n_inserted=n)

        rows.append(
            {
                "n": n,
                "m": m,
                "k": k,
                "fill_ratio": bf.fill_ratio,
                "empirical_fpr": empirical_fpr,
                "theoretical_fpr": theoretical_fpr,
                "abs_error": abs(empirical_fpr - theoretical_fpr),
                "zero_false_negatives_holds": present_check,
            }
        )

    return pd.DataFrame(rows)
