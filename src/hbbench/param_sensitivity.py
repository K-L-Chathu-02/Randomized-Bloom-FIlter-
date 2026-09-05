from __future__ import annotations

import pandas as pd

from .bloomfilter import BloomFilter
from .data_loader import generate_disjoint_absent_keys


def run_param_sensitivity(
    n: int,
    key_generator,
    *,
    m_values: list[int],
    k_values: list[int],
    n_absent_queries: int = 10_000,
    seed: int = 99,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    n:Fixed number of elements to insert for every (m, k) pair.
    key_generator:
        Callable(n) -> list[str] of n unique keys.
    m_values, k_values:
        Grids to sweep (full cross-product is tested).
    """
    keys = key_generator(n)
    absent_keys = generate_disjoint_absent_keys(keys, n_absent_queries, seed=seed)

    rows = []
    for m in m_values:
        for k in k_values:
            bf = BloomFilter(m=m, k=k)
            bf.insert_many(keys)

            false_positives = sum(bf.query(key) for key in absent_keys)
            empirical_fpr = false_positives / len(absent_keys)
            theoretical_fpr = bf.theoretical_fpr(n_inserted=n)
            k_star = BloomFilter.optimal_k(n, m)

            rows.append(
                {
                    "n": n,
                    "m": m,
                    "k": k,
                    "k_optimal_analytic": k_star,
                    "bits_per_element": m / n,
                    "fill_ratio": bf.fill_ratio,
                    "empirical_fpr": empirical_fpr,
                    "theoretical_fpr": theoretical_fpr,
                    "size_bytes": bf.size_in_bytes(),
                }
            )

    return pd.DataFrame(rows)


def find_best_k_per_m(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """From a param-sensitivity sweep, pick the empirically best k
    (lowest empirical_fpr) for each m, and compare it to the analytic
    optimum."""
    idx = sweep_df.groupby("m")["empirical_fpr"].idxmin()
    best = sweep_df.loc[idx].sort_values("m").reset_index(drop=True)
    best["k_matches_analytic"] = best["k"] == best["k_optimal_analytic"]
    return best
