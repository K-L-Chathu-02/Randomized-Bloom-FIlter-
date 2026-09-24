#!/usr/bin/env python3
"""
Isolates *why* BloomFilter (V2) queries are slower than HashSet (V1)
queries, rather than just re-measuring that they are.

    uv run scripts/run_overhead_experiment.py                  # synthetic, n=20000
    uv run scripts/run_overhead_experiment.py --mode real
    uv run scripts/run_overhead_experiment.py --n 100000 --k-values 1 2 4 6 8 10 12

Writes tables to results/tables/overhead_*.csv and plots to
results/plots/overhead_*.png, alongside the existing benchmark suite's
output, so they can be cited in the report the same way.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbbench import overhead_analysis as oa
from hbbench.bloomfilter import BloomFilter
from hbbench.data_loader import build_combined_real_url_pool, generate_synthetic_keys
from hbbench.hashset import HashSet

TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--n", type=int, default=20_000, help="number of keys to insert")
    parser.add_argument("--n-query", type=int, default=2_000, help="number of queries per timing")
    parser.add_argument("--n-trials", type=int, default=7, help="repeated timing trials")
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 10, 12],
        help="k values to sweep for the k-scaling regression",
    )
    parser.add_argument("--target-fpr", type=float, default=0.01)
    args = parser.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "synthetic":
        keys = generate_synthetic_keys(args.n, seed=42)
    else:
        pool = build_combined_real_url_pool(target_size=args.n)
        if len(pool) < args.n:
            print(f"[overhead] Real pool only has {len(pool)} URLs; using that instead of {args.n}.")
        keys = pool[: args.n]

    query_sample = keys[: min(args.n_query, len(keys))]

    hs = HashSet()
    hs.insert_many(keys)
    bf = BloomFilter.from_target_fpr(n_expected=len(keys), target_fpr=args.target_fpr)
    bf.insert_many(keys)
    print(f"[overhead] n={len(keys)}  BloomFilter m={bf.m}  k={bf.k}  (target_fpr={args.target_fpr})")

    # 1. cProfile: where does the time actually go, function by function?
    print("\n=== [1/5] cProfile breakdown (HashSet vs BloomFilter query) ===")
    profile_df = oa.compare_profiles(hs, bf, query_sample[: min(500, len(query_sample))])
    profile_df.to_csv(TABLES_DIR / f"overhead_profile_{args.mode}.csv", index=False)
    print(profile_df[["structure", "function", "n_calls", "tottime_per_call_us", "cumtime_s"]])

    # 2. Is it the hashing itself (a C extension) that costs more?
    print("\n=== [2/5] Hash-call isolation (1x vs 2x mmh3.hash64) ===")
    hash_df = oa.hash_component_benchmark(query_sample, n_trials=args.n_trials)
    hash_df.to_csv(TABLES_DIR / f"overhead_hash_component_{args.mode}.csv", index=False)
    print(hash_df[["component", "mean_us_per_call", "stdev_us_per_call"]])

    # 3. Does cost scale linearly in k, at a slope matching a bare Python loop?
    print("\n=== [3/5] k-scaling regression ===")
    k_df = oa.k_scaling_benchmark(
        m=bf.m, k_values=args.k_values, keys=keys, query_sample=query_sample, n_trials=args.n_trials
    )
    k_df.to_csv(TABLES_DIR / f"overhead_k_scaling_{args.mode}.csv", index=False)
    print(k_df[["k", "mean_us_per_call", "stdev_us_per_call"]])
    print(
        f"fit: {k_df.attrs['fit_slope_us_per_k']:.4f} us/k + {k_df.attrs['fit_intercept_us']:.4f} us "
        f"| bare for-loop iteration: {k_df.attrs['noop_loop_us_per_iteration']:.4f} us"
    )
    oa.plot_k_scaling(k_df, filename=f"overhead_k_scaling_{args.mode}.png")

    # 4. Ablation: remove the Python-level loop (two ways) and measure recovery.
    print("\n=== [4/5] Loop-removal ablation ===")
    ablation_df = oa.full_query_ablation(
        m=bf.m, k=bf.k, keys=keys, query_sample=query_sample, n_trials=args.n_trials
    )
    ablation_df.to_csv(TABLES_DIR / f"overhead_ablation_{args.mode}.csv", index=False)
    print(ablation_df[["variant", "mean_us_per_call", "stdev_us_per_call"]])
    print(
        f"gap over HashSet: {ablation_df.attrs['original_gap_over_hashset_us']:.4f} us | "
        f"recovered by vectorizing: {ablation_df.attrs['pct_gap_recovered_by_vectorizing']:.1f}% | "
        f"recovered by fusing: {ablation_df.attrs['pct_gap_recovered_by_fusing']:.1f}%"
    )
    oa.plot_ablation_bar(ablation_df, filename=f"overhead_ablation_{args.mode}.png")

    # 5. Static, timing-independent corroboration: bytecode instruction counts.
    print("\n=== [5/5] Bytecode instruction counts ===")
    bytecode_df = oa.bytecode_instruction_counts()
    bytecode_df.to_csv(TABLES_DIR / "overhead_bytecode_counts.csv", index=False)
    print(bytecode_df)

    print("\nDone. Tables -> results/tables/overhead_*.csv, Plots -> results/plots/overhead_*.png")


if __name__ == "__main__":
    main()
