#!/usr/bin/env python3
"""
    uv run scripts/run_benchmarks.py                     # default (synthetic, quick)
    uv run scripts/run_benchmarks.py --mode synthetic --full
    uv run scripts/run_benchmarks.py --mode real
    uv run scripts/run_benchmarks.py --mode both --full

See `--help` for all options.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbbench import benchmark_fpr, benchmark_latency, benchmark_memory, param_sensitivity, visualize  
from hbbench.data_loader import build_combined_real_url_pool, generate_synthetic_keys  

TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def synthetic_key_generator(n: int) -> list[str]:
    return generate_synthetic_keys(n, seed=42)


def make_real_key_generator(pool: list[str]):
    def _gen(n: int) -> list[str]:
        if n > len(pool):
            raise ValueError(f"Requested {n} keys but real pool only has {len(pool)}.")
        return pool[:n]

    return _gen


def run_suite(label: str, key_generator, sizes: list[int], fpr_sizes: list[int], param_n: int) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{label}] Memory footprint benchmark ===")
    mem_df = benchmark_memory.run_memory_benchmark(sizes, key_generator)
    mem_df.to_csv(TABLES_DIR / f"memory_{label}.csv", index=False)
    print(mem_df)
    visualize.plot_memory_footprint(mem_df, filename=f"memory_footprint_{label}.png")

    print(f"\n=== [{label}] Latency benchmark ===")
    lat_df = benchmark_latency.run_latency_benchmark(sizes, key_generator)
    lat_df.to_csv(TABLES_DIR / f"latency_{label}.csv", index=False)
    print(lat_df)
    for op in ("insert", "query_present", "query_absent"):
        visualize.plot_latency(lat_df, op, filename=f"latency_{op}_{label}.png")

    print(f"\n=== [{label}] False-positive rate benchmark ===")
    m = 8 * max(fpr_sizes)  # ~8 bits/element, a common default
    k = 6
    fpr_df = benchmark_fpr.run_fpr_benchmark(m, k, fpr_sizes, key_generator)
    fpr_df.to_csv(TABLES_DIR / f"fpr_{label}.csv", index=False)
    print(fpr_df)
    visualize.plot_fpr_empirical_vs_theoretical(fpr_df, filename=f"fpr_{label}.png")

    print(f"\n=== [{label}] Parameter sensitivity sweep ===")
    m_values = [int(param_n * bpe) for bpe in (4, 8, 12, 16, 20)]
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    sweep_df = param_sensitivity.run_param_sensitivity(param_n, key_generator, m_values=m_values, k_values=k_values)
    sweep_df.to_csv(TABLES_DIR / f"param_sensitivity_{label}.csv", index=False)
    best_df = param_sensitivity.find_best_k_per_m(sweep_df)
    best_df.to_csv(TABLES_DIR / f"param_sensitivity_best_{label}.csv", index=False)
    print(best_df)
    visualize.plot_param_sensitivity_heatmap(sweep_df, filename=f"param_sensitivity_heatmap_{label}.png")
    visualize.plot_param_sensitivity_vs_k(sweep_df, filename=f"param_sensitivity_vs_k_{label}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Hash Set vs Bloom Filter benchmark suite.")
    parser.add_argument("--mode", choices=["synthetic", "real", "both"], default="synthetic")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full proposal-scale sizes (10^3 .. 10^6). Slow — several minutes. "
        "Without this flag, a much faster --quick-equivalent size sweep runs instead.",
    )
    args = parser.parse_args()

    if args.full:
        sizes = [1_000, 10_000, 500_000, 800_000]
        fpr_sizes = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000]
        param_n = 100_000
    else:
        sizes = [1_000, 5_000, 20_000, 50_000]
        fpr_sizes = [1_000, 5_000, 10_000, 25_000, 50_000]
        param_n = 20_000

    if args.mode in ("synthetic", "both"):
        run_suite("synthetic", synthetic_key_generator, sizes, fpr_sizes, param_n)

    if args.mode in ("real", "both"):
        target = max(sizes) if not args.full else 1_000_000
        pool = build_combined_real_url_pool(target_size=target)
        real_sizes = [s for s in sizes if s <= len(pool)]
        real_fpr_sizes = [s for s in fpr_sizes if s <= len(pool)]
        real_param_n = min(param_n, len(pool))
        if not real_sizes:
            print(f"[run_benchmarks] Real pool only has {len(pool)} URLs — too small for requested sizes.")
        else:
            run_suite("real", make_real_key_generator(pool), real_sizes, real_fpr_sizes, real_param_n)

    print("\nDone. Tables -> results/tables/, Plots -> results/plots/")


if __name__ == "__main__":
    main()
