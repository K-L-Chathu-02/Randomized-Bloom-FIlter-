from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")

PLOTS_DIR = Path(__file__).resolve().parents[2] / "results" / "plots"


def _save(fig, filename: str) -> Path:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_memory_footprint(df: pd.DataFrame, filename: str = "memory_footprint.png", metric: str = "primary_bytes"):
    """Log-log plot of memory vs n for both structures.

    Defaults to `primary_bytes` (HashSet: tracemalloc peak; BloomFilter:
    exact m/8 closed form) since raw process RSS delta is dominated by
    OS page-granularity noise at small n. Pass metric="process_rss_delta_bytes"
    for the real-process-memory view instead.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    for structure, group in df.groupby("structure"):
        ax.plot(group["n"], group[metric], marker="o", label=structure)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of elements (n)")
    label = "Process RSS delta" if metric == "process_rss_delta_bytes" else "Memory footprint"
    ax.set_ylabel(f"{label} (bytes, log scale)")
    ax.set_title("Memory Footprint: HashSet (V1) vs BloomFilter (V2)")
    ax.legend()
    _save(fig, filename)
    return fig


def plot_latency(df: pd.DataFrame, operation: str, filename: str | None = None):
    """Log-log plot of mean per-op latency (microseconds) vs n, for one
    operation ('insert', 'query_present', or 'query_absent')."""
    sub = df[df["operation"] == operation]
    fig, ax = plt.subplots(figsize=(9, 6))
    for structure, group in sub.groupby("structure"):
        ax.errorbar(
            group["n"],
            group["mean_us_per_op"],
            yerr=(group["stdev_s"] / group["n_ops"]) * 1e6,
            marker="o",
            capsize=3,
            label=structure,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Number of elements (n)")
    ax.set_ylabel("Mean latency per operation (µs)")
    ax.set_title(f"Latency: {operation.replace('_', ' ').title()}")
    ax.legend()
    _save(fig, filename or f"latency_{operation}.png")
    return fig


def plot_fpr_empirical_vs_theoretical(df: pd.DataFrame, filename: str = "fpr_empirical_vs_theoretical.png"):
    """Empirical false-positive rate vs the theoretical curve, as n grows."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(df["n"], df["empirical_fpr"], marker="o", label="Empirical FPR")
    ax.plot(df["n"], df["theoretical_fpr"], linestyle="--", marker="x", label="Theoretical FPR")
    ax.set_xlabel("Number of elements inserted (n)")
    ax.set_ylabel("False-positive rate")
    ax.set_title("Bloom Filter: Empirical vs Theoretical False-Positive Rate")
    ax.legend()
    _save(fig, filename)
    return fig


def plot_param_sensitivity_heatmap(df: pd.DataFrame, filename: str = "param_sensitivity_heatmap.png"):
    """Heatmap of empirical FPR across the (m, k) grid."""
    pivot = df.pivot(index="k", columns="m", values="empirical_fpr")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis_r", ax=ax, cbar_kws={"label": "Empirical FPR"})
    ax.set_title("Parameter Sensitivity: Empirical FPR over (m, k)")
    ax.set_xlabel("m (bit array size)")
    ax.set_ylabel("k (number of hash functions)")
    _save(fig, filename)
    return fig


def plot_param_sensitivity_vs_k(df: pd.DataFrame, filename: str = "param_sensitivity_vs_k.png"):
    """For each m, empirical FPR vs k, with the analytic optimum k
    marked — shows the characteristic U-shaped curve."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for m, group in df.groupby("m"):
        group = group.sort_values("k")
        line, = ax.plot(group["k"], group["empirical_fpr"], marker="o", label=f"m={m}")
        k_star = group["k_optimal_analytic"].iloc[0]
        ax.axvline(k_star, color=line.get_color(), linestyle=":", alpha=0.6)
    ax.set_xlabel("k (number of hash functions)")
    ax.set_ylabel("Empirical false-positive rate")
    ax.set_title("FPR vs k (dotted lines = analytic optimum k*)")
    ax.legend()
    _save(fig, filename)
    return fig
