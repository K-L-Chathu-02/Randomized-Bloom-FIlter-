"""
Overhead-attribution experiments: HashSet (V1) vs BloomFilter (V2) query cost.

These experiments do NOT just re-measure "BloomFilter is slower" (the main
latency benchmark already shows that). They decompose *why*, by isolating
each candidate cause and testing it independently:

  1. cProfile breakdown       -> where does wall-clock time actually go,
                                  function by function, inside each query?
  2. Hash-call isolation      -> is it the *hashing* (mmh3, a C extension)
                                  that costs more, or something else?
  3. k-scaling regression     -> does query cost grow linearly with k, and
                                  does the *slope* match the cost of a bare
                                  CPython for-loop iteration doing nothing?
  4. Loop-removal ablation    -> if we replace the pure-Python position-
                                  generation loop with an equivalent
                                  vectorized (NumPy, C-level-looped) version
                                  and change NOTHING else, how much of the
                                  gap disappears? This is the causal test:
                                  algorithmic/hash cost is untouched, only
                                  the interpreter-loop is removed.

Each function returns a plain pandas.DataFrame so it slots into the same
`results/tables/*.csv` + `visualize.py` pipeline the rest of the suite uses.
"""

from __future__ import annotations

import cProfile
import dis
import io
import pstats
import time
from statistics import mean, stdev

import matplotlib.pyplot as plt
import mmh3
import numpy as np
import pandas as pd
import seaborn as sns

from .bloomfilter import BloomFilter
from .hashing import base_hashes, derive_positions, single_hash_index
from .hashset import HashSet
from .visualize import PLOTS_DIR, _save

sns.set_theme(style="whitegrid", context="talk")

# --------------------------------------------------------------------- #
# 0. shared micro-timing helper
# --------------------------------------------------------------------- #


def _time_calls(fn, args_list, n_trials: int = 7) -> dict:
    """Time `fn(*args)` once per element of `args_list`, repeated
    `n_trials` times over the whole pass. Returns per-call mean/stdev in
    microseconds. Mirrors the style of timing.repeated_trials so results
    are directly comparable to the rest of the suite.
    """
    n_ops = len(args_list)
    totals = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for args in args_list:
            fn(*args)
        totals.append(time.perf_counter() - start)
    per_call_us = [(t / n_ops) * 1e6 for t in totals]
    return {
        "n_ops": n_ops,
        "n_trials": n_trials,
        "mean_us_per_call": mean(per_call_us),
        "stdev_us_per_call": stdev(per_call_us) if n_trials > 1 else 0.0,
        "min_us_per_call": min(per_call_us),
    }


# --------------------------------------------------------------------- #
# 1. cProfile breakdown of a real query_many call
# --------------------------------------------------------------------- #


def profile_query_many(structure, items, label: str, top_n: int = 12) -> pd.DataFrame:
    """Profile `structure.query(item)` for every item in `items` and
    return the top_n functions by cumulative time, as a DataFrame you can
    dump straight to results/tables/.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    for item in items:
        structure.query(item)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(top_n)

    rows = []
    for func, (cc, nc, tt, ct, _callers) in stats.stats.items():
        filename, lineno, funcname = func
        rows.append(
            {
                "structure": label,
                "function": f"{funcname} ({filename.split('/')[-1]}:{lineno})",
                "n_calls": nc,
                "tottime_s": tt,
                "cumtime_s": ct,
                "tottime_per_call_us": (tt / nc) * 1e6 if nc else float("nan"),
            }
        )
    df = pd.DataFrame(rows).sort_values("cumtime_s", ascending=False).head(top_n)
    df.insert(0, "rank", range(1, len(df) + 1))
    return df.reset_index(drop=True)


def compare_profiles(hs: HashSet, bf: BloomFilter, query_sample: list[str]) -> pd.DataFrame:
    """Convenience wrapper: profile both structures on the same query
    sample and return one combined, labeled DataFrame."""
    hs_df = profile_query_many(hs, query_sample, "HashSet (V1)")
    bf_df = profile_query_many(bf, query_sample, "BloomFilter (V2)")
    return pd.concat([hs_df, bf_df], ignore_index=True)


# --------------------------------------------------------------------- #
# 2. Isolate the hashing component itself (mmh3 is a C extension, so if
#    hashing were the bottleneck, BloomFilter's *2* mmh3 calls vs
#    HashSet's *1* should roughly explain the gap on its own).
# --------------------------------------------------------------------- #


def hash_component_benchmark(items: list[str], n_trials: int = 7) -> pd.DataFrame:
    """Time the raw hash calls each structure's query() depends on, with
    no set/list/bitarray lookup attached. If hashing dominated, HashSet's
    single mmh3.hash64 call and BloomFilter's double call (base_hashes)
    would already differ by roughly the full observed query gap.
    """
    args_list = [(item,) for item in items]

    single = _time_calls(lambda item: mmh3.hash64(item, seed=0, signed=False), args_list, n_trials)
    double = _time_calls(base_hashes, args_list, n_trials)

    rows = [
        {"component": "single_hash_call (HashSet-equivalent, 1x mmh3.hash64)", **single},
        {"component": "double_hash_call (BloomFilter-equivalent, 2x mmh3.hash64)", **double},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# 3. k-scaling regression: does query cost scale linearly with k, and
#    does the slope match a bare Python for-loop's per-iteration cost?
# --------------------------------------------------------------------- #


def _noop_loop_per_iteration_us(max_k: int, n_trials: int = 9, n_reps: int = 200_000) -> float:
    """Cost of one bare `for _ in range(k): pass` iteration, i.e. the
    floor for ANY per-element Python-level loop, hashing or not."""
    durations = []
    for _ in range(n_trials):
        start = time.perf_counter()
        for _ in range(n_reps):
            for _ in range(max_k):
                pass
        durations.append(time.perf_counter() - start)
    total_iterations = n_reps * max_k
    return (mean(durations) / total_iterations) * 1e6


def k_scaling_benchmark(
    m: int,
    k_values: list[int],
    keys: list[str],
    query_sample: list[str],
    n_trials: int = 7,
) -> pd.DataFrame:
    """Hold the bit-array size m and the key set fixed; vary only k (the
    number of derived positions per query). If per-query cost grows
    ~linearly in k with a slope close to the bare-loop cost measured by
    `_noop_loop_per_iteration_us`, that is direct evidence the dominant
    cost is CPython dispatching k Python-level loop iterations, not the
    (fixed, k-independent-per-call) hashing itself.
    """
    rows = []
    for k in k_values:
        bf = BloomFilter(m=m, k=k)
        bf.insert_many(keys)
        args_list = [(item,) for item in query_sample]
        stats = _time_calls(bf.query, args_list, n_trials)
        rows.append({"k": k, **stats})

    df = pd.DataFrame(rows)

    # Least-squares fit: mean_us_per_call ~ intercept + slope * k
    slope, intercept = np.polyfit(df["k"], df["mean_us_per_call"], 1)
    df.attrs["fit_slope_us_per_k"] = slope
    df.attrs["fit_intercept_us"] = intercept
    df.attrs["noop_loop_us_per_iteration"] = _noop_loop_per_iteration_us(max(k_values))
    return df


# --------------------------------------------------------------------- #
# 4. Loop-removal ablation. Same hashes, same bit array, same semantics —
#    only the *Python-level loop* that builds the k positions is replaced
#    with a NumPy expression (a single vectorized C loop instead of k
#    CPython bytecode-dispatched iterations). If this alone closes most
#    of the HashSet/BloomFilter query gap, the interpreter loop -- not
#    the algorithm -- is the bottleneck.
# --------------------------------------------------------------------- #


def derive_positions_vectorized(item: str, k: int, m: int) -> np.ndarray:
    """Numerically identical to hashing.derive_positions, but the
    `[(h1 + i*h2) % m for i in range(k)]` Python loop is replaced with
    one NumPy call, moving the k-iteration loop from the CPython
    bytecode interpreter into NumPy's compiled C loop."""
    h1, h2 = base_hashes(item)
    i = np.arange(k, dtype=np.uint64)
    return (np.uint64(h1) + i * np.uint64(h2)) % np.uint64(m)


def query_vectorized(bf: BloomFilter, item: str) -> bool:
    """Same semantics as BloomFilter.query, using the vectorized position
    derivation above. Bit access still goes through bitarray's
    per-element __getitem__ (see caveat in the accompanying report text:
    this ablation isolates the *position-derivation* loop specifically,
    not the bit-test loop, since bitarray has no batched random-access
    read).

    CAVEAT this experiment is designed to surface: NumPy has its own
    fixed per-call dispatch/array-allocation overhead (~1-3us), which at
    small k can exceed what it saves. If this variant comes out slower
    than the original, that is a real, reportable result -- it means
    "the loop is slow" was too imprecise; the fixed per-call cost of
    *whatever* mechanism replaces the loop matters just as much as
    whether the loop is interpreted. Use it alongside query_fused below,
    which tests the same idea without introducing NumPy's own overhead.
    """
    positions = derive_positions_vectorized(item, bf.k, bf.m)
    return all(bf._bits[int(pos)] for pos in positions)


def query_fused(bf: BloomFilter, item: str) -> bool:
    """Same semantics as BloomFilter.query, but with the two separate
    Python-level loops it currently runs -- (a) the list comprehension
    inside derive_positions, and (b) the generator `all(... for pos in
    derive_positions(...))` that consumes it -- fused into a single loop,
    and the separate `derive_positions` function call inlined. Still pure
    Python, still k interpreter iterations: this isolates the cost of
    *function-call and double-loop* overhead specifically, without
    NumPy's confound, as a second, independent piece of evidence for
    where the fixed per-query overhead comes from.
    """
    h1, h2 = base_hashes(item)
    bits = bf._bits
    m = bf.m
    for i in range(bf.k):
        if not bits[(h1 + i * h2) % m]:
            return False
    return True


def positions_loop_ablation(k_values: list[int], m: int, items: list[str], n_trials: int = 7) -> pd.DataFrame:
    """Head-to-head: original (pure-Python loop) vs vectorized (NumPy)
    position derivation, at each k, with hashing held identical."""
    rows = []
    for k in k_values:
        args_list = [(item, k, m) for item in items]
        original = _time_calls(derive_positions, args_list, n_trials)
        vectorized = _time_calls(lambda item, k, m: derive_positions_vectorized(item, k, m), args_list, n_trials)
        rows.append({"k": k, "variant": "original_python_loop", **original})
        rows.append({"k": k, "variant": "vectorized_numpy", **vectorized})
    return pd.DataFrame(rows)


def full_query_ablation(
    m: int,
    k: int,
    keys: list[str],
    query_sample: list[str],
    n_trials: int = 7,
) -> pd.DataFrame:
    """End-to-end comparison at fixed (m, k): HashSet.query vs
    BloomFilter.query (original) vs query_vectorized (position-loop
    replaced). Shows how much of the HashSet/BloomFilter gap the single
    targeted change recovers.
    """
    hs = HashSet()
    hs.insert_many(keys)
    bf = BloomFilter(m=m, k=k)
    bf.insert_many(keys)

    args_list = [(item,) for item in query_sample]
    hs_stats = _time_calls(hs.query, args_list, n_trials)
    bf_stats = _time_calls(bf.query, args_list, n_trials)
    bf_vec_stats = _time_calls(lambda item: query_vectorized(bf, item), args_list, n_trials)
    bf_fused_stats = _time_calls(lambda item: query_fused(bf, item), args_list, n_trials)

    rows = [
        {"variant": "HashSet.query (V1)", **hs_stats},
        {"variant": "BloomFilter.query (V2, original)", **bf_stats},
        {"variant": "BloomFilter.query (V2, vectorized position loop)", **bf_vec_stats},
        {"variant": "BloomFilter.query (V2, fused single Python loop)", **bf_fused_stats},
    ]
    df = pd.DataFrame(rows)
    gap = df.loc[1, "mean_us_per_call"] - df.loc[0, "mean_us_per_call"]
    for name, idx in (("vectorizing", 2), ("fusing", 3)):
        recovered = df.loc[1, "mean_us_per_call"] - df.loc[idx, "mean_us_per_call"]
        df.attrs[f"gap_recovered_by_{name}_us"] = recovered
        df.attrs[f"pct_gap_recovered_by_{name}"] = (recovered / gap * 100) if gap else float("nan")
    df.attrs["original_gap_over_hashset_us"] = gap
    return df


# --------------------------------------------------------------------- #
# 5. Bytecode-level evidence: how many Python bytecode instructions does
#    each query path execute per element, independent of timing noise?
# --------------------------------------------------------------------- #


def bytecode_instruction_counts() -> pd.DataFrame:
    """Static bytecode instruction counts for the hot functions on each
    query path. Complements the wall-clock measurements with a
    hardware/timing-independent count of interpreter work per call."""
    targets = {
        "single_hash_index (HashSet query path)": single_hash_index,
        "derive_positions (BloomFilter query path, per k)": derive_positions,
        "base_hashes (BloomFilter query path)": base_hashes,
    }
    rows = []
    for name, fn in targets.items():
        n_instructions = sum(1 for _ in dis.get_instructions(fn))
        rows.append({"function": name, "bytecode_instructions": n_instructions})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# 6. Plots, matching the style/conventions of visualize.py
# --------------------------------------------------------------------- #


def plot_k_scaling(df: pd.DataFrame, filename: str = "overhead_k_scaling.png"):
    """Query cost vs k, with the fitted linear trend and the bare-loop
    floor annotated, so the slope-vs-floor comparison is visible at a
    glance."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.errorbar(
        df["k"], df["mean_us_per_call"], yerr=df["stdev_us_per_call"],
        marker="o", linestyle="none", capsize=4, label="measured BloomFilter.query",
    )
    slope = df.attrs.get("fit_slope_us_per_k")
    intercept = df.attrs.get("fit_intercept_us")
    if slope is not None:
        k_line = np.linspace(df["k"].min(), df["k"].max(), 100)
        ax.plot(k_line, intercept + slope * k_line, "--",
                label=f"linear fit ({slope:.3f} us/k + {intercept:.3f} us)")
    noop = df.attrs.get("noop_loop_us_per_iteration")
    if noop is not None:
        ax.axhline(0, color="none")  # keep autoscale sane
        ax.annotate(f"bare for-loop iteration: {noop:.4f} us",
                    xy=(0.02, 0.95), xycoords="axes fraction", fontsize=11)
    ax.set_xlabel("k (number of derived positions per query)")
    ax.set_ylabel("Query latency (us/call)")
    ax.set_title("BloomFilter query cost vs k")
    ax.legend()
    _save(fig, filename)
    return fig


def plot_ablation_bar(df: pd.DataFrame, filename: str = "overhead_ablation.png"):
    """Bar chart of the full_query_ablation() result: HashSet vs original
    BloomFilter vs the two loop-modified variants."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(df["variant"], df["mean_us_per_call"], yerr=df["stdev_us_per_call"], capsize=4)
    ax.set_ylabel("Query latency (us/call)")
    ax.set_title("Query cost by implementation variant")
    ax.set_xticks(range(len(df["variant"])))
    ax.set_xticklabels(df["variant"], rotation=20, ha="right")
    _save(fig, filename)
    return fig
