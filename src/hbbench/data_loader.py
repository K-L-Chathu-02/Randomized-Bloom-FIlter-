from __future__ import annotations

import hashlib
import random
import string
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Expected filenames + the column that holds the URL, for each known
# real-world dataset. If your downloaded file has a different name or
# column, either rename it to match, or call `load_real_dataset()` with
# explicit `path=` / `url_column=` arguments.
REAL_DATASET_SPECS = {
    "phishstorm": {
        "filename": "urlset.csv",
        "url_column_candidates": [ "domain"],
        "source": "https://research.aalto.fi/en/datasets/phishstorm-phishing-legitimate-url-dataset/",
    },
    "phiusiil": {
        "filename": "PhiUSIIL_Phishing_URL_Dataset 2.csv",
        "url_column_candidates": ["URL"],
        "source": "https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset",
    },
    "mendeley_1": {
        "filename": "Phishing URLs.csv",
        "url_column_candidates": ["url"],
        "source": "https://data.mendeley.com/datasets/vfszbj9b36/1",
    },
    "mendeley_2": {
        "filename": "URL dataset.csv",
        "url_column_candidates": ["url"],
        "source": "https://data.mendeley.com/datasets/vfszbj9b36/1",
    },
}


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------

def generate_synthetic_keys(
    n: int,
    *,
    min_len: int = 8,
    max_len: int = 24,
    seed: int = 42,
) -> list[str]:
    """Generate n unique random alphanumeric string keys.

    Uses a deterministic RNG (per `seed`) so benchmark runs are
    reproducible. Uniqueness is enforced (no duplicate keys), matching
    "randomly generated string keys" in the proposal's dataset section.
    """
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    keys: set[str] = set()
    
    max_attempts = n * 20
    attempts = 0
    while len(keys) < n and attempts < max_attempts:
        length = rng.randint(min_len, max_len)
        key = "".join(rng.choices(alphabet, k=length))
        keys.add(key)
        attempts += 1
    if len(keys) < n:
        raise RuntimeError(
            f"Could not generate {n} unique keys after {max_attempts} attempts "
            f"(got {len(keys)}). Increase max_len or reduce n."
        )
    return list(keys)


def generate_disjoint_absent_keys(present_keys: list[str], n: int, *, seed: int = 1337) -> list[str]:
    """Generate n keys guaranteed disjoint from `present_keys`, for
    false-positive-rate testing (querying a disjoint absent set)."""
    present_set = set(present_keys)
    rng = random.Random(seed)
    alphabet = string.ascii_letters + string.digits
    absent: set[str] = set()
    max_attempts = n * 20
    attempts = 0
    while len(absent) < n and attempts < max_attempts:
        length = rng.randint(8, 24)
        key = "".join(rng.choices(alphabet, k=length))
        if key not in present_set:
            absent.add(key)
        attempts += 1
    if len(absent) < n:
        raise RuntimeError(f"Could not generate {n} disjoint absent keys after {max_attempts} attempts.")
    return list(absent)


# --------------------------------------------------------------------------
# Real-world data
# --------------------------------------------------------------------------

@dataclass
class RealDataset:
    name: str
    source: str
    path: Path
    urls: list[str]


def _find_url_column(df: pd.DataFrame, candidates: list[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    # Fall back to case-insensitive match
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(
        f"Could not find a URL column among {candidates} in columns {list(df.columns)}. "
        "Pass url_column= explicitly."
    )


def load_real_dataset(
    name: str,
    *,
    path: str | Path | None = None,
    url_column: str | None = None,
) -> RealDataset:
    """Load one real-world dataset by short name (see REAL_DATASET_SPECS),
    or from an explicit `path` with an explicit `url_column`."""
    if name not in REAL_DATASET_SPECS and path is None:
        raise ValueError(
            f"Unknown dataset '{name}'. Known: {list(REAL_DATASET_SPECS)}. "
            "Or pass an explicit path= and url_column=."
        )

    spec = REAL_DATASET_SPECS.get(name, {})
    resolved_path = Path(path) if path is not None else DATA_RAW_DIR / spec["filename"]

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Expected dataset file not found: {resolved_path}\n"
            f"Download it from: {spec.get('source', '(see README)')}\n"
            f"and place it at that path (or pass an explicit path=)."
        )

    try:
        df = pd.read_csv(resolved_path)
    except UnicodeDecodeError:
        # Some downloaded datasets contain legacy or mixed byte values.
        df = pd.read_csv(resolved_path, encoding="latin-1",on_bad_lines="skip")
    col = url_column or _find_url_column(df, spec.get("url_column_candidates", ["url", "URL"]))
    urls = df[col].dropna().astype(str).drop_duplicates().tolist()
    return RealDataset(name=name, source=spec.get("source", str(resolved_path)), path=resolved_path, urls=urls)


def load_all_available_real_datasets() -> dict[str, RealDataset]:
    """Load whichever of the known real datasets are actually present in
    data/raw/, skipping (and warning about) any that are missing rather
    than failing the whole run."""
    loaded = {}
    for name in REAL_DATASET_SPECS:
        try:
            loaded[name] = load_real_dataset(name)
        except FileNotFoundError as e:
            print(f"[data_loader] Skipping '{name}': {e}")
    return loaded


def build_combined_real_url_pool(target_size: int = 1_000_000, seed: int = 7) -> list[str]:
    """Combine all available real datasets into a single deduplicated
    pool of URL strings, matching the proposal's "~1 million keys" real
    dataset target. If the combined unique pool is smaller than
    `target_size`, returns everything available (a warning is printed);
    it does NOT fabricate extra "real" URLs.
    """
    datasets = load_all_available_real_datasets()
    if not datasets:
        raise FileNotFoundError(
            "No real-world datasets found in data/raw/. See README.md 'Datasets' "
            "section for expected filenames and download links."
        )
    pool: set[str] = set()
    for ds in datasets.values():
        pool.update(ds.urls)
    pool_list = list(pool)
    rng = random.Random(seed)
    rng.shuffle(pool_list)
    if len(pool_list) < target_size:
        print(
            f"[data_loader] Combined real dataset pool has {len(pool_list)} unique URLs, "
            f"fewer than the requested target_size={target_size}. Using all available."
        )
        return pool_list
    return pool_list[:target_size]


def stable_hash_split(items: list[str], test_fraction: float = 0.2, seed: int = 0) -> tuple[list[str], list[str]]:
    """Deterministically split `items` into (train/insert, test/query)
    sets using a stable hash of each item, so the split is reproducible
    without needing to shuffle the whole list."""
    threshold = int(test_fraction * (2**32))
    train, test = [], []
    for item in items:
        h = int(hashlib.md5((str(seed) + item).encode("utf-8")).hexdigest()[:8], 16)
        (test if h < threshold else train).append(item)
    return train, test
