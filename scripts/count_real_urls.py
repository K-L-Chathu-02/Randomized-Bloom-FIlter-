#!/usr/bin/env python3
"""Print the total number of unique URLs in all available real datasets."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hbbench.data_loader import load_all_available_real_datasets


def main() -> None:
    datasets = load_all_available_real_datasets()
    if not datasets:
        raise FileNotFoundError("No real-world datasets found in data/raw/.")

    unique_urls: set[str] = set()
    for dataset in datasets.values():
        unique_urls.update(dataset.urls)

    print(f"Deduplicated URLs across {len(datasets)} dataset(s): {len(unique_urls)}")


if __name__ == "__main__":
    main()
