# Deterministic Hash Sets vs. Randomized Bloom Filters for Set-Membership Testing

CS4523 project — **Team RandomiX**
Chathurangi K.L. (220087R) · Navodi S.Y.A.C (220419N)

## Setup

### Option A — `uv` 

```bash
# Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

cd hashset-bloom-project
uv sync                 # creates .venv and installs all dependencies
```

### Option B — plain `pip`

```bash
cd hashset-bloom-project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # so `import hbbench` works from anywhere
```

### Apple Silicon (macOS) note

`bitarray` and `mmh3` both ship prebuilt `manylinux`/`macosx-arm64`
wheels on PyPI for recent Python versions, so `uv sync` / `pip install`
should **not** need to compile anything locally. If you hit a build
error anyway (e.g. from an unusually old Python or a corporate proxy
blocking wheel downloads), install the Xcode Command Line Tools first
(`xcode-select --install`) — that gives `pip` a C compiler to fall back
to a source build.

## Running the benchmarks

```bash
# Quick run (~1-2 min): smaller sizes, synthetic data only. Good for a sanity check.
uv run scripts/run_benchmarks.py

# Full proposal-scale run (10^3 .. 10^6 elements) — several minutes, synthetic data.
uv run scripts/run_benchmarks.py --full

# Real-world data (requires files in data/raw/ — see above)
uv run scripts/run_benchmarks.py --mode real
uv run scripts/run_benchmarks.py --mode both --full   # synthetic AND real, full scale

# Without uv:
python3 scripts/run_benchmarks.py --full
```

Each run writes:

- `results/tables/*.csv` — raw numbers for every benchmark axis
- `results/plots/*.png` — the corresponding comparative plots
