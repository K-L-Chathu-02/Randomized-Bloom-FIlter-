from __future__ import annotations

import mmh3


def base_hashes(item: str) -> tuple[int, int]:
    """
    Compute the two base 64-bit hashes used to derive all k positions.
    We use two different seeds of MurmurHash3 (x64, 128-bit variant,
    folded to 64 bits) so h1 and h2 are effectively independent.
    """
    h1 = mmh3.hash64(item, seed=0, signed=False)[0]
    h2 = mmh3.hash64(item, seed=0x9747b28c, signed=False)[0]
    # Ensure h2 is odd-ish / non-zero so double hashing doesn't degenerate
    # when combined with a power-of-two-ish m. If h2 lands on 0, nudge it.
    if h2 == 0:
        h2 = 1
    return h1, h2


def derive_positions(item: str, k: int, m: int) -> list[int]:
    """
    Derive k bit-array positions in [0, m) for `item` via double hashing.
    """
    h1, h2 = base_hashes(item)
    return [(h1 + i * h2) % m for i in range(k)]


def single_hash_index(item: str, n_buckets: int) -> int:
    """Single hash function used by the deterministic hash set's bucket
    array (separate from the Bloom filter's double hashing)."""
    return mmh3.hash64(item, seed=0, signed=False)[0] % n_buckets
