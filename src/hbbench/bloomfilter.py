from __future__ import annotations

import math
from dataclasses import dataclass, field

from bitarray import bitarray

from .hashing import derive_positions


@dataclass
class BloomFilter:
    """
    Parameters
    m:Size of the underlying bit array, in bits.
    k:Number of hash functions (derived positions) per element.
    """

    m: int
    k: int

    _bits: bitarray = field(init=False, repr=False)
    _count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be positive")
        if self.k <= 0:
            raise ValueError("k must be positive")
        self._bits = bitarray(self.m)
        self._bits.setall(0)
        self._count = 0

    # -- core operations -------------------------------------------------

    def insert(self, item: str) -> None:
        for pos in derive_positions(item, self.k, self.m):
            self._bits[pos] = 1
        self._count += 1

    def query(self, item: str) -> bool:
        """Returns False => definitely absent. Returns True => possibly
        present (may be a false positive)."""
        return all(self._bits[pos] for pos in derive_positions(item, self.k, self.m))

    def insert_many(self, items) -> None:
        for item in items:
            self.insert(item)

    def query_many(self, items) -> list[bool]:
        return [self.query(item) for item in items]

    # -- introspection -------------------------------------------------

    def __len__(self) -> int:
        """Number of insert() calls made (NOT unique-element count — a
        Bloom filter cannot distinguish a repeated insert from a new
        one without extra bookkeeping)."""
        return self._count

    def __contains__(self, item: str) -> bool:
        return self.query(item)

    @property
    def fill_ratio(self) -> float:
        """Fraction of bits currently set to 1."""
        return self._bits.count(1) / self.m

    def size_in_bytes(self) -> int:
        """Exact memory footprint of the bit array itself (m/8 bytes,
        rounded up), independent of how many elements were inserted or
        how large those elements are."""
        return len(self._bits.tobytes())

    def theoretical_fpr(self, n_inserted: int | None = None) -> float:
        """
        Theoretical false-positive probability after inserting n elements:

            p ≈ (1 - e^(-kn/m))^k

        If n_inserted is omitted, uses the number of insert() calls made
        so far.
        """
        n = self._count if n_inserted is None else n_inserted
        if n <= 0:
            return 0.0
        return (1 - math.exp(-self.k * n / self.m)) ** self.k

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def optimal_m(n: int, target_fpr: float) -> int:
        """
        Optimal bit-array size for n expected elements and a target
        false-positive rate p:

            m = - (n * ln p) / (ln 2)^2
        """
        if not (0 < target_fpr < 1):
            raise ValueError("target_fpr must be in (0, 1)")
        if n <= 0:
            raise ValueError("n must be positive")
        m = -(n * math.log(target_fpr)) / (math.log(2) ** 2)
        return max(1, math.ceil(m))

    @staticmethod
    def optimal_k(n: int, m: int) -> int:
        """
        Optimal number of hash functions for n expected elements and an
        m-bit array:

            k = (m / n) * ln 2
        """
        if n <= 0:
            raise ValueError("n must be positive")
        k = (m / n) * math.log(2)
        return max(1, round(k))

    @classmethod
    def from_target_fpr(cls, n_expected: int, target_fpr: float) -> "BloomFilter":
        """Construct a BloomFilter sized optimally for `n_expected`
        elements and a `target_fpr` false-positive rate."""
        m = cls.optimal_m(n_expected, target_fpr)
        k = cls.optimal_k(n_expected, m)
        return cls(m=m, k=k)
