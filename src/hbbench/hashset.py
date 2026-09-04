"""
V1 — Deterministic Hash Set 
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hashing import single_hash_index


@dataclass
class HashSet:
    """A separate-chaining hash set with dynamic resizing.

    Parameters
    initial_buckets:
        Starting number of buckets. Grows automatically to keep the
        load factor bounded.
    max_load_factor:
        Average chain length at which the table doubles in size.
    """

    initial_buckets: int = 16
    max_load_factor: float = 0.75

    _buckets: list[list[str]] = field(init=False, repr=False)
    _n_buckets: int = field(init=False, repr=False)
    _count: int = field(init=False, default=0)
    resize_events: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._n_buckets = max(1, self.initial_buckets)
        self._buckets = [[] for _ in range(self._n_buckets)]
        self._count = 0


    def insert(self, item: str) -> bool:
        """Insert `item`. Returns True if newly inserted, False if it was
        already present (idempotent, like a mathematical set)."""
        idx = single_hash_index(item, self._n_buckets)
        bucket = self._buckets[idx]
        if item in bucket:
            return False
        bucket.append(item)
        self._count += 1
        if self._count / self._n_buckets > self.max_load_factor:
            self._resize()
        return True

    def query(self, item: str) -> bool:
        """Exact membership test. Always correct — no false positives or
        negatives, by construction."""
        idx = single_hash_index(item, self._n_buckets)
        return item in self._buckets[idx]

    def delete(self, item: str) -> bool:
        """Remove `item` if present. Returns True if it was removed."""
        idx = single_hash_index(item, self._n_buckets)
        bucket = self._buckets[idx]
        if item in bucket:
            bucket.remove(item)
            self._count -= 1
            return True
        return False


    def insert_many(self, items) -> None:
        for item in items:
            self.insert(item)

    def query_many(self, items) -> list[bool]:
        return [self.query(item) for item in items]


    def __len__(self) -> int:
        return self._count

    def __contains__(self, item: str) -> bool:
        return self.query(item)

    @property
    def n_buckets(self) -> int:
        return self._n_buckets

    @property
    def load_factor(self) -> float:
        return self._count / self._n_buckets

    def max_chain_length(self) -> int:
        return max((len(b) for b in self._buckets), default=0)

    def _resize(self) -> None:
        """Double the bucket array and rehash every element."""
        old_buckets = self._buckets
        self._n_buckets *= 2
        self._buckets = [[] for _ in range(self._n_buckets)]
        for bucket in old_buckets:
            for item in bucket:
                idx = single_hash_index(item, self._n_buckets)
                self._buckets[idx].append(item)
        self.resize_events += 1


class BuiltinHashSet:
    """Thin wrapper around Python's built-in `set`, exposing the same
    insert/query interface as `HashSet` and `BloomFilter`.

    Included as a reference point: CPython's `set` is a highly optimized
    open-addressing hash table implemented in C. Comparing against it
    shows the overhead of the pure-Python `HashSet` above versus the
    practical "just use `set()`" baseline, while `HashSet` stays useful
    as the structure whose internals (buckets, chaining, load factor)
    can actually be inspected and instrumented.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: set[str] = set()

    def insert(self, item: str) -> bool:
        if item in self._data:
            return False
        self._data.add(item)
        return True

    def query(self, item: str) -> bool:
        return item in self._data

    def delete(self, item: str) -> bool:
        if item in self._data:
            self._data.discard(item)
            return True
        return False

    def insert_many(self, items) -> None:
        self._data.update(items)

    def query_many(self, items) -> list[bool]:
        return [item in self._data for item in items]

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: str) -> bool:
        return item in self._data
