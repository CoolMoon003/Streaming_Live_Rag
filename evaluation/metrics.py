"""
Phase 5 - deterministic metrics helpers.

Kept intentionally small and dependency-free: retrieval-quality metrics
(Recall@k, MRR) plus a simple wall-clock latency tracker. No new third-party
dependencies are introduced.
"""

import time
from contextlib import contextmanager
from typing import Any


def recall_at_k(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
    k: int,
) -> float:
    """
    1.0 if any expected chunk id appears in the top-k retrieved ids, else 0.0.

    Returns 0.0 (not an error) when expected_chunk_ids is empty, since an
    eval row with no labeled ground truth cannot contribute a hit.
    """
    if not expected_chunk_ids:
        return 0.0

    top_k = set(retrieved_chunk_ids[: max(0, k)])
    expected = set(expected_chunk_ids)

    return 1.0 if top_k & expected else 0.0


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    expected_chunk_ids: list[str],
) -> float:
    """1 / rank of the first retrieved chunk that is in expected_chunk_ids, else 0.0."""
    expected = set(expected_chunk_ids)

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in expected:
            return 1.0 / rank

    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


class LatencyTracker:
    """
    Accumulates wall-clock timings for named pipeline stages across many
    measured calls, and reports the mean per stage in milliseconds.

    This is intentionally a plain dict-backed accumulator, not an
    observability system: Phase 5 only asks for a simple structured
    latency summary.
    """

    def __init__(self):
        self._totals: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    @contextmanager
    def measure(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.record(stage, elapsed_ms)

    def record(self, stage: str, elapsed_ms: float) -> None:
        self._totals[stage] = self._totals.get(stage, 0.0) + elapsed_ms
        self._counts[stage] = self._counts.get(stage, 0) + 1

    def mean_ms(self, stage: str) -> float | None:
        count = self._counts.get(stage, 0)
        if count == 0:
            return None
        return self._totals[stage] / count

    def summary(self) -> dict[str, Any]:
        return {
            stage: round(self.mean_ms(stage), 2)
            for stage in self._totals
        }