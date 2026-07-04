from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    """Linear-interpolation percentile on an already-sorted sequence."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = pct / 100.0 * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def bootstrap_ci(
    per_query_values: Sequence[float],
    num_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean of a per-query metric.

    Resamples queries with replacement ``num_resamples`` times and returns the
    ``(alpha/2, 1 - alpha/2)`` percentiles of the resampled means. Deterministic
    given ``seed`` so reported intervals are reproducible.
    """
    n = len(per_query_values)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(seed)
    values = list(per_query_values)
    means: List[float] = []
    for _ in range(num_resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return _percentile(means, 100.0 * alpha / 2.0), _percentile(means, 100.0 * (1.0 - alpha / 2.0))


@dataclass(frozen=True)
class RetrievalResult:
    top1: float
    top5: float
    mean_reciprocal_rank: float
    query_count: int
    # Per-query outcomes retained so callers can compute confidence intervals or
    # run significance tests without re-ranking.
    top1_flags: Tuple[float, ...] = field(default=())
    top5_flags: Tuple[float, ...] = field(default=())
    reciprocal_ranks: Tuple[float, ...] = field(default=())

    def confidence_intervals(
        self, num_resamples: int = 1000, alpha: float = 0.05, seed: int = 13
    ) -> Dict[str, Tuple[float, float]]:
        return {
            "top1": bootstrap_ci(self.top1_flags, num_resamples, alpha, seed),
            "top5": bootstrap_ci(self.top5_flags, num_resamples, alpha, seed),
            "mean_reciprocal_rank": bootstrap_ci(self.reciprocal_ranks, num_resamples, alpha, seed),
        }

    def to_dict(self, bootstrap: bool = False, num_resamples: int = 1000, alpha: float = 0.05, seed: int = 13) -> Dict[str, object]:
        result: Dict[str, object] = {
            "top1": self.top1,
            "top5": self.top5,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "query_count": self.query_count,
        }
        if bootstrap:
            cis = self.confidence_intervals(num_resamples=num_resamples, alpha=alpha, seed=seed)
            result["confidence_level"] = 1.0 - alpha
            result["bootstrap_resamples"] = num_resamples
            for metric, (low, high) in cis.items():
                result[f"{metric}_ci_low"] = low
                result[f"{metric}_ci_high"] = high
        return result


def evaluate_rankings(
    query_fact_ids: Sequence[str],
    candidate_fact_ids: Sequence[str],
    rankings: Sequence[Sequence[int]],
) -> RetrievalResult:
    if len(query_fact_ids) != len(rankings):
        raise ValueError("query_fact_ids and rankings must have the same length")

    reciprocal_ranks: List[float] = []
    top1_flags: List[float] = []
    top5_flags: List[float] = []

    for query_fact_id, ranking in zip(query_fact_ids, rankings):
        first_match_rank = None
        for rank_idx, candidate_idx in enumerate(ranking, start=1):
            if candidate_fact_ids[candidate_idx] == query_fact_id:
                first_match_rank = rank_idx
                break

        if first_match_rank is None:
            reciprocal_ranks.append(0.0)
            top1_flags.append(0.0)
            top5_flags.append(0.0)
            continue

        reciprocal_ranks.append(1.0 / first_match_rank)
        top1_flags.append(1.0 if first_match_rank == 1 else 0.0)
        top5_flags.append(1.0 if first_match_rank <= 5 else 0.0)

    query_count = len(query_fact_ids)
    if query_count == 0:
        return RetrievalResult(top1=0.0, top5=0.0, mean_reciprocal_rank=0.0, query_count=0)

    return RetrievalResult(
        top1=sum(top1_flags) / query_count,
        top5=sum(top5_flags) / query_count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / query_count,
        query_count=query_count,
        top1_flags=tuple(top1_flags),
        top5_flags=tuple(top5_flags),
        reciprocal_ranks=tuple(reciprocal_ranks),
    )


def fact_ids_by_track(rows: Iterable[Dict[str, str]], track: str | None = None) -> Set[str]:
    return {row["fact_id"] for row in rows if track is None or row["track"] == track}
