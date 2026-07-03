from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set


@dataclass(frozen=True)
class RetrievalResult:
    top1: float
    top5: float
    mean_reciprocal_rank: float
    query_count: int

    def to_dict(self) -> Dict[str, float | int]:
        return {
            "top1": self.top1,
            "top5": self.top5,
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "query_count": self.query_count,
        }


def evaluate_rankings(
    query_fact_ids: Sequence[str],
    candidate_fact_ids: Sequence[str],
    rankings: Sequence[Sequence[int]],
) -> RetrievalResult:
    if len(query_fact_ids) != len(rankings):
        raise ValueError("query_fact_ids and rankings must have the same length")

    reciprocal_ranks: List[float] = []
    top1_hits = 0
    top5_hits = 0

    for query_fact_id, ranking in zip(query_fact_ids, rankings):
        first_match_rank = None
        for rank_idx, candidate_idx in enumerate(ranking, start=1):
            if candidate_fact_ids[candidate_idx] == query_fact_id:
                first_match_rank = rank_idx
                break

        if first_match_rank is None:
            reciprocal_ranks.append(0.0)
            continue

        reciprocal_ranks.append(1.0 / first_match_rank)
        if first_match_rank == 1:
            top1_hits += 1
        if first_match_rank <= 5:
            top5_hits += 1

    query_count = len(query_fact_ids)
    if query_count == 0:
        return RetrievalResult(top1=0.0, top5=0.0, mean_reciprocal_rank=0.0, query_count=0)

    return RetrievalResult(
        top1=top1_hits / query_count,
        top5=top5_hits / query_count,
        mean_reciprocal_rank=sum(reciprocal_ranks) / query_count,
        query_count=query_count,
    )


def fact_ids_by_track(rows: Iterable[Dict[str, str]], track: str | None = None) -> Set[str]:
    return {row["fact_id"] for row in rows if track is None or row["track"] == track}
