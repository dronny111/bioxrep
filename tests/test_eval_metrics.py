from __future__ import annotations

import numpy as np

from bioxrep.eval.invariance_ratio import pairwise_cosine_distances, sample_between_distances
from bioxrep.eval.retrieval import evaluate_rankings


def test_evaluate_rankings_keeps_per_query_metrics() -> None:
    result = evaluate_rankings(
        query_fact_ids=["a", "b", "c"],
        candidate_fact_ids=["b", "a", "c"],
        rankings=[[1, 0, 2], [0, 2, 1], [0, 1, 2]],
    )

    assert result.top1 == 2 / 3
    assert result.top5 == 1.0
    assert result.mean_reciprocal_rank == (1.0 + 1.0 + 1 / 3) / 3
    assert result.top1_flags == (1.0, 1.0, 0.0)
    assert result.reciprocal_ranks == (1.0, 1.0, 1 / 3)


def test_invariance_distance_helpers_are_deterministic() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    within = pairwise_cosine_distances(embeddings)
    between = sample_between_distances([embeddings[0], embeddings[1], embeddings[2]], max_pairs=2, seed=7)
    between_again = sample_between_distances([embeddings[0], embeddings[1], embeddings[2]], max_pairs=2, seed=7)

    assert within.tolist() == [1.0, 0.0, 1.0]
    assert between.tolist() == between_again.tolist()
    assert len(between) == 2
