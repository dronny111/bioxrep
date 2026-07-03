from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from bioxrep.baselines.char_ngram_retrieval import cosine, fit_idf, transform
from bioxrep.data.io import read_jsonl


def reciprocal_rank(labels: Sequence[bool], ranking: Sequence[int]) -> float:
    for rank_idx, candidate_idx in enumerate(ranking, start=1):
        if labels[candidate_idx]:
            return 1.0 / rank_idx
    return 0.0


def evaluate_hard_records(records: Sequence[Dict[str, Any]]) -> Dict[str, float | int]:
    if not records:
        raise ValueError("No hard retrieval records were provided")

    texts: List[str] = []
    for record in records:
        texts.append(str(record["query"]["text"]))
        texts.extend(str(candidate["text"]) for candidate in record["candidates"])

    idf = fit_idf(texts)
    top1_hits = 0
    top5_hits = 0
    reciprocal_ranks: List[float] = []
    candidate_counts: List[int] = []

    for record in records:
        query_vector = transform(str(record["query"]["text"]), idf)
        candidate_vectors = [transform(str(candidate["text"]), idf) for candidate in record["candidates"]]
        scores = [(idx, cosine(query_vector, vector)) for idx, vector in enumerate(candidate_vectors)]
        scores.sort(key=lambda item: item[1], reverse=True)
        ranking = [idx for idx, _ in scores]
        labels = [bool(candidate.get("is_positive")) for candidate in record["candidates"]]

        rr = reciprocal_rank(labels, ranking)
        reciprocal_ranks.append(rr)
        if ranking and labels[ranking[0]]:
            top1_hits += 1
        if any(labels[idx] for idx in ranking[:5]):
            top5_hits += 1
        candidate_counts.append(len(record["candidates"]))

    query_count = len(records)
    return {
        "top1": top1_hits / query_count,
        "top5": top5_hits / query_count,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / query_count,
        "query_count": query_count,
        "avg_candidates": sum(candidate_counts) / query_count,
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run char n-gram retrieval on BioXRep hard candidate sets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    metrics = evaluate_hard_records(records)
    metrics["input"] = str(args.input)
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
