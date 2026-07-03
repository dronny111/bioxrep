from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from bioxrep.data.io import flatten_equivalence_classes, read_jsonl
from bioxrep.eval.retrieval import evaluate_rankings


def load_rows(input_path: Path) -> List[Dict[str, object]]:
    rows = read_jsonl(input_path)
    if rows and "forms" in rows[0]:
        return flatten_equivalence_classes(rows)
    return rows


def canonical_key(row: Dict[str, object], key: str) -> str:
    if key == "fact_id":
        return str(row["fact_id"])

    attributes = row.get("attributes", {})
    if not isinstance(attributes, dict):
        raise ValueError("Expected row['attributes'] to be a dictionary")

    if key not in attributes:
        raise ValueError(f"Canonical key '{key}' not present in row attributes")
    return str(attributes[key])


def filter_rows(
    rows: Sequence[Dict[str, object]],
    track: str | None,
    notation: str | None,
    split: str | None,
) -> List[Dict[str, object]]:
    return [
        row
        for row in rows
        if (track is None or row["track"] == track)
        and (notation is None or row["notation"] == notation)
        and (split is None or row.get("split") == split)
    ]


def rank_by_canonical_key(
    queries: Sequence[Dict[str, object]],
    candidates: Sequence[Dict[str, object]],
    key: str,
) -> List[List[int]]:
    candidate_keys = [canonical_key(candidate, key) for candidate in candidates]
    rankings: List[List[int]] = []

    for query in queries:
        query_key = canonical_key(query, key)
        exact = [idx for idx, candidate_key in enumerate(candidate_keys) if candidate_key == query_key]
        rest = [idx for idx, candidate_key in enumerate(candidate_keys) if candidate_key != query_key]
        rankings.append(exact + rest)

    return rankings


def run_retrieval(
    input_path: Path,
    canonical_key_name: str,
    relevance_key_name: str,
    query_notation: str | None,
    candidate_notation: str | None,
    track: str | None,
    query_split: str | None,
    candidate_split: str | None,
    max_queries: int | None,
    max_candidates: int | None,
) -> Dict[str, float | int | str | None]:
    rows = load_rows(input_path)
    queries = filter_rows(rows, track, query_notation, query_split)
    candidates = filter_rows(rows, track, candidate_notation, candidate_split)

    if max_queries is not None:
        queries = queries[:max_queries]
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    if not queries:
        raise ValueError("No query rows matched the requested filters")
    if not candidates:
        raise ValueError("No candidate rows matched the requested filters")

    rankings = rank_by_canonical_key(queries, candidates, canonical_key_name)
    result = evaluate_rankings(
        query_fact_ids=[canonical_key(row, relevance_key_name) for row in queries],
        candidate_fact_ids=[canonical_key(row, relevance_key_name) for row in candidates],
        rankings=rankings,
    )

    metrics = result.to_dict()
    metrics.update(
        {
            "input": str(input_path),
            "canonical_key": canonical_key_name,
            "relevance_key": relevance_key_name,
            "query_notation": query_notation,
            "candidate_notation": candidate_notation,
            "query_split": query_split,
            "candidate_split": candidate_split,
            "track": track,
            "candidate_count": len(candidates),
            "max_queries": max_queries,
            "max_candidates": max_candidates,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a canonical-field teacher retrieval baseline.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--canonical-key", default="fact_id")
    parser.add_argument("--relevance-key", default="fact_id")
    parser.add_argument("--query-notation", default=None)
    parser.add_argument("--candidate-notation", default=None)
    parser.add_argument("--query-split", default=None)
    parser.add_argument("--candidate-split", default=None)
    parser.add_argument(
        "--track",
        choices=["variant", "lab", "gene_alias", "clinvar_gene", "clinvar_hgvs", "clinvar_summary"],
        default=None,
    )
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_retrieval(
        input_path=args.input,
        canonical_key_name=args.canonical_key,
        relevance_key_name=args.relevance_key,
        query_notation=args.query_notation,
        candidate_notation=args.candidate_notation,
        track=args.track,
        query_split=args.query_split,
        candidate_split=args.candidate_split,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
