from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from bioxrep.data.io import flatten_equivalence_classes, read_jsonl
from bioxrep.eval.retrieval import evaluate_rankings


Vector = Dict[str, float]


def char_ngrams(text: str, min_n: int = 2, max_n: int = 5) -> List[str]:
    normalized = f" {text.lower()} "
    grams: List[str] = []
    for n in range(min_n, max_n + 1):
        grams.extend(normalized[i : i + n] for i in range(0, max(0, len(normalized) - n + 1)))
    return grams


def fit_idf(texts: Sequence[str]) -> Dict[str, float]:
    doc_freq: Dict[str, int] = defaultdict(int)
    for text in texts:
        for gram in set(char_ngrams(text)):
            doc_freq[gram] += 1

    doc_count = len(texts)
    return {gram: math.log((1 + doc_count) / (1 + freq)) + 1.0 for gram, freq in doc_freq.items()}


def transform(text: str, idf: Dict[str, float]) -> Vector:
    counts = Counter(char_ngrams(text))
    vector = {gram: count * idf.get(gram, 0.0) for gram, count in counts.items() if gram in idf}
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0.0:
        return vector
    return {gram: value / norm for gram, value in vector.items()}


def cosine(left: Vector, right: Vector) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(gram, 0.0) for gram, value in left.items())


def rank_candidates(query_vectors: Sequence[Vector], candidate_vectors: Sequence[Vector]) -> List[List[int]]:
    rankings: List[List[int]] = []
    for query_vector in query_vectors:
        scores = [(idx, cosine(query_vector, candidate_vector)) for idx, candidate_vector in enumerate(candidate_vectors)]
        scores.sort(key=lambda item: item[1], reverse=True)
        rankings.append([idx for idx, _ in scores])
    return rankings


def split_query_candidate_rows(
    rows: Sequence[Dict[str, object]],
    query_notation: str | None,
    candidate_notation: str | None,
    track: str | None,
    query_split: str | None,
    candidate_split: str | None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    filtered = [row for row in rows if track is None or row["track"] == track]
    queries = [
        row
        for row in filtered
        if (query_notation is None or row["notation"] == query_notation)
        and (query_split is None or row.get("split") == query_split)
    ]
    candidates = [
        row
        for row in filtered
        if (candidate_notation is None or row["notation"] == candidate_notation)
        and (candidate_split is None or row.get("split") == candidate_split)
    ]

    if query_notation is None and candidate_notation is None:
        candidates = [row for row in filtered if candidate_split is None or row.get("split") == candidate_split]
    return queries, candidates


def remove_self_matches(
    queries: Sequence[Dict[str, object]],
    candidates: Sequence[Dict[str, object]],
    rankings: Sequence[Sequence[int]],
) -> List[List[int]]:
    cleaned: List[List[int]] = []
    for query, ranking in zip(queries, rankings):
        cleaned.append([idx for idx in ranking if candidates[idx]["form_id"] != query["form_id"]])
    return cleaned


def run_retrieval(
    input_path: Path,
    query_notation: str | None,
    candidate_notation: str | None,
    track: str | None,
    query_split: str | None,
    candidate_split: str | None,
    max_queries: int | None,
    max_candidates: int | None,
) -> Dict[str, float | int | str | None]:
    loaded_rows = read_jsonl(input_path)
    if loaded_rows and "forms" in loaded_rows[0]:
        rows = flatten_equivalence_classes(loaded_rows)
    else:
        rows = loaded_rows

    queries, candidates = split_query_candidate_rows(
        rows,
        query_notation,
        candidate_notation,
        track,
        query_split,
        candidate_split,
    )

    if not queries:
        raise ValueError("No query rows matched the requested filters")
    if not candidates:
        raise ValueError("No candidate rows matched the requested filters")

    if max_queries is not None:
        queries = queries[:max_queries]
    if max_candidates is not None:
        candidates = candidates[:max_candidates]

    idf_rows = list(queries) + list(candidates)
    idf = fit_idf([str(row["text"]) for row in idf_rows])
    query_vectors = [transform(str(row["text"]), idf) for row in queries]
    candidate_vectors = [transform(str(row["text"]), idf) for row in candidates]
    rankings = rank_candidates(query_vectors, candidate_vectors)
    rankings = remove_self_matches(queries, candidates, rankings)

    result = evaluate_rankings(
        query_fact_ids=[str(row["fact_id"]) for row in queries],
        candidate_fact_ids=[str(row["fact_id"]) for row in candidates],
        rankings=rankings,
    )
    metrics = result.to_dict()
    metrics.update(
        {
            "input": str(input_path),
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
    parser = argparse.ArgumentParser(description="Run a character n-gram retrieval baseline.")
    parser.add_argument("--input", type=Path, default=Path("data/bioxrep_synth.jsonl"))
    parser.add_argument("--query-notation", default=None)
    parser.add_argument("--candidate-notation", default=None)
    parser.add_argument("--query-split", default=None)
    parser.add_argument("--candidate-split", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument(
        "--track",
        choices=["variant", "lab", "gene_alias", "clinvar_gene", "clinvar_hgvs", "clinvar_summary"],
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_retrieval(
        input_path=args.input,
        query_notation=args.query_notation,
        candidate_notation=args.candidate_notation,
        track=args.track,
        query_split=args.query_split,
        candidate_split=args.candidate_split,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
