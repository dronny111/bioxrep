"""Flat-pool retrieval adapter for a trained contrastive char student.

Scores a trained BioXRep student encoder on the SAME shared (queries,
candidate pool) setup used by the SapBERT / char-ngram / BioSyn / canonical
baselines, via :func:`bioxrep.eval.retrieval.evaluate_rankings`. This mirrors
``sapbert_retrieval.py`` (identical query/candidate filtering, self-match
removal, and output JSON schema) but swaps the encoder for a byte-level char
student loaded from a checkpoint.

The existing ``bioxrep/eval/hard_student_retrieval.py`` scores a student on
per-query HARD candidate sets; this adapter instead ranks every query against
the full flat candidate pool so student numbers are directly comparable to the
dense/hybrid baselines in the HGNC alias comparison table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from bioxrep.baselines.sapbert_retrieval import (
    remove_self_matches,
    split_query_candidate_rows,
)
from bioxrep.data.io import flatten_equivalence_classes, read_jsonl
from bioxrep.eval.hard_student_retrieval import load_student
from bioxrep.eval.retrieval import evaluate_rankings
from bioxrep.train.train_contrastive_student import (
    encode_numeric_field_values,
    encode_text_tensors,
)


def encode_forms(
    model,
    rows: Sequence[Dict[str, object]],
    args: Dict[str, object],
    numeric_field_stats: Dict[str, Dict[str, float]],
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Batch-encode rows into L2-normalized student embeddings.

    Uses the same byte-level tokenization (``encode_text_tensors``) and numeric
    field encoding the student was trained with, so query and candidate vectors
    live in the checkpoint's embedding space.
    """
    max_length = int(args.get("max_length", 128))
    text_transform = str(args.get("text_transform", "none"))
    use_numeric = bool(numeric_field_stats) and args.get("numeric_feature_mode", "none") != "none"

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            id_list, mask_list = [], []
            for row in batch:
                ids, mask = encode_text_tensors(str(row["text"]), max_length, text_transform)
                id_list.append(ids)
                mask_list.append(mask)
            token_ids = torch.stack(id_list).to(device)
            mask = torch.stack(mask_list).to(device)

            numeric_values = numeric_mask = None
            if use_numeric:
                nv_list, nm_list = [], []
                for row in batch:
                    values, present = encode_numeric_field_values(
                        row.get("attributes", {}) or {}, numeric_field_stats
                    )
                    nv_list.append(values)
                    nm_list.append(present)
                numeric_values = torch.stack(nv_list).to(device)
                numeric_mask = torch.stack(nm_list).to(device)

            vecs = model.encode(token_ids, mask, numeric_values, numeric_mask)
            embeddings.append(vecs.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def rank_dense(query_emb: np.ndarray, candidate_emb: np.ndarray) -> np.ndarray:
    """Per query, candidate indices ranked by descending cosine.

    Student vectors are already L2-normalized, so cosine is a dot product.
    """
    scores = query_emb @ candidate_emb.T
    return np.argsort(-scores, axis=1)


def run_retrieval(
    checkpoint_path: Path,
    input_path: Path,
    query_notation: str | None,
    candidate_notation: str | None,
    track: str | None,
    query_split: str | None,
    candidate_split: str | None,
    max_queries: int | None,
    max_candidates: int | None,
    batch_size: int = 512,
    device: str | None = None,
    bootstrap: bool = False,
    bootstrap_resamples: int = 1000,
) -> Dict[str, object]:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

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

    model, args, numeric_field_stats = load_student(checkpoint_path, torch_device)

    query_emb = encode_forms(model, queries, args, numeric_field_stats, torch_device, batch_size)
    candidate_emb = encode_forms(model, candidates, args, numeric_field_stats, torch_device, batch_size)

    rankings = rank_dense(query_emb, candidate_emb)
    rankings = remove_self_matches(queries, candidates, rankings)

    result = evaluate_rankings(
        query_fact_ids=[str(row["fact_id"]) for row in queries],
        candidate_fact_ids=[str(row["fact_id"]) for row in candidates],
        rankings=rankings,
    )
    metrics = result.to_dict(bootstrap=bootstrap, num_resamples=bootstrap_resamples)
    metrics.update(
        {
            "input": str(input_path),
            "model": str(checkpoint_path),
            "scoring": "student_dense",
            "encoder": args.get("encoder"),
            "hidden_dim": args.get("hidden_dim"),
            "projection_dim": args.get("projection_dim"),
            "text_transform": args.get("text_transform"),
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
    parser = argparse.ArgumentParser(
        description="Score a trained contrastive char student on the shared flat candidate pool."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to char_*_student.pt checkpoint.")
    parser.add_argument("--input", type=Path, default=Path("data/bioxrep_hgnc_alias_symbol_heldout.jsonl"))
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
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap", action="store_true", help="Add percentile bootstrap CIs to the metrics.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_retrieval(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        query_notation=args.query_notation,
        candidate_notation=args.candidate_notation,
        track=args.track,
        query_split=args.query_split,
        candidate_split=args.candidate_split,
        max_queries=args.max_queries,
        max_candidates=args.max_candidates,
        batch_size=args.batch_size,
        device=args.device,
        bootstrap=args.bootstrap,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
