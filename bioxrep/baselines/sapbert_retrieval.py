"""SapBERT / BioSyn-style dense retrieval baseline for BioXRep.

This baseline mirrors the CLI and row-splitting behaviour of
``bioxrep.baselines.char_ngram_retrieval`` so its numbers are directly
comparable, but replaces the sparse character n-gram TF-IDF vectors with dense
embeddings from a pretrained biomedical encoder (SapBERT by default).

Two scoring modes are supported:

* ``dense``  -- rank candidates by cosine similarity of L2-normalized encoder
  embeddings. This is the SapBERT baseline.
* ``hybrid`` -- BioSyn-style score that linearly combines the dense cosine
  score with the repository's char n-gram TF-IDF (sparse) cosine score:
  ``score = (1 - lambda) * sparse + lambda * dense``. This mirrors the BioSyn
  design of fusing a sparse lexical scorer with a dense synonym-aware encoder.

Example
-------
SapBERT on the HGNC alias-symbol held-out task (same slice as the char n-gram
baseline in the README)::

    python3 -m bioxrep.baselines.sapbert_retrieval \\
        --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl \\
        --track gene_alias --query-split test --candidate-split train \\
        --candidate-notation approved_symbol --max-queries 200 \\
        --output outputs/sapbert_hgnc_alias_symbol_heldout.json

BioSyn-style hybrid on the same slice::

    python3 -m bioxrep.baselines.sapbert_retrieval \\
        --input data/bioxrep_hgnc_alias_symbol_heldout.jsonl \\
        --track gene_alias --query-split test --candidate-split train \\
        --candidate-notation approved_symbol --max-queries 200 \\
        --scoring hybrid --hybrid-lambda 0.7 \\
        --output outputs/biosyn_hybrid_hgnc_alias_symbol_heldout.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from bioxrep.baselines.char_ngram_retrieval import (
    cosine as sparse_cosine,
    fit_idf,
    transform as sparse_transform,
)
from bioxrep.data.io import flatten_equivalence_classes, read_jsonl
from bioxrep.eval.retrieval import evaluate_rankings


DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"


def split_query_candidate_rows(
    rows: Sequence[Dict[str, object]],
    query_notation: str | None,
    candidate_notation: str | None,
    track: str | None,
    query_split: str | None,
    candidate_split: str | None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Filter rows into query/candidate pools (same logic as char_ngram_retrieval)."""
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


def _mean_pool(last_hidden_state, attention_mask):
    """Mean-pool token embeddings using the attention mask."""
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def encode_texts(
    texts: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 128,
    max_length: int = 32,
    pooling: str = "cls",
    device: str | None = None,
) -> np.ndarray:
    """Embed texts with a pretrained encoder and L2-normalize the vectors."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[start : start + batch_size]]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            outputs = model(**encoded)
            if pooling == "mean":
                vecs = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            else:  # cls
                vecs = outputs.last_hidden_state[:, 0, :]
            vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
            embeddings.append(vecs.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def rank_dense(query_emb: np.ndarray, candidate_emb: np.ndarray) -> np.ndarray:
    """Return, per query, candidate indices ranked by descending cosine.

    Vectors are already L2-normalized, so cosine is a plain dot product.
    Returns an int array of shape (num_queries, num_candidates).
    """
    scores = query_emb @ candidate_emb.T
    return np.argsort(-scores, axis=1)


def rank_hybrid(
    query_texts: Sequence[str],
    candidate_texts: Sequence[str],
    query_emb: np.ndarray,
    candidate_emb: np.ndarray,
    hybrid_lambda: float,
) -> np.ndarray:
    """BioSyn-style fusion: (1 - lambda) * sparse + lambda * dense.

    Sparse scores use the repository char n-gram TF-IDF cosine so the lexical
    component is identical to the char_ngram baseline.
    """
    dense_scores = query_emb @ candidate_emb.T  # (Q, C), cosine

    idf = fit_idf([str(t) for t in list(query_texts) + list(candidate_texts)])
    candidate_vectors = [sparse_transform(str(t), idf) for t in candidate_texts]
    query_vectors = [sparse_transform(str(t), idf) for t in query_texts]

    rankings: List[List[int]] = []
    for q_idx, q_vec in enumerate(query_vectors):
        sparse_row = np.array(
            [sparse_cosine(q_vec, c_vec) for c_vec in candidate_vectors],
            dtype=np.float64,
        )
        fused = (1.0 - hybrid_lambda) * sparse_row + hybrid_lambda * dense_scores[q_idx]
        rankings.append(list(np.argsort(-fused)))
    return np.array(rankings, dtype=np.int64)


def remove_self_matches(
    queries: Sequence[Dict[str, object]],
    candidates: Sequence[Dict[str, object]],
    rankings: Sequence[Sequence[int]],
) -> List[List[int]]:
    """Drop any candidate that is the exact same form as the query (by form_id)."""
    cleaned: List[List[int]] = []
    for query, ranking in zip(queries, rankings):
        cleaned.append([int(idx) for idx in ranking if candidates[int(idx)]["form_id"] != query["form_id"]])
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
    model_name: str = DEFAULT_MODEL,
    scoring: str = "dense",
    hybrid_lambda: float = 0.7,
    pooling: str = "cls",
    batch_size: int = 128,
    max_length: int = 32,
) -> Dict[str, object]:
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

    query_texts = [str(row["text"]) for row in queries]
    candidate_texts = [str(row["text"]) for row in candidates]

    query_emb = encode_texts(
        query_texts, model_name=model_name, batch_size=batch_size, max_length=max_length, pooling=pooling
    )
    candidate_emb = encode_texts(
        candidate_texts, model_name=model_name, batch_size=batch_size, max_length=max_length, pooling=pooling
    )

    if scoring == "hybrid":
        rankings = rank_hybrid(query_texts, candidate_texts, query_emb, candidate_emb, hybrid_lambda)
    else:
        rankings = rank_dense(query_emb, candidate_emb)

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
            "model": model_name,
            "scoring": scoring,
            "hybrid_lambda": hybrid_lambda if scoring == "hybrid" else None,
            "pooling": pooling,
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
    parser = argparse.ArgumentParser(description="Run a SapBERT / BioSyn-style dense retrieval baseline.")
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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace encoder id.")
    parser.add_argument("--scoring", choices=["dense", "hybrid"], default="dense")
    parser.add_argument(
        "--hybrid-lambda",
        type=float,
        default=0.7,
        help="Dense weight in BioSyn-style fusion; sparse weight is (1 - lambda).",
    )
    parser.add_argument("--pooling", choices=["cls", "mean"], default="cls")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=32)
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
        model_name=args.model,
        scoring=args.scoring,
        hybrid_lambda=args.hybrid_lambda,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
