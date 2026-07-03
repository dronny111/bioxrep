"""Batch driver: compare SapBERT / BioSyn-hybrid / char n-gram / canonical
teacher on HGNC alias-normalization retrieval, over an identical candidate pool.

For each task slice, all methods score the SAME (queries, candidates) pool so
the numbers are directly comparable. The candidate pool is encoded with SapBERT
only once and cached to disk. The char n-gram scores reproduce the repository's
``char_ngram_retrieval`` TF-IDF cosine exactly (same idf formula, same
normalization) via a scipy-sparse implementation, so the numbers equal the
in-repo baseline while running fast enough over the full ~45k pool.

Outputs: one result JSON per (task, method) under ``outputs/``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import sparse

from bioxrep.baselines.char_ngram_retrieval import char_ngrams, fit_idf
from bioxrep.baselines.sapbert_retrieval import encode_texts, DEFAULT_MODEL
from bioxrep.data.io import flatten_equivalence_classes, read_jsonl
from bioxrep.eval.retrieval import evaluate_rankings

REPO = Path(__file__).resolve().parents[1]
INPUT = REPO / "data" / "bioxrep_hgnc_alias_symbol_heldout.jsonl"
OUTDIR = REPO / "outputs"
CACHEDIR = REPO / ".sapbert_emb_cache"
MAX_QUERIES = 2000
HYBRID_LAMBDA = 0.7

# (task_name, query_notation, query_split, candidate_notation, candidate_split)
TASKS = [
    ("alias_symbol_heldout", "alias_symbol", "test", "approved_symbol", "train"),
    ("prev_symbol", "prev_symbol", "train", "approved_symbol", "train"),
    ("alias_name", "alias_name", "train", "approved_symbol", "train"),
]


def filter_rows(rows, notation, split):
    return [
        r
        for r in rows
        if r["track"] == "gene_alias" and r["notation"] == notation and r.get("split") == split
    ]


def remove_self_matches(queries, candidates, rankings):
    cleaned = []
    for q, ranking in zip(queries, rankings):
        cleaned.append([int(i) for i in ranking if candidates[int(i)]["form_id"] != q["form_id"]])
    return cleaned


def build_sparse(texts: Sequence[str], idf: Dict[str, float], vocab: Dict[str, int]):
    """Repo-faithful char n-gram TF-IDF matrix, L2-normalized rows (CSR)."""
    from collections import Counter

    indptr = [0]
    indices: List[int] = []
    data: List[float] = []
    for text in texts:
        counts = Counter(char_ngrams(str(text)))
        row_idx: List[int] = []
        row_val: List[float] = []
        for gram, count in counts.items():
            j = vocab.get(gram)
            if j is None:
                continue
            w = count * idf.get(gram, 0.0)
            if w != 0.0:
                row_idx.append(j)
                row_val.append(w)
        norm = float(np.sqrt(np.sum(np.square(row_val)))) if row_val else 0.0
        if norm > 0.0:
            row_val = [v / norm for v in row_val]
        indices.extend(row_idx)
        data.extend(row_val)
        indptr.append(len(indices))
    return sparse.csr_matrix((data, indices, indptr), shape=(len(texts), len(vocab)), dtype=np.float64)


def metrics_dict(queries, candidates, rankings, extra):
    res = evaluate_rankings(
        query_fact_ids=[str(q["fact_id"]) for q in queries],
        candidate_fact_ids=[str(c["fact_id"]) for c in candidates],
        rankings=rankings,
    )
    m = res.to_dict()
    m.update(extra)
    return m


def canonical_rank(queries, candidates):
    """Rank candidates by exact match on attributes.symbol (structured upper bound)."""
    cand_keys = [str(c["attributes"]["symbol"]) for c in candidates]
    key_to_idxs: Dict[str, List[int]] = {}
    for i, k in enumerate(cand_keys):
        key_to_idxs.setdefault(k, []).append(i)
    all_idx = list(range(len(candidates)))
    rankings = []
    for q in queries:
        qk = str(q["attributes"]["symbol"])
        exact = key_to_idxs.get(qk, [])
        exact_set = set(exact)
        rankings.append(exact + [i for i in all_idx if i not in exact_set])
    return rankings


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    CACHEDIR.mkdir(exist_ok=True)
    rows = read_jsonl(INPUT)
    if rows and "forms" in rows[0]:
        rows = flatten_equivalence_classes(rows)

    # Shared candidate pool: approved_symbol / train (encode once).
    candidates = filter_rows(rows, "approved_symbol", "train")
    cand_texts = [str(c["text"]) for c in candidates]
    cand_cache = CACHEDIR / "approved_symbol_train_emb.npy"
    if cand_cache.exists():
        cand_emb = np.load(cand_cache)
        print(f"loaded cached candidate emb {cand_emb.shape}")
    else:
        print(f"encoding {len(cand_texts)} candidates ...")
        cand_emb = encode_texts(cand_texts, device="cpu", batch_size=128).astype(np.float32)
        np.save(cand_cache, cand_emb)
        print(f"encoded + cached candidate emb {cand_emb.shape}")

    summary = []
    for task_name, q_not, q_split, c_not, c_split in TASKS:
        assert c_not == "approved_symbol" and c_split == "train"
        queries = filter_rows(rows, q_not, q_split)[:MAX_QUERIES]
        q_texts = [str(q["text"]) for q in queries]
        print(f"\n=== task {task_name}: {len(queries)} queries x {len(candidates)} candidates ===")

        q_cache = CACHEDIR / f"{task_name}_query_emb.npy"
        if q_cache.exists():
            q_emb = np.load(q_cache)
        else:
            q_emb = encode_texts(q_texts, device="cpu", batch_size=128).astype(np.float32)
            np.save(q_cache, q_emb)

        common = {
            "input": str(INPUT.relative_to(REPO)),
            "task": task_name,
            "query_notation": q_not,
            "candidate_notation": c_not,
            "query_split": q_split,
            "candidate_split": c_split,
            "track": "gene_alias",
            "candidate_count": len(candidates),
            "max_queries": MAX_QUERIES,
        }

        # ---- SapBERT dense ----
        dense_scores = q_emb @ cand_emb.T  # (Q, C) cosine
        dense_rank = np.argsort(-dense_scores, axis=1, kind="stable")
        dense_rank = remove_self_matches(queries, candidates, dense_rank)
        m = metrics_dict(queries, candidates, dense_rank, {**common, "method": "sapbert_dense", "model": DEFAULT_MODEL})
        (OUTDIR / f"sapbert_dense_hgnc_{task_name}.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        summary.append(m)
        print(f"  sapbert_dense   top1={m['top1']:.4f} top5={m['top5']:.4f} mrr={m['mean_reciprocal_rank']:.4f}")

        # ---- char n-gram (repo-faithful TF-IDF cosine) ----
        idf = fit_idf(q_texts + cand_texts)
        vocab = {g: i for i, g in enumerate(idf.keys())}
        Q = build_sparse(q_texts, idf, vocab)
        C = build_sparse(cand_texts, idf, vocab)
        sparse_scores = np.asarray((Q @ C.T).todense())  # (Q, C) cosine
        cn_rank = np.argsort(-sparse_scores, axis=1, kind="stable")
        cn_rank = remove_self_matches(queries, candidates, cn_rank)
        m = metrics_dict(queries, candidates, cn_rank, {**common, "method": "char_ngram"})
        (OUTDIR / f"char_ngram_hgnc_{task_name}.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        summary.append(m)
        print(f"  char_ngram      top1={m['top1']:.4f} top5={m['top5']:.4f} mrr={m['mean_reciprocal_rank']:.4f}")

        # ---- BioSyn-style hybrid ----
        fused = (1.0 - HYBRID_LAMBDA) * sparse_scores + HYBRID_LAMBDA * dense_scores
        hy_rank = np.argsort(-fused, axis=1, kind="stable")
        hy_rank = remove_self_matches(queries, candidates, hy_rank)
        m = metrics_dict(queries, candidates, hy_rank, {**common, "method": "biosyn_hybrid", "hybrid_lambda": HYBRID_LAMBDA, "model": DEFAULT_MODEL})
        (OUTDIR / f"biosyn_hybrid_hgnc_{task_name}.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        summary.append(m)
        print(f"  biosyn_hybrid   top1={m['top1']:.4f} top5={m['top5']:.4f} mrr={m['mean_reciprocal_rank']:.4f}")

        # ---- canonical teacher (structured upper bound) ----
        ct_rank = canonical_rank(queries, candidates)
        ct_rank = remove_self_matches(queries, candidates, ct_rank)
        m = metrics_dict(queries, candidates, ct_rank, {**common, "method": "canonical_teacher", "canonical_key": "symbol"})
        (OUTDIR / f"canonical_teacher_hgnc_{task_name}.json").write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
        summary.append(m)
        print(f"  canonical_teach top1={m['top1']:.4f} top5={m['top5']:.4f} mrr={m['mean_reciprocal_rank']:.4f}")

    (OUTDIR / "hgnc_alias_comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {len(summary)} result rows to {OUTDIR}")


if __name__ == "__main__":
    main()
