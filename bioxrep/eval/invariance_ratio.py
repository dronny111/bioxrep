from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch

from bioxrep.baselines.student_retrieval import encode_forms
from bioxrep.data.io import flatten_equivalence_classes, read_jsonl
from bioxrep.eval.hard_student_retrieval import load_student


def filter_rows(
    rows: Sequence[Dict[str, Any]],
    track: str | None,
    split: str | None,
    notations: set[str],
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        if track is not None and row.get("track") != track:
            continue
        if split is not None and row.get("split") != split:
            continue
        if notations and row.get("notation") not in notations:
            continue
        filtered.append(row)
    return filtered


def pairwise_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.array([], dtype=np.float32)
    embeddings = np.asarray(embeddings, dtype=np.float64)
    scores = embeddings @ embeddings.T
    upper = np.triu_indices(len(embeddings), k=1)
    return 1.0 - scores[upper]


def sample_between_distances(
    centroids: Sequence[np.ndarray],
    max_pairs: int,
    seed: int,
) -> np.ndarray:
    if len(centroids) < 2:
        return np.array([], dtype=np.float32)
    normalized_centroids = [np.asarray(centroid, dtype=np.float64) for centroid in centroids]
    rng = random.Random(seed)
    distances: List[float] = []
    total_possible = len(normalized_centroids) * (len(normalized_centroids) - 1) // 2
    if total_possible <= max_pairs:
        for left_idx in range(len(normalized_centroids)):
            for right_idx in range(left_idx + 1, len(normalized_centroids)):
                distances.append(float(1.0 - normalized_centroids[left_idx] @ normalized_centroids[right_idx]))
    else:
        seen: set[tuple[int, int]] = set()
        while len(distances) < max_pairs:
            left_idx = rng.randrange(len(normalized_centroids))
            right_idx = rng.randrange(len(normalized_centroids))
            if left_idx == right_idx:
                continue
            pair = tuple(sorted((left_idx, right_idx)))
            if pair in seen:
                continue
            seen.add(pair)
            distances.append(float(1.0 - normalized_centroids[pair[0]] @ normalized_centroids[pair[1]]))
    return np.array(distances, dtype=np.float32)


def compute_invariance_ratio(
    checkpoint_path: Path,
    input_path: Path,
    track: str | None = None,
    split: str | None = None,
    notations: Sequence[str] = (),
    min_forms_per_fact: int = 2,
    max_facts: int | None = None,
    max_between_pairs: int = 10000,
    batch_size: int = 512,
    seed: int = 13,
    device: str | None = None,
) -> Dict[str, Any]:
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loaded_rows = read_jsonl(input_path)
    rows = flatten_equivalence_classes(loaded_rows) if loaded_rows and "forms" in loaded_rows[0] else loaded_rows
    rows = filter_rows(rows, track=track, split=split, notations=set(notations))
    if not rows:
        raise ValueError("No rows matched the requested filters")

    grouped_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[str(row["fact_id"])].append(row)
    eligible_groups = [group for group in grouped_rows.values() if len(group) >= min_forms_per_fact]
    if max_facts is not None:
        rng = random.Random(seed)
        rng.shuffle(eligible_groups)
        eligible_groups = eligible_groups[:max_facts]
    if not eligible_groups:
        raise ValueError("No fact groups had enough forms to compute within-class distances")

    selected_rows = [row for group in eligible_groups for row in group]
    model, args, numeric_field_stats = load_student(checkpoint_path, torch_device)
    embeddings = encode_forms(
        model=model,
        rows=selected_rows,
        args=args,
        numeric_field_stats=numeric_field_stats,
        device=torch_device,
        batch_size=batch_size,
    )

    cursor = 0
    within_distances: List[float] = []
    centroids: List[np.ndarray] = []
    for group in eligible_groups:
        group_embeddings = embeddings[cursor : cursor + len(group)]
        cursor += len(group)
        within_distances.extend(pairwise_cosine_distances(group_embeddings).tolist())
        centroid = group_embeddings.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids.append(centroid)

    between_distances = sample_between_distances(centroids, max_pairs=max_between_pairs, seed=seed)
    if not within_distances:
        raise ValueError("No within-class form pairs were available after filtering")
    if len(between_distances) == 0:
        raise ValueError("At least two fact groups are required to compute between-class distance")

    mean_within = float(np.mean(within_distances))
    mean_between = float(np.mean(between_distances))
    return {
        "checkpoint": str(checkpoint_path),
        "input": str(input_path),
        "track": track,
        "split": split,
        "notations": list(notations),
        "fact_count": len(eligible_groups),
        "form_count": len(selected_rows),
        "within_pair_count": len(within_distances),
        "between_pair_count": int(len(between_distances)),
        "mean_within_cosine_distance": mean_within,
        "mean_between_cosine_distance": mean_between,
        "invariance_ratio": mean_between / max(mean_within, 1e-12),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute between-class / within-class student embedding distance.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--track", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--notations", default="", help="Comma-separated notation filter.")
    parser.add_argument("--min-forms-per-fact", type=int, default=2)
    parser.add_argument("--max-facts", type=int, default=None)
    parser.add_argument("--max-between-pairs", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = None if args.device == "auto" else args.device
    notations = [notation.strip() for notation in args.notations.split(",") if notation.strip()]
    metrics = compute_invariance_ratio(
        checkpoint_path=args.checkpoint,
        input_path=args.input,
        track=args.track,
        split=args.split,
        notations=notations,
        min_forms_per_fact=args.min_forms_per_fact,
        max_facts=args.max_facts,
        max_between_pairs=args.max_between_pairs,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
