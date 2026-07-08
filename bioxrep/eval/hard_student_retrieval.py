from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

from bioxrep.data.io import read_jsonl
from bioxrep.eval.retrieval import bootstrap_ci
from bioxrep.models.char_encoder import CharCNNEncoder, CharMeanEncoder, CharTransformerEncoder
from bioxrep.train.train_contrastive_student import (
    ContrastiveStudent,
    NumericFeatureEncoder,
    encode_numeric_field_values,
)


def transform_text(text: str, mode: str) -> str:
    if mode == "mask_digits":
        return re.sub(r"\d", "#", text)
    if mode == "strip_digits":
        return re.sub(r"\d", "", text)
    return text


def encode_text(text: str, max_length: int, text_transform: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    text = transform_text(text, text_transform)
    encoded = text.encode("utf-8")[:max_length]
    ids = [byte + 1 for byte in encoded]
    length = len(ids)
    if length < max_length:
        ids.extend([0] * (max_length - length))
    mask = [1.0] * length + [0.0] * (max_length - length)
    return (
        torch.tensor([ids], dtype=torch.long, device=device),
        torch.tensor([mask], dtype=torch.float, device=device),
    )


def build_encoder(args: Mapping[str, Any]) -> torch.nn.Module:
    if args["encoder"] == "mean":
        return CharMeanEncoder(hidden_dim=args["hidden_dim"], projection_dim=args["projection_dim"])
    if args["encoder"] == "cnn":
        kernel_sizes = tuple(int(value) for value in str(args.get("kernel_sizes", "3,5,7")).split(",") if value.strip())
        return CharCNNEncoder(
            hidden_dim=args["hidden_dim"],
            projection_dim=args["projection_dim"],
            kernel_sizes=kernel_sizes,
            dropout=args.get("dropout", 0.1),
        )
    if args["encoder"] == "transformer":
        return CharTransformerEncoder(
            hidden_dim=args["hidden_dim"],
            projection_dim=args["projection_dim"],
            num_layers=args.get("transformer_layers", 2),
            num_heads=args.get("transformer_heads", 4),
            max_length=args.get("max_length", 512),
            dropout=args.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown encoder: {args['encoder']}")


def load_student(checkpoint_path: Path, device: torch.device) -> tuple[ContrastiveStudent, Dict[str, Any], Dict[str, Dict[str, float]]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]
    numeric_field_stats = checkpoint.get("numeric_field_stats", {})
    numeric_feature_encoder = None
    if numeric_field_stats and args.get("numeric_feature_mode", "none") != "none":
        numeric_feature_encoder = NumericFeatureEncoder(
            input_dim=len(numeric_field_stats),
            projection_dim=args["projection_dim"],
            mode=args["numeric_feature_mode"],
            fourier_dim=args.get("numeric_fourier_dim", 16),
        )
    state_dict = checkpoint["model_state_dict"]
    numeric_head_fields = [
        field for field in numeric_field_stats if f"numeric_heads.{field}.weight" in state_dict
    ]
    model = ContrastiveStudent(
        encoder=build_encoder(args),
        projection_dim=args["projection_dim"],
        attribute_vocabularies=checkpoint.get("attribute_vocabularies", {}),
        numeric_target_fields=numeric_head_fields,
        numeric_feature_encoder=numeric_feature_encoder,
        text_weight=args.get("text_weight", 1.0),
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, args, numeric_field_stats


def numeric_tensors(
    attributes: Mapping[str, Any],
    numeric_field_stats: Mapping[str, Mapping[str, float]],
    device: torch.device,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if not numeric_field_stats:
        return None, None
    values, mask = encode_numeric_field_values(attributes, numeric_field_stats)
    return (
        torch.tensor([values], dtype=torch.float, device=device),
        torch.tensor([mask], dtype=torch.float, device=device),
    )


@torch.no_grad()
def encode_form(
    model: ContrastiveStudent,
    text: str,
    attributes: Mapping[str, Any],
    args: Mapping[str, Any],
    numeric_field_stats: Mapping[str, Mapping[str, float]],
    device: torch.device,
) -> torch.Tensor:
    token_ids, mask = encode_text(
        text=text,
        max_length=args.get("max_length", 128),
        text_transform=args.get("text_transform", "none"),
        device=device,
    )
    numeric_values, numeric_mask = numeric_tensors(attributes, numeric_field_stats, device)
    return model.encode(token_ids, mask, numeric_values, numeric_mask)


def reciprocal_rank(labels: Sequence[bool], ranking: Sequence[int]) -> float:
    for rank_idx, candidate_idx in enumerate(ranking, start=1):
        if labels[candidate_idx]:
            return 1.0 / rank_idx
    return 0.0


@torch.no_grad()
def evaluate_hard_set(
    checkpoint_path: Path,
    input_path: Path,
    device: torch.device,
    bootstrap: bool = False,
    bootstrap_resamples: int = 1000,
) -> Dict[str, Any]:
    records = read_jsonl(input_path)
    if not records:
        raise ValueError("No hard retrieval records were provided")

    model, args, numeric_field_stats = load_student(checkpoint_path, device)
    top1_hits = 0
    top5_hits = 0
    top1_flags: List[float] = []
    top5_flags: List[float] = []
    reciprocal_ranks: List[float] = []
    candidate_counts: List[int] = []
    matched_decoy_counts: List[int] = []
    random_decoy_counts: List[int] = []

    for record in records:
        query = record["query"]
        query_embedding = encode_form(
            model=model,
            text=str(query["text"]),
            attributes=query.get("attributes", {}),
            args=args,
            numeric_field_stats=numeric_field_stats,
            device=device,
        )
        candidate_embeddings = [
            encode_form(
                model=model,
                text=str(candidate["text"]),
                attributes=candidate.get("attributes", {}),
                args=args,
                numeric_field_stats=numeric_field_stats,
                device=device,
            )
            for candidate in record["candidates"]
        ]
        candidate_matrix = torch.cat(candidate_embeddings, dim=0)
        scores = (query_embedding @ candidate_matrix.T).squeeze(0)
        ranking = torch.argsort(scores, descending=True).cpu().tolist()
        labels = [bool(candidate.get("is_positive")) for candidate in record["candidates"]]

        reciprocal_ranks.append(reciprocal_rank(labels, ranking))
        is_top1 = 1.0 if (ranking and labels[ranking[0]]) else 0.0
        is_top5 = 1.0 if any(labels[idx] for idx in ranking[:5]) else 0.0
        top1_hits += int(is_top1)
        top5_hits += int(is_top5)
        top1_flags.append(is_top1)
        top5_flags.append(is_top5)
        candidate_counts.append(len(record["candidates"]))
        matched_decoy_counts.append(int(record.get("matched_decoy_count", 0)))
        random_decoy_counts.append(int(record.get("random_decoy_count", 0)))

    query_count = len(records)
    total_decoys = sum(matched_decoy_counts) + sum(random_decoy_counts)
    metrics: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "input": str(input_path),
        "top1": top1_hits / query_count,
        "top5": top5_hits / query_count,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / query_count,
        "query_count": query_count,
        "avg_candidates": sum(candidate_counts) / query_count,
        "min_candidates": min(candidate_counts),
        "max_candidates": max(candidate_counts),
        # Pool composition, so top-k is read against how many decoys are truly hard.
        "avg_matched_decoys": sum(matched_decoy_counts) / query_count,
        "avg_random_decoys": sum(random_decoy_counts) / query_count,
        "matched_decoy_fraction": (sum(matched_decoy_counts) / total_decoys) if total_decoys else None,
        "encoder": args.get("encoder"),
        "numeric_fields": list(numeric_field_stats),
        "numeric_feature_mode": args.get("numeric_feature_mode", "none"),
        "text_transform": args.get("text_transform", "none"),
        "text_weight": args.get("text_weight", 1.0),
    }
    if bootstrap:
        metrics["confidence_level"] = 0.95
        metrics["bootstrap_resamples"] = bootstrap_resamples
        for name, flags in (("top1", top1_flags), ("top5", top5_flags), ("mean_reciprocal_rank", reciprocal_ranks)):
            low, high = bootstrap_ci(flags, num_resamples=bootstrap_resamples)
            metrics[f"{name}_ci_low"] = low
            metrics[f"{name}_ci_high"] = high
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BioXRep student on hard retrieval candidate sets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--bootstrap", action="store_true", help="Add percentile bootstrap CIs to the metrics.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    metrics = evaluate_hard_set(
        args.checkpoint,
        args.input,
        device,
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
