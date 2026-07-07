from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from bioxrep.data.io import read_jsonl
from bioxrep.models.char_encoder import CharCNNEncoder, CharMeanEncoder, CharTransformerEncoder


def normalize_attribute_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return None
    text = str(value).strip()
    return text or None


def build_attribute_vocabularies(rows: Sequence[Dict[str, object]], attribute_fields: Sequence[str]) -> Dict[str, Dict[str, int]]:
    vocabularies: Dict[str, Dict[str, int]] = {}
    for field in attribute_fields:
        counts: Counter[str] = Counter()
        for row in rows:
            attributes = row.get("attributes", {})
            if not isinstance(attributes, Mapping):
                continue
            normalized = normalize_attribute_value(attributes.get(field))
            if normalized is not None:
                counts[normalized] += 1
        if counts:
            vocabularies[field] = {value: index for index, value in enumerate(sorted(counts))}
    return vocabularies


def numeric_attribute_value(value: object) -> float | None:
    if value is None or isinstance(value, list):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def build_numeric_field_stats(rows: Sequence[Dict[str, object]], numeric_fields: Sequence[str]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for field in numeric_fields:
        values: List[float] = []
        for row in rows:
            attributes = row.get("attributes", {})
            if not isinstance(attributes, Mapping):
                continue
            value = numeric_attribute_value(attributes.get(field))
            if value is not None:
                values.append(value)
        if not values:
            continue
        min_value = min(values)
        max_value = max(values)
        scale = max(max_value - min_value, 1.0)
        stats[field] = {"min": min_value, "max": max_value, "scale": scale}
    return stats


def validate_numeric_fields(
    requested_fields: Sequence[str],
    numeric_field_stats: Mapping[str, Mapping[str, float]],
) -> List[str]:
    missing_fields = [field for field in requested_fields if field not in numeric_field_stats]
    if missing_fields:
        available = ", ".join(sorted(numeric_field_stats)) or "none"
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"Requested numeric field(s) have no numeric values in the training data: {missing}. "
            f"Available numeric fields from this request: {available}."
        )
    return list(numeric_field_stats)


def encode_numeric_field_values(
    attributes: Mapping[str, object],
    numeric_field_stats: Mapping[str, Mapping[str, float]],
) -> Tuple[List[float], List[float]]:
    normalized_values: List[float] = []
    present_mask: List[float] = []
    for field, field_stats in numeric_field_stats.items():
        value = numeric_attribute_value(attributes.get(field))
        if value is None:
            normalized_values.append(0.0)
            present_mask.append(0.0)
            continue
        normalized_values.append((value - field_stats["min"]) / field_stats["scale"])
        present_mask.append(1.0)
    return normalized_values, present_mask


def transform_text(text: str, text_transform: str) -> str:
    if text_transform == "mask_digits":
        return re.sub(r"\d", "#", text)
    if text_transform == "strip_digits":
        return re.sub(r"\d", "", text)
    return text


def encode_text_tensors(text: str, max_length: int, text_transform: str) -> Tuple[torch.Tensor, torch.Tensor]:
    text = transform_text(text, text_transform)
    encoded = text.encode("utf-8")[:max_length]
    ids = [byte + 1 for byte in encoded]
    length = len(ids)
    if length < max_length:
        ids.extend([0] * (max_length - length))
    mask = [1.0] * length + [0.0] * (max_length - length)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float)


class NumericFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, mode: str, fourier_dim: int = 16):
        super().__init__()
        self.mode = mode
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.fourier_dim = fourier_dim
        if mode == "explicit":
            self.project = nn.Sequential(
                nn.Linear(input_dim * 2, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        elif mode == "sinusoidal":
            self.project = nn.Sequential(
                nn.Linear(input_dim * fourier_dim * 2 + input_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        elif mode == "xval":
            # xVal-style continuous tokenization (Golkar et al., 2023): a single learned
            # embedding per field is scaled multiplicatively by the (normalized) value,
            # so magnitude is carried in the activation rather than in discrete tokens.
            self.value_embeddings = nn.Parameter(torch.randn(input_dim, projection_dim) * 0.02)
            self.project = nn.Sequential(
                nn.Linear(projection_dim, projection_dim),
                nn.GELU(),
                nn.Linear(projection_dim, projection_dim),
            )
        else:
            raise ValueError(f"Unknown numeric feature mode: {mode}")

    def forward(self, normalized_values: torch.Tensor, present_mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "explicit":
            features = torch.cat([normalized_values, present_mask], dim=1)
            return self.project(features)

        if self.mode == "xval":
            # Scale each field's learned embedding by its normalized value, zero out
            # absent fields, and sum across fields before projecting.
            scaled = (normalized_values * present_mask).unsqueeze(-1) * self.value_embeddings.unsqueeze(0)
            pooled = scaled.sum(dim=1)
            return self.project(pooled)

        frequencies = torch.pow(
            2.0,
            torch.arange(self.fourier_dim, device=normalized_values.device, dtype=normalized_values.dtype),
        ).view(1, 1, -1)
        phases = normalized_values.unsqueeze(-1) * frequencies * torch.pi
        sin_features = torch.sin(phases) * present_mask.unsqueeze(-1)
        cos_features = torch.cos(phases) * present_mask.unsqueeze(-1)
        features = torch.cat(
            [sin_features.reshape(normalized_values.shape[0], -1), cos_features.reshape(normalized_values.shape[0], -1), present_mask],
            dim=1,
        )
        return self.project(features)


class ContrastiveStudent(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int,
        attribute_vocabularies: Mapping[str, Mapping[str, int]],
        numeric_target_fields: Sequence[str],
        numeric_feature_encoder: NumericFeatureEncoder | None = None,
        text_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder = encoder
        self.numeric_feature_encoder = numeric_feature_encoder
        self.text_weight = text_weight
        self.attribute_heads = nn.ModuleDict(
            {field: nn.Linear(projection_dim, len(vocabulary)) for field, vocabulary in attribute_vocabularies.items()}
        )
        self.numeric_heads = nn.ModuleDict({field: nn.Linear(projection_dim, 1) for field in numeric_target_fields})

    def encode(
        self,
        token_ids: torch.Tensor,
        mask: torch.Tensor,
        numeric_values: torch.Tensor | None = None,
        numeric_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.encoder(token_ids, mask) * self.text_weight
        if self.numeric_feature_encoder is None or numeric_values is None or numeric_mask is None:
            return F.normalize(embeddings, dim=-1)
        numeric_embeddings = self.numeric_feature_encoder(numeric_values, numeric_mask)
        return F.normalize(embeddings + numeric_embeddings, dim=-1)

    def attribute_logits(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {field: head(embeddings) for field, head in self.attribute_heads.items()}

    def token_features(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.encoder, "token_features"):
            raise ValueError("The selected encoder does not expose token-level features for attention distillation")
        return self.encoder.token_features(token_ids, mask)


class PairDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, object]],
        max_length: int,
        attribute_vocabularies: Mapping[str, Mapping[str, int]],
        numeric_field_stats: Mapping[str, Mapping[str, float]],
        numeric_fields: Sequence[str],
        text_transform: str,
    ):
        self.rows = list(rows)
        self.max_length = max_length
        self.attribute_vocabularies = dict(attribute_vocabularies)
        self.numeric_field_stats = dict(numeric_field_stats)
        self.numeric_fields = list(numeric_fields)
        self.text_transform = text_transform

    def __len__(self) -> int:
        return len(self.rows)

    def transform_text(self, text: str) -> str:
        return transform_text(text, self.text_transform)

    def encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return encode_text_tensors(text, self.max_length, self.text_transform)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.rows[idx]
        left_ids, left_mask = self.encode_text(str(row["left_text"]))
        right_ids, right_mask = self.encode_text(str(row["right_text"]))
        attributes = row.get("attributes", {})
        attribute_labels: Dict[str, int] = {}
        numeric_values: List[float] = []
        numeric_mask: List[float] = []
        if isinstance(attributes, Mapping):
            for field, vocabulary in self.attribute_vocabularies.items():
                normalized = normalize_attribute_value(attributes.get(field))
                attribute_labels[field] = vocabulary.get(normalized, -1)
            numeric_values, numeric_mask = encode_numeric_field_values(attributes, self.numeric_field_stats)
        return {
            "left_ids": left_ids,
            "left_mask": left_mask,
            "right_ids": right_ids,
            "right_mask": right_mask,
            "fact_id": row["fact_id"],
            "attribute_labels": attribute_labels,
            "numeric_values": torch.tensor(numeric_values, dtype=torch.float),
            "numeric_mask": torch.tensor(numeric_mask, dtype=torch.float),
            "numeric_field_names": list(self.numeric_fields),
        }


class ClassFormDataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Dict[str, object]],
        max_length: int,
        attribute_vocabularies: Mapping[str, Mapping[str, int]],
        numeric_field_stats: Mapping[str, Mapping[str, float]],
        numeric_fields: Sequence[str],
        text_transform: str,
        forms_per_class: int,
        notation_filter: Sequence[str],
        seed: int,
    ):
        self.max_length = max_length
        self.attribute_vocabularies = dict(attribute_vocabularies)
        self.numeric_field_stats = dict(numeric_field_stats)
        self.numeric_fields = list(numeric_fields)
        self.text_transform = text_transform
        self.forms_per_class = forms_per_class
        self.notation_filter = set(notation_filter)
        self.rng = random.Random(seed)
        self.examples: List[Dict[str, object]] = []
        for example in examples:
            forms = self.filtered_forms(example)
            if len(forms) >= 2:
                self.examples.append(example)

    def filtered_forms(self, example: Mapping[str, object]) -> List[Dict[str, str]]:
        forms = example.get("forms", [])
        if not isinstance(forms, list):
            return []
        filtered: List[Dict[str, str]] = []
        for form in forms:
            if not isinstance(form, Mapping):
                continue
            notation = str(form.get("notation", ""))
            if self.notation_filter and notation not in self.notation_filter:
                continue
            text = str(form.get("text", "")).strip()
            if text:
                filtered.append({"text": text, "notation": notation})
        return filtered

    def __len__(self) -> int:
        return len(self.examples)

    def encode_text(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return encode_text_tensors(text, self.max_length, self.text_transform)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        example = self.examples[idx]
        forms = self.filtered_forms(example)
        if len(forms) > self.forms_per_class:
            forms = self.rng.sample(forms, self.forms_per_class)
        attributes = example.get("attributes", {})
        attribute_labels: Dict[str, int] = {}
        numeric_values: List[float] = []
        numeric_mask: List[float] = []
        if isinstance(attributes, Mapping):
            for field, vocabulary in self.attribute_vocabularies.items():
                normalized = normalize_attribute_value(attributes.get(field))
                attribute_labels[field] = vocabulary.get(normalized, -1)
            numeric_values, numeric_mask = encode_numeric_field_values(attributes, self.numeric_field_stats)

        token_ids = []
        masks = []
        for form in forms:
            ids, mask = self.encode_text(form["text"])
            token_ids.append(ids)
            masks.append(mask)

        return {
            "ids": torch.stack(token_ids),
            "mask": torch.stack(masks),
            "fact_id": str(example["fact_id"]),
            "form_count": len(forms),
            "attribute_labels": attribute_labels,
            "numeric_values": torch.tensor(numeric_values, dtype=torch.float),
            "numeric_mask": torch.tensor(numeric_mask, dtype=torch.float),
            "numeric_field_names": list(self.numeric_fields),
        }


class HardRetrievalDataset(Dataset):
    def __init__(self, records: Sequence[Dict[str, object]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.records[idx]


def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    attribute_fields = sorted({field for item in batch for field in item["attribute_labels"].keys()})
    return {
        "left_ids": torch.stack([item["left_ids"] for item in batch]),
        "left_mask": torch.stack([item["left_mask"] for item in batch]),
        "right_ids": torch.stack([item["right_ids"] for item in batch]),
        "right_mask": torch.stack([item["right_mask"] for item in batch]),
        "fact_id": [str(item["fact_id"]) for item in batch],
        "numeric_values": torch.stack([item["numeric_values"] for item in batch]),
        "numeric_mask": torch.stack([item["numeric_mask"] for item in batch]),
        "numeric_field_names": batch[0]["numeric_field_names"] if batch else [],
        "attribute_labels": {
            field: torch.tensor([item["attribute_labels"].get(field, -1) for item in batch], dtype=torch.long)
            for field in attribute_fields
        },
    }


def collate_class_forms(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    attribute_fields = sorted({field for item in batch for field in item["attribute_labels"].keys()})
    ids: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    fact_ids: List[str] = []
    numeric_values: List[torch.Tensor] = []
    numeric_masks: List[torch.Tensor] = []
    attribute_values: Dict[str, List[int]] = {field: [] for field in attribute_fields}
    for item in batch:
        form_count = int(item["form_count"])
        ids.extend(item["ids"][idx] for idx in range(form_count))
        masks.extend(item["mask"][idx] for idx in range(form_count))
        fact_ids.extend([str(item["fact_id"])] * form_count)
        numeric_values.extend([item["numeric_values"]] * form_count)
        numeric_masks.extend([item["numeric_mask"]] * form_count)
        for field in attribute_fields:
            attribute_values[field].extend([item["attribute_labels"].get(field, -1)] * form_count)

    return {
        "ids": torch.stack(ids),
        "mask": torch.stack(masks),
        "fact_id": fact_ids,
        "numeric_values": torch.stack(numeric_values),
        "numeric_mask": torch.stack(numeric_masks),
        "numeric_field_names": batch[0]["numeric_field_names"] if batch else [],
        "attribute_labels": {
            field: torch.tensor(values, dtype=torch.long)
            for field, values in attribute_values.items()
        },
    }


def flatten_hard_record_attributes(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for record in records:
        query = record.get("query", {})
        if isinstance(query, Mapping):
            rows.append({"attributes": query.get("attributes", {})})
        candidates = record.get("candidates", [])
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, Mapping):
                    rows.append({"attributes": candidate.get("attributes", {})})
    return rows


def split_rows(rows: List[Dict[str, object]], valid_fraction: float, seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    valid_size = max(1, int(len(shuffled) * valid_fraction))
    return shuffled[valid_size:], shuffled[:valid_size]


def positive_mask_from_fact_ids(fact_ids: Sequence[str], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[left_fact_id == right_fact_id for right_fact_id in fact_ids] for left_fact_id in fact_ids],
        dtype=torch.float,
        device=device,
    )


def contrastive_loss(
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    temperature: float,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    logits = left_embeddings @ right_embeddings.T / temperature
    left_log_probs = F.log_softmax(logits, dim=1)
    right_log_probs = F.log_softmax(logits.T, dim=1)
    positives_per_row = positive_mask.sum(dim=1).clamp_min(1.0)
    positives_per_col = positive_mask.sum(dim=0).clamp_min(1.0)
    left_loss = -((left_log_probs * positive_mask).sum(dim=1) / positives_per_row).mean()
    right_loss = -((right_log_probs * positive_mask.T).sum(dim=1) / positives_per_col).mean()
    return (left_loss + right_loss) / 2.0


def supervised_contrastive_loss(embeddings: torch.Tensor, temperature: float, positive_mask: torch.Tensor) -> torch.Tensor:
    logits = embeddings @ embeddings.T / temperature
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(self_mask, -1e4)
    positive_mask = positive_mask.masked_fill(self_mask, 0.0)
    positives_per_row = positive_mask.sum(dim=1)
    valid_rows = positives_per_row > 0
    if int(valid_rows.sum().item()) == 0:
        return torch.tensor(0.0, device=embeddings.device)
    log_probs = F.log_softmax(logits, dim=1)
    row_losses = -((log_probs * positive_mask).sum(dim=1) / positives_per_row.clamp_min(1.0))
    return row_losses[valid_rows].mean()


def byte_alignment_teacher_probs(
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    exact_weight: float = 2.0,
    digit_weight: float = 1.0,
) -> torch.Tensor:
    """Build a deterministic teacher distribution over target character positions.

    The teacher assigns mass to exact byte matches and a smaller mass to any
    digit-to-digit match. Rows with no match fall back to a uniform distribution
    over valid target positions, so every source character has a defined target
    attention distribution.
    """
    exact = (source_ids.unsqueeze(2) == target_ids.unsqueeze(1)).float()
    source_digits = ((source_ids >= ord("0") + 1) & (source_ids <= ord("9") + 1)).unsqueeze(2)
    target_digits = ((target_ids >= ord("0") + 1) & (target_ids <= ord("9") + 1)).unsqueeze(1)
    digit_matches = (source_digits & target_digits).float()

    target_valid = target_mask.unsqueeze(1)
    weights = (exact_weight * exact + digit_weight * digit_matches) * target_valid
    row_sums = weights.sum(dim=2, keepdim=True)
    target_counts = target_valid.sum(dim=2, keepdim=True).clamp_min(1.0)
    uniform = target_valid / target_counts
    teacher = torch.where(row_sums > 0, weights / row_sums.clamp_min(1e-8), uniform)
    return teacher * source_mask.unsqueeze(2)


def neural_teacher_attention_probs(
    teacher: "ContrastiveStudent",
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Build a *learned* teacher distribution over target character positions.

    A frozen, pretrained student encoder replaces the deterministic byte/digit
    rule: the teacher's own token-level cross-form attention
    ``softmax(f_src . f_tgt^T / (sqrt(d) * tau))`` becomes the KL target the
    (smaller/cheaper) student is trained to match. Same tensor shape and byte
    grid as ``byte_alignment_teacher_probs`` so it is a drop-in target for
    ``attention_kl_loss``. This is the neural-teacher form of the namesake
    attention-distillation method (vs. the hand-coded byte/digit teacher).
    """
    with torch.no_grad():
        source_features = teacher.token_features(source_ids, source_mask)
        target_features = teacher.token_features(target_ids, target_mask)
        feature_dim = max(1, source_features.shape[-1])
        scale = math.sqrt(feature_dim) * max(temperature, 1e-6)
        logits = source_features @ target_features.transpose(1, 2) / scale
        logits = logits.masked_fill(target_mask.unsqueeze(1) == 0, -1e4)
        probs = F.softmax(logits, dim=2)
        return probs * source_mask.unsqueeze(2)


def load_teacher_model(checkpoint_path: "os.PathLike[str] | str", device: torch.device) -> "ContrastiveStudent":
    """Reconstruct a frozen teacher ``ContrastiveStudent`` from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = dict(checkpoint["args"])
    namespace = argparse.Namespace(**saved_args)
    attribute_vocabularies = checkpoint.get("attribute_vocabularies", {})
    numeric_field_stats = checkpoint.get("numeric_field_stats", {})
    numeric_fields = sorted(numeric_field_stats)
    numeric_feature_encoder = None
    if numeric_field_stats and getattr(namespace, "numeric_feature_mode", "none") != "none":
        numeric_feature_encoder = NumericFeatureEncoder(
            input_dim=len(numeric_field_stats),
            projection_dim=namespace.projection_dim,
            mode=namespace.numeric_feature_mode,
            fourier_dim=getattr(namespace, "numeric_fourier_dim", 16),
        )
    model = ContrastiveStudent(
        build_encoder(namespace),
        namespace.projection_dim,
        attribute_vocabularies,
        numeric_target_fields=numeric_fields,
        numeric_feature_encoder=numeric_feature_encoder,
        text_weight=getattr(namespace, "text_weight", 1.0),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def attention_kl_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_ids: torch.Tensor,
    target_ids: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    exact_weight: float,
    digit_weight: float,
    teacher_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    if teacher_probs is None:
        teacher = byte_alignment_teacher_probs(
            source_ids=source_ids,
            target_ids=target_ids,
            source_mask=source_mask,
            target_mask=target_mask,
            exact_weight=exact_weight,
            digit_weight=digit_weight,
        )
    else:
        teacher = teacher_probs
    feature_dim = max(1, source_features.shape[-1])
    logits = source_features @ target_features.transpose(1, 2) / math.sqrt(feature_dim)
    logits = logits.masked_fill(target_mask.unsqueeze(1) == 0, -1e4)
    log_probs = F.log_softmax(logits, dim=2)
    row_losses = F.kl_div(log_probs, teacher, reduction="none").sum(dim=2)
    valid_rows = source_mask > 0
    if int(valid_rows.sum().item()) == 0:
        return torch.tensor(0.0, device=source_features.device)
    return row_losses[valid_rows].mean()


def pair_attention_distillation_loss(
    model: ContrastiveStudent,
    left_ids: torch.Tensor,
    left_mask: torch.Tensor,
    right_ids: torch.Tensor,
    right_mask: torch.Tensor,
    exact_weight: float,
    digit_weight: float,
    teacher_model: "ContrastiveStudent | None" = None,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    left_features = model.token_features(left_ids, left_mask)
    right_features = model.token_features(right_ids, right_mask)
    lr_teacher = rl_teacher = None
    if teacher_model is not None:
        lr_teacher = neural_teacher_attention_probs(
            teacher_model, left_ids, right_ids, left_mask, right_mask, teacher_temperature
        )
        rl_teacher = neural_teacher_attention_probs(
            teacher_model, right_ids, left_ids, right_mask, left_mask, teacher_temperature
        )
    left_to_right = attention_kl_loss(
        left_features,
        right_features,
        left_ids,
        right_ids,
        left_mask,
        right_mask,
        exact_weight,
        digit_weight,
        teacher_probs=lr_teacher,
    )
    right_to_left = attention_kl_loss(
        right_features,
        left_features,
        right_ids,
        left_ids,
        right_mask,
        left_mask,
        exact_weight,
        digit_weight,
        teacher_probs=rl_teacher,
    )
    return (left_to_right + right_to_left) / 2.0


def class_attention_distillation_loss(
    model: ContrastiveStudent,
    token_ids: torch.Tensor,
    mask: torch.Tensor,
    fact_ids: Sequence[str],
    exact_weight: float,
    digit_weight: float,
    teacher_model: "ContrastiveStudent | None" = None,
    teacher_temperature: float = 1.0,
) -> torch.Tensor:
    grouped_indices: Dict[str, List[int]] = {}
    for index, fact_id in enumerate(fact_ids):
        grouped_indices.setdefault(fact_id, []).append(index)

    left_indices: List[int] = []
    right_indices: List[int] = []
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        anchor = indices[0]
        for other in indices[1:]:
            left_indices.append(anchor)
            right_indices.append(other)

    if not left_indices:
        return torch.tensor(0.0, device=token_ids.device)

    left = torch.tensor(left_indices, dtype=torch.long, device=token_ids.device)
    right = torch.tensor(right_indices, dtype=torch.long, device=token_ids.device)
    return pair_attention_distillation_loss(
        model=model,
        left_ids=token_ids.index_select(0, left),
        left_mask=mask.index_select(0, left),
        right_ids=token_ids.index_select(0, right),
        right_mask=mask.index_select(0, right),
        exact_weight=exact_weight,
        digit_weight=digit_weight,
        teacher_model=teacher_model,
        teacher_temperature=teacher_temperature,
    )


def attribute_loss(
    model: ContrastiveStudent,
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    attribute_labels: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not model.attribute_heads:
        return torch.tensor(0.0, device=device), {}

    left_logits = model.attribute_logits(left_embeddings)
    right_logits = model.attribute_logits(right_embeddings)
    losses: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}

    for field, labels in attribute_labels.items():
        if field not in left_logits:
            continue
        field_labels = labels.to(device)
        valid_mask = field_labels >= 0
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            continue
        left_loss = F.cross_entropy(left_logits[field][valid_mask], field_labels[valid_mask])
        right_loss = F.cross_entropy(right_logits[field][valid_mask], field_labels[valid_mask])
        losses.append((left_loss + right_loss) / 2.0)
        left_predictions = left_logits[field][valid_mask].argmax(dim=1)
        right_predictions = right_logits[field][valid_mask].argmax(dim=1)
        metrics[f"attr_{field}_left_acc"] = float((left_predictions == field_labels[valid_mask]).float().mean().item())
        metrics[f"attr_{field}_right_acc"] = float((right_predictions == field_labels[valid_mask]).float().mean().item())

    if not losses:
        return torch.tensor(0.0, device=device), metrics
    return torch.stack(losses).mean(), metrics


def form_attribute_loss(
    model: ContrastiveStudent,
    embeddings: torch.Tensor,
    attribute_labels: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not model.attribute_heads:
        return torch.tensor(0.0, device=device), {}

    logits_by_field = model.attribute_logits(embeddings)
    losses: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}
    for field, labels in attribute_labels.items():
        if field not in logits_by_field:
            continue
        field_labels = labels.to(device)
        valid_mask = field_labels >= 0
        if int(valid_mask.sum().item()) == 0:
            continue
        logits = logits_by_field[field][valid_mask]
        labels_present = field_labels[valid_mask]
        losses.append(F.cross_entropy(logits, labels_present))
        predictions = logits.argmax(dim=1)
        metrics[f"attr_{field}_acc"] = float((predictions == labels_present).float().mean().item())
    if not losses:
        return torch.tensor(0.0, device=device), metrics
    return torch.stack(losses).mean(), metrics


def numeric_consistency_loss(
    model: ContrastiveStudent,
    left_embeddings: torch.Tensor,
    right_embeddings: torch.Tensor,
    numeric_values: torch.Tensor,
    numeric_mask: torch.Tensor,
    numeric_field_names: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not model.numeric_heads or not numeric_field_names:
        return torch.tensor(0.0, device=device), {}

    losses: List[torch.Tensor] = []
    metrics: Dict[str, float] = {}
    for field_index, field_name in enumerate(numeric_field_names):
        if field_name not in model.numeric_heads:
            continue
        target = numeric_values[:, field_index]
        present = numeric_mask[:, field_index] > 0
        if int(present.sum().item()) == 0:
            continue
        left_pred = model.numeric_heads[field_name](left_embeddings).squeeze(-1)
        right_pred = model.numeric_heads[field_name](right_embeddings).squeeze(-1)
        target_present = target[present]
        left_present = left_pred[present]
        right_present = right_pred[present]
        left_loss = F.mse_loss(left_present, target_present)
        right_loss = F.mse_loss(right_present, target_present)
        pair_loss = F.mse_loss(left_present, right_present)
        losses.append((left_loss + right_loss + pair_loss) / 3.0)
        metrics[f"num_{field_name}_left_mae"] = float(torch.abs(left_present - target_present).mean().item())
        metrics[f"num_{field_name}_right_mae"] = float(torch.abs(right_present - target_present).mean().item())

    if not losses:
        return torch.tensor(0.0, device=device), metrics
    return torch.stack(losses).mean(), metrics


@torch.no_grad()
def evaluate(
    model: ContrastiveStudent,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
    attribute_loss_weight: float,
    numeric_loss_weight: float,
    attention_distillation_weight: float,
    attention_teacher_exact_weight: float,
    attention_teacher_digit_weight: float,
    teacher_model: "ContrastiveStudent | None" = None,
    teacher_temperature: float = 1.0,
) -> Dict[str, float]:
    model.eval()
    total = 0
    top1 = 0
    top5 = 0
    losses: List[float] = []
    attention_losses: List[float] = []
    attribute_metric_history: Dict[str, List[float]] = {}
    numeric_metric_history: Dict[str, List[float]] = {}

    for batch in loader:
        left_ids = batch["left_ids"].to(device)
        left_mask = batch["left_mask"].to(device)
        right_ids = batch["right_ids"].to(device)
        right_mask = batch["right_mask"].to(device)
        numeric_values = batch["numeric_values"].to(device)
        numeric_mask = batch["numeric_mask"].to(device)
        left_embeddings = model.encode(left_ids, left_mask, numeric_values, numeric_mask)
        right_embeddings = model.encode(right_ids, right_mask, numeric_values, numeric_mask)
        logits = left_embeddings @ right_embeddings.T / temperature
        positive_mask = positive_mask_from_fact_ids(batch["fact_id"], device)
        batch_contrastive_loss = contrastive_loss(left_embeddings, right_embeddings, temperature, positive_mask)
        batch_attribute_loss, batch_attribute_metrics = attribute_loss(
            model,
            left_embeddings,
            right_embeddings,
            batch["attribute_labels"],
            device,
        )
        batch_numeric_loss, batch_numeric_metrics = numeric_consistency_loss(
            model,
            left_embeddings,
            right_embeddings,
            numeric_values,
            numeric_mask,
            batch["numeric_field_names"],
            device,
        )
        if attention_distillation_weight > 0:
            batch_attention_loss = pair_attention_distillation_loss(
                model=model,
                left_ids=left_ids,
                left_mask=left_mask,
                right_ids=right_ids,
                right_mask=right_mask,
                exact_weight=attention_teacher_exact_weight,
                digit_weight=attention_teacher_digit_weight,
                teacher_model=teacher_model,
                teacher_temperature=teacher_temperature,
            )
            attention_losses.append(float(batch_attention_loss.cpu()))
        else:
            batch_attention_loss = torch.tensor(0.0, device=device)
        losses.append(
            float(
                (
                    batch_contrastive_loss
                    + attribute_loss_weight * batch_attribute_loss
                    + numeric_loss_weight * batch_numeric_loss
                    + attention_distillation_weight * batch_attention_loss
                ).cpu()
            )
        )
        for key, value in batch_attribute_metrics.items():
            attribute_metric_history.setdefault(key, []).append(value)
        for key, value in batch_numeric_metrics.items():
            numeric_metric_history.setdefault(key, []).append(value)

        ranking = torch.argsort(logits, dim=1, descending=True)
        ranked_positive_mask = positive_mask.gather(dim=1, index=ranking)
        total += logits.shape[0]
        top1 += int(ranked_positive_mask[:, 0].sum().cpu())
        top5 += int(ranked_positive_mask[:, : min(5, ranking.shape[1])].any(dim=1).sum().cpu())

    metrics = {
        "loss": sum(losses) / max(1, len(losses)),
        "attention_distillation_loss": sum(attention_losses) / max(1, len(attention_losses)),
        "top1": top1 / max(1, total),
        "top5": top5 / max(1, total),
        "pairs": total,
    }
    for key, values in attribute_metric_history.items():
        metrics[key] = sum(values) / len(values)
    for key, values in numeric_metric_history.items():
        metrics[key] = sum(values) / len(values)
    return metrics


def reciprocal_rank(labels: Sequence[bool], ranking: Sequence[int]) -> float:
    for rank_idx, candidate_idx in enumerate(ranking, start=1):
        if labels[candidate_idx]:
            return 1.0 / rank_idx
    return 0.0


def encode_hard_forms(
    model: ContrastiveStudent,
    texts: Sequence[str],
    attributes: Sequence[Mapping[str, object]],
    max_length: int,
    text_transform: str,
    numeric_field_stats: Mapping[str, Mapping[str, float]],
    device: torch.device,
) -> torch.Tensor:
    encoded = [encode_text_tensors(text, max_length, text_transform) for text in texts]
    token_ids = torch.stack([item[0] for item in encoded]).to(device)
    mask = torch.stack([item[1] for item in encoded]).to(device)
    numeric_values = None
    numeric_mask = None
    if numeric_field_stats:
        encoded_numeric = [encode_numeric_field_values(item, numeric_field_stats) for item in attributes]
        numeric_values = torch.tensor([item[0] for item in encoded_numeric], dtype=torch.float, device=device)
        numeric_mask = torch.tensor([item[1] for item in encoded_numeric], dtype=torch.float, device=device)
    return model.encode(token_ids, mask, numeric_values, numeric_mask)


def hard_retrieval_training_step(
    model: ContrastiveStudent,
    records: Sequence[Dict[str, Any]],
    device: torch.device,
    max_length: int,
    text_transform: str,
    numeric_field_stats: Mapping[str, Mapping[str, float]],
    temperature: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    losses: List[torch.Tensor] = []
    top1 = 0
    top5 = 0
    reciprocal_ranks: List[float] = []

    for record in records:
        query = record["query"]
        candidates = record["candidates"]
        texts = [str(query["text"])] + [str(candidate["text"]) for candidate in candidates]
        attributes = [
            query.get("attributes", {}) if isinstance(query.get("attributes", {}), Mapping) else {},
            *[
                candidate.get("attributes", {}) if isinstance(candidate.get("attributes", {}), Mapping) else {}
                for candidate in candidates
            ],
        ]
        labels = torch.tensor([bool(candidate.get("is_positive")) for candidate in candidates], dtype=torch.bool, device=device)
        if int(labels.sum().item()) == 0:
            continue

        embeddings = encode_hard_forms(
            model=model,
            texts=texts,
            attributes=attributes,
            max_length=max_length,
            text_transform=text_transform,
            numeric_field_stats=numeric_field_stats,
            device=device,
        )
        scores = (embeddings[:1] @ embeddings[1:].T).squeeze(0) / temperature
        loss = -(torch.logsumexp(scores[labels], dim=0) - torch.logsumexp(scores, dim=0))
        losses.append(loss)

        ranking = torch.argsort(scores.detach(), descending=True).cpu().tolist()
        label_list = labels.detach().cpu().tolist()
        reciprocal_ranks.append(reciprocal_rank(label_list, ranking))
        if ranking and label_list[ranking[0]]:
            top1 += 1
        if any(label_list[idx] for idx in ranking[:5]):
            top5 += 1

    if not losses:
        return torch.tensor(0.0, device=device), {
            "hard_train_top1": 0.0,
            "hard_train_top5": 0.0,
            "hard_train_mean_reciprocal_rank": 0.0,
        }

    query_count = len(losses)
    return torch.stack(losses).mean(), {
        "hard_train_top1": top1 / query_count,
        "hard_train_top5": top5 / query_count,
        "hard_train_mean_reciprocal_rank": sum(reciprocal_ranks) / query_count,
    }


@torch.no_grad()
def evaluate_hard_retrieval(
    model: ContrastiveStudent,
    records: Sequence[Dict[str, Any]],
    device: torch.device,
    max_length: int,
    text_transform: str,
    numeric_field_stats: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    if not records:
        return {
            "top1": 0.0,
            "top5": 0.0,
            "mean_reciprocal_rank": 0.0,
            "queries": 0.0,
            "avg_candidates": 0.0,
        }

    model.eval()
    top1 = 0
    top5 = 0
    reciprocal_ranks: List[float] = []
    candidate_counts: List[int] = []

    for record in records:
        query = record["query"]
        texts = [str(query["text"])] + [str(candidate["text"]) for candidate in record["candidates"]]
        attributes = [
            query.get("attributes", {}) if isinstance(query.get("attributes", {}), Mapping) else {},
            *[
                candidate.get("attributes", {}) if isinstance(candidate.get("attributes", {}), Mapping) else {}
                for candidate in record["candidates"]
            ],
        ]
        embeddings = encode_hard_forms(
            model=model,
            texts=texts,
            attributes=attributes,
            max_length=max_length,
            text_transform=text_transform,
            numeric_field_stats=numeric_field_stats,
            device=device,
        )
        query_embedding = embeddings[:1]
        candidate_matrix = embeddings[1:]
        scores = (query_embedding @ candidate_matrix.T).squeeze(0)
        ranking = torch.argsort(scores, descending=True).cpu().tolist()
        labels = [bool(candidate.get("is_positive")) for candidate in record["candidates"]]

        reciprocal_ranks.append(reciprocal_rank(labels, ranking))
        if ranking and labels[ranking[0]]:
            top1 += 1
        if any(labels[idx] for idx in ranking[:5]):
            top5 += 1
        candidate_counts.append(len(record["candidates"]))

    query_count = len(records)
    return {
        "top1": top1 / query_count,
        "top5": top5 / query_count,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / query_count,
        "queries": float(query_count),
        "avg_candidates": sum(candidate_counts) / query_count,
    }


def build_encoder(args: argparse.Namespace) -> nn.Module:
    if args.encoder == "mean":
        return CharMeanEncoder(hidden_dim=args.hidden_dim, projection_dim=args.projection_dim)
    if args.encoder == "cnn":
        kernel_sizes = tuple(int(value) for value in args.kernel_sizes.split(",") if value.strip())
        return CharCNNEncoder(
            hidden_dim=args.hidden_dim,
            projection_dim=args.projection_dim,
            kernel_sizes=kernel_sizes,
            dropout=args.dropout,
        )
    if args.encoder == "transformer":
        return CharTransformerEncoder(
            hidden_dim=args.hidden_dim,
            projection_dim=args.projection_dim,
            num_layers=args.transformer_layers,
            num_heads=args.transformer_heads,
            max_length=args.max_length,
            dropout=args.dropout,
        )
    raise ValueError(f"Unknown encoder: {args.encoder}")


def train(args: argparse.Namespace) -> Dict[str, object]:
    active_inputs = [value is not None for value in (args.input, args.class_input, args.hard_train_input)]
    if sum(active_inputs) != 1:
        raise ValueError("Use exactly one of --input, --class-input, or --hard-train-input")

    hard_valid_records: List[Dict[str, Any]] = []
    if args.hard_valid_input is not None:
        hard_valid_records = read_jsonl(args.hard_valid_input)
        if args.max_hard_valid_queries is not None:
            hard_valid_records = hard_valid_records[: args.max_hard_valid_queries]
    if args.hard_train_input is not None and args.attention_distillation_weight > 0:
        raise ValueError("--attention-distillation-weight is supported for pair and class training, not --hard-train-input")

    class_train_examples: List[Dict[str, object]] = []
    hard_train_records: List[Dict[str, object]] = []
    train_rows: List[Dict[str, object]] = []
    valid_rows: List[Dict[str, object]] = []
    class_notations = [notation.strip() for notation in args.class_notations.split(",") if notation.strip()]
    if args.class_input is not None:
        class_train_examples = read_jsonl(args.class_input)
        if args.max_classes is not None:
            class_train_examples = class_train_examples[: args.max_classes]
        if args.valid_input is not None:
            valid_rows = read_jsonl(args.valid_input)
            if args.max_valid_pairs is not None:
                valid_rows = valid_rows[: args.max_valid_pairs]
        else:
            raise ValueError("--valid-input is currently required for class training so pair metrics remain comparable")
    elif args.hard_train_input is not None:
        hard_train_records = read_jsonl(args.hard_train_input)
        if args.max_hard_train_queries is not None:
            hard_train_records = hard_train_records[: args.max_hard_train_queries]
        if args.valid_input is not None:
            valid_rows = read_jsonl(args.valid_input)
            if args.max_valid_pairs is not None:
                valid_rows = valid_rows[: args.max_valid_pairs]
        else:
            raise ValueError("--valid-input is currently required for hard-negative training so pair metrics remain comparable")
    else:
        rows = read_jsonl(args.input)
        if args.max_pairs is not None:
            rows = rows[: args.max_pairs]
        if args.valid_input is not None:
            train_rows = rows
            valid_rows = read_jsonl(args.valid_input)
            if args.max_valid_pairs is not None:
                valid_rows = valid_rows[: args.max_valid_pairs]
        else:
            train_rows, valid_rows = split_rows(rows, args.valid_fraction, args.seed)

    if hard_train_records:
        stats_rows = flatten_hard_record_attributes(hard_train_records)
    else:
        stats_rows = class_train_examples if class_train_examples else train_rows
    attribute_fields = [field.strip() for field in args.attribute_fields.split(",") if field.strip()]
    attribute_vocabularies = build_attribute_vocabularies(stats_rows, attribute_fields)
    numeric_fields = [field.strip() for field in args.numeric_fields.split(",") if field.strip()]
    numeric_field_stats = build_numeric_field_stats(stats_rows, numeric_fields)
    effective_numeric_fields = validate_numeric_fields(numeric_fields, numeric_field_stats)

    if hard_train_records:
        train_loader = DataLoader(
            HardRetrievalDataset(hard_train_records),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda batch: batch,
        )
    elif class_train_examples:
        if args.forms_per_class < 2:
            raise ValueError("--forms-per-class must be at least 2")
        train_dataset = ClassFormDataset(
            class_train_examples,
            args.max_length,
            attribute_vocabularies,
            numeric_field_stats,
            effective_numeric_fields,
            args.text_transform,
            forms_per_class=args.forms_per_class,
            notation_filter=class_notations,
            seed=args.seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_class_forms,
        )
    else:
        train_loader = DataLoader(
            PairDataset(train_rows, args.max_length, attribute_vocabularies, numeric_field_stats, effective_numeric_fields, args.text_transform),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
    valid_loader = DataLoader(
        PairDataset(valid_rows, args.max_length, attribute_vocabularies, numeric_field_stats, effective_numeric_fields, args.text_transform),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    numeric_feature_encoder = None
    if numeric_field_stats and args.numeric_feature_mode != "none":
        numeric_feature_encoder = NumericFeatureEncoder(
            input_dim=len(numeric_field_stats),
            projection_dim=args.projection_dim,
            mode=args.numeric_feature_mode,
            fourier_dim=args.numeric_fourier_dim,
        )
    model = ContrastiveStudent(
        build_encoder(args),
        args.projection_dim,
        attribute_vocabularies,
        numeric_target_fields=effective_numeric_fields,
        numeric_feature_encoder=numeric_feature_encoder,
        text_weight=args.text_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    teacher_model = None
    if getattr(args, "attention_teacher_checkpoint", None) is not None:
        if args.attention_distillation_weight <= 0:
            raise ValueError("--attention-teacher-checkpoint requires --attention-distillation-weight > 0")
        teacher_model = load_teacher_model(args.attention_teacher_checkpoint, device)

    history: List[Dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: List[float] = []
        attention_losses: List[float] = []
        attribute_metric_history: Dict[str, List[float]] = {}
        numeric_metric_history: Dict[str, List[float]] = {}
        for batch in train_loader:
            if hard_train_records:
                loss, hard_train_metrics = hard_retrieval_training_step(
                    model=model,
                    records=batch,
                    device=device,
                    max_length=args.max_length,
                    text_transform=args.text_transform,
                    numeric_field_stats=numeric_field_stats,
                    temperature=args.temperature,
                )
                batch_attribute_metrics = {}
                batch_numeric_metrics = {}
                batch_attention_loss = torch.tensor(0.0, device=device)
                for key, value in hard_train_metrics.items():
                    attribute_metric_history.setdefault(key, []).append(value)
            else:
                numeric_values = batch["numeric_values"].to(device)
                numeric_mask = batch["numeric_mask"].to(device)
                if class_train_examples:
                    ids = batch["ids"].to(device)
                    mask = batch["mask"].to(device)
                    embeddings = model.encode(ids, mask, numeric_values, numeric_mask)
                    positive_mask = positive_mask_from_fact_ids(batch["fact_id"], device)
                    batch_contrastive_loss = supervised_contrastive_loss(embeddings, args.temperature, positive_mask)
                    batch_attribute_loss, batch_attribute_metrics = form_attribute_loss(
                        model,
                        embeddings,
                        batch["attribute_labels"],
                        device,
                    )
                    batch_numeric_loss, batch_numeric_metrics = torch.tensor(0.0, device=device), {}
                    if args.attention_distillation_weight > 0:
                        batch_attention_loss = class_attention_distillation_loss(
                            model=model,
                            token_ids=ids,
                            mask=mask,
                            fact_ids=batch["fact_id"],
                            exact_weight=args.attention_teacher_exact_weight,
                            digit_weight=args.attention_teacher_digit_weight,
                            teacher_model=teacher_model,
                            teacher_temperature=args.attention_teacher_temperature,
                        )
                    else:
                        batch_attention_loss = torch.tensor(0.0, device=device)
                else:
                    left_ids = batch["left_ids"].to(device)
                    left_mask = batch["left_mask"].to(device)
                    right_ids = batch["right_ids"].to(device)
                    right_mask = batch["right_mask"].to(device)
                    left_embeddings = model.encode(left_ids, left_mask, numeric_values, numeric_mask)
                    right_embeddings = model.encode(right_ids, right_mask, numeric_values, numeric_mask)
                    positive_mask = positive_mask_from_fact_ids(batch["fact_id"], device)
                    batch_contrastive_loss = contrastive_loss(left_embeddings, right_embeddings, args.temperature, positive_mask)
                    batch_attribute_loss, batch_attribute_metrics = attribute_loss(
                        model,
                        left_embeddings,
                        right_embeddings,
                        batch["attribute_labels"],
                        device,
                    )
                    batch_numeric_loss, batch_numeric_metrics = numeric_consistency_loss(
                        model,
                        left_embeddings,
                        right_embeddings,
                        numeric_values,
                        numeric_mask,
                        batch["numeric_field_names"],
                        device,
                    )
                    if args.attention_distillation_weight > 0:
                        batch_attention_loss = pair_attention_distillation_loss(
                            model=model,
                            left_ids=left_ids,
                            left_mask=left_mask,
                            right_ids=right_ids,
                            right_mask=right_mask,
                            exact_weight=args.attention_teacher_exact_weight,
                            digit_weight=args.attention_teacher_digit_weight,
                            teacher_model=teacher_model,
                            teacher_temperature=args.attention_teacher_temperature,
                        )
                    else:
                        batch_attention_loss = torch.tensor(0.0, device=device)
                loss = (
                    batch_contrastive_loss
                    + args.attribute_loss_weight * batch_attribute_loss
                    + args.numeric_loss_weight * batch_numeric_loss
                    + args.attention_distillation_weight * batch_attention_loss
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if args.attention_distillation_weight > 0:
                attention_losses.append(float(batch_attention_loss.detach().cpu()))
            for key, value in batch_attribute_metrics.items():
                attribute_metric_history.setdefault(key, []).append(value)
            for key, value in batch_numeric_metrics.items():
                numeric_metric_history.setdefault(key, []).append(value)

        valid_metrics = evaluate(
            model,
            valid_loader,
            device,
            args.temperature,
            args.attribute_loss_weight,
            args.numeric_loss_weight,
            args.attention_distillation_weight,
            args.attention_teacher_exact_weight,
            args.attention_teacher_digit_weight,
            teacher_model,
            args.attention_teacher_temperature,
        )
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": sum(losses) / max(1, len(losses)),
            "train_attention_distillation_loss": sum(attention_losses) / max(1, len(attention_losses)),
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        if hard_valid_records:
            hard_valid_metrics = evaluate_hard_retrieval(
                model=model,
                records=hard_valid_records,
                device=device,
                max_length=args.max_length,
                text_transform=args.text_transform,
                numeric_field_stats=numeric_field_stats,
            )
            epoch_metrics.update({f"hard_valid_{key}": value for key, value in hard_valid_metrics.items()})
        for key, values in attribute_metric_history.items():
            epoch_metrics[f"train_{key}"] = sum(values) / len(values)
        for key, values in numeric_metric_history.items():
            epoch_metrics[f"train_{key}"] = sum(values) / len(values)
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, sort_keys=True))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"char_{args.encoder}_student.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "history": history,
            "attribute_vocabularies": attribute_vocabularies,
            "numeric_field_stats": numeric_field_stats,
        },
        model_path,
    )

    result = {
        "model_path": str(model_path),
        "train_pairs": len(train_rows),
        "train_classes": len(class_train_examples),
        "hard_train_queries": len(hard_train_records),
        "valid_pairs": len(valid_rows),
        "hard_valid_queries": len(hard_valid_records),
        "hard_valid_input": str(args.hard_valid_input) if args.hard_valid_input is not None else None,
        "attribute_fields": sorted(attribute_vocabularies),
        "attribute_vocabularies": attribute_vocabularies,
        "numeric_fields": sorted(numeric_field_stats),
        "numeric_field_stats": numeric_field_stats,
        "attention_distillation_weight": args.attention_distillation_weight,
        "attention_teacher_exact_weight": args.attention_teacher_exact_weight,
        "attention_teacher_digit_weight": args.attention_teacher_digit_weight,
        "attention_teacher_checkpoint": (
            str(args.attention_teacher_checkpoint)
            if getattr(args, "attention_teacher_checkpoint", None) is not None
            else None
        ),
        "attention_teacher_kind": (
            "neural" if getattr(args, "attention_teacher_checkpoint", None) is not None else "byte_rule"
        ),
        "attention_teacher_temperature": getattr(args, "attention_teacher_temperature", 1.0),
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight BioXRep contrastive student.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--class-input", type=Path, default=None)
    parser.add_argument("--hard-train-input", type=Path, default=None)
    parser.add_argument("--class-notations", default="")
    parser.add_argument("--forms-per-class", type=int, default=4)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/contrastive_student"))
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-hard-train-queries", type=int, default=None)
    parser.add_argument("--valid-input", type=Path, default=None)
    parser.add_argument("--max-valid-pairs", type=int, default=None)
    parser.add_argument("--hard-valid-input", type=Path, default=None)
    parser.add_argument("--max-hard-valid-queries", type=int, default=None)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--encoder", choices=["mean", "cnn", "transformer"], default="mean")
    parser.add_argument("--kernel-sizes", default="3,5,7")
    parser.add_argument("--transformer-layers", type=int, default=2,
                        help="Number of self-attention layers (encoder=transformer).")
    parser.add_argument("--transformer-heads", type=int, default=4,
                        help="Number of attention heads (encoder=transformer).")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--attribute-fields", default="")
    parser.add_argument("--attribute-loss-weight", type=float, default=0.0)
    parser.add_argument("--numeric-fields", default="")
    parser.add_argument("--numeric-feature-mode", choices=["none", "explicit", "sinusoidal", "xval"], default="none")
    parser.add_argument("--numeric-fourier-dim", type=int, default=16)
    parser.add_argument("--numeric-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--attention-distillation-weight",
        type=float,
        default=0.0,
        help="Weight for the deterministic byte/digit alignment attention-distillation loss.",
    )
    parser.add_argument(
        "--attention-teacher-exact-weight",
        type=float,
        default=2.0,
        help="Teacher mass assigned to exact byte matches.",
    )
    parser.add_argument(
        "--attention-teacher-digit-weight",
        type=float,
        default=1.0,
        help="Teacher mass assigned to digit-to-digit matches.",
    )
    parser.add_argument(
        "--attention-teacher-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to a saved ContrastiveStudent checkpoint to use as a NEURAL "
            "attention-distillation teacher. When set, the frozen teacher's learned "
            "cross-form attention replaces the deterministic byte/digit teacher as the "
            "KL target. When unset, the deterministic byte/digit teacher is used."
        ),
    )
    parser.add_argument(
        "--attention-teacher-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for the neural teacher's attention distribution.",
    )
    parser.add_argument("--text-transform", choices=["none", "mask_digits", "strip_digits"], default="none")
    parser.add_argument("--text-weight", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args)
    print(json.dumps({"model_path": result["model_path"], "valid_pairs": result["valid_pairs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
