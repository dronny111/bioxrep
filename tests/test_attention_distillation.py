from __future__ import annotations

import torch

from bioxrep.models.char_encoder import CharCNNEncoder, CharMeanEncoder
from bioxrep.train.train_contrastive_student import (
    ContrastiveStudent,
    byte_alignment_teacher_probs,
    encode_text_tensors,
    neural_teacher_attention_probs,
    pair_attention_distillation_loss,
)


def test_byte_alignment_teacher_prefers_exact_match() -> None:
    source_ids, source_mask = encode_text_tensors("A12", max_length=4, text_transform="none")
    target_ids, target_mask = encode_text_tensors("BA1", max_length=4, text_transform="none")

    teacher = byte_alignment_teacher_probs(
        source_ids.unsqueeze(0),
        target_ids.unsqueeze(0),
        source_mask.unsqueeze(0),
        target_mask.unsqueeze(0),
    )

    assert torch.isclose(teacher[0, 0, 1], torch.tensor(1.0))
    assert teacher[0, 1, 2] > teacher[0, 1, 0]
    assert torch.isclose(teacher[0, :3].sum(dim=1), torch.ones(3)).all()
    assert float(teacher[0, 3].sum()) == 0.0


def test_attention_distillation_loss_is_finite_for_supported_encoders() -> None:
    left_ids, left_mask = encode_text_tensors("BRAF V600E", max_length=16, text_transform="none")
    right_ids, right_mask = encode_text_tensors("BRAF p.Val600Glu", max_length=16, text_transform="none")

    for encoder in (
        CharMeanEncoder(hidden_dim=8, projection_dim=8),
        CharCNNEncoder(hidden_dim=8, projection_dim=8, kernel_sizes=(3,)),
    ):
        model = ContrastiveStudent(
            encoder=encoder,
            projection_dim=8,
            attribute_vocabularies={},
            numeric_target_fields=[],
        )
        loss = pair_attention_distillation_loss(
            model=model,
            left_ids=left_ids.unsqueeze(0),
            left_mask=left_mask.unsqueeze(0),
            right_ids=right_ids.unsqueeze(0),
            right_mask=right_mask.unsqueeze(0),
            exact_weight=2.0,
            digit_weight=1.0,
        )

        assert torch.isfinite(loss)
        assert float(loss.detach()) >= 0.0


def test_neural_teacher_probs_are_valid_distributions() -> None:
    torch.manual_seed(0)
    teacher = ContrastiveStudent(
        encoder=CharCNNEncoder(hidden_dim=16, projection_dim=16, kernel_sizes=(3,)),
        projection_dim=16,
        attribute_vocabularies={},
        numeric_target_fields=[],
    )
    teacher.eval()
    source_ids, source_mask = encode_text_tensors("BRAF V600E", max_length=24, text_transform="none")
    target_ids, target_mask = encode_text_tensors("BRAF p.Val600Glu", max_length=24, text_transform="none")
    assert int((target_mask == 0).sum()) > 0  # guarantee padded target columns exist

    probs = neural_teacher_attention_probs(
        teacher,
        source_ids.unsqueeze(0),
        target_ids.unsqueeze(0),
        source_mask.unsqueeze(0),
        target_mask.unsqueeze(0),
    )

    # Rows for valid source positions sum to 1; masked source rows sum to 0.
    row_sums = probs.sum(dim=2).squeeze(0)
    valid = source_mask.bool()
    assert torch.allclose(row_sums[valid], torch.ones(int(valid.sum())), atol=1e-4)
    assert float(row_sums[~valid].abs().max()) == 0.0
    # No probability mass may land on padded target positions.
    padded_target_mass = probs[:, :, target_mask == 0]
    assert float(padded_target_mass.abs().max()) == 0.0


def test_neural_teacher_distillation_loss_flows_to_student() -> None:
    torch.manual_seed(0)
    teacher = ContrastiveStudent(
        encoder=CharCNNEncoder(hidden_dim=16, projection_dim=16, kernel_sizes=(3,)),
        projection_dim=16,
        attribute_vocabularies={},
        numeric_target_fields=[],
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student = ContrastiveStudent(
        encoder=CharCNNEncoder(hidden_dim=8, projection_dim=8, kernel_sizes=(3,)),
        projection_dim=8,
        attribute_vocabularies={},
        numeric_target_fields=[],
    )
    left_ids, left_mask = encode_text_tensors("TP53", max_length=12, text_transform="none")
    right_ids, right_mask = encode_text_tensors("tumor protein p53", max_length=12, text_transform="none")

    loss = pair_attention_distillation_loss(
        model=student,
        left_ids=left_ids.unsqueeze(0),
        left_mask=left_mask.unsqueeze(0),
        right_ids=right_ids.unsqueeze(0),
        right_mask=right_mask.unsqueeze(0),
        exact_weight=2.0,
        digit_weight=1.0,
        teacher_model=teacher,
    )

    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None for p in student.parameters())
