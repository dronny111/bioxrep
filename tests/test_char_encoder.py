from __future__ import annotations

import torch

from bioxrep.models.char_encoder import CharCNNEncoder


def _manual_branch_pool(features_by_position: torch.Tensor, mask: torch.Tensor, widths: list[int]) -> torch.Tensor:
    """Reference masked max-pool that slices strictly at per-branch boundaries."""
    expanded = mask.unsqueeze(1)
    outs = []
    start = 0
    for width in widths:
        chunk = features_by_position[:, start : start + width, :].masked_fill(expanded == 0, -1e4)
        outs.append(chunk.max(dim=2).values)
        start += width
    return torch.cat(outs, dim=1)


def test_charcnn_pools_at_branch_boundaries() -> None:
    """The encoder must pool each conv branch separately, not in fixed-width chunks.

    This guards the previous implementation, which stepped through the concatenated
    feature map in blocks of convs[0].out_channels and would mix channels across
    branches if the branches ever had different widths.
    """
    torch.manual_seed(0)
    enc = CharCNNEncoder(vocab_size=259, hidden_dim=8, projection_dim=16, kernel_sizes=(3, 5, 7))
    enc.eval()
    assert enc.branch_channels == [8, 8, 8]

    ids = torch.randint(1, 259, (3, 10))
    mask = torch.ones(3, 10)
    mask[0, 6:] = 0

    features_by_position = enc.token_features(ids, mask).transpose(1, 2)
    expected_pooled = _manual_branch_pool(features_by_position, mask, enc.branch_channels)
    with torch.no_grad():
        expected = torch.nn.functional.normalize(enc.projection(enc.dropout(expected_pooled)), dim=-1)
        out = enc(ids, mask)

    assert out.shape == (3, 16)
    assert torch.allclose(out, expected, atol=1e-6)
    # Output rows are L2-normalized.
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_charcnn_handles_unequal_branch_widths() -> None:
    """With differing kernel branches the pooled dim still equals the summed channels."""
    enc = CharCNNEncoder(vocab_size=64, hidden_dim=6, projection_dim=12, kernel_sizes=(3, 5))
    ids = torch.randint(1, 64, (2, 8))
    mask = torch.ones(2, 8)
    out = enc(ids, mask)
    assert out.shape == (2, 12)
    assert sum(enc.branch_channels) == enc.projection[0].in_features
