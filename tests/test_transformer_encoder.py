from __future__ import annotations

import torch

from bioxrep.models.char_encoder import CharTransformerEncoder


def test_transformer_output_is_normalized_and_correct_shape() -> None:
    torch.manual_seed(0)
    enc = CharTransformerEncoder(hidden_dim=32, projection_dim=16, num_layers=2, num_heads=4, max_length=32)
    enc.eval()
    ids = torch.randint(1, 257, (3, 12))
    mask = torch.ones(3, 12)
    out = enc(ids, mask)
    assert out.shape == (3, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_transformer_zeroes_padded_token_features() -> None:
    torch.manual_seed(0)
    enc = CharTransformerEncoder(hidden_dim=32, projection_dim=16, num_layers=1, num_heads=2, max_length=32)
    enc.eval()
    ids = torch.randint(1, 257, (2, 10))
    mask = torch.ones(2, 10)
    mask[0, 6:] = 0
    tf = enc.token_features(ids, mask)
    assert tf.shape == (2, 10, 32)
    # Padded positions must carry no signal into the mean pool.
    assert torch.allclose(tf[0, 6:], torch.zeros_like(tf[0, 6:]), atol=1e-6)


def test_transformer_is_invariant_to_trailing_padding() -> None:
    """Appending pad tokens (mask 0) must not change the pooled embedding.

    Padding is excluded via both src_key_padding_mask and the post-hoc mask, so a
    longer padded grid encodes to the same normalized vector as the short one.
    """
    torch.manual_seed(0)
    enc = CharTransformerEncoder(hidden_dim=32, projection_dim=16, num_layers=2, num_heads=4, max_length=64)
    enc.eval()
    ids = torch.randint(1, 257, (3, 15))
    mask = torch.ones(3, 15)
    mask[0, 10:] = 0  # row already has some padding
    ids[0, 10:] = 0
    with torch.no_grad():
        out = enc(ids, mask)
        ids2 = torch.cat([ids, torch.zeros(3, 20, dtype=torch.long)], dim=1)
        mask2 = torch.cat([mask, torch.zeros(3, 20)], dim=1)
        out2 = enc(ids2, mask2)
    assert torch.allclose(out, out2, atol=1e-5)
