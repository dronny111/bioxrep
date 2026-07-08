from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharMeanEncoder(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_dim: int = 128, projection_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.token_features(token_ids, mask)
        masked = features * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        pooled = masked.sum(dim=1) / lengths
        projected = self.projection(pooled)
        return F.normalize(projected, dim=-1)

    def token_features(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class CharCNNEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2) for kernel_size in kernel_sizes]
        )
        # Channel width of each conv branch. Cached so masked max-pooling can slice the
        # concatenated feature map at the correct per-branch boundaries instead of
        # assuming every branch shares convs[0].out_channels (a silent bug if the
        # branches are ever given different widths).
        self.branch_channels = [conv.out_channels for conv in self.convs]
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features_by_position = self.token_features(token_ids, mask).transpose(1, 2)
        pooled_outputs = []
        expanded_mask = mask.unsqueeze(1)
        start = 0
        for width in self.branch_channels:
            features = features_by_position[:, start : start + width, :]
            masked_features = features.masked_fill(expanded_mask == 0, -1e4)
            pooled_outputs.append(masked_features.max(dim=2).values)
            start += width
        pooled = torch.cat(pooled_outputs, dim=1)
        projected = self.projection(self.dropout(pooled))
        return F.normalize(projected, dim=-1)

    def token_features(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids).transpose(1, 2)
        features = [F.gelu(conv(embeddings)).transpose(1, 2) for conv in self.convs]
        return torch.cat(features, dim=2) * mask.unsqueeze(-1)


class CharTransformerEncoder(nn.Module):
    """Byte-level self-attention encoder.

    Consumes the *identical* byte tokenization as ``CharMeanEncoder`` /
    ``CharCNNEncoder`` (vocab 256, ``padding_idx=0``, ``byte + 1`` offset), so the
    only variable it changes relative to the char-CNN is the sequence-mixing
    architecture: strided convolutions are replaced by multi-head self-attention.
    This makes it a single-axis (convolution -> attention) architecture ablation
    rather than a tokenization change. Byte-level transformers are the
    tokenization-free family (CANINE, ByT5, Charformer); because BioXRep aliases and
    HGVS strings are short, the usual quadratic-length cost never bites here.

    Contract matches the other encoders: ``forward(token_ids, mask) -> normalized
    projection`` and ``token_features(token_ids, mask) -> per-position features`` so
    it is a drop-in for attention distillation and every eval path.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 128,
        projection_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        max_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        # Learned absolute positional embeddings over the fixed max byte grid.
        self.position_embedding = nn.Embedding(max_length, hidden_dim)
        self.max_length = max_length
        feedforward_dim = feedforward_dim if feedforward_dim is not None else hidden_dim * 4
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first=True and would only
        # emit a warning while silently disabling itself; turn it off explicitly.
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        features = self.token_features(token_ids, mask)
        masked = features * mask.unsqueeze(-1)
        lengths = mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        pooled = masked.sum(dim=1) / lengths
        projected = self.projection(self.dropout(pooled))
        return F.normalize(projected, dim=-1)

    def token_features(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[1]
        positions = torch.arange(seq_len, device=token_ids.device).clamp_max(self.max_length - 1)
        hidden = self.embedding(token_ids) + self.position_embedding(positions).unsqueeze(0)
        # TransformerEncoder treats True in the padding mask as "ignore".
        key_padding_mask = mask == 0
        encoded = self.transformer(hidden, src_key_padding_mask=key_padding_mask)
        return encoded * mask.unsqueeze(-1)
