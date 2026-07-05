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
        for start in range(0, features_by_position.shape[1], self.convs[0].out_channels):
            features = features_by_position[:, start : start + self.convs[0].out_channels, :]
            masked_features = features.masked_fill(expanded_mask == 0, -1e4)
            pooled_outputs.append(masked_features.max(dim=2).values)
        pooled = torch.cat(pooled_outputs, dim=1)
        projected = self.projection(self.dropout(pooled))
        return F.normalize(projected, dim=-1)

    def token_features(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids).transpose(1, 2)
        features = [F.gelu(conv(embeddings)).transpose(1, 2) for conv in self.convs]
        return torch.cat(features, dim=2) * mask.unsqueeze(-1)
