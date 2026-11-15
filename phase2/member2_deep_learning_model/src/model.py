"""
model.py
========
MultiTaskToxicNet: Multitask Deep Neural Network for Tox21 Molecular Toxicity.
Predicts all 12 nuclear receptor and stress response pathways simultaneously.
Extracts learned latent molecular representations for downstream hybrid modeling.

Phase 2 Improved Architecture:
  - Residual skip connections to ease gradient flow through deep trunk
  - LayerNorm (default) instead of BatchNorm for sparse fingerprint stability
  - GELU activation for smoother gradients
  - Per-task classification heads with dedicated capacity per endpoint
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Dense block with optional residual (skip) connection.
    If input and output dimensions differ, a learned linear projection
    adapts the skip path so addition is valid.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3,
                 norm_type: str = "layer_norm", activation: str = "gelu"):
        super().__init__()
        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        self.linear = nn.Linear(in_dim, out_dim)
        if norm_type == "batch_norm":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm_type == "layer_norm":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()
        self.act = act_cls()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip projection when dimensions change
        self.skip = nn.Linear(in_dim, out_dim, bias=False) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = self.linear(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.drop(out)
        return out + residual


class TaskHead(nn.Module):
    """Per-endpoint classification head with one hidden layer."""

    def __init__(self, latent_dim: int, head_hidden: int = 64, dropout: float = 0.2,
                 activation: str = "gelu"):
        super().__init__()
        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU
        self.head = nn.Sequential(
            nn.Linear(latent_dim, head_hidden),
            act_cls(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z).squeeze(-1)


class MultiTaskToxicNet(nn.Module):
    """
    Multitask Neural Network for predicting 12 Tox21 toxicity endpoints.

    Improved Architecture:
      Input (2053-dim)
         │
      ResidualBlock(512) + LayerNorm + GELU + Dropout
         │
      ResidualBlock(256) + LayerNorm + GELU + Dropout
         │
      ResidualBlock(128) + LayerNorm + GELU + Dropout  <── Latent Representation (z)
         │
      12 × TaskHead(128 → 64 → 1) ── Per-Endpoint Logits
    """

    def __init__(
        self,
        input_dim: int = 2053,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 128,
        num_endpoints: int = 12,
        dropout: float = 0.35,
        norm_type: str = "layer_norm",
        activation: str = "gelu",
        head_hidden: int = 64,
        head_dropout: float = 0.2,
        use_task_heads: bool = True,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.num_endpoints = num_endpoints
        self.dropout_rate = dropout
        self.norm_type = norm_type
        self.use_task_heads = use_task_heads
        self.input_dropout_rate = input_dropout

        # Input dropout: randomly drops raw Morgan bits during training.
        # With only 6,259 training molecules against a 2,053-dim input the trunk
        # otherwise memorises individual fingerprint bits within a few epochs
        # (train loss collapsing to ~0.02 while validation loss climbs). Dropping
        # input bits forces the network to rely on redundant substructure
        # evidence rather than on any single bit.
        self.input_drop = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()

        # Shared feature trunk with residual connections
        blocks: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            blocks.append(ResidualBlock(prev_dim, h_dim, dropout=dropout,
                                        norm_type=norm_type, activation=activation))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*blocks)

        # Bottleneck latent layer (embeddings for Member 3)
        self.latent_layer = ResidualBlock(prev_dim, latent_dim,
                                          dropout=dropout * 0.5,
                                          norm_type=norm_type,
                                          activation=activation)

        # Multi-task classification heads
        if use_task_heads:
            self.task_heads = nn.ModuleList([
                TaskHead(latent_dim, head_hidden=head_hidden,
                         dropout=head_dropout, activation=activation)
                for _ in range(num_endpoints)
            ])
            self.classifier = None
        else:
            # Backward-compatible single linear layer
            self.classifier = nn.Linear(latent_dim, num_endpoints)
            self.task_heads = None

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation (bottleneck embedding z)."""
        h = self.trunk(self.input_drop(x))
        z = self.latent_layer(h)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits (batch_size, 12)."""
        z = self.extract_latent(x)
        if self.use_task_heads and self.task_heads is not None:
            logits = torch.stack([head(z) for head in self.task_heads], dim=1)
        else:
            logits = self.classifier(z)
        return logits

    def forward_with_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and latent embeddings."""
        z = self.extract_latent(x)
        if self.use_task_heads and self.task_heads is not None:
            logits = torch.stack([head(z) for head in self.task_heads], dim=1)
        else:
            logits = self.classifier(z)
        return logits, z

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for extract_latent for downstream compatibility."""
        return self.extract_latent(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid predicted probabilities for all 12 endpoints."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)
