"""
model.py
========
MultiTaskToxicNet: Multitask Deep Neural Network for Tox21 Molecular Toxicity.
Predicts all 12 nuclear receptor and stress response pathways simultaneously.
Extracts learned latent molecular representations for downstream hybrid modeling.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskToxicNet(nn.Module):
    """
    Multitask Neural Network for predicting 12 Tox21 toxicity endpoints.

    Architecture:
      Input (2053-dim)
         │
      Dense(512) + BatchNorm + ReLU + Dropout
         │
      Dense(256) + BatchNorm + ReLU + Dropout
         │
      Dense(128) + BatchNorm + ReLU + Dropout  <── Latent Representation (z)
         │
      Dense(12) ── Multi-Task Logits
    """

    def __init__(
        self,
        input_dim: int = 2053,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 128,
        num_endpoints: int = 12,
        dropout: float = 0.35,
        norm_type: str = "batch_norm",
        activation: str = "relu",
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

        # Activation factory
        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        # Shared feature trunk
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            if norm_type == "batch_norm":
                layers.append(nn.BatchNorm1d(h_dim))
            elif norm_type == "layer_norm":
                layers.append(nn.LayerNorm(h_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)

        # Bottleneck latent layer (embeddings for Member 3)
        latent_layers: List[nn.Module] = [nn.Linear(prev_dim, latent_dim)]
        if norm_type == "batch_norm":
            latent_layers.append(nn.BatchNorm1d(latent_dim))
        elif norm_type == "layer_norm":
            latent_layers.append(nn.LayerNorm(latent_dim))
        latent_layers.append(act_cls())
        if dropout > 0:
            latent_layers.append(nn.Dropout(dropout * 0.5))
        self.latent_layer = nn.Sequential(*latent_layers)

        # Multi-task classification head
        self.classifier = nn.Linear(latent_dim, num_endpoints)

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation (bottleneck embedding z)."""
        h = self.trunk(x)
        z = self.latent_layer(h)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning raw logits (batch_size, 12)."""
        z = self.extract_latent(x)
        logits = self.classifier(z)
        return logits

    def forward_with_embeddings(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and latent embeddings."""
        z = self.extract_latent(x)
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
