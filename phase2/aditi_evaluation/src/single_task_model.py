"""
single_task_model.py
====================
Single-Task Deep Neural Network Architecture and Training Harness for Tox21.
Each network is an independent model predicting exactly ONE toxicological endpoint.
Maintains identical depth, hidden-layer capacity, normalization, and regularization as MultiTaskToxicNet.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from .metrics import compute_classification_metrics


class SingleTaskToxicNet(nn.Module):
    """
    Independent Deep Neural Network for a single toxicity endpoint.

    Architecture:
      Input (2053-dim)
         │
      Dense(512) + BatchNorm + ReLU + Dropout(0.35)
         │
      Dense(256) + BatchNorm + ReLU + Dropout(0.35)
         │
      Dense(128) + BatchNorm + ReLU + Dropout(0.175)  (Bottleneck)
         │
      Dense(1) ── Single Endpoint Logit
    """

    def __init__(
        self,
        input_dim: int = 2053,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 128,
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
        self.dropout_rate = dropout

        act_cls = nn.GELU if activation.lower() == "gelu" else nn.ReLU

        # Feature trunk
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

        # Bottleneck latent layer
        latent_layers: List[nn.Module] = [nn.Linear(prev_dim, latent_dim)]
        if norm_type == "batch_norm":
            latent_layers.append(nn.BatchNorm1d(latent_dim))
        elif norm_type == "layer_norm":
            latent_layers.append(nn.LayerNorm(latent_dim))
        latent_layers.append(act_cls())
        if dropout > 0:
            latent_layers.append(nn.Dropout(dropout * 0.5))
        self.latent_layer = nn.Sequential(*latent_layers)

        # Single-task classification head
        self.classifier = nn.Linear(latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning single-endpoint logits (batch_size, 1)."""
        h = self.trunk(x)
        z = self.latent_layer(h)
        logits = self.classifier(z)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid predicted probabilities (batch_size, 1)."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


class SingleTaskTrainer:
    """
    Dedicated training and validation harness for SingleTaskToxicNet.
    Monitors single-endpoint validation ROC-AUC for early stopping.
    """

    def __init__(
        self,
        model: SingleTaskToxicNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        pos_weight: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        endpoint_name: str = "Endpoint",
    ):
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.endpoint_name = endpoint_name

        pw_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
        )

        self.best_val_roc = -1.0
        self.best_epoch = 0
        self.best_weights: Optional[Dict[str, Any]] = None

    def fit(self, max_epochs: int = 100, patience: int = 15) -> Dict[str, Any]:
        """Train single-task model with early stopping."""
        epochs_no_improve = 0
        history: List[Dict[str, float]] = []

        for epoch in range(1, max_epochs + 1):
            # Training
            self.model.train()
            train_loss = 0.0
            n_train_batches = 0
            for X_b, y_b in self.train_loader:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(X_b)
                loss = self.criterion(logits, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                train_loss += loss.item()
                n_train_batches += 1
            train_loss /= max(1, n_train_batches)

            # Validation
            self.model.eval()
            val_loss = 0.0
            n_val_batches = 0
            all_val_preds = []
            all_val_targets = []
            with torch.no_grad():
                for X_b, y_b in self.val_loader:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    logits = self.model(X_b)
                    loss = self.criterion(logits, y_b)
                    val_loss += loss.item()
                    n_val_batches += 1
                    all_val_preds.append(torch.sigmoid(logits).cpu().numpy())
                    all_val_targets.append(y_b.cpu().numpy())
            val_loss /= max(1, n_val_batches)

            y_val_true = np.vstack(all_val_targets).ravel()
            y_val_prob = np.vstack(all_val_preds).ravel()

            if len(np.unique(y_val_true)) > 1:
                val_roc = float(roc_auc_score(y_val_true, y_val_prob))
                val_pr = float(average_precision_score(y_val_true, y_val_prob))
            else:
                val_roc = 0.5
                val_pr = 0.0

            self.scheduler.step(val_roc)

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_roc_auc": round(val_roc, 4),
                "val_pr_auc": round(val_pr, 4),
            })

            if val_roc > self.best_val_roc:
                self.best_val_roc = val_roc
                self.best_epoch = epoch
                self.best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)

        return {
            "best_epoch": self.best_epoch,
            "best_val_roc": self.best_val_roc,
            "history": pd.DataFrame(history),
        }

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predicted probabilities and ground truth labels from a loader."""
        self.model.eval()
        preds, targets = [], []
        for X_b, y_b in loader:
            X_b = X_b.to(self.device)
            logits = self.model(X_b)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y_b.numpy())
        return np.vstack(preds).ravel(), np.vstack(targets).ravel()

