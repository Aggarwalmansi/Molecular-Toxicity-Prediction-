"""
trainer.py
==========
Training and validation harness for MultiTaskToxicNet.
Includes:
- Masked loss computation
- Class weighting
- Learning rate scheduling
- Early stopping
- Training metrics logging and plotting
- Checkpointing
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .loss import MaskedWeightedBCEWithLogitsLoss
from .model import MultiTaskToxicNet
from phase2.data.src.data_loader import METRICS_DIR, PLOTS_DIR


class MultiTaskTrainer:
    """
    End-to-end trainer for MultiTaskToxicNet on Tox21.
    """

    def __init__(
        self,
        model: MultiTaskToxicNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        pos_weight: Optional[torch.Tensor] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        report_dir: Optional[Union[str, Path]] = None,
        target_assays: Optional[List[str]] = None,
        early_stopping_metric: str = "val_macro_roc_auc",
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        self.device = device
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.early_stopping_metric = early_stopping_metric
        self.target_assays = target_assays or [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
            "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
            "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
        ]

        # Checkpoints & Reports directories
        base_dir = Path(__file__).resolve().parent.parent
        self.checkpoint_dir = Path(checkpoint_dir or (base_dir / "checkpoints"))
        self.report_dir = Path(report_dir or METRICS_DIR)
        self.plot_dir = PLOTS_DIR

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        # Loss function with positive class weights
        self.criterion = MaskedWeightedBCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer and LR Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max" if early_stopping_metric == "val_macro_roc_auc" else "min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )

        self.history: List[Dict[str, Any]] = []
        self.best_score = float("-inf") if early_stopping_metric == "val_macro_roc_auc" else float("inf")
        self.best_val_loss = float("inf")
        self.best_val_roc = 0.0
        self.best_val_pr = 0.0
        self.best_epoch = 0

    def train_epoch(self) -> float:
        """Run one training epoch over train_loader."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0

        for X_batch, y_batch, mask_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            mask_batch = mask_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch, mask_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        return total_loss / max(1, total_batches)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, Dict[str, float], Dict[str, float]]:
        """
        Evaluate model on a dataloader.
        Returns:
            loss, macro_roc_auc, macro_pr_auc, per_endpoint_roc, per_endpoint_pr
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        all_logits = []
        all_targets = []
        all_masks = []

        for X_batch, y_batch, mask_batch in loader:
            X_b = X_batch.to(self.device)
            y_b = y_batch.to(self.device)
            m_b = mask_batch.to(self.device)

            logits = self.model(X_b)
            loss = self.criterion(logits, y_b, m_b)

            total_loss += loss.item()
            total_batches += 1

            all_logits.append(logits.cpu())
            all_targets.append(y_batch.cpu())
            all_masks.append(mask_batch.cpu())

        avg_loss = total_loss / max(1, total_batches)
        y_prob = torch.sigmoid(torch.cat(all_logits, dim=0)).numpy()
        y_true = torch.cat(all_targets, dim=0).numpy()
        mask = torch.cat(all_masks, dim=0).numpy().astype(bool)

        # Compute per-endpoint metrics strictly on observed labels
        per_endpoint_roc: Dict[str, float] = {}
        per_endpoint_pr: Dict[str, float] = {}

        for ep_idx, ep in enumerate(self.target_assays):
            obs = mask[:, ep_idx]
            y_obs = y_true[obs, ep_idx]
            p_obs = y_prob[obs, ep_idx]

            # Invariant: Must have both classes to compute ROC-AUC
            if len(np.unique(y_obs)) > 1:
                try:
                    roc = float(roc_auc_score(y_obs, p_obs))
                    pr = float(average_precision_score(y_obs, p_obs))
                    per_endpoint_roc[ep] = roc
                    per_endpoint_pr[ep] = pr
                except Exception:
                    per_endpoint_roc[ep] = 0.5
                    per_endpoint_pr[ep] = 0.0
            else:
                per_endpoint_roc[ep] = 0.5
                per_endpoint_pr[ep] = 0.0

        macro_roc = float(np.mean(list(per_endpoint_roc.values())))
        macro_pr = float(np.mean(list(per_endpoint_pr.values())))

        return avg_loss, macro_roc, macro_pr, per_endpoint_roc, per_endpoint_pr

    def fit(self, max_epochs: int = 100, patience: int = 15) -> Dict[str, Any]:
        """
        Train with early stopping and save best checkpoint.
        """
        epochs_no_improve = 0
        best_state = None
        best_val_roc = 0.0
        best_val_pr = 0.0

        is_maximize = self.early_stopping_metric == "val_macro_roc_auc"
        print(f"[MultiTaskTrainer] Training on device: {self.device} for up to {max_epochs} epochs (monitoring {self.early_stopping_metric}, patience={patience})")

        for epoch in range(1, max_epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_roc, val_pr, _, _ = self.evaluate(self.val_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            current_metric = val_roc if is_maximize else val_loss
            self.scheduler.step(current_metric)

            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_macro_roc_auc": round(val_roc, 4),
                "val_macro_pr_auc": round(val_pr, 4),
                "learning_rate": current_lr,
            }
            self.history.append(record)

            improved = (current_metric > self.best_score) if is_maximize else (current_metric < self.best_score)

            if epoch % 5 == 0 or epoch == 1 or improved:
                print(
                    f"Epoch {epoch:03d}/{max_epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val ROC-AUC: {val_roc:.4f} | "
                    f"Val PR-AUC: {val_pr:.4f} | "
                    f"LR: {current_lr:.1e}"
                )

            # Early stopping check
            if improved:
                self.best_score = current_metric
                self.best_epoch = epoch
                self.best_val_loss = val_loss
                self.best_val_roc = val_roc
                self.best_val_pr = val_pr
                best_val_roc = val_roc
                best_val_pr = val_pr
                # state_dict() returns references to the live parameter
                # tensors, not copies. Without .clone() these entries keep
                # tracking the model as training continues, so by the end of
                # fit() they hold the FINAL epoch's weights instead of the
                # best epoch's, and the restore below silently loads the wrong
                # model. Cloning freezes the values at this epoch.
                best_state = {
                    "epoch": epoch,
                    "model_state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    },
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_macro_roc_auc": val_roc,
                    "val_macro_pr_auc": val_pr,
                    "config": {
                        "input_dim": self.model.input_dim,
                        "hidden_dims": self.model.hidden_dims,
                        "latent_dim": self.model.latent_dim,
                        "num_endpoints": self.model.num_endpoints,
                        "dropout": self.model.dropout_rate,
                        "target_assays": self.target_assays,
                    },
                }
                torch.save(best_state, self.checkpoint_dir / "best_multitask_model.pt")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"[MultiTaskTrainer] Early stopping triggered at epoch {epoch}. Best epoch: {self.best_epoch}")
                    break

        # Save training history
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.report_dir / "training_history.csv", index=False)

        # Plot curves
        self._plot_training_curves(history_df)

        # Reload best model weights
        if best_state is not None:
            self.model.load_state_dict(best_state["model_state_dict"])
            print(f"[MultiTaskTrainer] Restored best model weights from epoch {self.best_epoch} (Val Loss: {self.best_val_loss:.4f}, Val ROC-AUC: {best_val_roc:.4f})")

        return {
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "best_val_macro_roc_auc": best_val_roc,
            "best_val_macro_pr_auc": best_val_pr,
        }

    def _plot_training_curves(self, df: pd.DataFrame):
        """Save publication-quality training curves."""
        # 1. Loss Curve
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
        plt.axvline(self.best_epoch, color="#2ca02c", linestyle="--", label=f"Best Epoch ({self.best_epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Masked Weighted BCE Loss")
        plt.title("MultiTaskToxicNet Training & Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / "loss_curve.png", dpi=200)
        plt.close()

        # 2. Validation ROC & PR AUC Curves
        plt.figure(figsize=(8, 5))
        plt.plot(df["epoch"], df["val_macro_roc_auc"], label="Val Macro ROC-AUC", color="#2ca02c", lw=2)
        plt.plot(df["epoch"], df["val_macro_pr_auc"], label="Val Macro PR-AUC", color="#9467bd", lw=2)
        plt.axvline(self.best_epoch, color="#d62728", linestyle="--", label=f"Best Epoch ({self.best_epoch})")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Metric Score")
        plt.title("Validation Macro ROC-AUC & PR-AUC Trajectory")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plot_dir / "val_metrics_curve.png", dpi=200)
        plt.close()

        # 3. Combined Dual-Panel Plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
        ax1.plot(df["epoch"], df["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
        ax1.axvline(self.best_epoch, color="#2ca02c", linestyle="--", label=f"Best ({self.best_epoch})")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Masked Weighted Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(df["epoch"], df["val_macro_roc_auc"], label="Macro ROC-AUC", color="#2ca02c", lw=2)
        ax2.plot(df["epoch"], df["val_macro_pr_auc"], label="Macro PR-AUC", color="#9467bd", lw=2)
        ax2.axvline(self.best_epoch, color="#d62728", linestyle="--", label=f"Best ({self.best_epoch})")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_title("Validation Macro Performance")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle("MultiTaskToxicNet Learning Dynamics (Phase 2 DL Model)", fontsize=13, weight="bold")
        plt.tight_layout()
        plt.savefig(self.plot_dir / "training_curves_summary.png", dpi=200)
        plt.close()
