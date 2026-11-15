"""
class_weight_ablation.py
========================
Phase 2 Member 4 - Task 5: Class-Weighting Ablation Study.
Evaluates the impact of positive class weighting on the multi-task deep neural network.
Compares:
  - Experiment A: Weighted Loss (training-set positive class weights)
  - Experiment B: Unweighted Loss (pos_weight=None)
Observation masking remains strictly enabled in both configurations.
Across 3 random seeds: 42, 21, 7 (Total 6 training runs).
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from phase2.data.src.data_loader import Tox21DataLoader, TARGET_ASSAYS
from phase2.deep_learning.src.loss import MaskedWeightedBCEWithLogitsLoss
from phase2.deep_learning.src.model import MultiTaskToxicNet
from .data_utils import set_seed, Tox21DataLoaderWrapper
from .metrics import compute_classification_metrics, evaluate_multitask_predictions
from phase2.data.src.data_loader import METRICS_DIR, PLOTS_DIR, PREDICTIONS_DIR

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"


class ClassWeightAblationStudy:
    """
    Executes and analyzes the Class Weighting ablation study across 3 random seeds.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.config_path = Path(config_path or (base_dir / "configs" / "class_weight_config.json"))
        self.output_dir = Path(output_dir or base_dir)

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.wrapper = Tox21DataLoaderWrapper()
        self.loader = self.wrapper.loader
        self.target_assays = list(TARGET_ASSAYS)
        self.seeds = self.config.get("seeds", [42, 21, 7])

        # Prepare directories
        self.ckpt_dir = self.output_dir / "checkpoints" / "class_weight"
        self.pred_dir = PREDICTIONS_DIR
        self.results_dir = METRICS_DIR
        self.plot_dir = PLOTS_DIR

        for d in [self.ckpt_dir, self.pred_dir, self.results_dir, self.plot_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def train_single_run(
        self,
        use_class_weights: bool,
        seed: int,
        config_name: str,
    ) -> Dict[str, Any]:
        """
        Train a multi-task network with or without positive class weights.
        """
        set_seed(seed)
        t_cfg = self.config["training"]
        batch_size = t_cfg["batch_size"]
        max_epochs = t_cfg["max_epochs"]
        lr = t_cfg["learning_rate"]
        weight_decay = t_cfg["weight_decay"]
        patience = t_cfg["early_stopping_patience"]

        train_loader, val_loader, test_loader = self.loader.get_pytorch_dataloaders(
            feature_set=self.config["feature_set"],
            batch_size=batch_size,
            shuffle_train=True,
        )

        pos_weights = None
        if use_class_weights:
            pos_weights = self.loader.get_class_weights(as_tensor=True).to(self.device)

        model = MultiTaskToxicNet(
            input_dim=self.config["input_dim"],
            hidden_dims=self.config["hidden_dims"],
            latent_dim=self.config["latent_dim"],
            num_endpoints=self.config["num_endpoints"],
            dropout=self.config["dropout"],
            norm_type=self.config["norm_type"],
            activation=self.config["activation"],
        ).to(self.device)

        criterion = MaskedWeightedBCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-5
        )

        history: List[Dict[str, float]] = []
        best_val_roc = -1.0
        best_epoch = 0
        best_weights = None
        epochs_no_improve = 0

        y_val, mask_val, _ = self.loader.load_labels(split="val")

        for epoch in range(1, max_epochs + 1):
            # Training
            model.train()
            train_loss = 0.0
            n_batches = 0
            for X_b, y_b, m_b in train_loader:
                X_b, y_b, m_b = X_b.to(self.device), y_b.to(self.device), m_b.to(self.device)
                optimizer.zero_grad()
                logits = model(X_b)
                loss = criterion(logits, y_b, m_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(1, n_batches)

            # Validation
            model.eval()
            val_loss = 0.0
            val_batches = 0
            all_val_logits = []
            with torch.no_grad():
                for X_b, y_b, m_b in val_loader:
                    X_b, y_b, m_b = X_b.to(self.device), y_b.to(self.device), m_b.to(self.device)
                    logits = model(X_b)
                    loss = criterion(logits, y_b, m_b)
                    val_loss += loss.item()
                    val_batches += 1
                    all_val_logits.append(logits.cpu())
            val_loss /= max(1, val_batches)

            val_probs = torch.sigmoid(torch.cat(all_val_logits, dim=0)).numpy()
            val_eval = evaluate_multitask_predictions(
                val_probs, y_val, mask_val, self.target_assays, thresholds=0.5
            )
            val_macro_roc = float(val_eval.loc[val_eval["Endpoint"] == "MACRO AVERAGE", "ROC-AUC"].values[0])
            val_macro_pr = float(val_eval.loc[val_eval["Endpoint"] == "MACRO AVERAGE", "PR-AUC"].values[0])

            scheduler.step(val_macro_roc)
            current_lr = float(optimizer.param_groups[0]["lr"])

            history.append({
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss": round(val_loss, 5),
                "val_macro_roc_auc": round(val_macro_roc, 4),
                "val_macro_pr_auc": round(val_macro_pr, 4),
                "lr": current_lr,
            })

            if val_macro_roc > best_val_roc:
                best_val_roc = val_macro_roc
                best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if best_weights is not None:
            model.load_state_dict(best_weights)

        # Save Checkpoint
        ckpt_file = self.ckpt_dir / f"model_{config_name}_seed_{seed}.pt"
        torch.save({
            "epoch": best_epoch,
            "seed": seed,
            "use_class_weights": use_class_weights,
            "config_name": config_name,
            "val_macro_roc_auc": best_val_roc,
            "model_state_dict": best_weights,
            "history": history,
        }, ckpt_file)

        # Generate Test Predictions
        model.eval()
        X_test, mids_test = self.loader.load_features("combined", split="test")
        y_test, mask_test, _ = self.loader.load_labels(split="test")
        with torch.no_grad():
            test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(self.device))
            test_probs = torch.sigmoid(test_logits).cpu().numpy()

        # Save test predictions CSV
        pred_rows = []
        for r_idx, mid in enumerate(mids_test):
            for ep_idx, ep in enumerate(self.target_assays):
                is_obs = bool(mask_test[r_idx, ep_idx])
                p = float(test_probs[r_idx, ep_idx])
                pred_rows.append({
                    "mol_id": mid,
                    "endpoint": ep,
                    "seed": seed,
                    "configuration": config_name,
                    "weighted": use_class_weights,
                    "predicted_prob": round(p, 6),
                    "predicted_label": int(p >= 0.5),
                    "true_label": int(y_test[r_idx, ep_idx]) if is_obs else "",
                    "observed": is_obs,
                })
        df_preds = pd.DataFrame(pred_rows)
        df_preds.to_csv(self.pred_dir / f"preds_{config_name}_seed_{seed}.csv", index=False)

        # Evaluate test metrics
        test_eval_df = evaluate_multitask_predictions(
            test_probs, y_test, mask_test, self.target_assays, thresholds=0.5
        )
        test_eval_df["Seed"] = seed
        test_eval_df["Configuration"] = config_name
        test_eval_df["Weighted"] = use_class_weights
        test_eval_df["Best_Epoch"] = best_epoch

        return {
            "config_name": config_name,
            "seed": seed,
            "weighted": use_class_weights,
            "best_epoch": best_epoch,
            "best_val_roc": best_val_roc,
            "history": pd.DataFrame(history),
            "test_eval": test_eval_df,
            "checkpoint_path": str(ckpt_file),
        }

    def run_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute all 6 training runs (2 configurations x 3 seeds).
        """
        configs = [
            ("weighted_loss", True),
            ("unweighted_loss", False),
        ]
        all_evals: List[pd.DataFrame] = []

        print("=" * 70)
        print("STARTING TASK 5: CLASS-WEIGHTING ABLATION EXPERIMENT")
        print("=" * 70)

        for cfg_name, weighted_flag in configs:
            for seed in self.seeds:
                print(f"\n---> Running Configuration: '{cfg_name}' (weighted={weighted_flag}) | Seed: {seed}")
                run_res = self.train_single_run(use_class_weights=weighted_flag, seed=seed, config_name=cfg_name)
                all_evals.append(run_res["test_eval"])
                macro_row = run_res["test_eval"][run_res["test_eval"]["Endpoint"] == "MACRO AVERAGE"].iloc[0]
                print(f"      Finished at epoch {run_res['best_epoch']} | Test Macro ROC-AUC: {macro_row['ROC-AUC']:.4f} | Recall: {macro_row['Recall']:.4f} | Precision: {macro_row['Precision']:.4f} | FN: {macro_row['FN']}")

        combined_df = pd.concat(all_evals, ignore_index=True)
        csv_path = self.results_dir / "class_weight_ablation.csv"
        combined_df.to_csv(csv_path, index=False)
        print(f"\n[Saved] Detailed class weight ablation results saved to: {csv_path}")

        # Summary across seeds
        summary_rows = []
        macro_df = combined_df[combined_df["Endpoint"] == "MACRO AVERAGE"]
        for cfg_name in ["weighted_loss", "unweighted_loss"]:
            sub = macro_df[macro_df["Configuration"] == cfg_name]
            summary_rows.append({
                "Configuration": cfg_name,
                "Weighted": True if cfg_name == "weighted_loss" else False,
                "ROC-AUC_mean": round(float(sub["ROC-AUC"].mean()), 4),
                "ROC-AUC_std": round(float(sub["ROC-AUC"].std()), 4),
                "PR-AUC_mean": round(float(sub["PR-AUC"].mean()), 4),
                "PR-AUC_std": round(float(sub["PR-AUC"].std()), 4),
                "F1-Score_mean": round(float(sub["F1-Score"].mean()), 4),
                "F1-Score_std": round(float(sub["F1-Score"].std()), 4),
                "Recall_mean": round(float(sub["Recall"].mean()), 4),
                "Recall_std": round(float(sub["Recall"].std()), 4),
                "Precision_mean": round(float(sub["Precision"].mean()), 4),
                "Precision_std": round(float(sub["Precision"].std()), 4),
                "FN_mean": round(float(sub["FN"].mean()), 1),
                "TP_mean": round(float(sub["TP"].mean()), 1),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = self.results_dir / "class_weight_ablation_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[Saved] Class weight ablation summary saved to: {summary_csv_path}")

        # Generate diagnostic plots
        self._plot_comparison(combined_df)
        self._plot_precision_recall_tradeoff(combined_df)

        return combined_df, summary_df

    def _plot_comparison(self, combined_df: pd.DataFrame):
        """Plot bar comparison of Recall, Precision, and ROC-AUC for weighted vs unweighted."""
        ep_df = combined_df[combined_df["Endpoint"] != "MACRO AVERAGE"]
        agg_df = ep_df.groupby(["Endpoint", "Configuration"]).agg({
            "Recall": "mean",
            "Precision": "mean",
            "ROC-AUC": "mean",
            "FN": "mean",
        }).reset_index()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        sns.barplot(
            data=agg_df, x="Endpoint", y="Recall", hue="Configuration",
            palette={"weighted_loss": "#2ca02c", "unweighted_loss": "#7f7f7f"},
            ax=ax1,
        )
        ax1.set_title("Class-Weighting Impact on Positive Recall (Sensitivity to Rare Toxic Compounds)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Active Compound Recall", fontsize=11)
        ax1.set_ylim(0.0, 1.0)
        ax1.legend(title="Loss Formulation", loc="upper right")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        sns.barplot(
            data=agg_df, x="Endpoint", y="FN", hue="Configuration",
            palette={"weighted_loss": "#2ca02c", "unweighted_loss": "#d62728"},
            ax=ax2,
        )
        ax2.set_title("False Negative Count by Endpoint (Lower is Safer for Hazard Screening)", fontsize=13, fontweight="bold")
        ax2.set_ylabel("False Negatives (Missed Toxins)", fontsize=11)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right", fontsize=10)
        ax2.legend(title="Loss Formulation", loc="upper right")
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        comp_path = self.plot_dir / "class_weight_comparison.png"
        plt.savefig(comp_path, dpi=300)
        plt.close()
        print(f"[Plot] Class weight comparison plot saved to: {comp_path}")

    def _plot_precision_recall_tradeoff(self, combined_df: pd.DataFrame):
        """Scatter/bar plot visualizing precision vs recall trade-off across endpoints."""
        macro_df = combined_df[combined_df["Endpoint"] == "MACRO AVERAGE"].groupby("Configuration").agg({
            "Precision": ["mean", "std"],
            "Recall": ["mean", "std"],
            "F1-Score": ["mean", "std"],
            "ROC-AUC": ["mean", "std"],
        })

        fig, ax = plt.subplots(figsize=(8, 6))
        ep_df = combined_df[combined_df["Endpoint"] != "MACRO AVERAGE"]
        agg_df = ep_df.groupby(["Endpoint", "Configuration"]).agg({"Precision": "mean", "Recall": "mean"}).reset_index()

        sns.scatterplot(
            data=agg_df, x="Recall", y="Precision", hue="Configuration",
            style="Configuration", s=120,
            palette={"weighted_loss": "#2ca02c", "unweighted_loss": "#d62728"},
            ax=ax
        )

        for _, row in agg_df.iterrows():
            ax.annotate(row["Endpoint"], (row["Recall"] + 0.01, row["Precision"]), fontsize=8, alpha=0.8)

        ax.set_title("Precision vs. Recall Trade-Off under Class Weighting", fontsize=13, fontweight="bold")
        ax.set_xlabel("Positive Recall (Sensitivity)", fontsize=11)
        ax.set_ylabel("Positive Precision", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_xlim(-0.02, 1.0)
        ax.set_ylim(-0.02, 1.0)

        plt.tight_layout()
        pr_path = self.plot_dir / "class_weight_precision_recall.png"
        plt.savefig(pr_path, dpi=300)
        plt.close()
        print(f"[Plot] Precision-Recall trade-off plot saved to: {pr_path}")


def main():
    study = ClassWeightAblationStudy()
    study.run_all()


if __name__ == "__main__":
    main()

