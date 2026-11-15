"""
multitask_vs_single_task.py
===========================
Phase 2 Member 4 - Task 6: Multitask vs Single-Task Learning Benchmark.
Compares:
  - Multitask Learning: One shared deep network predicting all 12 endpoints simultaneously
  - Single-Task Learning: 12 independent networks, each trained exclusively on one endpoint's observed data
Evaluated across 3 random seeds: 42, 21, 7 (Total 36 single-task + 3 multitask runs).
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

from phase2.data.src.data_loader import Tox21DataLoader, TARGET_ASSAYS
from phase2.deep_learning.src.model import MultiTaskToxicNet
from .data_utils import set_seed, Tox21DataLoaderWrapper
from .metrics import compute_classification_metrics, evaluate_multitask_predictions
from .single_task_model import SingleTaskToxicNet, SingleTaskTrainer
from .dropout_ablation import DropoutAblationStudy
from phase2.data.src.data_loader import METRICS_DIR, PLOTS_DIR, PREDICTIONS_DIR

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"


class MultitaskVsSingleTaskStudy:
    """
    Executes and analyzes the Multitask vs. Single-Task benchmark.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.config_path = Path(config_path or (base_dir / "configs" / "single_task_config.json"))
        self.output_dir = Path(output_dir or base_dir)

        with open(self.config_path, "r") as f:
            self.config = json.load(f)

        self.device = device or torch.device("cpu")
        self.wrapper = Tox21DataLoaderWrapper()
        self.loader = self.wrapper.loader
        self.target_assays = list(TARGET_ASSAYS)
        self.seeds = self.config.get("seeds", [42, 21, 7])

        # Prepare directories
        self.ckpt_dir = self.output_dir / "checkpoints" / "single_task"
        self.pred_dir = PREDICTIONS_DIR
        self.results_dir = METRICS_DIR
        self.plot_dir = PLOTS_DIR

        for d in [self.ckpt_dir, self.pred_dir, self.results_dir, self.plot_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def train_single_task_for_seed(self, seed: int) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Train 12 independent single-task networks for a given random seed.

        Returns
        -------
        df_eval : DataFrame of test evaluation metrics for all 12 endpoints + MACRO AVERAGE
        test_probs_matrix : (782, 12) array of predicted probabilities
        """
        set_seed(seed)
        t_cfg = self.config["training"]
        max_epochs = t_cfg["max_epochs"]
        patience = t_cfg["early_stopping_patience"]
        lr = t_cfg["learning_rate"]
        wd = t_cfg["weight_decay"]

        # Ground truth
        y_test, mask_test, mids_test = self.loader.load_labels(split="test")
        test_probs_matrix = np.zeros((len(mids_test), len(self.target_assays)), dtype=float)

        st_eval_rows = []

        print(f"\n[Single-Task] Training 12 independent endpoint models for Seed: {seed}...")
        for ep_idx, ep in enumerate(self.target_assays):
            tr_loader, va_loader, te_loader = self.wrapper.get_single_task_loaders(
                endpoint=ep, batch_size=t_cfg["batch_size"], feature_set=self.config["feature_set"]
            )
            pos_weight = self.wrapper.get_single_task_pos_weight(ep)

            model = SingleTaskToxicNet(
                input_dim=self.config["input_dim"],
                hidden_dims=self.config["hidden_dims"],
                latent_dim=self.config["latent_dim"],
                dropout=self.config["dropout"],
                norm_type=self.config["norm_type"],
                activation=self.config["activation"],
            )

            trainer = SingleTaskTrainer(
                model=model,
                train_loader=tr_loader,
                val_loader=va_loader,
                test_loader=te_loader,
                pos_weight=pos_weight,
                learning_rate=lr,
                weight_decay=wd,
                device=self.device,
                endpoint_name=ep,
            )

            fit_res = trainer.fit(max_epochs=max_epochs, patience=patience)

            # Evaluate test set
            test_preds, test_targets = trainer.predict(te_loader)
            m_metrics = compute_classification_metrics(test_targets, test_preds, threshold=0.5)
            m_metrics["Endpoint"] = ep
            m_metrics["Seed"] = seed
            m_metrics["Model"] = "Single-Task DL"
            m_metrics["Best_Epoch"] = fit_res["best_epoch"]
            st_eval_rows.append(m_metrics)

            # Place predictions into matrix for observed test instances
            obs_mask = mask_test[:, ep_idx].astype(bool)
            test_probs_matrix[obs_mask, ep_idx] = test_preds

            # Save checkpoint
            clean_ep = ep.replace("-", "_")
            ckpt_path = self.ckpt_dir / f"model_{clean_ep}_seed_{seed}.pt"
            torch.save({
                "endpoint": ep,
                "seed": seed,
                "best_epoch": fit_res["best_epoch"],
                "best_val_roc": fit_res["best_val_roc"],
                "model_state_dict": trainer.model.state_dict(),
            }, ckpt_path)

        df_st = pd.DataFrame(st_eval_rows)

        # Macro average row
        macro_dict: Dict[str, Any] = {
            "Endpoint": "MACRO AVERAGE",
            "Seed": seed,
            "Model": "Single-Task DL",
            "Positives": int(df_st["Positives"].sum()),
            "Negatives": int(df_st["Negatives"].sum()),
            "ROC-AUC": round(float(df_st["ROC-AUC"].mean()), 4),
            "PR-AUC": round(float(df_st["PR-AUC"].mean()), 4),
            "F1-Score": round(float(df_st["F1-Score"].mean()), 4),
            "Precision": round(float(df_st["Precision"].mean()), 4),
            "Recall": round(float(df_st["Recall"].mean()), 4),
            "Balanced Acc": round(float(df_st["Balanced Acc"].mean()), 4),
            "MCC": round(float(df_st["MCC"].mean()), 4),
            "TP": int(df_st["TP"].sum()),
            "FP": int(df_st["FP"].sum()),
            "TN": int(df_st["TN"].sum()),
            "FN": int(df_st["FN"].sum()),
        }
        df_st_full = pd.concat([df_st, pd.DataFrame([macro_dict])], ignore_index=True)

        # Save single-task predictions CSV
        pred_rows = []
        for r_idx, mid in enumerate(mids_test):
            for ep_idx, ep in enumerate(self.target_assays):
                is_obs = bool(mask_test[r_idx, ep_idx])
                p = float(test_probs_matrix[r_idx, ep_idx])
                pred_rows.append({
                    "mol_id": mid,
                    "endpoint": ep,
                    "seed": seed,
                    "model": "Single-Task DL",
                    "predicted_prob": round(p, 6),
                    "predicted_label": int(p >= 0.5),
                    "true_label": int(y_test[r_idx, ep_idx]) if is_obs else "",
                    "observed": is_obs,
                })
        df_pred_st = pd.DataFrame(pred_rows)
        df_pred_st.to_csv(self.pred_dir / f"preds_single_task_seed_{seed}.csv", index=False)

        return df_st_full, test_probs_matrix

    def get_or_train_multitask_for_seed(self, seed: int) -> pd.DataFrame:
        """
        Retrieve or train Multitask model for given seed.
        Checks if checkpoint exists from Task 4; if not, trains a run.
        """
        mt_ckpt_path = self.output_dir / "checkpoints" / "dropout" / f"model_original_dropout_seed_{seed}.pt"
        if mt_ckpt_path.exists():
            ckpt = torch.load(mt_ckpt_path, map_location=self.device)
            model = MultiTaskToxicNet(
                input_dim=self.config["input_dim"],
                hidden_dims=self.config["hidden_dims"],
                latent_dim=self.config["latent_dim"],
                num_endpoints=len(self.target_assays),
                dropout=self.config["dropout"],
            ).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            X_test, mids_test = self.loader.load_features("combined", split="test")
            y_test, mask_test, _ = self.loader.load_labels(split="test")
            with torch.no_grad():
                test_logits = model(torch.tensor(X_test, dtype=torch.float32).to(self.device))
                test_probs = torch.sigmoid(test_logits).cpu().numpy()

            df_eval = evaluate_multitask_predictions(test_probs, y_test, mask_test, self.target_assays, thresholds=0.5)
            df_eval["Seed"] = seed
            df_eval["Model"] = "MultiTask DL"
            return df_eval
        else:
            # Train on-the-fly
            study = DropoutAblationStudy(output_dir=self.output_dir, device=self.device)
            res = study.train_single_run(dropout_rate=0.35, seed=seed, config_name="original_dropout")
            df_eval = res["test_eval"]
            df_eval["Model"] = "MultiTask DL"
            return df_eval

    def run_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute full benchmark: 36 single-task runs + multitask comparison across 3 seeds.
        """
        all_evals: List[pd.DataFrame] = []

        print("=" * 70)
        print("STARTING TASK 6: MULTITASK VS SINGLE-TASK LEARNING BENCHMARK")
        print("=" * 70)

        for seed in self.seeds:
            # 1. Multitask run
            print(f"\n---> Evaluating Multitask DL for Seed: {seed}")
            df_mt = self.get_or_train_multitask_for_seed(seed)
            df_mt["Model"] = "MultiTask DL"
            all_evals.append(df_mt)
            mt_macro = df_mt[df_mt["Endpoint"] == "MACRO AVERAGE"].iloc[0]
            print(f"      MultiTask DL (Seed {seed}) -> Macro ROC-AUC: {mt_macro['ROC-AUC']:.4f} | PR-AUC: {mt_macro['PR-AUC']:.4f} | F1: {mt_macro['F1-Score']:.4f}")

            # 2. Single-task runs (12 independent models)
            df_st, _ = self.train_single_task_for_seed(seed)
            all_evals.append(df_st)
            st_macro = df_st[df_st["Endpoint"] == "MACRO AVERAGE"].iloc[0]
            print(f"      Single-Task DL (Seed {seed}) -> Macro ROC-AUC: {st_macro['ROC-AUC']:.4f} | PR-AUC: {st_macro['PR-AUC']:.4f} | F1: {st_macro['F1-Score']:.4f}")

        # Combine all evaluations into multitask_vs_single_task.csv
        combined_df = pd.concat(all_evals, ignore_index=True)
        csv_path = self.results_dir / "multitask_vs_single_task.csv"
        combined_df.to_csv(csv_path, index=False)
        print(f"\n[Saved] Detailed multitask vs single-task results saved to: {csv_path}")

        # Summary across seeds
        summary_rows = []
        macro_df = combined_df[combined_df["Endpoint"] == "MACRO AVERAGE"]
        for m_name in ["MultiTask DL", "Single-Task DL"]:
            sub = macro_df[macro_df["Model"] == m_name]
            summary_rows.append({
                "Model": m_name,
                "ROC-AUC_mean": round(float(sub["ROC-AUC"].mean()), 4),
                "ROC-AUC_std": round(float(sub["ROC-AUC"].std()), 4),
                "PR-AUC_mean": round(float(sub["PR-AUC"].mean()), 4),
                "PR-AUC_std": round(float(sub["PR-AUC"].std()), 4),
                "F1-Score_mean": round(float(sub["F1-Score"].mean()), 4),
                "F1-Score_std": round(float(sub["F1-Score"].std()), 4),
                "Precision_mean": round(float(sub["Precision"].mean()), 4),
                "Precision_std": round(float(sub["Precision"].std()), 4),
                "Recall_mean": round(float(sub["Recall"].mean()), 4),
                "Recall_std": round(float(sub["Recall"].std()), 4),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_csv_path = self.results_dir / "multitask_vs_single_task_summary.csv"
        summary_df.to_csv(summary_csv_path, index=False)
        print(f"[Saved] Multitask vs Single-Task summary saved to: {summary_csv_path}")

        # Generate plots
        self._plot_summary(summary_df)
        self._plot_endpoint_comparison(combined_df)

        return combined_df, summary_df

    def _plot_summary(self, summary_df: pd.DataFrame):
        """Plot overall macro performance bar chart with standard deviation error bars."""
        fig, ax = plt.subplots(figsize=(8, 5))
        metrics = ["ROC-AUC", "PR-AUC", "F1-Score"]
        x = np.arange(len(metrics))
        width = 0.35

        mt_means = [summary_df.loc[summary_df["Model"] == "MultiTask DL", f"{m}_mean"].values[0] for m in metrics]
        mt_stds = [summary_df.loc[summary_df["Model"] == "MultiTask DL", f"{m}_std"].values[0] for m in metrics]

        st_means = [summary_df.loc[summary_df["Model"] == "Single-Task DL", f"{m}_mean"].values[0] for m in metrics]
        st_stds = [summary_df.loc[summary_df["Model"] == "Single-Task DL", f"{m}_std"].values[0] for m in metrics]

        ax.bar(x - width/2, mt_means, width, yerr=mt_stds, capsize=5, label="MultiTask DL (Shared Trunk)", color="#1f77b4")
        ax.bar(x + width/2, st_means, width, yerr=st_stds, capsize=5, label="Single-Task DL (12 Independent Nets)", color="#ff7f0e")

        ax.set_ylabel("Macro Score across 12 Endpoints", fontsize=11)
        ax.set_title("MultiTask vs. Single-Task Deep Learning Performance (Mean ± SD, 3 Seeds)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 0.9)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plot_path = self.plot_dir / "multitask_vs_single_task.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Plot] Multitask summary plot saved to: {plot_path}")

    def _plot_endpoint_comparison(self, combined_df: pd.DataFrame):
        """Plot per-endpoint ROC-AUC comparison between MultiTask and Single-Task."""
        ep_df = combined_df[combined_df["Endpoint"] != "MACRO AVERAGE"]
        agg_df = ep_df.groupby(["Endpoint", "Model"]).agg({"ROC-AUC": "mean", "PR-AUC": "mean"}).reset_index()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        sns.barplot(
            data=agg_df, x="Endpoint", y="ROC-AUC", hue="Model",
            palette={"MultiTask DL": "#1f77b4", "Single-Task DL": "#ff7f0e"},
            ax=ax1,
        )
        ax1.set_title("Per-Endpoint ROC-AUC: MultiTask vs Single-Task (Mean across 3 Seeds)", fontsize=13, fontweight="bold")
        ax1.set_ylabel("Test ROC-AUC", fontsize=11)
        ax1.set_ylim(0.4, 0.95)
        ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax1.legend(loc="upper right")
        ax1.grid(axis="y", linestyle=":", alpha=0.6)

        sns.barplot(
            data=agg_df, x="Endpoint", y="PR-AUC", hue="Model",
            palette={"MultiTask DL": "#1f77b4", "Single-Task DL": "#ff7f0e"},
            ax=ax2,
        )
        ax2.set_title("Per-Endpoint PR-AUC: MultiTask vs Single-Task (Mean across 3 Seeds)", fontsize=13, fontweight="bold")
        ax2.set_ylabel("Test PR-AUC", fontsize=11)
        ax2.set_ylim(0.0, 0.75)
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha="right", fontsize=10)
        ax2.legend(loc="upper right")
        ax2.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plot_path = self.plot_dir / "multitask_single_task_endpoint_comparison.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"[Plot] Endpoint comparison plot saved to: {plot_path}")


def main():
    study = MultitaskVsSingleTaskStudy()
    study.run_all()


if __name__ == "__main__":
    main()

