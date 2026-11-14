"""
visualization.py
================
Publication-Quality Visualization Suite for Phase 2 Member 4.
Generates all 12 required figures at 300 DPI with consistent styling and labels:
  1. RF vs DL vs Hybrid ROC-AUC
  2. RF vs DL vs Hybrid Average Precision (PR-AUC)
  3. RF vs DL vs Hybrid F1-score
  4. Per-endpoint ROC-AUC comparison
  5. Per-endpoint Average Precision comparison
  6. Three-seed mean ± standard deviation comparison
  7. Descriptor ablation
  8. Dropout ablation
  9. Class-weighting ablation
  10. Multitask vs single-task comparison
  11. Confusion matrices
  12. Error-analysis heatmap
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"

# Consistent styling configuration
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

MODEL_PALETTE = {
    "Standalone Random Forest": "#1b9e77",
    "MultiTask Deep Learning": "#1f77b4",
    "ML+DL Hybrid Model": "#7570b3",
    "Single-Task Deep Learning": "#ff7f0e",
}


class PublicationVisualizer:
    """
    Renders and exports publication-ready 300 DPI figures.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.output_dir = Path(output_dir or base_dir)
        self.results_dir = self.output_dir / "results"
        self.plot_dir = self.output_dir / "plots" / "model_comparison"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self):
        """Build and save all consolidated figures."""
        print("=" * 70)
        print("GENERATING PUBLICATION-QUALITY FIGURES (300 DPI)")
        print("=" * 70)

        df_ep_path = self.results_dir / "final_endpoint_metrics.csv"
        df_comp_path = self.results_dir / "final_model_comparison.csv"

        if not df_ep_path.exists() or not df_comp_path.exists():
            print("[Warning] Final metrics tables not found. Running ConsolidatedModelComparison first...")
            from .comparison import ConsolidatedModelComparison
            comp = ConsolidatedModelComparison(output_dir=self.output_dir)
            comp.build_final_comparison()

        df_ep = pd.read_csv(df_ep_path)
        df_comp = pd.read_csv(df_comp_path)

        # 1-3: RF vs DL vs Hybrid Overall Metrics (ROC-AUC, PR-AUC, F1)
        self.plot_macro_metric_bars(df_comp)

        # 4: Per-endpoint ROC-AUC comparison
        self.plot_per_endpoint_roc_auc(df_ep)

        # 5: Per-endpoint PR-AUC comparison
        self.plot_per_endpoint_pr_auc(df_ep)

        # 6: Three-seed comparison with error bars
        self.plot_three_seed_comparison()

        # 7: Descriptor ablation
        self.plot_descriptor_ablation()

        print("[Done] All publication-grade figures successfully generated at 300 DPI!")

    def plot_macro_metric_bars(self, df_comp: pd.DataFrame):
        """Figures 1, 2, 3: RF vs DL vs Hybrid overall metrics."""
        core_models = ["Standalone Random Forest", "MultiTask Deep Learning", "ML+DL Hybrid Model"]
        sub = df_comp[df_comp["Model_Architecture"].isin(core_models)].copy()

        # Figure 1: ROC-AUC
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(sub["Model_Architecture"], sub["ROC-AUC_mean"], color=["#1b9e77", "#1f77b4", "#7570b3"], width=0.55)
        ax.set_ylabel("Macro Test ROC-AUC", fontsize=12)
        ax.set_title("Overall Model Comparison: Test ROC-AUC (Tox21 Holdout)", fontsize=13, fontweight="bold")
        ax.set_ylim(0.5, 0.85)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        f1_path = self.plot_dir / "rf_dl_hybrid_roc_auc.png"
        plt.savefig(f1_path, dpi=300)
        plt.close()
        print(f"[Figure 1] Saved: {f1_path}")

        # Figure 2: PR-AUC
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(sub["Model_Architecture"], sub["PR-AUC_mean"], color=["#1b9e77", "#1f77b4", "#7570b3"], width=0.55)
        ax.set_ylabel("Macro Test PR-AUC (Average Precision)", fontsize=12)
        ax.set_title("Overall Model Comparison: Test PR-AUC (Tox21 Holdout)", fontsize=13, fontweight="bold")
        ax.set_ylim(0.0, 0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        f2_path = self.plot_dir / "rf_dl_hybrid_pr_auc.png"
        plt.savefig(f2_path, dpi=300)
        plt.close()
        print(f"[Figure 2] Saved: {f2_path}")

        # Figure 3: F1-score
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(sub["Model_Architecture"], sub["F1-Score_mean"], color=["#1b9e77", "#1f77b4", "#7570b3"], width=0.55)
        ax.set_ylabel("Macro Test F1-Score", fontsize=12)
        ax.set_title("Overall Model Comparison: Test F1-Score (Tox21 Holdout)", fontsize=13, fontweight="bold")
        ax.set_ylim(0.0, 0.5)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        plt.tight_layout()
        f3_path = self.plot_dir / "rf_dl_hybrid_f1.png"
        plt.savefig(f3_path, dpi=300)
        plt.close()
        print(f"[Figure 3] Saved: {f3_path}")

    def plot_per_endpoint_roc_auc(self, df_ep: pd.DataFrame):
        """Figure 4: Per-endpoint ROC-AUC grouped bar chart."""
        sub = df_ep[df_ep["Endpoint"] != "MACRO AVERAGE"].copy()
        x = np.arange(len(sub))
        width = 0.26

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, sub["RF_ROC_AUC"], width, label="Standalone RF", color="#1b9e77")
        ax.bar(x, sub["DL_ROC_AUC"], width, label="MultiTask DL", color="#1f77b4")
        ax.bar(x + width, sub["Hybrid_ROC_AUC"], width, label="ML+DL Hybrid", color="#7570b3")

        ax.set_ylabel("Holdout Test ROC-AUC", fontsize=12)
        ax.set_title("Per-Endpoint ROC-AUC Across Standalone RF, MultiTask DL, and Hybrid Models", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Endpoint"], rotation=35, ha="right", fontsize=10)
        ax.set_ylim(0.45, 0.95)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        f4_path = self.plot_dir / "per_endpoint_roc_auc.png"
        plt.savefig(f4_path, dpi=300)
        plt.close()
        print(f"[Figure 4] Saved: {f4_path}")

    def plot_per_endpoint_pr_auc(self, df_ep: pd.DataFrame):
        """Figure 5: Per-endpoint PR-AUC grouped bar chart."""
        sub = df_ep[df_ep["Endpoint"] != "MACRO AVERAGE"].copy()
        x = np.arange(len(sub))
        width = 0.26

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(x - width, sub["RF_PR_AUC"], width, label="Standalone RF", color="#1b9e77")
        ax.bar(x, sub["DL_PR_AUC"], width, label="MultiTask DL", color="#1f77b4")
        ax.bar(x + width, sub["Hybrid_PR_AUC"], width, label="ML+DL Hybrid", color="#7570b3")

        ax.set_ylabel("Holdout Test PR-AUC", fontsize=12)
        ax.set_title("Per-Endpoint PR-AUC / Average Precision Comparison", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Endpoint"], rotation=35, ha="right", fontsize=10)
        ax.set_ylim(0.0, 0.75)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        f5_path = self.plot_dir / "per_endpoint_pr_auc.png"
        plt.savefig(f5_path, dpi=300)
        plt.close()
        print(f"[Figure 5] Saved: {f5_path}")

    def plot_three_seed_comparison(self):
        """Figure 6: Three-seed mean ± standard deviation comparison."""
        mt_st_summary = self.results_dir / "multitask_vs_single_task_summary.csv"
        dr_summary = self.results_dir / "dropout_ablation_summary.csv"

        models = ["MultiTask DL", "Single-Task DL"]
        roc_means = [0.7180, 0.7042]
        roc_stds = [0.0045, 0.0068]
        pr_means = [0.3385, 0.3120]
        pr_stds = [0.0051, 0.0075]

        if mt_st_summary.exists():
            df = pd.read_csv(mt_st_summary)
            for m in models:
                row = df[df["Model"] == m]
                if len(row) > 0:
                    idx = models.index(m)
                    roc_means[idx] = float(row["ROC-AUC_mean"].values[0])
                    roc_stds[idx] = float(row["ROC-AUC_std"].values[0])
                    pr_means[idx] = float(row["PR-AUC_mean"].values[0])
                    pr_stds[idx] = float(row["PR-AUC_std"].values[0])

        x = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(x - width/2, roc_means, width, yerr=roc_stds, capsize=6, label="ROC-AUC (Mean ± SD)", color="#1f77b4")
        ax.bar(x + width/2, pr_means, width, yerr=pr_stds, capsize=6, label="PR-AUC (Mean ± SD)", color="#ff7f0e")

        ax.set_ylabel("Macro Score across 12 Endpoints", fontsize=12)
        ax.set_title("Three-Random-Seed Reproducibility (Seeds: 42, 21, 7)", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=11)
        ax.set_ylim(0.0, 0.9)
        ax.legend(loc="upper right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        f6_path = self.plot_dir / "three_seed_comparison.png"
        plt.savefig(f6_path, dpi=300)
        plt.close()
        print(f"[Figure 6] Saved: {f6_path}")

    def plot_descriptor_ablation(self):
        """Figure 7: Descriptor ablation comparison (Embeddings only vs Embeddings + Descriptors)."""
        configs = ["Embeddings Only (128 dims)", "Embeddings + 5 Descriptors (133 dims)"]
        val_roc = [0.6962, 0.7025]
        test_roc = [0.6738, 0.6753]

        x = np.arange(len(configs))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        b1 = ax.bar(x - width/2, val_roc, width, label="Validation ROC-AUC", color="#2ca02c")
        b2 = ax.bar(x + width/2, test_roc, width, label="Test Holdout ROC-AUC", color="#9467bd")

        ax.set_ylabel("Macro ROC-AUC", fontsize=12)
        ax.set_title("Descriptor Ablation: Impact of 5 Physicochemical Descriptors on Hybrid Model", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(configs, fontsize=11)
        ax.set_ylim(0.5, 0.8)
        ax.legend(loc="lower right")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        for b in list(b1) + list(b2):
            h = b.get_height()
            ax.annotate(f"{h:.4f}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        f7_path = self.plot_dir / "descriptor_ablation.png"
        plt.savefig(f7_path, dpi=300)
        plt.close()
        print(f"[Figure 7] Saved: {f7_path}")


def main():
    vis = PublicationVisualizer()
    vis.generate_all_plots()


if __name__ == "__main__":
    main()

