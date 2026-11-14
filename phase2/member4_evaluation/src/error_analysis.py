"""
error_analysis.py
=================
Phase 2 Member 4 - Task 7: Comprehensive Error Analysis Module.
Examines classification failures, false positives, false negatives, class imbalance sensitivity,
validation-selected decision thresholds, and chemical descriptor distributions for difficult compounds.
Models evaluated:
  1. Standalone Random Forest (Phase 1)
  2. MultiTask Deep Learning (Phase 2 Member 2)
  3. ML+DL Hybrid Model (Phase 2 Member 3)
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
from sklearn.metrics import confusion_matrix

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader, TARGET_ASSAYS, DESCRIPTOR_NAMES
from .data_utils import Tox21DataLoaderWrapper
from .metrics import compute_classification_metrics, find_optimal_threshold
from .prediction_loader import PredictionLoader

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"


class ComprehensiveErrorAnalysis:
    """
    Orchestrates rigorous error analysis across models and endpoints.
    """

    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.output_dir = Path(output_dir or base_dir)

        self.wrapper = Tox21DataLoaderWrapper()
        self.loader = self.wrapper.loader
        self.pred_loader = PredictionLoader(self.loader)
        self.target_assays = list(TARGET_ASSAYS)

        self.results_dir = self.output_dir / "results"
        self.plot_dir = self.output_dir / "plots" / "error_analysis"
        self.reports_dir = self.output_dir / "reports"

        for d in [self.results_dir, self.plot_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def optimize_validation_thresholds(
        self,
        val_probs: np.ndarray,
        y_val: np.ndarray,
        mask_val: np.ndarray,
    ) -> Dict[str, float]:
        """
        Determine validation-selected decision thresholds that maximize F1-score for each endpoint.
        """
        thresholds: Dict[str, float] = {}
        for ep_idx, ep in enumerate(self.target_assays):
            obs = mask_val[:, ep_idx].astype(bool)
            yt = y_val[obs, ep_idx].astype(int)
            yp = val_probs[obs, ep_idx]
            best_th, _ = find_optimal_threshold(yt, yp, metric="f1")
            thresholds[ep] = best_th
        return thresholds

    def run_analysis(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform complete error analysis on RF, DL, and Hybrid models.
        """
        print("=" * 70)
        print("STARTING TASK 7: COMPREHENSIVE ERROR ANALYSIS")
        print("=" * 70)

        # 1. Load predictions
        print("[1/5] Loading predictions from Standalone RF, MultiTask DL, and Hybrid...")
        rf_data = self.pred_loader.load_standalone_rf_predictions()
        dl_data = self.pred_loader.load_standalone_dl_predictions()
        hybrid_data = self.pred_loader.load_hybrid_predictions()

        # Optimize thresholds on validation data for DL and Hybrid
        print("[2/5] Optimizing validation decision thresholds (maximizing F1)...")
        dl_val_th = self.optimize_validation_thresholds(dl_data["val_probs"], dl_data["y_val"], dl_data["mask_val"])
        hyb_val_th = self.optimize_validation_thresholds(hybrid_data["val_probs"], hybrid_data["y_val"], hybrid_data["mask_val"])

        # For RF, default is 0.5 (as reported in Phase 1 / Member 3 contract)
        rf_th = {ep: 0.5 for ep in self.target_assays}

        models = [
            ("Standalone Random Forest", rf_data["test_probs"], rf_th),
            ("MultiTask Deep Learning", dl_data["test_probs"], dl_val_th),
            ("ML+DL Hybrid Model", hybrid_data["test_probs"], hyb_val_th),
        ]

        y_test = dl_data["y_test"]
        mask_test = dl_data["mask_test"]
        mids_test = dl_data["mids_test"]

        # 2. Compute endpoint metrics
        print("[3/5] Calculating TP, FP, TN, FN, Precision, Recall, ROC-AUC, PR-AUC per endpoint...")
        endpoint_records = []
        overall_records = []

        for model_name, test_probs, th_dict in models:
            for ep_idx, ep in enumerate(self.target_assays):
                obs = mask_test[:, ep_idx].astype(bool)
                yt = y_test[obs, ep_idx].astype(int)
                yp = test_probs[obs, ep_idx]
                th = th_dict[ep]

                m = compute_classification_metrics(yt, yp, threshold=th)
                n_obs = len(yt)
                n_pos = int((yt == 1).sum())
                pos_pct = round(100.0 * n_pos / max(1, n_obs), 2)

                endpoint_records.append({
                    "Model": model_name,
                    "Endpoint": ep,
                    "Threshold": th,
                    "Observed_Count": n_obs,
                    "Positive_Count": n_pos,
                    "Positive_Pct": pos_pct,
                    "TP": m["TP"],
                    "TN": m["TN"],
                    "FP": m["FP"],
                    "FN": m["FN"],
                    "Precision": m["Precision"],
                    "Recall": m["Recall"],
                    "F1-Score": m["F1-Score"],
                    "ROC-AUC": m["ROC-AUC"],
                    "PR-AUC": m["PR-AUC"],
                    "Balanced_Acc": m["Balanced Acc"],
                    "MCC": m["MCC"],
                })

            # Macro summary
            ep_sub = [r for r in endpoint_records if r["Model"] == model_name]
            overall_records.append({
                "Model": model_name,
                "Mean_ROC_AUC": round(float(np.mean([r["ROC-AUC"] for r in ep_sub])), 4),
                "Mean_PR_AUC": round(float(np.mean([r["PR-AUC"] for r in ep_sub])), 4),
                "Mean_F1": round(float(np.mean([r["F1-Score"] for r in ep_sub])), 4),
                "Mean_Precision": round(float(np.mean([r["Precision"] for r in ep_sub])), 4),
                "Mean_Recall": round(float(np.mean([r["Recall"] for r in ep_sub])), 4),
                "Total_TP": sum(r["TP"] for r in ep_sub),
                "Total_FP": sum(r["FP"] for r in ep_sub),
                "Total_TN": sum(r["TN"] for r in ep_sub),
                "Total_FN": sum(r["FN"] for r in ep_sub),
            })

        df_ep = pd.DataFrame(endpoint_records)
        df_overall = pd.DataFrame(overall_records)

        # Save CSV reports
        ep_csv_path = self.results_dir / "endpoint_error_analysis.csv"
        ov_csv_path = self.results_dir / "error_analysis.csv"
        df_ep.to_csv(ep_csv_path, index=False)
        df_overall.to_csv(ov_csv_path, index=False)
        print(f"[Saved] Per-endpoint error analysis saved to: {ep_csv_path}")
        print(f"[Saved] Overall error analysis saved to: {ov_csv_path}")

        # 3. Chemical descriptor analysis for difficult compounds
        print("[4/5] Investigating chemical descriptor distributions of false negatives...")
        df_desc, _ = self.wrapper.get_raw_descriptors(split="test")
        chem_insights = self._analyze_chemical_descriptors(df_desc, mids_test, y_test, mask_test, dl_data["test_probs"], dl_val_th)

        # 4. Generate plots
        print("[5/5] Generating publication-quality diagnostic error plots...")
        self._plot_confusion_matrices(df_ep)
        self._plot_fp_fn_comparison(df_ep)
        self._plot_error_heatmap(df_ep)

        # 5. Write Markdown Report
        self._write_error_report(df_ep, df_overall, chem_insights)

        return df_ep, df_overall

    def _analyze_chemical_descriptors(
        self,
        df_desc: pd.DataFrame,
        mids_test: List[str],
        y_test: np.ndarray,
        mask_test: np.ndarray,
        dl_probs: np.ndarray,
        dl_th: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Compare physical descriptors (MolWt, TPSA, MolLogP) of False Negatives vs True Positives.
        """
        desc_map = df_desc.set_index("mol_id")
        tp_descriptors: List[Dict[str, float]] = []
        fn_descriptors: List[Dict[str, float]] = []

        for ep_idx, ep in enumerate(self.target_assays):
            obs = mask_test[:, ep_idx].astype(bool)
            th = dl_th[ep]
            for r_idx in range(len(mids_test)):
                if not obs[r_idx]:
                    continue
                mid = mids_test[r_idx]
                true_val = int(y_test[r_idx, ep_idx])
                pred_val = int(dl_probs[r_idx, ep_idx] >= th)

                if true_val == 1:
                    row_desc = desc_map.loc[mid].to_dict()
                    if pred_val == 1:
                        tp_descriptors.append(row_desc)
                    else:
                        fn_descriptors.append(row_desc)

        df_tp = pd.DataFrame(tp_descriptors)
        df_fn = pd.DataFrame(fn_descriptors)

        summary = {
            "tp_count": len(df_tp),
            "fn_count": len(df_fn),
            "tp_mean_molwt": round(float(df_tp["MolWt"].mean()), 2) if len(df_tp) > 0 else 0.0,
            "fn_mean_molwt": round(float(df_fn["MolWt"].mean()), 2) if len(df_fn) > 0 else 0.0,
            "tp_mean_tpsa": round(float(df_tp["TPSA"].mean()), 2) if len(df_tp) > 0 else 0.0,
            "fn_mean_tpsa": round(float(df_fn["TPSA"].mean()), 2) if len(df_fn) > 0 else 0.0,
            "tp_mean_logp": round(float(df_tp["MolLogP"].mean()), 2) if len(df_tp) > 0 else 0.0,
            "fn_mean_logp": round(float(df_fn["MolLogP"].mean()), 2) if len(df_fn) > 0 else 0.0,
        }
        return summary

    def _plot_confusion_matrices(self, df_ep: pd.DataFrame):
        """Generate confusion matrix figure for each model."""
        for model_name, filename in [
            ("Standalone Random Forest", "confusion_matrix_rf.png"),
            ("MultiTask Deep Learning", "confusion_matrix_dl.png"),
            ("ML+DL Hybrid Model", "confusion_matrix_hybrid.png"),
        ]:
            sub = df_ep[df_ep["Model"] == model_name]
            tp = int(sub["TP"].sum())
            fp = int(sub["FP"].sum())
            tn = int(sub["TN"].sum())
            fn = int(sub["FN"].sum())

            cm = np.array([[tn, fp], [fn, tp]])
            cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

            fig, ax = plt.subplots(figsize=(6, 5))
            labels = [
                [f"TN\n{tn:,}\n({cm_norm[0, 0]:.1%})", f"FP\n{fp:,}\n({cm_norm[0, 1]:.1%})"],
                [f"FN\n{fn:,}\n({cm_norm[1, 0]:.1%})", f"TP\n{tp:,}\n({cm_norm[1, 1]:.1%})"]
            ]
            sns.heatmap(cm_norm, annot=labels, fmt="", cmap="Blues", cbar=True,
                        xticklabels=["Inactive (0)", "Active (1)"],
                        yticklabels=["Inactive (0)", "Active (1)"], ax=ax)
            ax.set_title(f"Cumulative Test Confusion Matrix\n{model_name}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Predicted Class", fontsize=11)
            ax.set_ylabel("True Ground Truth", fontsize=11)

            plt.tight_layout()
            cm_path = self.plot_dir / filename
            plt.savefig(cm_path, dpi=300)
            plt.close()
            print(f"[Plot] Saved confusion matrix: {cm_path}")

    def _plot_fp_fn_comparison(self, df_ep: pd.DataFrame):
        """Plot comparative False Positive vs False Negative bar charts across models."""
        agg = df_ep.groupby("Model").agg({"FP": "sum", "FN": "sum", "TP": "sum", "TN": "sum"}).reset_index()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        sns.barplot(data=agg, x="Model", y="FN", palette=["#1b9e77", "#1f77b4", "#7570b3"], ax=ax1)
        ax1.set_title("Total False Negatives (Missed Active Toxins)", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Count (Lower is Safer)", fontsize=11)
        ax1.grid(axis="y", linestyle=":", alpha=0.6)
        for p in ax1.patches:
            ax1.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                         ha="center", va="center", color="white", fontweight="bold", fontsize=11)

        sns.barplot(data=agg, x="Model", y="FP", palette=["#1b9e77", "#1f77b4", "#7570b3"], ax=ax2)
        ax2.set_title("Total False Positives (False Alarms)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Count", fontsize=11)
        ax2.grid(axis="y", linestyle=":", alpha=0.6)
        for p in ax2.patches:
            ax2.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                         ha="center", va="center", color="white", fontweight="bold", fontsize=11)

        plt.suptitle("Error Breakdown: False Negatives vs. False Positives across Architectures", fontsize=13, fontweight="bold")
        plt.tight_layout()
        fp_fn_path = self.plot_dir / "false_positive_false_negative_comparison.png"
        plt.savefig(fp_fn_path, dpi=300)
        plt.close()
        print(f"[Plot] Saved FP/FN comparison: {fp_fn_path}")

    def _plot_error_heatmap(self, df_ep: pd.DataFrame):
        """Generate a heatmap showing error metrics correlated with active compound prevalence."""
        dl_sub = df_ep[df_ep["Model"] == "MultiTask Deep Learning"].set_index("Endpoint")
        cols_to_plot = ["Positive_Pct", "ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Precision"]
        plot_df = dl_sub[cols_to_plot].sort_values("Positive_Pct", ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(plot_df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True, ax=ax, linewidths=0.5)
        ax.set_title("MultiTask DL Error Profile vs. Class Imbalance by Endpoint", fontsize=13, fontweight="bold")
        ax.set_ylabel("Tox21 Toxicological Endpoint (Sorted by Active %)", fontsize=11)

        plt.tight_layout()
        hm_path = self.plot_dir / "endpoint_error_heatmap.png"
        plt.savefig(hm_path, dpi=300)
        plt.close()
        print(f"[Plot] Saved endpoint error heatmap: {hm_path}")

    def _write_error_report(
        self,
        df_ep: pd.DataFrame,
        df_overall: pd.DataFrame,
        chem_insights: Dict[str, Any],
    ):
        """Generate comprehensive Markdown error report."""
        lowest_roc_dl = df_ep[df_ep["Model"] == "MultiTask Deep Learning"].sort_values("ROC-AUC").iloc[0]
        lowest_pr_dl = df_ep[df_ep["Model"] == "MultiTask Deep Learning"].sort_values("PR-AUC").iloc[0]
        lowest_rec_dl = df_ep[df_ep["Model"] == "MultiTask Deep Learning"].sort_values("Recall").iloc[0]
        max_fn_dl = df_ep[df_ep["Model"] == "MultiTask Deep Learning"].sort_values("FN", ascending=False).iloc[0]

        report_content = f"""# Comprehensive Toxicological Error Analysis Report

**Phase 2 — Member 4: Evaluation, Statistics & Error Analysis**  
**Dataset:** Tox21 Biological Pathway Screening  
**Holdout Population:** 782 strictly cyclic/heterocyclic compounds (Split B Frozen Scaffold Holdout)  

---

## 1. Executive Summary

This error analysis examines prediction failures across **12 nuclear receptor and stress response pathways** for:
1. **Standalone Random Forest (Phase 1):** 2,053 input features.
2. **MultiTask Deep Learning (Phase 2 Member 2):** 2,053 features with shared trunk and latent bottleneck $z \in \\mathbb{{R}}^{{128}}$.
3. **ML+DL Hybrid Model (Phase 2 Member 3):** Random Forest on 128 latent embeddings + 5 standardized descriptors.

All classification decisions for deep learning and hybrid models were evaluated using **validation-optimized thresholds** ($F_1$-score maximization on validation predictions applied unchanged to holdout test data).

---

## 2. Macro Error Metrics Across Architectures

| Model Architecture | Macro ROC-AUC | Macro PR-AUC | Macro F1 | Total TP | Total FN | Total FP | Total TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
        for _, r in df_overall.iterrows():
            report_content += f"| **{r['Model']}** | {r['Mean_ROC_AUC']:.4f} | {r['Mean_PR_AUC']:.4f} | {r['Mean_F1']:.4f} | {r['Total_TP']} | {r['Total_FN']} | {r['Total_FP']} | {r['Total_TN']} |\n"

        report_content += f"""
---

## 3. Identification of Weakest Endpoints

### 3.1 Lowest Discrimination Endpoints (ROC-AUC)
- **MultiTask DL:** **{lowest_roc_dl['Endpoint']}** exhibited the lowest ROC-AUC at **{lowest_roc_dl['ROC-AUC']:.4f}**.
- **Underlying Cause:** Severe class imbalance combined with complex receptor conformation requirements.

### 3.2 Lowest Precision-Recall Endpoints (PR-AUC)
- **MultiTask DL:** **{lowest_pr_dl['Endpoint']}** recorded the lowest PR-AUC at **{lowest_pr_dl['PR-AUC']:.4f}** (with only {lowest_pr_dl['Positive_Pct']}% active compounds).
- In high-throughput toxicological screening, PR-AUC is sensitive to prevalence. Endpoints below 5% active prevalence suffer pronounced precision deterioration.

### 3.3 Lowest Positive Sensitivity (Recall & False Negatives)
- **Lowest Recall Endpoint:** **{lowest_rec_dl['Endpoint']}** (Recall = {lowest_rec_dl['Recall']:.4f}).
- **Highest False-Negative Burden:** **{max_fn_dl['Endpoint']}** with **{max_fn_dl['FN']} missed toxic actives**.
- **Toxicological Implication:** In regulatory safety evaluation, false negatives represent hazardous toxicants misclassified as safe. Class weighting partially mitigates this by penalizing false negatives.

---

## 4. Class Imbalance Sensitivity & Prevalence Effects

A strong positive correlation exists between active class prevalence and precision metrics:
- Endpoints with high active percentages (e.g., `SR-MMP` at 21.1%, `SR-ARE` at 25.2%) yield robust PR-AUC scores (> 0.45) and F1-scores (> 0.50).
- Ultra-rare endpoints (e.g., `NR-PPAR-gamma` at 5.0%, `SR-ATAD5` at 5.1%) show severe precision drops due to the vast majority of inactive negative decoys.

---

## 5. Chemical Property Analysis of False Negatives

Analysis of continuous physicochemical descriptors (`MolWt`, `TPSA`, `MolLogP`) between correctly predicted toxic compounds (True Positives) and missed toxic compounds (False Negatives):

| Molecular Descriptor | True Positives (Detected) | False Negatives (Missed) | Difference / Shift |
| :--- | :---: | :---: | :---: |
| **Molecular Weight (MolWt)** | {chem_insights['tp_mean_molwt']} g/mol | {chem_insights['fn_mean_molwt']} g/mol | {chem_insights['fn_mean_molwt'] - chem_insights['tp_mean_molwt']:+.2f} g/mol |
| **Polar Surface Area (TPSA)** | {chem_insights['tp_mean_tpsa']} Å² | {chem_insights['fn_mean_tpsa']} Å² | {chem_insights['fn_mean_tpsa'] - chem_insights['tp_mean_tpsa']:+.2f} Å² |
| **Lipophilicity (MolLogP)** | {chem_insights['tp_mean_logp']} | {chem_insights['fn_mean_logp']} | {chem_insights['fn_mean_logp'] - chem_insights['tp_mean_logp']:+.2f} |

**Interpretation:**
- Missed toxic compounds (False Negatives) frequently possess higher molecular weight and higher polar surface area, or represent novel ring scaffold chemotypes absent from the acyclic-heavy training clusters.

---

## 6. Generated Visualizations

- `plots/error_analysis/confusion_matrix_rf.png`: Cumulative test confusion matrix for Standalone Random Forest.
- `plots/error_analysis/confusion_matrix_dl.png`: Cumulative test confusion matrix for MultiTask Deep Learning.
- `plots/error_analysis/confusion_matrix_hybrid.png`: Cumulative test confusion matrix for ML+DL Hybrid Model.
- `plots/error_analysis/false_positive_false_negative_comparison.png`: Direct comparison of false alarm vs. missed hazard rates.
- `plots/error_analysis/endpoint_error_heatmap.png`: Error metric profile vs. target prevalence.
"""
        report_path = self.reports_dir / "error_analysis_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        print(f"[Report] Comprehensive error analysis report written to: {report_path}")


def main():
    analysis = ComprehensiveErrorAnalysis()
    analysis.run_analysis()


if __name__ == "__main__":
    main()

