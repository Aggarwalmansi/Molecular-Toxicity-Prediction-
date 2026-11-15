"""
comparison.py
=============
Phase 2 Member 4 - Final Consolidated Model Comparison Module.
Aggregates and formats benchmark comparisons across:
  1. Standalone Random Forest (Phase 1)
  2. MultiTask Deep Learning (Phase 2 Member 2)
  3. ML+DL Hybrid Model (Phase 2 Member 3)
  4. Single-Task Deep Learning (Task 6)
  5. Descriptor Ablation (Task 3)
  6. Dropout Ablation (Task 4)
  7. Class-Weighting Ablation (Task 5)
Produces certified CSV summaries and JSON metadata for Member 5 and faculty viva.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from phase2.data.src.data_loader import Tox21DataLoader, TARGET_ASSAYS
from .data_utils import Tox21DataLoaderWrapper
from .metrics import compute_classification_metrics
from .prediction_loader import PredictionLoader
from phase2.data.src.data_loader import METRICS_DIR

os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"


class ConsolidatedModelComparison:
    """
    Builds authoritative comparison tables across all architectures, seeds, and ablations.
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
        self.target_assays = list(TARGET_ASSAYS)
        self.results_dir = METRICS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def load_completed_experiment_results(self) -> Dict[str, pd.DataFrame]:
        """Load results from Phase 1, Phase 2 Member 2, Member 3, and Member 4 tasks."""
        root = self.base_dir.parents[1]
        results: Dict[str, pd.DataFrame] = {}

        # 1. Standalone RF (Phase 1)
        rf_metrics_path = root / "results" / "metrics.csv"
        if rf_metrics_path.exists():
            df_rf = pd.read_csv(rf_metrics_path)
            results["standalone_rf"] = df_rf[df_rf["model"] == "Random Forest"]

        # 2. Member 2 DL
        dl_eval_path = root / "phase2" / "deep_learning" / "reports" / "test_evaluation.csv"
        if dl_eval_path.exists():
            results["standalone_dl"] = pd.read_csv(dl_eval_path)

        # 3. Member 3 Hybrid
        hyb_eval_path = root / "phase2" / "hybrid" / "reports" / "test_evaluation.csv"
        if hyb_eval_path.exists():
            results["hybrid"] = pd.read_csv(hyb_eval_path)

        # 4. Member 4 Dropout Ablation
        dropout_path = self.results_dir / "dropout_ablation.csv"
        if dropout_path.exists():
            results["dropout"] = pd.read_csv(dropout_path)

        # 5. Member 4 Class Weight Ablation
        cw_path = self.results_dir / "class_weight_ablation.csv"
        if cw_path.exists():
            results["class_weight"] = pd.read_csv(cw_path)

        # 6. Member 4 Multitask vs Single Task
        mt_st_path = self.results_dir / "multitask_vs_single_task.csv"
        if mt_st_path.exists():
            results["multitask_single_task"] = pd.read_csv(mt_st_path)

        return results

    def build_final_comparison(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Synthesize all experimental results into final evaluation tables.
        """
        data = self.load_completed_experiment_results()

        # Build final_model_comparison.csv
        comparison_rows = []

        # 1. Standalone RF
        if "standalone_rf" in data:
            rf_df = data["standalone_rf"]
            comparison_rows.append({
                "Model_Architecture": "Standalone Random Forest",
                "Phase": "Phase 1 Baseline",
                "Features": "2,053 (2048 Morgan + 5 Descriptors)",
                "ROC-AUC_mean": round(float(rf_df["test_roc_auc"].mean()), 4),
                "PR-AUC_mean": round(float(rf_df["test_pr_auc"].mean()), 4),
                "F1-Score_mean": round(float(rf_df["f1_score"].mean()), 4),
                "Balanced_Acc_mean": round(float(rf_df["balanced_accuracy"].mean()), 4),
                "MCC_mean": round(float(rf_df["mcc"].mean()), 4),
                "Notes": "Original Phase 1 competitive benchmark",
            })

        # 2. Standalone MultiTask DL (Member 2)
        if "standalone_dl" in data:
            dl_df = data["standalone_dl"]
            dl_macro = dl_df[dl_df["Endpoint"] == "MACRO AVERAGE"].iloc[0]
            comparison_rows.append({
                "Model_Architecture": "MultiTask Deep Learning",
                "Phase": "Phase 2 Member 2",
                "Features": "2,053 -> 128 bottleneck -> 12 outputs",
                "ROC-AUC_mean": round(float(dl_macro["ROC-AUC"]), 4),
                "PR-AUC_mean": round(float(dl_macro["PR-AUC"]), 4),
                "F1-Score_mean": round(float(dl_macro["F1-Score"]), 4),
                "Balanced_Acc_mean": round(float(dl_macro["Balanced Acc"]), 4),
                "MCC_mean": round(float(dl_macro["MCC"]), 4),
                "Notes": "Canonical MultiTaskToxicNet production model",
            })

        # 3. ML+DL Hybrid Model (Member 3)
        if "hybrid" in data:
            hyb_df = data["hybrid"]
            hyb_macro = hyb_df[hyb_df["Endpoint"] == "MACRO AVERAGE"].iloc[0]
            comparison_rows.append({
                "Model_Architecture": "ML+DL Hybrid Model",
                "Phase": "Phase 2 Member 3",
                "Features": "133 (128 embeddings + 5 descriptors)",
                "ROC-AUC_mean": round(float(hyb_macro["ROC-AUC"]), 4),
                "PR-AUC_mean": round(float(hyb_macro["PR-AUC"]), 4),
                "F1-Score_mean": round(float(hyb_macro["F1-Score"]), 4),
                "Balanced_Acc_mean": round(float(hyb_macro["Balanced Acc"]), 4),
                "MCC_mean": round(float(hyb_macro["MCC"]), 4),
                "Notes": "Negative result honestly reported (bottleneck compression penalty)",
            })

        # 4. Single-Task Deep Learning
        if "multitask_single_task" in data:
            st_df = data["multitask_single_task"]
            st_macro = st_df[(st_df["Endpoint"] == "MACRO AVERAGE") & (st_df["Model"] == "Single-Task DL")]
            if len(st_macro) > 0:
                comparison_rows.append({
                    "Model_Architecture": "Single-Task Deep Learning",
                    "Phase": "Phase 2 Member 4",
                    "Features": "2,053 per endpoint (12 independent nets)",
                    "ROC-AUC_mean": round(float(st_macro["ROC-AUC"].mean()), 4),
                    "PR-AUC_mean": round(float(st_macro["PR-AUC"].mean()), 4),
                    "F1-Score_mean": round(float(st_macro["F1-Score"].mean()), 4),
                    "Balanced_Acc_mean": round(float(st_macro["Balanced Acc"].mean()), 4),
                    "MCC_mean": round(float(st_macro["MCC"].mean()), 4),
                    "Notes": "Trained on individual observed endpoints across 3 seeds",
                })

        # 5. Ablations: No Dropout
        if "dropout" in data:
            dr_df = data["dropout"]
            no_drop_macro = dr_df[(dr_df["Endpoint"] == "MACRO AVERAGE") & (dr_df["Configuration"] == "no_dropout")]
            if len(no_drop_macro) > 0:
                comparison_rows.append({
                    "Model_Architecture": "DL Ablation: No Dropout (0.0)",
                    "Phase": "Phase 2 Member 4",
                    "Features": "2,053 canonical input",
                    "ROC-AUC_mean": round(float(no_drop_macro["ROC-AUC"].mean()), 4),
                    "PR-AUC_mean": round(float(no_drop_macro["PR-AUC"].mean()), 4),
                    "F1-Score_mean": round(float(no_drop_macro["F1-Score"].mean()), 4),
                    "Balanced_Acc_mean": round(float(no_drop_macro["Balanced Acc"].mean()), 4),
                    "MCC_mean": round(float(no_drop_macro["MCC"].mean()), 4),
                    "Notes": "Quantifies overfitting penalty when dropout regularization is removed",
                })

        # 6. Ablations: Unweighted Loss
        if "class_weight" in data:
            cw_df = data["class_weight"]
            unw_macro = cw_df[(cw_df["Endpoint"] == "MACRO AVERAGE") & (cw_df["Configuration"] == "unweighted_loss")]
            if len(unw_macro) > 0:
                comparison_rows.append({
                    "Model_Architecture": "DL Ablation: Unweighted Loss",
                    "Phase": "Phase 2 Member 4",
                    "Features": "2,053 canonical input",
                    "ROC-AUC_mean": round(float(unw_macro["ROC-AUC"].mean()), 4),
                    "PR-AUC_mean": round(float(unw_macro["PR-AUC"].mean()), 4),
                    "F1-Score_mean": round(float(unw_macro["F1-Score"].mean()), 4),
                    "Balanced_Acc_mean": round(float(unw_macro["Balanced Acc"].mean()), 4),
                    "MCC_mean": round(float(unw_macro["MCC"].mean()), 4),
                    "Notes": "Evaluates sensitivity and false-negative burden without class weighting",
                })

        # 7. Descriptor Ablation (Member 3 reported)
        comparison_rows.append({
            "Model_Architecture": "Hybrid Ablation: No Descriptors",
            "Phase": "Phase 2 Member 3 / Member 4",
            "Features": "128 (Embeddings only, no 5 descriptors)",
            "ROC-AUC_mean": 0.6738,
            "PR-AUC_mean": 0.2985,
            "F1-Score_mean": 0.2910,
            "Balanced_Acc_mean": 0.5890,
            "MCC_mean": 0.2680,
            "Notes": "Descriptors show negligible difference (-0.0015 on validation)",
        })

        df_final_comparison = pd.DataFrame(comparison_rows)
        comp_csv_path = self.results_dir / "final_model_comparison.csv"
        df_final_comparison.to_csv(comp_csv_path, index=False)
        print(f"[Saved] Final model comparison table saved to: {comp_csv_path}")

        # Build final_endpoint_metrics.csv
        # Matrix of Endpoint vs Model for ROC-AUC
        ep_rows = []
        for ep in self.target_assays:
            row: Dict[str, Any] = {"Endpoint": ep}
            if "standalone_rf" in data:
                rf_sub = data["standalone_rf"][data["standalone_rf"]["endpoint"] == ep]
                if len(rf_sub) > 0:
                    row["RF_ROC_AUC"] = round(float(rf_sub["test_roc_auc"].values[0]), 4)
                    row["RF_PR_AUC"] = round(float(rf_sub["test_pr_auc"].values[0]), 4)

            if "standalone_dl" in data:
                dl_sub = data["standalone_dl"][data["standalone_dl"]["Endpoint"] == ep]
                if len(dl_sub) > 0:
                    row["DL_ROC_AUC"] = round(float(dl_sub["ROC-AUC"].values[0]), 4)
                    row["DL_PR_AUC"] = round(float(dl_sub["PR-AUC"].values[0]), 4)

            if "hybrid" in data:
                hyb_sub = data["hybrid"][data["hybrid"]["Endpoint"] == ep]
                if len(hyb_sub) > 0:
                    row["Hybrid_ROC_AUC"] = round(float(hyb_sub["ROC-AUC"].values[0]), 4)
                    row["Hybrid_PR_AUC"] = round(float(hyb_sub["PR-AUC"].values[0]), 4)

            if "multitask_single_task" in data:
                st_sub = data["multitask_single_task"]
                st_ep = st_sub[(st_sub["Endpoint"] == ep) & (st_sub["Model"] == "Single-Task DL")]
                if len(st_ep) > 0:
                    row["SingleTask_ROC_AUC"] = round(float(st_ep["ROC-AUC"].mean()), 4)
                    row["SingleTask_PR_AUC"] = round(float(st_ep["PR-AUC"].mean()), 4)

            ep_rows.append(row)

        df_final_ep = pd.DataFrame(ep_rows)
        # Add Macro Average row
        macro_ep_row: Dict[str, Any] = {"Endpoint": "MACRO AVERAGE"}
        for c in df_final_ep.columns:
            if c != "Endpoint":
                macro_ep_row[c] = round(float(df_final_ep[c].mean()), 4)
        df_final_ep_full = pd.concat([df_final_ep, pd.DataFrame([macro_ep_row])], ignore_index=True)

        ep_csv_path = self.results_dir / "final_endpoint_metrics.csv"
        df_final_ep_full.to_csv(ep_csv_path, index=False)
        print(f"[Saved] Final per-endpoint metrics table saved to: {ep_csv_path}")

        # JSON Metadata summary
        summary_metadata = {
            "project": "Molecular Toxicity Prediction using Tox21 Dataset",
            "phase": "Phase 2 - Member 4 Evaluation & Ablation Studies",
            "certified_split": "Split B Frozen Scaffold Split (6,259 train / 782 val / 782 test)",
            "random_seeds": [42, 21, 7],
            "num_endpoints": 12,
            "models_evaluated": list(df_final_comparison["Model_Architecture"].unique()),
            "macro_performance_summary": df_final_comparison.to_dict(orient="records"),
        }
        json_path = self.results_dir / "experiment_summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_metadata, f, indent=2)
        print(f"[Saved] Experiment metadata summary saved to: {json_path}")

        return df_final_comparison, df_final_ep_full, summary_metadata


def main():
    comparison = ConsolidatedModelComparison()
    comparison.build_final_comparison()


if __name__ == "__main__":
    main()

