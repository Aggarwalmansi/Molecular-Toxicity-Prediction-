"""
evaluate_hybrid.py
==================
Scores the hybrid model and compares it against the two standalone models.

The three approaches being compared:
  1. Standalone Random Forest  - Phase 1, trained on 2053 fingerprint + descriptor features
  2. Standalone Deep Learning  - Member 2's MultiTaskToxicNet
  3. Hybrid                    - Random Forest on 128 DL features + 5 descriptors

The standalone numbers are read from the result files those members already
produced. We do not retrain or re-run their models, so the comparison uses
their own reported results.

All three were scored on the same 782 test molecules, using only observed
labels, with the same sklearn metric functions and the same 0.5 decision
threshold. That is what makes the comparison fair.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .hybrid_model import TARGET_ASSAYS, get_module_dir


def get_project_root():
    """Returns the repository root folder."""
    return Path(__file__).resolve().parents[3]


def evaluate_predictions(probabilities, labels, mask, threshold=0.5):
    """
    Computes per-endpoint metrics using only the observed labels.

    For each endpoint we select the molecules whose label was measured, then
    score the predictions for those molecules. Molecules with a missing label
    contribute nothing to that endpoint's score.

    ROC-AUC and PR-AUC use the predicted probabilities directly, so they do
    not depend on the threshold. F1, precision, recall, balanced accuracy and
    MCC need a hard 0/1 decision, which is where the threshold comes in.
    """
    rows = []

    for endpoint_index, endpoint in enumerate(TARGET_ASSAYS):
        observed = mask[:, endpoint_index]
        y_true = labels[observed, endpoint_index].astype(int)
        y_prob = probabilities[observed, endpoint_index]
        y_pred = (y_prob >= threshold).astype(int)

        # roc_auc_score fails if a split happens to contain only one class
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        else:
            roc_auc = 0.5
            pr_auc = 0.0

        rows.append(
            {
                "Endpoint": endpoint,
                "Test Actives": int((y_true == 1).sum()),
                "Test Inactives": int((y_true == 0).sum()),
                "Missing": int((~observed).sum()),
                "ROC-AUC": round(float(roc_auc), 4),
                "PR-AUC": round(float(pr_auc), 4),
                "F1-Score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "Balanced Acc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
                "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
                "Precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "Recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            }
        )

    results = pd.DataFrame(rows)

    # Macro average = plain mean across the 12 endpoints, so every endpoint
    # counts equally regardless of how many molecules it has.
    macro = {"Endpoint": "MACRO AVERAGE"}
    for column in results.columns:
        if column == "Endpoint":
            continue
        if column in ["Test Actives", "Test Inactives", "Missing"]:
            macro[column] = int(results[column].sum())
        else:
            macro[column] = round(float(results[column].mean()), 4)

    return pd.concat([results, pd.DataFrame([macro])], ignore_index=True)


def load_standalone_rf_results():
    """
    Reads the Phase 1 Random Forest test results from results/metrics.csv.

    Column names are renamed to match our own table so the three models can be
    compared side by side.
    """
    metrics_file = get_project_root() / "results" / "metrics.csv"
    all_metrics = pd.read_csv(metrics_file)
    forest_metrics = all_metrics[all_metrics["model"] == "Random Forest"]

    renamed = pd.DataFrame(
        {
            "Endpoint": forest_metrics["endpoint"].values,
            "ROC-AUC": forest_metrics["test_roc_auc"].values,
            "PR-AUC": forest_metrics["test_pr_auc"].values,
            "F1-Score": forest_metrics["f1_score"].values,
            "Balanced Acc": forest_metrics["balanced_accuracy"].values,
            "MCC": forest_metrics["mcc"].values,
        }
    )
    return renamed


def load_standalone_dl_results():
    """Reads Member 2's test results, dropping their macro average row."""
    results_file = (
        get_project_root()
        / "phase2"
        / "deep_learning"
        / "reports"
        / "test_evaluation.csv"
    )
    dl_results = pd.read_csv(results_file)
    return dl_results[dl_results["Endpoint"] != "MACRO AVERAGE"].reset_index(drop=True)


def build_comparison(hybrid_results, metric="ROC-AUC"):
    """
    Builds a per-endpoint table comparing the three approaches on one metric.

    Also adds the hybrid-minus-standalone differences, which is how we judge
    whether the hybrid actually helped.
    """
    standalone_rf = load_standalone_rf_results().set_index("Endpoint")
    standalone_dl = load_standalone_dl_results().set_index("Endpoint")
    hybrid = hybrid_results[hybrid_results["Endpoint"] != "MACRO AVERAGE"].set_index("Endpoint")

    comparison = pd.DataFrame(
        {
            "Standalone RF": standalone_rf[metric],
            "Standalone DL": standalone_dl[metric],
            "Hybrid": hybrid[metric],
        }
    )
    comparison["Hybrid - RF"] = (comparison["Hybrid"] - comparison["Standalone RF"]).round(4)
    comparison["Hybrid - DL"] = (comparison["Hybrid"] - comparison["Standalone DL"]).round(4)

    macro_row = pd.DataFrame(
        [comparison.mean().round(4)], index=pd.Index(["MACRO AVERAGE"], name="Endpoint")
    )
    return pd.concat([comparison, macro_row])


def save_predictions(probabilities, labels, mask, mol_ids, output_file):
    """
    Saves the hybrid test predictions in long format, one row per
    molecule-endpoint pair, so Member 5 can reuse them without retraining.

    The observed column records whether a true label existed, so nobody
    accidentally scores against a missing label.
    """
    rows = []
    for row_index in range(len(mol_ids)):
        for endpoint_index, endpoint in enumerate(TARGET_ASSAYS):
            is_observed = bool(mask[row_index, endpoint_index])
            true_label = labels[row_index, endpoint_index]
            rows.append(
                {
                    "mol_id": mol_ids[row_index],
                    "endpoint": endpoint,
                    "predicted_probability": round(
                        float(probabilities[row_index, endpoint_index]), 6
                    ),
                    "predicted_label": int(probabilities[row_index, endpoint_index] >= 0.5),
                    "true_label": "" if not is_observed else int(true_label),
                    "observed": is_observed,
                }
            )

    predictions = pd.DataFrame(rows)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_file, index=False)
    return predictions
