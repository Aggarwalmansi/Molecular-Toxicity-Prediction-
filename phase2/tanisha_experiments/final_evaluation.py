"""
final_evaluation.py
===================
A5: Full comparison of every candidate on all six metrics, plus per-endpoint
threshold tuning.

Why tune the threshold: at the fixed 0.5 cutoff the models find only about a
quarter of the toxic molecules. For a toxicity screen that is the expensive
kind of mistake. Lowering the cutoff trades precision for recall. We choose
one cutoff per endpoint by maximising F1 on the VALIDATION split, then apply
those cutoffs unchanged to the test split.

Both the fixed-threshold and tuned-threshold results are reported, so nobody
can mistake a threshold change for a genuine model improvement: ROC-AUC and
PR-AUC do not depend on the threshold at all and are identical in both.

Run from the repository root:
    python -m phase2.experiments.final_evaluation
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

from phase2.experiments.ensemble import (
    DL_SIX_RUNS,
    blend,
    load_predictions,
    make_wide,
)
from phase2.data.src.data_loader import METRICS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def score_all_metrics(wide, score_column, thresholds=None):
    """
    Computes every metric per endpoint, using observed labels only.

    thresholds is a dictionary of endpoint -> cutoff. When it is None we use
    0.5 everywhere.
    """
    observed = wide[wide["observed"]]
    rows = []

    for endpoint, group in observed.groupby("endpoint", observed=True):
        y_true = group["true_label"].astype(int).values
        y_prob = group[score_column].values
        if len(np.unique(y_true)) < 2:
            continue

        cutoff = 0.5 if thresholds is None else thresholds.get(endpoint, 0.5)
        y_pred = (y_prob >= cutoff).astype(int)

        rows.append(
            {
                "endpoint": endpoint,
                "threshold": round(float(cutoff), 4),
                "roc_auc": roc_auc_score(y_true, y_prob),
                "pr_auc": average_precision_score(y_true, y_prob),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "balanced_acc": balanced_accuracy_score(y_true, y_pred),
                "mcc": matthews_corrcoef(y_true, y_pred),
            }
        )

    return pd.DataFrame(rows)


def choose_thresholds_on_validation(val_wide, score_column):
    """
    Picks one cutoff per endpoint by maximising F1 on validation.

    We search the candidate cutoffs that actually occur in the validation
    predictions, which is enough to find the best achievable F1.
    """
    observed = val_wide[val_wide["observed"]]
    thresholds = {}

    for endpoint, group in observed.groupby("endpoint", observed=True):
        y_true = group["true_label"].astype(int).values
        y_prob = group[score_column].values
        if len(np.unique(y_true)) < 2:
            thresholds[endpoint] = 0.5
            continue

        candidates = np.unique(np.quantile(y_prob, np.linspace(0.01, 0.99, 99)))
        best_cutoff = 0.5
        best_f1 = -1.0
        for cutoff in candidates:
            score = f1_score(y_true, (y_prob >= cutoff).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_cutoff = float(cutoff)
        thresholds[endpoint] = best_cutoff

    return thresholds


def get_candidates(val, test):
    """
    Builds the score column for every candidate model on both splits.

    Returns a dictionary of name -> (val table, test table, column name).
    """
    candidates = {}

    # Single models that already have a column
    for name in ["rf_phase1", "dl_member2_checkpoint", "dl_weighted_seed42"]:
        candidates[name] = (val, test, name)

    # Deep learning six-run ensemble
    candidates["dl6_rank_avg"] = (
        blend(val, DL_SIX_RUNS, None, True),
        blend(test, DL_SIX_RUNS, None, True),
        "blend_score",
    )

    # The validation-selected winner from ensemble.py
    rf_and_dl = ["rf_phase1"] + DL_SIX_RUNS
    weights = [3] + [1] * len(DL_SIX_RUNS)
    candidates["rf_plus_dl6_rank_w3"] = (
        blend(val, rf_and_dl, weights, True),
        blend(test, rf_and_dl, weights, True),
        "blend_score",
    )

    return candidates


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()
    val = make_wide(predictions, "val")
    test = make_wide(predictions, "test")

    candidates = get_candidates(val, test)

    fixed_rows = []
    tuned_rows = []
    per_endpoint_tables = []

    for name in candidates:
        val_table, test_table, column = candidates[name]

        # Fixed 0.5 threshold
        test_fixed = score_all_metrics(test_table, column)
        val_fixed = score_all_metrics(val_table, column)
        fixed_rows.append(
            {
                "model": name,
                "val_roc_auc": round(val_fixed["roc_auc"].mean(), 4),
                "val_pr_auc": round(val_fixed["pr_auc"].mean(), 4),
                "test_roc_auc": round(test_fixed["roc_auc"].mean(), 4),
                "test_pr_auc": round(test_fixed["pr_auc"].mean(), 4),
                "test_f1": round(test_fixed["f1"].mean(), 4),
                "test_precision": round(test_fixed["precision"].mean(), 4),
                "test_recall": round(test_fixed["recall"].mean(), 4),
                "test_mcc": round(test_fixed["mcc"].mean(), 4),
            }
        )

        # Thresholds chosen on validation, applied to test
        thresholds = choose_thresholds_on_validation(val_table, column)
        test_tuned = score_all_metrics(test_table, column, thresholds)
        tuned_rows.append(
            {
                "model": name,
                "test_roc_auc": round(test_tuned["roc_auc"].mean(), 4),
                "test_pr_auc": round(test_tuned["pr_auc"].mean(), 4),
                "test_f1": round(test_tuned["f1"].mean(), 4),
                "test_precision": round(test_tuned["precision"].mean(), 4),
                "test_recall": round(test_tuned["recall"].mean(), 4),
                "test_mcc": round(test_tuned["mcc"].mean(), 4),
            }
        )

        test_tuned["model"] = name
        per_endpoint_tables.append(test_tuned)

    fixed = pd.DataFrame(fixed_rows)
    tuned = pd.DataFrame(tuned_rows)

    fixed.to_csv(METRICS_DIR / "final_comparison_fixed_threshold.csv", index=False)
    tuned.to_csv(METRICS_DIR / "final_comparison_tuned_threshold.csv", index=False)
    pd.concat(per_endpoint_tables).to_csv(
        METRICS_DIR / "final_per_endpoint_tuned.csv", index=False
    )

    print("=" * 96)
    print("FIXED THRESHOLD 0.5")
    print("=" * 96)
    print(fixed.to_string(index=False))

    print()
    print("=" * 96)
    print("THRESHOLD TUNED PER ENDPOINT ON VALIDATION, APPLIED TO TEST")
    print("=" * 96)
    print(tuned.to_string(index=False))

    print()
    print("ROC-AUC and PR-AUC are identical in both tables because they do not")
    print("depend on the threshold. Only F1, precision, recall and MCC change.")


if __name__ == "__main__":
    main()
