"""
metrics.py
==========
Comprehensive evaluation metric functions for Tox21 toxicity prediction.
Supports:
- Threshold-independent metrics: ROC-AUC, PR-AUC / Average Precision.
- Threshold-dependent metrics: F1, Precision, Recall, Balanced Accuracy, MCC.
- Confusion matrix counts: TP, TN, FP, FN.
- Validation-based threshold optimization (F1 maximization).
- Non-parametric bootstrap confidence intervals.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Union[float, int]]:
    """
    Compute full suite of classification metrics for binary predictions.

    Parameters
    ----------
    y_true : 1D array of ground truth binary labels (0 or 1).
    y_prob : 1D array of predicted probabilities in [0, 1].
    threshold : Decision threshold for binarizing probabilities.

    Returns
    -------
    dict with metrics and confusion counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    # Area under curves
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = 0.5
        pr_auc = float(n_pos / max(1, len(y_true)))

    # Threshold-dependent metrics
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "F1-Score": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Balanced Acc": round(bal_acc, 4),
        "MCC": round(mcc, 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Positives": n_pos,
        "Negatives": n_neg,
        "Threshold": round(threshold, 4),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    steps: int = 100,
) -> Tuple[float, float]:
    """
    Determine the optimal classification threshold using validation predictions.

    Parameters
    ----------
    y_true : Ground truth validation labels.
    y_prob : Predicted validation probabilities.
    metric : Metric to optimize ('f1' or 'balanced_acc').
    steps : Number of candidate thresholds to test in [0.01, 0.99].

    Returns
    -------
    best_threshold : float
    best_score : float
    """
    thresholds = np.linspace(0.01, 0.99, steps)
    best_thresh = 0.5
    best_val = -1.0

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "balanced_acc":
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric '{metric}'")

        if score > best_val:
            best_val = score
            best_thresh = float(th)

    return round(best_thresh, 4), round(best_val, 4)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "ROC-AUC",
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute non-parametric bootstrap confidence interval for a metric.

    Returns
    -------
    (mean_score, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores: List[float] = []

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        yt_b = y_true[idx]
        yp_b = y_prob[idx]

        if len(np.unique(yt_b)) < 2:
            continue

        if metric == "ROC-AUC":
            val = roc_auc_score(yt_b, yp_b)
        elif metric == "PR-AUC":
            val = average_precision_score(yt_b, yp_b)
        elif metric == "F1-Score":
            val = f1_score(yt_b, (yp_b >= threshold).astype(int), zero_division=0)
        else:
            m = compute_classification_metrics(yt_b, yp_b, threshold)
            val = m.get(metric, 0.0)

        scores.append(val)

    if len(scores) == 0:
        return 0.5, 0.5, 0.5

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, alpha * 100))
    upper = float(np.percentile(scores, (1.0 - alpha) * 100))
    mean_val = float(np.mean(scores))

    return round(mean_val, 4), round(lower, 4), round(upper, 4)


def evaluate_multitask_predictions(
    probabilities: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    target_assays: List[str],
    thresholds: Optional[Union[float, Dict[str, float], List[float]]] = None,
) -> pd.DataFrame:
    """
    Evaluate multi-task predictions per endpoint across observed labels only.

    Parameters
    ----------
    probabilities : (N, 12) array of model probabilities.
    labels : (N, 12) array of ground truth labels (0, 1, or NaN).
    mask : (N, 12) boolean observation mask.
    target_assays : List of 12 assay names.
    thresholds : Optional threshold(s). Can be float, dict {assay: th}, or list of 12 floats.

    Returns
    -------
    pd.DataFrame containing per-endpoint metrics and a MACRO AVERAGE row.
    """
    rows = []

    for ep_idx, ep in enumerate(target_assays):
        obs = mask[:, ep_idx].astype(bool)
        y_true = labels[obs, ep_idx].astype(int)
        y_prob = probabilities[obs, ep_idx]

        # Determine threshold
        if thresholds is None:
            th = 0.5
        elif isinstance(thresholds, dict):
            th = thresholds.get(ep, 0.5)
        elif isinstance(thresholds, (list, np.ndarray)):
            th = float(thresholds[ep_idx])
        else:
            th = float(thresholds)

        res = compute_classification_metrics(y_true, y_prob, threshold=th)
        res["Endpoint"] = ep
        res["Missing"] = int((~obs).sum())
        rows.append(res)

    df = pd.DataFrame(rows)

    # Reorder columns
    col_order = [
        "Endpoint", "Positives", "Negatives", "Missing", "Threshold",
        "ROC-AUC", "PR-AUC", "F1-Score", "Precision", "Recall",
        "Balanced Acc", "MCC", "TP", "FP", "TN", "FN"
    ]
    cols = [c for c in col_order if c in df.columns] + [c for c in df.columns if c not in col_order]
    df = df[cols]

    # Calculate Macro Average
    macro: Dict[str, Any] = {"Endpoint": "MACRO AVERAGE"}
    sum_cols = ["Positives", "Negatives", "Missing", "TP", "FP", "TN", "FN"]
    mean_cols = ["Threshold", "ROC-AUC", "PR-AUC", "F1-Score", "Precision", "Recall", "Balanced Acc", "MCC"]

    for col in sum_cols:
        if col in df.columns:
            macro[col] = int(df[col].sum())
    for col in mean_cols:
        if col in df.columns:
            macro[col] = round(float(df[col].mean()), 4)

    return pd.concat([df, pd.DataFrame([macro])], ignore_index=True)

