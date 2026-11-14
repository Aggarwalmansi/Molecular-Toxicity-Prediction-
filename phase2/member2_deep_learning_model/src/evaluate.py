"""
evaluate.py
===========
Rigorous holdout test evaluation for MultiTaskToxicNet across all 12 Tox21 endpoints.
Calculates ROC-AUC, PR-AUC, F1-Score, Balanced Accuracy, MCC, Precision, and Recall.
Generates comprehensive diagnostic reports and publication-ready figures.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

import torch

from phase2.member1_data_pipeline.src.data_loader import TARGET_ASSAYS
from .model import MultiTaskToxicNet


def evaluate_multitask_model(
    model: MultiTaskToxicNet,
    loader: Any,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    target_assays: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Perform full benchmark evaluation on the certified holdout test set (782 molecules).

    Parameters
    ----------
    model : MultiTaskToxicNet
        Trained multi-task neural network.
    loader : Tox21DataLoader
        Phase 2 Member 1 data loader.
    output_dir : Path or str, optional
        Directory to save results.
    device : torch.device, optional
        Computing device.
    threshold : float, default 0.5
        Binary decision threshold for classification metrics.

    Returns
    -------
    pd.DataFrame
        Table with metrics for all 12 endpoints + Macro Average row.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    base_dir = Path(__file__).resolve().parent.parent
    report_dir = Path(output_dir or (base_dir / "reports"))
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    # Load holdout test features and ground truth labels
    X_test, mids_x = loader.load_features("combined", split="test")
    y_test, mask_test, mids_y = loader.load_labels(split="test")
    assert mids_x == mids_y, "Test compound ID misalignment!"

    with torch.no_grad():
        x_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        y_prob = torch.sigmoid(logits).cpu().numpy()

    records = []
    assays = target_assays or getattr(loader, "TARGET_ASSAYS", TARGET_ASSAYS)

    for ep_idx, ep in enumerate(assays):
        obs = mask_test[:, ep_idx].astype(bool)
        y_true = y_test[obs, ep_idx]
        p_pred = y_prob[obs, ep_idx]
        y_bin = (p_pred >= threshold).astype(int)

        n_pos = int((y_true == 1.0).sum())
        n_neg = int((y_true == 0.0).sum())
        n_miss = int((~obs).sum())

        roc_auc = float(roc_auc_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_true, p_pred)) if len(np.unique(y_true)) > 1 else 0.0
        f1 = float(f1_score(y_true, y_bin, zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_bin))
        mcc = float(matthews_corrcoef(y_true, y_bin))
        prec = float(precision_score(y_true, y_bin, zero_division=0))
        rec = float(recall_score(y_true, y_bin, zero_division=0))

        records.append({
            "Endpoint": ep,
            "Test Actives": n_pos,
            "Test Inactives": n_neg,
            "Missing": n_miss,
            "ROC-AUC": round(roc_auc, 4),
            "PR-AUC": round(pr_auc, 4),
            "F1-Score": round(f1, 4),
            "Balanced Acc": round(bal_acc, 4),
            "MCC": round(mcc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
        })

    df_eval = pd.DataFrame(records)

    # Append Macro Average row
    macro_row = {
        "Endpoint": "MACRO AVERAGE",
        "Test Actives": int(df_eval["Test Actives"].sum()),
        "Test Inactives": int(df_eval["Test Inactives"].sum()),
        "Missing": int(df_eval["Missing"].sum()),
        "ROC-AUC": round(df_eval["ROC-AUC"].mean(), 4),
        "PR-AUC": round(df_eval["PR-AUC"].mean(), 4),
        "F1-Score": round(df_eval["F1-Score"].mean(), 4),
        "Balanced Acc": round(df_eval["Balanced Acc"].mean(), 4),
        "MCC": round(df_eval["MCC"].mean(), 4),
        "Precision": round(df_eval["Precision"].mean(), 4),
        "Recall": round(df_eval["Recall"].mean(), 4),
    }
    df_eval_full = pd.concat([df_eval, pd.DataFrame([macro_row])], ignore_index=True)

    # Save CSV
    csv_path = report_dir / "test_evaluation.csv"
    df_eval_full.to_csv(csv_path, index=False)
    print(f"[evaluate] Saved full evaluation report to {csv_path}")

    # Plot per-endpoint performance
    _plot_per_endpoint(df_eval, plot_dir)

    return df_eval_full


def _plot_per_endpoint(df: pd.DataFrame, plot_dir: Path):
    """Plot per-endpoint ROC-AUC and PR-AUC horizontal comparison."""
    endpoints = df["Endpoint"].tolist()
    roc_scores = df["ROC-AUC"].tolist()
    pr_scores = df["PR-AUC"].tolist()

    x = np.arange(len(endpoints))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, roc_scores, width, label="ROC-AUC", color="#1f77b4")
    bars2 = ax.bar(x + width/2, pr_scores, width, label="PR-AUC", color="#ff7f0e")

    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("MultiTaskToxicNet Test Holdout Performance Across 12 Tox21 Endpoints", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(endpoints, rotation=45, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="Random Baseline (ROC=0.5)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    # Add score annotations above bars
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(plot_dir / "per_endpoint_roc_pr.png", dpi=200)
    plt.close()
