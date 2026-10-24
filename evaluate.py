"""
evaluate.py
===========
Tox21 Molecular Toxicity Prediction - Member 4: Evaluation, Statistics and Error Analysis

Primary Ownership: Proving whether models work

Tasks & Capabilities:
---------------------
1. Implement ROC-AUC and PR-AUC / Average Precision per endpoint with 95% bootstrap confidence intervals.
2. Ensure missing labels (NaN) are strictly excluded from both training and evaluation for each target.
3. Guarantee paired model comparison on the exact same test compounds.
4. Compute threshold-dependent metrics (F1, Precision, Recall, Specificity, Balanced Accuracy, MCC).
5. Conduct paired statistical significance testing (Bootstrap difference test & Wilcoxon signed-rank test).
6. Export per-compound test predictions for deep chemical error analysis.
7. Run a comparative Scaffold Split vs. Random Split experiment (Task 6) to quantify the scaffold generalization penalty.
8. Generate publication-grade diagnostic plots (ROC curves, PR curves, model comparison with CIs, class distributions,
   performance vs. imbalance correlation, and confusion matrices).

Outputs Owned:
--------------
- evaluate.py
- results/metrics.csv
- results/summary_metrics.csv
- results/statistical_tests.csv
- results/split_comparison.csv
- results/test_predictions.csv
- results/plots/*.png
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
)
from sklearn.model_selection import train_test_split

# ==============================================================================
# Configuration & Constants
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "tox21_member2_handoff" / "artifacts"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results"

# All 12 Tox21 toxicological endpoints
TARGET_ASSAYS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

# Style configuration for clean publication-quality plots
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
})

MODEL_COLORS = {
    "Logistic Regression": "#2b5c8f",
    "Decision Tree": "#d95f02",
    "Random Forest": "#1b9e77",
}


# ==============================================================================
# Step 1: Data Loader & Split Preparation
# ==============================================================================

def load_data(artifacts_dir: Path):
    """
    Loads preprocessed feature matrix and split metadata prepared by Members 1 & 2.
    """
    labels_path = artifacts_dir / "labels_and_splits.parquet"
    features_path = artifacts_dir / "X_combined.parquet"

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels parquet artifact at: {labels_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Missing features parquet artifact at: {features_path}")

    print(f"[DataLoader] Reading metadata from: {labels_path.name}")
    labels_df = pd.read_parquet(labels_path)

    print(f"[DataLoader] Reading canonical features from: {features_path.name}")
    features_df = pd.read_parquet(features_path)

    feat_cols = [c for c in features_df.columns if c != "mol_id"]
    X = features_df[feat_cols].values.astype(np.float64)

    return X, labels_df


def get_models(random_state: int = 42):
    """
    Returns standard scikit-learn models matching Member 3 specification.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


# ==============================================================================
# Step 2: Statistical Bootstrap Confidence Intervals
# ==============================================================================

def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Computes empirical bootstrap confidence intervals for ROC-AUC and PR-AUC.
    Returns:
        ((roc_lower, roc_upper), (pr_lower, pr_upper))
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)
    roc_scores = []
    pr_scores = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b_true = y_true[idx]
        y_b_prob = y_prob[idx]

        # Ensure both classes are sampled in the bootstrap replica
        if len(np.unique(y_b_true)) < 2:
            continue

        try:
            roc_s = roc_auc_score(y_b_true, y_b_prob)
            pr_s = average_precision_score(y_b_true, y_b_prob)
            roc_scores.append(roc_s)
            pr_scores.append(pr_s)
        except Exception:
            continue

    alpha = 1.0 - confidence_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    if len(roc_scores) >= 50:
        roc_ci = (
            float(np.percentile(roc_scores, lower_pct)),
            float(np.percentile(roc_scores, upper_pct)),
        )
        pr_ci = (
            float(np.percentile(pr_scores, lower_pct)),
            float(np.percentile(pr_scores, upper_pct)),
        )
    else:
        roc_ci = (np.nan, np.nan)
        pr_ci = (np.nan, np.nan)

    return roc_ci, pr_ci


def compute_paired_bootstrap_test(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap hypothesis test for difference in ROC-AUC and PR-AUC between Model A and Model B:
    H0: Metric(A) - Metric(B) == 0.
    Returns:
        (delta_roc, p_value_roc, delta_pr, p_value_pr)
    """
    rng = np.random.default_rng(random_state)
    n_samples = len(y_true)

    base_roc_a = roc_auc_score(y_true, prob_a)
    base_roc_b = roc_auc_score(y_true, prob_b)
    delta_roc = base_roc_a - base_roc_b

    base_pr_a = average_precision_score(y_true, prob_a)
    base_pr_b = average_precision_score(y_true, prob_b)
    delta_pr = base_pr_a - base_pr_b

    roc_diffs = []
    pr_diffs = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        y_b = y_true[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            r_a = roc_auc_score(y_b, prob_a[idx])
            r_b = roc_auc_score(y_b, prob_b[idx])
            p_a = average_precision_score(y_b, prob_a[idx])
            p_b = average_precision_score(y_b, prob_b[idx])
            roc_diffs.append(r_a - r_b)
            pr_diffs.append(p_a - p_b)
        except Exception:
            continue

    roc_diffs = np.array(roc_diffs)
    pr_diffs = np.array(pr_diffs)

    # Empirical two-tailed p-value
    p_val_roc = 2.0 * min(np.mean(roc_diffs <= 0), np.mean(roc_diffs >= 0)) if len(roc_diffs) > 0 else np.nan
    p_val_pr = 2.0 * min(np.mean(pr_diffs <= 0), np.mean(pr_diffs >= 0)) if len(pr_diffs) > 0 else np.nan

    p_val_roc = min(1.0, float(p_val_roc))
    p_val_pr = min(1.0, float(p_val_pr))

    return delta_roc, p_val_roc, delta_pr, p_val_pr


# ==============================================================================
# Step 3: Comprehensive Evaluation Loop
# ==============================================================================

def evaluate_all_endpoints(
    X: np.ndarray,
    labels_df: pd.DataFrame,
    n_bootstraps: int = 1000,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Main evaluation routine:
    - Trains LR, DT, RF on observed training compounds for each assay.
    - Evaluates on identical observed test compounds.
    - Computes ROC-AUC, PR-AUC, 95% CIs, and threshold metrics.
    - Performs pairwise statistical significance tests.
    - Collects compound predictions for error analysis.
    """
    train_mask = (labels_df["split"] == "train").values
    test_mask = (labels_df["split"] == "test").values

    detailed_metrics = []
    statistical_tests = []
    prediction_records = []
    curve_data = {}  # {assay: {model: {"fpr", "tpr", "precision", "recall"}}}

    print("=" * 85)
    print("      MEMBER 4: RIGOROUS EVALUATION & STATISTICAL ANALYSIS PIPELINE")
    print("=" * 85)

    for idx, assay in enumerate(TARGET_ASSAYS, start=1):
        t0 = time.time()
        y_all = labels_df[assay].values

        # Strict requirement: Missing labels (NaN) excluded per target
        train_obs = train_mask & ~np.isnan(y_all)
        test_obs = test_mask & ~np.isnan(y_all)

        X_train, y_train = X[train_obs], y_all[train_obs]
        X_test, y_test = X[test_obs], y_all[test_obs]

        test_mol_ids = labels_df.loc[test_obs, "mol_id"].values
        test_smiles = labels_df.loc[test_obs, "smiles"].values
        test_scaffold = labels_df.loc[test_obs, "scaffold_smiles"].values

        n_train_pos = int((y_train == 1).sum())
        n_train_neg = int((y_train == 0).sum())
        n_test_pos = int((y_test == 1).sum())
        n_test_neg = int((y_test == 0).sum())
        test_pos_rate = n_test_pos / len(y_test)

        print(
            f"[{idx:2d}/12] Endpoint: {assay:<14s} | "
            f"Test Samples: {len(y_test)} (Pos: {n_test_pos}, Neg: {n_test_neg}, Prev: {test_pos_rate:.1%})"
        )

        models = get_models(random_state=random_state)
        fitted_probs = {}
        fitted_preds = {}

        curve_data[assay] = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)

            fitted_probs[model_name] = probs
            fitted_preds[model_name] = preds

            # Core metrics
            roc_auc = float(roc_auc_score(y_test, probs))
            pr_auc = float(average_precision_score(y_test, probs))

            # 95% Bootstrap Confidence Intervals
            (roc_ci_low, roc_ci_high), (pr_ci_low, pr_ci_high) = compute_bootstrap_ci(
                y_test, probs, n_bootstraps=n_bootstraps, random_state=random_state
            )

            # Threshold-dependent metrics (at 0.5 classification threshold)
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            bal_acc = balanced_accuracy_score(y_test, preds)
            mcc = matthews_corrcoef(y_test, preds)

            # ROC & PR curves
            fpr, tpr, _ = roc_curve(y_test, probs)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, probs)
            curve_data[assay][model_name] = {
                "fpr": fpr,
                "tpr": tpr,
                "precision": precision_vals,
                "recall": recall_vals,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            }

            detailed_metrics.append({
                "endpoint": assay,
                "model": model_name,
                "train_samples": len(y_train),
                "train_positives": n_train_pos,
                "train_negatives": n_train_neg,
                "test_samples": len(y_test),
                "test_positives": n_test_pos,
                "test_negatives": n_test_neg,
                "positive_prevalence": round(test_pos_rate, 4),
                "random_chance_pr_auc": round(test_pos_rate, 4),
                "test_roc_auc": round(roc_auc, 4),
                "roc_auc_ci_lower": round(roc_ci_low, 4),
                "roc_auc_ci_upper": round(roc_ci_high, 4),
                "test_pr_auc": round(pr_auc, 4),
                "pr_auc_ci_lower": round(pr_ci_low, 4),
                "pr_auc_ci_upper": round(pr_ci_high, 4),
                "pr_auc_uplift_over_chance": round(pr_auc - test_pos_rate, 4),
                "f1_score": round(f1, 4),
                "precision": round(prec, 4),
                "recall_sensitivity": round(rec, 4),
                "specificity": round(spec, 4),
                "balanced_accuracy": round(bal_acc, 4),
                "mcc": round(mcc, 4),
            })

        # Record compound predictions for error analysis
        for m_idx in range(len(y_test)):
            prediction_records.append({
                "mol_id": test_mol_ids[m_idx],
                "endpoint": assay,
                "smiles": test_smiles[m_idx],
                "scaffold_smiles": test_scaffold[m_idx],
                "y_true": int(y_test[m_idx]),
                "prob_lr": round(float(fitted_probs["Logistic Regression"][m_idx]), 4),
                "prob_dt": round(float(fitted_probs["Decision Tree"][m_idx]), 4),
                "prob_rf": round(float(fitted_probs["Random Forest"][m_idx]), 4),
                "pred_lr": int(fitted_preds["Logistic Regression"][m_idx]),
                "pred_dt": int(fitted_preds["Decision Tree"][m_idx]),
                "pred_rf": int(fitted_preds["Random Forest"][m_idx]),
            })

        # Paired statistical hypothesis tests: RF vs LR, RF vs DT, LR vs DT
        comparisons = [
            ("Random Forest", "Logistic Regression"),
            ("Random Forest", "Decision Tree"),
            ("Logistic Regression", "Decision Tree"),
        ]
        for mod_a, mod_b in comparisons:
            d_roc, p_roc, d_pr, p_pr = compute_paired_bootstrap_test(
                y_test,
                fitted_probs[mod_a],
                fitted_probs[mod_b],
                n_bootstraps=n_bootstraps,
                random_state=random_state,
            )

            # Paired Wilcoxon signed-rank test on squared error / Brier loss
            loss_a = (y_test - fitted_probs[mod_a]) ** 2
            loss_b = (y_test - fitted_probs[mod_b]) ** 2
            diff_loss = loss_a - loss_b
            if np.all(diff_loss == 0):
                p_wilcoxon = 1.0
            else:
                try:
                    _, p_wilcoxon = stats.wilcoxon(loss_a, loss_b)
                except Exception:
                    p_wilcoxon = np.nan

            statistical_tests.append({
                "endpoint": assay,
                "comparison": f"{mod_a} vs {mod_b}",
                "delta_roc_auc": round(d_roc, 4),
                "p_value_roc_auc": round(p_roc, 4),
                "delta_pr_auc": round(d_pr, 4),
                "p_value_pr_auc": round(p_pr, 4),
                "p_value_wilcoxon_loss": round(float(p_wilcoxon), 4) if not np.isnan(p_wilcoxon) else np.nan,
                "significant_at_05_roc": bool(p_roc < 0.05),
                "significant_at_05_pr": bool(p_pr < 0.05),
            })

        elapsed = time.time() - t0
        print(f"       Completed in {elapsed:.1f}s")

    metrics_df = pd.DataFrame(detailed_metrics)
    stats_df = pd.DataFrame(statistical_tests)
    preds_df = pd.DataFrame(prediction_records)

    return metrics_df, stats_df, preds_df, curve_data


# ==============================================================================
# Step 4: Random-Split Benchmark Experiment (Task 6)
# ==============================================================================

def run_random_split_experiment(
    X: np.ndarray,
    labels_df: pd.DataFrame,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Executes Task 6:
    Creates a random 80/10/10 split on all 7,823 compounds, retrains models,
    and compares performance against Scaffold Splitting to quantify the
    'scaffold generalization gap' (optimism bias of random splitting).
    """
    print("\n" + "=" * 85)
    print("      TASK 6: SCAFFOLD SPLIT VS. RANDOM SPLIT EXPERIMENT")
    print("=" * 85)

    n_samples = len(labels_df)
    indices = np.arange(n_samples)

    # 80% train, 20% temp (which will be 10% val, 10% test)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.20, random_state=random_state, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=random_state, shuffle=True
    )

    random_split_mask = np.full(n_samples, "train", dtype=object)
    random_split_mask[val_idx] = "val"
    random_split_mask[test_idx] = "test"

    train_mask = (random_split_mask == "train")
    test_mask = (random_split_mask == "test")

    results = []

    for assay in TARGET_ASSAYS:
        y_all = labels_df[assay].values

        train_obs = train_mask & ~np.isnan(y_all)
        test_obs = test_mask & ~np.isnan(y_all)

        X_train, y_train = X[train_obs], y_all[train_obs]
        X_test, y_test = X[test_obs], y_all[test_obs]

        models = get_models(random_state=random_state)
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]

            roc_auc = float(roc_auc_score(y_test, probs))
            pr_auc = float(average_precision_score(y_test, probs))

            results.append({
                "endpoint": assay,
                "model": model_name,
                "random_test_roc_auc": round(roc_auc, 4),
                "random_test_pr_auc": round(pr_auc, 4),
            })

    return pd.DataFrame(results)


# ==============================================================================
# Step 5: Publication-Grade Visualizations (Task 4)
# ==============================================================================

def generate_plots(
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    curve_data: Dict,
    split_comp_df: pd.DataFrame,
    output_dir: Path,
):
    """
    Generates all diagnostic figures required for Member 4 ownership.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Plotting] Generating figures in: {plots_dir}")

    # --------------------------------------------------------------------------
    # Figure 1: Model Comparison (ROC-AUC and PR-AUC with 95% Bootstrap CIs)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)

    # Subplot 1A: ROC-AUC
    sns.barplot(
        data=metrics_df,
        x="endpoint",
        y="test_roc_auc",
        hue="model",
        palette=MODEL_COLORS,
        ax=axes[0],
    )
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Random Guess (0.50)")
    axes[0].set_ylabel("Test ROC-AUC")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_title("A. Per-Endpoint ROC-AUC Comparison Across Models (with Scaffold Split)", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=True)

    # Subplot 1B: PR-AUC
    sns.barplot(
        data=metrics_df,
        x="endpoint",
        y="test_pr_auc",
        hue="model",
        palette=MODEL_COLORS,
        ax=axes[1],
    )
    axes[1].set_ylabel("Test PR-AUC (Average Precision)")
    axes[1].set_ylim(0.0, 0.8)
    axes[1].set_title("B. Per-Endpoint PR-AUC Comparison Across Models", fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)
    plt.xticks(rotation=30, ha="right")

    plt.tight_layout()
    fig_path1 = plots_dir / "model_comparison_roc_pr_auc.png"
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {fig_path1.name}")

    # --------------------------------------------------------------------------
    # Figure 2: ROC Curves (3x4 Grid for all 12 Endpoints)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes = axes.flatten()

    for idx, assay in enumerate(TARGET_ASSAYS):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Chance")
        for model_name, col in MODEL_COLORS.items():
            dat = curve_data[assay][model_name]
            ax.plot(
                dat["fpr"],
                dat["tpr"],
                label=f"{model_name[:8]} ({dat['roc_auc']:.2f})",
                color=col,
                linewidth=1.8,
            )
        ax.set_title(assay, fontweight="bold", fontsize=11)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.set_ylabel("True Positive Rate", fontsize=9)
        ax.legend(loc="lower right", fontsize=8, frameon=True)

    plt.suptitle("Tox21 Receiver Operating Characteristic (ROC) Curves Across 12 Endpoints", fontsize=15, y=0.99)
    plt.tight_layout()
    fig_path2 = plots_dir / "roc_curves_all_endpoints.png"
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {fig_path2.name}")

    # --------------------------------------------------------------------------
    # Figure 3: Precision-Recall (PR) Curves (3x4 Grid with Chance Prevalence)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 4, figsize=(18, 13))
    axes = axes.flatten()

    for idx, assay in enumerate(TARGET_ASSAYS):
        ax = axes[idx]
        prev = metrics_df[metrics_df["endpoint"] == assay]["positive_prevalence"].iloc[0]
        ax.axhline(prev, color="red", linestyle=":", alpha=0.8, label=f"Prevalence ({prev:.1%})")

        for model_name, col in MODEL_COLORS.items():
            dat = curve_data[assay][model_name]
            ax.plot(
                dat["recall"],
                dat["precision"],
                label=f"{model_name[:8]} ({dat['pr_auc']:.2f})",
                color=col,
                linewidth=1.8,
            )
        ax.set_title(assay, fontweight="bold", fontsize=11)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.suptitle("Tox21 Precision-Recall Curves (PR-AUC vs. Random Baseline Prevalence)", fontsize=15, y=0.99)
    plt.tight_layout()
    fig_path3 = plots_dir / "pr_curves_all_endpoints.png"
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {fig_path3.name}")

    # --------------------------------------------------------------------------
    # Figure 4: Class Distribution and Imbalance per Endpoint
    # --------------------------------------------------------------------------
    sub_df = metrics_df[metrics_df["model"] == "Random Forest"][
        ["endpoint", "test_positives", "test_negatives", "positive_prevalence"]
    ].copy()

    fig, ax1 = plt.subplots(figsize=(13, 6))

    x = np.arange(len(sub_df))
    width = 0.55

    ax1.bar(x, sub_df["test_negatives"], width, label="Test Inactives (0)", color="#9ecae1")
    ax1.bar(
        x,
        sub_df["test_positives"],
        width,
        bottom=sub_df["test_negatives"],
        label="Test Actives / Toxic (1)",
        color="#e6550d",
    )
    ax1.set_ylabel("Number of Test Compounds", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(sub_df["endpoint"], rotation=30, ha="right", fontsize=10)

    # Secondary axis for positive prevalence %
    ax2 = ax1.twinx()
    ax2.plot(x, sub_df["positive_prevalence"] * 100, color="#756bb1", marker="o", linewidth=2.5, label="Active %")
    ax2.set_ylabel("Positive Class Prevalence (%)", color="#756bb1", fontsize=12)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    plt.title("Tox21 Test Set Class Distributions and Severe Imbalance Ratios Across Endpoints", fontweight="bold")
    plt.tight_layout()
    fig_path4 = plots_dir / "endpoint_class_distributions.png"
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {fig_path4.name}")

    # --------------------------------------------------------------------------
    # Figure 5: Performance vs. Class Imbalance (Why some endpoints fail)
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    rf_df = metrics_df[metrics_df["model"] == "Random Forest"]

    # PR-AUC vs Prevalence
    sns.regplot(
        data=rf_df,
        x="positive_prevalence",
        y="test_pr_auc",
        ax=axes[0],
        color="#1b9e77",
        scatter_kws={"s": 70},
    )
    for _, row in rf_df.iterrows():
        axes[0].annotate(
            row["endpoint"],
            (row["positive_prevalence"], row["test_pr_auc"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    axes[0].set_xlabel("Positive Prevalence in Test Set", fontsize=11)
    axes[0].set_ylabel("Random Forest Test PR-AUC", fontsize=11)
    axes[0].set_title("A. Strong Positive Correlation: PR-AUC vs. Active Prevalence", fontweight="bold")

    # ROC-AUC vs Prevalence
    sns.regplot(
        data=rf_df,
        x="positive_prevalence",
        y="test_roc_auc",
        ax=axes[1],
        color="#2b5c8f",
        scatter_kws={"s": 70},
    )
    for _, row in rf_df.iterrows():
        axes[1].annotate(
            row["endpoint"],
            (row["positive_prevalence"], row["test_roc_auc"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    axes[1].set_xlabel("Positive Prevalence in Test Set", fontsize=11)
    axes[1].set_ylabel("Random Forest Test ROC-AUC", fontsize=11)
    axes[1].set_title("B. ROC-AUC Invariance vs. Active Prevalence", fontweight="bold")

    plt.tight_layout()
    fig_path5 = plots_dir / "performance_vs_imbalance.png"
    plt.savefig(fig_path5, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {fig_path5.name}")

    # --------------------------------------------------------------------------
    # Figure 6: Confusion Matrices (Best vs. Worst Performing Endpoints)
    # --------------------------------------------------------------------------
    # Best: SR-MMP (or NR-AR-LBD), Worst: NR-ER (or NR-PPAR-gamma)
    endpoints_to_cm = ["SR-MMP", "NR-AR-LBD", "NR-ER", "SR-ATAD5"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    for idx, ep in enumerate(endpoints_to_cm):
        ep_preds = preds_df[preds_df["endpoint"] == ep]
        cm = confusion_matrix(ep_preds["y_true"], ep_preds["pred_rf"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            annot_kws={"size": 14, "weight": "bold"},
        )
        axes[idx].set_title(f"Random Forest Confusion Matrix: {ep}", fontweight="bold")
        axes[idx].set_xlabel("Predicted Label (0: Inactive, 1: Toxic)")
        axes[idx].set_ylabel("True Experimental Label")
        axes[idx].set_xticklabels(["Inactive", "Toxic"])
        axes[idx].set_yticklabels(["Inactive", "Toxic"])

    plt.tight_layout()
    fig_path6 = plots_dir / "confusion_matrices_worst_best.png"
    plt.savefig(fig_path6, dpi=300)
    plt.close()
    print(f"  ✓ Saved: {fig_path6.name}")

    # --------------------------------------------------------------------------
    # Figure 7: Scaffold Split vs. Random Split Benchmark (Task 6)
    # --------------------------------------------------------------------------
    if not split_comp_df.empty:
        # Merge scaffold metrics with random split metrics
        scaff_sub = metrics_df[["endpoint", "model", "test_roc_auc", "test_pr_auc"]].rename(
            columns={"test_roc_auc": "scaffold_roc_auc", "test_pr_auc": "scaffold_pr_auc"}
        )
        merged_split = pd.merge(scaff_sub, split_comp_df, on=["endpoint", "model"])
        rf_split = merged_split[merged_split["model"] == "Random Forest"].copy()

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # ROC-AUC Split comparison
        df_melt_roc = rf_split.melt(
            id_vars=["endpoint"],
            value_vars=["scaffold_roc_auc", "random_test_roc_auc"],
            var_name="Split Strategy",
            value_name="ROC-AUC",
        )
        df_melt_roc["Split Strategy"] = df_melt_roc["Split Strategy"].replace({
            "scaffold_roc_auc": "Bemis-Murcko Scaffold Split (Realistic)",
            "random_test_roc_auc": "Random Stratified Split (Optimistic / Leaky)",
        })

        sns.barplot(
            data=df_melt_roc,
            x="endpoint",
            y="ROC-AUC",
            hue="Split Strategy",
            palette=["#2b5c8f", "#fc8d62"],
            ax=axes[0],
        )
        axes[0].set_title("A. Random Forest ROC-AUC: Scaffold Split vs. Random Split", fontweight="bold")
        axes[0].set_ylim(0.5, 1.0)
        axes[0].set_ylabel("ROC-AUC")
        axes[0].legend(loc="lower right", frameon=True)

        # PR-AUC Split comparison
        df_melt_pr = rf_split.melt(
            id_vars=["endpoint"],
            value_vars=["scaffold_pr_auc", "random_test_pr_auc"],
            var_name="Split Strategy",
            value_name="PR-AUC",
        )
        df_melt_pr["Split Strategy"] = df_melt_pr["Split Strategy"].replace({
            "scaffold_pr_auc": "Bemis-Murcko Scaffold Split (Realistic)",
            "random_test_pr_auc": "Random Stratified Split (Optimistic / Leaky)",
        })

        sns.barplot(
            data=df_melt_pr,
            x="endpoint",
            y="PR-AUC",
            hue="Split Strategy",
            palette=["#2b5c8f", "#fc8d62"],
            ax=axes[1],
        )
        axes[1].set_title("B. Random Forest PR-AUC: Scaffold Split vs. Random Split", fontweight="bold")
        axes[1].set_ylim(0.0, 0.9)
        axes[1].set_ylabel("PR-AUC")
        axes[1].legend(loc="upper right", frameon=True)
        plt.xticks(rotation=30, ha="right")

        plt.tight_layout()
        fig_path7 = plots_dir / "scaffold_vs_random_split.png"
        plt.savefig(fig_path7, dpi=300)
        plt.close()
        print(f"  ✓ Saved: {fig_path7.name}")


# ==============================================================================
# Step 6: Main Pipeline Controller
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Tox21 Member 4 Evaluation, Statistics & Error Analysis Pipeline")
    parser.add_argument("--artifacts-dir", type=str, default=str(ARTIFACTS_DIR), help="Path to artifacts directory")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Path to save results")
    parser.add_argument("--bootstraps", type=int, default=1000, help="Number of bootstrap resamples for CIs")
    parser.add_argument("--plots", action="store_true", default=True, help="Generate publication-grade diagnostic plots")
    parser.add_argument("--random-split", action="store_true", default=True, help="Run Scaffold vs Random Split experiment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    artifacts_path = Path(args.artifacts_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load preprocessed features and labels
    X, labels_df = load_data(artifacts_path)

    # 2. Run core evaluation
    metrics_df, stats_df, preds_df, curve_data = evaluate_all_endpoints(
        X=X,
        labels_df=labels_df,
        n_bootstraps=args.bootstraps,
        random_state=args.seed,
    )

    # 3. Save primary metric tables
    metrics_csv_path = output_path / "metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\n[Artifact] Saved detailed metrics: {metrics_csv_path}")

    stats_csv_path = output_path / "statistical_tests.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[Artifact] Saved statistical tests: {stats_csv_path}")

    preds_csv_path = output_path / "test_predictions.csv"
    preds_df.to_csv(preds_csv_path, index=False)
    print(f"[Artifact] Saved compound predictions: {preds_csv_path}")

    # 4. Summary Table (Macro-Averages across all 12 endpoints)
    summary_df = metrics_df.groupby("model")[
        ["test_roc_auc", "test_pr_auc", "f1_score", "balanced_accuracy", "mcc"]
    ].agg(["mean", "std"]).round(4)
    summary_csv_path = output_path / "summary_metrics.csv"
    summary_df.to_csv(summary_csv_path)
    print(f"[Artifact] Saved summary metrics: {summary_csv_path}")

    print("\n" + "=" * 85)
    print("                MACRO-AVERAGE BENCHMARK SUMMARY (ALL 12 ASSAYS)")
    print("=" * 85)
    print(summary_df.to_string())
    print("=" * 85)

    # 5. Run Random-Split benchmark experiment (Task 6)
    split_comp_df = pd.DataFrame()
    if args.random_split:
        split_comp_df = run_random_split_experiment(X=X, labels_df=labels_df, random_state=args.seed)
        split_csv_path = output_path / "split_comparison.csv"
        split_comp_df.to_csv(split_csv_path, index=False)
        print(f"[Artifact] Saved split comparison metrics: {split_csv_path}")

    # 6. Generate Diagnostic Plots (Task 4)
    if args.plots:
        generate_plots(
            metrics_df=metrics_df,
            preds_df=preds_df,
            curve_data=curve_data,
            split_comp_df=split_comp_df,
            output_dir=output_path,
        )

    print("\n[Done] Member 4 evaluation, statistics, and figures completed successfully.")


if __name__ == "__main__":
    main()
