"""
final_comparison.py
===================
Puts every Phase 1 and Phase 2 model into one table and picks the winner in
each of the four categories the project needs:

  best standalone ML model          (Random Forest family)
  best standalone DL model          (one network, one input type)
  best feature-level hybrid         (two feature sources fused inside a model)
  best prediction ensemble          (several finished models combined)

The selection rule is the same everywhere and it is the whole point of this
script: **every winner is chosen by its VALIDATION score. The test score is
computed once afterwards and never used to choose anything.**

Inputs (produced by the other scripts in this folder):
  predictions/val_test_predictions.parquet        Phase 1 RF + saved DL runs
  predictions/ann_transformer_predictions.parquet Experiment 2 and ablations
  predictions/feature_hybrid_predictions.parquet  RF/ExtraTrees on DL embeddings

Run from the repository root:
    python -m phase2.experiments.final_comparison

Approximate runtime: about one minute. Nothing is trained here.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phase2.experiments.ensemble import DL_SIX_RUNS, blend, make_wide, score_macro
from phase2.experiments.final_evaluation import (
    choose_thresholds_on_validation,
    score_all_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = PROJECT_ROOT / "phase2" / "experiments" / "predictions"
RESULTS_DIR = PROJECT_ROOT / "phase2" / "experiments" / "results"
PLOTS_DIR = PROJECT_ROOT / "phase2" / "experiments" / "plots"

PREDICTION_FILES = [
    "val_test_predictions.parquet",
    "ann_transformer_predictions.parquet",
    "feature_hybrid_predictions.parquet",
]

# Which models belong to which category. Anything matching a prefix is a
# candidate for that category; the best one is then chosen on validation.
CATEGORY_PREFIXES = {
    "standalone_ml": ["rf_phase1"],
    "standalone_dl": ["dl_", "ann_only_", "smiles_only_"],
    "feature_hybrid": ["fh_", "hybrid_seed"],
}


def load_all_predictions():
    """Reads every prediction file that exists and stacks them."""
    tables = []
    for filename in PREDICTION_FILES:
        path = PREDICTIONS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run the script that produces it first."
            )
        tables.append(pd.read_parquet(path))

    predictions = pd.concat(tables, ignore_index=True)

    # Safety check: every model must cover exactly the same molecule-endpoint
    # rows, otherwise a comparison between them would not be like for like.
    counts = predictions.groupby("model").size()
    if counts.nunique() != 1:
        raise ValueError(f"Models cover different numbers of rows:\n{counts}")

    return predictions


def list_models(predictions, prefixes):
    """Returns the model names that start with any of the given prefixes."""
    names = sorted(predictions["model"].unique())
    return [n for n in names if any(n.startswith(p) for p in prefixes)]


def pick_best_single_model(val, candidates):
    """Returns the candidate with the highest validation macro ROC-AUC."""
    scored = []
    for name in candidates:
        roc, pr = score_macro(val, name)
        scored.append({"candidate": name, "val_roc_auc": round(roc, 4),
                       "val_pr_auc": round(pr, 4)})
    table = pd.DataFrame(scored).sort_values("val_roc_auc", ascending=False)
    return table.iloc[0]["candidate"], table


def pick_best_ensemble(val, predictions):
    """
    Tries a short list of prediction ensembles and returns the best on
    validation.

    We stay with rank averaging because the models produce probabilities on
    very different scales: a forest and a neural network cannot be averaged
    directly without letting whichever one is more confident take over.
    Ranking turns each model into an equal voter.
    """
    available = set(predictions["model"].unique())
    dl_runs = [name for name in DL_SIX_RUNS if name in available]
    at_runs = [f"hybrid_seed{seed}" for seed in [7, 21, 42] if f"hybrid_seed{seed}" in available]

    options = {}
    options["dl6_rank_avg"] = (dl_runs, None)

    for rf_weight in [1, 2, 3, 6]:
        options[f"rf_plus_dl6_rank_w{rf_weight}"] = (
            ["rf_phase1"] + dl_runs,
            [rf_weight] + [1] * len(dl_runs),
        )

    if at_runs:
        options["at3_rank_avg"] = (at_runs, None)
        for rf_weight in [1, 3]:
            options[f"rf_plus_at3_rank_w{rf_weight}"] = (
                ["rf_phase1"] + at_runs,
                [rf_weight] + [1] * len(at_runs),
            )
        # The forest, the six fingerprint networks and the three hybrids,
        # each family carrying the same total weight.
        options["rf_plus_dl6_plus_at3_rank"] = (
            ["rf_phase1"] + dl_runs + at_runs,
            [3] + [0.5] * len(dl_runs) + [1.0] * len(at_runs),
        )

    scored = []
    for name in options:
        columns, weights = options[name]
        roc, pr = score_macro(blend(val, columns, weights, True), "blend_score")
        scored.append({"candidate": name, "val_roc_auc": round(roc, 4),
                       "val_pr_auc": round(pr, 4)})

    table = pd.DataFrame(scored).sort_values("val_roc_auc", ascending=False)
    best = table.iloc[0]["candidate"]
    return best, options[best], table, options


def score_candidate(val, test, columns, weights, is_blend):
    """
    Scores one candidate on validation and on test, at the fixed 0.5 cutoff
    and at per-endpoint cutoffs tuned on validation.
    """
    if is_blend:
        val_table = blend(val, columns, weights, True)
        test_table = blend(test, columns, weights, True)
        column = "blend_score"
    else:
        val_table, test_table, column = val, test, columns

    val_fixed = score_all_metrics(val_table, column)
    thresholds = choose_thresholds_on_validation(val_table, column)
    test_tuned = score_all_metrics(test_table, column, thresholds)

    row = {
        "val_roc_auc": round(val_fixed["roc_auc"].mean(), 4),
        "val_pr_auc": round(val_fixed["pr_auc"].mean(), 4),
        "test_roc_auc": round(test_tuned["roc_auc"].mean(), 4),
        "test_pr_auc": round(test_tuned["pr_auc"].mean(), 4),
        "test_precision": round(test_tuned["precision"].mean(), 4),
        "test_recall": round(test_tuned["recall"].mean(), 4),
        "test_f1": round(test_tuned["f1"].mean(), 4),
        "test_mcc": round(test_tuned["mcc"].mean(), 4),
    }
    return row, test_tuned


def seed_spread(val, test, predictions):
    """
    Mean and standard deviation over seeds for every configuration that was
    trained more than once. This is how we tell a real difference from luck.
    """
    rows = []
    families = {
        "dl_weighted": [f"dl_weighted_seed{s}" for s in [7, 21, 42]],
        "dl_unweighted": [f"dl_unweighted_seed{s}" for s in [7, 21, 42]],
        "dl_no_dropout": [f"dl_no_dropout_seed{s}" for s in [7, 21, 42]],
        "ann_only": [f"ann_only_seed{s}" for s in [7, 21, 42]],
        "smiles_only": [f"smiles_only_seed{s}" for s in [7, 21, 42]],
        "ann_transformer_hybrid": [f"hybrid_seed{s}" for s in [7, 21, 42]],
    }
    available = set(predictions["model"].unique())

    for family in families:
        members = [m for m in families[family] if m in available]
        if len(members) < 2:
            continue
        val_scores = [score_macro(val, m)[0] for m in members]
        test_scores = [score_macro(test, m)[0] for m in members]
        rows.append(
            {
                "configuration": family,
                "seeds": len(members),
                "val_roc_mean": round(float(np.mean(val_scores)), 4),
                "val_roc_std": round(float(np.std(val_scores, ddof=1)), 4),
                "test_roc_mean": round(float(np.mean(test_scores)), 4),
                "test_roc_std": round(float(np.std(test_scores, ddof=1)), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("val_roc_mean", ascending=False)


def make_plots(comparison, per_endpoint, spread):
    """Three simple figures: overall scores, per-endpoint scores, seed spread."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Validation and test ROC-AUC for the selected models, side by side.
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(comparison))
    axis.bar(positions - 0.2, comparison["val_roc_auc"], 0.4, label="validation")
    axis.bar(positions + 0.2, comparison["test_roc_auc"], 0.4, label="test")
    axis.set_xticks(positions)
    axis.set_xticklabels(comparison["model"], rotation=20, ha="right", fontsize=8)
    axis.set_ylabel("macro ROC-AUC")
    axis.set_ylim(0.5, 0.9)
    axis.set_title("Selected model per category (chosen on validation)")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "category_winners_roc_auc.png", dpi=150)
    plt.close(figure)

    # 2. Per-endpoint test ROC-AUC, so nobody claims a win from one endpoint.
    pivot = per_endpoint.pivot(index="endpoint", columns="model", values="roc_auc")
    figure, axis = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="bar", ax=axis, width=0.8)
    axis.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    axis.set_ylabel("test ROC-AUC")
    axis.set_ylim(0.4, 1.0)
    axis.set_title("Test ROC-AUC on each of the 12 endpoints")
    axis.legend(fontsize=7)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "per_endpoint_roc_auc.png", dpi=150)
    plt.close(figure)

    # 3. Seed spread: the error bar is what makes a small gap meaningless.
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.errorbar(
        spread["configuration"], spread["test_roc_mean"],
        yerr=spread["test_roc_std"], fmt="o", capsize=5,
    )
    axis.set_ylabel("test ROC-AUC")
    axis.set_title("Mean and standard deviation over three seeds")
    axis.tick_params(axis="x", rotation=20, labelsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(PLOTS_DIR / "seed_spread.png", dpi=150)
    plt.close(figure)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_all_predictions()
    val = make_wide(predictions, "val")
    test = make_wide(predictions, "test")

    print("=" * 78)
    print("STEP 1: PICK A WINNER IN EACH CATEGORY, USING VALIDATION ONLY")
    print("=" * 78)

    selections = {}
    selection_tables = []

    for category in ["standalone_ml", "standalone_dl", "feature_hybrid"]:
        candidates = list_models(predictions, CATEGORY_PREFIXES[category])
        best, table = pick_best_single_model(val, candidates)
        table["category"] = category
        selection_tables.append(table)
        selections[category] = (best, best, None, False)
        print(f"  {category:16s} {len(candidates):2d} candidates -> {best}")

    best_ensemble, (columns, weights), ensemble_table, _ = pick_best_ensemble(val, predictions)
    ensemble_table["category"] = "prediction_ensemble"
    selection_tables.append(ensemble_table)
    selections["prediction_ensemble"] = (best_ensemble, columns, weights, True)
    print(f"  {'ensemble':16s} {len(ensemble_table):2d} candidates -> {best_ensemble}")

    pd.concat(selection_tables, ignore_index=True)[
        ["category", "candidate", "val_roc_auc", "val_pr_auc"]
    ].to_csv(RESULTS_DIR / "validation_selection_all_candidates.csv", index=False)

    print()
    print("=" * 78)
    print("STEP 2: SCORE THE WINNERS ON TEST, ONCE")
    print("=" * 78)

    comparison_rows = []
    per_endpoint_tables = []

    for category in selections:
        label, columns, weights, is_blend = selections[category]
        row, test_tuned = score_candidate(val, test, columns, weights, is_blend)
        row["category"] = category
        row["model"] = label
        comparison_rows.append(row)

        test_tuned["model"] = label
        test_tuned["category"] = category
        per_endpoint_tables.append(test_tuned)

    column_order = [
        "category", "model", "val_roc_auc", "val_pr_auc",
        "test_roc_auc", "test_pr_auc", "test_precision",
        "test_recall", "test_f1", "test_mcc",
    ]
    comparison = pd.DataFrame(comparison_rows)[column_order]
    comparison.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)

    per_endpoint = pd.concat(per_endpoint_tables, ignore_index=True)
    per_endpoint.to_csv(RESULTS_DIR / "final_per_endpoint_comparison.csv", index=False)

    print(comparison.to_string(index=False))
    print()
    print("Precision, recall, F1 and MCC use per-endpoint cutoffs tuned on")
    print("validation. ROC-AUC and PR-AUC do not depend on a cutoff.")

    print()
    print("=" * 78)
    print("STEP 3: SEED SPREAD — HOW BIG A GAP IS ACTUALLY MEANINGFUL")
    print("=" * 78)
    spread = seed_spread(val, test, predictions)
    spread.to_csv(RESULTS_DIR / "seed_spread.csv", index=False)
    print(spread.to_string(index=False))

    print()
    print("=" * 78)
    print("STEP 4: PER-ENDPOINT CHECK")
    print("=" * 78)
    pivot = per_endpoint.pivot(index="endpoint", columns="model", values="roc_auc").round(3)
    print(pivot.to_string())
    best_counts = pivot.idxmax(axis=1).value_counts()
    print()
    print("Endpoints where each model is the best:")
    print(best_counts.to_string())

    make_plots(comparison, per_endpoint, spread)
    print()
    print(f"Figures written to {PLOTS_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
