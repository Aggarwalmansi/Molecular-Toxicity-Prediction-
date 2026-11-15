"""
ensemble.py
===========
A4: Tests whether combining model predictions improves performance.

The rule we follow strictly: every choice (which models to combine, which
blending method, which weight) is made using VALIDATION predictions only.
The winning combination is applied to the test set exactly once, at the end.

Two ways of combining are tried:

  Simple averaging  - average the predicted probabilities.
  Rank averaging    - replace each model's probabilities by their rank order
                      within an endpoint, scale to 0-1, then average.

Rank averaging helps when models are individually good at ordering molecules
but produce probabilities on different scales. A Random Forest and a neural
network are exactly that case: averaging their raw probabilities lets whichever
model outputs more extreme numbers dominate, while averaging ranks treats them
as equal voters.

Run from the repository root:
    python -m phase2.experiments.ensemble
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from phase2.data.src.data_loader import METRICS_DIR, PREDICTIONS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_FILE = PREDICTIONS_DIR / "val_test_predictions.parquet"

# The six runs of the same architecture that Member 4 produced.
DL_SIX_RUNS = [
    "dl_weighted_seed7",
    "dl_weighted_seed21",
    "dl_weighted_seed42",
    "dl_unweighted_seed7",
    "dl_unweighted_seed21",
    "dl_unweighted_seed42",
]


def load_predictions():
    """Reads the long prediction table produced by generate_predictions.py."""
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {PREDICTIONS_FILE}. Run generate_predictions.py first."
        )
    return pd.read_parquet(PREDICTIONS_FILE)


def make_wide(predictions, split):
    """
    Reshapes one split into a wide table: one row per molecule-endpoint pair,
    one column per model.

    Keeping the true label and the observed flag alongside makes it impossible
    to accidentally line up a prediction with the wrong molecule.
    """
    subset = predictions[predictions["split"] == split]
    wide = subset.pivot_table(
        index=["mol_id", "endpoint"],
        columns="model",
        values="predicted_prob",
        observed=True,
    )
    labels = subset.drop_duplicates(subset=["mol_id", "endpoint"]).set_index(["mol_id", "endpoint"])
    wide["true_label"] = labels["true_label"]
    wide["observed"] = labels["observed"]
    return wide.reset_index()


def rank_normalise(wide, model_columns):
    """
    Replaces each model's probabilities by ranks scaled to 0-1, within each
    endpoint separately.

    Ranking per endpoint matters because endpoints have very different positive
    rates. Ranking uses no labels, so it leaks nothing.
    """
    ranked = wide.copy()
    for column in model_columns:
        ranked[column] = wide.groupby("endpoint", observed=True)[column].rank(pct=True)
    return ranked


def score_macro(wide, score_column):
    """Computes macro ROC-AUC and PR-AUC over observed labels only."""
    observed = wide[wide["observed"]]
    roc_scores = []
    pr_scores = []
    for _, group in observed.groupby("endpoint", observed=True):
        y_true = group["true_label"].astype(int)
        if y_true.nunique() > 1:
            roc_scores.append(roc_auc_score(y_true, group[score_column]))
            pr_scores.append(average_precision_score(y_true, group[score_column]))
    return float(np.mean(roc_scores)), float(np.mean(pr_scores))


def blend(wide, columns, weights=None, use_ranks=False):
    """
    Combines the given model columns into a single score column.

    weights lets us give one model more influence, for example 0.5 Random
    Forest and 0.5 deep learning ensemble.
    """
    table = rank_normalise(wide, columns) if use_ranks else wide

    if weights is None:
        combined = table[columns].mean(axis=1)
    else:
        weight_array = np.array(weights, dtype=float)
        weight_array = weight_array / weight_array.sum()
        combined = (table[columns].values * weight_array).sum(axis=1)

    result = wide.copy()
    result["blend_score"] = combined
    return result


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions()

    val = make_wide(predictions, "val")
    test = make_wide(predictions, "test")

    print("=" * 72)
    print("STEP 1: INDIVIDUAL MODELS ON VALIDATION")
    print("=" * 72)
    individual_rows = []
    for model in DL_SIX_RUNS + ["rf_phase1"]:
        val_roc, val_pr = score_macro(val, model)
        individual_rows.append({"candidate": model, "val_roc_auc": val_roc, "val_pr_auc": val_pr})
        print(f"  {model:28s} val ROC {val_roc:.4f}  PR {val_pr:.4f}")

    dl_values = [row["val_roc_auc"] for row in individual_rows if row["candidate"] != "rf_phase1"]
    print()
    print(f"  Six DL runs: mean val ROC-AUC {np.mean(dl_values):.4f} "
          f"(std {np.std(dl_values, ddof=1):.4f})")

    print()
    print("=" * 72)
    print("STEP 2: CANDIDATE ENSEMBLES, SCORED ON VALIDATION ONLY")
    print("=" * 72)

    candidates = {}

    # Combining the six deep learning runs with each other
    candidates["dl6_mean_prob"] = (DL_SIX_RUNS, None, False)
    candidates["dl6_rank_avg"] = (DL_SIX_RUNS, None, True)

    # Combining the Random Forest with the deep learning runs.
    # Giving the forest weight 6 makes it count as much as the six runs together.
    rf_and_dl = ["rf_phase1"] + DL_SIX_RUNS
    for rf_weight in [1, 2, 3, 6, 12]:
        weights = [rf_weight] + [1] * len(DL_SIX_RUNS)
        candidates[f"rf_plus_dl6_rank_w{rf_weight}"] = (rf_and_dl, weights, True)
        candidates[f"rf_plus_dl6_prob_w{rf_weight}"] = (rf_and_dl, weights, False)

    rows = []
    for name in candidates:
        columns, weights, use_ranks = candidates[name]
        blended = blend(val, columns, weights, use_ranks)
        val_roc, val_pr = score_macro(blended, "blend_score")
        rows.append({"candidate": name, "val_roc_auc": val_roc, "val_pr_auc": val_pr})
        print(f"  {name:28s} val ROC {val_roc:.4f}  PR {val_pr:.4f}")

    all_candidates = pd.DataFrame(individual_rows + rows).sort_values(
        "val_roc_auc", ascending=False
    )
    all_candidates.to_csv(METRICS_DIR / "ensemble_validation_selection.csv", index=False)

    best_name = all_candidates.iloc[0]["candidate"]
    print()
    print(f"  Best on validation: {best_name} ({all_candidates.iloc[0]['val_roc_auc']:.4f})")

    print()
    print("=" * 72)
    print("STEP 3: APPLY THE VALIDATION WINNER TO TEST, ONCE")
    print("=" * 72)

    final_rows = []
    for name in ["rf_phase1", "dl6_rank_avg", best_name]:
        if name in candidates:
            columns, weights, use_ranks = candidates[name]
            val_blend = blend(val, columns, weights, use_ranks)
            test_blend = blend(test, columns, weights, use_ranks)
            val_roc, val_pr = score_macro(val_blend, "blend_score")
            test_roc, test_pr = score_macro(test_blend, "blend_score")
        else:
            val_roc, val_pr = score_macro(val, name)
            test_roc, test_pr = score_macro(test, name)

        final_rows.append(
            {
                "model": name,
                "val_roc_auc": round(val_roc, 4),
                "val_pr_auc": round(val_pr, 4),
                "test_roc_auc": round(test_roc, 4),
                "test_pr_auc": round(test_pr, 4),
            }
        )

    final = pd.DataFrame(final_rows).drop_duplicates(subset=["model"])
    final.to_csv(METRICS_DIR / "ensemble_final.csv", index=False)
    print(final.to_string(index=False))

    # Save the winning ensemble's test predictions so later steps can reuse them
    columns, weights, use_ranks = candidates[best_name]
    test_blend = blend(test, columns, weights, use_ranks)
    val_blend = blend(val, columns, weights, use_ranks)
    saved = pd.concat(
        [
            val_blend.assign(split="val"),
            test_blend.assign(split="test"),
        ]
    )[["mol_id", "endpoint", "split", "blend_score", "true_label", "observed"]]
    saved = saved.rename(columns={"blend_score": "predicted_prob"})
    saved["model"] = best_name
    saved.to_parquet(METRICS_DIR / "best_ensemble_predictions.parquet", index=False)
    print()
    print(f"Saved winning ensemble predictions ({best_name}) for reuse.")


if __name__ == "__main__":
    main()
