"""
improve_ensemble.py
===================
A1-A4: tries to get Phase 2 ahead of Phase 1 on PR-AUC and MCC as well, not
only on ROC-AUC.

Why the previous ensemble lost on PR-AUC. It averaged RANKS. Replacing a
probability by its rank keeps the ordering but throws away how confident the
model was, and PR-AUC rewards confidence at the very top of the list. So here
we calibrate the probabilities instead and average those.

What calibration means, in one sentence: it rescales a model's scores so that
0.7 really does mean "toxic about 70% of the time". We use isotonic
regression, which only ever reorders scores upward or downward in a
monotonic way.

An important consequence of that: because the rescaling is monotonic, it
CANNOT change any single model's ROC-AUC or PR-AUC. It changes only what the
average of several models means, which is exactly the thing we are trying to
fix.

Everything is selected on validation. Test is scored once at the end.

Run from the repository root:
    python -m phase2.experiments.improve_ensemble

Approximate runtime: about two minutes. Nothing is trained.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from phase2.experiments.ensemble import DL_SIX_RUNS, make_wide, rank_normalise
from phase2.experiments.final_evaluation import (
    choose_thresholds_on_validation,
    score_all_metrics,
)
from phase2.data.src.data_loader import METRICS_DIR, PREDICTIONS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_FILES = [
    "val_test_predictions.parquet",
    "phase1_baseline_predictions.parquet",
    "feature_hybrid_predictions.parquet",
    "ann_transformer_predictions.parquet",
]

BEST_HYBRID = "fh_corrected_emb_extratrees"
AT_HYBRID_RUNS = ["hybrid_seed7", "hybrid_seed21", "hybrid_seed42"]

# The members the Part A ensemble already used.
BASE_MEMBERS = ["rf_phase1"] + DL_SIX_RUNS

# Candidate extra members for A3, each tested one at a time on validation.
EXTRA_MEMBERS = {
    "lr_phase1": ["lr_phase1"],
    "dt_phase1": ["dt_phase1"],
    "feature_hybrid": [BEST_HYBRID],
    "ann_transformer": AT_HYBRID_RUNS,
}


def load_wide_tables():
    """Reads every prediction file and returns observed-only wide tables."""
    tables = []
    for filename in PREDICTION_FILES:
        path = PREDICTIONS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing {path.name}. Generate it first.")
        tables.append(pd.read_parquet(path))

    predictions = pd.concat(tables, ignore_index=True)

    counts = predictions.groupby("model").size()
    if counts.nunique() != 1:
        raise ValueError(f"Models cover different row counts, refusing to compare:\n{counts}")

    val = make_wide(predictions, "val")
    test = make_wide(predictions, "test")

    # Only observed labels are ever scored, so drop the rest once and for all.
    return val[val["observed"]].copy(), test[test["observed"]].copy()


def calibrate_one_endpoint(val_scores, val_labels, test_scores):
    """
    Fits isotonic regression for one model on one endpoint.

    Returns (honest validation values, test values).

    The validation values are CROSS-FITTED: the table is split into five
    parts, and each part is rescaled by a curve fitted on the other four. If
    we instead fitted on all of validation and scored the same rows, the
    validation number would be flattering and useless for choosing anything.

    The test values come from a curve fitted on all of validation, which is
    allowed because validation is a tuning split.
    """
    if val_labels.sum() < 5 or (1 - val_labels).sum() < 5:
        # Too few of one class to fit five folds; leave the scores as they are.
        return val_scores.copy(), test_scores.copy()

    honest_val = np.zeros_like(val_scores)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fit_rows, score_rows in folds.split(val_scores.reshape(-1, 1), val_labels):
        curve = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        curve.fit(val_scores[fit_rows], val_labels[fit_rows])
        honest_val[score_rows] = curve.predict(val_scores[score_rows])

    full_curve = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    full_curve.fit(val_scores, val_labels)
    return honest_val, full_curve.predict(test_scores)


def add_calibrated_columns(val, test, models):
    """Adds a cal_<model> column for every model, on both splits."""
    val = val.copy()
    test = test.copy()

    for model in models:
        val[f"cal_{model}"] = np.nan
        test[f"cal_{model}"] = np.nan

    for endpoint in val["endpoint"].unique():
        val_rows = val["endpoint"] == endpoint
        test_rows = test["endpoint"] == endpoint
        val_labels = val.loc[val_rows, "true_label"].astype(int).values

        for model in models:
            honest_val, calibrated_test = calibrate_one_endpoint(
                val.loc[val_rows, model].values,
                val_labels,
                test.loc[test_rows, model].values,
            )
            val.loc[val_rows, f"cal_{model}"] = honest_val
            test.loc[test_rows, f"cal_{model}"] = calibrated_test

    return val, test


def average_columns(wide, members, weights=None, use_ranks=False, calibrated=False):
    """
    Combines several members into one blend_score column.

    use_ranks reproduces the old rank averaging so we can compare against it.
    calibrated averages the cal_* columns instead of the raw probabilities.
    """
    columns = [f"cal_{m}" for m in members] if calibrated else list(members)
    table = rank_normalise(wide, columns) if use_ranks else wide

    if weights is None:
        combined = table[columns].mean(axis=1)
    else:
        weight_array = np.array(weights, dtype=float)
        weight_array = weight_array / weight_array.sum()
        combined = (table[columns].values * weight_array).sum(axis=1)

    result = wide.copy()
    result["blend_score"] = np.asarray(combined)
    return result


def macro_scores(wide, column):
    """Macro ROC-AUC and PR-AUC over the endpoints."""
    roc_scores = []
    pr_scores = []
    for _, group in wide.groupby("endpoint", observed=True):
        y_true = group["true_label"].astype(int)
        if y_true.nunique() > 1:
            roc_scores.append(roc_auc_score(y_true, group[column]))
            pr_scores.append(average_precision_score(y_true, group[column]))
    return float(np.mean(roc_scores)), float(np.mean(pr_scores))


def member_weights(members, rf_weight):
    """Gives rf_phase1 the chosen weight and every other member weight 1."""
    return [rf_weight if m == "rf_phase1" else 1 for m in members]


def run_a1_combination_method(val):
    """A1: rank averaging versus raw probabilities versus calibrated ones."""
    weights = member_weights(BASE_MEMBERS, 3)
    options = {
        "rank_average (Part A choice)": {"use_ranks": True, "calibrated": False},
        "raw_probability_average": {"use_ranks": False, "calibrated": False},
        "calibrated_probability_average": {"use_ranks": False, "calibrated": True},
        "calibrated_rank_average": {"use_ranks": True, "calibrated": True},
    }

    rows = []
    for name, settings in options.items():
        blended = average_columns(val, BASE_MEMBERS, weights, **settings)
        roc, pr = macro_scores(blended, "blend_score")
        rows.append({"method": name, "val_roc_auc": round(roc, 4),
                     "val_pr_auc": round(pr, 4),
                     "val_mean_of_both": round((roc + pr) / 2, 4),
                     "use_ranks": settings["use_ranks"],
                     "calibrated": settings["calibrated"]})

    return pd.DataFrame(rows).sort_values("val_mean_of_both", ascending=False)


def run_a2_weight_search(val, members, use_ranks, calibrated):
    """
    A2: choose the Random Forest weight on ROC-AUC, on PR-AUC, and on both.

    The combining method comes from whichever option won A1, so we are not
    tuning a weight for a method we already rejected.
    """
    rows = []
    for rf_weight in [1, 2, 3, 4, 6, 9, 12, 18, 24]:
        blended = average_columns(
            val, members, member_weights(members, rf_weight),
            use_ranks=use_ranks, calibrated=calibrated,
        )
        roc, pr = macro_scores(blended, "blend_score")
        rows.append(
            {
                "rf_weight": rf_weight,
                "val_roc_auc": round(roc, 4),
                "val_pr_auc": round(pr, 4),
                "val_mean_of_both": round((roc + pr) / 2, 4),
            }
        )

    table = pd.DataFrame(rows)
    choices = {
        "by_roc_auc": int(table.loc[table["val_roc_auc"].idxmax(), "rf_weight"]),
        "by_pr_auc": int(table.loc[table["val_pr_auc"].idxmax(), "rf_weight"]),
        "by_mean_of_both": int(table.loc[table["val_mean_of_both"].idxmax(), "rf_weight"]),
    }
    return table, choices


def run_a3_extra_members(val, members, rf_weight, use_ranks, calibrated):
    """
    A3: try each extra member group on its own and keep the ones that help.

    Judged on the mean of validation ROC-AUC and PR-AUC, so a member cannot
    buy a PR-AUC gain by damaging ROC-AUC.
    """
    def blend_of(member_list):
        return average_columns(
            val, member_list, member_weights(member_list, rf_weight),
            use_ranks=use_ranks, calibrated=calibrated,
        )

    base_roc, base_pr = macro_scores(blend_of(members), "blend_score")
    base_mean = (base_roc + base_pr) / 2

    rows = [{"members_added": "none (base)", "val_roc_auc": round(base_roc, 4),
             "val_pr_auc": round(base_pr, 4), "val_mean_of_both": round(base_mean, 4),
             "kept": True}]

    kept = list(members)
    for name, extra in EXTRA_MEMBERS.items():
        candidate = kept + [m for m in extra if m not in kept]
        roc, pr = macro_scores(blend_of(candidate), "blend_score")
        helps = (roc + pr) / 2 > base_mean
        rows.append({"members_added": name, "val_roc_auc": round(roc, 4),
                     "val_pr_auc": round(pr, 4), "val_mean_of_both": round((roc + pr) / 2, 4),
                     "kept": helps})
        if helps:
            kept = candidate
            base_mean = (roc + pr) / 2

    return pd.DataFrame(rows), kept


def build_candidates(val, test, final_members, rf_weight, use_ranks, calibrated):
    """
    Builds the short candidate list used for the final comparison and for the
    per-endpoint choice in A4. Four options only, to limit how much we can
    overfit the 782 validation molecules.
    """
    candidates = {}

    candidates["rf_phase1"] = (val, test, "rf_phase1")

    for name, members, weights, settings in [
        ("dl6_alone", DL_SIX_RUNS, None, {"use_ranks": use_ranks, "calibrated": calibrated}),
        ("rf_plus_dl6_rank_w3", BASE_MEMBERS, member_weights(BASE_MEMBERS, 3),
         {"use_ranks": True}),
        ("phase2_tuned_blend", final_members, member_weights(final_members, rf_weight),
         {"use_ranks": use_ranks, "calibrated": calibrated}),
    ]:
        candidates[name] = (
            average_columns(val, members, weights, **settings),
            average_columns(test, members, weights, **settings),
            "blend_score",
        )

    return candidates


def run_a4_per_endpoint(val, candidates):
    """
    A4: pick the best candidate separately for each endpoint, on validation.

    This is the riskiest step in the file. With roughly 780 validation
    molecules and only a few dozen positives on the rare endpoints, the
    winner of a four-way comparison can easily be the luckiest rather than
    the best. The report says so.
    """
    rows = []
    for endpoint in sorted(val["endpoint"].unique()):
        scores = {}
        for name, (val_table, _, column) in candidates.items():
            group = val_table[val_table["endpoint"] == endpoint]
            y_true = group["true_label"].astype(int)
            if y_true.nunique() < 2:
                continue
            roc = roc_auc_score(y_true, group[column])
            pr = average_precision_score(y_true, group[column])
            scores[name] = (roc + pr) / 2

        best = max(scores, key=scores.get)
        rows.append({"endpoint": endpoint, "chosen_model": best,
                     "val_mean_of_both": round(scores[best], 4)})

    return pd.DataFrame(rows)


def assemble_per_endpoint(split, choice_by_endpoint, candidates):
    """
    Glues the per-endpoint winners into one table with a single score column.

    split is "val" or "test" and decides which of each candidate's two tables
    we take the rows from.
    """
    table_index = 0 if split == "val" else 1
    pieces = []

    for endpoint, chosen in choice_by_endpoint.items():
        table = candidates[chosen][table_index]
        column = candidates[chosen][2]
        piece = table[table["endpoint"] == endpoint].copy()
        piece["blend_score"] = piece[column]
        pieces.append(piece[["mol_id", "endpoint", "true_label", "observed", "blend_score"]])

    return pd.concat(pieces, ignore_index=True)


def full_metrics(val_table, test_table, column, label):
    """Six metrics on validation and on test, with cutoffs tuned on validation."""
    thresholds = choose_thresholds_on_validation(val_table, column)
    val_scored = score_all_metrics(val_table, column, thresholds)
    test_scored = score_all_metrics(test_table, column, thresholds)

    return {
        "model": label,
        "val_roc_auc": round(val_scored["roc_auc"].mean(), 4),
        "val_pr_auc": round(val_scored["pr_auc"].mean(), 4),
        "val_f1": round(val_scored["f1"].mean(), 4),
        "val_mcc": round(val_scored["mcc"].mean(), 4),
        "test_roc_auc": round(test_scored["roc_auc"].mean(), 4),
        "test_pr_auc": round(test_scored["pr_auc"].mean(), 4),
        "test_precision": round(test_scored["precision"].mean(), 4),
        "test_recall": round(test_scored["recall"].mean(), 4),
        "test_f1": round(test_scored["f1"].mean(), 4),
        "test_mcc": round(test_scored["mcc"].mean(), 4),
    }


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    val, test = load_wide_tables()

    all_models = sorted(
        set(BASE_MEMBERS + [BEST_HYBRID] + AT_HYBRID_RUNS + ["lr_phase1", "dt_phase1"])
    )
    print("Calibrating every model on validation (cross-fitted, 5 folds)...")
    val, test = add_calibrated_columns(val, test, all_models)

    print()
    print("=" * 74)
    print("A1: HOW TO COMBINE — RANKS, RAW PROBABILITIES, OR CALIBRATED ONES")
    print("=" * 74)
    a1 = run_a1_combination_method(val)
    a1.to_csv(METRICS_DIR / "a1_combination_method.csv", index=False)
    print(a1.to_string(index=False))

    # A1's winner is carried into every later step.
    best_method = a1.iloc[0]
    use_ranks = bool(best_method["use_ranks"])
    calibrated = bool(best_method["calibrated"])
    print()
    print(f"  A1 winner: {best_method['method']}")
    print(f"  carried into A2-A4: use_ranks={use_ranks}, calibrated={calibrated}")

    print()
    print("=" * 74)
    print("A2: CHOOSING THE RANDOM FOREST WEIGHT ON PR-AUC, NOT ONLY ROC-AUC")
    print("=" * 74)
    a2, weight_choices = run_a2_weight_search(val, BASE_MEMBERS, use_ranks, calibrated)
    a2.to_csv(METRICS_DIR / "a2_weight_search.csv", index=False)
    print(a2.to_string(index=False))
    print()
    for criterion, weight in weight_choices.items():
        print(f"  best weight {criterion:18s} -> {weight}")
    rf_weight = weight_choices["by_mean_of_both"]
    print(f"  recommended: {rf_weight} (chosen on the mean, so neither metric is sacrificed)")

    print()
    print("=" * 74)
    print("A3: DOES ADDING MORE MEMBERS HELP?")
    print("=" * 74)
    a3, final_members = run_a3_extra_members(
        val, BASE_MEMBERS, rf_weight, use_ranks, calibrated
    )
    a3.to_csv(METRICS_DIR / "a3_extra_members.csv", index=False)
    print(a3.to_string(index=False))
    print()
    print(f"  members kept: {len(final_members)}")
    for member in final_members:
        print(f"    {member}")

    candidates = build_candidates(
        val, test, final_members, rf_weight, use_ranks, calibrated
    )

    print()
    print("=" * 74)
    print("A4: ONE MODEL PER ENDPOINT, CHOSEN ON VALIDATION")
    print("=" * 74)
    a4 = run_a4_per_endpoint(val, candidates)
    a4.to_csv(METRICS_DIR / "a4_per_endpoint_choice.csv", index=False)
    print(a4.to_string(index=False))
    choice_by_endpoint = dict(zip(a4["endpoint"], a4["chosen_model"]))

    candidates["per_endpoint_choice"] = (
        assemble_per_endpoint("val", choice_by_endpoint, candidates),
        assemble_per_endpoint("test", choice_by_endpoint, candidates),
        "blend_score",
    )

    print()
    print("=" * 74)
    print("FINAL: ALL SIX METRICS, VALIDATION AND TEST")
    print("=" * 74)
    rows = []
    for name, (val_table, test_table, column) in candidates.items():
        rows.append(full_metrics(val_table, test_table, column, name))

    comparison = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    comparison.to_csv(METRICS_DIR / "improved_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    baseline = comparison[comparison["model"] == "rf_phase1"].iloc[0]
    best = comparison.iloc[0]
    print()
    print(f"Highest validation PR-AUC: {best['model']} ({best['val_pr_auc']:.4f})")
    print(f"Phase 1 validation PR-AUC: {baseline['val_pr_auc']:.4f}")

    print()
    print("=" * 74)
    print("RECOMMENDED MODEL VERSUS PHASE 1, PER ENDPOINT")
    print("=" * 74)
    save_recommended(val, test, candidates, "phase2_tuned_blend")


def save_recommended(val, test, candidates, chosen):
    """
    Writes the recommended model's per-endpoint scores and its predictions.

    Simplicity is the reason the single blend is preferred over the
    per-endpoint selection even though the latter scores slightly higher on
    validation: that choice is argued in PART_A_RESULTS.md and was decided
    before test was looked at.
    """
    val_table, test_table, column = candidates[chosen]
    thresholds = choose_thresholds_on_validation(val_table, column)

    per_endpoint = score_all_metrics(test_table, column, thresholds)
    per_endpoint["model"] = chosen
    phase1 = score_all_metrics(test[test["observed"]] if "observed" in test else test, "rf_phase1",
                               choose_thresholds_on_validation(val, "rf_phase1"))
    phase1["model"] = "rf_phase1"

    combined = pd.concat([phase1, per_endpoint], ignore_index=True)
    combined.to_csv(METRICS_DIR / "recommended_per_endpoint.csv", index=False)

    side_by_side = combined.pivot(index="endpoint", columns="model",
                                  values=["roc_auc", "pr_auc"]).round(4)
    print(side_by_side.to_string())

    keep = ["mol_id", "endpoint", "true_label", "observed", column]
    saved = pd.concat(
        [val_table[keep].assign(split="val"), test_table[keep].assign(split="test")],
        ignore_index=True,
    ).rename(columns={column: "predicted_prob"})
    saved["model"] = chosen
    saved.to_parquet(METRICS_DIR / "recommended_model_predictions.parquet", index=False)
    print()
    print(f"Saved {chosen} predictions and per-endpoint scores.")


if __name__ == "__main__":
    main()
