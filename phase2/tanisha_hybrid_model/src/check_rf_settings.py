"""
check_rf_settings.py
====================
Checks whether the hybrid model's weak result is caused by the Random Forest
settings rather than by the features themselves.

The main pipeline deliberately uses the same forest settings as the Phase 1
baseline (n_estimators=100, max_depth=12), so that the comparison isolates the
effect of the features. But a fair question in the viva is: "maybe the hybrid
only looks worse because 133 dense features need a different forest?"

This script answers that by trying several settings and scoring them on the
VALIDATION split only. The test split is never touched here.

Run from the repository root:
    python -m phase2.hybrid.src.check_rf_settings
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from phase2.data.src.data_loader import Tox21DataLoader
from phase2.hybrid.src.evaluate_hybrid import evaluate_predictions
from phase2.hybrid.src.hybrid_features import build_hybrid_features
from phase2.hybrid.src.hybrid_model import get_module_dir
from phase2.data.src.data_loader import METRICS_DIR

# Settings to try. The first one matches the Phase 1 baseline exactly.
SETTINGS_TO_TRY = [
    {"max_depth": 12},
    {"max_depth": 20},
    {"max_depth": None},
    {"max_depth": None, "min_samples_leaf": 5},
    {"max_depth": None, "max_features": "log2"},
    {"max_depth": 12, "n_estimators": 300},
]


def score_one_setting(train_features, train_labels, train_mask, val_features, val_labels, val_mask, settings):
    """Trains 12 forests with one set of settings and returns the validation macro scores."""
    probabilities = np.zeros((len(val_features), 12))

    for endpoint_index in range(12):
        observed = np.where(train_mask[:, endpoint_index])[0]
        forest = RandomForestClassifier(
            n_estimators=settings.get("n_estimators", 100),
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            max_depth=settings.get("max_depth", 12),
            min_samples_leaf=settings.get("min_samples_leaf", 1),
            max_features=settings.get("max_features", "sqrt"),
        )
        forest.fit(train_features[observed], train_labels[observed, endpoint_index].astype(int))
        probabilities[:, endpoint_index] = forest.predict_proba(val_features)[:, 1]

    results = evaluate_predictions(probabilities, val_labels, val_mask)
    macro = results.iloc[-1]
    return float(macro["ROC-AUC"]), float(macro["PR-AUC"])


def get_phase1_rf_validation_scores():
    """Reads the Phase 1 Random Forest validation scores for reference."""
    results_file = get_module_dir().parents[1] / "model_evaluation_results.csv"
    phase1 = pd.read_csv(results_file)
    forest_rows = phase1[phase1["model"] == "Random Forest"]
    return float(forest_rows["val_roc_auc"].mean()), float(forest_rows["val_pr_auc"].mean())


def main():
    loader = Tox21DataLoader()
    train_labels, train_mask, _ = loader.load_labels(split="train")
    val_labels, val_mask, _ = loader.load_labels(split="val")
    train_features, _ = build_hybrid_features(loader, "train")
    val_features, _ = build_hybrid_features(loader, "val")

    print("Validation macro scores for the hybrid feature set (133 features)")
    print("The test split is not used in this script.")
    print()

    rows = []
    for settings in SETTINGS_TO_TRY:
        roc_auc, pr_auc = score_one_setting(
            train_features, train_labels, train_mask,
            val_features, val_labels, val_mask, settings,
        )
        label = ", ".join(f"{k}={v}" for k, v in settings.items())
        print(f"  {label:45s} ROC-AUC {roc_auc:.4f}   PR-AUC {pr_auc:.4f}")
        rows.append({"settings": label, "val_roc_auc": round(roc_auc, 4), "val_pr_auc": round(pr_auc, 4)})

    phase1_roc, phase1_pr = get_phase1_rf_validation_scores()
    print()
    print(f"  {'Phase 1 standalone RF (2053 features)':45s} ROC-AUC {phase1_roc:.4f}   PR-AUC {phase1_pr:.4f}")
    rows.append(
        {
            "settings": "Phase 1 standalone RF (2053 features, max_depth=12)",
            "val_roc_auc": round(phase1_roc, 4),
            "val_pr_auc": round(phase1_pr, 4),
        }
    )

    best = max(rows[:-1], key=lambda r: r["val_roc_auc"])
    print()
    print(f"Best hybrid setting on validation: {best['settings']} ({best['val_roc_auc']:.4f})")
    if best["val_roc_auc"] < phase1_roc:
        print("Even the best hybrid setting stays below the standalone Random Forest,")
        print("so the gap is caused by the features, not by the forest settings.")

    output_file = METRICS_DIR / "hybrid_rf_settings_check.csv"
    pd.DataFrame(rows).to_csv(output_file, index=False)
    print(f"\nSaved to {output_file.name}")


if __name__ == "__main__":
    main()
