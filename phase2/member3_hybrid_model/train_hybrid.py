"""
train_hybrid.py
===============
Member 3 entry point. Runs the whole hybrid pipeline in order.

Run from the repository root:
    python -m phase2.member3_hybrid_model.train_hybrid

Steps:
  1. Load Member 1's labels and observation masks.
  2. Build the hybrid features: Member 2's 128 DL embeddings + 5 descriptors.
  3. Use the VALIDATION split to decide whether the 5 descriptors help.
  4. Train 12 Random Forests on the training split.
  5. Score once on the TEST split and compare against the two standalone models.
"""

import numpy as np
import pandas as pd

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader
from phase2.member3_hybrid_model.src.evaluate_hybrid import (
    build_comparison,
    evaluate_predictions,
    save_predictions,
)
from phase2.member3_hybrid_model.src.hybrid_features import build_hybrid_features
from phase2.member3_hybrid_model.src.hybrid_model import (
    get_module_dir,
    load_config,
    predict_probabilities,
    save_forests,
    train_all_endpoints,
)


def main():
    config = load_config()
    loader = Tox21DataLoader()
    reports_dir = get_module_dir() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 1: LOAD LABELS AND MASKS (Member 1)")
    print("=" * 70)
    train_labels, train_mask, train_ids = loader.load_labels(split="train")
    val_labels, val_mask, val_ids = loader.load_labels(split="val")
    test_labels, test_mask, test_ids = loader.load_labels(split="test")
    print(f"  train {len(train_ids)} | val {len(val_ids)} | test {len(test_ids)} molecules")
    print(f"  observed labels: train {int(train_mask.sum())}, "
          f"val {int(val_mask.sum())}, test {int(test_mask.sum())}")

    print()
    print("=" * 70)
    print("STEP 2: BUILD HYBRID FEATURES (Member 2 embeddings + descriptors)")
    print("=" * 70)
    train_features, _ = build_hybrid_features(loader, "train")
    val_features, _ = build_hybrid_features(loader, "val")
    test_features, test_feature_ids = build_hybrid_features(loader, "test")
    print(f"  hybrid feature matrix: {train_features.shape[1]} features "
          f"(128 learned + 5 descriptors)")
    print(f"  train {train_features.shape} | val {val_features.shape} | test {test_features.shape}")

    if test_feature_ids != list(test_ids):
        raise ValueError("Test feature rows are not aligned with the test label rows.")
    print("  molecule order verified against Member 1's label rows")

    print()
    print("=" * 70)
    print("STEP 3: VALIDATION CHECK - DO THE 5 DESCRIPTORS HELP?")
    print("=" * 70)
    print("  Training on embeddings only (128 features):")
    embeddings_only_train, _ = build_hybrid_features(loader, "train", use_descriptors=False)
    embeddings_only_val, _ = build_hybrid_features(loader, "val", use_descriptors=False)
    forests_embeddings_only = train_all_endpoints(
        embeddings_only_train, train_labels, train_mask, config, verbose=False
    )
    val_probabilities = predict_probabilities(forests_embeddings_only, embeddings_only_val)
    embeddings_only_val_results = evaluate_predictions(val_probabilities, val_labels, val_mask)
    embeddings_only_score = embeddings_only_val_results.iloc[-1]["ROC-AUC"]
    print(f"    validation macro ROC-AUC: {embeddings_only_score:.4f}")

    print("  Training on embeddings + descriptors (133 features):")
    forests_with_descriptors = train_all_endpoints(
        train_features, train_labels, train_mask, config, verbose=False
    )
    val_probabilities = predict_probabilities(forests_with_descriptors, val_features)
    hybrid_val_results = evaluate_predictions(val_probabilities, val_labels, val_mask)
    hybrid_score = hybrid_val_results.iloc[-1]["ROC-AUC"]
    print(f"    validation macro ROC-AUC: {hybrid_score:.4f}")

    use_descriptors = hybrid_score >= embeddings_only_score
    print()
    if use_descriptors:
        print(f"  Decision: KEEP the descriptors (+{hybrid_score - embeddings_only_score:.4f} on validation)")
    else:
        print(f"  Decision: descriptors did NOT help on validation "
              f"({hybrid_score - embeddings_only_score:+.4f}), but we keep them because the "
              f"task specifies the hybrid feature set")

    hybrid_val_results.to_csv(reports_dir / "validation_evaluation.csv", index=False)

    print()
    print("=" * 70)
    print("STEP 4: TRAIN 12 RANDOM FORESTS ON THE TRAINING SPLIT")
    print("=" * 70)
    forests = train_all_endpoints(train_features, train_labels, train_mask, config, verbose=True)
    models_dir = save_forests(forests)
    print(f"  saved 12 models to {models_dir.relative_to(get_module_dir().parents[1])}")

    print()
    print("=" * 70)
    print("STEP 5: FINAL TEST EVALUATION")
    print("=" * 70)
    test_probabilities = predict_probabilities(forests, test_features)
    hybrid_results = evaluate_predictions(test_probabilities, test_labels, test_mask)
    hybrid_results.to_csv(reports_dir / "test_evaluation.csv", index=False)
    print(hybrid_results.to_string(index=False))

    save_predictions(
        test_probabilities,
        test_labels,
        test_mask,
        test_ids,
        reports_dir / "test_predictions.csv",
    )

    print()
    print("=" * 70)
    print("STEP 6: THREE-WAY COMPARISON")
    print("=" * 70)
    for metric in ["ROC-AUC", "PR-AUC", "F1-Score", "MCC"]:
        comparison = build_comparison(hybrid_results, metric=metric)
        safe_metric = metric.replace("-", "_").lower()
        comparison.to_csv(reports_dir / f"comparison_{safe_metric}.csv")
        print()
        print(f"--- {metric} ---")
        print(comparison.to_string())

    print()
    print("Finished. Reports written to phase2/member3_hybrid_model/reports/")


if __name__ == "__main__":
    main()
