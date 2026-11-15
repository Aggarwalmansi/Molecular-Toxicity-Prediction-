"""
train_hybrid.py
===============
Member 3 entry point. Runs the whole hybrid pipeline in order.

Run from the repository root:
    python -m phase2.member3_hybrid_model.train_hybrid

Steps:
  1. Load Member 1's labels and observation masks.
  2. Build the FUSED features: 2,048 Morgan bits + 5 descriptors + 128 DL
     embeddings = 2,181 per molecule.
  3. Validation ablation: embeddings-only (128) vs embeddings+descriptors (133)
     vs the fused 2,181 set, so the fusion decision is evidence-based.
  4. Train 12 Random Forests on the fused training features.
  5. Tune the blending weight alpha on the validation split only.
  6. Score the fused forest and the blended ensemble once on the 782 test
     molecules, and compare against the Phase 1 and Member 2 standalone models.
"""

import numpy as np
import pandas as pd

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader
from phase2.member3_hybrid_model.src.dl_predictions import (
    load_trained_dl_model,
    predict_dl_probabilities,
)
from phase2.member3_hybrid_model.src.evaluate_hybrid import (
    build_comparison,
    evaluate_predictions,
    save_predictions,
)
from phase2.member3_hybrid_model.src.hybrid_features import build_hybrid_features
from phase2.member3_hybrid_model.src.hybrid_model import (
    blend_predictions,
    get_module_dir,
    load_config,
    predict_probabilities,
    save_forests,
    train_all_endpoints,
)

PHASE1_TEST_ROC_AUC = 0.7541
PHASE1_TEST_PR_AUC = 0.3962


def macro_roc_auc(results):
    """Pulls the macro-average ROC-AUC out of an evaluation table."""
    return float(results.iloc[-1]["ROC-AUC"])


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
    print("STEP 2: BUILD FUSED FEATURES (raw 2053 + DL embeddings 128)")
    print("=" * 70)
    train_features, train_feature_ids = build_hybrid_features(loader, "train")
    val_features, val_feature_ids = build_hybrid_features(loader, "val")
    test_features, test_feature_ids = build_hybrid_features(loader, "test")
    print(f"  fused feature matrix: {train_features.shape[1]} features "
          f"(2048 Morgan bits + 5 descriptors + 128 learned)")
    print(f"  train {train_features.shape} | val {val_features.shape} | test {test_features.shape}")

    # Row-for-row ID verification on every split. A silent misalignment would
    # pair one molecule's features with another molecule's labels.
    for split_name, feature_ids, label_ids in [
        ("train", train_feature_ids, train_ids),
        ("val", val_feature_ids, val_ids),
        ("test", test_feature_ids, test_ids),
    ]:
        if list(feature_ids) != list(label_ids):
            raise ValueError(f"{split_name} feature rows are not aligned with its label rows.")
    print("  molecule order verified against Member 1's label rows on all 3 splits")

    print()
    print("=" * 70)
    print("STEP 3: VALIDATION ABLATION - DOES FEATURE FUSION HELP?")
    print("=" * 70)
    ablation_rows = []

    arms = [
        ("embeddings only (128)", dict(use_raw_features=False, use_descriptors=False)),
        ("embeddings + descriptors (133)", dict(use_raw_features=False, use_descriptors=True)),
        ("raw only (2053)", dict(use_raw_features=True, use_embeddings=False)),
        ("FUSED raw + embeddings (2181)", dict()),
    ]

    for arm_name, kwargs in arms:
        arm_train, _ = build_hybrid_features(loader, "train", **kwargs)
        arm_val, _ = build_hybrid_features(loader, "val", **kwargs)
        arm_forests = train_all_endpoints(arm_train, train_labels, train_mask, config, verbose=False)
        arm_val_probs = predict_probabilities(arm_forests, arm_val)
        arm_results = evaluate_predictions(arm_val_probs, val_labels, val_mask)
        score = macro_roc_auc(arm_results)
        ablation_rows.append({
            "Feature set": arm_name,
            "Features": int(arm_train.shape[1]),
            "Val macro ROC-AUC": round(score, 4),
            "Val macro PR-AUC": round(float(arm_results.iloc[-1]["PR-AUC"]), 4),
        })
        print(f"  {arm_name:34s} {arm_train.shape[1]:5d} features  "
              f"val macro ROC-AUC {score:.4f}")
        if arm_name.startswith("FUSED"):
            hybrid_val_results = arm_results
            hybrid_val_probs = arm_val_probs
            fused_val_score = score
        if arm_name.startswith("raw only"):
            raw_only_val_score = score
            raw_only_val_probs = arm_val_probs
            raw_only_forests = arm_forests

    ablation = pd.DataFrame(ablation_rows)
    ablation.to_csv(reports_dir / "feature_ablation_validation.csv", index=False)

    best_arm = ablation.loc[ablation["Val macro ROC-AUC"].idxmax(), "Feature set"]
    print()
    print(f"  Best feature set on validation: {best_arm}")
    hybrid_val_results.to_csv(reports_dir / "validation_evaluation.csv", index=False)

    # Input-level fusion turns out to HURT the forest: the 128 dense embedding
    # columns win far more splits than their 6% share of the columns warrants,
    # because a continuous feature offers thousands of candidate thresholds
    # where a binary Morgan bit offers one. So the forest that feeds the
    # ensemble is whichever arm validation prefers, and the learned
    # representation is combined at the probability level instead.
    fusion_hurts = raw_only_val_score > fused_val_score
    if fusion_hurts:
        print(f"  Input-level fusion COSTS {fused_val_score - raw_only_val_score:+.4f} "
              f"validation macro ROC-AUC against raw features alone.")
        print("  -> the ensemble will blend the raw-feature forest with the DL model, "
              "combining the learned representation at the probability level.")
        blend_arm_name = "raw 2053"
        blend_arm_kwargs = dict(use_raw_features=True, use_embeddings=False)
    else:
        print("  Input-level fusion helps; the fused forest will feed the ensemble.")
        blend_arm_name = "fused 2181"
        blend_arm_kwargs = {}

    print()
    print("=" * 70)
    print("STEP 4: TRAIN 12 RANDOM FORESTS ON THE FUSED FEATURES")
    print("=" * 70)
    rf_settings = config["random_forest"]
    print(f"  RandomForest(n_estimators={rf_settings['n_estimators']}, "
          f"max_depth={rf_settings['max_depth']}, "
          f"min_samples_leaf={rf_settings['min_samples_leaf']}, "
          f"class_weight='{rf_settings['class_weight']}')")
    forests = train_all_endpoints(train_features, train_labels, train_mask, config, verbose=True)
    models_dir = save_forests(forests)
    print(f"  saved 12 models to {models_dir.relative_to(get_module_dir().parents[1])}")

    print()
    print("=" * 70)
    print("STEP 5: TUNE BLENDING WEIGHT ALPHA ON VALIDATION ONLY")
    print("=" * 70)
    dl_model = load_trained_dl_model()
    dl_val_probs, dl_val_ids = predict_dl_probabilities(loader, "val", model=dl_model)
    dl_test_probs, dl_test_ids = predict_dl_probabilities(loader, "test", model=dl_model)
    if list(dl_val_ids) != list(val_ids) or list(dl_test_ids) != list(test_ids):
        raise ValueError("DL probability rows are not aligned with Member 1's label rows.")

    dl_val_results = evaluate_predictions(dl_val_probs, val_labels, val_mask)
    print(f"  fused RF (2181)     val macro ROC-AUC {fused_val_score:.4f}")
    print(f"  raw RF (2053)       val macro ROC-AUC {raw_only_val_score:.4f}")
    print(f"  DL alone            val macro ROC-AUC {macro_roc_auc(dl_val_results):.4f}")
    print(f"  blending arm: {blend_arm_name}")
    print()

    blend_rf_val_probs = raw_only_val_probs if fusion_hurts else hybrid_val_probs

    alpha_grid = config.get("ensemble", {}).get("alpha_grid", [0.5, 0.55, 0.6, 0.65, 0.7])
    alpha_rows = []
    best_alpha = None
    best_alpha_score = -np.inf

    for alpha in alpha_grid:
        blended_val = blend_predictions(blend_rf_val_probs, dl_val_probs, alpha=alpha)
        blended_results = evaluate_predictions(blended_val, val_labels, val_mask)
        score = macro_roc_auc(blended_results)
        alpha_rows.append({
            "RF arm": blend_arm_name,
            "alpha": alpha,
            "Val macro ROC-AUC": round(score, 4),
            "Val macro PR-AUC": round(float(blended_results.iloc[-1]["PR-AUC"]), 4),
        })
        print(f"  alpha={alpha:.2f}  val macro ROC-AUC {score:.4f}")
        if score > best_alpha_score:
            best_alpha_score = score
            best_alpha = alpha

    alpha_table = pd.DataFrame(alpha_rows)
    alpha_table.to_csv(reports_dir / "alpha_tuning_validation.csv", index=False)
    print()
    print(f"  Selected alpha = {best_alpha:.2f} (val macro ROC-AUC {best_alpha_score:.4f})")

    print()
    print("=" * 70)
    print("STEP 6: FINAL TEST EVALUATION (782 holdout molecules)")
    print("=" * 70)
    test_probabilities = predict_probabilities(forests, test_features)
    hybrid_results = evaluate_predictions(test_probabilities, test_labels, test_mask)
    hybrid_results.to_csv(reports_dir / "test_evaluation.csv", index=False)
    print("--- Feature-fused Random Forest (2,181 features) ---")
    print(hybrid_results.to_string(index=False))

    save_predictions(
        test_probabilities,
        test_labels,
        test_mask,
        test_ids,
        reports_dir / "test_predictions.csv",
    )

    # The blending arm was chosen on validation; score it once on test.
    if fusion_hurts:
        raw_test_features, _ = build_hybrid_features(loader, "test", use_raw_features=True,
                                                     use_embeddings=False)
        blend_rf_test_probs = predict_probabilities(raw_only_forests, raw_test_features)
        raw_results = evaluate_predictions(blend_rf_test_probs, test_labels, test_mask)
        raw_results.to_csv(reports_dir / "raw_rf_test_evaluation.csv", index=False)
        print()
        print("--- Raw-feature Random Forest (2,053 features, upgraded settings) ---")
        print(raw_results.to_string(index=False))
    else:
        blend_rf_test_probs = test_probabilities
        raw_results = None

    blended_test = blend_predictions(blend_rf_test_probs, dl_test_probs, alpha=best_alpha)
    ensemble_results = evaluate_predictions(blended_test, test_labels, test_mask)
    ensemble_results.to_csv(reports_dir / "ensemble_test_evaluation.csv", index=False)
    print()
    print(f"--- Blended ensemble ({blend_arm_name} RF + DL, alpha={best_alpha:.2f}) ---")
    print(ensemble_results.to_string(index=False))

    save_predictions(
        blended_test,
        test_labels,
        test_mask,
        test_ids,
        reports_dir / "ensemble_test_predictions.csv",
    )

    print()
    print("=" * 70)
    print("STEP 7: THREE-WAY COMPARISON")
    print("=" * 70)
    for metric in ["ROC-AUC", "PR-AUC", "F1-Score", "MCC"]:
        comparison = build_comparison(hybrid_results, metric=metric)
        safe_metric = metric.replace("-", "_").lower()
        comparison.to_csv(reports_dir / f"comparison_{safe_metric}.csv")
        print()
        print(f"--- {metric} ---")
        print(comparison.to_string())

    print()
    print("=" * 70)
    print("PHASE 2 vs PHASE 1 BASELINE")
    print("=" * 70)
    fused_roc = macro_roc_auc(hybrid_results)
    fused_pr = float(hybrid_results.iloc[-1]["PR-AUC"])
    ensemble_roc = macro_roc_auc(ensemble_results)
    ensemble_pr = float(ensemble_results.iloc[-1]["PR-AUC"])

    summary_rows = [
        {"Model": "Phase 1 Random Forest (baseline)",
         "Test macro ROC-AUC": PHASE1_TEST_ROC_AUC,
         "Test macro PR-AUC": PHASE1_TEST_PR_AUC,
         "ROC-AUC vs Phase 1": 0.0},
        {"Model": "Phase 2 DL MultiTaskToxicNet",
         "Test macro ROC-AUC": round(float(evaluate_predictions(
             dl_test_probs, test_labels, test_mask).iloc[-1]["ROC-AUC"]), 4),
         "Test macro PR-AUC": round(float(evaluate_predictions(
             dl_test_probs, test_labels, test_mask).iloc[-1]["PR-AUC"]), 4),
         "ROC-AUC vs Phase 1": None},
        {"Model": "Phase 2 fused RF (2,181 features)",
         "Test macro ROC-AUC": round(fused_roc, 4),
         "Test macro PR-AUC": round(fused_pr, 4),
         "ROC-AUC vs Phase 1": round(fused_roc - PHASE1_TEST_ROC_AUC, 4)},
    ]
    if raw_results is not None:
        raw_roc = macro_roc_auc(raw_results)
        summary_rows.append({
            "Model": "Phase 2 raw RF (2,053 features, upgraded)",
            "Test macro ROC-AUC": round(raw_roc, 4),
            "Test macro PR-AUC": round(float(raw_results.iloc[-1]["PR-AUC"]), 4),
            "ROC-AUC vs Phase 1": round(raw_roc - PHASE1_TEST_ROC_AUC, 4),
        })
    summary_rows.append({
        "Model": f"Phase 2 blended ensemble ({blend_arm_name} + DL, alpha={best_alpha:.2f})",
        "Test macro ROC-AUC": round(ensemble_roc, 4),
        "Test macro PR-AUC": round(ensemble_pr, 4),
        "ROC-AUC vs Phase 1": round(ensemble_roc - PHASE1_TEST_ROC_AUC, 4),
    })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(reports_dir / "phase1_vs_phase2_summary.csv", index=False)
    print(summary.to_string(index=False))

    best_roc = max(fused_roc, ensemble_roc)
    if raw_results is not None:
        best_roc = max(best_roc, macro_roc_auc(raw_results))
    print()
    if best_roc > PHASE1_TEST_ROC_AUC:
        print(f"  Phase 2 BEATS the Phase 1 baseline: {best_roc:.4f} > {PHASE1_TEST_ROC_AUC:.4f} "
              f"(+{best_roc - PHASE1_TEST_ROC_AUC:.4f})")
    else:
        print(f"  Phase 2 does NOT yet beat the Phase 1 baseline: "
              f"{best_roc:.4f} <= {PHASE1_TEST_ROC_AUC:.4f}")

    print()
    print("Finished. Reports written to phase2/member3_hybrid_model/reports/")


if __name__ == "__main__":
    main()
