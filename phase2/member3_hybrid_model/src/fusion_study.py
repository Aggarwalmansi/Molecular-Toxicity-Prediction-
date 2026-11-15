"""
fusion_study.py
===============
Validation-only study of how to combine Member 1's raw features with Member 2's
learned embeddings.

Why this exists
---------------
Simply concatenating the 128 dense embeddings onto the 2,053 raw features made
the Random Forest *worse*, not better: validation macro ROC-AUC fell from
0.8044 (raw only) to 0.7800 (fused). The cause is how Random Forests choose
splits. At each node the forest samples sqrt(2181) ~ 47 candidate columns and
picks the one with the best impurity gain. A continuous embedding column offers
thousands of candidate thresholds, while a binary Morgan bit offers exactly
one, so the embeddings win far more splits than their 6% share of the columns
would suggest. They crowd out the fingerprint evidence the Phase 1 forest
relied on.

This script tests the ways around that on the VALIDATION split only:
  - max_features settings that re-balance how often bits get sampled
  - duplicating/omitting blocks
  - blending the raw-feature forest with the DL model instead of fusing inputs

The test split is never touched here. Whatever wins on validation is what
train_hybrid.py then scores once on test.

Run from the repository root:
    python -m phase2.member3_hybrid_model.src.fusion_study
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader
from phase2.member3_hybrid_model.src.dl_predictions import (
    load_trained_dl_model,
    predict_dl_probabilities,
)
from phase2.member3_hybrid_model.src.evaluate_hybrid import evaluate_predictions
from phase2.member3_hybrid_model.src.hybrid_features import build_hybrid_features
from phase2.member3_hybrid_model.src.hybrid_model import (
    TARGET_ASSAYS,
    blend_predictions,
    get_module_dir,
    select_observed_rows,
)


def train_and_score(x_train, y_train, mask_train, x_val, y_val, mask_val, **rf_kwargs):
    """Trains 12 forests with the given settings and returns (val table, val probs)."""
    probs = np.zeros((x_val.shape[0], len(TARGET_ASSAYS)))
    for idx in range(len(TARGET_ASSAYS)):
        xt, yt, _ = select_observed_rows(x_train, y_train, mask_train, idx)
        forest = RandomForestClassifier(random_state=42, n_jobs=-1, **rf_kwargs)
        forest.fit(xt, yt)
        probs[:, idx] = forest.predict_proba(x_val)[:, 1]
    return evaluate_predictions(probs, y_val, mask_val), probs


def macro(results, metric="ROC-AUC"):
    return float(results.iloc[-1][metric])


def main():
    loader = Tox21DataLoader()
    y_train, mask_train, _ = loader.load_labels(split="train")
    y_val, mask_val, val_ids = loader.load_labels(split="val")

    raw_train, _ = build_hybrid_features(loader, "train", use_embeddings=False)
    raw_val, _ = build_hybrid_features(loader, "val", use_embeddings=False)
    fused_train, _ = build_hybrid_features(loader, "train")
    fused_val, _ = build_hybrid_features(loader, "val")

    base = dict(n_estimators=200, max_depth=15, min_samples_leaf=2, class_weight="balanced")
    phase1 = dict(n_estimators=100, max_depth=12, class_weight="balanced")

    rows = []

    print("=" * 78)
    print("A. RANDOM FOREST FEATURE-SET AND max_features STUDY (validation only)")
    print("=" * 78)
    experiments = [
        ("raw 2053, Phase 1 RF settings", raw_train, raw_val, phase1),
        ("raw 2053, upgraded RF", raw_train, raw_val, base),
        ("raw 2053, upgraded RF, max_features=0.1", raw_train, raw_val,
         {**base, "max_features": 0.1}),
        ("fused 2181, upgraded RF", fused_train, fused_val, base),
        ("fused 2181, upgraded RF, max_features=0.1", fused_train, fused_val,
         {**base, "max_features": 0.1}),
        ("fused 2181, upgraded RF, max_features=0.3", fused_train, fused_val,
         {**base, "max_features": 0.3}),
        ("fused 2181, upgraded RF, max_features=log2", fused_train, fused_val,
         {**base, "max_features": "log2"}),
    ]

    rf_val_probs = {}
    for name, xt, xv, kwargs in experiments:
        results, probs = train_and_score(xt, y_train, mask_train, xv, y_val, mask_val, **kwargs)
        rf_val_probs[name] = probs
        rows.append({
            "Configuration": name,
            "Features": int(xt.shape[1]),
            "Val macro ROC-AUC": round(macro(results), 4),
            "Val macro PR-AUC": round(macro(results, "PR-AUC"), 4),
        })
        print(f"  {name:46s} val macro ROC-AUC {macro(results):.4f}  "
              f"PR-AUC {macro(results, 'PR-AUC'):.4f}")

    print()
    print("=" * 78)
    print("B. PROBABILITY BLENDING WITH MEMBER 2's NETWORK (validation only)")
    print("=" * 78)
    dl_model = load_trained_dl_model()
    dl_val_probs, dl_val_ids = predict_dl_probabilities(loader, "val", model=dl_model)
    if list(dl_val_ids) != list(val_ids):
        raise ValueError("DL probability rows are not aligned with the validation label rows.")
    dl_results = evaluate_predictions(dl_val_probs, y_val, mask_val)
    print(f"  DL alone{'':38s} val macro ROC-AUC {macro(dl_results):.4f}")
    print()

    blend_rows = []
    for rf_name in ["raw 2053, upgraded RF", "fused 2181, upgraded RF"]:
        for alpha in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            blended = blend_predictions(rf_val_probs[rf_name], dl_val_probs, alpha=alpha)
            results = evaluate_predictions(blended, y_val, mask_val)
            blend_rows.append({
                "RF arm": rf_name,
                "alpha": alpha,
                "Val macro ROC-AUC": round(macro(results), 4),
                "Val macro PR-AUC": round(macro(results, "PR-AUC"), 4),
            })
            print(f"  {rf_name:34s} alpha={alpha:.2f}  "
                  f"val macro ROC-AUC {macro(results):.4f}")

    out_dir = get_module_dir() / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "fusion_study_features.csv", index=False)
    blends = pd.DataFrame(blend_rows)
    blends.to_csv(out_dir / "fusion_study_blending.csv", index=False)

    best = blends.loc[blends["Val macro ROC-AUC"].idxmax()]
    print()
    print("=" * 78)
    print(f"Best blend on validation: {best['RF arm']} at alpha={best['alpha']:.2f} "
          f"-> {best['Val macro ROC-AUC']:.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
