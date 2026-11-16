> **Current application:** The Streamlit predictor now serves the revised residual DL, 2,181-feature fusion RF, stronger raw-feature RF and 80/20 probability blend. The serving bundle uses 200 trees, depth 15 and minimum leaf size 2. Macro reporting rounds endpoint metrics to four decimals before averaging.

# Phase 2 — ML + DL Hybrid Model

**Role:** Hybrid modelling — combining the data pipeline module’s features with the deep learning module’s learned representation
**Status:** Complete. Phase 2 now beats the Phase 1 baseline.
**Headline finding:** Combining the two models at the **probability** level works
(+0.0144 test ROC-AUC over Phase 1). Combining them at the **input** level does
not, and the measurements explaining why are in section 3.

---

## 1. What the hybrid is

Two ways of combining a Random Forest with the deep learning module’s network were implemented
and measured.

**(a) Feature fusion — concatenate the inputs.**

```
2048 Morgan bits + 5 descriptors          the deep learning module’s network (frozen)
        (2053 raw)                           2053 -> 512 -> 256 -> 128 (z)
             |                                            |
             +------------- CONCATENATE ------------------+
                                |
                    2181 features per molecule
                                |
                 12 Random Forests, one per endpoint
```

This replaces the original design, which fed the forest only the 128 embeddings
plus 5 descriptors (133 features) and **discarded all 2,048 Morgan bits**. That
is what dropped test ROC-AUC to 0.6785.

**(b) Probability blending — combine the outputs.**

```
Random Forest on raw features  ->  P_rf  --\
                                            >--  alpha * P_rf + (1-alpha) * P_dl
the deep learning module’s network             ->  P_dl  --/
```

`alpha` is tuned on the **validation split only** over
[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9], then applied once to test.
Validation selected **alpha = 0.80**.

Diagram: `reports/plots/hybrid_architecture.png`

---

## 2. Results

All models were scored on the **same 782 test molecules**, using only observed
labels, with the same scikit-learn metric functions and the same 0.5 decision
threshold.

### Macro average across the 12 endpoints (test split)

| Model | Features | ROC-AUC | PR-AUC | vs Phase 1 |
|---|---|---|---|---|
| Standalone Random Forest (Phase 1 baseline) | 2053 | 0.7541 | 0.3962 | — |
| Standalone Deep Learning (Deep learning, retrained) | 2053 → 12 | 0.7266 | 0.3298 | −0.0275 |
| Hybrid, feature fusion | 2181 | 0.7156 | 0.3284 | −0.0385 |
| Random Forest, upgraded settings | 2053 | 0.7674 | **0.4063** | **+0.0133** |
| **Hybrid, probability blending (alpha=0.80)** | 2053 + DL | **0.7685** | 0.3983 | **+0.0144** |

The blended ensemble is the strongest model and beats the Phase 1 baseline on
both ROC-AUC (+0.0144) and PR-AUC (+0.0021).

Two separate changes produced the gain, and they are worth separating:

1. **Upgraded forest settings** (200 trees, depth 15, `min_samples_leaf=2`
   instead of 100 trees / depth 12) are worth **+0.0133** on their own.
   Validation: 0.7866 → 0.8044.
2. **Blending in the network's probabilities** adds a further **+0.0011**.
   Validation: 0.8044 → 0.8079.

So most of the improvement is better forest capacity, and the learned
representation contributes a small additional amount. Stating that split
honestly matters more than attributing the whole gain to the hybrid.

Full tables: `reports/phase1_vs_phase2_summary.csv`,
`reports/ensemble_test_evaluation.csv`, `reports/raw_rf_test_evaluation.csv`,
`reports/comparison_roc_auc.csv` (also `pr_auc`, `f1_score`, `mcc`).
Test predictions: `reports/ensemble_test_predictions.csv`.

---

## 3. Why feature fusion fails but blending works

This is the central finding, and it is measured rather than asserted.

### The measurement

`src/fusion_study.py`, validation split only:

| Configuration | Val macro ROC-AUC |
|---|---|
| raw 2053, Phase 1 forest settings | 0.7866 |
| **raw 2053, upgraded forest** | **0.8044** |
| fused 2181, upgraded forest | 0.7800 |
| fused 2181, `max_features=0.1` | 0.7783 |
| fused 2181, `max_features=0.3` | 0.7658 |
| fused 2181, `max_features=log2` | 0.7881 |
| embeddings only (128) | 0.7699 |
| embeddings + descriptors (133) | 0.7733 |

Adding the 128 embeddings to the 2,053 raw features **costs 0.0244 validation
ROC-AUC**. No `max_features` setting recovers it.

### The reason

A Random Forest picks each split by impurity gain over `sqrt(n_features)`
sampled columns — about 47 of 2,181. A continuous embedding column offers
thousands of candidate thresholds; a binary Morgan bit offers exactly one. So
the 128 embeddings win far more splits than their 6% share of the columns
warrants, and they crowd out the fingerprint evidence the forest was good at.
The embeddings also come from a network that fits the training molecules much
better than the held-out ones, so the columns that dominate the splits are the
ones carrying the most training-set-specific noise.

Probability blending avoids the problem entirely. Each model is fitted on the
representation that suits it — trees on sparse binary bits, the network on dense
combinations — and only the two outputs are averaged. Because the two make
differently-shaped errors, averaging cancels part of the uncorrelated component.

A three-way blend (raw forest + fused forest + network) was also tested and
reached 0.8083 on validation against 0.8079 for the two-way blend. That
difference is noise on 782 molecules, so the simpler two-way blend was kept.

### What this means for the original design

The first version of this module was not merely under-tuned: discarding the
2,048 Morgan bits removed the signal the Phase 1 forest depended on, and no
forest hyperparameter could compensate. Fusing the bits back in fixes that
specific loss (0.6785 → 0.7156) but introduces the crowding problem above,
which is why the shipped ensemble blends probabilities instead.

---

## 4. Handling of missing labels

Missing labels are never treated as negatives. For each endpoint, the training
set is filtered to only the molecules whose label for *that* endpoint was
measured, using the data pipeline module’s observation mask:

```python
observed = mask[:, endpoint_index]
x = features[observed]
y = labels[observed, endpoint_index]
```

This is why a separate forest is trained per endpoint rather than one
multi-output model: a molecule can have a label for SR-MMP but not for NR-AR,
and per-endpoint training keeps that molecule for SR-MMP instead of discarding
it or inventing a label.

Training molecule counts per endpoint match the Phase 1 counts exactly (for
example NR-AR: 5853 molecules, 233 active), confirming the filtering is correct.

Evaluation applies the same rule: a molecule contributes to an endpoint's score
only if its label for that endpoint exists.

---

## 5. Leakage prevention

| Control | How |
|---|---|
| Splits | the data pipeline module’s frozen scaffold split, unchanged (6259 / 782 / 782) |
| Descriptor scaling | Fitted on training data only in Phase 1, reused as-is |
| Model fitting | Forests fitted on the training split only |
| Model selection | Feature set, forest settings and the blending weight alpha all decided on **validation** |
| Blending weight | alpha swept on validation only; test scored once at the selected alpha=0.80 |
| Test usage | Touched once, for the final numbers in section 2 |
| Row alignment | Molecule order checked between raw features, embeddings, DL probabilities and labels on all three splits; mismatch raises an error |

**One caveat to state openly:** the training embeddings come from a network that
was itself trained on those same training labels, so the training embeddings
are partly fitted to their own labels. This does not affect the reported test
numbers, because the test embeddings come from a network that never saw any
test label. It is the standard caveat for stacking a model on another model's
learned features, and it is worth mentioning if asked.

---

## 6. Files

```
phase2/member3_hybrid_model/
├── README.md                          # this file
├── train_hybrid.py                    # entry point, runs the whole pipeline
├── configs/
│   └── hybrid_config.json             # feature sources + forest settings
├── src/
│   ├── hybrid_features.py             # builds the 2181-feature fused matrix
│   ├── hybrid_model.py                # 12 forests + blend_predictions()
│   ├── dl_predictions.py              # scores the deep learning module’s checkpoint for blending
│   ├── fusion_study.py                # validation-only study behind section 3
│   ├── evaluate_hybrid.py             # metrics + three-way comparison
│   ├── check_rf_settings.py           # validation-only settings check
│   └── make_diagram.py                # architecture + comparison plots
├── tests/
│   └── test_hybrid.py                 # fusion + blending test suite
├── models/                            # 12 saved forests, rf_<endpoint>.joblib
└── reports/
    ├── phase1_vs_phase2_summary.csv   # the headline table in section 2
    ├── test_evaluation.csv            # fused forest per-endpoint test metrics
    ├── raw_rf_test_evaluation.csv     # upgraded raw forest test metrics
    ├── ensemble_test_evaluation.csv   # blended ensemble test metrics
    ├── ensemble_test_predictions.csv  # 9384 rows: 782 molecules x 12 endpoints
    ├── test_predictions.csv           # same, for the fused forest
    ├── validation_evaluation.csv      # per-endpoint validation metrics
    ├── feature_ablation_validation.csv
    ├── alpha_tuning_validation.csv    # the alpha sweep
    ├── fusion_study_features.csv      # section 3 feature-set table
    ├── fusion_study_blending.csv      # section 3 blending table
    ├── rf_settings_check.csv
    ├── comparison_roc_auc.csv         # (also pr_auc, f1_score, mcc)
    └── plots/
        ├── hybrid_architecture.png
        └── model_comparison.png
```

### What each main file does

- **`hybrid_features.py`** — reads the data pipeline module’s raw 2,053 features and the deep learning module’s
  saved embeddings, checks the molecule order matches row-for-row, and
  concatenates them into 2,181 columns. The `use_raw_features` /
  `use_embeddings` / `use_descriptors` switches produce the four ablation arms
  in section 3.
- **`hybrid_model.py`** — `select_observed_rows()` applies the missing-label
  mask; `train_all_endpoints()` fits the 12 forests; `save_forests()` writes
  them to `models/`; `blend_predictions()` implements
  `alpha * P_rf + (1 - alpha) * P_dl`.
- **`dl_predictions.py`** — rebuilds the deep learning module’s network from their own config
  and checkpoint and scores a split with it. Nothing is retrained, so the
  blended numbers use their model exactly as trained.
- **`fusion_study.py`** — the validation-only evidence for section 3: the
  feature-set comparison, the `max_features` attempts, and the alpha sweep.
- **`evaluate_hybrid.py`** — `evaluate_predictions()` computes per-endpoint
  metrics on observed labels only; `build_comparison()` joins our results with
  the result files the other two members already produced.
- **`check_rf_settings.py`** — the validation-only evidence that the result is
  not a hyperparameter artifact.

---

## 7. How to run

From the repository root:

```bash
python -m phase2.member3_hybrid_model.train_hybrid
```

Then, optionally:

```bash
python -m phase2.member3_hybrid_model.src.fusion_study
```

```bash
python -m phase2.member3_hybrid_model.src.check_rf_settings
```

```bash
python -m phase2.member3_hybrid_model.src.make_diagram
```

Reusing the saved forests without retraining:

```python
from phase2.member3_hybrid_model.src import load_forests, predict_probabilities
forests = load_forests()
probabilities = predict_probabilities(forests, test_features)
```

**Version note.** the model artifacts were serialized with `scikit-learn==1.9.1` because the 12
files in `models/` were pickled with that version, and scikit-learn does not
guarantee that a model saved by one version loads correctly in another. If you
are on a different version, either install the pinned one or just retrain —
`train_hybrid.py` rebuilds all 12 forests in about 2 seconds and overwrites
`models/` with files matching your own version.

---

## 8. Note for the team

While reading the Phase 1 outputs I found that `results/metrics.csv` and
`model_evaluation_results.csv` disagree on the Random Forest test scores — for
NR-AR they give 0.7387 and 0.6922 ROC-AUC respectively, and the macro averages
are 0.7541 and 0.7498. They appear to come from two different Phase 1 runs.

This module uses `results/metrics.csv` for test comparisons, because that is the
file containing the full metric set (F1, MCC, balanced accuracy), and
`model_evaluation_results.csv` only for validation scores, which are the only
place validation numbers were saved. The conclusion is unaffected: the blended
ensemble scores above the Phase 1 forest under either file. Someone should confirm
which Phase 1 run is authoritative before the final report.
