# Phase 2 — Member 3: ML + DL Hybrid Model

**Role:** Hybrid modelling — Random Forest on learned Deep Learning features
**Status:** Complete, with a negative result that is reported honestly
**Headline finding:** On this dataset the hybrid **does not improve** on either standalone model.

---

## 1. What the hybrid is

```
Member 2's trained network (frozen)              5 molecular descriptors
   2053 features                                 MolWt, TPSA, MolLogP,
        ↓                                        NumHDonors, NumHAcceptors
   Dense 512 → Dense 256 → Dense 128  (z)                 ↓
        ↓                                                 ↓
   128 learned features  ──── CONCATENATE ──── 5 descriptors
                                   ↓
                         133 features per molecule
                                   ↓
                  12 Random Forests, one per endpoint
                                   ↓
                     12 toxicity probabilities
```

Diagram: `reports/plots/hybrid_architecture.png`

The learned features are the **128-dimensional bottleneck embedding** from
`MultiTaskToxicNet.latent_layer`, the layer just before Member 2's 12-output
classifier. Member 2 already extracted these for all three splits, so we read
their saved parquet files rather than re-running the network. Their model is
used exactly as they trained it, and none of their code was modified.

---

## 2. Results

All three approaches were scored on the **same 782 test molecules**, using only
observed labels, with the same scikit-learn metric functions and the same 0.5
decision threshold.

### Macro average across the 12 endpoints (test split)

| Model | Features | ROC-AUC | PR-AUC | F1 | MCC |
|---|---|---|---|---|---|
| **Standalone Random Forest** (Phase 1) | 2053 | **0.7541** | **0.3962** | **0.3969** | **0.3141** |
| **Standalone Deep Learning** (Member 2) | 2053 → 12 | 0.7179 | 0.3376 | 0.3590 | 0.2856 |
| **Hybrid** (Member 3) | 133 | 0.6785 | 0.3037 | 0.3155 | 0.2819 |

**The hybrid is the weakest of the three.** It is 0.0756 ROC-AUC below the
standalone Random Forest and 0.0394 below the standalone deep model.

Per endpoint, the hybrid beats the standalone Random Forest on ROC-AUC in only
**1 of 12** cases (SR-ARE, +0.0082). Its single clear win is NR-AR-LBD against
the deep model (+0.0616).

Full tables: `reports/comparison_roc_auc.csv`, `comparison_pr_auc.csv`,
`comparison_f1_score.csv`, `comparison_mcc.csv`.
Chart: `reports/plots/model_comparison.png`.

### Is this just bad Random Forest settings?

No. `src/check_rf_settings.py` tries six forest configurations and scores them
on the **validation split only**:

| Hybrid forest settings | Val ROC-AUC | Val PR-AUC |
|---|---|---|
| max_depth=12 (matches Phase 1) | 0.7125 | 0.3616 |
| max_depth=20 | 0.7119 | 0.3601 |
| max_depth=None | 0.7115 | 0.3547 |
| max_depth=None, min_samples_leaf=5 | **0.7346** | **0.3736** |
| max_depth=None, max_features=log2 | 0.7193 | 0.3604 |
| max_depth=12, n_estimators=300 | 0.7249 | 0.3639 |
| **Phase 1 standalone RF (2053 features)** | **0.7866** | **0.4643** |

Even the best hybrid setting (0.7346) stays well below the standalone Random
Forest (0.7866) **on validation**, where no test data was involved. The gap is
caused by the features, not by the forest settings.

### Do the 5 descriptors help?

Measured on validation: embeddings alone score 0.7140, embeddings + descriptors
score 0.7125. The descriptors make **no useful difference** (−0.0015). This is
expected, because those same 5 descriptors were already part of the 2053 inputs
the network saw, so their information is largely present in the embedding
already.

---

## 3. Why the hybrid loses — what to say in the viva

Three honest reasons, in order of importance:

1. **The bottleneck throws information away.** The 128-dim embedding is a
   compression of 2053 features. The Random Forest can only use what survived
   that compression, while the standalone forest sees all 2048 Morgan bits.
   Individual substructure bits that a tree can split on cleanly are blended
   into dense combinations in the embedding.

2. **The embedding was shaped for a linear classifier, not for trees.** Member 2
   trained `z` so that a single `Linear(128, 12)` layer could separate the
   classes. A Random Forest splits on one axis at a time, so a representation
   tuned for linear separation does not suit it. This shows up clearly: the
   linear head on `z` scores 0.7179, while the forest on the same `z` scores
   0.6785 — the forest is a worse reader of this representation than the layer
   it was designed for.

3. **Random Forests are already strong on sparse binary fingerprints.** Trees
   handle high-dimensional sparse binary inputs well, which is exactly what
   Morgan bits are. There was little room for a learned representation to add
   value here.

**This is a legitimate scientific result, not a failed task.** The project
brief asks us to show whether the hybrid improves performance, and the evidence
says it does not. Reporting that correctly is worth more than inventing an
improvement.

---

## 4. Handling of missing labels

Missing labels are never treated as negatives. For each endpoint, the training
set is filtered to only the molecules whose label for *that* endpoint was
measured, using Member 1's observation mask:

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
| Splits | Member 1's frozen scaffold split, unchanged (6259 / 782 / 782) |
| Descriptor scaling | Fitted on training data only in Phase 1, reused as-is |
| Model fitting | Forests fitted on the training split only |
| Model selection | Descriptor question and forest settings decided on **validation** |
| Test usage | Touched once, for the final numbers in section 2 |
| Row alignment | Molecule order checked between embeddings, descriptors and labels; mismatch raises an error |

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
├── requirements.txt
├── train_hybrid.py                    # entry point, runs the whole pipeline
├── configs/
│   └── hybrid_config.json             # feature sources + forest settings
├── src/
│   ├── hybrid_features.py             # builds the 133-feature matrix
│   ├── hybrid_model.py                # trains / saves / loads the 12 forests
│   ├── evaluate_hybrid.py             # metrics + three-way comparison
│   ├── check_rf_settings.py           # validation-only settings check
│   └── make_diagram.py                # architecture + comparison plots
├── models/                            # 12 saved forests, rf_<endpoint>.joblib
└── reports/
    ├── test_evaluation.csv            # hybrid per-endpoint test metrics
    ├── validation_evaluation.csv      # hybrid per-endpoint validation metrics
    ├── test_predictions.csv           # 9384 rows: 782 molecules x 12 endpoints
    ├── rf_settings_check.csv
    ├── comparison_roc_auc.csv         # (also pr_auc, f1_score, mcc)
    └── plots/
        ├── hybrid_architecture.png
        └── model_comparison.png
```

### What each main file does

- **`hybrid_features.py`** — reads Member 2's saved embeddings and Member 1's
  scaled descriptors, checks the molecule order matches, and joins them into
  133 columns. The `use_descriptors=False` switch is what produced the
  descriptor ablation in section 2.
- **`hybrid_model.py`** — `select_observed_rows()` applies the missing-label
  mask; `train_all_endpoints()` fits the 12 forests; `save_forests()` writes
  them to `models/`. The forest settings are deliberately identical to the
  Phase 1 baseline so the comparison isolates the features.
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

**Version note.** `requirements.txt` pins `scikit-learn==1.9.1` because the 12
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
place validation numbers were saved. The conclusion is unaffected: the hybrid
scores below the standalone forest under either file. Someone should confirm
which Phase 1 run is authoritative before the final report.
