# Part A — Phase 2 Model Improvement Results

Branch: `phase2-improvements`. No Phase 1 file was modified. No file was deleted.

Every number below came from running the scripts in `phase2/experiments/`.
All selection was done on validation; test was scored once per candidate.

---

## 1. The seven known findings — all verified

| # | Finding | Verdict |
|---|---|---|
| 1 | Member 2's checkpoint (0.7179) underperforms six reruns (0.7403–0.7603) | **Confirmed**, and the cause found |
| 2 | Rank-averaging six DL runs gives 0.7681 test ROC-AUC | **Confirmed** — measured 0.7680 |
| 3 | RF + six-seed DL rank-average gives ~0.7767 | **Confirmed** — measured 0.7779 after re-selecting the weight on validation |
| 4 | Hybrid training embeddings 99.95% linearly separable | **Confirmed** — measured 99.52% |
| 5 | ExtraTrees lifts the hybrid from 0.6785 to 0.7123 | **Confirmed and exceeded** — reached 0.7294 |
| 6 | Seed-to-seed spread ≈ 0.02 ROC-AUC | **Confirmed** — std 0.0036–0.0105 per configuration |
| 7 | Two Phase 1 result files disagree | **Resolved** — see section 5 |

---

## 2. Root cause of the deep learning problem (A2)

`phase2/member2_deep_learning_model/src/trainer.py` line 250 did this:

```python
best_state = {"model_state_dict": self.model.state_dict(), ...}
```

`state_dict()` returns a dictionary whose values are **references to the live
parameter tensors**, not copies. As training continues those tensors are
updated in place, so `best_state` silently follows the model. Minimal proof:

```
weights at the "best" moment were 1.0, then training continued to 99.0
  aliased["weight"] now reads: [99.0, 99.0, 99.0]
  cloned ["weight"] now reads: [1.0, 1.0, 1.0]
```

Consequences:

1. `torch.save()` ran *inside* the improvement branch, so **the `.pt` file on
   disk is correct** (best epoch).
2. But the restore at the end of `fit()` loaded the **final epoch's** weights,
   and the "Restored best model weights" message was false.
3. Everything computed after `fit()` used the final-epoch model: the reported
   test metrics **and the 128-dim embeddings handed to the hybrid**.

Measured cost: test ROC-AUC **0.7179** (final epoch, what was reported) versus
**0.7318** (the actual saved checkpoint).

**Fixed** by cloning the tensors. Member 4's code already did this correctly
(`class_weight_ablation.py` line 180), which is why their reruns scored higher.

### The deeper issue: the network overfits in two epochs

| epoch | train loss | val loss | val ROC-AUC |
|---|---|---|---|
| 1 | 1.080 | 1.240 | 0.7700 |
| **2** | **0.804** | **1.243** | **0.7966** ← best |
| 3 | 0.636 | 1.522 | 0.7772 |
| 10 | 0.242 | 3.060 | 0.7553 |
| 17 | 0.151 | 3.997 | 0.7541 |

Validation loss triples while training loss falls 7×. Early stopping worked
correctly; the model simply has nothing left to learn after epoch 2. Every one
of Member 4's nine runs also peaks at epoch 2.

### Weighted vs unweighted loss (3 seeds each)

| Configuration | Test ROC-AUC | Test PR-AUC | Val ROC-AUC |
|---|---|---|---|
| weighted | 0.7471 ± 0.0065 | 0.3286 ± 0.0145 | 0.7877 ± 0.0049 |
| unweighted | 0.7484 ± 0.0105 | 0.3489 ± 0.0177 | 0.7842 ± 0.0054 |
| no dropout | 0.7476 ± 0.0036 | 0.3207 ± 0.0209 | 0.7862 ± 0.0105 |

The gap between weighted and unweighted is **0.0012**, far inside the seed
spread. **Class weighting makes no real difference to ranking quality.** It
does change the 0.5-threshold operating point a lot (recall 0.64 vs 0.18), but
that is a threshold effect, handled properly in section 4.

Architecture was **not** changed — no experiment supported it.

---

## 3. Hybrid model (A3)

Selected on validation; test shown for the record.

| Variant | Features | Val ROC-AUC | Test ROC-AUC |
|---|---|---|---|
| **corrected_emb_extratrees** | 133 | **0.7913** | 0.7294 |
| corrected_emb_plus_morgan_rf | 2181 | 0.7838 | 0.7239 |
| member4_emb_extratrees | 133 | 0.7801 | 0.7484 |
| corrected_emb_rf | 133 | 0.7736 | 0.7175 |
| member4_emb_plus_morgan_rf | 2181 | 0.7696 | 0.7424 |
| original_hybrid_rf | 133 | 0.7125 | 0.6785 |

Two things fixed the hybrid, together worth **+0.051 test ROC-AUC**:

1. **Using embeddings from the correct checkpoint** (0.6785 → 0.7175). The
   final-epoch embeddings were 99.52% linearly separable on training data
   because the network had memorised it; the best-epoch embeddings are 95.44%.
2. **ExtraTrees instead of Random Forest** (0.7175 → 0.7294). ExtraTrees splits
   at random thresholds, which suits dense continuous embeddings better than
   Random Forest's greedy axis-aligned splits.

Adding Morgan fingerprints back **did not help** (0.7913 → 0.7838 on validation).

**Honest caveat:** `member4_emb_extratrees` scores higher on test (0.7484) but
lower on validation (0.7801). We report the validation winner, because picking
the test winner would be exactly the leakage this project is trying to avoid.

**Out-of-fold embeddings were not implemented.** They require training K
networks from scratch (~5 minutes for 5 folds). Given that the hybrid is not
the recommended model, I stopped and left this for your decision.

---

## 4. Ensembles and thresholds (A4, A5)

Selection on validation only; the winner was applied to test once.
Winner: `rf_plus_dl6_rank_w3` — rank-average of the Phase 1 Random Forest
(weight 3) with the six DL runs (weight 1 each).

### Fixed 0.5 threshold

| Model | Val ROC | Test ROC | Test PR | Test F1 | Test MCC |
|---|---|---|---|---|---|
| rf_phase1 (Phase 1 baseline) | 0.7978 | 0.7541 | **0.3962** | **0.3969** | **0.3141** |
| dl_member2_checkpoint | 0.7966 | 0.7318 | 0.3419 | 0.3111 | 0.2332 |
| dl6_rank_avg | 0.8060 | 0.7680 | 0.3641 | 0.2754 | 0.2090 |
| **rf_plus_dl6_rank_w3** | **0.8191** | **0.7779** | 0.3903 | 0.2806 | 0.2210 |

The ensemble's F1 and MCC look terrible here, and that is an artefact, not a
result: rank-averaged scores are uniformly spread over 0–1, so a 0.5 cutoff
flags the top half of all molecules. **For rank-based scores the 0.5 threshold
is meaningless.** Use the tuned table.

### Thresholds tuned per endpoint on validation, applied unchanged to test

| Model | Test ROC | Test PR | Test F1 | Test Precision | Test Recall | Test MCC |
|---|---|---|---|---|---|---|
| rf_phase1 | 0.7541 | **0.3962** | 0.3979 | 0.4627 | 0.3703 | **0.3394** |
| dl6_rank_avg | 0.7680 | 0.3641 | 0.3725 | 0.3965 | 0.4609 | 0.3120 |
| **rf_plus_dl6_rank_w3** | **0.7779** | 0.3903 | **0.4000** | 0.4233 | 0.4509 | 0.3385 |

ROC-AUC and PR-AUC are identical in both tables because they do not depend on
the threshold.

Threshold tuning helped every model: the Phase 1 forest's MCC rose from 0.3141
to 0.3394 and its precision from 0.3642 to 0.4627, with no retraining.

---

## 5. Which Phase 1 file is authoritative (finding 7)

Refitting the Phase 1 Random Forest with the exact settings in
`train_models.py` reproduces **`results/metrics.csv`** exactly: macro test
ROC-AUC **0.7541**, matching per-endpoint values.

`model_evaluation_results.csv` (macro 0.7498) is a different, earlier run and
is **not** reproducible from the current code. It is, however, the only file
containing Phase 1 **validation** scores.

**Decision:** use `results/metrics.csv` for all test comparisons. Both files
are kept; neither is deleted.

---

## 6. Data handling verification (A6) — all checks pass

```
1. split sizes: train 6259, val 782, test 782  | total 7823
   overlaps: train/val 0, train/test 0, val/test 0
2. mol_id row alignment: combined, morgan, descriptors_scaled -> True on all splits
3. rows with observed=False carrying a label: 0
   observed rows with a missing label: 0
   observed labels are 0/1 only: True
4. observed rows identical for every model: val 7049, test 7095
```

The loaders raise an error on any molecule-order mismatch rather than
continuing silently.

---

## 7. Recommendation

**Final Phase 2 model: `rf_plus_dl6_rank_w3`** — rank-average of the Phase 1
Random Forest (weight 3) with the six deep learning runs (weight 1 each), with
per-endpoint thresholds tuned on validation.

| Metric | Phase 1 RF | Phase 2 ensemble | Change |
|---|---|---|---|
| Test ROC-AUC | 0.7541 | **0.7779** | **+0.0238** |
| Test PR-AUC | **0.3962** | 0.3903 | −0.0059 |
| Test F1 (tuned) | 0.3979 | **0.4000** | +0.0021 |
| Test MCC (tuned) | **0.3394** | 0.3385 | −0.0009 |

**Stated plainly: Phase 2 beats Phase 1 on ROC-AUC by 0.0238, which is larger
than the 0.02 seed-noise threshold. On PR-AUC, F1 and MCC the two are tied —
the differences are within noise, and PR-AUC is marginally worse.**

This should be reported as "the ensemble ranks molecules better, and matches
the baseline elsewhere", not as a win across the board.

Why this model rather than the hybrid: the best hybrid reaches 0.7294 test
ROC-AUC, below the Phase 1 forest's 0.7541. The feature-based hybrid remains a
negative result and should stay in the report as one.

### Files produced

```
phase2/experiments/
├── generate_predictions.py      # A1: val + test predictions for 11 models
├── ensemble.py                  # A4: validation-selected blending
├── hybrid_experiments.py        # A3: hybrid variants + separability analysis
├── final_evaluation.py          # A5: six metrics, fixed and tuned thresholds
├── predictions/
│   └── val_test_predictions.parquet
└── results/
    ├── ensemble_validation_selection.csv
    ├── ensemble_final.csv
    ├── best_ensemble_predictions.parquet
    ├── hybrid_variants.csv
    ├── final_comparison_fixed_threshold.csv
    ├── final_comparison_tuned_threshold.csv
    └── final_per_endpoint_tuned.csv
```

One file changed outside this folder: `member2_deep_learning_model/src/trainer.py`
(the `.clone()` fix, 1 logical line plus a comment).
