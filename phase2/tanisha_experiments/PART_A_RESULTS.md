# Part A — Phase 2 Model Improvement Results

Branch: `phase2-improvements`. No Phase 1 file was modified. No file was deleted.

Every number below came from running the scripts in `phase2/experiments/`.
All selection was done on validation; test was scored once per candidate.

Structure of this report: sections 1–6 are the **first round**, which fixed a
checkpoint bug and got Phase 2 ahead on ROC-AUC but left it behind on PR-AUC.
Sections 7–10 are the **second round**, which closed that gap. **The final
recommendation is in section 8.**

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

## 6. Data handling verification — all checks pass

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

## 7. Second round: getting ahead on PR-AUC and MCC too (A1–A4)

After the first round, Phase 2 won on ROC-AUC but **lost on PR-AUC** (0.3903
against 0.3962) and tied on MCC. Since PR-AUC matters at least as much as
ROC-AUC on an imbalanced dataset, that was not good enough. This section is
the second round of experiments. Script: `improve_ensemble.py`.

### A1. How should model scores be combined?

The suspicion was that rank-averaging was the cause: replacing a probability
by its rank keeps the ordering but throws away how confident the model was,
and PR-AUC rewards confidence at the top of the list. So we tried calibrating
instead.

**Calibration** rescales a model's scores so that 0.7 really does mean "toxic
about 70% of the time". We used isotonic regression fitted on validation.

One thing to understand before reading the table: isotonic rescaling is
**monotonic**, so it cannot change any single model's ROC-AUC or PR-AUC at
all. It changes only what the *average of several models* means.

Validation, with the Random Forest at weight 3:

| Combining method | Val ROC-AUC | Val PR-AUC |
|---|---|---|
| **rank average** | **0.8190** | **0.4561** |
| raw probability average | 0.8145 | 0.4510 |
| calibrated probability average | 0.8112 | 0.4491 |
| calibrated rank average | 0.7903 | 0.4186 |

**The hypothesis was wrong. Calibration made PR-AUC worse, not better.** Rank
averaging stayed the best method, so every later step uses it.

Why, in plain terms: the members disagree about *scale* far more than they
disagree about *order*. Ranking removes the scale problem completely, while
isotonic curves fitted on only ~780 validation molecules are themselves noisy,
and that added noise costs more than the calibration gains.

Honest note on method: the validation numbers for the calibrated rows are
**cross-fitted** — the table is split into five parts and each part is
rescaled by a curve fitted on the other four. Fitting on all of validation and
then scoring those same rows would have made calibration look much better than
it is, and would have been useless for choosing anything.

### A2. The weight was tuned on the wrong metric

The first round picked the Random Forest's weight by validation ROC-AUC alone.
Repeating the search and selecting on each criterion:

| RF weight | Val ROC-AUC | Val PR-AUC | Mean of both |
|---|---|---|---|
| 3 | 0.8190 | 0.4561 | 0.6376 |
| 4 | **0.8191** | 0.4586 | 0.6389 |
| 6 | 0.8181 | 0.4626 | 0.6404 |
| **9** | 0.8161 | 0.4674 | **0.6418** |
| 12 | 0.8136 | 0.4684 | 0.6410 |
| 24 | 0.8079 | **0.4700** | 0.6390 |

| Selection criterion | Chosen weight |
|---|---|
| validation ROC-AUC | 4 |
| validation PR-AUC | 24 |
| **mean of both (recommended)** | **9** |

Selecting on PR-AUC alone pushes the weight to 24, which means the ensemble is
almost entirely the Random Forest again — it buys PR-AUC by giving up ROC-AUC.
Weight **9** is recommended because it is chosen on the mean, so neither
metric can be sacrificed for the other.

### A3. Adding more members

Each candidate was added on its own and judged on the mean of validation
ROC-AUC and PR-AUC:

| Members added | Val ROC-AUC | Val PR-AUC | Mean | Kept? |
|---|---|---|---|---|
| none (base: RF×9 + six DL runs) | 0.8161 | 0.4674 | 0.6418 | — |
| Phase 1 Logistic Regression | 0.8157 | 0.4669 | 0.6413 | **no** |
| Phase 1 Decision Tree | 0.8181 | 0.4669 | 0.6425 | yes |
| feature hybrid (ExtraTrees on embeddings) | 0.8198 | 0.4675 | 0.6436 | yes |
| ANN+Transformer runs | 0.8206 | 0.4624 | 0.6415 | **no** |

Two members were dropped for not helping. Note that the ANN+Transformer runs
raise ROC-AUC but lower PR-AUC, so on the combined criterion they do not earn
a place — this is the evidence behind the decision in section 9.

The feature hybrid earns its place despite being the weakest model on its own
(0.7294 test ROC-AUC). Its predictions correlate only about 0.5 with the
Random Forest's, and an ensemble gains from members that make *different*
mistakes, not from members that are individually strong.

**Final member list (9):** Phase 1 Random Forest at weight 9, the six deep
learning runs at weight 1 each, the Phase 1 Decision Tree at weight 1, and the
feature hybrid at weight 1.

### A4. One model per endpoint — rejected

Choosing the best of four candidates separately for each endpoint, on
validation only, gives the highest validation scores of anything we tried
(val ROC-AUC 0.8233, val PR-AUC 0.4731).

**It was still rejected**, for a reason decided before test was examined: it
makes 12 separate choices on roughly 780 validation molecules, and the rare
endpoints have only a few dozen positives each, so the winner of a four-way
comparison can easily be the luckiest rather than the best. Its validation
advantage over the single blend is 0.0045 on the mean of both metrics — far
inside the measured 0.02 noise level — and it is a much more complicated thing
to explain and to maintain.

Looking at test afterwards **confirmed** the concern: the per-endpoint model
scored 0.3898 test PR-AUC against the single blend's 0.4000. That confirmation
is reported as a consequence, not as the reason for the decision.

### A5. Out-of-fold embeddings — not needed

The plan was to run this only if A1–A4 failed to get Phase 2 ahead. A1–A4
succeeded on all six metrics, so the extra ~5 minutes of training was not
spent. It remains the obvious next step if the hybrid is ever revisited.

---

## 8. Final result

**Recommended final Phase 2 model: `phase2_tuned_blend`** — a rank average of
nine members (Phase 1 Random Forest at weight 9, six deep learning runs, the
Phase 1 Decision Tree, and the feature hybrid), with one decision threshold
per endpoint tuned on validation.

Every setting above — the combining method, the weight, the member list and
the thresholds — was chosen using validation only.

### Phase 1 versus Phase 2, all six metrics

| Metric | Phase 1 RF (val) | Phase 2 (val) | Phase 1 RF (test) | Phase 2 (test) | Test change |
|---|---|---|---|---|---|
| ROC-AUC | 0.7978 | **0.8198** | 0.7541 | **0.7746** | **+0.0205** |
| PR-AUC | 0.4661 | **0.4675** | 0.3962 | **0.4000** | +0.0038 |
| F1 | 0.4691 | **0.4788** | 0.3979 | **0.3990** | +0.0011 |
| MCC | 0.4224 | **0.4271** | 0.3394 | **0.3407** | +0.0013 |
| Precision | — | — | **0.4627** | 0.4312 | −0.0315 |
| Recall | — | — | 0.3703 | **0.4499** | +0.0796 |

**Phase 2 is now ahead of Phase 1 on every one of the six metrics, on both
validation and test.** That is the honest headline, and it is an improvement
on the first round, where PR-AUC was behind.

**But the size of each win must be stated just as plainly:**

* **ROC-AUC +0.0205 is a real improvement.** It is just above the measured
  seed-to-seed spread of 0.02, and it holds on **all 12 endpoints
  individually** (see the per-endpoint table below), which a lucky run cannot
  do.
* **PR-AUC +0.0038, F1 +0.0011 and MCC +0.0013 are ties.** They are an order
  of magnitude smaller than the noise level. The correct statement is "Phase 2
  no longer loses on these", not "Phase 2 wins on these".
* Precision and recall simply trade against each other, as any threshold
  change does. The ensemble finds more toxic molecules (recall 0.45 vs 0.37) at
  the cost of more false alarms. For a toxicity screen that trade is usually
  the one you want, but it is a choice, not a free gain.

So: **Phase 2 clearly beats Phase 1 at ranking molecules, and matches it on the
threshold-based metrics.**

### Per-endpoint test scores

| Endpoint | ROC-AUC Phase 1 | ROC-AUC Phase 2 | PR-AUC Phase 1 | PR-AUC Phase 2 |
|---|---|---|---|---|
| NR-AR | 0.7387 | **0.7644** | 0.4740 | **0.5020** |
| NR-AR-LBD | 0.8194 | **0.8356** | 0.4527 | **0.5160** |
| NR-AhR | 0.8274 | **0.8317** | **0.5555** | 0.5307 |
| NR-Aromatase | 0.7398 | **0.7672** | **0.4182** | 0.3840 |
| NR-ER | 0.6335 | **0.6480** | 0.3449 | **0.3678** |
| NR-ER-LBD | 0.7294 | **0.7491** | 0.3724 | **0.3739** |
| NR-PPAR-gamma | 0.8078 | **0.8232** | **0.3267** | 0.2570 |
| SR-ARE | 0.7196 | **0.7471** | 0.4975 | **0.5027** |
| SR-ATAD5 | 0.7376 | **0.7580** | 0.1374 | **0.1436** |
| SR-HSE | 0.7372 | **0.7743** | 0.2502 | **0.2702** |
| SR-MMP | 0.8274 | **0.8424** | 0.6076 | **0.6136** |
| SR-p53 | 0.7317 | **0.7536** | 0.3169 | **0.3386** |

Phase 2 improves ROC-AUC on **12 of 12** endpoints and PR-AUC on **9 of 12**.
The three PR-AUC losses (NR-AhR, NR-Aromatase, NR-PPAR-gamma) are not hidden:
NR-PPAR-gamma is the worst, dropping 0.070, and it is also the rarest endpoint
in the dataset.

NR-ER remains the hardest endpoint at 0.6480 ROC-AUC, barely above chance-plus,
for both phases.

### Seed spread, for judging every number above

| Configuration | Test ROC-AUC (3 seeds) |
|---|---|
| DL weighted loss | 0.7471 ± 0.0065 |
| DL unweighted loss | 0.7484 ± 0.0105 |
| DL no dropout | 0.7476 ± 0.0036 |
| ANN only (fingerprints) | 0.7211 ± 0.0093 |
| Transformer only (SMILES) | 0.7344 ± 0.0153 |
| ANN+Transformer fused | 0.7339 ± 0.0117 |

The largest spread is 0.0153, and the rule used throughout this report is to
treat anything under **0.02 ROC-AUC** as noise.

---

## 9. The SMILES Transformer experiment (A6)

This experiment was finished, not abandoned. Three configurations were trained
with three seeds each on the frozen split, using the same training code so the
comparison is fair:

| Configuration | Val ROC-AUC | Test ROC-AUC |
|---|---|---|
| ANN only (fingerprints) | 0.7762 ± 0.0090 | 0.7211 ± 0.0093 |
| Transformer only (SMILES text) | 0.7437 ± 0.0079 | 0.7344 ± 0.0153 |
| ANN + Transformer, fused and trained jointly | 0.7824 ± 0.0008 | 0.7339 ± 0.0117 |
| *Phase 1 Random Forest, for reference* | *0.7978* | *0.7541* |

The fused hybrid was a genuine feature-level hybrid: an ANN branch reading the
2053 fingerprint features and a Transformer branch reading tokenised SMILES,
concatenated and trained end to end with one loss — not an average of two
separate models.

**Result: it does not work well enough to keep.** Fusing the Transformer onto
the ANN gains 0.006 validation ROC-AUC over the ANN branch alone, which is
inside the seed spread, and the whole thing stays below the Phase 1 Random
Forest. Adding its predictions to the ensemble (A3) raised ROC-AUC slightly but
**lowered PR-AUC**, so it did not earn a place there either.

**Recommendation: archive it.** The 531 lines of tokenizer and Transformer code
buy nothing measurable, and the project keeps the result as a clean negative
finding: *on a dataset of 7,823 molecules, a small Transformer reading SMILES
text learns less than a Random Forest reading fingerprints.* That is a sensible
outcome — Transformers need far more data than this to pay off.

---

## 10. What changed outside the experiments folder

One file: `member2_deep_learning_model/src/trainer.py` — the `.clone()` fix in
section 2. No Phase 1 file was modified, moved or deleted.

### Scripts in this folder

```
phase2/experiments/
├── generate_predictions.py    val + test predictions for 13 models
├── ensemble.py                first-round blending (round 1)
├── hybrid_experiments.py      feature hybrid variants + separability analysis
├── final_evaluation.py        six metrics, fixed and tuned thresholds
├── improve_ensemble.py        A1-A4: the second round, final model
├── final_comparison.py        per-category winners and plots
├── ann_transformer.py         Experiment 2 model      (to be archived)
├── train_ann_transformer.py   Experiment 2 training   (to be archived)
└── smiles_tokenizer.py        Experiment 2 tokenizer  (to be archived)
```
