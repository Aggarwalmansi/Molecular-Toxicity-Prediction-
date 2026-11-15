# Phase 2: Close the Gap on Phase 1, Then Simplify the Project

I am working on a college project called **Molecular Toxicity Prediction using the Tox21 dataset**.

Phase 1 is finished and frozen. Phase 2's Part A improvement work is also already finished — the results are in `phase2/experiments/PART_A_RESULTS.md`. **Read that file first.** Do not redo that work.

I want two things now:
1. Phase 2 should clearly beat Phase 1 — not just on one metric.
2. The project folders and code should be simple enough for all five of us to explain in a viva.

Work in order: Step 0, then Part A, then stop for my approval, then Part B.

---

## STEP 0 — Do This Before Anything Else

**My Part A work is uncommitted and could be lost.** On branch `phase2-improvements` these are modified but not committed:
- `phase2/member2_deep_learning_model/src/trainer.py` (the `.clone()` checkpoint fix)
- all 12 `phase2/member3_hybrid_model/models/rf_*.joblib` files
- the whole untracked `phase2/experiments/` folder

Commit all of it first, with a clear message, before touching anything else. Then confirm `git status` is clean.

---

## RULE 1 — Do Not Touch Phase 1 (strict)

These are Phase 1 and must not be modified, moved, renamed or deleted:

```
train_models.py            evaluate.py            app.py
model_evaluation_results.csv
results/                   (metrics.csv, test_predictions.csv, plots/, etc.)
tox21_project/
tox21_member2_handoff/
requirements.txt           requirements-eval.txt
```

You may **read** from them. Phase 2 code reading `results/metrics.csv` or the handoff artifacts is fine and should keep working. `app.py` is my live presentation dashboard — if it stops running, you have broken a rule.

All restructuring happens **inside `phase2/` only**.

---

## RULE 2 — Keep the Code Simple

All five team members must be able to read and explain this code in a viva.

- Simple Python, clear variable names, basic functions.
- No new classes unless genuinely needed. No clever abstractions.
- No function longer than about 50 lines.
- Comments only where the reason is not obvious. Do not comment every line.
- No new libraries beyond what is already installed. Reuse existing code and existing loaders.
- Prefer adding to an existing script over creating a new one.
- Do not make code complicated to make it look professional.

---

## RULE 3 — Data Handling (unchanged, still strict)

- Use the existing Phase 1 scaffold split. Never make a new split.
- Train on train. **Select and tune on validation. Touch test only for the final number.**
- Never use test results to pick a model, weight, threshold or setting.
- A missing label is never non-toxic — keep using the observation mask.
- Compare all models on the same test molecules and the same observed labels.
- Fail loudly on any molecule-order mismatch.
- `phase2/experiments/predictions/val_test_predictions.parquet` already holds validation and test predictions for 11 models. Reuse it. Do not regenerate predictions that already exist.

---

# PART A — Make Phase 2 Properly Better Than Phase 1

## Where the project actually stands

Current recommended model is `rf_plus_dl6_rank_w3`, with thresholds tuned on validation:

| Metric | Phase 1 RF | Phase 2 ensemble | Verdict |
|---|---|---|---|
| Test ROC-AUC | 0.7541 | **0.7779** | Phase 2 wins by 0.0238 |
| Test PR-AUC | **0.3962** | 0.3903 | **Phase 1 still wins** |
| Test F1 (tuned) | 0.3979 | 0.4000 | tied |
| Test MCC (tuned) | **0.3394** | 0.3385 | tied |

**This is the problem to solve.** On an imbalanced toxicity dataset, PR-AUC matters at least as much as ROC-AUC, and right now Phase 2 does not beat Phase 1 on it — not even on validation (val PR-AUC 0.4564 vs 0.4661).

**Goal: get Phase 2 ahead of Phase 1 on PR-AUC and MCC as well, without weakening ROC-AUC.** Try the experiments below in this order and stop when the validation numbers stop improving.

## A1. Average calibrated probabilities instead of ranks (try this first)

Rank-averaging throws away the actual probability values, which is the most likely reason PR-AUC dropped — PR-AUC cares about how confident the top predictions are, and ranks flatten that.

Instead:
- For each model and each endpoint, calibrate the predicted probabilities using the **validation** split (`sklearn.calibration.CalibratedClassifierCV` with `method="isotonic"`, or a simple logistic/Platt fit on the validation predictions — whichever is simpler to read).
- Then average the calibrated probabilities.
- Compare against the current rank-average on validation.

Calibration is just "rescaling scores so that 0.7 really means 70%". Explain it in one or two simple sentences in the results file.

## A2. Choose the blend weight on PR-AUC, not only ROC-AUC

The current weight (RF × 3, each DL run × 1) was selected on validation ROC-AUC. Redo the weight search selecting on **validation PR-AUC**, and also on the **average of validation ROC-AUC and PR-AUC**. Report all three choices and which you recommend.

## A3. Add more members to the ensemble

More diverse members usually lift PR-AUC. All of these already exist:
- `prob_lr` and `prob_dt` — the Phase 1 Logistic Regression and Decision Tree columns already saved in `results/test_predictions.csv`. You will need their validation counterparts; generate those by **reading Phase 1's code, not editing it**.
- The best hybrid (`corrected_emb_extratrees`). Its predictions correlate only about 0.5 with the Random Forest, so it may add value in a blend even though it is weaker alone.

Select every addition on validation. Drop any member that does not help.

## A4. Per-endpoint model selection

Instead of one blend for all 12 endpoints, pick the best model or blend **per endpoint** using validation only. The weak endpoints (NR-ER ≈ 0.63, SR-p53, SR-ATAD5, SR-HSE) may prefer a different member than the strong ones.

Be careful: choosing among many options on only ~780 validation molecules can overfit the validation set. Keep the candidate list short (4–6 options), and say in the report that this risk exists.

## A5. Out-of-fold embeddings for the hybrid (only if time allows)

Still not implemented. Train the network 5 times on 4/5 of the training set, take each held-out fold's embeddings from a network that never saw it, then fit ExtraTrees on those. About 5 minutes of training.

Do this only if A1–A4 have not already got Phase 2 ahead. If you do it, explain in simple English how you guaranteed no fold's embeddings came from a network that saw that fold's labels.

## A6. Decide the fate of the unfinished Transformer experiment

`phase2/experiments/` contains `ann_transformer.py`, `train_ann_transformer.py`, `smiles_tokenizer.py` (531 lines total), 16 MB of checkpoints for `ann_only` seeds 7/21/42 and `smiles_only` seeds 7/21, an **empty** log file, and no results.

Tell me which you recommend, then wait for my answer:
- **(a) Finish it:** train the missing seed, evaluate it, and test whether it helps the ensemble on validation. Keep it only if it does.
- **(b) Archive it:** move the scripts and checkpoints to `archive/` and note in the report that a SMILES Transformer was attempted but not completed.

My preference is (b) unless you think it can realistically improve PR-AUC, because 531 lines of tokenizer and Transformer code is hard for my team to explain in a viva.

## A7. Report Honestly

- Report ROC-AUC, PR-AUC, F1, Precision, Recall and MCC for every candidate, on validation and on test.
- Report mean ± standard deviation across seeds where seeds apply.
- Treat any difference smaller than 0.02 ROC-AUC as noise, since that is the measured seed spread.
- **If Phase 2 still does not beat Phase 1 on PR-AUC after these experiments, say so plainly.** Do not reframe it, and do not pick the test winner over the validation winner to make it look better. An honest tie is an acceptable result.
- Write the outcome into `phase2/experiments/PART_A_RESULTS.md` (update it, do not create a second results file).

---

## STOP HERE

Show me:
1. The final comparison table — Phase 1 versus the best Phase 2 model on all six metrics.
2. Which model you recommend as final, and why, based on validation.
3. Your answer on the Transformer experiment (A6).
4. Your proposed folder plan for Part B, listing exactly what you will rename or move and what reads from each item.

**Wait for my approval before changing any folder or file name.**

---

# PART B — Simplify the Phase 2 Folders

Only after I approve. Everything here happens inside `phase2/`.

## B1. Use Function Names Instead of Member Names

Right now `phase2/` looks like five separate student projects. Rename with `git mv` so history is kept:

```
phase2/member1_data_pipeline      ->  phase2/data/
phase2/member2_deep_learning_model ->  phase2/deep_learning/
phase2/member3_hybrid_model        ->  phase2/hybrid/
phase2/member4_evaluation          ->  phase2/evaluation/
phase2/experiments                 ->  phase2/experiments/   (keep as is)
```

After each rename, grep the whole repo for the old path and update every import and string. There are roughly 27 such references. Then run every script and every test to confirm nothing broke.

## B2. One Place for Results

Collect the scattered `reports/`, `results/` and `plots/` folders inside `phase2/` into:

```
phase2/results/
├── metrics/      (all result CSVs)
└── plots/        (all PNGs)
```

Keep **`results/` at the repo root untouched** — that belongs to Phase 1.

## B3. Deal With the 236 MB of Checkpoints

`phase2/member4_evaluation/checkpoints/single_task/` alone is **168 MB** (36 model files from the single-task ablation), and the repo total is over 280 MB. That is too heavy for a college submission.

- The single-task ablation's **conclusion is already saved** in its results CSVs, so the 36 checkpoints are not needed to reproduce the finding.
- Move them to `archive/` and add `archive/` to `.gitignore`, or keep them out of git with a note in the README saying how to regenerate them.
- **Do not delete them from my disk.** Move only.
- Keep the checkpoints that are actually loaded by current code: `deep_learning/checkpoints/best_multitask_model.pt` and the six `class_weight` models used by the ensemble.

## B4. Keep Only What Is Needed

- Merge the four per-member READMEs into **one** `README.md` at the repo root.
- **Before removing any member README, move its technical content into `docs/`.** `phase2/member3_hybrid_model/README.md` holds the hybrid's negative-result analysis and leakage discussion, which I need for my viva. Delete boilerplate only, never analysis.
- Merge the four `phase2/*/requirements.txt` files into one `phase2/requirements.txt`.
- Do not delete anything. Move unwanted files to `archive/` with `git mv`, and ask me before any real delete — my environment blocks deletes by default, and a blocked delete can leave `.git` lock files that break later git commands.
- Do not create empty folders, duplicate scripts, extra documentation, or a second copy of any prediction file.

## B5. Target Structure

```
molecular-toxicity-prediction/
├── phase1 files at root (untouched: train_models.py, evaluate.py, app.py, results/, tox21_*/)
│
├── phase2/
│   ├── data/            # loader and split handling
│   ├── deep_learning/   # the MLP: model, trainer, embeddings
│   ├── hybrid/          # feature-based hybrid model
│   ├── evaluation/      # metrics, ablations, error analysis
│   ├── experiments/     # final improvement + ensemble scripts
│   ├── results/
│   │   ├── metrics/
│   │   └── plots/
│   └── requirements.txt
│
├── docs/                # detailed analyses kept from member READMEs
├── archive/             # heavy checkpoints and abandoned experiments (gitignored)
├── README.md            # one README for the whole project
└── .gitignore
```

## B6. One Final README

In simple English: project introduction, problem statement, dataset, Phase 1 work, Phase 2 work, models used, hybrid model explanation, architecture diagram, **Phase 1 vs Phase 2 comparison table**, install and run instructions, folder structure, and team contributions in a single section.

**Note:** Phase 1 and Phase 2 use different member numbering — in `app.py`, Member 3 is baseline classifiers, while in `phase2/`, Member 3 is the hybrid model. Ask me who did what before writing the contributions section. Do not guess from folder names.

---

## FINAL CHECKS

Before telling me it is done:

- every import works
- every script under `phase2/` runs
- every test under `phase2/*/tests/` passes
- saved models still load
- `streamlit run app.py` works and every tab loads
- the README's install and run steps work when followed from scratch
- `git status` is clean and the work is committed
- no Phase 1 file has been modified (`git diff` against the Phase 1 paths shows nothing)

Run a small test before any expensive experiment. Tell me the expected runtime before anything that retrains a model.

**Do not claim Phase 2 improved on a metric unless the validation and test numbers support it.**
