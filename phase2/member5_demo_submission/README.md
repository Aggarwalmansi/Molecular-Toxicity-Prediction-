# Inference demo, setup and reproduction guide

This package connects the team's saved models to new SMILES inputs. It does not retrain the DL or hybrid models. All paths resolve relative to the repository, so commands work independently of a contributor's home directory.

## Run locally

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Only needed when the standalone RF files are absent:
python -m phase2.member5_demo_submission.prepare_models
streamlit run app.py
```

Open **Live Demo & Molecule Inspector → Live SMILES Predictor**, select a preset or enter a SMILES string, then click **Predict all 12 endpoints**. The table and chart compare RF, multitask DL and hybrid activity scores. Download CSV includes endpoint, model, threshold, class and canonical SMILES. There is no training inside the dashboard. Member 5 uses separately rebuilt hybrid forests matched to the frozen checkpoint, because the original exported embeddings fail checkpoint parity. The original Member 3 forests are preserved.

Command-line prediction:

```bash
python -m phase2.member5_demo_submission.predict --smiles 'CCO' --output predictions.csv
python -m unittest discover -s phase2/member5_demo_submission/tests -v
```

## Inference contract

1. Reject empty/invalid SMILES; canonicalize valid molecules with RDKit.
2. Compute 2,048 Morgan bits (radius 2) followed by MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors.
3. Scale the five descriptors with the committed training-only scaler; never fit on user input.
4. Load Member 2's checkpoint on CPU in evaluation mode. Disable gradient tracking and dropout.
5. Feed its 128-dimensional bottleneck plus the five scaled descriptors to Member 3's endpoint forests.
6. Feed the original 2,053 features to the exported standalone RFs.
7. Use 0.5 as the explicitly stated default threshold. Activity scores are not calibrated human toxicity probabilities. Inactive predictions do not establish safety.

Only trusted repository checkpoint/joblib files should be loaded. Missing optional standalone RFs are reported, not substituted. Missing or incompatible required artifacts produce an error.

## Benchmark provenance

The app reads `phase2/member4_evaluation/results/final_model_comparison.csv` rather than introducing another hardcoded Phase 2 table. Member 2's historical evaluation reports macro ROC-AUC **0.7179**, average precision **0.3376**. Member 4's separate three-seed multitask experiment reports ROC-AUC **0.7472 ± 0.0065**. These must not be presented as the same checkpoint result. Hybrid ROC-AUC is **0.6753**; the experiment does not demonstrate an improvement over RF or DL. Bottleneck information loss is a possible explanation, not a proven causal mechanism.

## Submission files

- `report/main.tex`: editable IEEE-format report with methods, measured results, limitations, contributions and references.
- `presentation/`: editable PowerPoint slide deck.
- `CONTRIBUTIONS.csv`: ownership record; replace member labels with actual names and add commit references before submitting.

No student names, roll numbers or individual contribution evidence were supplied. Fill those details and have each member confirm their section. Verify the course's problem-specific requirements against the original project PDF before final submission; only the general rubric was supplied here.

## Integration correction

The original Member 2 exported embeddings do not match its saved checkpoint. The trainer restored a mutable in-memory state dictionary instead of frozen weights. The trainer now reloads the on-disk checkpoint for future exports. `prepare_models` creates a separate checkpoint-consistent hybrid under Member 5, trained only on the frozen training partition. Validation/test metrics for this bundle are in `models/inference_metrics.csv`; these are distinct from the historical Member 3 result. Bundle metadata records checkpoint SHA-256 and library versions. Historical hybrid joblib files were serialized with a different scikit-learn version, another reason to keep the demo refit separate.

Verified demo test metrics (ROC-AUC / average precision): RF 0.7498 / 0.3959; DL 0.7318 / 0.3419; rebuilt hybrid 0.7182 / 0.3335. Differences from historical RF numbers may include library-version effects; use the measured bundle table when presenting live predictions.
