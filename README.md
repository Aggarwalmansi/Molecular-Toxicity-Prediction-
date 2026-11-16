# Molecular Toxicity Prediction — Team T028, Section D

An end-to-end Tox21 assay activity prediction project: chemical validation, molecular features, ML, residual multitask DL, feature fusion, probability blending and a unified Streamlit application.

## Current results

Same frozen scaffold test partition (782 molecules); each endpoint uses observed labels only. Values are macro averages over 12 assays. “PR-AUC” in source CSVs means average precision.

| Model | ROC-AUC | Average precision |
|---|---:|---:|
| Phase 1 RF | 0.7541 | 0.3962 |
| Residual multitask DL | 0.7266 | 0.3298 |
| Feature-fusion RF | 0.7156 | 0.3284 |
| Stronger RF | 0.7674 | 0.4063 |
| Probability blend | 0.7685 | 0.3983 |

**Probability blending has the highest ROC-AUC (0.7685); stronger RF has the highest average precision (0.4063).** Stronger RF settings contribute +0.0133 ROC-AUC over Phase 1; blending adds +0.0011 and reduces AP from 0.4063 to 0.3983. Significance of that small increment is not established.

Sources: `phase2/member3_hybrid_model/reports/phase1_vs_phase2_summary.csv`, `raw_rf_test_evaluation.csv`, `ensemble_test_evaluation.csv`, and `phase2/member2_deep_learning_model/reports/test_evaluation.csv`.

## Dataset and features

7,831 raw records; 8 invalid structures excluded; 7,823 valid molecules. Frozen split: 6,259 train / 782 validation / 782 test. There are 16,012 missing assay entries (17.06%); they are masked, never interpreted as inactive. All 1,775 acyclic molecules are training-only, so holdout evaluation concerns ring-containing scaffolds.

Inputs are 2,048 radius-2 Morgan bits plus five training-standardized descriptors, ordered MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors. Scaling and all forest fitting use training data only.

## Current models

The residual network uses 2,053 → 512 → 256 → 128, LayerNorm, GELU and 12 heads (128 → 64 → 1). It has 2,532,492 trainable parameters. Input/trunk/bottleneck/head dropout: 0.30/0.50/0.25/0.20. AdamW: learning rate 0.0003, weight decay 0.001; batch size 64; maximum 150 epochs; early stopping patience 20 on validation macro ROC-AUC. Cosine warm restarts: T_0=20, T_mult=2, eta_min=1e-6; gradient clipping norm 2.0. Positive weights are min(sqrt(N_negative/N_positive), 3.0), using training counts only.

Feature fusion joins all 2,053 original features with 128 neural embeddings (2,181 total). Stronger raw-feature and fused forests use 200 trees, max_depth=15, min_samples_leaf=2, balanced weights and seed 42. Probability blending uses **0.80 × P_RF + 0.20 × P_DL**, with raw-feature arm and alpha chosen on validation data. The five descriptors are already in the original block; they are not added twice.

Best DL checkpoint: epoch 3; validation ROC-AUC 0.7917; test ROC-AUC 0.7266 / AP 0.3298 / recall 0.2984 at threshold 0.5. Trunk regularization delays the best checkpoint but does not eliminate overfitting. Continuous embedding split-threshold effects are a possible explanation for weaker fusion, not an established cause.

## Application and reproducibility

[Open the deployed application](https://molecular-toxicity-prediction.onrender.com/). To reproduce locally:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Rebuild local serving forests and metrics if required:
python -m phase2.member5_report_ppt.prepare_models
streamlit run app.py
```

Prediction serves four methods (48 scores): residual DL, feature-fusion RF, stronger raw-feature RF and the 80/20 blend. The Phase 1 baseline is retained as a historical comparison, not served as the stronger RF.

## Earlier experiments

`phase2/member4_evaluation/` retains **earlier-network** ablations. Multitask ROC-AUC 0.7472 ± 0.0065 versus single-task 0.7020 ± 0.0076 is a separate three-seed study, not the residual model's performance. ± denotes seed standard deviation. Old 0.7179/0.6753 tables and 133-feature experiments are historical and must not replace the current table above.

## Project materials

- [Data pipeline](phase2/member1_data_pipeline/README.md)
- [Residual neural model](phase2/member2_deep_learning_model/README.md)
- [Fusion and blending experiments](phase2/member3_hybrid_model/README.md)
- [Earlier-network evaluation](phase2/member4_evaluation/README.md)
- [Setup and reproduction guide](phase2/member5_report_ppt/README.md)
- [Formal report](phase2/member5_report_ppt/report/main.tex)
- [Presentation](phase2/member5_report_ppt/presentation/Molecular_Toxicity_T028_refined.pptx)

## Team T028 — Section D

| Member | Name | Enrollment number | Contribution |
|---|---|---|---|
| 1 | Mansi Agarwal | 2401010264 | Data pipeline and feature engineering |
| 2 | Nachiket Amlekar | 2401020039 | Multitask deep learning |
| 3 | Tanisha Dhiman | 2401010475 | Hybrid models and probability blending |
| 4 | Aditi Singh | 2401020006 | Evaluation and ablation studies |
| 5 | Adamya Tiwari | 2401010025 | Application, report, presentation and reproduction guide |

Assay activity scores are uncalibrated and do not establish human safety.

Reporting convention: endpoint metrics are rounded to four decimals before macro averaging, matching the current evaluation scripts. Full-precision averaging may differ in the last displayed decimal.
