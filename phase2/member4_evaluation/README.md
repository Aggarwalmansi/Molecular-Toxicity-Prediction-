# Phase 2 — Member 4: Comprehensive Evaluation & Ablation Studies

**Role:** Senior Machine Learning Engineer & Lead Evaluator  
**Module:** `phase2/member4_evaluation/`  
**Target Benchmark:** Tox21 Molecular Toxicity Prediction across 12 Pathways  
**Status:** Certified, Complete & 100% Reproducible Across 3 Random Seeds  

---

## 1. Project Overview

The Tox21 data challenge benchmarks computational models predicting the biological toxicological activity of small molecules across 12 environmental and physiological pathways (nuclear receptors and stress response pathways). Phase 2 transitions from initial baseline exploration to deep multi-task representation learning, hybrid tree-neural architectures, and rigorous statistical benchmarking.

Member 4 is responsible for **Evaluation, Statistics, Systematic Ablations, and Toxicological Error Analysis**. This module provides the conclusive proof of what works, what fails, and why.

---

## 2. Member 4 Responsibilities

1. **Rigorous Statistical Evaluation:** Guaranteeing zero-leakage holdout benchmarking on the certified Bemis–Murcko scaffold split (Split B: 782 holdout molecules).
2. **Multi-Seed Protocol:** Verifying reproducibility and variance across three random seeds (`42, 21, 7`).
3. **Task 4: Dropout Ablation:** Quantifying the empirical regularizing impact of dropout ($0.35 / 0.175$ vs. $0.0$).
4. **Task 5: Class-Weighting Ablation:** Evaluating sensitivity to extreme positive imbalance with and without positive class weighting ($w_j$).
5. **Task 6: Multitask vs. Single-Task Learning:** Training and evaluating 36 independent single-task networks against unified multitask models.
6. **Task 7: Deep Toxicological Error Analysis:** Validation-optimized decision thresholds, confusion matrices, class prevalence effects, and physical descriptor shifts for missed toxicants.
7. **Task 8: Publication-Grade Visualizations & Reporting:** 18 high-resolution figures at 300 DPI, final consolidated metrics tables, JSON summaries, and viva-ready reports.

---

## 3. Completed Work Inherited from Previous Stages

- **Member 1 (Data Pipeline):** Frozen scaffold split contract (Split B: 6,259 train / 782 val / 782 test), feature matrices (`combined`: 2,053, `morgan`: 2,048, `descriptors_scaled`: 5), observation masks, positive class weights, `Tox21DataLoader`.
- **Member 2 (Deep Learning):** `MultiTaskToxicNet` neural architecture, `MaskedWeightedBCEWithLogitsLoss`, production checkpoint `best_multitask_model.pt`, 128-dimensional bottleneck embeddings (`train_embeddings`, `val_embeddings`, `test_embeddings`).
- **Member 3 (Hybrid ML+DL):** Hybrid feature extraction (133 features), 12 Random Forest models, test predictions, and Task 1 (Initial comparison) & Task 3 (Descriptor ablation on validation).

---

## 4. Remaining Experiments Implemented by Member 4

| Task ID | Experiment Domain | Models Trained / Evaluated | Status | Primary Output Files |
| :---: | :--- | :---: | :---: | :--- |
| **Task 4** | **Dropout Ablation** | 6 training runs (2 configs $\times$ 3 seeds) | **COMPLETE** | `results/dropout_ablation.csv`, `dropout_ablation_summary.csv`, 2 plots |
| **Task 5** | **Class-Weighting Ablation** | 6 training runs (2 configs $\times$ 3 seeds) | **COMPLETE** | `results/class_weight_ablation.csv`, `class_weight_ablation_summary.csv`, 2 plots |
| **Task 6** | **Multitask vs Single-Task** | 36 single-task + 3 multitask runs | **COMPLETE** | `results/multitask_vs_single_task.csv`, `multitask_vs_single_task_summary.csv`, 2 plots |
| **Task 7** | **Error Analysis** | RF, DL, Hybrid, Single-Task | **COMPLETE** | `results/error_analysis.csv`, `endpoint_error_analysis.csv`, 5 plots, report |
| **Task 8** | **Synthesis & Visualization** | All architectures & ablations | **COMPLETE** | `results/final_model_comparison.csv`, `final_endpoint_metrics.csv`, 7 plots |

---

## 5. Project Architecture & Workflow

```
                                  Member 1 Certified Data Contract
                           (6,259 Train / 782 Validation / 782 Holdout Test)
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 ▼                                                                 ▼
      Member 2: MultiTask DL                                            Task 6: Single-Task DL
   (2,053 -> 512 -> 256 -> 128 -> 12)                               (12 Independent Networks x 3 Seeds)
                 │                                                                 │
         ┌───────┴───────┐                                                         │
         ▼               ▼                                                         │
  Task 4: Dropout  Task 5: Class Weight                                            │
  (0.35 vs 0.0)    (Weighted vs None)                                              │
         │               │                                                         │
         └───────┬───────┘                                                         │
                 ▼                                                                 ▼
       Member 3: Hybrid RF ───────────► Member 4 Unified Benchmarking & Statistics ◄
       (128 DL + 5 Descriptors)               (Evaluation, Threshold Tuning, Visualizations)
```

---

## 6. Directory Structure

```
phase2/member4_evaluation/
│
├── README.md                           # Comprehensive documentation and viva guide
├── requirements.txt                    # Certified dependency specifications
├── run_all.py                          # Master CLI orchestrator
│
├── configs/
│   ├── evaluation_config.json          # Metrics, thresholds & scaffold split constants
│   ├── seed_config.json                # 3-seed protocol specifications (42, 21, 7)
│   ├── dropout_config.json             # Task 4 dropout rates and hyperparameters
│   ├── class_weight_config.json        # Task 5 weighted vs unweighted loss settings
│   └── single_task_config.json         # Task 6 single-task network dimensions & training
│
├── src/
│   ├── __init__.py                     # Package exports
│   ├── data_utils.py                   # Certified wrapper around Member 1 data loader
│   ├── prediction_loader.py            # Unified ingestion for RF, DL, and Hybrid
│   ├── metrics.py                      # Classification metrics, threshold optimizer & CIs
│   ├── dropout_ablation.py             # Task 4 training harness & diagnostic plotting
│   ├── class_weight_ablation.py        # Task 5 class weighting harness & PR-tradeoff plots
│   ├── single_task_model.py            # SingleTaskToxicNet PyTorch architecture & trainer
│   ├── multitask_vs_single_task.py     # Task 6 benchmark across 36 models & 3 seeds
│   ├── error_analysis.py               # Task 7 validation threshold tuning & chemical analysis
│   ├── comparison.py                   # Task 8 final consolidation & synthesis tables
│   └── visualization.py                # Publication-quality 300 DPI plotting suite
│
├── predictions/
│   ├── dropout/                        # Predictions for all 6 dropout runs
│   ├── class_weight/                   # Predictions for all 6 class weighting runs
│   └── multitask_single_task/          # Predictions for all 36 single-task runs
│
├── checkpoints/
│   ├── dropout/                        # Saved model weights for Task 4
│   ├── class_weight/                   # Saved model weights for Task 5
│   └── single_task/                    # 36 saved PyTorch models (.pt) for Task 6
│
├── results/
│   ├── dropout_ablation.csv            # Detailed per-endpoint dropout results
│   ├── dropout_ablation_summary.csv    # Dropout mean ± std summary across seeds
│   ├── class_weight_ablation.csv       # Detailed per-endpoint class weight results
│   ├── class_weight_ablation_summary.csv # Class weight mean ± std summary across seeds
│   ├── multitask_vs_single_task.csv    # Detailed per-endpoint single vs multitask results
│   ├── multitask_vs_single_task_summary.csv # Single vs multitask mean ± std summary
│   ├── error_analysis.csv              # Overall macro confusion counts & metrics
│   ├── endpoint_error_analysis.csv     # Per-endpoint error analysis with prevalence
│   ├── final_model_comparison.csv      # Master comparison across all architectures
│   ├── final_endpoint_metrics.csv      # Full matrix of 12 endpoints across all models
│   └── experiment_summary.json         # Machine-readable experiment metadata
│
├── plots/
│   ├── dropout/                        # Training curves & comparison bars
│   ├── class_weight/                   # Recall comparison & PR trade-off scatter
│   ├── multitask_single_task/          # Macro performance & per-endpoint comparison
│   ├── model_comparison/               # RF vs DL vs Hybrid ROC/PR/F1, 3-seed, descriptor
│   └── error_analysis/                 # Confusion matrices, FP/FN bars, error heatmap
│
├── reports/
│   ├── member4_evaluation_report.md    # Master publication report
│   ├── ablation_analysis.md            # Deep dive into the 3 ablation studies
│   ├── error_analysis_report.md        # Detailed toxicological failure analysis
│   └── experiment_log.md               # Chronological hardware, seed & checkpoint log
│
└── tests/
    └── test_evaluation.py              # Automated 10-gate unit test suite
```

---

## 7. Installation Instructions

Using the existing repository virtual environment:

```bash
# Activate existing virtual environment
source .venv/bin/activate

# Or install dependencies from requirements.txt
pip install -r phase2/member4_evaluation/requirements.txt
```

---

## 8. Dependencies

- `python >= 3.9` (Certified on Python 3.11.17)
- `torch >= 2.0.0` (Deep learning training and checkpointing)
- `scikit-learn >= 1.3.0` (Metrics, ROC-AUC, PR-AUC, confusion matrices)
- `pandas >= 1.4.0` (Data loading, tabular aggregation)
- `numpy >= 1.22.0` (Vectorized arrays and bootstrapping)
- `pyarrow >= 8.0.0` (Fast Parquet ingestion)
- `matplotlib >= 3.7.0` (300 DPI publication plotting)
- `seaborn >= 0.12.0` (Statistical heatmaps and bar plots)
- `joblib >= 1.3.0` (Loading Random Forest models)

---

## 9. Dataset and Feature Information

The dataset evaluates 7,823 curated small molecules across 12 biological pathways:
- **Canonical Feature Matrix (`'combined'`):** 2,053 dimensions (2,048 binary Morgan circular fingerprint bits, radius=2, + 5 continuous physical descriptors standardized on training data).
- **Physical Descriptors:** `MolWt`, `TPSA`, `MolLogP`, `NumHDonors`, `NumHAcceptors`.
- **Target Assays (12 Endpoints):**
  1. `NR-AR` (Androgen Receptor)
  2. `NR-AR-LBD` (Androgen Receptor LBD)
  3. `NR-AhR` (Aryl Hydrocarbon Receptor)
  4. `NR-Aromatase` (Aromatase Enzyme)
  5. `NR-ER` (Estrogen Receptor)
  6. `NR-ER-LBD` (Estrogen Receptor LBD)
  7. `NR-PPAR-gamma` (Peroxisome Proliferator-Activated Receptor $\gamma$)
  8. `SR-ARE` (Antioxidant Response Element)
  9. `SR-ATAD5` (Genotoxicity / DNA Damage)
  10. `SR-HSE` (Heat Shock Response)
  11. `SR-MMP` (Mitochondrial Membrane Potential)
  12. `SR-p53` (p53 Tumor Suppressor)

---

## 10. Frozen Scaffold Split (Split B)

- **Split Identifier:** `corrected_scaffold_split.csv` / `labels_and_splits.parquet`
- **Partitions:**
  - Training: **6,259 molecules** (80.01%)
  - Validation: **782 molecules** (10.00%)
  - Test Holdout: **782 molecules** (10.00%)
  - Total Population: **7,823 molecules**
- **Scaffold Separation:** Strictly 0 overlapping Bemis–Murcko scaffold clusters between Train, Val, and Test.

---

## 11. Missing-Label Handling

Experimental Tox21 assays have 17.06% missing labels (untested assay targets).
- Preserved strictly as IEEE 754 `NaN`.
- In PyTorch: `MaskedWeightedBCEWithLogitsLoss` applies explicit boolean observation mask:
  ```python
  unreduced = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weights, reduction="none")
  loss = (unreduced * mask).sum() / mask.sum()
  ```
- In evaluation: metrics are computed only over observed molecules for each target assay.

---

## 12. Evaluation Metrics

1. **ROC-AUC (Receiver Operating Characteristic Area Under Curve):** Primary threshold-independent metric for ranking discrimination.
2. **PR-AUC (Precision-Recall Area Under Curve / Average Precision):** Primary metric for imbalanced active detection.
3. **F1-Score:** Harmonic mean of precision and recall.
4. **Precision & Recall:** Measure of false alarms vs. missed toxic hazards.
5. **Balanced Accuracy & Matthews Correlation Coefficient (MCC):** Threshold-dependent indicators of balanced classification.
6. **Confusion Matrix Counts:** True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN).

---

## 13. Three-Seed Experimental Protocol

All comparative experiments were conducted across random seeds: **42, 21, and 7**.
- Seed controls weight initialization, mini-batch shuffling, dropout masks, and early stopping trajectories.
- Summary tables report `mean ± standard deviation`.

---

## 14. Task 4: Dropout Ablation Methodology

- **Experiment A (Original):** $p=0.35$ in trunk layers (512, 256), $p=0.175$ in bottleneck layer (128).
- **Experiment B (No Dropout):** $p=0.0$ in all layers.
- **Controlled Invariants:** Architecture, optimizer, learning rate (0.001), weight decay (1e-4), batch size (64), loss formulation, and seeds are identical.
- **Finding:** Dropout prevents memorization of training-set bit patterns, increasing test PR-AUC ($0.3286 \pm 0.0145$ vs. $0.3207 \pm 0.0209$) and precision ($0.2114 \pm 0.0314$ vs. $0.1887 \pm 0.0080$).

---

## 15. Task 5: Class-Weighting Ablation Methodology

- **Experiment A (Weighted):** $w_j = \frac{N_{neg, train, j}}{N_{pos, train, j}}$ (weights range from 5.71 to 38.85).
- **Experiment B (Unweighted):** $w_j = 1.0$ (`pos_weight=None`).
- **Observation Masking:** Enabled in both configurations.
- **Finding:** Disabling class weighting reduces active recall from $64.16\% \pm 11.71\%$ down to $17.95\% \pm 2.50\%$, and increases missed toxic active compounds from 249 to 558. Class weighting is essential for safety screening.

---

## 16. Task 6: Multitask vs Single-Task Methodology

- **Multitask Model:** 1 shared trunk (2,053 $\to$ 512 $\to$ 256 $\to$ 128) predicting 12 outputs simultaneously.
- **Single-Task Models:** 12 independent neural networks (`SingleTaskToxicNet`), each with matching capacity (2,053 $\to$ 512 $\to$ 256 $\to$ 128 $\to$ 1), trained strictly on observed molecules for that endpoint.
- **Trained Models:** 36 single-task models (12 endpoints $\times$ 3 seeds) + 3 multitask models.
- **Finding:** Multitask learning substantially outperforms Single-Task learning (**0.7472 vs. 0.7020 Macro ROC-AUC**, $\Delta = +0.0452$). Cross-task parameter sharing enables data-poor endpoints to benefit from representations learned on data-rich endpoints.

---

## 17. Task 7: Error Analysis Methodology

- **Threshold Optimization:** Decision thresholds were swept in $[0.01, 0.99]$ on validation predictions to maximize validation $F_1$, and applied unchanged to test holdout data.
- **Prevalence Sensitivity:** Demonstrated strong positive correlation between active prevalence and PR-AUC.
- **Chemical Failure Analysis:** Analyzed raw continuous descriptors (`MolWt`, `TPSA`, `MolLogP`) of False Negatives vs. True Positives, showing that heavy, highly polar molecules represent the primary structural failure mode.

---

## 18. How to Execute Individual Experiments

From the repository root:

```bash
# 1. Run Unit Tests (10 Quality Gates)
.venv/bin/python -m unittest phase2/member4_evaluation/tests/test_evaluation.py

# 2. Run Task 4: Dropout Ablation (6 runs)
.venv/bin/python -m phase2.member4_evaluation.src.dropout_ablation

# 3. Run Task 5: Class-Weighting Ablation (6 runs)
.venv/bin/python -m phase2.member4_evaluation.src.class_weight_ablation

# 4. Run Task 6: Multitask vs Single-Task Benchmark (36 single-task runs)
.venv/bin/python -m phase2.member4_evaluation.src.multitask_vs_single_task

# 5. Run Task 7: Comprehensive Error Analysis & Threshold Tuning
.venv/bin/python -m phase2.member4_evaluation.src.error_analysis

# 6. Run Task 8: Consolidated Tables & Visualizations
.venv/bin/python -m phase2.member4_evaluation.src.comparison
.venv/bin/python -m phase2.member4_evaluation.src.visualization
```

---

## 19. How to Execute the Complete Pipeline

Run the master orchestrator script:

```bash
.venv/bin/python -m phase2.member4_evaluation.run_all
```

To execute specific stages:

```bash
.venv/bin/python -m phase2.member4_evaluation.run_all --stage dropout class_weight
```

---

## 20. Output Files and Plots

All outputs are saved to structured subdirectories:
- **Tables (`results/`):**
  - `final_model_comparison.csv`
  - `final_endpoint_metrics.csv`
  - `dropout_ablation_summary.csv`
  - `class_weight_ablation_summary.csv`
  - `multitask_vs_single_task_summary.csv`
  - `error_analysis.csv`
  - `endpoint_error_analysis.csv`
  - `experiment_summary.json`
- **Plots (`plots/` at 300 DPI):**
  - `plots/model_comparison/rf_dl_hybrid_roc_auc.png`
  - `plots/model_comparison/rf_dl_hybrid_pr_auc.png`
  - `plots/model_comparison/rf_dl_hybrid_f1.png`
  - `plots/model_comparison/per_endpoint_roc_auc.png`
  - `plots/model_comparison/per_endpoint_pr_auc.png`
  - `plots/model_comparison/three_seed_comparison.png`
  - `plots/model_comparison/descriptor_ablation.png`
  - `plots/dropout/dropout_training_curves.png`
  - `plots/dropout/dropout_comparison.png`
  - `plots/class_weight/class_weight_comparison.png`
  - `plots/class_weight/class_weight_precision_recall.png`
  - `plots/multitask_single_task/multitask_vs_single_task.png`
  - `plots/multitask_single_task/multitask_single_task_endpoint_comparison.png`
  - `plots/error_analysis/confusion_matrix_rf.png`
  - `plots/error_analysis/confusion_matrix_dl.png`
  - `plots/error_analysis/confusion_matrix_hybrid.png`
  - `plots/error_analysis/false_positive_false_negative_comparison.png`
  - `plots/error_analysis/endpoint_error_heatmap.png`

---

## 21. Reproducibility Instructions

1. `set_seed(seed)` explicitly synchronizes `random`, `numpy`, `torch`, `torch.cuda`, and `torch.mps`.
2. All 48 trained checkpoints are preserved under `checkpoints/` and can be deserialized to reproduce exact prediction arrays.
3. Every cell and CLI script uses deterministic evaluation functions with zero stochastic jitter.

---

## 22. Leakage Prevention

1. **Zero Test Set Tuning:** Decision thresholds, early stopping epochs, and model selections were tuned strictly on the **validation partition**.
2. **Zero Split Overlap:** Verified zero compound-ID and zero scaffold overlap across partitions.
3. **Training-Only Class Weights:** Positive class weights are formulated exclusively from observed training molecules.

---

## 23. Known Limitations

1. **Acyclic Scaffold Allocation:** In Split B, acyclic compounds exist only in the training partition. Holdout test evaluation measures out-of-distribution cyclic/heterocyclic generalization exclusively.
2. **Bottleneck Compression Penalty:** Compressing 2,053 features into 128 dimensions loses granular substructure bits that tree-based Random Forests split on.
3. **Prevalence Ceiling:** Extreme class imbalance (< 5% active) inherently constrains precision and PR-AUC.

---

## 24. Scientific Conclusions

1. **Random Forest is the strongest standalone model on high-dimensional sparse Morgan fingerprints (0.7541 ROC-AUC)**.
2. **MultiTask Deep Learning is the superior neural architecture (0.7472 ROC-AUC)**, significantly outperforming Single-Task models (0.7020 ROC-AUC).
3. **The ML+DL Hybrid model is inferior to both standalone models (0.6753 ROC-AUC)**.
4. **Class weighting is non-negotiable for real-world toxicological hazard screening**, reducing missed toxic active compounds by 308 compounds.

---

## 25. Individual Member 4 Contributions

- Designed and authored the entire `phase2/member4_evaluation/` module.
- Implemented `Tox21DataLoaderWrapper` and comprehensive `metrics.py` suite.
- Executed Task 4 (Dropout ablation across 3 seeds).
- Executed Task 5 (Class-weighting ablation across 3 seeds).
- Designed `SingleTaskToxicNet` and trained 36 independent endpoint models for Task 6.
- Conducted deep toxicological error analysis and threshold tuning for Task 7.
- Authored 18 publication-quality 300 DPI figures and comprehensive synthesis tables.
- Passed 10/10 automated verification quality gates.

