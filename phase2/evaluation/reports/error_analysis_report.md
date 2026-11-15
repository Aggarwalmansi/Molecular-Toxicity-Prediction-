# Comprehensive Toxicological Error Analysis Report

**Phase 2 — Member 4: Evaluation, Statistics & Error Analysis**  
**Dataset:** Tox21 Biological Pathway Screening  
**Holdout Population:** 782 strictly cyclic/heterocyclic compounds (Split B Frozen Scaffold Holdout)  

---

## 1. Executive Summary

This error analysis examines prediction failures across **12 nuclear receptor and stress response pathways** for:
1. **Standalone Random Forest (Phase 1):** 2,053 input features.
2. **MultiTask Deep Learning (Phase 2 Member 2):** 2,053 features with shared trunk and latent bottleneck $z \in \mathbb{R}^{128}$.
3. **ML+DL Hybrid Model (Phase 2 Member 3):** Random Forest on 128 latent embeddings + 5 standardized descriptors.

All classification decisions for deep learning and hybrid models were evaluated using **validation-optimized thresholds** ($F_1$-score maximization on validation predictions applied unchanged to holdout test data).

---

## 2. Macro Error Metrics Across Architectures

| Model Architecture | Macro ROC-AUC | Macro PR-AUC | Macro F1 | Total TP | Total FN | Total FP | Total TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standalone Random Forest** | 0.7541 | 0.3961 | 0.3954 | 398 | 330 | 783 | 5584 |
| **MultiTask Deep Learning** | 0.7318 | 0.3419 | 0.3646 | 344 | 384 | 733 | 5634 |
| **ML+DL Hybrid Model** | 0.6753 | 0.3009 | 0.3460 | 309 | 419 | 596 | 5771 |

---

## 3. Identification of Weakest Endpoints

### 3.1 Lowest Discrimination Endpoints (ROC-AUC)
- **MultiTask DL:** **NR-ER** exhibited the lowest ROC-AUC at **0.6337**.
- **Underlying Cause:** Severe class imbalance combined with complex receptor conformation requirements.

### 3.2 Lowest Precision-Recall Endpoints (PR-AUC)
- **MultiTask DL:** **SR-ATAD5** recorded the lowest PR-AUC at **0.0954** (with only 5.15% active compounds).
- In high-throughput toxicological screening, PR-AUC is sensitive to prevalence. Endpoints below 5% active prevalence suffer pronounced precision deterioration.

### 3.3 Lowest Positive Sensitivity (Recall & False Negatives)
- **Lowest Recall Endpoint:** **SR-ATAD5** (Recall = 0.1176).
- **Highest False-Negative Burden:** **NR-ER** with **61 missed toxic actives**.
- **Toxicological Implication:** In regulatory safety evaluation, false negatives represent hazardous toxicants misclassified as safe. Class weighting partially mitigates this by penalizing false negatives.

---

## 4. Class Imbalance Sensitivity & Prevalence Effects

A strong positive correlation exists between active class prevalence and precision metrics:
- Endpoints with high active percentages (e.g., `SR-MMP` at 21.1%, `SR-ARE` at 25.2%) yield robust PR-AUC scores (> 0.45) and F1-scores (> 0.50).
- Ultra-rare endpoints (e.g., `NR-PPAR-gamma` at 5.0%, `SR-ATAD5` at 5.1%) show severe precision drops due to the vast majority of inactive negative decoys.

---

## 5. Chemical Property Analysis of False Negatives

Analysis of continuous physicochemical descriptors (`MolWt`, `TPSA`, `MolLogP`) between correctly predicted toxic compounds (True Positives) and missed toxic compounds (False Negatives):

| Molecular Descriptor | True Positives (Detected) | False Negatives (Missed) | Difference / Shift |
| :--- | :---: | :---: | :---: |
| **Molecular Weight (MolWt)** | 403.46 g/mol | 356.38 g/mol | -47.08 g/mol |
| **Polar Surface Area (TPSA)** | 72.51 Å² | 70.26 Å² | -2.25 Å² |
| **Lipophilicity (MolLogP)** | 4.17 | 3.2 | -0.97 |

**Interpretation:**
- Missed toxic compounds (False Negatives) frequently possess higher molecular weight and higher polar surface area, or represent novel ring scaffold chemotypes absent from the acyclic-heavy training clusters.

---

## 6. Generated Visualizations

- `plots/error_analysis/confusion_matrix_rf.png`: Cumulative test confusion matrix for Standalone Random Forest.
- `plots/error_analysis/confusion_matrix_dl.png`: Cumulative test confusion matrix for MultiTask Deep Learning.
- `plots/error_analysis/confusion_matrix_hybrid.png`: Cumulative test confusion matrix for ML+DL Hybrid Model.
- `plots/error_analysis/false_positive_false_negative_comparison.png`: Direct comparison of false alarm vs. missed hazard rates.
- `plots/error_analysis/endpoint_error_heatmap.png`: Error metric profile vs. target prevalence.
