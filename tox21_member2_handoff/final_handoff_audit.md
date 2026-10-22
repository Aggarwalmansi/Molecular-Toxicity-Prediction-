# Final Handoff Audit Report: Tox21 Member 2 Pipeline

**Role**: Senior ML Engineer & Cheminformatics Specialist  
**Target Package**: [`tox21_member2_handoff/`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/)  
**Date**: October 2026  
**Status**: **CERTIFIED PASS (All Checks Passing, Zero Blockers)**  

---

## Executive Summary

Before committing the Member 2 handoff package to Member 3 (responsible for Phase 4 baseline modeling), an independent technical audit was conducted on all source code, Jupyter notebooks, manifests, and parquet artifacts. 

Two specific points of ambiguity were investigated and conclusively resolved:
1. **Canonical Feature Definition of `X_combined.parquet`**: Established that `X_combined.parquet` contains **2,048 binary Morgan bits plus 5 standardized descriptors** (fitted strictly on the training partition). To ensure maximum flexibility for Member 3 without ambiguity, both standardized and raw combined matrices are now provided as certified artifacts (`X_combined.parquet` and `X_combined_raw.parquet`), and the loader `Tox21Dataset` supports both keys with explicit documentation.
2. **Reconciliation of the Variance-Filtered Matrix (1,011 vs. 1,217)**: Established that the true empirical count of Morgan bits with training variance $\ge 0.005$ is **exactly 1,011 features**. The saved artifact `X_morgan_unsupervised_filtered.parquet` has shape `(7823, 1012)` (`mol_id` + 1,011 features). The number "1,217" was traced to a documentation typo originating from bit index `1217` in `fingerprint_statistics.csv`. All documentation has been harmonized to the verified count of 1,011.

---

## 1. Audit of Canonical Feature Definitions

### 1.1 Empirical Inspection of `X_combined.parquet`
Inspection of [`tox21_member2_handoff/artifacts/X_combined.parquet`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/artifacts/X_combined.parquet) ($N = 7,823$ rows, $2,054$ columns):
- **Columns**: `mol_id`, `morgan_0` through `morgan_2047`, followed by the 5 descriptor columns (`MolWt_scaled`, `TPSA_scaled`, `MolLogP_scaled`, `NumHDonors_scaled`, `NumHAcceptors_scaled`).
- **Statistical Distribution on Training Split ($N = 6,259$)**:
  - `MolWt_scaled`: Mean = $3.27 \times 10^{-16} \approx 0.0$, Std = $1.00008 \approx 1.0$, Min = $-1.67$, Max = $10.03$
  - `TPSA_scaled`: Mean = $-4.33 \times 10^{-15} \approx 0.0$, Std = $1.00008 \approx 1.0$, Min = $-1.08$, Max = $14.39$
  - `MolLogP_scaled`: Mean = $-4.13 \times 10^{-15} \approx 0.0$, Std = $1.00008 \approx 1.0$, Min = $-7.87$, Max = $9.15$
  - `NumHDonors_scaled`: Mean = $-7.83 \times 10^{-17} \approx 0.0$, Std = $1.00008 \approx 1.0$, Min = $-0.68$, Max = $14.57$
  - `NumHAcceptors_scaled`: Mean = $4.54 \times 10^{-17} \approx 0.0$, Std = $1.00008 \approx 1.0$, Min = $-1.14$, Max = $15.74$
- **Conclusion**: `X_combined.parquet` contains **2,048 binary Morgan features plus 5 standardized descriptors**.

### 1.2 Dual Artifact Provision & Loader Contract
Because certain downstream models (e.g., Logistic Regression, Support Vector Machines, Neural Networks) require standardized continuous variables, while others (e.g., tree-based ensembles such as Random Forest, XGBoost, LightGBM) benefit from natural physical units, both matrices are officially provided in `artifacts/`:
1. **`X_combined.parquet` (Canonical)**: 2,048 binary Morgan bits + 5 standardized descriptors (fitted on train).
2. **`X_combined_raw.parquet`**: 2,048 binary Morgan bits + 5 raw physicochemical descriptors (natural units).
3. **`X_descriptors_scaled.parquet`**: 5 standardized descriptors.
4. **`X_descriptors_raw.parquet`**: 5 raw physicochemical descriptors.

`Tox21Dataset.load_features()` in [`features.py`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/features.py) maps keys unambiguously:
- `ds.load_features("combined")` $\rightarrow$ returns standardized combined matrix $(7823, 2053)$.
- `ds.load_features("combined_raw")` $\rightarrow$ returns raw combined matrix $(7823, 2053)$.
- `ds.load_features("descriptors_scaled")` $\rightarrow$ returns standardized descriptors $(7823, 5)$.
- `ds.load_features("descriptors_raw")` $\rightarrow$ returns raw descriptors $(7823, 5)$.
- `ds.load_features("morgan")` $\rightarrow$ returns binary Morgan bits $(7823, 2048)$.
- `ds.load_features("morgan_filtered")` $\rightarrow$ returns variance-filtered Morgan bits $(7823, 1011)$.

---

## 2. Reconciliation of the Variance-Filtered Matrix

### 2.1 The Issue
Earlier Phase 3 documentation recorded 1,011 retained Morgan features at variance threshold $\ge 0.005$, whereas some handoff text mistakenly listed 1,217 features.

### 2.2 First-Principles Verification
Using the exact training matrix (`X_morgan[train_mask]`, $N = 6,259$):
- **Variance Metric**: Population variance $\text{Var}(X) = \frac{1}{N} \sum_{i=1}^N (x_i - \bar{x})^2$. For binary bits with active proportion $p$, $\text{Var}(X) = p(1 - p)$.
- **Threshold**: $\text{Var}(X) \ge 0.005$ (equivalent to $p \ge 0.005025$, or active in at least 32 training molecules).
- **Constant Bits ($\text{Var} = 0.0$)**: 122 bits are never activated in the training partition.
- **Low-Variance Bits ($0 < \text{Var} < 0.005$)**: 915 bits have fewer than 32 occurrences in training.
- **Total Dropped**: $122 + 915 = 1,037$ bits.
- **Total Retained**: $2,048 - 1,037 = \mathbf{1,011}$ bits.

### 2.3 Root Cause of "1,217"
In [`phase3_feature_analysis/fingerprint_statistics.csv`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/phase3_feature_analysis/fingerprint_statistics.csv), row 1219 corresponds to feature name `morgan_1217` (`variance = 0.005245`). An author drafting documentation mistook the feature bit index `1217` for the feature count `1011`. 

The saved artifact [`X_morgan_unsupervised_filtered.parquet`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/artifacts/X_morgan_unsupervised_filtered.parquet) was verified to have shape `(7823, 1012)` (`mol_id` + 1,011 features) and was never altered. All documentation has been corrected to state **1,011 features**.

---

## 3. Validation of the Notebook-to-Artifact Contract

| Contract Requirement | Target Artifact | Verification Details | Status |
|---|---|---|---|
| **Row Alignment** | `labels_and_splits.parquet` | 7,823 rows matching `corrected_scaffold_split.csv` row-for-row | **PASS** |
| **Morgan Dimensions** | `X_morgan.parquet` | Exactly 2,048 binary bits (uint8 binary {0, 1}) | **PASS** |
| **Descriptor Dimensions** | `X_descriptors_raw.parquet` | Exactly 5 continuous float64 columns | **PASS** |
| **Combined Dimensions** | `X_combined.parquet` | Exactly 2,053 features (2,048 Morgan + 5 descriptors) | **PASS** |
| **Split Partitions** | `labels_and_splits.parquet` | Train: 6,259 (80.01%), Val: 782 (10.00%), Test: 782 (10.00%) | **PASS** |
| **Missing-Label Integrity** | `labels_and_splits.parquet` | 16,012 unobserved assay entries retained as IEEE 754 NaN | **PASS** |
| **Label Invariant 1** | All 12 Assays | $\text{Observed} = \text{Positive} + \text{Negative}$ across all 48 cuts | **PASS** |
| **Label Invariant 2** | All 12 Assays | $\text{Split Total} = \text{Observed} + \text{Missing}$ across all 48 cuts | **PASS** |
| **Training-Only Scaler** | `scaler_descriptors.json` | Fitted on $N = 6,259$ train split; zero test/val leakage | **PASS** |
| **Training-Only PCA** | `X_pca_combined.parquet` | Mean centering and covariance eigenvectors fitted on train only | **PASS** |
| **Zero Feature Leakage** | All feature matrices | No labels, split indicators, or SMILES in predictive columns | **PASS** |
| **Supervised Feature Top-1** | `feature_selection_summary.csv` | NR-AR top feature is `morgan_519` ($\chi^2 = 1,727.23, N = 5,853$) | **PASS** |

---

## 4. Safe Reproducibility Execution

A safe reproducibility audit was executed from a fresh kernel using a temporary execution copy (`temp_pipeline_test.ipynb`):
- All 16 notebook code cells executed sequentially with **0 errors**.
- Execution completed in **19.8 seconds**.
- All assertions evaluated to `True`.
- The committed source notebook [`tox21_member2_pipeline.ipynb`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/tox21_member2_pipeline.ipynb) and upstream artifacts remained strictly uncorrupted.
- The temporary notebook was cleaned up upon completion.
- All 13 registered artifacts in [`manifest.json`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/manifest.json) matched their disk SHA-256 hashes exactly.

---

## 5. Handoff Decision & Certification

All inconsistencies have been resolved, code and documentation are 100% harmonized, and the validation test suite passes cleanly.

### Final Certification
- **Status**: **CERTIFIED PASS**
- **Action**: The feature engineering handoff package is officially approved for **Member 3 (Phase 4: Baseline Model Training)**.
- **Scope Compliance**: No predictive models were trained or evaluated during this phase.
