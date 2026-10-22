# Final Handoff Change Log: Tox21 Member 2 Pipeline

**Role**: Senior ML Engineer & Cheminformatics Specialist  
**Package**: [`tox21_member2_handoff/`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/)  
**Date**: October 2026  

---

## 1. Summary of Changes and Resolutions

### Change 1: Clarification of `X_combined.parquet` and Addition of `X_combined_raw.parquet`
- **Issue**: Ambiguity regarding whether `X_combined.parquet` contained raw or standardized descriptors.
- **Audit Finding**: Empirical evaluation of `X_combined.parquet` confirmed that descriptor columns have mean $\approx 0.0$ and std $\approx 1.0$ on the training partition ($N = 6,259$). It contains **2,048 binary Morgan features plus 5 standardized descriptors**.
- **Action Taken**:
  1. Maintained `X_combined.parquet` as the canonical modeling matrix with standardized descriptors.
  2. Created [`tox21_member2_handoff/artifacts/X_combined_raw.parquet`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/artifacts/X_combined_raw.parquet) containing 2,048 binary Morgan features plus 5 raw continuous descriptors (`MolWt`, `TPSA`, `MolLogP`, `NumHDonors`, `NumHAcceptors` in natural units).
  3. Updated `Tox21Dataset.load_features()` in [`features.py`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/features.py) to support both `'combined'` (standardized) and `'combined_raw'` (raw) keys with explicit documentation.
  4. Updated [`README.md`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/README.md) and [`manifest.json`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/manifest.json) to catalog both representations.

### Change 2: Reconciliation of the Variance-Filtered Matrix (1,011 vs. 1,217)
- **Issue**: Discrepancy between earlier reports of 1,011 retained Morgan features and some documentation listing 1,217 features.
- **Audit Finding**:
  - Training variance $\text{Var}(X) = p(1 - p)$ computed on the training partition ($N = 6,259$) with threshold $\ge 0.005$ yields **exactly 1,011 retained features** (122 constant bits and 915 low-variance bits dropped; $2,048 - 1,037 = 1,011$).
  - The physical parquet artifact [`X_morgan_unsupervised_filtered.parquet`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/artifacts/X_morgan_unsupervised_filtered.parquet) has shape `(7823, 1012)` (`mol_id` + 1,011 features) and has always contained 1,011 features.
  - The number "1,217" was a clerical typo tracing back to row index 1219 (`morgan_1217`) in `phase3_feature_analysis/fingerprint_statistics.csv`.
- **Action Taken**:
  1. Corrected all occurrences of "1,217" to "1,011" in [`README.md`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/README.md) and [`features.py`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/features.py).
  2. Verified that the physical parquet file was already 100% correct and required no modification.

### Change 3: Synchronization of `manifest.json` Checksums
- **Action**: Recalculated exact byte sizes and SHA-256 hashes for all 13 artifacts in `tox21_member2_handoff/artifacts/` and updated [`manifest.json`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/manifest.json).

### Change 4: Safe Reproducibility Validation Execution
- **Action**: Tested pipeline reproducibility by copying [`tox21_member2_pipeline.ipynb`](file:///Users/mansiagarwal/molecular%20toxicity-%20grp_project/tox21_member2_handoff/tox21_member2_pipeline.ipynb) to a temporary test notebook, executing all 16 code cells in a clean environment, verifying 0 errors in 19.8s, and cleaning up the temporary notebook without touching the committed source notebook.

---

## 2. Updated File Inventory

| File | Change Type | Description |
|---|---|---|
| `features.py` | Modified | Added `'combined_raw'` to `Tox21Dataset.load_features()`; corrected filtered feature docstring to 1,011. |
| `README.md` | Modified | Updated directory inventory and feature table to include `X_combined_raw.parquet` and correct filtered count to 1,011. |
| `manifest.json` | Modified | Cataloged all 13 artifacts with updated SHA-256 checksums. |
| `artifacts/X_combined_raw.parquet` | Created | Added raw combined feature matrix (2,048 Morgan bits + 5 raw descriptors). |
| `final_handoff_audit.md` | Created | Comprehensive final audit report for Member 3 handoff. |
| `final_handoff_checks.txt` | Created | Automated verification test suite log (13/13 PASS). |
| `final_handoff_change_log.md` | Created | This change log document. |
