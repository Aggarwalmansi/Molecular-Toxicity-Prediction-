# Molecular Toxicity Prediction (Tox21 Data Challenge)

This repository contains the collaborative molecular toxicity prediction project on the Tox21 10,000 environmental chemical library across 12 toxicological endpoints (Nuclear Receptors and Stress Response pathways).

---

## Team Workflow & Repository Structure

The project is structured into modular phase deliverables:

```
.
├── README.md                      # Project root overview and quickstart for Member 3
├── tox21_project/                 # Member 1: Dataset Preprocessing, EDA, and Scaffold Split
└── tox21_member2_handoff/         # Member 2: Feature Engineering, PCA, Feature Selection, and Certified Handoff
    ├── README.md                  # Comprehensive Member 2 documentation and usage guide
    ├── tox21_member2_pipeline.ipynb # Clean sequential pipeline notebook (16 sections)
    ├── features.py                # Reusable library and Tox21Dataset loader
    ├── requirements.txt           # Minimal environment dependencies
    ├── manifest.json              # Checksums and file size specifications
    ├── validation_results.txt     # Test suite results (16/16 checks PASS)
    ├── feature_dictionary.csv     # Dictionary defining all 2,053 features
    ├── final_handoff_audit.md     # Independent scientific audit report
    ├── final_handoff_checks.txt   # Detailed audit assertions and proofs
    ├── final_handoff_change_log.md# Handoff change log
    └── artifacts/                 # Certified feature matrices and metadata
        ├── X_combined.parquet     # (7823, 2054) Canonical: 2,048 Morgan bits + 5 standardized descriptors
        ├── X_combined_raw.parquet # (7823, 2054) 2,048 Morgan bits + 5 raw descriptors
        ├── X_morgan.parquet       # (7823, 2049) Fixed-length Morgan circular fingerprints (radius=2)
        ├── X_morgan_unsupervised_filtered.parquet # (7823, 1012) 1,011 filtered Morgan bits (Var >= 0.005)
        ├── X_descriptors_scaled.parquet # (7823, 6) 5 standardized physicochemical descriptors
        ├── X_descriptors_raw.parquet    # (7823, 6) 5 raw physicochemical descriptors
        ├── X_pca_combined.parquet       # (7823, 50) Top 50 principal components of combined features
        ├── X_pca_morgan.parquet         # (7823, 50) Top 50 principal components of Morgan fingerprints
        ├── X_pca_descriptors.parquet    # (7823, 5)  5 principal components of descriptors
        ├── corrected_scaffold_split.csv # (7823, 17) Canonical 80/10/10 scaffold split source
        ├── labels_and_splits.parquet    # (7823, 17) Compound IDs, split masks, and 12 target assays
        ├── scaler_descriptors.json      # Training-fitted scaler parameters (mean, std)
        ├── endpoint_top50_features.csv  # Top 50 associated features per endpoint (Chi2/ANOVA)
        └── feature_selection_summary.csv# Summary of top feature associations per endpoint
├── train_models.py                # Member 3: Baseline model training
├── evaluate.py                    # Member 4: Comprehensive evaluation, bootstrap CIs, paired tests, and plotting
├── requirements-eval.txt          # Member 4: Minimal dependencies for evaluation and analysis
└── results/                       # Member 4: Official evaluation metrics, statistics, and diagnostic plots
    ├── metrics.csv                # Per-endpoint metrics with 95% bootstrap CIs and threshold stats
    ├── summary_metrics.csv        # Macro-averaged summary table across all 12 endpoints
    ├── statistical_tests.csv      # Paired Bootstrap and Wilcoxon signed-rank significance tests
    ├── split_comparison.csv       # Scaffold Split vs. Random Split benchmark (Task 6)
    ├── test_predictions.csv       # Compound-level predictions with SMILES, scaffolds, probabilities, and labels
    ├── error_analysis.md          # Scientific error analysis report with chemical and statistical failure modes
    └── plots/                     # 7 publication-grade diagnostic figures (300 DPI)
```

---

## Member 3 Quickstart: Baseline Model Training

Member 3 can clone the repository and begin model training immediately using the `Tox21Dataset` loader:

```python
import sys
sys.path.append("tox21_member2_handoff")
from features import Tox21Dataset

# 1. Initialize dataset loader
ds = Tox21Dataset("tox21_member2_handoff/artifacts")

# 2. Load canonical combined features (7,823 molecules x 2,053 features)
X_all = ds.load_features("combined")
X_train = X_all[ds.train_idx]  # Shape: (6259, 2053)
X_val   = X_all[ds.val_idx]    # Shape: (782, 2053)
X_test  = X_all[ds.test_idx]   # Shape: (782, 2053)

# 3. Load target labels and observed-label masks (e.g., NR-AR)
y_train, mask_train = ds.get_labels("NR-AR", split="train")
y_val, mask_val     = ds.get_labels("NR-AR", split="val")
y_test, mask_test   = ds.get_labels("NR-AR", split="test")

# 4. Train only on observed molecules (missing labels strictly masked)
X_train_obs = X_train[mask_train]  # Shape: (5853, 2053)
y_train_obs = y_train[mask_train]  # 233 active, 5,620 inactive

# Fit model (e.g. LogisticRegression, RandomForestClassifier, LightGBM)
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train_obs, y_train_obs)
```

For full feature definitions, PCA analysis, and univariate rankings, see [`tox21_member2_handoff/README.md`](tox21_member2_handoff/README.md).

---

## Member 4: Evaluation, Statistics and Error Analysis

Member 4 provides rigorous statistical benchmarking, error diagnosis, and proves whether models truly generalize:

### Running the Evaluation Suite
```bash
# 1. Install dependencies
pip install -r requirements-eval.txt

# 2. Run full evaluation, statistical tests, Task 6 random-split experiment, and generate all plots
python evaluate.py
```

### Macro-Averaged Benchmark Summary (All 12 Endpoints)

| Model | Test ROC-AUC (Mean ± Std) | Test PR-AUC (Mean ± Std) | F1-Score (Mean ± Std) | Balanced Accuracy | Matthews Corr. Coeff. (MCC) |
|---|---|---|---|---|---|
| **Random Forest** | **0.7541 ± 0.0570** | **0.3962 ± 0.1323** | **0.3969 ± 0.1070** | **0.6684 ± 0.0507** | **0.3141 ± 0.1177** |
| **Logistic Regression** | 0.7152 ± 0.0596 | 0.2949 ± 0.1319 | 0.3268 ± 0.1355 | 0.6358 ± 0.0761 | 0.2413 ± 0.1233 |
| **Decision Tree** | 0.6428 ± 0.0597 | 0.2091 ± 0.0957 | 0.2629 ± 0.1182 | 0.6295 ± 0.0592 | 0.1616 ± 0.0859 |

### Key Takeaways
- **Statistical Superiority**: Random Forest significantly outperforms Logistic Regression ($p < 0.01$) and Decision Tree ($p < 0.001$) across almost all endpoints in PR-AUC and probabilistic prediction error (Wilcoxon signed-rank test).
- **Scaffold Generalization Gap (Task 6)**: Random splitting artificially inflates macro ROC-AUC by **+0.0840** (up to **+0.1616** on `SR-p53` and **+0.1610** on `NR-ER`). Scaffold splitting is crucial to avoid false optimism in molecular property prediction.
- **Scientific Error Analysis**: Full failure mode analysis on underperforming targets (`NR-ER`, `NR-PPAR-gamma`, `SR-ATAD5`) is documented in [`results/error_analysis.md`](results/error_analysis.md).
