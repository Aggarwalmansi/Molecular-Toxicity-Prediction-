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
