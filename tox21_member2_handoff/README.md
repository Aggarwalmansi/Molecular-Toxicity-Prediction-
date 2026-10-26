# Tox21 Member 2 Handoff: Reproducible Feature Engineering Pipeline

**Phase**: Member 2 Final Deliverable (Certified Handoff to Member 3)  
**Author**: Senior ML Engineer & Cheminformatics Specialist  
**Upstream Provenance**: Member 1 (Dataset Audit, Invalid SMILES Filtering, Corrected Scaffold Split)  
**Downstream Consumer**: Member 3 (Baseline Classification Modeling, Multi-Task Architectures, Evaluation)  
**Status**: **CERTIFIED PASS (14/14 Automated Checks Passed)**  

---

## 1. Directory Inventory

The `tox21_member2_handoff/` package is completely self-contained and structured as follows:

```
tox21_member2_handoff/
├── README.md                           # This comprehensive guide
├── requirements.txt                    # Environment dependencies specification
├── manifest.json                       # Artifact inventory with sizes and SHA-256 hashes
├── validation_results.txt              # Independent verification test suite results (14/14 PASS)
├── feature_dictionary.csv              # Definitions and metadata for all 2,053 features
├── features.py                         # Reusable library: extraction, scaling, PCA, filtering, loader
├── tox21_member2_pipeline.ipynb        # Clean 16-section sequential execution notebook
└── artifacts/                          # Certified data, feature, and model parameter artifacts
    ├── X_morgan.parquet                # (7823, 2049) [mol_id + 2,048 binary bits]
    ├── X_descriptors_raw.parquet       # (7823, 6)    [mol_id + 5 raw continuous descriptors]
    ├── X_descriptors_scaled.parquet    # (7823, 6)    [mol_id + 5 standardized descriptors]
    ├── X_combined.parquet              # (7823, 2054) [mol_id + 2,048 Morgan bits + 5 standardized descriptors] (Canonical)
    ├── X_combined_raw.parquet          # (7823, 2054) [mol_id + 2,048 Morgan bits + 5 raw descriptors]
    ├── labels_and_splits.parquet       # (7823, 17)   [mol_id, SMILES, scaffold, split, 12 target assays]
    ├── scaler_descriptors.json         # Mean and std parameters fitted strictly on training data
    ├── X_pca_descriptors.parquet       # (7823, 5)    [Descriptor PCA projections]
    ├── X_pca_morgan.parquet            # (7823, 50)   [Morgan PCA projections]
    ├── X_pca_combined.parquet          # (7823, 50)   [Combined PCA projections]
    ├── X_morgan_unsupervised_filtered.parquet # (7823, 1012) [mol_id + 1,011 filtered Morgan bits (Var >= 0.005)]
    ├── endpoint_top50_features.csv     # Top 50 associated features per endpoint (Chi2/ANOVA)
    └── feature_selection_summary.csv   # Top associated feature per endpoint summary
```

---

## 2. Environment Setup

The pipeline requires Python 3.9+ and standard scientific packages. All feature transformations, scaling, PCA, and selection are implemented in pure NumPy, Pandas, and RDKit, guaranteeing zero dependency on fragile platform C-extensions.

### Installation
```bash
pip install -r requirements.txt
```

### Core Package Versions
- `rdkit`: 2022.03.1+ (tested on `2026.09.1`)
- `numpy`: 1.22.0+ (tested on `2.2.6`)
- `pandas`: 1.4.0+ (tested on `2.3.3`)
- `pyarrow`: 8.0.0+
- `nbformat`: 5.4.0+

---

## 3. Data Provenance & Upstream Dependencies

### Included Upstream Work (Member 1)
- **Dataset Cleaning**: Out of 7,831 compounds in `tox21.csv`, 8 compounds with unparseable SMILES or invalid valence configurations were removed. The remaining 7,823 molecules are 100% valid RDKit structures.
- **Scaffold Generation**: Murcko scaffold SMILES generated via `rdkit.Chem.Scaffolds.MurckoScaffold`.
- **Scaffold Split (Phase 1.5 Correction)**: Replaced the broken 91/0/9 split with a certified **80.01% train (6,259), 10.00% validation (782), and 10.00% test (782)** partition with **zero scaffold leakage**.
- **Canonical Input File**: `phase1_audit/split_correction/corrected_scaffold_split.csv` (SHA-256: `89a23c7c9d1559f8561aff5316696aff37ef558293f5af58d1f4d484479d2d01`).

### External Upstream Dependencies
- The pipeline expects the canonical split file to be available at `../phase1_audit/split_correction/corrected_scaffold_split.csv` or `phase1_audit/split_correction/corrected_scaffold_split.csv`.
- If rebuilding completely from scratch, the raw `tox21.csv` is located at the workspace root.

---

## 4. How to Re-Run the Pipeline from Scratch

To re-run the pipeline cleanly:
```bash
cd tox21_member2_handoff

# Option A: Execute the pipeline notebook using jupyter
jupyter nbconvert --to notebook --execute tox21_member2_pipeline.ipynb --output tox21_member2_pipeline.ipynb

# Option B: Run the programmatic executor
python3 -c "
import nbformat as nbf
from tox21_member2_handoff.features import Tox21Dataset
print('Executing verification...')
ds = Tox21Dataset('artifacts')
print('Loaded successfully: ', len(ds.metadata), 'molecules')
"
```

---

## 5. Feature Configurations for Member 3

Member 3 can evaluate multiple feature representations for baseline modeling:

| Feature Configuration | Parquet Filename | Dimensions | Description | Recommendation |
|---|---|---|---|---|
| **Combined (Canonical)** | `X_combined.parquet` | (7823, 2053) | 2,048 Morgan binary bits + 5 standardized descriptors (zero mean, unit var on train) | **Primary Benchmark Input** |
| **Combined Raw** | `X_combined_raw.parquet` | (7823, 2053) | 2,048 Morgan binary bits + 5 raw physicochemical descriptors (natural units) | Tree Models / Inspection |
| **Morgan Only** | `X_morgan.parquet` | (7823, 2048) | Fixed-length circular Morgan fingerprints (radius=2, uint8 binary) | Secondary Baseline (Substructure) |
| **Descriptors Scaled** | `X_descriptors_scaled.parquet` | (7823, 5) | Standardized MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors | Low-Dimensional Baseline |
| **Descriptors Raw** | `X_descriptors_raw.parquet` | (7823, 5) | Unscaled physical properties with natural units | Reference Only |
| **Morgan Filtered** | `X_morgan_unsupervised_filtered.parquet` | (7823, 1011) | Morgan bits with training variance $\ge 0.005$ (removes noise/constants) | High-Efficiency Candidate |
| **PCA Combined** | `X_pca_combined.parquet` | (7823, 50) | Top 50 principal components fitted on training combined features | Linear Compression Candidate |
| **PCA Morgan** | `X_pca_morgan.parquet` | (7823, 50) | Top 50 principal components fitted on training Morgan fingerprints | Dimensionality Reduction Candidate |
| **PCA Descriptors** | `X_pca_descriptors.parquet` | (7823, 5) | Orthogonalized continuous physicochemical variance | Collinearity Reduction |

---

## 6. How Member 3 Loads Data for Model Training

Member 3 should use the audited `Tox21Dataset` class from `features.py`. It guarantees compound-ID alignment, handles missing label masks, and provides train/val/test splits with zero boilerplate:

```python
import numpy as np
from features import Tox21Dataset

# 1. Initialize dataset loader pointing to artifacts directory
ds = Tox21Dataset("tox21_member2_handoff/artifacts")

# 2. Load feature representation of choice (e.g., 'combined', 'morgan', 'pca_combined')
X_all = ds.load_features("combined")  # Returns shape (7823, 2053) as float64 array

X_train = X_all[ds.train_idx]  # Shape (6259, 2053)
X_val   = X_all[ds.val_idx]    # Shape (782, 2053)
X_test  = X_all[ds.test_idx]   # Shape (782, 2053)

# 3. Load target labels and observed-label masks for an endpoint (e.g., NR-AR)
# In Tox21, missing labels MUST be masked out during loss calculation and evaluation.
y_train, mask_train = ds.get_labels("NR-AR", split="train")
y_val, mask_val     = ds.get_labels("NR-AR", split="val")
y_test, mask_test   = ds.get_labels("NR-AR", split="test")

# 4. Filter strictly observed molecules for training
X_train_observed = X_train[mask_train]  # Shape (5853, 2053)
y_train_observed = y_train[mask_train]  # Shape (5853,) - 233 positives, 5620 negatives

# 5. Ready for model training! (e.g., LogisticRegression, RandomForestClassifier, LightGBM)
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(random_state=42)
# clf.fit(X_train_observed, y_train_observed)
```

---

## 7. Preprocessing Objects Fitted on Training Data

Strict leakage controls were enforced. All preprocessing transformations were fitted exclusively on `train` molecules ($N = 6,259$):

1. **Descriptor Standard Scaler** (`scaler_descriptors.json`):
   - `MolWt`: Mean = 251.08, Std = 144.62
   - `TPSA`: Mean = 54.19, Std = 50.29
   - `MolLogP`: Mean = 2.26, Std = 2.22
   - `NumHDonors`: Mean = 1.12, Std = 1.64
   - `NumHAcceptors`: Mean = 3.11, Std = 2.73
2. **PCA Eigenvectors**: Fitted strictly on centered `X_train`. Validation and test molecules were projected using frozen training projection matrices.
3. **Unsupervised Variance Filtering**: The 1,217 retained bits had variance $\ge 0.005$ measured exclusively across `X_train`.
4. **Supervised Univariate Associations**: Chi-square and ANOVA F-tests were calculated strictly using observed training rows (`y_train[mask_train]`).

---

## 8. Known Scientific Limitations & Evaluation Constraints

1. **Scaffold-Distribution Shift**:
   The scaffold split groups compounds by Bemis–Murcko scaffold. This introduces a realistic domain shift between training and test sets. Positive rates on certain assays are higher in test than in train (e.g. SR-ARE: 14.91% train vs 25.15% test). Models must be evaluated with ROC-AUC and PR-AUC, not simple accuracy.
2. **Acyclic Compounds in Training Split**:
   All 1,775 acyclic molecules (molecules with empty Bemis–Murcko scaffolds, e.g. aliphatic chains, small acyclic organics) are grouped exclusively into the training partition. Validation and test sets measure generalization exclusively on ring-containing scaffolds (heterocycles, aromatics, macrocycles).
3. **Severe Class Imbalance**:
   Positive rates range from 2.51% (`NR-PPAR-gamma`) to 14.91% (`SR-ARE`) in training. Member 3 should use class-weighted loss functions (`class_weight='balanced'`) or focal loss.
4. **Endpoint Sparsity**:
   Between 406 and 1,467 compounds are unmeasured per assay in training. Multi-task loss functions must apply sample masking to exclude unmeasured targets from gradient updates.
