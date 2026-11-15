# Phase 2 — Member 1: Data Pipeline & Literature

**Role:** ML Infrastructure & Cheminformatics Data Engineer
**Status:** Certified & Frozen Ground Truth
**Target Architecture:** Multi-Task Deep Learning & Hybrid ML Modeling across 12 Tox21 Endpoints

---

## 1. Executive Summary & Module Responsibility

Member 1 owns the **foundation and ground truth** for all Phase 2 modeling and evaluation:
1. **Frozen Dataset & Split Contract:** Freezes the single authoritative Bemis–Murcko scaffold split (Split B: **6,259 train / 782 validation / 782 test**, total **7,823 compounds**).
2. **Feature Handoff Contract:** Connects precomputed Phase 1 feature matrices (`X_combined`, `X_morgan_unsupervised_filtered`, `X_descriptors_scaled`) without re-computation or data leakage.
3. **Labels & Observation Masks:** Preserves experimental missing labels strictly as IEEE 754 `NaN` and provides explicit boolean observation masks for all 12 endpoints.
4. **Class Imbalance & Positive Weights:** Formulates exact positive class weighting tensors strictly from observed training molecules.
5. **Literature Foundation:** Curates 4 peer-reviewed, verified landmark papers justifying the multi-task, scaffold-split, and curation methodology.

---

## 2. Directory Structure

```
phase2/member1_data_pipeline/
├── README.md                           # This documentation and data contract
├── configs/
│   └── split_config.json               # Frozen split metadata, counts, checksums & weights
├── reports/
│   ├── class_imbalance.csv             # Per-endpoint statistics across train, val, test, overall
│   └── validation_summary.json         # Automated verification results (48/48 invariants pass)
├── literature/
│   └── literature_table.csv            # 4 verified scientific publications with active DOIs
├── src/
│   ├── __init__.py                     # Package exports
│   └── data_loader.py                  # High-level loader & PyTorch dataset/dataloader generator
└── tests/
    └── test_data_pipeline.py           # Automated test suite (10/10 quality gates pass)
```

---

## 3. The Frozen Split Contract (Split B)

### 3.1 Split Identity & Checksums
Downstream members must consume **Split B exclusively** (`tox21_member2_handoff/artifacts/corrected_scaffold_split.csv` and `labels_and_splits.parquet`).

| Split Partition | Molecule Count | Proportion | Cyclic Scaffolds | Acyclic Molecules | Role in Phase 2 Pipeline |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Train** | **6,259** | 80.01% | 998 | 1,775 | Multi-task DL & ML training |
| **Validation** | **782** | 10.00% | 703 | 0 | Early stopping & hyperparameter tuning |
| **Test** | **782** | 10.00% | 703 | 0 | Unbiased holdout benchmark (Member 4) |
| **Total Valid** | **7,823** | 100.00% | 2,404 | 1,775 | Certified handoff population |

- **Zero Overlap:** Train $\cap$ Val = 0, Train $\cap$ Test = 0, Val $\cap$ Test = 0 (for both molecule IDs and Bemis–Murcko scaffolds).
- **Exact Checksum (SHA-256):**
  - `corrected_scaffold_split.csv`: `89a23c7c9d1559f8561aff5316696aff37ef558293f5af58d1f4d484479d2d01`
  - `labels_and_splits.parquet`: `b62a2bc92c2bfa572d1f157ebd49a7b12819eec0b616c9bc52c9cda1f2aba1ab`

### 3.2 Methodological Note on Acyclic Chemotypes
In Split B, all 1,775 acyclic compounds (empty Bemis–Murcko scaffold `""`) were clustered into the training partition to prevent fractional leakage. Consequently:
- **Validation and Test partitions evaluate out-of-distribution cyclic and heterocyclic scaffold generalization only.**
- Models are tested on entirely novel ring cores (macrocycles, heterocycles, fused aromatics). They are not evaluated on purely acyclic molecules in holdout test. This must be noted in individual viva presentations and report sections.

---

## 4. Supported Feature Spaces

All feature matrices maintain **100.0% row-for-row compound-ID alignment** with the frozen split:

| Feature Set Identifier | Dimensions ($D$) | Description | Primary Consumer |
| :--- | :---: | :--- | :--- |
| `'combined'` | **2,053** | 2,048 binary Morgan bits (radius=2) + 5 standardized descriptors | **Member 2 (DL Canonical)** |
| `'morgan_filtered'` | **1,011** | High-variance Morgan bits ($\text{Var}_{\text{train}} \ge 0.005$) | Member 2 (Ablation / Compact DL) |
| `'descriptors_scaled'` | **5** | Continuous descriptors standardized on train ($Z$-score) | **Member 3 (Hybrid RF)** |
| `'morgan'` | **2,048** | Binary circular fingerprints (ECFP4-like) | Member 4 (Ablation study) |
| `'descriptors_raw'` | **5** | Raw continuous descriptors (`MolWt`, `TPSA`, `MolLogP`, `NumHDonors`, `NumHAcceptors`) | Baseline inspection |
| `'combined_raw'` | **2,053** | 2,048 Morgan bits + 5 raw continuous descriptors | Baseline inspection |

---

## 5. Label Masking & Class Weighting Contract

### 5.1 Masked Loss Formulation (for Member 2 DL Model)
Because 17.06% of assay readouts are unmeasured, Member 2's multitask loss must multiply the binary cross-entropy by the boolean observation mask $m_{ij} \in \{0, 1\}$:

$$\mathcal{L}_{\text{multitask}} = \frac{1}{\sum_{i=1}^B \sum_{j=1}^{12} m_{ij}} \sum_{i=1}^B \sum_{j=1}^{12} m_{ij} \cdot \left[ - w_j y_{ij} \log(\hat{y}_{ij}) - (1 - y_{ij}) \log(1 - \hat{y}_{ij}) \right]$$

> [!WARNING]
> **CRITICAL PYTORCH LOSS MASKING REQUIREMENT:**
> In PyTorch batches (`Tox21TorchDataset`), missing labels (NaNs) are replaced with zero (`0.0`) in the label tensor `y_batch` to prevent NaN gradient corruption during backpropagation.
> Consequently, downstream training **MUST** apply the observation mask `mask_batch` to the element-wise unreduced loss before reduction (`reduction="none"`):
> ```python
> unreduced_loss = F.binary_cross_entropy_with_logits(logits, y_batch, pos_weight=pos_weights, reduction="none")
> loss = (unreduced_loss * mask_batch).sum() / mask_batch.sum()
> ```
> If `reduction="mean"` is used directly without applying `mask_batch`, missing labels will be silently treated as negative/inactive, introducing severe label contamination and violating core project rules.

### 5.2 Exact Positive Class Weights ($w_j = \frac{N_{\text{neg, train}}}{N_{\text{pos, train}}}$)
Weights are calculated strictly from training partition observed labels:

| Endpoint | Target Description | Train Actives ($N_{\text{pos}}$) | Train Inactives ($N_{\text{neg}}$) | Positive Weight ($w_j$) |
| :--- | :--- | :---: | :---: | :---: |
| **NR-AR** | Androgen Receptor | 233 | 5,620 | **24.12** |
| **NR-AR-LBD** | Androgen Receptor (LBD) | 190 | 5,314 | **27.97** |
| **NR-AhR** | Aryl Hydrocarbon Receptor | 583 | 4,731 | **8.11** |
| **NR-Aromatase** | Aromatase Enzyme | 206 | 4,593 | **22.30** |
| **NR-ER** | Estrogen Receptor | 629 | 4,460 | **7.09** |
| **NR-ER-LBD** | Estrogen Receptor (LBD) | 289 | 5,352 | **18.52** |
| **NR-PPAR-gamma** | PPAR-$\gamma$ | 133 | 5,167 | **38.85** |
| **SR-ARE** | Antioxidant Response Element | 724 | 4,132 | **5.71** |
| **SR-ATAD5** | ATAD5 Genotoxicity | 192 | 5,549 | **28.901** |
| **SR-HSE** | Heat Shock Response | 282 | 5,032 | **17.84** |
| **SR-MMP** | Mitochondrial Potential | 712 | 4,080 | **5.73** |
| **SR-p53** | p53 Tumor Suppressor | 285 | 5,232 | **18.36** |

---

## 6. Downstream Member Quickstart & API Guide

### 6.1 Installation
```bash
pip install -r requirements.txt
```

### 6.2 For Member 2: Deep Learning Model
```python
from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader

loader = Tox21DataLoader()

# 1. Option A: Native PyTorch DataLoaders (batch_size=64)
# Returns (X, y, mask) tensors where mask indicates valid observed targets
train_loader, val_loader, test_loader = loader.get_pytorch_dataloaders(
    feature_set="combined",  # 2,053 features (Morgan + scaled descriptors)
    batch_size=64,
    shuffle_train=True
)

# 2. Get positive class weight tensor for BCEWithLogitsLoss
pos_weights = loader.get_class_weights(as_tensor=True)  # torch.FloatTensor of shape (12,)

# 3. Training Loop Step Example:
# for X_batch, y_batch, mask_batch in train_loader:
#     logits = model(X_batch)
#     loss = (F.binary_cross_entropy_with_logits(
#         logits, y_batch, pos_weight=pos_weights, reduction="none"
#     ) * mask_batch).sum() / mask_batch.sum()
```

### 6.3 For Member 3: ML + DL Hybrid Model
```python
from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader
import numpy as np

loader = Tox21DataLoader()

# 1. Load scaled 5 continuous descriptors (fitted strictly on train)
X_desc_train, _ = loader.load_features("descriptors_scaled", split="train") # (6259, 5)
X_desc_test, _  = loader.load_features("descriptors_scaled", split="test")  # (782, 5)

# 2. Assume Member 2 provides learned latent embeddings: z_train (6259, D_latent), z_test (782, D_latent)
# X_hybrid_train = np.hstack([z_train, X_desc_train])
# X_hybrid_test  = np.hstack([z_test, X_desc_test])

# 3. Train one Random Forest per endpoint using observed masks:
for ep_idx, ep in enumerate(loader.TARGET_ASSAYS):
    y_tr, mask_tr, _ = loader.load_labels(split="train", endpoints=[ep])
    obs = mask_tr[:, 0]  # boolean filter
    # rf.fit(X_hybrid_train[obs], y_tr[obs, 0])
```

### 6.4 For Member 4: Rigorous Evaluation & Ablation Studies
```python
from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader

loader = Tox21DataLoader()

# Load exact holdout test data
X_test, test_mids = loader.load_features("combined", split="test")
y_test, mask_test, test_mids_y = loader.load_labels(split="test")
assert test_mids == test_mids_y  # Guaranteed row-for-row alignment

# Evaluate each model on identical test instances:
# for ep_idx, ep in enumerate(loader.TARGET_ASSAYS):
#     obs = mask_test[:, ep_idx]
#     y_true = y_test[obs, ep_idx]
#     roc = roc_auc_score(y_true, y_pred_prob[obs])
```

### 6.5 For Member 5: Streamlit App & Presentation Report
- Load `phase2/member1_data_pipeline/reports/class_imbalance.csv` directly for interactive data distribution charts in Streamlit.
- Reference `phase2/member1_data_pipeline/literature/literature_table.csv` for the IEEE report literature section.

---

## 7. Quality Gates & Test Suite Verification

Run the automated verification suite:

```bash
python3 -m unittest phase2/member1_data_pipeline/tests/test_data_pipeline.py
```

### Automated Quality Gates Result:
- **Test 1:** Exact Split Counts (6,259 / 782 / 782) — **PASS**
- **Test 2:** Zero Molecule-ID Overlap — **PASS**
- **Test 3:** Zero Bemis–Murcko Scaffold Overlap — **PASS**
- **Test 4:** Acyclic Allocation Policy (1,775 in train, 0 in val/test) — **PASS**
- **Test 5:** Missing Label Preservation & Boolean Mask Integrity — **PASS**
- **Test 6:** Label Invariants ($Obs = Pos + Neg$, $Tot = Obs + Miss$) 48/48 — **PASS**
- **Test 7:** Evaluability Gate ($\ge 20$ positives per endpoint in val/test) — **PASS**
- **Test 8:** Training-Only Class Weight Formulation — **PASS**
- **Test 9:** Feature Dimension Constants (2,053 / 1,011 / 5) — **PASS**
- **Test 10:** Verified Scientific Literature Table (4 papers with DOIs) — **PASS**

---

## 8. Verified Scientific References

1. **Mayr et al. (2016)**, *DeepTox: Toxicity Prediction using Deep Learning*. *Frontiers in Environmental Science*, 3, 80. [DOI: 10.3389/fenvs.2015.00080](https://doi.org/10.3389/fenvs.2015.00080)
2. **Wu et al. (2018)**, *MoleculeNet: a benchmark for molecular machine learning*. *Chemical Science*, 9(2), 513–530. [DOI: 10.1039/c7sc02664a](https://doi.org/10.1039/c7sc02664a)
3. **Huang et al. (2014)**, *Profiling of the Tox21 10K compound library for agonists and antagonists of the estrogen receptor alpha signaling pathway*. *Scientific Reports*, 4, 5664. [DOI: 10.1038/srep05664](https://doi.org/10.1038/srep05664)
4. **Fourches et al. (2010)**, *Trust, but Verify: On the Importance of Chemical Structure Curation in Cheminformatics and QSAR Modeling Research*. *Journal of Chemical Information and Modeling*, 50(7), 1189–1204. [DOI: 10.1021/ci100176x](https://doi.org/10.1021/ci100176x)
