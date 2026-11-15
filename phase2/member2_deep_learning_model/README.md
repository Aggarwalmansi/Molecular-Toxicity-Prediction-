# Phase 2 — Member 2: Multi-Task Deep Learning Model

**Role:** Deep Learning Research Engineer & Representation Learning Specialist  
**Status:** Trained, Verified & Production Ready  
**Target Architecture:** Multi-Task Deep Neural Network (`MultiTaskToxicNet`) predicting all 12 Tox21 biological pathways simultaneously with Masked Loss, Class Weighting, and Latent Representation Extraction  

---

## 1. Executive Summary & Deliverables

Member 2 is responsible for the deep learning core of Phase 2, predicting all 12 Tox21 biological endpoints simultaneously using hard parameter sharing:
1. **Multi-Task Neural Network (`MultiTaskToxicNet`):** PyTorch deep neural network accepting canonical 2,053-dim molecular features (`X_combined`: 2,048-bit Morgan circular fingerprints + 5 standardized physical descriptors) and predicting 12 target assays in a single forward pass.
2. **Masked & Weighted Loss (`MaskedWeightedBCEWithLogitsLoss`):** Strictly excludes unmeasured assay targets (17.06% missing labels) from backpropagation while applying exact training-set positive class weights ($w_j = 5.71 \to 38.85$) to counter severe activity imbalance.
3. **Regularization & Early Stopping:** Dropout ($p=0.35$), weight decay ($1\times 10^{-4}$), batch normalization, and early stopping monitoring validation macro ROC-AUC with learning rate scheduling.
4. **Checkpoint Deliverable:** Certified best model checkpoint saved at `checkpoints/best_multitask_model.pt`.
5. **Learned Molecular Features (Embeddings Handoff for Member 3):** 128-dimensional continuous latent representations $z \in \mathbb{R}^{128}$ extracted from the bottleneck layer for all 7,823 compounds (`train_embeddings.parquet`, `val_embeddings.parquet`, `test_embeddings.parquet`) with row-for-row compound ID alignment.
6. **Holdout Test Benchmark:** Evaluated on the 782-compound holdout test set across all 12 endpoints (ROC-AUC, PR-AUC, F1, Balanced Accuracy, MCC).

---

## 2. Directory Structure

```
phase2/member2_deep_learning_model/
├── README.md                           # This documentation & handoff contract
├── train.py                            # End-to-end training & orchestration script
├── configs/
│   └── model_config.json               # Model dimensions, hyperparameters & training config
├── checkpoints/
│   └── best_multitask_model.pt         # Best model weights, optimizer state & metadata
├── embeddings/
│   ├── train_embeddings.parquet        # Latent representations (6,259 × 128) with molecule_id
│   ├── val_embeddings.parquet          # Latent representations (782 × 128) with molecule_id
│   ├── test_embeddings.parquet         # Latent representations (782 × 128) with molecule_id
│   ├── train_embeddings.npy            # Raw numpy array (6,259, 128)
│   ├── val_embeddings.npy              # Raw numpy array (782, 128)
│   ├── test_embeddings.npy             # Raw numpy array (782, 128)
│   └── embeddings_metadata.json        # Checksums, dimensions & partition summary
├── reports/
│   ├── test_evaluation.csv             # Holdout test metrics across all 12 endpoints
│   ├── training_history.csv            # Per-epoch loss, validation ROC-AUC, PR-AUC, and LR
│   └── plots/
│       ├── loss_curve.png              # Train vs. validation loss curve
│       ├── val_metrics_curve.png       # Validation macro ROC-AUC & PR-AUC trajectory
│       ├── training_curves_summary.png # Dual-panel summary figure
│       └── per_endpoint_roc_pr.png     # Test ROC-AUC & PR-AUC comparison by endpoint
├── src/
│   ├── __init__.py                     # Package exports
│   ├── model.py                        # MultiTaskToxicNet PyTorch architecture
│   ├── loss.py                         # MaskedWeightedBCEWithLogitsLoss module
│   ├── trainer.py                      # MultiTaskTrainer engine with early stopping
│   ├── extract_embeddings.py           # Latent representation extractor
│   └── evaluate.py                     # 12-endpoint holdout test evaluator
└── tests/
    └── test_dl_model.py                # Automated test suite (10/10 quality gates pass)
```

---

## 3. Architecture & Mathematical Formulation

### 3.1 Network Topology

```
                  Canonical Input Feature Vector (2,053 dims)
                [2,048 Morgan circular bits + 5 standardized descriptors]
                                    │
                                    ▼
       Linear(2,053 → 512) ── BatchNorm1d ── ReLU ── Dropout(0.35)
                                    │
                                    ▼
       Linear(512 → 256)   ── BatchNorm1d ── ReLU ── Dropout(0.35)
                                    │
                                    ▼
       Linear(256 → 128)   ── BatchNorm1d ── ReLU ── Dropout(0.175)
                                    │
                         ┌──────────┴──────────┐
                         │ Latent Bottleneck z │ ──▶ Exported as 128-dim Embeddings
                         │  (128 dimensions)   │     for Member 3 (Hybrid RF Model)
                         └──────────┬──────────┘
                                    │
                                    ▼
                         Linear(128 → 12) Output
                                    │
                                    ▼
                         12 Multi-Task Logits
```

- **Total Trainable Parameters:** **1,219,212**
- **Hard Parameter Sharing:** The lower representations capture cross-cutting biological mechanisms (cellular stress, electrophilic binding, DNA damage sensing) across both Nuclear Receptors (NR) and Stress Response (SR) pathways (Mayr et al., 2016).

### 3.2 Masked & Weighted Multi-Task Loss

Missing experimental assay labels ($m_{ij} = 0$) are masked out before reduction. Positive class weights $w_j$ penalize false negatives on rare active compounds:

$$\mathcal{L}_{\text{multitask}} = \frac{1}{\sum_{i=1}^B \sum_{j=1}^{12} m_{ij}} \sum_{i=1}^B \sum_{j=1}^{12} m_{ij} \cdot \left[ - w_j y_{ij} \log(\sigma(\hat{z}_{ij})) - (1 - y_{ij}) \log(1 - \sigma(\hat{z}_{ij})) \right]$$

Where:
- $m_{ij} \in \{0, 1\}$ is the boolean observation mask (1 if compound $i$ was tested in assay $j$).
- $w_j = \frac{N_{\text{neg, train}}}{N_{\text{pos, train}}}$ is the training positive class weight.
- $\sigma(\cdot)$ is the standard sigmoid activation function.

---

## 4. Holdout Test Set Performance (Split B, 782 Compounds)

Evaluated strictly on the held-out test partition with zero scaffold overlap with training or validation compounds:

| Endpoint | Target Description | Test Actives | Test Inactives | Test ROC-AUC | Test PR-AUC | F1-Score | Balanced Acc | MCC | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NR-AR** | Androgen Receptor | 39 | 669 | **0.7395** | **0.3996** | 0.4308 | 0.6705 | 0.4137 | 0.5385 | 0.3590 |
| **NR-AR-LBD** | Androgen Receptor (LBD) | 21 | 610 | **0.7468** | **0.4717** | 0.5000 | 0.7734 | 0.4847 | 0.4444 | 0.5714 |
| **NR-AhR** | Aryl Hydrocarbon Receptor | 96 | 533 | **0.7417** | **0.4005** | 0.4171 | 0.6543 | 0.3156 | 0.4286 | 0.4062 |
| **NR-Aromatase** | Aromatase Enzyme | 48 | 462 | **0.7459** | **0.3657** | 0.3269 | 0.6349 | 0.2519 | 0.3036 | 0.3542 |
| **NR-ER** | Estrogen Receptor | 82 | 457 | **0.5951** | **0.2522** | 0.2893 | 0.5812 | 0.1666 | 0.2987 | 0.2805 |
| **NR-ER-LBD** | Estrogen Receptor (LBD) | 33 | 616 | **0.6942** | **0.2152** | 0.2632 | 0.6247 | 0.2203 | 0.2326 | 0.3030 |
| **NR-PPAR-gamma** | PPAR-$\gamma$ | 29 | 548 | **0.7528** | **0.2336** | 0.3667 | 0.6714 | 0.3322 | 0.3548 | 0.3793 |
| **SR-ARE** | Antioxidant Response Element | 123 | 366 | **0.7334** | **0.4655** | 0.5040 | 0.6687 | 0.3338 | 0.4961 | 0.5122 |
| **SR-ATAD5** | ATAD5 Genotoxicity | 34 | 626 | **0.6654** | **0.1331** | 0.1918 | 0.5774 | 0.1451 | 0.1795 | 0.2059 |
| **SR-HSE** | Heat Shock Response | 52 | 526 | **0.6877** | **0.2309** | 0.1957 | 0.5571 | 0.1287 | 0.2250 | 0.1731 |
| **SR-MMP** | Mitochondrial Potential | 106 | 396 | **0.8326** | **0.5976** | 0.5727 | 0.7328 | 0.4535 | 0.5526 | 0.5943 |
| **SR-p53** | p53 Tumor Suppressor | 65 | 558 | **0.6802** | **0.2862** | 0.2500 | 0.5781 | 0.1809 | 0.2979 | 0.2154 |
| **MACRO AVERAGE** | — | **728** | **6,367** | **0.7179** | **0.3376** | **0.3590** | **0.6437** | **0.2856** | **0.3627** | **0.3629** |

---

## 5. Downstream Integration & Handoff Contract

### 5.1 For Member 3: Hybrid ML Model
Member 3 can consume the extracted 128-dimensional latent representations directly:

```python
import pandas as pd
import numpy as np
from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader

# 1. Load scaled continuous descriptors (Member 1)
loader = Tox21DataLoader()
X_desc_train, mids_desc = loader.load_features("descriptors_scaled", split="train") # (6259, 5)

# 2. Load learned molecular embeddings (Member 2)
df_emb = pd.read_parquet("phase2/member2_deep_learning_model/embeddings/train_embeddings.parquet")
assert df_emb["molecule_id"].tolist() == mids_desc  # Guaranteed alignment
z_train = df_emb.drop(columns=["molecule_id"]).to_numpy() # (6259, 128)

# 3. Form hybrid feature matrix:
X_hybrid_train = np.hstack([z_train, X_desc_train]) # (6259, 133)
print(f"Hybrid input space ready: {X_hybrid_train.shape}")
```

### 5.2 For Member 4: Benchmark Comparison
Member 4 evaluates all models on the identical holdout test split (`test_evaluation.csv`):
- Macro ROC-AUC: **0.7179**
- Macro PR-AUC: **0.3376**
- Peak Endpoints: **SR-MMP (0.8326)**, **NR-PPAR-gamma (0.7528)**, **NR-AR-LBD (0.7468)**, **NR-Aromatase (0.7459)**, **NR-AhR (0.7417)**.

### 5.3 For Member 5: Dashboard & Presentation
- **Training plots**: Available under `reports/plots/` (`loss_curve.png`, `val_metrics_curve.png`, `per_endpoint_roc_pr.png`, `training_curves_summary.png`).
- **Interactive Checkpoint**: `checkpoints/best_multitask_model.pt` can be loaded with PyTorch directly in the Streamlit application.

---

## 6. Verification & Automated Test Suite

Run the Member 2 test suite:

```bash
python3 -m unittest phase2/member2_deep_learning_model/tests/test_dl_model.py
```

### Automated Quality Gates Result:
- **Test 1:** Forward output shapes match specification (12 logits, 128 embeddings) — **PASS**
- **Test 2:** Sigmoid output probabilities strictly bounded in $[0, 1]$ — **PASS**
- **Test 3:** Masked loss invariance: unobserved label modification produces identical loss and gradient — **PASS**
- **Test 4:** Zero gradient propagation for completely masked positions — **PASS**
- **Test 5:** Checkpoint save and deserialization reproduces exact predictions — **PASS**
- **Test 6:** Real batch integration with Member 1's `Tox21DataLoader` — **PASS**
- **Test 7:** Embeddings artifacts integrity (6,259 train, 782 val, 782 test) — **PASS**
- **Test 8:** Saved production checkpoint loads and passes validation sanity check — **PASS**
- **Test 9:** Test evaluation report completeness (12 endpoints + MACRO AVERAGE) — **PASS**
- **Test 10:** Generated diagnostic plots verified on disk — **PASS**

---

## 7. How to Retrain or Run Inference

```bash
# Retrain model with default hyperparameters
python3 phase2/member2_deep_learning_model/train.py

# Retrain with custom hyperparameters
python3 phase2/member2_deep_learning_model/train.py --epochs 80 --lr 0.0008 --batch-size 64 --patience 15
```
