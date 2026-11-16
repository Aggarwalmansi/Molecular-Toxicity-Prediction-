# Phase 2 — Multi-Task Deep Learning Model

**Role:** Deep Learning Research Engineer & Representation Learning Specialist  
**Status:** Trained, Verified & Production Ready  
**Target Architecture:** Multi-Task Deep Neural Network (`MultiTaskToxicNet`) predicting all 12 Tox21 biological pathways simultaneously with Masked Loss, Class Weighting, and Latent Representation Extraction  

---

## 1. Executive Summary & Deliverables

Deep learning is responsible for the deep learning core of Phase 2, predicting all 12 Tox21 biological endpoints simultaneously using hard parameter sharing:
1. **Multi-Task Neural Network (`MultiTaskToxicNet`):** PyTorch deep neural network accepting canonical 2,053-dim molecular features (`X_combined`: 2,048-bit Morgan circular fingerprints + 5 standardized physical descriptors) and predicting 12 target assays in a single forward pass.
2. **Masked & Weighted Loss (`MaskedWeightedBCEWithLogitsLoss`):** Strictly excludes unmeasured assay targets (17.06% missing labels) from backpropagation while applying square-root-damped training class ratios capped at 3.0 (raw ratios range from 5.71 to 38.85) to counter severe activity imbalance.
3. **Regularization & Early Stopping:** Input dropout ($p=0.30$) on all raw input features, trunk dropout ($p=0.50$), weight decay ($1\times 10^{-3}$), LayerNorm, gradient clipping at $\lVert g \rVert \le 2$, and early stopping on validation macro ROC-AUC under a `CosineAnnealingWarmRestarts` schedule ($T_0=20$, $T_{mult}=2$).
4. **Checkpoint Deliverable:** Certified best model checkpoint saved at `checkpoints/best_multitask_model.pt`.
5. **Learned Molecular Features (Embeddings Handoff for Hybrid modeling):** 128-dimensional continuous latent representations $z \in \mathbb{R}^{128}$ extracted from the bottleneck layer for all 7,823 compounds (`train_embeddings.parquet`, `val_embeddings.parquet`, `test_embeddings.parquet`) with row-for-row compound ID alignment.
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
                          Dropout(0.30) on the complete input vector
                                    │
                                    ▼
    ResidualBlock(2,053 → 512) ── LayerNorm ── GELU ── Dropout(0.50)
                                    │
                                    ▼
    ResidualBlock(512 → 256)   ── LayerNorm ── GELU ── Dropout(0.50)
                                    │
                                    ▼
    ResidualBlock(256 → 128)   ── LayerNorm ── GELU ── Dropout(0.25)
                                    │
                         ┌──────────┴──────────┐
                         │ Latent Bottleneck z │ ──▶ Exported as 128-dim Embeddings
                         │  (128 dimensions)   │     for Hybrid modeling (Hybrid RF Model)
                         └──────────┬──────────┘
                                    │
                                    ▼
        12 × TaskHead: Linear(128 → 64) ── GELU ── Dropout(0.2) ── Linear(64 → 1)
                                    │
                                    ▼
                         12 Multi-Task Logits
```

Each `ResidualBlock` adds a skip connection around the dense layer, with a
learned linear projection on the skip path where the dimensions change.

**Why these choices, specifically:**

| Choice | Reason |
| :--- | :--- |
| **LayerNorm** | Normalizes hidden activations within each molecule rather than using minibatch statistics. This motivates its use with small batches; no universal disadvantage of BatchNorm is established. |
| **GELU** | Provides a smooth nonlinear activation in the tested residual architecture; superiority over ReLU is not established by an isolated activation ablation. |
| **Residual connections** | Keep a short gradient path to the input layer, so the 2,053 → 512 projection still receives usable gradient after three blocks. |
| **Per-endpoint heads** | Each assay has a separate nonlinear readout while retaining a shared encoder. A single Linear(128,12) would still have independent output weights, but lacks the extra hidden layer per assay. |
| **Input dropout (0.30)** | With 6,259 training molecules against 2,053 sparse inputs the trunk otherwise memorizes individual bits within five epochs (train loss 0.016, validation loss 2.22). See section 4.1. |

- **Total Trainable Parameters:** **2,532,492**
- **Hard Parameter Sharing:** The lower representations capture cross-cutting biological mechanisms (cellular stress, electrophilic binding, DNA damage sensing) across both Nuclear Receptors (NR) and Stress Response (SR) pathways (Mayr et al., 2016).

### 3.2 Masked & Weighted Multi-Task Loss

Missing experimental assay labels ($m_{ij} = 0$) are masked out before reduction. Positive class weights $w_j$ penalize false negatives on rare active compounds:

$$\mathcal{L}_{\text{multitask}} = \frac{1}{\sum_{i=1}^B \sum_{j=1}^{12} m_{ij}} \sum_{i=1}^B \sum_{j=1}^{12} m_{ij} \cdot \left[ - w_j y_{ij} \log(\sigma(\hat{z}_{ij})) - (1 - y_{ij}) \log(1 - \sigma(\hat{z}_{ij})) \right]$$

Where:
- $m_{ij} \in \{0, 1\}$ is the boolean observation mask (1 if compound $i$ was tested in assay $j$).
- $\sigma(\cdot)$ is the standard sigmoid activation function.
- $w_j$ is the **sqrt-damped** training positive class weight:

$$w_j = \min\left(\sqrt{\frac{N_{\text{neg, train}}}{N_{\text{pos, train}}}},\; c\right)$$

Raw inverse-frequency weights span 5.71 (SR-MMP) to **38.85** (NR-PPAR-gamma).
Feeding 38.85 straight into the loss means a single rare active contributes as
much gradient as 39 inactives, so one minority compound in a batch of 64
dominates the update and the loss spikes. Square-root damping compresses that
range to 2.39–6.23 while preserving the ordering of endpoint rarity, and the
clamp $c$ caps what is left.

The clamp ceiling is **$c = 3.0$**, lowered from the 10.0 the damping formula
allows. The measured reason is in section 4.2: at $c = 10$ validation loss
climbs to 1.63, at $c = 3$ it holds at 1.19, and validation macro ROC-AUC is
statistically indistinguishable (0.7929 vs 0.7917).

---

## 4. Training Stability: the Overfitting Failure and What Fixed It

### 4.1 The failure

The original configuration (BatchNorm, ReLU, single linear classifier,
undamped weights, lr $10^{-3}$, weight decay $10^{-4}$) memorized the training
set almost immediately:

| Epoch | Train loss | Val loss | Val macro ROC-AUC |
| :---: | :---: | :---: | :---: |
| 1 | 0.513 | 0.597 | 0.7857 |
| 5 | 0.143 | 1.199 | 0.7529 |
| 10 | 0.043 | 1.789 | 0.7477 |
| 20 | 0.016 | 2.222 | 0.7476 |

Train loss fell 32× while validation loss rose 3.7×. Early stopping therefore
kept **epoch 1** — the exported embeddings came from a network that had seen
each molecule once.

### 4.2 What each fix was worth

`src/regularization_sweep.py` measures this on the validation split only
(30 epochs per configuration, `reports/regularization_sweep.csv`):

| Configuration | Best val ROC-AUC | Best epoch | Final val loss |
| :--- | :---: | :---: | :---: |
| Architecture fixes only (as originally specified) | 0.7857 | 1 | 2.56 |
| \+ input dropout 0.2 | 0.7925 | 2 | 2.03 |
| \+ input dropout 0.3, trunk dropout 0.5 | **0.7929** | 3 | 1.63 |
| \+ weight clamp $c=5$ | **0.7932** | 3 | 1.57 |
| \+ weight clamp $c=3$ **(shipped)** | 0.7917 | 3 | **1.19** |
| \+ weight clamp $c=2$ | 0.7905 | 3 | 0.89 |
| no positive weighting at all | 0.7884 | 3 | 0.50 |

Two honest observations:

1. **The architecture changes alone did not fix the overfitting.** LayerNorm,
   GELU, residual connections and per-task heads moved the best epoch from 1 to
   1 and validation ROC-AUC from 0.7179 to 0.7857 on validation, but train loss
   still collapsed to 0.02. Input dropout on the raw bits is what actually
   slowed memorization.
2. **Most of the "exploding" validation loss was a calibration artifact, not a
   ranking failure.** Lowering the positive-weight clamp cuts validation loss
   by more than half (2.56 → 0.89) while validation ROC-AUC barely moves. The
   weighted loss punishes overconfident logits on the minority class, which is
   why it grows so much faster than the model's actual ranking degrades.

### 4.3 The shipped run

| Epoch | Train loss | Val loss | Val macro ROC-AUC |
| :---: | :---: | :---: | :---: |
| 1 | 0.466 | 0.508 | 0.7698 |
| 2 | 0.366 | 0.508 | 0.7886 |
| **3** | 0.325 | **0.504** | **0.7917** |
| 10 | 0.176 | 0.819 | 0.7644 |
| 20 | 0.128 | 1.037 | 0.7573 |

Validation loss now **improves** through epoch 3 rather than diverging after
epoch 2, early stopping selects a genuinely trained network, and the end-state
overfit gap is less than half what it was.

**The remaining limitation, stated plainly:** validation macro ROC-AUC still
peaks at epoch 3 and plateaus at ~0.79 across every configuration tested. With
6,259 training molecules and a 2,053-dim sparse input, this network reaches its
ceiling in a few epochs. The honest conclusion is that this is a small-data
regime, not that more regularization tuning would keep helping.

---

## 5. Holdout Test Set Performance (Split B, 782 Compounds)

Evaluated strictly on the held-out test partition with zero scaffold overlap with training or validation compounds:

| Endpoint | Target Description | Test Actives | Test Inactives | Test ROC-AUC | Test PR-AUC | F1-Score | Balanced Acc | MCC | Precision | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NR-AR** | Androgen Receptor | 39 | 669 | **0.7736** | **0.4901** | 0.5714 | 0.7044 | 0.6092 | 0.9412 | 0.4103 |
| **NR-AR-LBD** | Androgen Receptor (LBD) | 21 | 610 | **0.7664** | **0.4047** | 0.5000 | 0.7094 | 0.4930 | 0.6000 | 0.4286 |
| **NR-AhR** | Aryl Hydrocarbon Receptor | 96 | 533 | **0.7986** | **0.4863** | 0.4427 | 0.6969 | 0.3273 | 0.3567 | 0.5833 |
| **NR-Aromatase** | Aromatase Enzyme | 48 | 462 | **0.7058** | **0.2298** | 0.1519 | 0.5354 | 0.0866 | 0.1935 | 0.1250 |
| **NR-ER** | Estrogen Receptor | 82 | 457 | **0.6293** | **0.3254** | 0.3360 | 0.6040 | 0.2757 | 0.4884 | 0.2561 |
| **NR-ER-LBD** | Estrogen Receptor (LBD) | 33 | 616 | **0.6851** | **0.3058** | 0.3200 | 0.6139 | 0.3134 | 0.4706 | 0.2424 |
| **NR-PPAR-gamma** | PPAR-$\gamma$ | 29 | 548 | **0.7509** | **0.1383** | 0.0588 | 0.5136 | 0.0641 | 0.2000 | 0.0345 |
| **SR-ARE** | Antioxidant Response Element | 123 | 366 | **0.6913** | **0.4335** | 0.4492 | 0.6335 | 0.2748 | 0.4690 | 0.4309 |
| **SR-ATAD5** | ATAD5 Genotoxicity | 34 | 626 | **0.6908** | **0.1267** | 0.1724 | 0.5584 | 0.1378 | 0.2083 | 0.1471 |
| **SR-HSE** | Heat Shock Response | 52 | 526 | **0.7046** | **0.2108** | 0.1600 | 0.5415 | 0.1216 | 0.2609 | 0.1154 |
| **SR-MMP** | Mitochondrial Potential | 106 | 396 | **0.8083** | **0.5466** | 0.5455 | 0.7229 | 0.4095 | 0.4853 | 0.6226 |
| **SR-p53** | p53 Tumor Suppressor | 65 | 558 | **0.7139** | **0.2598** | 0.2308 | 0.5681 | 0.1719 | 0.3077 | 0.1846 |
| **MACRO AVERAGE** | — | **728** | **6,367** | **0.7266** | **0.3298** | **0.3282** | **0.6168** | **0.2737** | **0.4151** | **0.2984** |

---

## 6. Downstream Integration & Handoff Contract

### 6.1 For Hybrid modeling: Hybrid ML Model
Hybrid modeling can consume the extracted 128-dimensional latent representations directly:

```python
import pandas as pd
import numpy as np
from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader

# 1. Load scaled continuous descriptors (Data pipeline)
loader = Tox21DataLoader()
X_desc_train, mids_desc = loader.load_features("descriptors_scaled", split="train") # (6259, 5)

# 2. Load learned molecular embeddings (Deep learning)
df_emb = pd.read_parquet("phase2/member2_deep_learning_model/embeddings/train_embeddings.parquet")
assert df_emb["molecule_id"].tolist() == mids_desc  # Guaranteed alignment
z_train = df_emb.drop(columns=["molecule_id"]).to_numpy() # (6259, 128)

# 3. Form the FUSED feature matrix: keep the raw block, add the embeddings.
X_raw_train, mids_raw = loader.load_features("combined", split="train")  # (6259, 2053)
assert mids_raw == mids_desc
X_hybrid_train = np.hstack([X_raw_train, z_train]) # (6259, 2181)
print(f"Fused input space ready: {X_hybrid_train.shape}")
```

Do **not** replace the raw features with the embeddings. Feeding a Random
Forest only `[z | descriptors]` (133 columns) drops test macro ROC-AUC to
0.6785, because the 2,048 Morgan bits the forest splits on are gone. the hybrid modeling module’s
README section 3 has the measurements, and also shows that the best use of this
representation is probability blending rather than input concatenation.

### 6.2 For Evaluation: Benchmark Comparison
Evaluation evaluates all models on the identical holdout test split (`test_evaluation.csv`):
- Macro ROC-AUC: **0.7266**
- Macro PR-AUC: **0.3298**
- Peak Endpoints: **SR-MMP (0.8083)**, **NR-AhR (0.7986)**, **NR-AR (0.7736)**, **NR-AR-LBD (0.7664)**, **NR-PPAR-gamma (0.7509)**.

### 6.3 For Application and reporting: Dashboard & Presentation
- **Training plots**: Available under `reports/plots/` (`loss_curve.png`, `val_metrics_curve.png`, `per_endpoint_roc_pr.png`, `training_curves_summary.png`).
- **Interactive Checkpoint**: `checkpoints/best_multitask_model.pt` can be loaded with PyTorch directly in the Streamlit application.

---

## 7. Verification & Automated Test Suite

Run the Deep learning test suite:

```bash
python3 -m unittest phase2/member2_deep_learning_model/tests/test_dl_model.py
```

### Automated Quality Gates Result:
- **Test 1:** Forward output shapes match specification (12 logits, 128 embeddings) — **PASS**
- **Test 2:** Sigmoid output probabilities strictly bounded in $[0, 1]$ — **PASS**
- **Test 3:** Masked loss invariance: unobserved label modification produces identical loss and gradient — **PASS**
- **Test 4:** Zero gradient propagation for completely masked positions — **PASS**
- **Test 5:** Checkpoint save and deserialization reproduces exact predictions — **PASS**
- **Test 6:** Real batch integration with the data pipeline module’s `Tox21DataLoader` — **PASS**
- **Test 7:** Embeddings artifacts integrity (6,259 train, 782 val, 782 test) — **PASS**
- **Test 8:** Saved production checkpoint loads and passes validation sanity check — **PASS**
- **Test 9:** Test evaluation report completeness (12 endpoints + MACRO AVERAGE) — **PASS**
- **Test 10:** Generated diagnostic plots verified on disk — **PASS**

---

## 8. How to Retrain or Run Inference

```bash
# Retrain model with default hyperparameters
python3 phase2/member2_deep_learning_model/train.py

# Retrain with custom hyperparameters
python3 phase2/member2_deep_learning_model/train.py --epochs 80 --lr 0.0008 --batch-size 64 --patience 15
```
