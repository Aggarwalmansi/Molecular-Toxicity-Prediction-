# Molecular Toxicity Prediction (Tox21)

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Hosted Live App](https://img.shields.io/badge/Streamlit-Live%20App-ff4b4b.svg)](https://molecular-toxicity-prediction.onrender.com/)

An end-to-end computational toxicology system benchmarking deep multitask neural architectures, tree ensembles, and hybrid representations across **12 biological pathways** (nuclear receptors and stress response pathways) using the **NIH Tox21 dataset**.

**Hosted Interactive Application:** [https://molecular-toxicity-prediction.onrender.com/](https://molecular-toxicity-prediction.onrender.com/)

---

## 1. Project Overview & Phase 2 Architecture

Phase 2 scales the project from initial single-endpoint baselines into an enterprise-grade, multi-stage deep learning and cheminformatics platform. The system is partitioned into modular, certified packages:

```
Molecular-Toxicity-Prediction-/
├── app.py                              # Interactive Streamlit web application
├── phase2/
│   ├── member1_data_pipeline/          # Certified scaffold split contract & masked data loaders
│   ├── member2_deep_learning_model/    # MultiTaskToxicNet & 128-dim latent representation learning
│   ├── member3_hybrid_model/           # Hybrid Random Forest on latent embeddings + descriptors
│   └── member4_evaluation/             # Comprehensive benchmark, 48 ablation models & error analysis
```

### Team Responsibilities & Core Modules

* **[Phase 2 — Member 1: Data Pipeline & Literature](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member1_data_pipeline/README.md):**
  * Freezes the authoritative Bemis–Murcko scaffold split (Split B: **6,259 train / 782 validation / 782 test**, total **7,823 compounds**).
  * Formulates the exact 12-endpoint observation mask and positive class weight tensors ($w_j = 5.71 \to 38.85$) strictly from training data.
  * Curates 4 peer-reviewed literature foundations justifying multi-task learning and scaffold split methodology.

* **[Phase 2 — Member 2: Deep Learning Multi-Task Model](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member2_deep_learning_model/README.md):**
  * Engineers `MultiTaskToxicNet` (1,219,212 parameters) predicting all 12 endpoints simultaneously via hard parameter sharing.
  * Implements `MaskedWeightedBCEWithLogitsLoss` strictly excluding 17.06% missing assay labels from gradient backpropagation.
  * Extracts 128-dimensional bottleneck embeddings ($z \in \mathbb{R}^{128}$) for downstream hybrid modeling.

* **[Phase 2 — Member 3: ML + DL Hybrid Model](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member3_hybrid_model/README.md):**
  * Trains 12 Random Forests on hybrid representations (128 latent embeddings + 5 standardized continuous descriptors = 133 features).
  * Evaluates descriptor ablation and honestly reports a negative result: hybrid modeling underperforms standalone models on compressed representations.

* **[Phase 2 — Member 4: Comprehensive Evaluation & Systematic Ablations](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member4_evaluation/README.md):**
  * Trains and evaluates **48 PyTorch model checkpoints** across 3 random seeds (**42, 21, 7**) under zero test-set leakage.
  * **Dropout Ablation:** Compares $p=0.35/0.175$ vs. $p=0.0$, demonstrating that dropout prevents fingerprint bit co-adaptation.
  * **Class-Weighting Ablation:** Shows that unweighted loss collapses active compound recall from 64.16% to 17.95% (surging false negatives from 249 to 558).
  * **MultiTask vs. Single-Task Learning:** Trains 36 independent single-task neural networks; proves MultiTask learning achieves a **+0.0452 ROC-AUC advantage** via shared biological representations.
  * **Toxicological Error Analysis:** Sweeps validation optimal $F_1$ decision thresholds; discovers that missed toxic compounds are heavier (+52.7 g/mol) and more polar (+12.4 Å² PSA).
  * **Publication Artifacts:** Generates **18 high-resolution 300 DPI figures**, 10 CSV result matrices, and 4 detailed scientific reports.

---

## 2. Master Architectural Benchmark (Holdout Test Split B)

All models were evaluated on the **exact same 782 holdout test molecules** with zero scaffold overlap against training or validation data:

| Model Architecture | Input Features | Holdout ROC-AUC | Holdout PR-AUC | $F_1$-Score | Balanced Accuracy | MCC | Key Takeaway |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standalone Random Forest** | 2,053 (2,048 Morgan + 5 Descriptors) | **0.7541** | **0.3962** | **0.3969** | **0.6684** | **0.3141** | **Best overall benchmark** on sparse binary fingerprints |
| **MultiTask Deep Learning** | 2,053 $\to$ 128 bottleneck $\to$ 12 heads | **0.7472 ± 0.0065** | **0.3286 ± 0.0145** | **0.3049 ± 0.0253** | **0.6974 ± 0.0253** | **0.2543 ± 0.0177** | Multi-task shared trunk transfers biological representations |
| **Single-Task Deep Learning** | 2,053 per endpoint (12 independent nets) | **0.7020 ± 0.0076** | **0.3197 ± 0.0084** | 0.3335 ± 0.0167 | 0.6485 ± 0.0040 | 0.2500 ± 0.0118 | Lacks cross-task transfer; fails on data-poor assays ($\Delta = -0.0452$) |
| **ML+DL Hybrid Model** | 133 (128 embeddings + 5 descriptors) | **0.6753** | **0.3009** | 0.2933 | 0.5921 | 0.2712 | Information-loss penalty from 128-dim bottleneck compression |

---

## 3. Per-Endpoint Benchmark Breakdown

Comparison of holdout test ROC-AUC across all 12 Tox21 pathways:

| Endpoint | Target Pathway | Standalone RF ROC-AUC | MultiTask DL ROC-AUC | Single-Task DL ROC-AUC | ML+DL Hybrid ROC-AUC | Active Prevalence |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **NR-AR** | Androgen Receptor | 0.7387 | **0.7395** | 0.7078 | 0.6495 | 5.51% |
| **NR-AR-LBD** | Androgen Receptor (LBD) | **0.8194** | 0.7468 | 0.7790 | 0.8138 | 3.33% |
| **NR-AhR** | Aryl Hydrocarbon Receptor | **0.8274** | 0.7417 | 0.7773 | 0.6939 | 15.26% |
| **NR-Aromatase** | Aromatase Enzyme | 0.7398 | **0.7459** | 0.6748 | 0.6859 | 9.41% |
| **NR-ER** | Estrogen Receptor | **0.6335** | 0.5951 | 0.6114 | 0.5808 | 15.21% |
| **NR-ER-LBD** | Estrogen Receptor (LBD) | **0.7294** | 0.6942 | 0.6729 | 0.6390 | 5.08% |
| **NR-PPAR-gamma** | PPAR-$\gamma$ | **0.8078** | 0.7528 | 0.6715 | 0.6746 | 5.03% |
| **SR-ARE** | Antioxidant Response | 0.7196 | **0.7334** | 0.7312 | 0.7200 | 25.15% |
| **SR-ATAD5** | Genotoxicity / DNA Damage | **0.7376** | 0.6654 | 0.6665 | 0.6302 | 5.15% |
| **SR-HSE** | Heat Shock Response | **0.7372** | 0.6877 | 0.6669 | 0.6344 | 9.00% |
| **SR-MMP** | Mitochondrial Potential | 0.8274 | **0.8326** | 0.7935 | 0.7772 | 21.12% |
| **SR-p53** | p53 Tumor Suppressor | **0.7317** | 0.6802 | 0.6714 | 0.6040 | 10.43% |
| **MACRO AVERAGE** | — | **0.7541** | **0.7179** | **0.7020** | **0.6753** | **10.80%** |

---

## 4. Key Scientific Insights & Systematic Ablations

1. **Why Random Forest Beats Deep Learning on Fingerprints:**  
   Morgan fingerprints (2,048 bits) are sparse and orthogonal. Decision trees execute axis-aligned orthogonal splits directly on informative substructure bits without requiring gradient optimization over high-dimensional continuous weights.
2. **Why the Hybrid Model Fails (The Bottleneck Penalty):**  
   Compressing 2,053 features into 128 dimensions creates an irreversible information loss penalty. Decision trees can no longer cleanly split on individual functional groups because they are blended into continuous linear combinations.
3. **MultiTask Learning Transfers Representations (+0.0452 ROC-AUC):**  
   Single-task models overfit on endpoints with low active counts (e.g., `NR-PPAR-gamma` with only 133 train actives). MultiTask learning leverages universal cellular stress signals learned on data-rich pathways (`SR-ARE`, `SR-MMP`) to regularize data-poor pathways.
4. **Positive Class Weighting is Non-Negotiable for Safety Screening:**  
   Without class weighting ($w_j = 1.0$), active recall collapses from **64.16% down to 17.95%**, causing **308 hazardous compounds to be missed as false negatives**.
5. **Physicochemical Failure Mode Analysis:**  
   Profiling prediction errors with continuous RDKit descriptors revealed that missed active compounds (false negatives) are **heavier (+52.7 g/mol)** and **more polar (+12.4 Å² PSA)** with significantly more hydrogen-bonding groups than detected active compounds.

---

## 5. Quickstart & Execution

### 5.1 Environment Setup
```bash
# Clone the repository
git clone https://github.com/Aggarwalmansi/Molecular-Toxicity-Prediction-.git
cd Molecular-Toxicity-Prediction-

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Run the Interactive Streamlit App
```bash
streamlit run app.py
```

### 5.3 Run Member Verification & Evaluation Suites
```bash
# Member 1 Data Pipeline Tests (10/10 PASS)
.venv/bin/python -m unittest phase2/member1_data_pipeline/tests/test_data_pipeline.py

# Member 2 Deep Learning Tests (10/10 PASS)
.venv/bin/python -m unittest phase2/member2_deep_learning_model/tests/test_dl_model.py

# Member 4 Comprehensive Evaluation Tests (10/10 PASS)
.venv/bin/python -m unittest phase2/member4_evaluation/tests/test_evaluation.py

# Run Full Member 4 Ablation & Evaluation Orchestrator
.venv/bin/python -m phase2.member4_evaluation.run_all
```

---

## 6. Detailed Member Documentation

For in-depth mathematical formulations, code APIs, hyperparameter configurations, and viva guides:
- [Phase 2 — Member 1: Data Pipeline & Literature](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member1_data_pipeline/README.md)
- [Phase 2 — Member 2: Deep Learning Model](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member2_deep_learning_model/README.md)
- [Phase 2 — Member 3: ML + DL Hybrid Model](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member3_hybrid_model/README.md)
- [Phase 2 — Member 4: Evaluation & Ablation Studies](file:///Users/aditisingh1619/Desktop/Molecular-Toxicity-Prediction-/phase2/member4_evaluation/README.md)
