# Molecular Toxicity Prediction (Tox21)

An end-to-end machine learning system for predicting chemical toxicity across 12 biological pathways (Nuclear Receptors & Stress Response) using the Tox21 dataset.

**Hosted Live App:** [https://molecular-toxicity-prediction.onrender.com/](https://molecular-toxicity-prediction.onrender.com/)

---

## Repository Overview

1. **Data Curation & Scaffold Split**:
   - 7,823 validated RDKit chemical compounds.
   - Strict 80/10/10 **Bemis–Murcko scaffold split** (zero scaffold leakage) ensuring realistic evaluation on unseen chemical cores.
2. **Feature Engineering**:
   - **Morgan Fingerprints**: 2,048-bit circular fingerprints (radius=2 / ECFP4) capturing local 2D chemical substructures.
   - **Physicochemical Descriptors**: 5 continuous molecular descriptors (`MolWt`, `MolLogP`, `TPSA`, `NumHDonors`, `NumHAcceptors`) standardized on training data.
   - **Variance Filtering**: Unsupervised filtering ($\text{Var} \ge 0.005$) reducing sparse bits to 1,011 high-variance Morgan features.
3. **Principal Component Analysis (PCA)**:
   - Dimensionality reduction extracting the top 50 principal components from Morgan and combined feature spaces to evaluate linear variance and reduce collinearity.
4. **Model Training**:
   - Baseline classifiers trained per endpoint: **Random Forest** (best performer), **Logistic Regression**, and **Decision Tree**.
5. **Interactive Streamlit App**:
   - Web application with live SMILES input, 2D molecular structure rendering, and real-time toxicity prediction across all 12 pathways.

---

## Overall Model Performance

Macro-averaged evaluation metrics across all 12 Tox21 endpoints on the scaffold test set:

| Model                         | Test ROC-AUC (Mean ± Std) | Test PR-AUC (Mean ± Std) |          F1-Score          |     Balanced Accuracy     |            MCC            |
| :---------------------------- | :------------------------: | :------------------------: | :------------------------: | :------------------------: | :------------------------: |
| **Random Forest**       | **0.7541 ± 0.0570** | **0.3962 ± 0.1323** | **0.3969 ± 0.1070** | **0.6684 ± 0.0507** | **0.3141 ± 0.1177** |
| **Logistic Regression** |      0.7152 ± 0.0596      |      0.2949 ± 0.1319      |      0.3268 ± 0.1355      |      0.6358 ± 0.0761      |      0.2413 ± 0.1233      |
| **Decision Tree**       |      0.6428 ± 0.0597      |      0.2091 ± 0.0957      |      0.2629 ± 0.1182      |      0.6295 ± 0.0592      |      0.1616 ± 0.0859      |

---

## Per-Endpoint Test Metrics

Performance breakdown across all 12 nuclear receptor (NR) and stress response (SR) assays:

| Endpoint                | Target Description               |    RF ROC-AUC    |    RF PR-AUC    | LR ROC-AUC | DT ROC-AUC |
| :---------------------- | :------------------------------- | :--------------: | :--------------: | :--------: | :--------: |
| **NR-AR**         | Androgen Receptor                | **0.7387** | **0.4740** |   0.7247   |   0.6839   |
| **NR-AR-LBD**     | Androgen Receptor (LBD)          | **0.8194** | **0.4527** |   0.7909   |   0.7452   |
| **NR-AhR**        | Aryl Hydrocarbon Receptor        | **0.8274** | **0.5555** |   0.7758   |   0.6769   |
| **NR-Aromatase**  | Aromatase Enzyme                 | **0.7398** | **0.4182** |   0.7098   |   0.6340   |
| **NR-ER**         | Estrogen Receptor                | **0.6335** | **0.3449** |   0.5823   |   0.5342   |
| **NR-ER-LBD**     | Estrogen Receptor (LBD)          | **0.7294** | **0.3724** |   0.7108   |   0.6383   |
| **NR-PPAR-gamma** | PPAR-gamma                       | **0.8078** | **0.3267** |   0.7569   |   0.6402   |
| **SR-ARE**        | Antioxidant Response Element     | **0.7196** | **0.4975** |   0.7181   |   0.6185   |
| **SR-ATAD5**      | ATAD5 Genotoxicity               | **0.7376** | **0.1374** |   0.6812   |   0.5989   |
| **SR-HSE**        | Heat Shock Response              | **0.7372** | **0.2502** |   0.6775   |   0.6828   |
| **SR-MMP**        | Mitochondrial Membrane Potential | **0.8274** | **0.6076** |   0.7876   |   0.6994   |
| **SR-p53**        | p53 Tumor Suppressor             | **0.7317** | **0.3169** |   0.6669   |   0.5612   |

---

## Error Analysis & Key Insights

- **Scaffold Generalization Gap**: Random splitting produces an artificial inflation of **+0.0840 macro ROC-AUC** (up to **+0.16** on `SR-p53` and `NR-ER`). This confirms models memorize scaffold cores under random split, proving that scaffold splitting is essential for real-world generalization.
- **High-Performing Endpoints (`SR-MMP`, `NR-AhR`)**: Driven by clear physical drivers (lipophilicity/charge for MMP, planar aromatic rings for AhR) well-captured by descriptors and Morgan bits.
- **Challenging Endpoints (`NR-ER`, `SR-ATAD5`, `NR-PPAR-gamma`)**: Limited by extreme class imbalance (<5% actives), high mechanistic diversity (multiple DNA damage pathways in ATAD5), and out-of-distribution scaffold cores in test sets.

---

## Quickstart

Run the interactive dashboard locally:

```bash
# Clone the repository
git clone https://github.com/Aggarwalmansi/Molecular-Toxicity-Prediction-.git
cd Molecular-Toxicity-Prediction-

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```
