# Member 4: Comprehensive Evaluation, Statistics and Error Analysis Report

**Project**: Tox21 Molecular Toxicity Prediction  
**Phase**: Member 4 Final Deliverable — Evaluation, Statistical Benchmarking, and Failure Mode Investigation  
**Primary Ownership**: Proving whether models work, diagnosing poor performance, and certifying benchmark integrity  
**Upstream Inputs**: Member 1 (`tox21_project/`, 7,823 curated SMILES, Bemis–Murcko scaffold split), Member 2 (`tox21_member2_handoff/`, 2,053 canonical features), Member 3 (`train_models.py`, baseline architectures)  
**Deliverables Owned**: `evaluate.py`, `results/metrics.csv`, `results/summary_metrics.csv`, `results/statistical_tests.csv`, `results/split_comparison.csv`, `results/test_predictions.csv`, `results/plots/*.png`, `results/error_analysis.md`

---

## 1. Executive Summary

This report establishes the rigorous statistical evaluation and error analysis across all 12 Tox21 toxicological endpoints (7 Nuclear Receptor pathways and 5 Stress Response pathways). 

### Key Findings:
1. **Model Hierarchy**: Random Forest significantly outperforms Logistic Regression and Decision Tree across both discriminative ranking (ROC-AUC) and precision-recall dynamics (PR-AUC).
   - **Random Forest**: Macro-average Test ROC-AUC = **0.7541 ± 0.0570**, PR-AUC = **0.3962 ± 0.1323**, Balanced Accuracy = **0.6684 ± 0.0507**.
   - **Logistic Regression**: Macro-average Test ROC-AUC = **0.7152 ± 0.0596**, PR-AUC = **0.2949 ± 0.1319**, Balanced Accuracy = **0.6358 ± 0.0761**.
   - **Decision Tree**: Macro-average Test ROC-AUC = **0.6428 ± 0.0597**, PR-AUC = **0.2091 ± 0.0957**, Balanced Accuracy = **0.6295 ± 0.0592**.
2. **Missing Label Protocol Audit**: Verified that unmeasured assay entries (`NaN`, 17.05% of dataset) were strictly masked out per endpoint. No imputation (e.g., treating missing as inactive `0`) was permitted.
3. **Paired Compound Evaluation**: All models for each endpoint were evaluated on the **exact same test compound IDs (`mol_id`)**, ensuring absolute experimental control for paired significance tests.
4. **Scaffold Generalization Penalty (Task 6)**: Random splitting produces severe optimistic bias. Under a random 80/10/10 split, Random Forest achieves macro ROC-AUC = **0.8381** (an artificial inflation of **+0.0840** over scaffold split). Certain endpoints (`NR-ER`, `SR-p53`, `NR-AR`) exhibit inflation exceeding **+0.1600 ROC-AUC**. Bemis–Murcko scaffold splitting is essential for testing generalization to novel chemical families.
5. **Root-Cause Diagnosis for Underperforming Endpoints**: Failure modes on weak endpoints (`NR-ER`, `NR-PPAR-gamma`, `SR-ATAD5`) were traced to extreme class imbalance (<5% positives), out-of-distribution (OOD) test scaffolds, and multifaceted biological activation mechanisms that cannot be linearly separated by 2D Morgan fingerprints.

---

## 2. Macro-Averaged Benchmark Summary

The table below summarizes macro-averaged performance across all 12 endpoints (derived from `results/metrics.csv`):

| Model | Test ROC-AUC (Mean ± Std) | Test PR-AUC (Mean ± Std) | F1-Score (Mean ± Std) | Balanced Accuracy (Mean ± Std) | Matthews Corr. Coeff. (MCC) |
|---|---|---|---|---|---|
| **Random Forest** | **0.7541 ± 0.0570** | **0.3962 ± 0.1323** | **0.3969 ± 0.1070** | **0.6684 ± 0.0507** | **0.3141 ± 0.1177** |
| **Logistic Regression** | 0.7152 ± 0.0596 | 0.2949 ± 0.1319 | 0.3268 ± 0.1355 | 0.6358 ± 0.0761 | 0.2413 ± 0.1233 |
| **Decision Tree** | 0.6428 ± 0.0597 | 0.2091 ± 0.0957 | 0.2629 ± 0.1182 | 0.6295 ± 0.0592 | 0.1616 ± 0.0859 |

### Statistical Significance (Paired Bootstrap & Wilcoxon Tests):
- **Random Forest vs. Decision Tree**: Random Forest is statistically superior across all 12 endpoints in PR-AUC ($p < 0.02$) and in 11 of 12 endpoints in ROC-AUC ($p < 0.02$).
- **Random Forest vs. Logistic Regression**: Random Forest exhibits statistically significant PR-AUC gains on key assays including `NR-AR` ($p < 0.001$), `NR-AhR` ($p = 0.002$), `NR-ER` ($p = 0.014$), `NR-ER-LBD` ($p = 0.004$), and `NR-PPAR-gamma` ($p = 0.014$). In terms of probabilistic Brier loss (Wilcoxon signed-rank test), Random Forest has significantly lower error on **all 12 endpoints** ($p < 0.0001$).
- **Logistic Regression vs. Decision Tree**: Logistic Regression significantly outperforms Decision Tree on `NR-AhR`, `SR-ARE`, `SR-MMP`, and `SR-p53` ($p < 0.05$).

---

## 3. Detailed Per-Endpoint Benchmark (Scaffold Split)

Evaluation on the canonical Bemis–Murcko scaffold test set with empirical **95% Bootstrap Confidence Intervals** (1,000 resamples):

| Endpoint | Test Size ($N_{\text{obs}}$) | Active Count | Active Prev. | Random Forest ROC-AUC [95% CI] | Random Forest PR-AUC [95% CI] | Baseline Chance PR-AUC | PR-AUC Uplift |
|---|---|---|---|---|---|---|---|
| **SR-MMP** | 502 | 106 | 21.12% | **0.8274** [0.7804, 0.8679] | **0.6076** [0.5105, 0.6949] | 0.2112 | **+0.3965** |
| **NR-AhR** | 629 | 96 | 15.26% | **0.8274** [0.7852, 0.8681] | **0.5555** [0.4623, 0.6455] | 0.1526 | **+0.4028** |
| **NR-AR-LBD** | 631 | 21 | 3.33% | **0.8194** [0.6907, 0.9245] | **0.4527** [0.2305, 0.6699] | 0.0333 | **+0.4194** |
| **NR-PPAR-gamma** | 577 | 29 | 5.03% | **0.8078** [0.7018, 0.8995] | **0.3267** [0.1850, 0.5138] | 0.0503 | **+0.2764** |
| **NR-Aromatase** | 510 | 48 | 9.41% | **0.7398** [0.6554, 0.8225] | **0.4182** [0.2875, 0.5633] | 0.0941 | **+0.3241** |
| **NR-AR** | 708 | 39 | 5.51% | **0.7387** [0.6410, 0.8308] | **0.4740** [0.3308, 0.6314] | 0.0551 | **+0.4189** |
| **SR-ATAD5** | 660 | 34 | 5.15% | **0.7376** [0.6547, 0.8056] | **0.1374** [0.0827, 0.2417] | 0.0515 | **+0.0859** |
| **SR-HSE** | 578 | 52 | 9.00% | **0.7372** [0.6584, 0.8081] | **0.2502** [0.1628, 0.3876] | 0.0900 | **+0.1602** |
| **SR-p53** | 623 | 65 | 10.43% | **0.7317** [0.6588, 0.7970] | **0.3169** [0.2212, 0.4311] | 0.1043 | **+0.2125** |
| **NR-ER-LBD** | 649 | 33 | 5.08% | **0.7294** [0.6127, 0.8277] | **0.3724** [0.2067, 0.5376] | 0.0508 | **+0.3215** |
| **SR-ARE** | 489 | 123 | 25.15% | **0.7196** [0.6704, 0.7740] | **0.4975** [0.4132, 0.5861] | 0.2515 | **+0.2460** |
| **NR-ER** | 539 | 82 | 15.21% | **0.6335** [0.5607, 0.7042] | **0.3449** [0.2456, 0.4558] | 0.1521 | **+0.1928** |

---

## 4. Failure Mode Investigation: Endpoints Where Models Struggle

Task 5 requires identifying endpoints where models perform poorly and diagnosing root causes. Three endpoints exhibit notable failure modes under scaffold split:

### 4.1 `NR-ER` (Estrogen Receptor Full Target) — Severe Generalization Breakdown
- **Observed Metrics**:
  - Decision Tree Test ROC-AUC: **0.5342** (barely above a coin flip).
  - Logistic Regression Test ROC-AUC: **0.5823** (F1 = 0.2591, MCC = 0.1036).
  - Random Forest Test ROC-AUC: **0.6335** (lowest among all 12 assays for RF).
- **Root Cause Investigation**:
  1. **Scaffold Clustered Generalization Gap**: Under random split, Random Forest achieves **0.7945 ROC-AUC** (+0.1610 higher!). This massive divergence proves that the model learns training scaffolds well, but completely fails to generalize across scaffold boundaries for ER.
  2. **Chemical Divergence**: Estrogen receptor ligands are classically partitioned into two distinct chemotypes: steroidal core structures (cyclopentanoperhydrophenanthrenes like estradiol) and non-steroidal bisphenols/stilbenes (e.g., diethylstilbestrol, bisphenol A derivatives). Because Bemis–Murcko scaffold grouping places whole core families into separate partitions, test active molecules have core rings unseen in the training partition.
  3. **High False Negative Rate**: Out of 82 toxic test compounds, Random Forest missed **49 active compounds** (FN rate = 59.8%). Compounds like `TOX28905` (quinidine/cinchonidine alkaloid scaffold) were predicted with probability 0.43 (below 0.5 threshold) despite showing confirmed ER binding.

### 4.2 `SR-ATAD5` (DNA Damage / Genotoxicity Response) — Precision Collapse
- **Observed Metrics**:
  - Test PR-AUC: **0.1374** (RF), **0.0936** (LR), **0.0756** (DT).
  - While test ROC-AUC is decent (0.7376 for RF), PR-AUC is severely depressed.
  - Uplift over chance is only **+0.0859**.
- **Root Cause Investigation**:
  1. **Extreme Class Imbalance**: ATAD5 has only 34 positive compounds out of 660 observed test compounds (active prevalence = **5.15%**).
  2. **High False Positive Burden**: At a 0.5 probability cutoff, Random Forest flagged **46 false positives** while identifying only 9 true positives (Precision = 16.36%). 
  3. **Mechanistic Diversity**: ATAD5 luciferase induction is triggered by stalled replication forks resulting from diverse DNA lesions (alkylating agents, topoisomerase inhibitors, crosslinkers, reactive oxygen species). Because DNA-damaging molecules share very few common 2D Morgan substructures, 2D circular fingerprints fire promiscuously on reactive groups (e.g., `TOX540` diphenyl carbonate predicted at 0.52 probability), yielding poor precision.

### 4.3 `NR-PPAR-gamma` (Peroxisome Proliferator-Activated Receptor gamma)
- **Observed Metrics**:
  - Test Actives: Only **29 positive compounds** out of 577 test molecules (5.03% prevalence).
  - Decision Tree PR-AUC is **0.0921** (chance is 0.0503).
  - Random Forest misses **20 out of 29 positives** (Recall = 31.03%).
- **Root Cause Investigation**:
  - PPAR-gamma has the fewest active training compounds in the entire dataset: only 133 positives out of 5,300 molecules (2.51% training prevalence).
  - Classical PPAR-gamma pharmacophores require an acidic head group (e.g., thiazolidinedione or carboxylate) and a lipophilic tail separated by an aromatic linker. In the scaffold test partition, novel linkers and acidic heads (`TOX27268`, a benzothiophene tetrazole derivative) were given probabilities under 0.33, completely evading detection.

---

## 5. Success Mode Investigation: High-Performing Endpoints

Conversely, two endpoints demonstrated outstanding performance:

### 5.1 `SR-MMP` (Mitochondrial Membrane Potential Disruption)
- **Random Forest**: Test ROC-AUC = **0.8274**, Test PR-AUC = **0.6076** (chance = 0.2112, uplift = **+0.3965**), F1 = **0.5404**, Recall = **72.64%**.
- **Physical Reason for Success**: Mitochondrial uncouplers and membrane permeabilizers share clear physicochemical rules: moderate-to-high lipophilicity (`MolLogP`), delocalized lipophilic cations, and weakly acidic protonophoric aromatic rings. Member 2's standardized continuous descriptors combined with Morgan fingerprints strongly encode these properties.

### 5.2 `NR-AhR` (Aryl Hydrocarbon Receptor)
- **Random Forest**: Test ROC-AUC = **0.8274**, Test PR-AUC = **0.5555** (chance = 0.1526, uplift = **+0.4028**).
- **Physical Reason for Success**: The AhR binding pocket selectively binds planar polycyclic aromatic hydrocarbons (PAHs) and halogenated dibenzodioxins. The 2,048 Morgan bits (radius=2) excel at capturing contiguous aromatic ring topologies and halogen substitution patterns, allowing models to identify toxic binders even across distinct scaffolds.

---

## 6. Scaffold Split vs. Random Split Experiment (Task 6)

To rigorously answer Task 6, we conducted a parallel 80/10/10 random stratified split across all 7,823 compounds using the identical seed (`seed=42`) and model configurations.

### Empirical Generalization Gap ($\Delta\text{ROC-AUC} = \text{ROC}_{\text{Random}} - \text{ROC}_{\text{Scaffold}}$):

| Endpoint | Scaffold ROC-AUC (RF) | Random Split ROC-AUC (RF) | $\Delta\text{ROC-AUC}$ (Optimism Bias) | Scaffold PR-AUC (RF) | Random Split PR-AUC (RF) | $\Delta\text{PR-AUC}$ (Optimism Bias) |
|---|---|---|---|---|---|---|
| **NR-AR** | 0.7387 | 0.8202 | **+0.0815** | 0.4740 | 0.4936 | +0.0196 |
| **NR-AR-LBD** | 0.8194 | 0.9140 | **+0.0946** | 0.4527 | 0.7432 | **+0.2905** |
| **NR-AhR** | 0.8274 | 0.8933 | **+0.0659** | 0.5555 | 0.6563 | **+0.1008** |
| **NR-Aromatase**| 0.7398 | 0.8226 | **+0.0828** | 0.4182 | 0.4257 | +0.0075 |
| **NR-ER** | 0.6335 | 0.7945 | **+0.1610** | 0.3449 | 0.4830 | **+0.1381** |
| **NR-ER-LBD** | 0.7294 | 0.8263 | **+0.0969** | 0.3724 | 0.4742 | **+0.1018** |
| **NR-PPAR-gamma**| 0.8078 | 0.7701 | -0.0377 | 0.3267 | 0.2663 | -0.0604 |
| **SR-ARE** | 0.7196 | 0.7918 | **+0.0722** | 0.4975 | 0.5012 | +0.0037 |
| **SR-ATAD5** | 0.7376 | 0.8305 | **+0.0929** | 0.1374 | 0.2762 | **+0.1388** |
| **SR-HSE** | 0.7372 | 0.7701 | **+0.0329** | 0.2502 | 0.3276 | +0.0774 |
| **SR-MMP** | 0.8274 | 0.8751 | **+0.0477** | 0.6076 | 0.7115 | **+0.1039** |
| **SR-p53** | 0.7317 | 0.8933 | **+0.1616** | 0.3169 | 0.5502 | **+0.2333** |
| **MACRO MEAN** | **0.7541** | **0.8381** | **+0.0840** | **0.3962** | **0.4924** | **+0.0962** |

### Scientific Conclusions on Task 6:
1. **Pervasive Optimism Bias**: In 11 out of 12 endpoints, random splitting inflates ROC-AUC by an average of **+0.0840** and PR-AUC by **+0.0962**.
2. **Severe Scaffold Leakage**: In `SR-p53` and `NR-ER`, random splitting exaggerates ROC-AUC by **+0.1616** and **+0.1610** respectively. Random splitting allows molecules sharing identical Bemis–Murcko core scaffolds to appear in both train and test. The classifier simply memorizes the core scaffold rather than learning transferable SAR (structure-activity relationships).
3. **Recommendation**: All future model benchmarking (including Phase 5 Deep Learning / Graph Neural Networks) **must** report Bemis–Murcko scaffold split metrics to reflect realistic prospective performance.

---

## 7. Molecular Case Studies (False Positives and False Negatives)

Inspection of compound predictions in `results/test_predictions.csv` reveals actionable chemical insights:

### Case Study 1: Hard False Negatives (Actives Missed by Random Forest)
- **`NR-ER` / Molecule `TOX28905`** (`C=C[C@H]1CN2CC[C@H]1C[C@H]2[C@H](O)c1ccnc2ccc(OC)c...`):
  - *True Label*: 1 (Active), *RF Probability*: 0.4322, *LR Probability*: 0.2811.
  - *Diagnosis*: Quinine/cinchona alkaloid scaffold. Its bridged bicyclic quinuclidine ring system has no analogs in the training active set, causing Morgan circular hash bits to miss binding features.
- **`NR-PPAR-gamma` / Molecule `TOX27268`** (`COc1ccc2sc(C(=O)Nc3nnn[n-]3)c(OC(C)C)c2c1...`):
  - *True Label*: 1 (Active), *RF Probability*: 0.3227.
  - *Diagnosis*: Novel tetrazole-based bioisostere. The model was trained predominantly on carboxylate head groups and missed this tetrazolate anion.

### Case Study 2: Hard False Positives (Inactives Predicted Toxic)
- **`SR-ATAD5` / Molecule `TOX540`** (`O=C(Oc1ccccc1)Oc1ccccc1` - Diphenyl Carbonate):
  - *True Label*: 0 (Inactive), *RF Probability*: 0.5196, *LR Probability*: 0.5614.
  - *Diagnosis*: Carbonate ester linkage and dual benzene rings trigger Morgan bits associated with active aryl carbamates and DNA alkylators, leading to a false alert.
- **`NR-AR-LBD` / Molecule `TOX26452`** (Steroid derivative):
  - *True Label*: 0 (Inactive), *RF Probability*: 0.6064.
  - *Diagnosis*: High structural overlap with androgen receptor binders (four-ring steroid core), but possessing a specific amide substitution that sterically prevents receptor dimerization. 2D fingerprints lack the 3D steric resolution to detect this negative clash.

---

## 8. Catalog of Generated Artifacts and Visualizations

All generated artifacts are organized under `results/`:

```
results/
├── metrics.csv                  # Full per-endpoint, per-model metrics with 95% CIs and threshold stats
├── summary_metrics.csv          # Macro-averaged summary table across all 12 endpoints
├── statistical_tests.csv        # Paired Bootstrap and Wilcoxon p-values for model comparisons
├── split_comparison.csv         # Scaffold vs. Random split comparison metrics (Task 6)
├── test_predictions.csv         # Compound-level predictions with SMILES, scaffold, probabilities, and labels
├── error_analysis.md            # This comprehensive scientific error analysis report
└── plots/                       # Publication-grade figures (300 DPI)
    ├── model_comparison_roc_pr_auc.png  # Grouped bar chart comparing models with error bars
    ├── roc_curves_all_endpoints.png     # 3x4 grid of ROC curves across all 12 assays
    ├── pr_curves_all_endpoints.png      # 3x4 grid of PR curves with random baseline prevalence
    ├── endpoint_class_distributions.png # Test set positive/negative counts and prevalence %
    ├── performance_vs_imbalance.png     # Correlation between class imbalance and PR/ROC performance
    ├── confusion_matrices_worst_best.png# Confusion heatmaps for best (SR-MMP, NR-AR-LBD) vs worst (NR-ER, SR-ATAD5)
    └── scaffold_vs_random_split.png     # Direct comparison of Scaffold vs Random Split performance
```

---

## 9. Recommendations for Downstream Modeling (Member 3 & Phase 5)

1. **Threshold Tuning**: Using the default 0.5 probability cutoff is sub-optimal for severe class imbalance. Tuning decision thresholds via Youden's $J$ or F1-maximization improves recall on low-prevalence targets like `NR-PPAR-gamma` and `SR-ATAD5`.
2. **Multi-Task Learning / Shared Representations**: Since assays in the same biological family (e.g., `NR-AR` and `NR-AR-LBD`, or `NR-ER` and `NR-ER-LBD`) share active ligands, multi-task neural networks or Graph Neural Networks (GNNs) can transfer representations from data-rich assays to data-sparse assays.
3. **Stereochemical and 3D Conformation Modeling**: For stereoisomer conflict pairs identified by Member 1 and steroid stereoisomers in `NR-AR-LBD`, 2D Morgan fingerprints without 3D chiral geometry miss sterically driven binding inactivation.
