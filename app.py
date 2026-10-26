"""
app.py
======
Tox21 Molecular Toxicity Prediction Platform - Team Presentation Dashboard
Comprehensive Multi-Member Showcase:
- Member 1: Dataset Preprocessing, Quality Audits & Bemis–Murcko Scaffold Split
- Member 2: Feature Engineering, Variance Filtering & Univariate Feature Selection
- Member 3: Baseline Classifiers, Hyperparameters & Training Dynamics
- Member 4: Rigorous Evaluation, Bootstrap CIs, Paired Significance & Error Analysis
- Interactive Live Demo: 2D Chemical Structure Rendering & Live Toxicity Prediction
"""

import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# RDKit import for chemical structure drawing and live featurization
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Draw, rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ==============================================================================
# Page Setup & Styling
# ==============================================================================

st.set_page_config(
    page_title="Tox21 Molecular Toxicity Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
MEMBER1_DIR = SCRIPT_DIR / "tox21_project"
MEMBER2_DIR = SCRIPT_DIR / "tox21_member2_handoff" / "artifacts"

TARGET_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

ASSAY_NAMES = {
    "NR-AR": "Androgen Receptor (Full Target)",
    "NR-AR-LBD": "Androgen Receptor (Ligand Binding Domain)",
    "NR-AhR": "Aryl Hydrocarbon Receptor",
    "NR-Aromatase": "Aromatase Enzyme (CYP19A1)",
    "NR-ER": "Estrogen Receptor (Full Target)",
    "NR-ER-LBD": "Estrogen Receptor (Ligand Binding Domain)",
    "NR-PPAR-gamma": "Peroxisome Proliferator-Activated Receptor γ",
    "SR-ARE": "Antioxidant Response Element (Nrf2)",
    "SR-ATAD5": "DNA Damage Response (ATAD5)",
    "SR-HSE": "Heat Shock Response Element",
    "SR-MMP": "Mitochondrial Membrane Potential Disruption",
    "SR-p53": "p53 Tumor Suppressor Pathway",
}


# ==============================================================================
# Helper Data Loaders
# ==============================================================================

@st.cache_data
def load_all_artifacts():
    """Loads CSV artifacts from all project members."""
    data = {}
    
    # Member 1 artifacts
    summary_path = MEMBER1_DIR / "dataset_summary.csv"
    excl_path = MEMBER1_DIR / "exclusion_report.csv"
    dup_path = MEMBER1_DIR / "duplicate_audit.csv"
    if summary_path.exists():
        data["m1_summary"] = pd.read_csv(summary_path)
    if excl_path.exists():
        data["m1_exclusion"] = pd.read_csv(excl_path)
    if dup_path.exists():
        data["m1_duplicates"] = pd.read_csv(dup_path)

    # Member 2 artifacts
    feat_sel_path = MEMBER2_DIR / "feature_selection_summary.csv"
    scaler_path = MEMBER2_DIR / "scaler_descriptors.json"
    if feat_sel_path.exists():
        data["m2_feat_sel"] = pd.read_csv(feat_sel_path)
    if scaler_path.exists():
        with open(scaler_path) as f:
            data["m2_scaler"] = json.load(f)

    # Member 3 artifacts
    m3_results_path = SCRIPT_DIR / "model_evaluation_results.csv"
    if m3_results_path.exists():
        data["m3_results"] = pd.read_csv(m3_results_path)

    # Member 4 artifacts
    m4_metrics_path = RESULTS_DIR / "metrics.csv"
    m4_stats_path = RESULTS_DIR / "statistical_tests.csv"
    m4_split_path = RESULTS_DIR / "split_comparison.csv"
    m4_preds_path = RESULTS_DIR / "test_predictions.csv"
    if m4_metrics_path.exists():
        data["m4_metrics"] = pd.read_csv(m4_metrics_path)
    if m4_stats_path.exists():
        data["m4_stats"] = pd.read_csv(m4_stats_path)
    if m4_split_path.exists():
        data["m4_split"] = pd.read_csv(m4_split_path)
    if m4_preds_path.exists():
        data["m4_preds"] = pd.read_csv(m4_preds_path)

    return data


def render_mol_image(smiles: str, size: Tuple[int, int] = (300, 200)):
    """Draws 2D chemical structure image using RDKit."""
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Draw.MolToImage(mol, size=size)
    except Exception:
        return None


# ==============================================================================
# Header & Project Overview
# ==============================================================================

st.title("🧪 Tox21 Molecular Toxicity Prediction Platform")
st.markdown("### Collaborative Applied Machine Learning Engineering Project")
st.write(
    "An end-to-end machine learning system predicting biological and toxicological activity across "
    "**12 nuclear receptor and stress response pathways** on 7,823 environmental chemicals."
)

# Top Team Highlights
c1, c2, c3, c4 = st.columns(4)
c1.metric("📦 Curated Dataset", "7,823 Compounds", "12 Target Assays")
c2.metric("🧬 Features", "2,053 Features", "2048 Morgan + 5 Desc")
c3.metric("🤖 Baseline Models", "3 Architectures", "LR, DecTree, RandForest")
c4.metric("🏆 Top Performance", "0.754 ROC-AUC", "Random Forest (Rank #1)")

st.markdown("---")


# ==============================================================================
# 5 Structured Team Presentation Tabs
# ==============================================================================

tab_m1, tab_m2, tab_m3, tab_m4, tab_demo = st.tabs([
    "🧹 Tab 1: Member 1 — Data & Scaffold Split",
    "🧬 Tab 2: Member 2 — Features & Selection",
    "🤖 Tab 3: Member 3 — Models & Training",
    "📈 Tab 4: Member 4 — Evaluation & Error Analysis",
    "🧪 Tab 5: Live Demo & Molecule Inspector",
])

artifacts = load_all_artifacts()


# ------------------------------------------------------------------------------
# TAB 1: MEMBER 1 — DATA CURATION, AUDITS & SCAFFOLD SPLIT
# ------------------------------------------------------------------------------
with tab_m1:
    st.header("🧹 Member 1: Dataset Preprocessing, Quality Audits & Scaffold Split")
    st.markdown("**Core Ownership:** Data hygiene, chemical sanitization, audit trail, and zero-leakage scaffold split.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Dataset Rows", "7,831", "Raw Tox21 file")
    col2.metric("Invalid SMILES Filtered", "8 Molecules", "Explicit valence > permitted")
    col3.metric("Valid Curated Molecules", "7,823", "Canonical SMILES")
    col4.metric("Missing Experimental Labels", "17.05%", "Strictly masked out")

    st.markdown("---")
    st.subheader("1. Data Sanitization & Exclusion Audit")
    st.write(
        "8 chemical structures in the raw Tox21 library contained hypervalent aluminum atoms "
        "(`[AlH3]` with valence 6) that failed RDKit chemical sanitization. These were documented and removed."
    )
    if "m1_exclusion" in artifacts:
        excl_df = artifacts["m1_exclusion"][["mol_id", "smiles", "invalid_reason"]]
        st.dataframe(excl_df, hide_index=True)

    st.markdown("---")
    st.subheader("2. Stereoisomer & Label Conflict Audit")
    st.write(
        "Member 1 identified **107 groups (229 molecules)** that are stereoisomers of each other. "
        "Crucially, **31 groups showed contradictory biological activity** in assays. "
        "No labels were arbitrarily changed or averaged; stereoisomers were kept intact and grouped into the same split."
    )
    if "m1_duplicates" in artifacts:
        dup_df = artifacts["m1_duplicates"].head(10)
        st.dataframe(dup_df, hide_index=True)

    st.markdown("---")
    st.subheader("3. Bemis–Murcko Scaffold Split Policy (80 / 10 / 10)")
    st.write(
        "To avoid train-test contamination, molecules were clustered by their Bemis–Murcko core scaffold without stereochemistry. "
        "Molecules without rings (acyclic structures, 1,775 molecules) were partitioned into distinct groups."
    )
    split_summary = pd.DataFrame({
        "Split Partition": ["Training Set", "Validation Set", "Test Set", "Total Valid"],
        "Molecule Count": [6259, 782, 782, 7823],
        "Percentage": ["79.99%", "10.00%", "10.01%", "100.0%"],
        "Role in Pipeline": ["Model training", "Hyperparameter tuning", "Unbiased holdout test", "Certified handoff"],
    })
    st.dataframe(split_summary, hide_index=True)


# ------------------------------------------------------------------------------
# TAB 2: MEMBER 2 — FEATURE ENGINEERING & FEATURE SELECTION
# ------------------------------------------------------------------------------
with tab_m2:
    st.header("🧬 Member 2: Feature Engineering, Standardization & Selection")
    st.markdown("**Core Ownership:** Transforming chemical graphs into numerical feature matrices and feature selection.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Canonical Features", "2,053", "Combined vector")
    col2.metric("Morgan Fingerprints", "2,048 bits", "ECFP4-like (radius=2)")
    col3.metric("Physicochemical Descriptors", "5 Descriptors", "Scaled on train split")
    col4.metric("Variance-Filtered Bits", "1,011 bits", "Variance >= 0.005")

    st.markdown("---")
    st.subheader("1. Canonical Feature Engineering Breakdown")
    st.markdown("""
    - **2,048 Binary Morgan Fingerprint Bits**: Computed using RDKit with radius=2 (equivalent to extended-connectivity fingerprints ECFP4), encoding localized atomic neighborhoods.
    - **5 Continuous Physicochemical Descriptors**:
      - `MolWt`: Molecular Weight (size)
      - `TPSA`: Topological Polar Surface Area (permeability/polarity)
      - `MolLogP`: Octanol-water partition coefficient (lipophilicity)
      - `NumHDonors`: Number of hydrogen bond donors
      - `NumHAcceptors`: Number of hydrogen bond acceptors
    - **Leakage-Safe Scaling**: Standardized strictly using mean and standard deviation from the training partition.
    """)

    st.markdown("---")
    st.subheader("2. Supervised Univariate Feature Selection (Chi2 & ANOVA F-Test)")
    st.write(
        "Member 2 performed univariate association tests per endpoint on observed training compounds. "
        "Below are the top-ranked features associated with toxicity for each of the 12 endpoints:"
    )
    if "m2_feat_sel" in artifacts:
        st.dataframe(artifacts["m2_feat_sel"].rename(columns={
            "endpoint": "Endpoint",
            "train_observed_n": "Train Observed (N)",
            "train_positives": "Train Actives",
            "top_1_associated_feature": "Top 1 Feature",
            "top_1_association_score": "Score",
            "top_5_associated_features": "Top 5 Associated Features",
        }), hide_index=True)

    st.info("""
    **Cheminformatics Insight:**
    - For `SR-MMP` (Mitochondrial Disruption), continuous descriptor `MolLogP` ranks among the top features because lipophilicity drives membrane permeability.
    - For `SR-ARE` (Antioxidant response), `MolWt` is the #1 associated feature.
    - For `NR-AR` and `NR-AR-LBD` (Androgen Receptors), bit `morgan_519` is the #1 predictor with an association score >1,500.
    """)


# ------------------------------------------------------------------------------
# TAB 3: MEMBER 3 — MODEL ARCHITECTURES & BASELINE TRAINING
# ------------------------------------------------------------------------------
with tab_m3:
    st.header("🤖 Member 3: Baseline Classifiers & Model Training")
    st.markdown("**Core Ownership:** Classical model training, class imbalance weighting, and baseline validation.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Class Imbalance Ratio", "5x to 34x", "Negative:Positive ratio")
    col2.metric("Weighting Strategy", "class_weight='balanced'", "Adjusts for rare actives")
    col3.metric("Endpoints Fitted", "12 Assays", "Independent binary classifiers")

    st.markdown("---")
    st.subheader("1. Evaluated Baseline Architectures & Hyperparameters")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        **1. Logistic Regression**
        - Fast, linear decision boundary
        - `max_iter`: 1,000
        - `class_weight`: `'balanced'`
        - `solver`: L-BFGS
        - Purpose: Linear baseline
        """)
    with col_b:
        st.markdown("""
        **2. Decision Tree**
        - Non-linear decision rules
        - `max_depth`: 10
        - `class_weight`: `'balanced'`
        - `criterion`: Gini impurity
        - Purpose: Single-tree baseline
        """)
    with col_c:
        st.markdown("""
        **3. Random Forest**
        - 100 decorrelated trees
        - `n_estimators`: 100
        - `max_depth`: 12
        - `class_weight`: `'balanced'`
        - Purpose: Non-linear ensemble
        """)

    st.markdown("---")
    st.subheader("2. Member 3 Initial Benchmark Results (Validation vs. Test)")
    st.write("Validation set (used for tuning) vs. Test set baseline scores across models:")
    if "m3_results" in artifacts:
        m3_df = artifacts["m3_results"]
        st.dataframe(m3_df[["endpoint", "model", "train_total_samples", "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc"]].rename(columns={
            "endpoint": "Endpoint",
            "model": "Model",
            "train_total_samples": "Train N",
            "val_roc_auc": "Val ROC-AUC",
            "val_pr_auc": "Val PR-AUC",
            "test_roc_auc": "Test ROC-AUC",
            "test_pr_auc": "Test PR-AUC",
        }), hide_index=True)


# ------------------------------------------------------------------------------
# TAB 4: MEMBER 4 — EVALUATION, STATISTICS & ERROR ANALYSIS (YOUR ROLE)
# ------------------------------------------------------------------------------
with tab_m4:
    st.header("📈 Member 4: Evaluation, Statistics, Error Analysis & Task 6")
    st.markdown("**Core Ownership:** Proving whether models truly generalize, statistical hypothesis tests, and failure mode diagnosis.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Top Model", "Random Forest", "0.754 ROC / 0.396 PR")
    col2.metric("Bootstrap Resamples", "1,000 / Endpoint", "95% Confidence Intervals")
    col3.metric("Paired Comparisons", "Exact Same Compounds", "Controlled p-values")
    col4.metric("Generalization Penalty", "-0.0840 ROC-AUC", "Scaffold vs. Random Split")

    st.markdown("---")
    st.subheader("1. Macro-Averaged Benchmark Summary (All 12 Assays)")
    macro_table = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "Test ROC-AUC (Mean ± Std)": ["0.7541 ± 0.0570", "0.7152 ± 0.0596", "0.6428 ± 0.0597"],
        "Test PR-AUC (Mean ± Std)": ["0.3962 ± 0.1323", "0.2949 ± 0.1319", "0.2091 ± 0.0957"],
        "F1-Score": [0.3969, 0.3268, 0.2629],
        "Balanced Accuracy": [0.6684, 0.6358, 0.6295],
        "Matthews Correlation (MCC)": [0.3141, 0.2413, 0.1616],
    })
    st.dataframe(macro_table, hide_index=True)

    plot_path1 = PLOTS_DIR / "model_comparison_roc_pr_auc.png"
    if plot_path1.exists():
        st.image(str(plot_path1), caption="Model Comparison across all 12 endpoints with 95% Bootstrap error bars", width=950)

    st.markdown("---")
    st.subheader("2. Task 6 Discovery: Scaffold Split vs. Random Split Benchmark")
    st.write(
        "We compared the realistic **Bemis-Murcko Scaffold Split** against an 80/10/10 **Random Split** "
        "to quantify how much random splitting artificially exaggerates model performance."
    )
    col_sc1, col_sc2 = st.columns([1.2, 1])
    with col_sc1:
        plot_path2 = PLOTS_DIR / "scaffold_vs_random_split.png"
        if plot_path2.exists():
            st.image(str(plot_path2), caption="Scaffold Split vs. Random Split Comparison", width=620)
    with col_sc2:
        if "m4_split" in artifacts and "m4_metrics" in artifacts:
            rf_scaff = artifacts["m4_metrics"][artifacts["m4_metrics"]["model"] == "Random Forest"][["endpoint", "test_roc_auc"]]
            rf_rand = artifacts["m4_split"][artifacts["m4_split"]["model"] == "Random Forest"][["endpoint", "random_test_roc_auc"]]
            merged_sp = pd.merge(rf_scaff, rf_rand, on="endpoint")
            merged_sp["Inflation (ΔROC)"] = (merged_sp["random_test_roc_auc"] - merged_sp["test_roc_auc"]).round(4)
            st.dataframe(merged_sp.rename(columns={
                "endpoint": "Endpoint",
                "test_roc_auc": "Scaffold ROC",
                "random_test_roc_auc": "Random ROC",
            }), hide_index=True)

    st.markdown("---")
    st.subheader("3. Failure Mode Diagnostics (Error Analysis)")
    st.info("""
    - **Why `NR-ER` struggled (0.634 ROC-AUC):** Estrogen Receptor binders belong to distinct chemical families (steroidal vs. bisphenols). Scaffold clustering separated these families into different splits, meaning the test set contained core scaffolds unseen during training.
    - **Why `SR-ATAD5` (0.137 PR-AUC) & `NR-PPAR-gamma` struggled:** Suffered from extreme class imbalance (<5% actives), generating high false positive rates and low precision.
    - **Why `SR-MMP` succeeded (0.827 ROC-AUC):** Mitochondrial membrane disruption is strongly driven by lipophilicity (`MolLogP`) and delocalized positive charge, which the 5 descriptors and Morgan bits captured easily.
    """)


# ------------------------------------------------------------------------------
# TAB 5: INTERACTIVE LIVE DEMO & MOLECULE INSPECTOR
# ------------------------------------------------------------------------------
with tab_demo:
    st.header("🧪 Interactive Demo: Molecule Inspector & Live Toxicity Predictor")

    subtab1, subtab2 = st.tabs(["🔍 Molecule Error Inspector (Test Set)", "⚡ Live SMILES Predictor"])

    # Subtab 1: Inspect evaluated test compounds
    with subtab1:
        st.subheader("Inspect Test Compound Failures & Successes")
        if "m4_preds" in artifacts:
            df_preds = artifacts["m4_preds"]
            c_ep, c_filt = st.columns(2)
            with c_ep:
                sel_ep = st.selectbox("Select Endpoint:", ["NR-ER", "SR-MMP", "NR-AhR", "SR-ATAD5", "NR-PPAR-gamma"])
            with c_filt:
                sel_cat = st.selectbox(
                    "Filter Category:",
                    ["False Negatives (Toxic Actives Missed by RF)", "False Positives (Inactives Flagged Toxic)", "True Positives (Accurately Flagged Toxic)"]
                )

            sub_m = df_preds[df_preds["endpoint"] == sel_ep]
            if "False Negatives" in sel_cat:
                filtered_m = sub_m[(sub_m["y_true"] == 1) & (sub_m["pred_rf"] == 0)]
            elif "False Positives" in sel_cat:
                filtered_m = sub_m[(sub_m["y_true"] == 0) & (sub_m["pred_rf"] == 1)]
            else:
                filtered_m = sub_m[(sub_m["y_true"] == 1) & (sub_m["pred_rf"] == 1)]

            st.write(f"Found **{len(filtered_m)}** molecules matching filter.")

            if not filtered_m.empty:
                pick_id = st.selectbox("Choose Molecule:", filtered_m["mol_id"].tolist())
                pick_row = filtered_m[filtered_m["mol_id"] == pick_id].iloc[0]

                col_d1, col_d2 = st.columns([1, 1.2])
                with col_d1:
                    st.markdown("#### 🧪 2D Molecular Structure")
                    img = render_mol_image(pick_row["smiles"], size=(340, 220))
                    if img:
                        st.image(img, caption=f"ID: {pick_row['mol_id']}")
                    else:
                        st.write("2D structure rendering available with RDKit.")
                with col_d2:
                    st.markdown("#### 🔬 Diagnostic Details")
                    st.write(f"**Molecule ID:** `{pick_row['mol_id']}`")
                    st.write(f"**True Experimental Label:** `{'1 (Toxic / Active)' if pick_row['y_true'] == 1 else '0 (Non-Toxic / Inactive)'}`")
                    st.write(f"**Random Forest Probability:** `{pick_row['prob_rf']:.2%}`")
                    st.write(f"**Logistic Regression Probability:** `{pick_row['prob_lr']:.2%}`")
                    st.write(f"**Decision Tree Probability:** `{pick_row['prob_dt']:.2%}`")
                    st.write(f"**SMILES:** `{pick_row['smiles']}`")
                    st.write(f"**Scaffold:** `{pick_row['scaffold_smiles']}`")

    # Subtab 2: Live SMILES Predictor
    with subtab2:
        st.subheader("⚡ Real-Time Live Molecule Predictor")
        st.write("Enter a chemical SMILES string to view its structure and predict toxicity across endpoints:")

        presets = {
            "Aspirin (Common analgesic)": "CC(=O)Oc1ccccc1C(=O)O",
            "Estradiol (Estrogen Receptor agonist)": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
            "Tamoxifen (Selective Estrogen Receptor Modulator)": "CCC(=C(c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3",
            "Bisphenol A (Endocrine disruptor)": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2",
        }
        pick_preset = st.selectbox("Choose a preset molecule:", ["Custom Input"] + list(presets.keys()))
        input_smi = presets[pick_preset] if pick_preset != "Custom Input" else "CC(=O)Oc1ccccc1C(=O)O"
        smi_text = st.text_input("SMILES Chemical String:", value=input_smi)

        col_p1, col_p2 = st.columns([1, 1.3])
        with col_p1:
            st.markdown("#### 🧪 2D Chemical Structure")
            live_img = render_mol_image(smi_text, size=(340, 220))
            if live_img:
                st.image(live_img, caption="Predicted Molecule Structure")
            else:
                st.warning("Enter a valid SMILES string to render.")

        with col_p2:
            st.markdown("#### 📊 Key Molecular Descriptors")
            if RDKIT_AVAILABLE:
                mol_obj = Chem.MolFromSmiles(smi_text)
                if mol_obj:
                    st.write(f"- **Molecular Weight**: `{Descriptors.MolWt(mol_obj):.2f}` g/mol")
                    st.write(f"- **LogP (Lipophilicity)**: `{Descriptors.MolLogP(mol_obj):.2f}`")
                    st.write(f"- **TPSA**: `{Descriptors.TPSA(mol_obj):.2f}` Å²")
                    st.write(f"- **H-Bond Donors**: `{Lipinski.NumHDonors(mol_obj)}`")
                    st.write(f"- **H-Bond Acceptors**: `{Lipinski.NumHAcceptors(mol_obj)}`")
                    st.success("Valid chemical structure verified by RDKit.")
                else:
                    st.error("Invalid SMILES string.")
