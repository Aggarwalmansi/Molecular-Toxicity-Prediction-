"""
app.py
======
Tox21 Molecular Toxicity Prediction Platform - Interactive Frontend
Built with Streamlit for Member 4: Evaluation, Statistics and Error Analysis

Features:
- Executive Benchmark Overview & Statistical Matrices
- Per-Endpoint Diagnostic Curves (ROC, PR, Confusion Matrices)
- Scaffold Split vs. Random Split Generalization Analysis (Task 6)
- Chemical Error Analysis & Interactive Molecule Explorer with 2D Structure Rendering
- Real-Time Live Toxicity Prediction Playground for Custom SMILES
- Scientific Audit Report Viewer
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Optional RDKit import for chemical structure drawing and live featurization
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, Draw, rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ==============================================================================
# Page Configuration & Styling
# ==============================================================================

st.set_page_config(
    page_title="Tox21 Molecular Toxicity Prediction Platform",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern biotech dark UI theme
st.markdown("""
<style>
    /* Main container background */
    .stApp {
        background-color: #0b111e;
        color: #e2e8f0;
    }
    
    /* Header card */
    .header-box {
        background: linear-gradient(135deg, #132238 0%, #0d1726 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #1e3a5f;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 14px;
    }
    
    .badge-container {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    
    .custom-badge {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid #334155;
    }
    
    /* KPI Metric Cards */
    .kpi-card {
        background: #111c2d;
        border: 1px solid #1e2e45;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    
    .kpi-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    
    .kpi-sub {
        font-size: 0.75rem;
        color: #10b981;
        margin-top: 4px;
    }
    
    /* Tables */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# File Paths & Data Loading
# ==============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
ARTIFACTS_DIR = SCRIPT_DIR / "tox21_member2_handoff" / "artifacts"

TARGET_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
    "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

ASSAY_NAMES = {
    "NR-AR": "Androgen Receptor (Full)",
    "NR-AR-LBD": "Androgen Receptor (Ligand Binding Domain)",
    "NR-AhR": "Aryl Hydrocarbon Receptor",
    "NR-Aromatase": "Aromatase Enzyme (CYP19A1)",
    "NR-ER": "Estrogen Receptor (Full)",
    "NR-ER-LBD": "Estrogen Receptor (Ligand Binding Domain)",
    "NR-PPAR-gamma": "Peroxisome Proliferator-Activated Receptor γ",
    "SR-ARE": "Antioxidant Response Element (Nrf2)",
    "SR-ATAD5": "DNA Damage Response (ATAD5)",
    "SR-HSE": "Heat Shock Response Element",
    "SR-MMP": "Mitochondrial Membrane Potential Disruption",
    "SR-p53": "p53 Tumor Suppressor Pathway",
}


@st.cache_data
def load_all_metrics():
    """Loads precomputed evaluation artifacts."""
    metrics_csv = RESULTS_DIR / "metrics.csv"
    summary_csv = RESULTS_DIR / "summary_metrics.csv"
    stats_csv = RESULTS_DIR / "statistical_tests.csv"
    split_csv = RESULTS_DIR / "split_comparison.csv"
    preds_csv = RESULTS_DIR / "test_predictions.csv"

    data = {}
    if metrics_csv.exists():
        data["metrics"] = pd.read_csv(metrics_csv)
    if summary_csv.exists():
        data["summary"] = pd.read_csv(summary_csv)
    if stats_csv.exists():
        data["stats"] = pd.read_csv(stats_csv)
    if split_csv.exists():
        data["split"] = pd.read_csv(split_csv)
    if preds_csv.exists():
        data["preds"] = pd.read_csv(preds_csv)
    return data


@st.cache_resource
def load_and_train_rf_models():
    """
    Fits and caches Random Forest models on canonical training data for live inference.
    """
    labels_path = ARTIFACTS_DIR / "labels_and_splits.parquet"
    features_path = ARTIFACTS_DIR / "X_combined.parquet"
    scaler_path = ARTIFACTS_DIR / "scaler_descriptors.json"

    if not (labels_path.exists() and features_path.exists() and scaler_path.exists()):
        return None, None

    labels_df = pd.read_parquet(labels_path)
    features_df = pd.read_parquet(features_path)
    with open(scaler_path) as f:
        scaler = json.load(f)

    feat_cols = [c for c in features_df.columns if c != "mol_id"]
    X = features_df[feat_cols].values.astype(np.float64)
    train_mask = (labels_df["split"] == "train").values

    models = {}
    for assay in TARGET_ASSAYS:
        y_all = labels_df[assay].values
        obs = train_mask & ~np.isnan(y_all)
        X_tr = X[obs]
        y_tr = y_all[obs]

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        models[assay] = rf

    return models, scaler


# ==============================================================================
# Helper Functions: Featurization & Rendering
# ==============================================================================

def featurize_smiles(smiles: str, scaler: dict) -> Tuple[np.ndarray, dict, str]:
    """
    Featurizes custom SMILES into canonical 2,053 features (2,048 Morgan + 5 scaled descriptors).
    """
    if not RDKIT_AVAILABLE:
        return None, None, "RDKit is not installed in the active environment."

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None, "Invalid SMILES string. RDKit could not parse the chemical structure."

    # 1. 2,048-bit Morgan Fingerprints (radius=2)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprintAsNumPy(mol).astype(np.float64)

    # 2. 5 Physicochemical Descriptors
    raw_desc = {
        "MolWt": float(Descriptors.MolWt(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "MolLogP": float(Descriptors.MolLogP(mol)),
        "NumHDonors": float(Lipinski.NumHDonors(mol)),
        "NumHAcceptors": float(Lipinski.NumHAcceptors(mol)),
    }

    desc_arr = np.array([
        raw_desc["MolWt"],
        raw_desc["TPSA"],
        raw_desc["MolLogP"],
        raw_desc["NumHDonors"],
        raw_desc["NumHAcceptors"],
    ])

    # Standardize using training-fitted scaler parameters
    mean_arr = np.array(scaler["mean"])
    std_arr = np.array(scaler["std"])
    scaled_desc = (desc_arr - mean_arr) / std_arr

    # Canonical 2,053 combined features
    combined = np.hstack([fp, scaled_desc])
    return combined, raw_desc, None


def render_mol_image(smiles: str, size: Tuple[int, int] = (320, 220)):
    """Draws 2D chemical structure image using RDKit."""
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


# ==============================================================================
# Header Banner
# ==============================================================================

st.markdown("""
<div class="header-box">
    <div class="header-title">🧪 Tox21 Molecular Toxicity Prediction Platform</div>
    <div class="header-subtitle">
        Reproducible Machine Learning Evaluation, Statistical Rigor & Chemical Error Analysis across 12 Assays
    </div>
    <div class="badge-container">
        <span class="custom-badge">🏛️ Member 4 Final Deliverable</span>
        <span class="custom-badge">🧬 Tox21 10,000 Library</span>
        <span class="custom-badge">🛡️ Strict Missing-Label Masking</span>
        <span class="custom-badge">🔬 Bemis–Murcko Scaffold Split</span>
        <span class="custom-badge">📈 95% Bootstrap Confidence Intervals</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# Navigation Sidebar
# ==============================================================================

st.sidebar.image("https://raw.githubusercontent.com/rdkit/rdkit/master/Docs/Book/source/images/RDKit_logo.png", width=160)
st.sidebar.title("Navigation")

navigation_mode = st.sidebar.radio(
    "Select View:",
    [
        "📊 Executive Benchmark & Summary",
        "🔬 Per-Endpoint Deep Dive & Curves",
        "⚖️ Scaffold Split vs. Random Split (Task 6)",
        "🔍 Chemical Error Analysis & Molecule Explorer",
        "🧪 Live Toxicity Prediction Playground",
        "📖 Scientific Report & Documentation",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Quick Stats")
st.sidebar.markdown("""
- **Molecules**: 7,823 curated compounds
- **Features**: 2,048 Morgan bits + 5 descriptors
- **Assays**: 12 (7 NR + 5 SR pathways)
- **Splits**: 80% Train, 10% Val, 10% Test
- **Missing Data**: 17.05% strictly masked
""")

data_cache = load_all_metrics()
metrics_df = data_cache.get("metrics", pd.DataFrame())
summary_df = data_cache.get("summary", pd.DataFrame())
stats_df = data_cache.get("stats", pd.DataFrame())
split_df = data_cache.get("split", pd.DataFrame())
preds_df = data_cache.get("preds", pd.DataFrame())


# ==============================================================================
# View 1: Executive Benchmark & Summary
# ==============================================================================

if navigation_mode == "📊 Executive Benchmark & Summary":
    st.subheader("📊 Executive Benchmark & Macro-Averaged Summary")

    # Top KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Top Model (Scaffold Split)</div>
            <div class="kpi-value">Random Forest</div>
            <div class="kpi-sub">Outperforms LR & DT (p < 0.01)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Macro ROC-AUC (RF)</div>
            <div class="kpi-value">0.7541</div>
            <div class="kpi-sub">± 0.0570 Std Dev across 12 assays</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Macro PR-AUC (RF)</div>
            <div class="kpi-value">0.3962</div>
            <div class="kpi-sub">+0.2931 Uplift over baseline chance</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-label">Scaffold Generalization Gap</div>
            <div class="kpi-value">-0.0840</div>
            <div class="kpi-sub">Penalty vs. leaky random splitting</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 🏆 Macro-Averaged Model Comparison Table")
    macro_table = pd.DataFrame({
        "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "Test ROC-AUC": ["0.7541 ± 0.0570", "0.7152 ± 0.0596", "0.6428 ± 0.0597"],
        "Test PR-AUC": ["0.3962 ± 0.1323", "0.2949 ± 0.1319", "0.2091 ± 0.0957"],
        "F1-Score": ["0.3969 ± 0.1070", "0.3268 ± 0.1355", "0.2629 ± 0.1182"],
        "Balanced Accuracy": ["0.6684 ± 0.0507", "0.6358 ± 0.0761", "0.6295 ± 0.0592"],
        "MCC": ["0.3141 ± 0.1177", "0.2413 ± 0.1233", "0.1616 ± 0.0859"],
        "Statistical Rank": ["🥇 1st Place", "🥈 2nd Place", "🥉 3rd Place"],
    })
    st.dataframe(macro_table, hide_index=True)

    st.markdown("---")
    st.markdown("### 📈 Per-Endpoint Model Benchmark Chart (ROC-AUC & PR-AUC)")
    fig_path = PLOTS_DIR / "model_comparison_roc_pr_auc.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Per-Endpoint Comparison of Models with 95% Bootstrap Confidence Intervals", width=1050)

    st.markdown("---")
    st.markdown("### 🔬 Paired Statistical Hypothesis Tests")
    st.markdown(r"""
    All models were compared on the **exact same test compound IDs (`mol_id`)** per target.
    - **Paired Bootstrap Difference Test**: Evaluates whether $\Delta\text{ROC}$ or $\Delta\text{PR} \ne 0$.
    - **Wilcoxon Signed-Rank Test**: Evaluates continuous Brier squared probabilistic error per compound.
    """)
    if not stats_df.empty:
        filter_comp = st.selectbox(
            "Filter Pairwise Comparison:",
            ["Random Forest vs Logistic Regression", "Random Forest vs Decision Tree", "Logistic Regression vs Decision Tree"]
        )
        filtered_stats = stats_df[stats_df["comparison"] == filter_comp]
        st.dataframe(filtered_stats, hide_index=True)


# ==============================================================================
# View 2: Per-Endpoint Deep Dive & Curves
# ==============================================================================

elif navigation_mode == "🔬 Per-Endpoint Deep Dive & Curves":
    st.subheader("🔬 Toxicological Endpoint Deep Dive")

    selected_endpoint = st.selectbox(
        "Select Target Assay:",
        TARGET_ASSAYS,
        format_func=lambda x: f"{x} — {ASSAY_NAMES.get(x, '')}",
    )

    endpoint_sub = metrics_df[metrics_df["endpoint"] == selected_endpoint]

    if not endpoint_sub.empty:
        rf_row = endpoint_sub[endpoint_sub["model"] == "Random Forest"].iloc[0]

        # Endpoint Summary Badges
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pathway Group", "Nuclear Receptor" if selected_endpoint.startswith("NR") else "Stress Response")
        c2.metric("Test Compounds ($N_{\text{obs}}$)", f"{int(rf_row['test_samples']):,}")
        c3.metric("Actives in Test", f"{int(rf_row['test_positives'])} ({rf_row['positive_prevalence']:.1%})")
        c4.metric("Baseline Chance PR-AUC", f"{rf_row['random_chance_pr_auc']:.4f}")

        st.markdown("#### Performance Metrics Across Models for " + selected_endpoint)
        display_cols = [
            "model", "test_roc_auc", "roc_auc_ci_lower", "roc_auc_ci_upper",
            "test_pr_auc", "pr_auc_ci_lower", "pr_auc_ci_upper",
            "f1_score", "balanced_accuracy", "precision", "recall_sensitivity", "mcc"
        ]
        st.dataframe(endpoint_sub[display_cols].rename(columns={
            "model": "Model",
            "test_roc_auc": "ROC-AUC",
            "roc_auc_ci_lower": "ROC CI Low",
            "roc_auc_ci_upper": "ROC CI High",
            "test_pr_auc": "PR-AUC",
            "pr_auc_ci_lower": "PR CI Low",
            "pr_auc_ci_upper": "PR CI High",
            "f1_score": "F1",
            "balanced_accuracy": "Bal Acc",
            "recall_sensitivity": "Recall",
            "precision": "Precision",
            "mcc": "MCC",
        }), hide_index=True)

    st.markdown("---")
    st.markdown("### 📊 Diagnostic Curves Across All 12 Endpoints")
    curve_tab1, curve_tab2, curve_tab3 = st.tabs(["ROC Curves Matrix (12 Assays)", "PR Curves Matrix (12 Assays)", "Confusion Matrix Focus"])
    with curve_tab1:
        st.image(str(PLOTS_DIR / "roc_curves_all_endpoints.png"), caption="3x4 Matrix of Receiver Operating Characteristic (ROC) Curves", width=1100)
    with curve_tab2:
        st.image(str(PLOTS_DIR / "pr_curves_all_endpoints.png"), caption="3x4 Matrix of Precision-Recall (PR) Curves vs. Baseline Chance Prevalence", width=1100)
    with curve_tab3:
        st.image(str(PLOTS_DIR / "confusion_matrices_worst_best.png"), caption="Confusion Heatmaps for High vs. Low Performing Endpoints", width=950)


# ==============================================================================
# View 3: Scaffold Split vs. Random Split (Task 6 Proof)
# ==============================================================================

elif navigation_mode == "⚖️ Scaffold Split vs. Random Split (Task 6)":
    st.subheader("⚖️ Scaffold Split vs. Random Split Experiment (Task 6)")

    st.markdown("""
    **Core Question**: *Does Random Splitting give a dangerous, artificially inflated illusion of model accuracy?*
    
    In cheminformatics and molecular discovery, standard random splitting frequently leaks core Bemis–Murcko molecular scaffolds between training and test sets. The model simply memorizes the shape rather than learning transferable structure-activity relationships.
    """)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.image(str(PLOTS_DIR / "scaffold_vs_random_split.png"), caption="Comparison of Random Forest Performance: Scaffold Split vs. Random Split", width=700)
    with col2:
        st.markdown("### 📉 The Generalization Penalty Table")
        if not split_df.empty:
            # Merge scaffold metrics with random split
            rf_scaff = metrics_df[metrics_df["model"] == "Random Forest"][["endpoint", "test_roc_auc", "test_pr_auc"]]
            rf_rand = split_df[split_df["model"] == "Random Forest"]
            merged = pd.merge(rf_scaff, rf_rand, on="endpoint")
            merged["Delta ROC-AUC"] = (merged["random_test_roc_auc"] - merged["test_roc_auc"]).round(4)
            merged["Delta PR-AUC"] = (merged["random_test_pr_auc"] - merged["test_pr_auc"]).round(4)
            
            st.dataframe(
                merged[["endpoint", "test_roc_auc", "random_test_roc_auc", "Delta ROC-AUC", "Delta PR-AUC"]].rename(columns={
                    "endpoint": "Assay",
                    "test_roc_auc": "Scaffold ROC",
                    "random_test_roc_auc": "Random ROC",
                    "Delta ROC-AUC": "Inflation (ΔROC)",
                    "Delta PR-AUC": "Inflation (ΔPR)",
                }),
                hide_index=True
            )

    st.info("""
    **Key Takeaways**:
    1. **Pervasive Optimism Bias**: In 11 of 12 endpoints, random splitting inflates ROC-AUC by an average of **+0.0840** and PR-AUC by **+0.0962**.
    2. **Extreme Scaffold Leakage**: Endpoints like `SR-p53` (+0.1616) and `NR-ER` (+0.1610) suffer extreme scaffold leakage under random split.
    3. **Gold Standard**: Bemis–Murcko scaffold splitting is strictly necessary to evaluate real-world discovery readiness.
    """)


# ==============================================================================
# View 4: Chemical Error Analysis & Molecule Explorer
# ==============================================================================

elif navigation_mode == "🔍 Chemical Error Analysis & Molecule Explorer":
    st.subheader("🔍 Chemical Error Analysis & Interactive Molecule Inspector")

    st.markdown("""
    Inspect individual compounds evaluated in the test partition. Identify why specific molecules caused **False Negatives** (dangerous missed toxicities) or **False Positives** (wasteful false alarms).
    """)

    if not preds_df.empty:
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            inspect_endpoint = st.selectbox(
                "Filter Assay:",
                TARGET_ASSAYS,
                index=TARGET_ASSAYS.index("NR-ER") if "NR-ER" in TARGET_ASSAYS else 0,
                format_func=lambda x: f"{x} ({ASSAY_NAMES.get(x, '')})"
            )
        with col_ctrl2:
            error_filter = st.selectbox(
                "Filter Prediction Category:",
                ["🔴 False Negatives (Actives Missed by RF)", "🟡 False Positives (Inactives Flagged Toxic)", "🟢 True Positives (Correctly Flagged Toxic)"]
            )

        sub_preds = preds_df[preds_df["endpoint"] == inspect_endpoint].copy()

        if "False Negatives" in error_filter:
            filtered_mols = sub_preds[(sub_preds["y_true"] == 1) & (sub_preds["pred_rf"] == 0)]
        elif "False Positives" in error_filter:
            filtered_mols = sub_preds[(sub_preds["y_true"] == 0) & (sub_preds["pred_rf"] == 1)]
        else:
            filtered_mols = sub_preds[(sub_preds["y_true"] == 1) & (sub_preds["pred_rf"] == 1)]

        st.markdown(f"Found **{len(filtered_mols)} compounds** in category `{error_filter}` for `{inspect_endpoint}`.")

        if not filtered_mols.empty:
            selected_mol_id = st.selectbox(
                "Select Molecule to Inspect:",
                filtered_mols["mol_id"].tolist(),
                format_func=lambda x: f"{x} — SMILES: {filtered_mols[filtered_mols['mol_id'] == x]['smiles'].iloc[0][:50]}..."
            )

            chosen_mol = filtered_mols[filtered_mols["mol_id"] == selected_mol_id].iloc[0]

            col_m1, col_m2 = st.columns([1, 1.2])
            with col_m1:
                st.markdown("#### 🧪 2D Molecular Structure")
                img = render_mol_image(chosen_mol["smiles"])
                if img:
                    st.image(img, caption=f"Compound: {chosen_mol['mol_id']}", width=340)
                else:
                    st.warning("Structure rendering requires RDKit.")

            with col_m2:
                st.markdown("#### 🔬 Compound Diagnostics")
                st.markdown(f"**Molecule ID**: `{chosen_mol['mol_id']}`")
                st.markdown(f"**True Experimental Label**: `{'1 (Toxic / Active)' if chosen_mol['y_true'] == 1 else '0 (Non-Toxic / Inactive)'}`")
                st.markdown(f"**Random Forest Probability**: `{chosen_mol['prob_rf']:.4f}`")
                st.markdown(f"**Logistic Regression Probability**: `{chosen_mol['prob_lr']:.4f}`")
                st.markdown(f"**Decision Tree Probability**: `{chosen_mol['prob_dt']:.4f}`")
                st.markdown(f"**SMILES**: `{chosen_mol['smiles']}`")
                st.markdown(f"**Core Bemis-Murcko Scaffold**: `{chosen_mol['scaffold_smiles']}`")


# ==============================================================================
# View 5: Live Toxicity Prediction Playground
# ==============================================================================

elif navigation_mode == "🧪 Live Toxicity Prediction Playground":
    st.subheader("🧪 Real-Time Live Toxicity Prediction Playground")
    st.markdown("""
    Input any valid SMILES chemical string to instantly calculate 2,048-bit Morgan Fingerprints, standardize the 5 continuous physicochemical descriptors, and predict toxicological risk across all 12 Tox21 assays using the trained Random Forest models.
    """)

    # Quick chemical presets
    presets = {
        "Estradiol (Estrogen Receptor agonist)": "CC12CCC3C(C1CCC2O)CCC4=C3C=CC(=C4)O",
        "Aspirin (Common analgesic)": "CC(=O)Oc1ccccc1C(=O)O",
        "Tamoxifen (Selective Estrogen Receptor Modulator)": "CCC(=C(c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3",
        "Bisphenol A (Endocrine disruptor)": "CC(C)(c1ccc(O)cc1)c2ccc(O)cc2",
        "Benzo[a]pyrene (Potent AhR binder & carcinogen)": "c1ccc2c(c1)cc3ccc4cccc5c4c3c2cc5",
    }

    selected_preset = st.selectbox("Or choose a pre-filled chemical preset:", ["Custom Input"] + list(presets.keys()))

    if selected_preset != "Custom Input":
        default_smiles = presets[selected_preset]
    else:
        default_smiles = "CC(=O)Oc1ccccc1C(=O)O"

    smiles_input = st.text_input("Enter Molecule SMILES string:", value=default_smiles)

    if st.button("🚀 Predict Toxicity Across 12 Assays", type="primary"):
        models_dict, scaler = load_and_train_rf_models()
        if models_dict is None:
            st.error("Missing model training artifacts in tox21_member2_handoff/artifacts.")
        else:
            features, raw_desc, err = featurize_smiles(smiles_input, scaler)
            if err:
                st.error(err)
            else:
                col_res1, col_res2 = st.columns([1, 1.3])
                with col_res1:
                    st.markdown("#### 🧪 Chemical Structure")
                    mol_img = render_mol_image(smiles_input, size=(360, 240))
                    if mol_img:
                        st.image(mol_img, caption="2D Chemical Representation", width=360)

                    st.markdown("#### 📊 Physicochemical Descriptors")
                    st.write(f"- **Molecular Weight**: `{raw_desc['MolWt']:.2f}` g/mol")
                    st.write(f"- **MolLogP (Lipophilicity)**: `{raw_desc['MolLogP']:.2f}`")
                    st.write(f"- **TPSA**: `{raw_desc['TPSA']:.2f}` Å²")
                    st.write(f"- **H-Bond Donors**: `{int(raw_desc['NumHDonors'])}`")
                    st.write(f"- **H-Bond Acceptors**: `{int(raw_desc['NumHAcceptors'])}`")

                with col_res2:
                    st.markdown("#### 🎯 Toxicity Risk Predictions")
                    pred_rows = []
                    for assay in TARGET_ASSAYS:
                        rf_model = models_dict[assay]
                        prob_toxic = float(rf_model.predict_proba(features.reshape(1, -1))[0, 1])

                        if prob_toxic >= 0.50:
                            risk_badge = "🔴 HIGH RISK (Toxic)"
                        elif prob_toxic >= 0.35:
                            risk_badge = "🟡 BORDERLINE"
                        else:
                            risk_badge = "🟢 LOW RISK (Safe)"

                        pred_rows.append({
                            "Assay Target": assay,
                            "Pathway Description": ASSAY_NAMES.get(assay, ""),
                            "Toxicity Probability": f"{prob_toxic:.1%}",
                            "Risk Classification": risk_badge,
                        })

                    pred_df = pd.DataFrame(pred_rows)
                    st.dataframe(pred_df, hide_index=True)


# ==============================================================================
# View 6: Scientific Report & Documentation
# ==============================================================================

elif navigation_mode == "📖 Scientific Report & Documentation":
    st.subheader("📖 Scientific Error Analysis Report & Documentation")
    report_path = RESULTS_DIR / "error_analysis.md"
    if report_path.exists():
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.warning("Error analysis report not found in results/error_analysis.md")
