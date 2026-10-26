"""
app.py
======
Tox21 Molecular Toxicity Prediction - Streamlit Dashboard (Simplified)
Member 4: Evaluation, Statistics, and Error Analysis

Designed to be simple, clean, and easy to explain during presentation/viva.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

# Optional RDKit import for drawing chemical molecules
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# ==============================================================================
# Page Setup
# ==============================================================================

st.set_page_config(
    page_title="Tox21 Toxicity Prediction Dashboard",
    page_icon="🧪",
    layout="wide",
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"


# ==============================================================================
# Header
# ==============================================================================

st.title("🧪 Tox21 Molecular Toxicity Prediction Dashboard")
st.markdown("**Member 4 Role:** Evaluation, Statistics and Error Analysis")
st.write(
    "This dashboard summarizes the performance of three baseline Machine Learning models "
    "(Random Forest, Logistic Regression, Decision Tree) across 12 toxicological endpoints in the Tox21 dataset."
)

st.markdown("---")


# ==============================================================================
# Top Metric Highlights (Easy to explain in 10 seconds)
# ==============================================================================

col1, col2, col3, col4 = st.columns(4)
col1.metric("Top Model", "Random Forest", "Rank #1")
col2.metric("Mean ROC-AUC", "0.754", "+0.039 over LogReg")
col3.metric("Mean PR-AUC", "0.396", "+0.101 over LogReg")
col4.metric("Scaffold Generalization Gap", "-0.084", "Inflation in Random Split")

st.markdown("---")


# ==============================================================================
# 3 Simple Tabs for Presentation
# ==============================================================================

tab1, tab2, tab3 = st.tabs([
    "📊 1. Model Benchmark & Results",
    "⚖️ 2. Scaffold vs. Random Split (Task 6)",
    "🔬 3. Molecule Inspector & Error Analysis",
])


# ------------------------------------------------------------------------------
# TAB 1: MODEL BENCHMARK
# ------------------------------------------------------------------------------
with tab1:
    st.subheader("Model Performance Comparison (All 12 Endpoints)")
    st.write(
        "We evaluated three classical machine learning models trained on 2,048-bit Morgan "
        "fingerprints + 5 physicochemical descriptors using a 80/10/10 scaffold split."
    )

    # Clean Macro Table
    summary_data = {
        "Model": ["Random Forest", "Logistic Regression", "Decision Tree"],
        "Test ROC-AUC": [0.7541, 0.7152, 0.6428],
        "Test PR-AUC": [0.3962, 0.2949, 0.2091],
        "F1-Score": [0.3969, 0.3268, 0.2629],
        "Balanced Accuracy": [0.6684, 0.6358, 0.6295],
        "Matthews Correlation (MCC)": [0.3141, 0.2413, 0.1616],
    }
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, hide_index=True)

    # Plot
    plot_path1 = PLOTS_DIR / "model_comparison_roc_pr_auc.png"
    if plot_path1.exists():
        st.markdown("#### Per-Endpoint ROC-AUC and PR-AUC Comparison")
        st.image(str(plot_path1), caption="Performance across all 12 endpoints (with 95% Bootstrap error bars)", width=950)

    # Simple 3-bullet explanation
    st.info("""
    **Key Takeaways to Explain:**
    1. **Random Forest performed best overall** (0.754 ROC-AUC, 0.396 PR-AUC) because it captures non-linear combinations of molecular bits.
    2. **Logistic Regression** is a solid linear baseline (0.715 ROC-AUC), while **Decision Tree** overfitted on sparse binary bits (0.643 ROC-AUC).
    3. **Strict Missing-Label Masking:** Missing experimental labels (17.05%) were never replaced with 0; models were evaluated only on observed lab data.
    """)


# ------------------------------------------------------------------------------
# TAB 2: SCAFFOLD SPLIT VS RANDOM SPLIT (TASK 6)
# ------------------------------------------------------------------------------
with tab2:
    st.subheader("Task 6: Why Scaffold Splitting Matters (Generalization Gap)")
    st.write(
        "We compared the realistic **Bemis-Murcko Scaffold Split** against a standard **Random Split** "
        "to test if random splitting gives an inflated illusion of accuracy."
    )

    col_s1, col_s2 = st.columns([1.2, 1])

    with col_s1:
        plot_path2 = PLOTS_DIR / "scaffold_vs_random_split.png"
        if plot_path2.exists():
            st.image(str(plot_path2), caption="Random Forest: Scaffold Split vs Random Split", width=650)

    with col_s2:
        st.markdown("#### The Optimism Bias Table")
        split_csv = RESULTS_DIR / "split_comparison.csv"
        metrics_csv = RESULTS_DIR / "metrics.csv"

        if split_csv.exists() and metrics_csv.exists():
            df_split = pd.read_csv(split_csv)
            df_metrics = pd.read_csv(metrics_csv)

            rf_scaff = df_metrics[df_metrics["model"] == "Random Forest"][["endpoint", "test_roc_auc"]]
            rf_rand = df_split[df_split["model"] == "Random Forest"][["endpoint", "random_test_roc_auc"]]
            merged_split = pd.merge(rf_scaff, rf_rand, on="endpoint")
            merged_split["Inflation (ΔROC)"] = (merged_split["random_test_roc_auc"] - merged_split["test_roc_auc"]).round(4)

            st.dataframe(
                merged_split.rename(columns={
                    "endpoint": "Endpoint",
                    "test_roc_auc": "Scaffold ROC",
                    "random_test_roc_auc": "Random ROC",
                }),
                hide_index=True
            )

    st.success("""
    **Simple Viva Explanation:**
    - In **Random Split**, molecules with identical core scaffolds end up in both train and test. The model simply memorizes the shape (+0.084 average inflation).
    - In **Scaffold Split**, entire scaffold families are separated. It tests if the model can truly predict toxicity on **novel, unseen chemical structures**.
    """)


# ------------------------------------------------------------------------------
# TAB 3: MOLECULE INSPECTOR & ERROR ANALYSIS
# ------------------------------------------------------------------------------
with tab3:
    st.subheader("Molecule Inspector & Failure Mode Analysis")
    st.write(
        "Inspect individual chemical compounds evaluated in the test set to understand "
        "why certain endpoints succeeded (e.g., SR-MMP) and others struggled (e.g., NR-ER)."
    )

    preds_csv = RESULTS_DIR / "test_predictions.csv"
    if preds_csv.exists():
        df_preds = pd.read_csv(preds_csv)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            chosen_endpoint = st.selectbox(
                "Select Endpoint:",
                ["NR-ER", "SR-MMP", "NR-AhR", "SR-ATAD5", "NR-PPAR-gamma"],
                index=0,
            )
        with col_f2:
            chosen_type = st.selectbox(
                "Filter Compound Type:",
                ["False Negatives (Toxic Actives Missed)", "False Positives (Inactives Flagged Toxic)", "True Positives (Correctly Flagged Toxic)"]
            )

        sub = df_preds[df_preds["endpoint"] == chosen_endpoint]
        if "False Negatives" in chosen_type:
            filtered = sub[(sub["y_true"] == 1) & (sub["pred_rf"] == 0)]
        elif "False Positives" in chosen_type:
            filtered = sub[(sub["y_true"] == 0) & (sub["pred_rf"] == 1)]
        else:
            filtered = sub[(sub["y_true"] == 1) & (sub["pred_rf"] == 1)]

        st.write(f"Found **{len(filtered)}** compounds in this category.")

        if not filtered.empty:
            sample_row = filtered.iloc[0]
            selected_id = st.selectbox("Choose Molecule ID:", filtered["mol_id"].tolist())
            selected_row = filtered[filtered["mol_id"] == selected_id].iloc[0]

            col_m1, col_m2 = st.columns([1, 1.2])

            with col_m1:
                st.markdown("#### 🧪 2D Chemical Structure")
                if RDKIT_AVAILABLE:
                    mol = Chem.MolFromSmiles(selected_row["smiles"])
                    if mol:
                        img = Draw.MolToImage(mol, size=(320, 220))
                        st.image(img, caption=f"Compound: {selected_row['mol_id']}")
                else:
                    st.info("RDKit structure drawing available when run in environment.")

            with col_m2:
                st.markdown("#### 🔬 Compound Details")
                st.write(f"**Molecule ID:** `{selected_row['mol_id']}`")
                st.write(f"**True Label:** `{'1 (Toxic / Active)' if selected_row['y_true'] == 1 else '0 (Non-Toxic / Inactive)'}`")
                st.write(f"**Random Forest Probability:** `{selected_row['prob_rf']:.2%}`")
                st.write(f"**Logistic Regression Probability:** `{selected_row['prob_lr']:.2%}`")
                st.write(f"**Decision Tree Probability:** `{selected_row['prob_dt']:.2%}`")
                st.write(f"**SMILES:** `{selected_row['smiles']}`")

    st.warning("""
    **Error Analysis Summary to Speak in Viva:**
    - **Why NR-ER performed poorly (0.634 ROC-AUC):** Estrogen Receptor binders belong to distinct chemical families (steroidal vs. bisphenols). The scaffold split separated them, so the test set contained scaffolds unseen in training.
    - **Why SR-MMP performed well (0.827 ROC-AUC):** Mitochondrial disruption is strongly driven by lipophilicity (MolLogP) and charge, which our 5 descriptors and Morgan bits captured easily.
    """)
