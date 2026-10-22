"""
data_pipeline.py
----------------
Simple helper functions for loading, checking and cleaning the Tox21 dataset.

Used by eda.ipynb. Every function returns a normal pandas DataFrame or a
dictionary so the results can be displayed and checked in the notebook.

The original CSV file is only READ, never written.
"""

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Hide RDKit warning messages (we record the errors ourselves instead).
RDLogger.DisableLog("rdApp.*")

ID_COLUMN = "mol_id"
SMILES_COLUMN = "smiles"

TARGET_COLUMNS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


# 1. Loading and basic checks

def load_dataset(path):
    """Read the CSV file exactly as it is (no values are changed)."""
    df = pd.read_csv(path)
    return df


def check_columns(df):
    """Compare the real column names with the expected ones."""
    expected = TARGET_COLUMNS + [ID_COLUMN, SMILES_COLUMN]
    missing_columns = [col for col in expected if col not in df.columns]
    extra_columns = [col for col in df.columns if col not in expected]
    return {"missing_columns": missing_columns, "extra_columns": extra_columns}


def check_target_values(df):
    """
    For every target column, list the values that appear and count any value
    that is NOT 0, 1 or missing. Nothing is converted or removed here.
    """
    rows = []
    for col in TARGET_COLUMNS:
        values = df[col].dropna()
        unexpected = []
        for value in values.unique():
            # Accept 0/1 stored as numbers or as text ("0", "1", "0.0", "1.0")
            if str(value).strip() not in ["0", "1", "0.0", "1.0"]:
                unexpected.append(value)
        n_unexpected = int(values.isin(unexpected).sum())
        rows.append({
            "assay": col,
            "dtype": str(df[col].dtype),
            "values_found": sorted([str(v) for v in values.unique()]),
            "n_missing": int(df[col].isnull().sum()),
            "n_unexpected_values": n_unexpected,
            "unexpected_values": unexpected,
        })
    return pd.DataFrame(rows)


# 2. Missing labels and class imbalance


def label_statistics(df):
    """
    Per-assay counts. Missing labels stay missing: they are never counted as 0.
    """
    n_rows = len(df)
    rows = []
    for col in TARGET_COLUMNS:
        positives = int((df[col] == 1).sum())
        negatives = int((df[col] == 0).sum())
        observed = int(df[col].notnull().sum())
        missing = int(df[col].isnull().sum())

        if observed > 0:
            positive_rate = 100 * positives / observed
        else:
            positive_rate = float("nan")

        if positives > 0:
            neg_to_pos = negatives / positives
        else:
            neg_to_pos = float("nan")

        rows.append({
            "assay": col,
            "positive": positives,
            "negative": negatives,
            "observed": observed,
            "missing": missing,
            "missing_pct": round(100 * missing / n_rows, 2),
            "positive_rate_pct": round(positive_rate, 2),
            "neg_to_pos_ratio": round(neg_to_pos, 2),
        })
    return pd.DataFrame(rows)


def overall_label_counts(df):
    """Total observed / missing labels across all 12 assays."""
    observed = int(df[TARGET_COLUMNS].notnull().sum().sum())
    missing = int(df[TARGET_COLUMNS].isnull().sum().sum())
    total_cells = len(df) * len(TARGET_COLUMNS)
    return {
        "rows": len(df),
        "total_label_cells": total_cells,
        "observed_labels": observed,
        "missing_labels": missing,
        "missing_pct": round(100 * missing / total_cells, 2),
    }



# 3. SMILES validation and canonicalization


def canonicalize_smiles(smiles):
    """
    Return (canonical_smiles, smiles_no_stereo, error_message).
    For an invalid SMILES the first two values are None.
    """
    if pd.isnull(smiles) or str(smiles).strip() == "":
        return None, None, "empty SMILES"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Parse again without sanitization to get RDKit's actual error message
        raw_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if raw_mol is None:
            return None, None, "SMILES syntax could not be parsed"
        try:
            Chem.SanitizeMol(raw_mol)
            error = "RDKit could not parse SMILES"
        except Exception as e:
            error = "RDKit sanitization failed: " + str(e)
        return None, None, error

    canonical = Chem.MolToSmiles(mol)                            # keeps stereochemistry
    no_stereo = Chem.MolToSmiles(mol, isomericSmiles=False)      # ignores stereochemistry
    return canonical, no_stereo, ""


def add_canonical_smiles(df):
    """
    Return a COPY of df with four new columns:
    canonical_smiles, smiles_no_stereo, smiles_valid, invalid_reason.
    Original columns (mol_id, smiles, 12 labels) are untouched.
    """
    result = df.copy()
    canonical_list = []
    no_stereo_list = []
    valid_list = []
    reason_list = []

    for smiles in result[SMILES_COLUMN]:
        canonical, no_stereo, error = canonicalize_smiles(smiles)
        canonical_list.append(canonical)
        no_stereo_list.append(no_stereo)
        valid_list.append(canonical is not None)
        reason_list.append(error)

    result["canonical_smiles"] = canonical_list
    result["smiles_no_stereo"] = no_stereo_list
    result["smiles_valid"] = valid_list
    result["invalid_reason"] = reason_list
    return result



# 4. Duplicates and conflicting labels


def duplicate_counts(df_checked):
    """
    Count different kinds of duplicates.
    df_checked must be the output of add_canonical_smiles().
    'Extra copies' = rows that repeat an earlier row (df.duplicated()).
    """
    valid = df_checked[df_checked["smiles_valid"]]
    original_columns = TARGET_COLUMNS + [ID_COLUMN, SMILES_COLUMN]
    return {
        "exact_duplicate_rows": int(df_checked[original_columns].duplicated().sum()),
        "duplicate_mol_ids": int(df_checked[ID_COLUMN].duplicated().sum()),
        "duplicate_raw_smiles": int(df_checked[SMILES_COLUMN].duplicated().sum()),
        "duplicate_canonical_smiles": int(valid["canonical_smiles"].duplicated().sum()),
        "duplicate_ignoring_stereo": int(valid["smiles_no_stereo"].duplicated().sum()),
    }


def duplicate_groups(df_checked, group_column):
    """
    Find every group of valid molecules that share the same value in
    group_column (e.g. 'canonical_smiles' or 'smiles_no_stereo').
    For each group and each assay, flag a conflict when the observed
    labels contain both 0 and 1. Labels are NOT changed.
    """
    valid = df_checked[df_checked["smiles_valid"]]
    repeated = valid[valid[group_column].duplicated(keep=False)]

    rows = []
    for key, group in repeated.groupby(group_column):
        conflicting_assays = []
        for col in TARGET_COLUMNS:
            observed_values = group[col].dropna().unique()
            if len(observed_values) > 1:
                conflicting_assays.append(col)

        rows.append({
            "group_type": group_column,
            "group_key": key,
            "n_records": len(group),
            "mol_ids": ";".join(group[ID_COLUMN]),
            "original_smiles": ";".join(group[SMILES_COLUMN]),
            "has_label_conflict": len(conflicting_assays) > 0,
            "conflicting_assays": ";".join(conflicting_assays),
        })

    columns = ["group_type", "group_key", "n_records", "mol_ids", "original_smiles",
               "has_label_conflict", "conflicting_assays"]
    return pd.DataFrame(rows, columns=columns)



# 5. Summary table


def build_dataset_summary(overall_info, stats_original, stats_valid, split_counts, split_class_stats):
    """
    Put everything into one long table with columns:
    section, assay, metric, value
    """
    rows = []

    for metric, value in overall_info.items():
        rows.append({"section": "overall", "assay": "ALL", "metric": metric, "value": value})

    for section, stats in [("assay_original", stats_original),
                           ("assay_after_invalid_removed", stats_valid)]:
        for _, row in stats.iterrows():
            for metric in stats.columns:
                if metric != "assay":
                    rows.append({"section": section, "assay": row["assay"],
                                 "metric": metric, "value": row[metric]})

    for _, row in split_counts.iterrows():
        rows.append({"section": "split_size", "assay": "ALL",
                     "metric": row["split"] + "_count", "value": row["count"]})
        rows.append({"section": "split_size", "assay": "ALL",
                     "metric": row["split"] + "_pct", "value": row["pct"]})

    for _, row in split_class_stats.iterrows():
        for metric in ["positive", "negative", "missing", "positive_rate_pct"]:
            rows.append({"section": "split_class_" + row["split"], "assay": row["assay"],
                         "metric": metric, "value": row[metric]})

    return pd.DataFrame(rows)
