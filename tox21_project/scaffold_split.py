"""
scaffold_split.py
-----------------
Reproducible Bemis-Murcko scaffold split for the Tox21 dataset.

Grouping policy (explained in README.md and eda.ipynb):
  * Molecules WITH rings are grouped by their Bemis-Murcko scaffold
    (computed without stereochemistry, so stereoisomers share a scaffold).
  * Molecules WITHOUT rings have an empty scaffold "". They are NOT put into
    one giant group. Instead each distinct acyclic structure (ignoring
    stereochemistry) forms its own small group. These groups are shuffled and
    shared out like any other small group, so train, validation and test all
    contain some acyclic molecules.
  * Every group goes to exactly one split, so identical structures and
    stereoisomers can never be in two different splits.

Allocation (same idea as the "balanced scaffold split" used in Chemprop):
  1. Groups larger than half of the validation target are "big" groups.
  2. Big groups and small groups are each shuffled with random.Random(seed).
  3. Big groups are placed first, then small groups. Each group goes to
     train if it still fits, otherwise to validation if it fits, otherwise to test.
"""

import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

SPLIT_NAMES = ["train", "validation", "test"]


def get_scaffold(canonical_smiles):
    """Bemis-Murcko scaffold SMILES without stereochemistry ('' = no rings)."""
    mol = Chem.MolFromSmiles(canonical_smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def add_scaffold_groups(df_valid):
    """
    Return a copy of df_valid with the columns:
      scaffold_smiles, has_ring_scaffold, group_key, scaffold_id
    df_valid must contain 'canonical_smiles' and 'smiles_no_stereo'.
    """
    result = df_valid.copy()

    scaffolds = []
    group_keys = []
    for canonical, no_stereo in zip(result["canonical_smiles"], result["smiles_no_stereo"]):
        scaffold = get_scaffold(canonical)
        scaffolds.append(scaffold)
        if scaffold != "":
            group_keys.append("SCAFFOLD:" + scaffold)
        else:
            # acyclic molecule -> its own group, shared with its stereoisomers only
            group_keys.append("ACYCLIC:" + no_stereo)

    result["scaffold_smiles"] = scaffolds
    result["has_ring_scaffold"] = result["scaffold_smiles"] != ""
    result["group_key"] = group_keys

    # Give each group a short readable ID: G00001 = largest group, etc.
    group_sizes = result["group_key"].value_counts()
    sorted_keys = sorted(group_sizes.index, key=lambda key: (-group_sizes[key], key))
    id_lookup = {}
    for number, key in enumerate(sorted_keys, start=1):
        id_lookup[key] = "G" + str(number).zfill(5)
    result["scaffold_id"] = result["group_key"].map(id_lookup)

    return result


def scaffold_split(df_grouped, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Assign every row of df_grouped to 'train', 'validation' or 'test'.
    Returns a copy with a new 'split' column.
    """
    n_total = len(df_grouped)
    train_target = train_frac * n_total
    val_target = val_frac * n_total

    # Row positions belonging to each group (sorted by key so the starting
    # order does not depend on the row order of the CSV file)
    groups = {}
    for position, key in enumerate(df_grouped["group_key"]):
        if key not in groups:
            groups[key] = []
        groups[key].append(position)
    group_keys = sorted(groups.keys())

    big_groups = [key for key in group_keys if len(groups[key]) > val_target / 2]
    small_groups = [key for key in group_keys if len(groups[key]) <= val_target / 2]

    rng = random.Random(seed)
    rng.shuffle(big_groups)
    rng.shuffle(small_groups)

    split_of_group = {}
    n_train = 0
    n_val = 0
    for key in big_groups + small_groups:
        size = len(groups[key])
        if n_train + size <= train_target:
            split_of_group[key] = "train"
            n_train += size
        elif n_val + size <= val_target:
            split_of_group[key] = "validation"
            n_val += size
        else:
            split_of_group[key] = "test"

    result = df_grouped.copy()
    result["split"] = result["group_key"].map(split_of_group)
    return result


def split_sizes(df_split):
    """Count and percentage of molecules in each split."""
    rows = []
    for name in SPLIT_NAMES:
        count = int((df_split["split"] == name).sum())
        rows.append({"split": name, "count": count,
                     "pct": round(100 * count / len(df_split), 2)})
    return pd.DataFrame(rows)


def class_distribution_by_split(df_split, target_columns):
    """Positive / negative / missing counts for every assay in every split."""
    rows = []
    for name in SPLIT_NAMES:
        part = df_split[df_split["split"] == name]
        for col in target_columns:
            positives = int((part[col] == 1).sum())
            negatives = int((part[col] == 0).sum())
            observed = positives + negatives
            if observed > 0:
                rate = round(100 * positives / observed, 2)
            else:
                rate = float("nan")
            rows.append({"split": name, "assay": col, "positive": positives,
                         "negative": negatives, "missing": int(part[col].isnull().sum()),
                         "positive_rate_pct": rate})
    return pd.DataFrame(rows)


def count_keys_in_several_splits(df, column):
    """How many distinct values of `column` appear in more than one split."""
    splits_per_value = df.groupby(column)["split"].nunique()
    return int((splits_per_value > 1).sum())


def validate_split(manifest, n_input_rows):
    """
    Run the leakage / consistency checks on the split manifest.
    The manifest has one row per ORIGINAL CSV row (valid and invalid).
    Returns a table: check, result (PASS/FAIL), detail.
    """
    checks = []

    def add(check, passed, detail):
        checks.append({"check": check, "result": "PASS" if passed else "FAIL",
                       "detail": detail})

    eligible = manifest[manifest["smiles_valid"]]
    excluded = manifest[~manifest["smiles_valid"]]

    add("manifest has one row per input row",
        len(manifest) == n_input_rows and manifest["original_row_index"].is_unique,
        f"{len(manifest)} manifest rows vs {n_input_rows} input rows")

    ok_labels = eligible["split"].isin(SPLIT_NAMES).all()
    add("every eligible compound has exactly one split",
        ok_labels and eligible["mol_id"].is_unique,
        f"{len(eligible)} eligible, {int(eligible['split'].isin(SPLIT_NAMES).sum())} assigned")

    add("invalid SMILES are excluded (not in any split)",
        (excluded["split"] == "excluded").all(),
        f"{len(excluded)} excluded rows")

    n = count_keys_in_several_splits(eligible, "scaffold_id")
    add("no scaffold group crosses splits", n == 0, f"{n} groups in >1 split")

    ring = eligible[eligible["has_ring_scaffold"]]
    n = count_keys_in_several_splits(ring, "scaffold_smiles")
    add("no non-empty Murcko scaffold crosses splits", n == 0, f"{n} scaffolds in >1 split")

    n = count_keys_in_several_splits(eligible, "canonical_smiles")
    add("no canonical structure in more than one split", n == 0, f"{n} structures in >1 split")

    n = count_keys_in_several_splits(eligible, "smiles_no_stereo")
    add("no stereoisomer set in more than one split", n == 0, f"{n} stereo-sets in >1 split")

    n = count_keys_in_several_splits(eligible, "mol_id")
    add("no compound ID in more than one split", n == 0, f"{n} IDs in >1 split")

    acyclic = eligible[~eligible["has_ring_scaffold"]]
    counts = acyclic["split"].value_counts()
    detail = ", ".join(f"{name}={int(counts.get(name, 0))}" for name in SPLIT_NAMES)
    add("acyclic (empty-scaffold) molecules: documented spread across splits",
        True, detail + " (by design, see policy)")

    return pd.DataFrame(checks)
