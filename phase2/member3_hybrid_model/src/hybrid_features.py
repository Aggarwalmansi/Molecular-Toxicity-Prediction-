"""
hybrid_features.py
==================
Builds the hybrid feature set for Member 3.

The hybrid input is two blocks joined side by side:

    [ 128 learned DL features ] + [ 5 molecular descriptors ] = 133 features

The 128 learned features are the bottleneck embeddings already extracted by
Member 2 from the trained MultiTaskToxicNet. We read them from the saved
parquet files instead of re-running the network, so Member 2's model is used
exactly as they trained it.

The 5 descriptors (MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors) are loaded
through Member 1's Tox21DataLoader, which reads the Phase 1 scaled descriptor
file. Those values were standardised in Phase 1 using training statistics
only, so reusing them introduces no leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DESCRIPTOR_NAMES = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]


def get_embeddings_dir():
    """Returns the folder where Member 2 saved the extracted embeddings."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "phase2" / "member2_deep_learning_model" / "embeddings"


def load_dl_embeddings(split):
    """
    Loads Member 2's learned features for one split.

    Returns:
        embeddings: array of shape (n_molecules, 128)
        mol_ids:    list of molecule ids in the same row order
    """
    embeddings_file = get_embeddings_dir() / f"{split}_embeddings.parquet"
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"Missing {embeddings_file}. Run Member 2's extract_embeddings step first."
        )

    table = pd.read_parquet(embeddings_file)
    mol_ids = table["molecule_id"].tolist()
    embedding_columns = [c for c in table.columns if c != "molecule_id"]
    embeddings = table[embedding_columns].values.astype(np.float64)
    return embeddings, mol_ids


def build_hybrid_features(loader, split, use_descriptors=True):
    """
    Builds the hybrid feature matrix for one split.

    The molecule order from Member 2's embeddings is checked against the order
    from Member 1's loader. If they ever stop matching we stop immediately,
    because silently mismatched rows would pair one molecule's features with
    another molecule's labels.

    use_descriptors=False returns the 128 embeddings on their own. We use that
    on the validation split to check whether the 5 descriptors actually add
    anything.

    Returns:
        features: array of shape (n_molecules, 133) or (n_molecules, 128)
        mol_ids:  list of molecule ids
    """
    embeddings, embedding_mol_ids = load_dl_embeddings(split)
    descriptors, descriptor_mol_ids = loader.load_features("descriptors_scaled", split=split)

    if embedding_mol_ids != list(descriptor_mol_ids):
        raise ValueError(
            f"Molecule order mismatch on split '{split}' between Member 2's "
            "embeddings and Member 1's descriptor table."
        )

    if not use_descriptors:
        return embeddings, embedding_mol_ids

    features = np.concatenate([embeddings, descriptors.astype(np.float64)], axis=1)
    return features, embedding_mol_ids


def get_feature_names(use_descriptors=True):
    """Returns the column names of the hybrid feature matrix, for feature importance plots."""
    names = [f"emb_{i:03d}" for i in range(128)]
    if use_descriptors:
        names = names + DESCRIPTOR_NAMES
    return names
