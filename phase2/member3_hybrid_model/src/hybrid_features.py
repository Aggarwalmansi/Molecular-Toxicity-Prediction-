"""
hybrid_features.py
==================
Builds the hybrid feature set for Member 3.

Phase 2 fix — feature FUSION instead of feature REPLACEMENT
-----------------------------------------------------------
The first version of this module handed the Random Forest only the 128
continuous bottleneck embeddings plus the 5 descriptors (133 features). That
threw away the 2,048 raw Morgan bits the Phase 1 forest was actually good at,
and test macro ROC-AUC collapsed from 0.7541 to 0.6785.

The hybrid input is now the two blocks joined side by side:

    [ 2,048 Morgan bits + 5 scaled descriptors ] + [ 128 learned DL features ]
    = 2,181 features

So the forest keeps every signal Phase 1 had and *gains* the DL representation
on top, which is what "hybrid" was supposed to mean.

The 128 learned features are the bottleneck embeddings already extracted by
Member 2 from the trained MultiTaskToxicNet. We read them from the saved
parquet files instead of re-running the network, so Member 2's model is used
exactly as they trained it.

The raw 2,053-dim block ('combined') is loaded through Member 1's
Tox21DataLoader: 2,048 binary Morgan bits plus the 5 descriptors
(MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors). Those descriptors were
standardised in Phase 1 using training statistics only, so reusing them
introduces no leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DESCRIPTOR_NAMES = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]

N_MORGAN_BITS = 2048
N_DESCRIPTORS = 5
N_EMBEDDINGS = 128
N_RAW_FEATURES = N_MORGAN_BITS + N_DESCRIPTORS      # 2053
N_FUSED_FEATURES = N_RAW_FEATURES + N_EMBEDDINGS    # 2181


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


def load_raw_features(loader, split):
    """
    Loads Member 1's canonical 2,053-dim block for one split:
    2,048 Morgan bits followed by the 5 scaled descriptors.

    Returns:
        features: array of shape (n_molecules, 2053)
        mol_ids:  list of molecule ids in the same row order
    """
    features, mol_ids = loader.load_features("combined", split=split)
    features = np.asarray(features, dtype=np.float64)
    if features.shape[1] != N_RAW_FEATURES:
        raise ValueError(
            f"Expected {N_RAW_FEATURES} raw features on split '{split}', "
            f"got {features.shape[1]}."
        )
    return features, list(mol_ids)


def build_hybrid_features(loader, split, use_raw_features=True, use_descriptors=True,
                          use_embeddings=True):
    """
    Builds the hybrid feature matrix for one split.

    Molecule IDs from Member 2's embeddings are checked row-for-row against the
    IDs from Member 1's loader before anything is concatenated. If they ever
    stop matching we stop immediately, because silently mismatched rows would
    pair one molecule's features with another molecule's labels.

    Parameters
    ----------
    use_raw_features : bool, default True
        Include the 2,048 Morgan bits + 5 descriptors block (2,053 features).
    use_descriptors : bool, default True
        Only consulted when use_raw_features is False. Appends the 5 scaled
        descriptors to the embeddings, reproducing the old 133-feature
        ablation arm so the fused model can be compared against it.
    use_embeddings : bool, default True
        Include Member 2's 128 latent embeddings.

    Returns
    -------
    features : array of shape (n_molecules, D)
        D = 2181 for the default fused configuration.
    mol_ids : list of molecule ids, aligned with the feature rows.
    """
    if not use_raw_features and not use_embeddings:
        raise ValueError("At least one of use_raw_features / use_embeddings must be True.")

    blocks = []
    reference_ids = None

    if use_raw_features:
        raw_features, raw_mol_ids = load_raw_features(loader, split)
        reference_ids = raw_mol_ids
        blocks.append(raw_features)

    if use_embeddings:
        embeddings, embedding_mol_ids = load_dl_embeddings(split)
        if reference_ids is None:
            reference_ids = embedding_mol_ids
        elif embedding_mol_ids != reference_ids:
            raise ValueError(
                f"Molecule order mismatch on split '{split}' between Member 2's "
                "embeddings and Member 1's feature table."
            )
        blocks.append(embeddings)

    if not use_raw_features and use_descriptors:
        descriptors, descriptor_mol_ids = loader.load_features("descriptors_scaled", split=split)
        if list(descriptor_mol_ids) != reference_ids:
            raise ValueError(
                f"Molecule order mismatch on split '{split}' between Member 1's "
                "descriptor table and the embedding rows."
            )
        blocks.append(np.asarray(descriptors, dtype=np.float64))

    features = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)

    if features.shape[0] != len(reference_ids):
        raise ValueError(
            f"Row count mismatch on split '{split}': {features.shape[0]} feature rows "
            f"for {len(reference_ids)} molecule ids."
        )

    return features, reference_ids


def get_feature_names(use_raw_features=True, use_descriptors=True, use_embeddings=True):
    """Returns the column names of the hybrid feature matrix, for feature importance plots."""
    names = []
    if use_raw_features:
        names += [f"morgan_{i:04d}" for i in range(N_MORGAN_BITS)]
        names += DESCRIPTOR_NAMES
    if use_embeddings:
        names += [f"emb_{i:03d}" for i in range(N_EMBEDDINGS)]
    if not use_raw_features and use_descriptors:
        names += DESCRIPTOR_NAMES
    return names
