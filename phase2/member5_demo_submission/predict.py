"""Inference only: reuse the team's frozen preprocessing and trained models.

Run from the repository root:
python -m phase2.member5_demo_submission.predict --smiles 'CCO'
"""
import argparse
import json
import hashlib
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from phase2.member2_deep_learning_model.src.model import MultiTaskToxicNet

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / 'tox21_member2_handoff/artifacts'
DL_DIR = ROOT / 'phase2/member2_deep_learning_model'
HYBRID_DIR = Path(__file__).resolve().parent / 'models/hybrid'
ML_DIR = Path(__file__).resolve().parent / 'models'
CONFIG = json.loads((DL_DIR / 'configs/model_config.json').read_text())
ENDPOINTS = CONFIG['target_assays']


def featurize(smiles):
    """Match the saved Morgan and descriptor order exactly; reject invalid input."""
    from rdkit import Chem
    from tox21_member2_handoff.features import (
        compute_morgan_fingerprints, compute_physicochemical_descriptors,
    )
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('Enter a non-empty SMILES string.')
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid SMILES: enter a valid molecular structure.')
    canonical = Chem.MolToSmiles(mol)
    bits, failures = compute_morgan_fingerprints([canonical])
    descriptors, errors = compute_physicochemical_descriptors([canonical])
    if failures or errors or not np.isfinite(descriptors).all():
        raise ValueError('Could not calculate molecular features.')
    scaler = json.loads((ARTIFACTS / 'scaler_descriptors.json').read_text())
    std = np.asarray(scaler['std'])
    scaled = (descriptors - scaler['mean']) / np.where(std == 0, 1, std)
    return canonical, np.concatenate([bits, scaled], axis=1).astype(np.float32), scaled


@lru_cache(maxsize=2)
def load_neural_model(verify=True):
    """Load the trained network, refusing a checkpoint the bundle was not built from.

    prepare_models passes verify=False: it is the step that re-stamps the hash,
    so it must be able to load a checkpoint the current bundle predates.
    Prediction always verifies, so the demo can never score a molecule with a
    network the shipped Random Forests were not fitted against.
    """
    checkpoint_path = DL_DIR / 'checkpoints/best_multitask_model.pt'
    metadata_path = ML_DIR / 'bundle.json'
    if verify and metadata_path.exists():
        expected = json.loads(metadata_path.read_text())['checkpoint_sha256']
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != expected:
            raise ValueError('Checkpoint changed. Run prepare_models again before predicting.')
    checkpoint = torch.load(checkpoint_path,
                            map_location='cpu', weights_only=True)
    config = {**CONFIG, **checkpoint['config']}
    keys = ['input_dim', 'hidden_dims', 'latent_dim', 'num_endpoints',
            'dropout', 'norm_type', 'activation']
    if config['target_assays'] != ENDPOINTS:
        raise ValueError('Checkpoint endpoint order does not match the model config.')
    model = MultiTaskToxicNet(**{k: config[k] for k in keys})
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.eval()


@lru_cache(maxsize=36)
def load_forest(directory, endpoint):
    # Only load trusted model files shipped with this repository.
    return joblib.load(Path(directory) / f"rf_{endpoint.replace('-', '_')}.joblib")


def predict(smiles, threshold=0.5):
    """Return all available model scores; do not train or access labels here."""
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between zero and one.')
    canonical, features, descriptors = featurize(smiles)
    with torch.inference_mode():
        logits, embedding = load_neural_model().forward_with_embeddings(torch.from_numpy(features))
        neural = torch.sigmoid(logits).numpy()[0]
    hybrid = np.concatenate([embedding.numpy(), descriptors], axis=1)
    rows = []
    for i, endpoint in enumerate(ENDPOINTS):
        scores = {'Multitask DL': float(neural[i])}
        for name, directory, inputs in [('Hybrid', HYBRID_DIR, hybrid),
                                         ('Random Forest', ML_DIR, features)]:
            path = directory / f"rf_{endpoint.replace('-', '_')}.joblib"
            if path.exists():
                forest = load_forest(str(directory), endpoint)
                positive_column = list(forest.classes_).index(1)
                scores[name] = float(forest.predict_proba(inputs)[0, positive_column])
        for name, score in scores.items():
            rows.append({'endpoint': endpoint, 'model': name, 'activity_score': score,
                         'prediction': 'Active' if score >= threshold else 'Inactive',
                         'threshold': threshold, 'canonical_smiles': canonical})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        result = predict(args.smiles)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.output:
        result.to_csv(args.output, index=False)
    else:
        print(result.to_csv(index=False))


if __name__ == '__main__':
    main()
