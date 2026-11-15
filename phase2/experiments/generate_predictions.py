"""
generate_predictions.py
=======================
A1: Generates validation AND test predictions for every model we want to
compare or combine.

Why this script exists: the repository previously saved test predictions only.
Without validation predictions you cannot choose an ensemble weight, a blend
method or a decision threshold honestly, because the only scores available to
choose from would be test scores.

Output: one long-format table with columns
    mol_id, endpoint, split, model, predicted_prob, true_label, observed

Run from the repository root:
    python -m phase2.experiments.generate_predictions

Approximate runtime: under one minute. Nothing here trains a neural network;
the deep learning predictions come from checkpoints that already exist.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from phase2.data.src.data_loader import Tox21DataLoader
from phase2.deep_learning.src.model import MultiTaskToxicNet
from phase2.hybrid.src.hybrid_model import TARGET_ASSAYS
from phase2.data.src.data_loader import PREDICTIONS_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLITS = ["val", "test"]


def build_long_table(probabilities_by_split, model_name, loader):
    """
    Turns a dictionary of {split: probability array} into long format.

    One row per molecule, endpoint and split. The observed column records
    whether a real label existed, so nothing downstream can accidentally
    score against a missing label.
    """
    rows = []
    for split in probabilities_by_split:
        probabilities = probabilities_by_split[split]
        labels, mask, mol_ids = loader.load_labels(split=split)

        for row_index in range(len(mol_ids)):
            for endpoint_index, endpoint in enumerate(TARGET_ASSAYS):
                is_observed = bool(mask[row_index, endpoint_index])
                rows.append(
                    {
                        "mol_id": mol_ids[row_index],
                        "endpoint": endpoint,
                        "split": split,
                        "model": model_name,
                        "predicted_prob": float(probabilities[row_index, endpoint_index]),
                        "true_label": float(labels[row_index, endpoint_index]) if is_observed else np.nan,
                        "observed": is_observed,
                    }
                )
    return pd.DataFrame(rows)


def predict_with_network(checkpoint_path, loader, dropout):
    """
    Loads one saved network and predicts probabilities for validation and test.

    No training happens here. We read the weights that were already saved and
    run them forward in eval mode, which turns dropout off.

    The dropout value matters even for loading: when dropout is 0 the model
    builds no Dropout layer at all, which shifts the numbering of every later
    layer. Passing the wrong value makes load_state_dict fail.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model = MultiTaskToxicNet(
        input_dim=2053,
        hidden_dims=[512, 256],
        latent_dim=128,
        num_endpoints=12,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities_by_split = {}
    for split in SPLITS:
        features, _ = loader.load_features("combined", split=split)
        with torch.no_grad():
            logits = model(torch.tensor(features, dtype=torch.float32))
            probabilities_by_split[split] = torch.sigmoid(logits).numpy()

    return probabilities_by_split


def predict_with_phase1_forest(loader):
    """
    Retrains the Phase 1 Random Forest and predicts validation and test.

    Phase 1 saved its scores but not its fitted models, so we refit with the
    exact settings from train_models.py. Takes about 3 seconds in total.
    The training split alone is used for fitting.
    """
    train_features, _ = loader.load_features("combined", split="train")
    train_labels, train_mask, _ = loader.load_labels(split="train")

    split_features = {}
    for split in SPLITS:
        split_features[split], _ = loader.load_features("combined", split=split)

    probabilities_by_split = {}
    for split in SPLITS:
        probabilities_by_split[split] = np.zeros((len(split_features[split]), len(TARGET_ASSAYS)))

    for endpoint_index in range(len(TARGET_ASSAYS)):
        observed = np.where(train_mask[:, endpoint_index])[0]
        forest = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        forest.fit(train_features[observed], train_labels[observed, endpoint_index].astype(int))

        for split in SPLITS:
            probabilities_by_split[split][:, endpoint_index] = forest.predict_proba(
                split_features[split]
            )[:, 1]

    return probabilities_by_split


def predict_with_phase1_baseline(kind, loader):
    """
    Refits the Phase 1 Logistic Regression or Decision Tree.

    Phase 1 saved only test predictions for these two, and an ensemble weight
    has to be chosen on validation, so we need validation predictions as well.
    The settings below are copied from train_models.py (get_models). That file
    is Phase 1 and is only read, never edited.
    """
    if kind == "lr":
        def make_model():
            return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    elif kind == "dt":
        def make_model():
            return DecisionTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown baseline '{kind}'. Use 'lr' or 'dt'.")

    train_features, _ = loader.load_features("combined", split="train")
    train_labels, train_mask, _ = loader.load_labels(split="train")

    split_features = {}
    probabilities_by_split = {}
    for split in SPLITS:
        split_features[split], _ = loader.load_features("combined", split=split)
        probabilities_by_split[split] = np.zeros(
            (len(split_features[split]), len(TARGET_ASSAYS))
        )

    for endpoint_index in range(len(TARGET_ASSAYS)):
        observed = np.where(train_mask[:, endpoint_index])[0]
        model = make_model()
        model.fit(train_features[observed], train_labels[observed, endpoint_index].astype(int))
        for split in SPLITS:
            probabilities_by_split[split][:, endpoint_index] = model.predict_proba(
                split_features[split]
            )[:, 1]

    return probabilities_by_split


def add_phase1_baselines():
    """
    Writes validation and test predictions for the two Phase 1 baselines into
    their own file, so the eleven predictions that already exist are left
    alone rather than regenerated.
    """
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    loader = Tox21DataLoader()

    tables = []
    for kind, model_name in [("lr", "lr_phase1"), ("dt", "dt_phase1")]:
        print(f"  refitting {model_name} on the training split")
        probabilities = predict_with_phase1_baseline(kind, loader)
        tables.append(build_long_table(probabilities, model_name, loader))

    output_file = PREDICTIONS_DIR / "phase1_baseline_predictions.parquet"
    pd.concat(tables, ignore_index=True).to_parquet(output_file, index=False)
    print(f"  saved {output_file.name}")


def get_network_runs():
    """
    Lists the saved deep learning runs to generate predictions for.

    These are the nine multi-task ablation runs (three configurations, three
    seeds each) plus the main deep learning checkpoint.

    The no_dropout checkpoints were archived to archive/ to keep the
    repository small, so those three runs are skipped unless you move them
    back. Nothing downstream needs them: the ensemble uses the six
    class_weight runs, which are still here.
    """
    member4_dir = PROJECT_ROOT / "phase2" / "evaluation" / "checkpoints"
    runs = []

    for configuration, folder in [
        ("weighted", "class_weight"),
        ("unweighted", "class_weight"),
        ("no_dropout", "dropout"),
    ]:
        for seed in [7, 21, 42]:
            if configuration == "no_dropout":
                filename = f"model_no_dropout_seed_{seed}.pt"
            else:
                filename = f"model_{configuration}_loss_seed_{seed}.pt"
            path = member4_dir / folder / filename
            dropout = 0.0 if configuration == "no_dropout" else 0.35
            if path.exists():
                runs.append((f"dl_{configuration}_seed{seed}", path, dropout))

    member2_checkpoint = (
        PROJECT_ROOT / "phase2" / "deep_learning" / "checkpoints" / "best_multitask_model.pt"
    )
    if member2_checkpoint.exists():
        runs.append(("dl_member2_checkpoint", member2_checkpoint, 0.35))

    return runs


def main():
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    loader = Tox21DataLoader()

    all_tables = []

    print("Generating validation and test predictions")
    print("=" * 60)

    print("Deep learning runs (loading saved checkpoints, no training):")
    for model_name, checkpoint_path, dropout in get_network_runs():
        probabilities = predict_with_network(checkpoint_path, loader, dropout)
        all_tables.append(build_long_table(probabilities, model_name, loader))
        print(f"  {model_name}")

    print("Phase 1 Random Forest (refitting on the training split, about 3 seconds):")
    forest_probabilities = predict_with_phase1_forest(loader)
    all_tables.append(build_long_table(forest_probabilities, "rf_phase1", loader))
    print("  rf_phase1")

    predictions = pd.concat(all_tables, ignore_index=True)

    output_file = PREDICTIONS_DIR / "val_test_predictions.parquet"
    predictions.to_parquet(output_file, index=False)

    print()
    print(f"Saved {len(predictions):,} rows to {output_file.relative_to(PROJECT_ROOT)}")
    print(f"  models: {predictions['model'].nunique()}")
    print(f"  splits: {sorted(predictions['split'].unique())}")
    print(f"  observed rows per split:")
    for split in SPLITS:
        subset = predictions[(predictions["split"] == split) & predictions["observed"]]
        print(f"    {split}: {len(subset) // predictions['model'].nunique():,} per model")


if __name__ == "__main__":
    main()
