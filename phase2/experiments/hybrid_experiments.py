"""
hybrid_experiments.py
=====================
A3: Improves the ML + DL hybrid model.

Background. The original hybrid trained a Random Forest on 128 embeddings
taken from Member 2's network. Those embeddings turned out to come from the
network's FINAL epoch rather than its best epoch, because of the state_dict
aliasing bug fixed in deep_learning/src/trainer.py. By the final
epoch the network had memorised the training set (train loss 0.15, validation
loss 4.0), so the training embeddings were almost perfectly separable while
the test embeddings were not. The second stage therefore learned a geometry
that does not exist at test time.

This script measures that effect and tests several fixes, choosing between
them on the VALIDATION split only.

Run from the repository root:
    python -m phase2.experiments.hybrid_experiments

Approximate runtime: about 3 minutes. No neural network is trained here; the
embeddings come from checkpoints that already exist.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from phase2.data.src.data_loader import Tox21DataLoader
from phase2.deep_learning.src.model import MultiTaskToxicNet
from phase2.hybrid.src.hybrid_model import TARGET_ASSAYS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "phase2" / "experiments" / "results"
SPLITS = ["train", "val", "test"]


def extract_embeddings(checkpoint_path, loader, dropout=0.35):
    """
    Runs a saved network forward and returns its 128-dim latent embeddings.

    eval() turns dropout and batch-norm updating off, so the embeddings are
    deterministic.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultiTaskToxicNet(
        input_dim=2053, hidden_dims=[512, 256], latent_dim=128,
        num_endpoints=12, dropout=dropout,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    embeddings = {}
    for split in SPLITS:
        features, _ = loader.load_features("combined", split=split)
        with torch.no_grad():
            embeddings[split] = model.extract_latent(
                torch.tensor(features, dtype=torch.float32)
            ).numpy().astype(np.float64)
    return embeddings


def measure_separability(embeddings, labels, mask, endpoint_index=0):
    """
    Measures how linearly separable the TRAINING embeddings are.

    A logistic regression that can fit the training labels almost perfectly
    means the network memorised them. The second-stage model then learns from
    a geometry that will not be present at test time.
    """
    observed = np.where(mask[:, endpoint_index])[0]
    x = embeddings[observed]
    y = labels[observed, endpoint_index].astype(int)

    classifier = LogisticRegression(max_iter=2000, class_weight="balanced")
    classifier.fit(x, y)
    return float(classifier.score(x, y))


def build_features(embeddings, loader, use_descriptors=True, use_morgan=False):
    """Joins the embedding block with the optional descriptor and Morgan blocks."""
    features = {}
    for split in SPLITS:
        parts = [embeddings[split]]
        if use_descriptors:
            descriptors, _ = loader.load_features("descriptors_scaled", split=split)
            parts.append(descriptors)
        if use_morgan:
            morgan, _ = loader.load_features("morgan", split=split)
            parts.append(morgan)
        features[split] = np.concatenate(parts, axis=1)
    return features


def make_classifier(kind):
    """Builds the second-stage model. RF matches the Phase 1 baseline settings."""
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    if kind == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=1000, min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier kind: {kind}")


def run_variant(features, loader, classifier_kind):
    """
    Trains one second-stage model per endpoint and scores validation and test.

    Missing labels are excluded per endpoint using the observation mask, so a
    missing label is never treated as non-toxic.
    """
    train_labels, train_mask, _ = loader.load_labels(split="train")

    scores = {}
    probabilities = {}
    for split in ["val", "test"]:
        probabilities[split] = np.zeros((len(features[split]), len(TARGET_ASSAYS)))


    for endpoint_index in range(len(TARGET_ASSAYS)):
        observed = np.where(train_mask[:, endpoint_index])[0]
        model = make_classifier(classifier_kind)
        model.fit(features["train"][observed], train_labels[observed, endpoint_index].astype(int))
        for split in ["val", "test"]:
            probabilities[split][:, endpoint_index] = model.predict_proba(features[split])[:, 1]

    for split in ["val", "test"]:
        labels, mask, _ = loader.load_labels(split=split)
        roc_scores = []
        pr_scores = []
        for endpoint_index in range(len(TARGET_ASSAYS)):
            observed = mask[:, endpoint_index]
            y_true = labels[observed, endpoint_index].astype(int)
            if len(np.unique(y_true)) > 1:
                y_prob = probabilities[split][observed, endpoint_index]
                roc_scores.append(roc_auc_score(y_true, y_prob))
                pr_scores.append(average_precision_score(y_true, y_prob))
        scores[f"{split}_roc_auc"] = round(float(np.mean(roc_scores)), 4)
        scores[f"{split}_pr_auc"] = round(float(np.mean(pr_scores)), 4)

    return scores, probabilities


def build_long_table(probabilities, loader, model_name):
    """
    Puts one variant's predictions into the same long format the other scripts
    use, so the final comparison can read every model from one place.
    """
    tables = []
    for split in ["val", "test"]:
        labels, mask, mol_ids = loader.load_labels(split=split)
        for endpoint_index, endpoint in enumerate(TARGET_ASSAYS):
            tables.append(
                pd.DataFrame(
                    {
                        "mol_id": mol_ids,
                        "endpoint": endpoint,
                        "split": split,
                        "model": model_name,
                        "predicted_prob": probabilities[split][:, endpoint_index],
                        "true_label": labels[:, endpoint_index],
                        "observed": mask[:, endpoint_index],
                    }
                )
            )
    return pd.concat(tables, ignore_index=True)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    loader = Tox21DataLoader()
    train_labels, train_mask, _ = loader.load_labels(split="train")

    member2_checkpoint = (
        PROJECT_ROOT / "phase2" / "deep_learning" / "checkpoints" / "best_multitask_model.pt"
    )
    member4_checkpoint = (
        PROJECT_ROOT / "phase2" / "evaluation" / "checkpoints" / "class_weight" / "model_weighted_loss_seed_42.pt"
    )

    print("=" * 72)
    print("STEP 1: WHY THE ORIGINAL HYBRID FAILED")
    print("=" * 72)

    # The embeddings the original hybrid actually used, saved by Member 2
    original_embeddings_file = (
        PROJECT_ROOT / "phase2" / "deep_learning" / "embeddings" / "train_embeddings.parquet"
    )
    original_train = pd.read_parquet(original_embeddings_file)
    original_train = original_train.drop(columns=["molecule_id"]).values.astype(np.float64)

    original_separability = measure_separability(original_train, train_labels, train_mask)
    print(f"  Training embeddings used by the ORIGINAL hybrid (final-epoch model):")
    print(f"    logistic regression training accuracy = {original_separability:.4f}")

    corrected = extract_embeddings(member2_checkpoint, loader)
    corrected_separability = measure_separability(corrected["train"], train_labels, train_mask)
    print(f"  Training embeddings from the CORRECT best-epoch checkpoint:")
    print(f"    logistic regression training accuracy = {corrected_separability:.4f}")
    print()
    print("  The first number being near 1.0 means the network had memorised the")
    print("  training labels, so the hybrid learned a pattern that does not exist")
    print("  at test time.")

    print()
    print("=" * 72)
    print("STEP 2: HYBRID VARIANTS, SELECTED ON VALIDATION")
    print("=" * 72)

    member4_embeddings = extract_embeddings(member4_checkpoint, loader)

    variants = [
        ("original_hybrid_rf", {"embeddings": "original", "classifier": "rf",
                                "descriptors": True, "morgan": False}),
        ("corrected_emb_rf", {"embeddings": "corrected", "classifier": "rf",
                              "descriptors": True, "morgan": False}),
        ("corrected_emb_extratrees", {"embeddings": "corrected", "classifier": "extratrees",
                                      "descriptors": True, "morgan": False}),
        ("corrected_emb_plus_morgan_rf", {"embeddings": "corrected", "classifier": "rf",
                                          "descriptors": True, "morgan": True}),
        ("member4_emb_extratrees", {"embeddings": "member4", "classifier": "extratrees",
                                    "descriptors": True, "morgan": False}),
        ("member4_emb_plus_morgan_rf", {"embeddings": "member4", "classifier": "rf",
                                        "descriptors": True, "morgan": True}),
    ]

    # The original embeddings only exist as saved files, for all three splits
    original_embeddings = {}
    for split in SPLITS:
        name = "val" if split == "val" else split
        table = pd.read_parquet(
            PROJECT_ROOT / "phase2" / "deep_learning" / "embeddings" / f"{name}_embeddings.parquet"
        )
        original_embeddings[split] = table.drop(columns=["molecule_id"]).values.astype(np.float64)

    embedding_sources = {
        "original": original_embeddings,
        "corrected": corrected,
        "member4": member4_embeddings,
    }

    rows = []
    prediction_tables = []
    for name, settings in variants:
        embeddings = embedding_sources[settings["embeddings"]]
        features = build_features(
            embeddings, loader,
            use_descriptors=settings["descriptors"],
            use_morgan=settings["morgan"],
        )
        scores, probabilities = run_variant(features, loader, settings["classifier"])
        scores["variant"] = name
        scores["n_features"] = features["train"].shape[1]
        rows.append(scores)
        prediction_tables.append(build_long_table(probabilities, loader, f"fh_{name}"))
        print(f"  {name:32s} ({scores['n_features']:4d} feat)  "
              f"val ROC {scores['val_roc_auc']:.4f}  test ROC {scores['test_roc_auc']:.4f}")

    predictions_dir = PROJECT_ROOT / "phase2" / "experiments" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_tables, ignore_index=True).to_parquet(
        predictions_dir / "feature_hybrid_predictions.parquet", index=False
    )

    results = pd.DataFrame(rows)[
        ["variant", "n_features", "val_roc_auc", "val_pr_auc", "test_roc_auc", "test_pr_auc"]
    ].sort_values("val_roc_auc", ascending=False)
    results.to_csv(RESULTS_DIR / "hybrid_variants.csv", index=False)

    print()
    print(results.to_string(index=False))
    best = results.iloc[0]
    print()
    print(f"  Best on validation: {best['variant']} "
          f"(val {best['val_roc_auc']:.4f} -> test {best['test_roc_auc']:.4f})")


if __name__ == "__main__":
    main()
