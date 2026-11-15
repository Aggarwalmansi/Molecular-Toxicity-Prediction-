"""
train_ann_transformer.py
========================
Trains Experiment 2 (the ANN + Transformer hybrid) and its two ablations.

Three configurations are trained with the SAME code and the SAME splits, so
any difference between them comes from the architecture alone:

  ann_only       fingerprint branch only      (sanity check against Member 2)
  smiles_only    Transformer branch only      (how much is in the text alone)
  hybrid         both branches, fused         (Experiment 2)

Each is trained with three random seeds. We report mean and standard
deviation, because a single lucky run must not decide anything.

Rules followed here:
  * The frozen Phase 1 scaffold split is used exactly as it is.
  * Missing labels are excluded through the observation mask. A missing label
    is never read as "not toxic".
  * The token vocabulary is built on TRAINING molecules only.
  * Early stopping watches VALIDATION macro ROC-AUC. Test is never looked at
    during training.

Run from the repository root:
    python -m phase2.experiments.train_ann_transformer
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from phase2.experiments.ann_transformer import AnnTransformerHybrid
from phase2.experiments.smiles_tokenizer import build_vocabulary, encode_many, split_smiles
from phase2.data.src.data_loader import TARGET_ASSAYS, Tox21DataLoader
from phase2.deep_learning.src.loss import MaskedWeightedBCEWithLogitsLoss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_FILE = PROJECT_ROOT / "tox21_member2_handoff" / "artifacts" / "corrected_scaffold_split.csv"
RESULTS_DIR = PROJECT_ROOT / "phase2" / "experiments" / "results"
PREDICTIONS_DIR = PROJECT_ROOT / "phase2" / "experiments" / "predictions"
CHECKPOINT_DIR = PROJECT_ROOT / "phase2" / "experiments" / "checkpoints"

MAX_LENGTH = 128        # covers about 99% of molecules without cutting
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 6
LEARNING_RATE = 1e-3
SEEDS = [7, 21, 42]
SPLITS = ["train", "val", "test"]


def load_smiles_by_split():
    """Reads the canonical SMILES for each split, in the loader's row order."""
    table = pd.read_csv(SPLIT_FILE)
    by_split = {}
    for split in SPLITS:
        wanted = "val" if split == "val" else split
        rows = table[table["split"] == wanted]
        by_split[split] = {
            "mol_ids": rows["mol_id"].tolist(),
            "smiles": rows["canonical_smiles"].tolist(),
        }
    return by_split


def build_tensors(loader, smiles_by_split, vocabulary):
    """
    Builds the fingerprint, token, label and mask tensors for every split.

    The assert is the safety net: if the SMILES table and the feature table
    ever fall out of order, we stop instead of silently pairing the wrong
    molecule with the wrong label.
    """
    data = {}
    for split in SPLITS:
        features, feature_ids = loader.load_features("combined", split=split)
        labels, mask, label_ids = loader.load_labels(split=split)
        assert feature_ids == label_ids == smiles_by_split[split]["mol_ids"], (
            f"Molecule order mismatch in the {split} split."
        )

        tokens = encode_many(smiles_by_split[split]["smiles"], vocabulary, MAX_LENGTH)

        data[split] = {
            "fingerprints": torch.tensor(features, dtype=torch.float32),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            # NaNs become 0.0 so the tensor maths works; the mask makes sure
            # those positions contribute exactly zero loss.
            "labels": torch.tensor(np.nan_to_num(labels, nan=0.0), dtype=torch.float32),
            "mask": torch.tensor(mask.astype(np.float32), dtype=torch.float32),
            "mol_ids": feature_ids,
            "raw_labels": labels,
            "raw_mask": mask,
        }
    return data


def macro_roc_auc(probabilities, raw_labels, raw_mask):
    """Average ROC-AUC over the endpoints, using observed labels only."""
    scores = []
    for endpoint_index in range(len(TARGET_ASSAYS)):
        observed = raw_mask[:, endpoint_index]
        y_true = raw_labels[observed, endpoint_index].astype(int)
        if len(np.unique(y_true)) > 1:
            scores.append(roc_auc_score(y_true, probabilities[observed, endpoint_index]))
    return float(np.mean(scores))


def predict(model, split_data):
    """Runs the model over a whole split and returns sigmoid probabilities."""
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(split_data["mol_ids"]), 256):
            stop = start + 256
            logits = model(
                split_data["fingerprints"][start:stop],
                split_data["tokens"][start:stop],
            )
            outputs.append(torch.sigmoid(logits))
    return torch.cat(outputs).numpy()


def train_one_model(data, vocabulary, use_fingerprints, use_smiles, seed, class_weights):
    """
    Trains one configuration with one seed and returns the best model.

    Early stopping keeps the weights from the epoch with the highest
    VALIDATION macro ROC-AUC. The .clone() below is deliberate: state_dict()
    hands back references to the live tensors, so without cloning the saved
    copy would keep following the model and we would end up restoring the last
    epoch instead of the best one.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = AnnTransformerHybrid(
        vocab_size=len(vocabulary),
        max_length=MAX_LENGTH,
        use_fingerprints=use_fingerprints,
        use_smiles=use_smiles,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = MaskedWeightedBCEWithLogitsLoss(pos_weight=torch.tensor(class_weights))

    train_set = TensorDataset(
        data["train"]["fingerprints"],
        data["train"]["tokens"],
        data["train"]["labels"],
        data["train"]["mask"],
    )
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    best_score = -1.0
    best_epoch = 0
    best_weights = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for fingerprints, tokens, labels, mask in train_loader:
            optimiser.zero_grad()
            logits = model(fingerprints, tokens)
            loss = loss_function(logits, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            running_loss += loss.item()

        val_probabilities = predict(model, data["val"])
        val_score = macro_roc_auc(
            val_probabilities, data["val"]["raw_labels"], data["val"]["raw_mask"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(running_loss / len(train_loader), 4),
                "val_roc_auc": round(val_score, 4),
            }
        )

        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_weights)
    return model, best_score, best_epoch, history


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    loader = Tox21DataLoader()
    smiles_by_split = load_smiles_by_split()

    vocabulary = build_vocabulary(smiles_by_split["train"]["smiles"])
    token_lengths = [len(split_smiles(s)) for s in smiles_by_split["train"]["smiles"]]
    print(f"Vocabulary built on training molecules only: {len(vocabulary)} tokens")
    print(f"Token length: median {int(np.median(token_lengths))}, "
          f"99th percentile {int(np.percentile(token_lengths, 99))}, "
          f"max {max(token_lengths)}, cut at {MAX_LENGTH}")
    print(f"Molecules longer than {MAX_LENGTH} tokens: "
          f"{sum(1 for n in token_lengths if n > MAX_LENGTH)} of {len(token_lengths)}")

    with open(RESULTS_DIR / "smiles_vocabulary.json", "w", encoding="utf-8") as handle:
        json.dump(vocabulary, handle, indent=2)

    data = build_tensors(loader, smiles_by_split, vocabulary)
    class_weights = loader.get_class_weights()

    configurations = [
        ("ann_only", True, False),
        ("smiles_only", False, True),
        ("hybrid", True, True),
    ]

    summary_rows = []
    prediction_rows = []
    history_rows = []

    for name, use_fingerprints, use_smiles in configurations:
        for seed in SEEDS:
            started = time.time()
            model, val_score, best_epoch, history = train_one_model(
                data, vocabulary, use_fingerprints, use_smiles, seed, class_weights
            )
            elapsed = time.time() - started

            test_probabilities = predict(model, data["test"])
            val_probabilities = predict(model, data["val"])
            test_score = macro_roc_auc(
                test_probabilities, data["test"]["raw_labels"], data["test"]["raw_mask"]
            )

            run_name = f"{name}_seed{seed}"
            torch.save(
                {"model_state_dict": model.state_dict(), "vocab_size": len(vocabulary),
                 "use_fingerprints": use_fingerprints, "use_smiles": use_smiles,
                 "best_epoch": best_epoch},
                CHECKPOINT_DIR / f"{run_name}.pt",
            )

            summary_rows.append(
                {
                    "configuration": name,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "val_roc_auc": round(val_score, 4),
                    "test_roc_auc": round(test_score, 4),
                    "minutes": round(elapsed / 60, 2),
                }
            )
            for record in history:
                record.update({"configuration": name, "seed": seed})
                history_rows.append(record)

            # Long format, matching generate_predictions.py so ensemble.py can
            # read these alongside the existing models.
            for split, probabilities in [("val", val_probabilities), ("test", test_probabilities)]:
                split_data = data[split]
                for endpoint_index, endpoint in enumerate(TARGET_ASSAYS):
                    prediction_rows.append(
                        pd.DataFrame(
                            {
                                "mol_id": split_data["mol_ids"],
                                "endpoint": endpoint,
                                "split": split,
                                "model": run_name,
                                "predicted_prob": probabilities[:, endpoint_index],
                                "true_label": split_data["raw_labels"][:, endpoint_index],
                                "observed": split_data["raw_mask"][:, endpoint_index],
                            }
                        )
                    )

            print(f"  {run_name:22s} best epoch {best_epoch:2d}  "
                  f"val {val_score:.4f}  test {test_score:.4f}  ({elapsed/60:.1f} min)")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "ann_transformer_runs.csv", index=False)
    pd.DataFrame(history_rows).to_csv(RESULTS_DIR / "ann_transformer_history.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_parquet(
        PREDICTIONS_DIR / "ann_transformer_predictions.parquet", index=False
    )

    grouped = summary.groupby("configuration").agg(
        val_mean=("val_roc_auc", "mean"),
        val_std=("val_roc_auc", "std"),
        test_mean=("test_roc_auc", "mean"),
        test_std=("test_roc_auc", "std"),
    ).round(4)
    grouped.to_csv(RESULTS_DIR / "ann_transformer_summary.csv")

    print()
    print("Mean +/- standard deviation over three seeds:")
    print(grouped.to_string())
    print()
    print("Selection is made on the validation column. The test column is")
    print("printed for the record only.")


if __name__ == "__main__":
    main()
