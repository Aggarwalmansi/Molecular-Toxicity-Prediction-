"""
regularization_sweep.py
=======================
Diagnostic sweep used to cure the early-overfitting failure in MultiTaskToxicNet.

With the architecture fixes alone (LayerNorm, GELU, residual trunk, per-task
heads, sqrt-damped weights) the network still memorised the 6,259 training
molecules within about five epochs: train loss fell from 0.51 to 0.016 while
validation loss climbed from 0.60 to 2.22, so early stopping kept epoch 1 and
the learned embeddings were barely trained.

This script trains short runs over a grid of regularization strengths and
reports, for each, the best validation macro ROC-AUC and the epoch it occurred
on. A healthy configuration peaks well after epoch 2 and keeps validation loss
flat rather than exploding.

Run from the repository root:
    python -m phase2.member2_deep_learning_model.src.regularization_sweep
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader
from phase2.member2_deep_learning_model.src.loss import MaskedWeightedBCEWithLogitsLoss
from phase2.member2_deep_learning_model.src.model import MultiTaskToxicNet

TARGET_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

# Each entry varies only the regularization knobs; architecture stays fixed.
GRID = [
    {"name": "baseline (as specified)", "input_dropout": 0.0, "dropout": 0.30,
     "hidden_dims": [512, 256], "weight_decay": 1e-3, "lr": 3e-4},
    {"name": "input_drop 0.2", "input_dropout": 0.2, "dropout": 0.30,
     "hidden_dims": [512, 256], "weight_decay": 1e-3, "lr": 3e-4},
    {"name": "input_drop 0.2 + trunk 0.5", "input_dropout": 0.2, "dropout": 0.50,
     "hidden_dims": [512, 256], "weight_decay": 1e-3, "lr": 3e-4},
    {"name": "input_drop 0.3 + trunk 0.5", "input_dropout": 0.3, "dropout": 0.50,
     "hidden_dims": [512, 256], "weight_decay": 1e-3, "lr": 3e-4},
    {"name": "input_drop 0.3 + trunk 0.5 + wd 1e-2", "input_dropout": 0.3, "dropout": 0.50,
     "hidden_dims": [512, 256], "weight_decay": 1e-2, "lr": 3e-4},
    {"name": "small trunk [256,128] + input_drop 0.2", "input_dropout": 0.2, "dropout": 0.40,
     "hidden_dims": [256, 128], "weight_decay": 1e-2, "lr": 3e-4},
    {"name": "input_drop 0.3 + trunk 0.5 + wd 1e-2 + lr 1e-4", "input_dropout": 0.3,
     "dropout": 0.50, "hidden_dims": [512, 256], "weight_decay": 1e-2, "lr": 1e-4},
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, macro_roc_auc, macro_pr_auc) on observed labels only."""
    model.eval()
    total_loss, batches = 0.0, 0
    logits_all, targets_all, masks_all = [], [], []

    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits = model(x)
        total_loss += criterion(logits, y, m).item()
        batches += 1
        logits_all.append(logits.cpu())
        targets_all.append(y.cpu())
        masks_all.append(m.cpu())

    probs = torch.sigmoid(torch.cat(logits_all)).numpy()
    y_true = torch.cat(targets_all).numpy()
    mask = torch.cat(masks_all).numpy().astype(bool)

    rocs, prs = [], []
    for idx in range(len(TARGET_ASSAYS)):
        obs = mask[:, idx]
        y_obs, p_obs = y_true[obs, idx], probs[obs, idx]
        if len(np.unique(y_obs)) > 1:
            rocs.append(roc_auc_score(y_obs, p_obs))
            prs.append(average_precision_score(y_obs, p_obs))
        else:
            rocs.append(0.5)
            prs.append(0.0)

    return total_loss / max(1, batches), float(np.mean(rocs)), float(np.mean(prs))


def run_one(spec, data, epochs, device, seed=42):
    """Trains one configuration and returns its validation trajectory summary."""
    set_seed(seed)
    train_loader, val_loader, pos_weights = data

    model = MultiTaskToxicNet(
        input_dim=2053,
        hidden_dims=spec["hidden_dims"],
        latent_dim=128,
        num_endpoints=12,
        dropout=spec["dropout"],
        norm_type="layer_norm",
        activation="gelu",
        head_hidden=64,
        head_dropout=0.2,
        use_task_heads=True,
        input_dropout=spec["input_dropout"],
    ).to(device)

    criterion = MaskedWeightedBCEWithLogitsLoss(
        pos_weight=None if spec.get("weight_cap") == "none" else pos_weights,
        use_sqrt_damping=True,
        weight_cap=10.0 if spec.get("weight_cap") in (None, "none") else spec["weight_cap"],
    )
    optimizer = optim.AdamW(model.parameters(), lr=spec["lr"],
                            weight_decay=spec["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )

    best_roc, best_epoch, best_loss_at_best = -np.inf, 0, np.nan
    final_train_loss, final_val_loss = np.nan, np.nan

    for epoch in range(1, epochs + 1):
        model.train()
        running, batches = 0.0, 0
        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y, m)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running += loss.item()
            batches += 1
        scheduler.step(epoch)

        train_loss = running / max(1, batches)
        val_loss, val_roc, _ = evaluate(model, val_loader, criterion, device)
        final_train_loss, final_val_loss = train_loss, val_loss

        if val_roc > best_roc:
            best_roc, best_epoch, best_loss_at_best = val_roc, epoch, val_loss

    return {
        "config": spec["name"],
        "input_dropout": spec["input_dropout"],
        "trunk_dropout": spec["dropout"],
        "hidden_dims": str(spec["hidden_dims"]),
        "weight_decay": spec["weight_decay"],
        "lr": spec["lr"],
        "weight_cap": spec.get("weight_cap", 10.0),
        "best_val_roc_auc": round(best_roc, 4),
        "best_epoch": best_epoch,
        "val_loss_at_best": round(best_loss_at_best, 4),
        "final_train_loss": round(final_train_loss, 4),
        "final_val_loss": round(final_val_loss, 4),
        "train_val_loss_gap": round(final_val_loss - final_train_loss, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Regularization sweep for MultiTaskToxicNet")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    device = pick_device()
    loader = Tox21DataLoader()
    train_loader, val_loader, _ = loader.get_pytorch_dataloaders(
        feature_set="combined", batch_size=64, shuffle_train=True
    )
    pos_weights = loader.get_class_weights(as_tensor=True)
    data = (train_loader, val_loader, pos_weights)

    print(f"[sweep] device={device}, {args.epochs} epochs per configuration, "
          f"{len(GRID)} configurations")
    rows = []
    for spec in GRID:
        row = run_one(spec, data, args.epochs, device)
        rows.append(row)
        print(f"  {row['config']:46s} best val ROC-AUC {row['best_val_roc_auc']:.4f} "
              f"@ epoch {row['best_epoch']:3d} | final train {row['final_train_loss']:.3f} "
              f"val {row['final_val_loss']:.3f}")

    results = pd.DataFrame(rows).sort_values("best_val_roc_auc", ascending=False)
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "regularization_sweep.csv", index=False)
    print()
    print(results.to_string(index=False))
    print(f"\n[sweep] written to {out_dir / 'regularization_sweep.csv'}")


if __name__ == "__main__":
    main()
