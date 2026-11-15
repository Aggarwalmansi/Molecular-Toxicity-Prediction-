"""
train.py
========
End-to-End Orchestrator for Phase 2 Member 2: Deep Learning Model.
Trains MultiTaskToxicNet on canonical 2,053-dim features, applies masked loss and
class weighting, logs learning curves, restores best checkpoint, extracts molecular
embeddings for Member 3, and conducts holdout test benchmark evaluation.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader
from phase2.member2_deep_learning_model.src.model import MultiTaskToxicNet
from phase2.member2_deep_learning_model.src.trainer import MultiTaskTrainer
from phase2.member2_deep_learning_model.src.extract_embeddings import extract_and_save_embeddings
from phase2.member2_deep_learning_model.src.evaluate import evaluate_multitask_model


def set_seed(seed: int = 42):
    """Ensure strict reproducibility across all runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def main():
    parser = argparse.ArgumentParser(description="Train Phase 2 MultiTaskToxicNet for Tox21")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "configs" / "model_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # CLI overrides
    batch_size = args.batch_size or config["training"]["batch_size"]
    max_epochs = args.epochs or config["training"]["max_epochs"]
    learning_rate = args.lr or config["training"]["learning_rate"]
    patience = args.patience or config["training"]["early_stopping_patience"]
    weight_decay = config["training"]["weight_decay"]

    print("=" * 70)
    print("PHASE 2 — MEMBER 2: MULTI-TASK DEEP LEARNING MODEL TRAINING")
    print("=" * 70)
    print(f"Feature space: {config['feature_set']} ({config['input_dim']} dims)")
    print(f"Architecture:  Input({config['input_dim']}) -> {config['hidden_dims']} -> Bottleneck({config['latent_dim']}) -> 12 TaskHeads")
    print(f"Improvements: LayerNorm, GELU, Residual Connections, Per-Task Heads, Sqrt-Damped Weights")
    print(f"Regularization: InputDropout={config.get('input_dropout', 0.0)}, Dropout={config['dropout']}, HeadDropout={config.get('head_dropout', 0.2)}, Weight Decay={weight_decay}")
    print(f"Training Plan: Batch Size={batch_size}, Max Epochs={max_epochs}, Patience={patience}, LR={learning_rate}")
    print("=" * 70)

    # 1. Initialize Member 1 Data Loader
    print("[1/5] Loading data from Member 1 certified pipeline...")
    loader = Tox21DataLoader()
    train_loader, val_loader, test_loader = loader.get_pytorch_dataloaders(
        feature_set=config["feature_set"],
        batch_size=batch_size,
        shuffle_train=True,
    )
    pos_weights = loader.get_class_weights(as_tensor=True)
    print(f"  └── DataLoaders created! Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    print(f"  └── Raw positive class weights (range: {pos_weights.min().item():.2f} - {pos_weights.max().item():.2f})")
    print(f"  └── Sqrt damping will be applied (cap={config['training'].get('weight_cap', 10.0)})")

    # 2. Instantiate Model
    print("[2/5] Initializing MultiTaskToxicNet (Improved Architecture)...")
    model = MultiTaskToxicNet(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        num_endpoints=config["num_endpoints"],
        dropout=config["dropout"],
        norm_type=config["norm_type"],
        activation=config["activation"],
        head_hidden=config.get("head_hidden", 64),
        head_dropout=config.get("head_dropout", 0.2),
        use_task_heads=config.get("use_task_heads", True),
        input_dropout=config.get("input_dropout", 0.0),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  └── Total trainable parameters: {total_params:,}")

    # 3. Train Model
    print("[3/5] Starting model training with masked loss and early stopping...")
    early_metric = config["training"].get("early_stopping_metric", "val_macro_roc_auc")
    trainer = MultiTaskTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        pos_weight=pos_weights,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        checkpoint_dir=base_dir / "checkpoints",
        report_dir=base_dir / "reports",
        target_assays=config["target_assays"],
        early_stopping_metric=early_metric,
        use_sqrt_damping=config["training"].get("use_sqrt_damping", True),
        weight_cap=config["training"].get("weight_cap", 10.0),
        scheduler_config=config["training"].get("lr_scheduler"),
    )
    fit_summary = trainer.fit(max_epochs=max_epochs, patience=patience)
    print(f"  └── Training completed at epoch {fit_summary['best_epoch']}!")
    print(f"      Best Val Loss: {fit_summary['best_val_loss']:.4f} | Best Val ROC-AUC: {fit_summary['best_val_macro_roc_auc']:.4f}")

    # 4. Extract Latent Molecular Embeddings for Member 3
    print("[4/5] Extracting learned molecular embeddings for Member 3 (Hybrid RF)...")
    emb_metadata = extract_and_save_embeddings(
        model=trainer.model,
        loader=loader,
        output_dir=base_dir / "embeddings",
        device=trainer.device,
    )

    # 5. Holdout Test Set Benchmark Evaluation
    print("[5/5] Conducting final holdout test evaluation (782 compounds)...")
    eval_df = evaluate_multitask_model(
        model=trainer.model,
        loader=loader,
        output_dir=base_dir / "reports",
        device=trainer.device,
    )
    print("\n" + "=" * 70)
    print("FINAL HOLDOUT TEST PERFORMANCE SUMMARY:")
    print("=" * 70)
    print(eval_df.to_string(index=False))
    print("=" * 70)
    print(f"[Done] All artifacts and handoffs successfully written to {base_dir}")


if __name__ == "__main__":
    main()
