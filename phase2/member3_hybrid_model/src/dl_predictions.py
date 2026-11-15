"""
dl_predictions.py
=================
Loads Member 2's trained MultiTaskToxicNet and scores a split with it, so the
hybrid Random Forest can be blended against it.

We rebuild the network from Member 2's own model_config.json and load their
saved checkpoint weights. Nothing is retrained here: the probabilities are
exactly what Member 2's model produces, which keeps the ensemble honest.
"""

from pathlib import Path

import json
import numpy as np
import torch

from phase2.member2_deep_learning_model.src.model import MultiTaskToxicNet


def get_member2_dir():
    """Returns the member2_deep_learning_model folder."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "phase2" / "member2_deep_learning_model"


def pick_device():
    """Uses Apple MPS or CUDA when available, otherwise CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_trained_dl_model(device=None):
    """
    Rebuilds MultiTaskToxicNet from Member 2's config and loads the best checkpoint.

    Returns the model in eval mode on the chosen device.
    """
    member2_dir = get_member2_dir()
    config_file = member2_dir / "configs" / "model_config.json"
    checkpoint_file = member2_dir / "checkpoints" / "best_multitask_model.pt"

    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_file}. Run "
            "`python -m phase2.member2_deep_learning_model.train` first."
        )

    with open(config_file, "r") as f:
        config = json.load(f)

    device = device or pick_device()

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
    )

    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_dl_probabilities(loader, split, model=None, device=None):
    """
    Scores one split with Member 2's network.

    Returns:
        probabilities: array of shape (n_molecules, 12) of sigmoid outputs
        mol_ids:       list of molecule ids in the same row order
    """
    device = device or pick_device()
    model = model or load_trained_dl_model(device=device)

    features, mol_ids = loader.load_features("combined", split=split)
    x = torch.tensor(np.asarray(features, dtype=np.float32), dtype=torch.float32).to(device)

    with torch.no_grad():
        probabilities = torch.sigmoid(model(x)).cpu().numpy().astype(np.float64)

    return probabilities, list(mol_ids)
