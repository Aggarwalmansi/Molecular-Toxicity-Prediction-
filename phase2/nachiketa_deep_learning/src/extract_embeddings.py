"""
extract_embeddings.py
=====================
Extracts learned latent molecular representations (bottleneck embeddings z)
from MultiTaskToxicNet for Train, Validation, and Holdout Test splits.
These embeddings are consumed directly by Member 3 to construct the Hybrid ML Model.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from .model import MultiTaskToxicNet


def sha256_file(filepath: Union[str, Path]) -> str:
    """Compute sha256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def extract_and_save_embeddings(
    model: MultiTaskToxicNet,
    loader: Any,
    output_dir: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 128,
) -> Dict[str, Any]:
    """
    Extract latent bottleneck embeddings for train, val, and test splits.

    Parameters
    ----------
    model : MultiTaskToxicNet
        Trained model with restored best weights.
    loader : Tox21DataLoader
        Phase 2 Member 1 data loader instance.
    output_dir : Path or str, optional
        Target directory to save parquet and npy embeddings.
    device : torch.device, optional
        Computing device.
    batch_size : int, default 128
        Batch size for inference.

    Returns
    -------
    dict
        Metadata summarizing extracted embeddings.
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    base_dir = Path(__file__).resolve().parent.parent
    out_dir = Path(output_dir or (base_dir / "embeddings"))
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    model.to(device)

    splits = ["train", "val", "test"]
    metadata: Dict[str, Any] = {
        "latent_dim": model.latent_dim,
        "source_checkpoint": "best_multitask_model.pt",
        "partitions": {},
    }

    print(f"[extract_embeddings] Extracting {model.latent_dim}-dim embeddings using {device}...")

    for split in splits:
        X_split, mids = loader.load_features("combined", split=split)
        n_samples = len(X_split)

        all_embs = []
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch_x = torch.tensor(X_split[i : i + batch_size], dtype=torch.float32).to(device)
                z = model.get_embeddings(batch_x)
                all_embs.append(z.cpu().numpy())

        embeddings = np.vstack(all_embs).astype(np.float32)
        assert len(embeddings) == len(mids), f"Length mismatch: {len(embeddings)} vs {len(mids)}"

        # Save Parquet with molecule_id column
        emb_cols = [f"emb_{k:03d}" for k in range(model.latent_dim)]
        df_emb = pd.DataFrame(embeddings, columns=emb_cols)
        df_emb.insert(0, "molecule_id", mids)

        parquet_path = out_dir / f"{split}_embeddings.parquet"
        df_emb.to_parquet(parquet_path, index=False)

        # Also save raw .npy for convenient numpy loading
        npy_path = out_dir / f"{split}_embeddings.npy"
        np.save(npy_path, embeddings)

        metadata["partitions"][split] = {
            "count": n_samples,
            "dimensions": model.latent_dim,
            "parquet_file": parquet_path.name,
            "parquet_sha256": sha256_file(parquet_path),
            "npy_file": npy_path.name,
            "npy_sha256": sha256_file(npy_path),
        }
        print(f"  └── {split.capitalize()}: {n_samples} compounds -> {parquet_path.name} ({embeddings.shape})")

    meta_path = out_dir / "embeddings_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[extract_embeddings] Saved all embeddings and metadata to {out_dir}")
    return metadata
