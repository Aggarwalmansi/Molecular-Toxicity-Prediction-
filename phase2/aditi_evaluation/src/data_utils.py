"""
data_utils.py
=============
Data utility wrappers and validation helpers for Member 4 Evaluation & Ablation Studies.
Interfaces directly with Member 1's frozen scaffold split and feature contracts.
"""

import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from phase2.data.src.data_loader import Tox21DataLoader, TARGET_ASSAYS, DESCRIPTOR_NAMES

# Ensure matplotlib cache directory is set to writable directory
os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig"


def set_seed(seed: int = 42) -> None:
    """Ensure strict multi-platform reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class Tox21DataLoaderWrapper:
    """
    Certified interface around Member 1's Tox21DataLoader.
    Guarantees validation of split sizes, feature dimensions, and compound ID alignment.
    """

    def __init__(self, data_loader: Optional[Tox21DataLoader] = None):
        self.loader = data_loader or Tox21DataLoader()
        self.target_assays = list(TARGET_ASSAYS)
        self.validate_invariants()

    def validate_invariants(self) -> None:
        """Verify non-negotiable data contracts from Member 1."""
        train_idx = self.loader.get_split_indices("train")
        val_idx = self.loader.get_split_indices("val")
        test_idx = self.loader.get_split_indices("test")

        assert len(train_idx) == 6259, f"Train split count mismatch: {len(train_idx)} != 6259"
        assert len(val_idx) == 782, f"Validation split count mismatch: {len(val_idx)} != 782"
        assert len(test_idx) == 782, f"Test split count mismatch: {len(test_idx)} != 782"

        # Check feature shapes
        X_train, mids_x = self.loader.load_features("combined", split="train")
        y_train, mask_train, mids_y = self.loader.load_labels(split="train")
        assert X_train.shape == (6259, 2053), f"Combined feature shape mismatch: {X_train.shape}"
        assert y_train.shape == (6259, 12), f"Label shape mismatch: {y_train.shape}"
        assert mask_train.shape == (6259, 12), f"Mask shape mismatch: {mask_train.shape}"
        assert mids_x == mids_y, "Train molecule ID misalignment between features and labels"

    def get_multitask_loaders(
        self,
        batch_size: int = 64,
        feature_set: str = "combined",
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Obtain PyTorch DataLoaders for MultiTask modeling."""
        return self.loader.get_pytorch_dataloaders(
            feature_set=feature_set,
            batch_size=batch_size,
            shuffle_train=True,
        )

    def get_pos_weights(self, as_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
        """Obtain positive class weights strictly from training observed compounds."""
        return self.loader.get_class_weights(as_tensor=as_tensor)

    def get_single_task_data(
        self,
        endpoint: str,
        split: str = "train",
        feature_set: str = "combined",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract observed molecules for a single target assay.
        Filters out missing labels (NaN) using observation mask.

        Returns
        -------
        X_obs : np.ndarray of shape (N_obs, D)
        y_obs : np.ndarray of shape (N_obs, 1)
        mids_obs : List of observed molecule IDs
        """
        if endpoint not in self.target_assays:
            raise ValueError(f"Endpoint '{endpoint}' not in target assays {self.target_assays}")

        ep_idx = self.target_assays.index(endpoint)
        X, mids = self.loader.load_features(feature_set=feature_set, split=split)
        y, mask, mids_y = self.loader.load_labels(split=split)
        assert mids == mids_y, f"ID mismatch for {split}"

        obs_mask = mask[:, ep_idx].astype(bool)
        X_obs = X[obs_mask]
        y_obs = y[obs_mask, ep_idx : ep_idx + 1].astype(np.float32)
        mids_obs = [mids[i] for i, is_obs in enumerate(obs_mask) if is_obs]

        return X_obs, y_obs, mids_obs

    def get_single_task_pos_weight(self, endpoint: str) -> float:
        """Compute positive class weight for a single endpoint strictly on train."""
        _, y_train, _ = self.get_single_task_data(endpoint, split="train")
        n_pos = float((y_train == 1.0).sum())
        n_neg = float((y_train == 0.0).sum())
        if n_pos > 0:
            return float(n_neg / n_pos)
        return 1.0

    def get_single_task_loaders(
        self,
        endpoint: str,
        batch_size: int = 64,
        feature_set: str = "combined",
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Generate PyTorch DataLoaders for a single endpoint with observed labels only."""
        X_tr, y_tr, _ = self.get_single_task_data(endpoint, "train", feature_set)
        X_va, y_va, _ = self.get_single_task_data(endpoint, "val", feature_set)
        X_te, y_te, _ = self.get_single_task_data(endpoint, "test", feature_set)

        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32))
        val_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.float32))
        test_ds = TensorDataset(torch.tensor(X_te, dtype=torch.float32), torch.tensor(y_te, dtype=torch.float32))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def get_raw_descriptors(self, split: str = "test") -> Tuple[pd.DataFrame, List[str]]:
        """Load continuous molecular descriptors for chemical error analysis."""
        X_desc, mids = self.loader.load_features("descriptors_raw", split=split)
        df_desc = pd.DataFrame(X_desc, columns=DESCRIPTOR_NAMES)
        df_desc["mol_id"] = mids
        return df_desc, mids

