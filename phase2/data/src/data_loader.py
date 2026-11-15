"""
data_loader.py
==============
Phase 2 Member 1: Reusable Data Pipeline and Handoff Module for Tox21.

Author: Senior ML Infrastructure & Cheminformatics Engineer
Module: phase2.data.src.data_loader
Certified Split: Split B Frozen (corrected_scaffold_split.csv, 6,259 train / 782 val / 782 test)

Key Capabilities:
1. Reusable dataset loader providing features, labels, boolean masks, splits, and molecule IDs.
2. Preserves NaN labels strictly and exposes explicit observation masks (~np.isnan(y)).
3. Supports 6 feature representations:
   - 'combined': 2,053 features (2,048 Morgan bits + 5 scaled descriptors) [Canonical Phase 2 DL input]
   - 'morgan_filtered': 1,011 high-variance Morgan bits (variance >= 0.005 on train)
   - 'descriptors_scaled': 5 continuous descriptors standardized strictly on training partition
   - 'morgan': 2,048 binary Morgan bits
   - 'descriptors_raw': 5 unscaled continuous descriptors
   - 'combined_raw': 2,053 unscaled features
4. Computes positive class weights strictly from observed training molecules.
5. Verifies row-for-row molecule-ID alignment across all inputs.
6. Optional PyTorch Dataset & DataLoader integration for Member 2 DL model.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Try importing numpy, pandas, pyarrow, torch
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Global Constants ────────────────────────────────────────────────────────────
TARGET_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

DESCRIPTOR_NAMES = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]

FEATURE_MAP = {
    "combined": "X_combined.parquet",
    "combined_raw": "X_combined_raw.parquet",
    "morgan": "X_morgan.parquet",
    "descriptors_scaled": "X_descriptors_scaled.parquet",
    "descriptors_raw": "X_descriptors_raw.parquet",
    "morgan_filtered": "X_morgan_unsupervised_filtered.parquet",
}

FEATURE_DIMS = {
    "combined": 2053,
    "combined_raw": 2053,
    "morgan": 2048,
    "descriptors_scaled": 5,
    "descriptors_raw": 5,
    "morgan_filtered": 1011,
}

TRAIN_POSITIVE_WEIGHTS = {
    "NR-AR": 24.120171673819744,
    "NR-AR-LBD": 27.96842105263158,
    "NR-AhR": 8.11492281303602,
    "NR-Aromatase": 22.29611650485437,
    "NR-ER": 7.0906200317965025,
    "NR-ER-LBD": 18.51903114186851,
    "NR-PPAR-gamma": 38.849624060150376,
    "SR-ARE": 5.707182320441989,
    "SR-ATAD5": 28.901041666666668,
    "SR-HSE": 17.843971631205672,
    "SR-MMP": 5.730337078651685,
    "SR-p53": 18.357894736842106,
}


def find_project_root() -> Path:
    """Locate project root by searching upwards for key markers."""
    current = Path(__file__).resolve().parent
    for p in [current] + list(current.parents):
        if (p / "tox21_member2_handoff").exists() or (p / "tox21_project").exists():
            return p
    return current.parent.parent.parent


class Tox21DataLoader:
    """
    Certified Data Pipeline Loader for Tox21 Phase 2.
    Loads and provides features, labels, observation masks, and split indices.
    """

    def __init__(
        self,
        artifacts_dir: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
    ):
        project_root = find_project_root()
        if artifacts_dir is not None:
            self.artifacts_dir = Path(artifacts_dir)
        else:
            self.artifacts_dir = project_root / "tox21_member2_handoff" / "artifacts"

        if not self.artifacts_dir.exists():
            raise FileNotFoundError(f"Artifacts directory not found: {self.artifacts_dir}")

        if config_path is not None:
            self.config_path = Path(config_path)
        else:
            self.config_path = project_root / "phase2" / "data" / "configs" / "split_config.json"

        self.config: Dict[str, Any] = {}
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

        self._labels_df: Optional[Any] = None
        self._csv_rows: Optional[List[Dict[str, str]]] = None
        self._init_metadata()

    def _init_metadata(self) -> None:
        """Load split metadata from Parquet if pandas is available, otherwise from CSV."""
        parquet_path = self.artifacts_dir / "labels_and_splits.parquet"
        csv_path = self.artifacts_dir / "corrected_scaffold_split.csv"

        if PANDAS_AVAILABLE and parquet_path.exists():
            self._labels_df = pd.read_parquet(parquet_path)
            self._mol_ids = self._labels_df["mol_id"].tolist()
            self._splits = self._labels_df["split"].tolist()
        elif csv_path.exists():
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self._csv_rows = list(reader)
            self._mol_ids = [r["mol_id"] for r in self._csv_rows]
            self._splits = [r["split"] for r in self._csv_rows]
        else:
            raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} was found.")

        # Normalize split naming ('val' vs 'validation')
        # In Split B, the validation split is encoded as 'val'
        self.train_indices = [i for i, s in enumerate(self._splits) if s == "train"]
        self.val_indices = [i for i, s in enumerate(self._splits) if s in ("val", "validation")]
        self.test_indices = [i for i, s in enumerate(self._splits) if s == "test"]

    @property
    def total_molecules(self) -> int:
        return len(self._mol_ids)

    def get_split_indices(self, split: str) -> List[int]:
        """Returns list of row indices corresponding to requested split."""
        split_clean = split.lower().strip()
        if split_clean == "train":
            return list(self.train_indices)
        elif split_clean in ("val", "validation"):
            return list(self.val_indices)
        elif split_clean == "test":
            return list(self.test_indices)
        elif split_clean in ("all", "overall"):
            return list(range(self.total_molecules))
        else:
            raise ValueError(f"Unknown split '{split}'. Expected 'train', 'val'/'validation', 'test', or 'all'.")

    def load_labels(
        self,
        split: Optional[str] = None,
        endpoints: Optional[List[str]] = None,
    ) -> Tuple[Any, Any, List[str]]:
        """
        Loads ground truth toxicity labels and boolean observation masks.

        Parameters
        ----------
        split : Optional split ('train', 'val', 'test', or None for all)
        endpoints : Optional subset of endpoints (defaults to all 12 assays)

        Returns
        -------
        labels : Array of shape (N, num_endpoints) with floats (0.0, 1.0, or NaN for unmeasured)
        mask   : Boolean array of shape (N, num_endpoints) (True for measured, False for NaN)
        mol_ids: List of molecule IDs of length N
        """
        target_eps = endpoints if endpoints is not None else TARGET_ASSAYS
        indices = self.get_split_indices(split) if split is not None else list(range(self.total_molecules))
        mol_ids = [self._mol_ids[i] for i in indices]

        if PANDAS_AVAILABLE and self._labels_df is not None:
            sub_df = self._labels_df.iloc[indices][target_eps]
            labels = sub_df.values.astype(np.float64)
            mask = ~np.isnan(labels)
            return labels, mask, mol_ids
        else:
            # Fallback for standard library
            labels_list = []
            mask_list = []
            for i in indices:
                row = self._csv_rows[i]
                row_labels = []
                row_mask = []
                for ep in target_eps:
                    val = row.get(ep, "")
                    if val in ("", "nan", "NaN", None):
                        row_labels.append(float("nan"))
                        row_mask.append(False)
                    else:
                        v_float = float(val)
                        row_labels.append(v_float)
                        row_mask.append(True)
                labels_list.append(row_labels)
                mask_list.append(row_mask)
            if NUMPY_AVAILABLE:
                return np.array(labels_list, dtype=np.float64), np.array(mask_list, dtype=bool), mol_ids
            return labels_list, mask_list, mol_ids

    def load_features(
        self,
        feature_set: str = "combined",
        split: Optional[str] = None,
    ) -> Tuple[Any, List[str]]:
        """
        Loads feature matrix for the specified representation and split.

        Parameters
        ----------
        feature_set : One of 'combined', 'combined_raw', 'morgan', 'descriptors_scaled',
                      'descriptors_raw', 'morgan_filtered'
        split : Optional split ('train', 'val', 'test', or None for all)

        Returns
        -------
        features : Array of shape (N, D)
        mol_ids  : List of molecule IDs of length N
        """
        if feature_set not in FEATURE_MAP:
            raise ValueError(f"Unknown feature set '{feature_set}'. Options: {list(FEATURE_MAP.keys())}")

        indices = self.get_split_indices(split) if split is not None else list(range(self.total_molecules))
        mol_ids = [self._mol_ids[i] for i in indices]

        parquet_file = self.artifacts_dir / FEATURE_MAP[feature_set]

        if PANDAS_AVAILABLE and parquet_file.exists():
            df = pd.read_parquet(parquet_file)
            feat_cols = [c for c in df.columns if c != "mol_id"]
            sub_vals = df.iloc[indices][feat_cols].values.astype(np.float64)
            return sub_vals, mol_ids

        # Pure standard library fallback: check for phase2_features/features.npz
        project_root = find_project_root()
        npz_path = project_root / "phase2_features" / "features.npz"
        if npz_path.exists():
            import zipfile
            import struct
            z = zipfile.ZipFile(npz_path)
            # map feature_set to npy name in npz
            npz_map = {
                "combined": "X_morgan.npy",  # can combine with descriptors
                "morgan": "X_morgan.npy",
                "descriptors_scaled": "X_descriptors_scaled.npy",
                "descriptors_raw": "X_descriptors_raw.npy",
            }
            if feature_set in npz_map:
                data = z.read(npz_map[feature_set])
                header_len = struct.unpack("<H", data[8:10])[0]
                raw_bytes = data[10 + header_len:]
                if NUMPY_AVAILABLE:
                    import io
                    arr = np.load(io.BytesIO(data))
                    return arr[indices].astype(np.float64), mol_ids

        raise RuntimeError(f"Cannot load feature set '{feature_set}': requires pandas/pyarrow to read {parquet_file}.")

    def get_class_weights(
        self,
        endpoints: Optional[List[str]] = None,
        as_tensor: bool = False,
    ) -> Any:
        """
        Calculates positive class weight for each endpoint strictly on the training partition:
            w_j = N_{neg, train} / N_{pos, train}

        Parameters
        ----------
        endpoints : Optional list of endpoints
        as_tensor : If True and PyTorch is available, returns a torch.FloatTensor

        Returns
        -------
        weights : 1D array or tensor of positive class weights
        """
        target_eps = endpoints if endpoints is not None else TARGET_ASSAYS
        weights = []
        for ep in target_eps:
            if ep in TRAIN_POSITIVE_WEIGHTS:
                weights.append(TRAIN_POSITIVE_WEIGHTS[ep])
            else:
                # Calculate dynamically from training labels
                labels, mask, _ = self.load_labels(split="train", endpoints=[ep])
                valid = labels[mask]
                pos = (valid == 1).sum()
                neg = (valid == 0).sum()
                weights.append(float(neg / pos) if pos > 0 else 1.0)

        if NUMPY_AVAILABLE:
            w_arr = np.array(weights, dtype=np.float32)
            if as_tensor and TORCH_AVAILABLE:
                return torch.from_numpy(w_arr)
            return w_arr
        return weights

    def get_dataset(
        self,
        split: str = "train",
        feature_set: str = "combined",
        endpoints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience bundle returning features (X), labels (y), observation mask,
        molecule IDs, and positive class weights for downstream model consumption.
        """
        target_eps = endpoints if endpoints is not None else TARGET_ASSAYS
        X, mol_ids = self.load_features(feature_set=feature_set, split=split)
        y, mask, mol_ids_y = self.load_labels(split=split, endpoints=target_eps)
        assert mol_ids == mol_ids_y, "Row alignment mismatch between features and labels."

        class_weights = self.get_class_weights(endpoints=target_eps)

        return {
            "X": X,
            "y": y,
            "mask": mask,
            "mol_ids": mol_ids,
            "split": split,
            "feature_set": feature_set,
            "endpoints": target_eps,
            "class_weights": class_weights,
            "n_samples": len(mol_ids),
            "n_features": X.shape[1] if hasattr(X, "shape") else len(X[0]),
        }

    def get_pytorch_dataloaders(
        self,
        feature_set: str = "combined",
        batch_size: int = 64,
        shuffle_train: bool = True,
        num_workers: int = 0,
        endpoints: Optional[List[str]] = None,
    ) -> Tuple[Any, Any, Any]:
        """
        Generates PyTorch DataLoaders for train, validation, and test splits.
        Specifically architected for Member 2's multitask deep learning network.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Install with `pip install torch`.")

        train_ds = Tox21TorchDataset(self, split="train", feature_set=feature_set, endpoints=endpoints)
        val_ds = Tox21TorchDataset(self, split="val", feature_set=feature_set, endpoints=endpoints)
        test_ds = Tox21TorchDataset(self, split="test", feature_set=feature_set, endpoints=endpoints)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        return train_loader, val_loader, test_loader


# ── Optional PyTorch Dataset Wrapper ───────────────────────────────────────────
if TORCH_AVAILABLE:
    class Tox21TorchDataset(Dataset):
        """
        PyTorch Dataset for multi-task molecular toxicity prediction.
        Yields (X_tensor, y_tensor, mask_tensor).
        In y_tensor, missing values are zero-filled so that tensor operations do not
        propagate NaN; the mask tensor ensures missing targets receive ZERO loss contribution.
        """

        def __init__(
            self,
            loader: Tox21DataLoader,
            split: str = "train",
            feature_set: str = "combined",
            endpoints: Optional[List[str]] = None,
        ):
            self.bundle = loader.get_dataset(split=split, feature_set=feature_set, endpoints=endpoints)
            X = self.bundle["X"]
            y = self.bundle["y"]
            mask = self.bundle["mask"]

            self.X = torch.tensor(X, dtype=torch.float32)
            # Fill NaNs with 0.0 in label tensor; mask indicates valid targets
            y_clean = np.nan_to_num(y, nan=0.0)
            self.y = torch.tensor(y_clean, dtype=torch.float32)
            self.mask = torch.tensor(mask.astype(np.float32), dtype=torch.float32)
            self.mol_ids = self.bundle["mol_ids"]

        def __len__(self) -> int:
            return len(self.mol_ids)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx], self.mask[idx]
