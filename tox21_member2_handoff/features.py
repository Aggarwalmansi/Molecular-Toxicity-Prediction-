"""
features.py
===========
Reproducible Feature Engineering and Transformation Pipeline for Tox21.

Member 2 Core Handoff Module.

Capabilities:
1. Molecular Feature Extraction:
   - Morgan fingerprints (radius=2, 2,048 bits, binary ECFP4-like) via RDKit
   - 5 Physicochemical descriptors: MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors
2. Leakage-Safe Preprocessing:
   - LeakageSafeStandardScaler: Standardizes descriptors strictly using training mean and std
   - LeakageSafePCA: Eigendecomposition PCA fitted strictly on training data
3. Leakage-Safe Feature Selection:
   - UnsupervisedFeatureFilter: Removes constant and low-variance features on training data
   - SupervisedFeatureSelector: Univariate Chi-square (Yates-corrected) & ANOVA F-test
     using strictly observed labels in training data (NaNs excluded, zero negative imputation)
4. Data Loader & Verification:
   - Tox21Dataset: Unified loader ensuring row alignment, stable compound IDs, and mask extraction
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator

# ── Global Constants ────────────────────────────────────────────────────────────
RDKIT_VERSION = rdBase.rdkitVersion
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048
DESCRIPTOR_NAMES = ["MolWt", "TPSA", "MolLogP", "NumHDonors", "NumHAcceptors"]
TARGET_ASSAYS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE",
    "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]


# ── Feature Extraction ──────────────────────────────────────────────────────────
def get_morgan_generator(radius: int = MORGAN_RADIUS, n_bits: int = MORGAN_NBITS):
    """Initialize modern RDKit Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def compute_morgan_fingerprints(
    smiles_list: List[str],
    radius: int = MORGAN_RADIUS,
    n_bits: int = MORGAN_NBITS,
) -> Tuple[np.ndarray, List[dict]]:
    """
    Generate fixed-length binary Morgan fingerprints (ECFP4-like).
    Returns (fps_matrix, failures).
    """
    gen = get_morgan_generator(radius=radius, n_bits=n_bits)
    fps = []
    failures = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append({"index": idx, "smiles": smi, "error": "SMILES parse failure"})
            fps.append(np.zeros(n_bits, dtype=np.uint8))
            continue
        try:
            arr = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
            fps.append(arr)
        except Exception as exc:
            failures.append({"index": idx, "smiles": smi, "error": str(exc)})
            fps.append(np.zeros(n_bits, dtype=np.uint8))

    fps_matrix = np.vstack(fps)
    return fps_matrix, failures


def compute_physicochemical_descriptors(
    smiles_list: List[str],
) -> Tuple[np.ndarray, List[dict]]:
    """
    Calculate 5 fundamental physicochemical descriptors:
    MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors.
    Returns (desc_matrix, failures).
    """
    desc_rows = []
    failures = []

    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            failures.append({"index": idx, "smiles": smi, "error": "SMILES parse failure"})
            desc_rows.append([np.nan] * len(DESCRIPTOR_NAMES))
            continue
        try:
            mw = float(Descriptors.MolWt(mol))
            tpsa = float(Descriptors.TPSA(mol))
            logp = float(Descriptors.MolLogP(mol))
            hbd = float(Lipinski.NumHDonors(mol))
            hba = float(Lipinski.NumHAcceptors(mol))
            desc_rows.append([mw, tpsa, logp, hbd, hba])
        except Exception as exc:
            failures.append({"index": idx, "smiles": smi, "error": str(exc)})
            desc_rows.append([np.nan] * len(DESCRIPTOR_NAMES))

    desc_matrix = np.array(desc_rows, dtype=np.float64)
    return desc_matrix, failures


# ── Leakage-Safe Standard Scaler ────────────────────────────────────────────────
class LeakageSafeStandardScaler:
    """
    StandardScaler that fits strictly on training data to prevent leakage.
    Guarantees that constant features (std=0) do not trigger division by zero.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "LeakageSafeStandardScaler":
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_train.shape}")
        self.mean_ = np.mean(X_train, axis=0)
        self.std_ = np.std(X_train, axis=0)
        self.std_[self.std_ == 0.0] = 1.0  # Avoid division by zero
        self.n_samples_ = int(len(X_train))
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Scaler is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_names": self.feature_names or [],
            "mean": self.mean_.tolist() if self.mean_ is not None else [],
            "std": self.std_.tolist() if self.std_ is not None else [],
            "n_training_samples": getattr(self, "n_samples_", None),
            "fit_split": "train",
        }

    def save_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeakageSafeStandardScaler":
        scaler = cls(feature_names=data.get("feature_names"))
        scaler.mean_ = np.array(data["mean"], dtype=np.float64)
        scaler.std_ = np.array(data["std"], dtype=np.float64)
        scaler.is_fitted = True
        return scaler

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> "LeakageSafeStandardScaler":
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


# ── Leakage-Safe PCA ────────────────────────────────────────────────────────────
class LeakageSafePCA:
    """
    Principal Component Analysis via covariance eigendecomposition.
    Fitted strictly on training partition to guarantee zero data leakage.
    Self-contained pure-NumPy implementation.
    """

    def __init__(self, n_components: Optional[int] = None, name: str = "PCA"):
        self.n_components = n_components
        self.name = name
        self.mean_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "LeakageSafePCA":
        X_train = np.asarray(X_train, dtype=np.float64)
        N, D = X_train.shape
        self.mean_ = np.mean(X_train, axis=0)
        self.feature_names_in_ = feature_names

        X_c = X_train - self.mean_
        # Covariance matrix formulation
        if N > D:
            cov = np.dot(X_c.T, X_c) / (N - 1)
            evals, evecs = np.linalg.eigh(cov)
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]
            total_var = np.sum(evals)
            k = self.n_components if self.n_components is not None else D
            self.components_ = evecs[:, :k].T
            self.explained_variance_ = evals[:k]
        else:
            gram = np.dot(X_c, X_c.T) / (N - 1)
            evals, evecs = np.linalg.eigh(gram)
            idx = np.argsort(evals)[::-1]
            evals = np.maximum(evals[idx], 0.0)
            evecs = evecs[:, idx]
            total_var = np.sum(evals)
            k = self.n_components if self.n_components is not None else min(N, D)
            components = []
            for i in range(k):
                if evals[i] > 1e-12:
                    v = np.dot(X_c.T, evecs[:, i])
                    v = v / np.linalg.norm(v)
                else:
                    v = np.zeros(D)
                components.append(v)
            self.components_ = np.vstack(components)
            self.explained_variance_ = evals[:k]

        self.explained_variance_ratio_ = self.explained_variance_ / total_var
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("PCA model is not fitted yet.")
        X = np.asarray(X, dtype=np.float64)
        X_c = X - self.mean_
        return np.dot(X_c, self.components_.T)

    def fit_transform(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> np.ndarray:
        return self.fit(X_train, feature_names).transform(X_train)


# ── Feature Selection ───────────────────────────────────────────────────────────
class UnsupervisedFeatureFilter:
    """
    Evaluates constant, duplicate, and low-variance features strictly on training data.
    """

    def __init__(self, variance_threshold: float = 0.005):
        self.variance_threshold = variance_threshold
        self.constant_indices_: List[int] = []
        self.retained_indices_: List[int] = []
        self.variances_: Optional[np.ndarray] = None
        self.feature_names_in_: Optional[List[str]] = None

    def fit(self, X_train: np.ndarray, feature_names: Optional[List[str]] = None) -> "UnsupervisedFeatureFilter":
        X_train = np.asarray(X_train, dtype=np.float64)
        self.variances_ = np.var(X_train, axis=0)
        self.feature_names_in_ = feature_names

        self.constant_indices_ = np.where(self.variances_ == 0.0)[0].tolist()
        self.retained_indices_ = np.where(self.variances_ >= self.variance_threshold)[0].tolist()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.retained_indices_:
            raise RuntimeError("Filter is not fitted yet.")
        X = np.asarray(X)
        return X[:, self.retained_indices_]

    @property
    def retained_feature_names(self) -> List[str]:
        if self.feature_names_in_ is None:
            return [f"feature_{i}" for i in self.retained_indices_]
        return [self.feature_names_in_[i] for i in self.retained_indices_]


class SupervisedFeatureSelector:
    """
    Univariate statistical feature association selector.
    Fitted strictly on training split observed labels (NaNs excluded).
    Uses Yates-corrected Chi-square for binary Morgan bits and ANOVA F-test for descriptors.
    """

    def __init__(self, target_assays: List[str] = TARGET_ASSAYS):
        self.target_assays = target_assays
        self.rankings_: Dict[str, pd.DataFrame] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train_df: pd.DataFrame,
        feature_names: List[str],
        n_morgan_bits: int = MORGAN_NBITS,
    ) -> "SupervisedFeatureSelector":
        X_train = np.asarray(X_train, dtype=np.float64)

        for assay in self.target_assays:
            y_arr = y_train_df[assay].values.astype(np.float64)
            mask = ~np.isnan(y_arr)
            X_sub = X_train[mask]
            y_sub = y_arr[mask]

            N = len(y_sub)
            n_pos = int(np.sum(y_sub == 1))
            n_neg = int(np.sum(y_sub == 0))

            if n_pos == 0 or n_neg == 0:
                continue

            scores = np.zeros(X_train.shape[1], dtype=np.float64)
            test_types = []

            # 1. Chi-square with Yates correction for binary Morgan bits
            for j in range(n_morgan_bits):
                col = X_sub[:, j]
                O11 = np.sum((y_sub == 1) & (col == 1))
                O10 = np.sum((y_sub == 0) & (col == 1))
                O01 = np.sum((y_sub == 1) & (col == 0))
                O00 = np.sum((y_sub == 0) & (col == 0))

                c1 = O11 + O10
                c0 = O01 + O00
                if c1 == 0 or c0 == 0:
                    scores[j] = 0.0
                else:
                    E11 = (n_pos * c1) / N
                    E10 = (n_neg * c1) / N
                    E01 = (n_pos * c0) / N
                    E00 = (n_neg * c0) / N

                    chi2 = 0.0
                    for O, E in [(O11, E11), (O10, E10), (O01, E01), (O00, E00)]:
                        if E > 0:
                            num = max(0.0, abs(O - E) - 0.5) ** 2
                            chi2 += num / E
                    scores[j] = chi2
                test_types.append("Chi2_Yates")

            # 2. ANOVA F-statistic for continuous descriptors
            for j in range(n_morgan_bits, X_train.shape[1]):
                col = X_sub[:, j]
                pos_vals = col[y_sub == 1]
                neg_vals = col[y_sub == 0]

                grand_mean = np.mean(col)
                ssb = n_pos * (np.mean(pos_vals) - grand_mean) ** 2 + n_neg * (np.mean(neg_vals) - grand_mean) ** 2
                ssw = np.sum((pos_vals - np.mean(pos_vals)) ** 2) + np.sum((neg_vals - np.mean(neg_vals)) ** 2)

                df_b = 1
                df_w = N - 2
                msb = ssb / df_b
                msw = ssw / df_w if df_w > 0 else 1.0

                f_stat = msb / msw if msw > 1e-12 else 0.0
                scores[j] = f_stat
                test_types.append("ANOVA_F")

            df_rank = pd.DataFrame({
                "feature_name": feature_names,
                "feature_index": np.arange(len(feature_names)),
                "association_score": scores,
                "test_type": test_types,
            }).sort_values("association_score", ascending=False).reset_index(drop=True)
            df_rank["rank"] = np.arange(1, len(df_rank) + 1)
            self.rankings_[assay] = df_rank

        return self

    def get_top_k(self, assay: str, k: int = 50) -> pd.DataFrame:
        if assay not in self.rankings_:
            raise KeyError(f"Assay '{assay}' not fitted.")
        return self.rankings_[assay].head(k)


# ── Unified Tox21 Dataset Container ─────────────────────────────────────────────
class Tox21Dataset:
    """
    Standard container for loading, slicing, and preparing Tox21 features for Member 3.
    """

    def __init__(self, artifacts_dir: Union[str, Path]):
        self.artifacts_dir = Path(artifacts_dir)
        self.metadata = pd.read_parquet(self.artifacts_dir / "labels_and_splits.parquet")
        self.train_idx = np.where(self.metadata["split"] == "train")[0]
        self.val_idx = np.where(self.metadata["split"] == "val")[0]
        self.test_idx = np.where(self.metadata["split"] == "test")[0]

    def load_features(self, feature_set: str = "combined", return_df: bool = False) -> Union[np.ndarray, pd.DataFrame]:
        """
        Load feature matrix by name:
        - 'combined': 2,053 features (2,048 Morgan bits + 5 training-standardized descriptors) [Canonical]
        - 'combined_raw': 2,053 features (2,048 Morgan bits + 5 raw physicochemical descriptors)
        - 'morgan': 2,048 binary bits
        - 'descriptors_scaled': 5 standardized descriptors (zero mean, unit variance on train)
        - 'descriptors_raw': 5 raw descriptors (MolWt, TPSA, MolLogP, NumHDonors, NumHAcceptors)
        - 'morgan_filtered': 1,011 variance-filtered bits (training variance >= 0.005)
        - 'pca_combined': 50 combined PCs
        - 'pca_morgan': 50 Morgan PCs
        - 'pca_descriptors': 5 descriptor PCs
        """
        file_map = {
            "combined": "X_combined.parquet",
            "combined_raw": "X_combined_raw.parquet",
            "morgan": "X_morgan.parquet",
            "descriptors_scaled": "X_descriptors_scaled.parquet",
            "descriptors_raw": "X_descriptors_raw.parquet",
            "morgan_filtered": "X_morgan_unsupervised_filtered.parquet",
            "pca_combined": "X_pca_combined.parquet",
            "pca_morgan": "X_pca_morgan.parquet",
            "pca_descriptors": "X_pca_descriptors.parquet",
        }
        if feature_set not in file_map:
            raise ValueError(f"Unknown feature set '{feature_set}'. Options: {list(file_map.keys())}")
        df = pd.read_parquet(self.artifacts_dir / file_map[feature_set])
        if "mol_id" in df.columns:
            feat_cols = [c for c in df.columns if c != "mol_id"]
            df = df[feat_cols]
        if return_df:
            return df
        return df.values.astype(np.float64)

    def get_labels(self, assay: str, split: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (labels, observed_mask) for a specific assay and optional split.
        Missing values remain NaN; observed_mask indicates valid experimental outcomes.
        """
        if split == "train":
            idx = self.train_idx
        elif split == "val":
            idx = self.val_idx
        elif split == "test":
            idx = self.test_idx
        else:
            idx = np.arange(len(self.metadata))

        y = self.metadata.iloc[idx][assay].values.astype(np.float64)
        mask = ~np.isnan(y)
        return y, mask
