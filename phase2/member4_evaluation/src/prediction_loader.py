"""
prediction_loader.py
====================
Unified prediction ingestion and evaluation harness for:
1. Standalone Random Forest (Phase 1)
2. Standalone MultiTask Deep Learning (Phase 2 Member 2)
3. ML + DL Hybrid Model (Phase 2 Member 3)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import joblib

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader, TARGET_ASSAYS
from phase2.member2_deep_learning_model.src.model import MultiTaskToxicNet


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


class PredictionLoader:
    """
    Standardized loader for all models evaluated in Phase 2 Member 4.
    Ensures identical test compounds, strict observation masking, and consistent shapes.
    """

    def __init__(self, data_loader: Optional[Tox21DataLoader] = None):
        self.root = get_project_root()
        self.loader = data_loader or Tox21DataLoader()
        self.target_assays = list(TARGET_ASSAYS)

        # Ground truth labels and masks
        self.y_val, self.mask_val, self.mids_val = self.loader.load_labels(split="val")
        self.y_test, self.mask_test, self.mids_test = self.loader.load_labels(split="test")

    def load_standalone_rf_predictions(self) -> Dict[str, Any]:
        """
        Load Standalone Random Forest predictions.
        Reads test predictions from results/test_predictions.csv and reconstructs (782, 12) probability matrix.
        """
        test_pred_path = self.root / "results" / "test_predictions.csv"
        df_pred = pd.read_csv(test_pred_path)

        # Build (782, 12) probability array aligned with self.mids_test and self.target_assays
        test_probs = np.zeros((len(self.mids_test), len(self.target_assays)), dtype=float)

        # Group by endpoint and mol_id
        for ep_idx, ep in enumerate(self.target_assays):
            ep_df = df_pred[df_pred["endpoint"] == ep].set_index("mol_id")
            for m_idx, mid in enumerate(self.mids_test):
                if mid in ep_df.index:
                    test_probs[m_idx, ep_idx] = float(ep_df.loc[mid, "prob_rf"])

        return {
            "model_name": "Standalone Random Forest",
            "test_probs": test_probs,
            "y_test": self.y_test,
            "mask_test": self.mask_test,
            "mids_test": self.mids_test,
        }

    def load_standalone_dl_predictions(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """
        Run inference using Member 2 MultiTaskToxicNet certified checkpoint on val and test splits.
        """
        if device is None:
            device = torch.device("cpu")

        ckpt_path = Path(checkpoint_path or (
            self.root / "phase2" / "member2_deep_learning_model" / "checkpoints" / "best_multitask_model.pt"
        ))
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Deep learning checkpoint not found at {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=device)
        model = MultiTaskToxicNet(
            input_dim=ckpt["config"].get("input_dim", 2053),
            hidden_dims=ckpt["config"].get("hidden_dims", [512, 256]),
            latent_dim=ckpt["config"].get("latent_dim", 128),
            num_endpoints=ckpt["config"].get("num_endpoints", 12),
            dropout=ckpt["config"].get("dropout", 0.35),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        # Load features
        X_val, mids_val_x = self.loader.load_features("combined", split="val")
        X_test, mids_test_x = self.loader.load_features("combined", split="test")
        assert mids_val_x == self.mids_val
        assert mids_test_x == self.mids_test

        with torch.no_grad():
            val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
            val_probs = torch.sigmoid(model(val_t)).cpu().numpy()

            test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            test_probs = torch.sigmoid(model(test_t)).cpu().numpy()

        return {
            "model_name": "MultiTask Deep Learning",
            "val_probs": val_probs,
            "test_probs": test_probs,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "mask_val": self.mask_val,
            "mask_test": self.mask_test,
            "mids_val": self.mids_val,
            "mids_test": self.mids_test,
        }

    def load_hybrid_predictions(self) -> Dict[str, Any]:
        """
        Load Member 3 Hybrid Model predictions.
        Reads saved Random Forest models from phase2/member3_hybrid_model/models/ and predicts on val and test.
        """
        hybrid_dir = self.root / "phase2" / "member3_hybrid_model"
        models_dir = hybrid_dir / "models"

        # Load Member 2 embeddings + scaled descriptors
        train_emb = pd.read_parquet(self.root / "phase2" / "member2_deep_learning_model" / "embeddings" / "train_embeddings.parquet")
        val_emb = pd.read_parquet(self.root / "phase2" / "member2_deep_learning_model" / "embeddings" / "val_embeddings.parquet")
        test_emb = pd.read_parquet(self.root / "phase2" / "member2_deep_learning_model" / "embeddings" / "test_embeddings.parquet")

        X_desc_val, _ = self.loader.load_features("descriptors_scaled", split="val")
        X_desc_test, _ = self.loader.load_features("descriptors_scaled", split="test")

        z_val = val_emb.drop(columns=["molecule_id"]).to_numpy()
        z_test = test_emb.drop(columns=["molecule_id"]).to_numpy()

        X_val_hybrid = np.hstack([z_val, X_desc_val])
        X_test_hybrid = np.hstack([z_test, X_desc_test])

        val_probs = np.zeros((len(self.mids_val), len(self.target_assays)), dtype=float)
        test_probs = np.zeros((len(self.mids_test), len(self.target_assays)), dtype=float)

        for ep_idx, ep in enumerate(self.target_assays):
            model_file = models_dir / f"rf_{ep.replace('-', '_')}.joblib"
            if not model_file.exists():
                model_file = models_dir / f"rf_{ep}.joblib"
            rf = joblib.load(model_file)

            val_p = rf.predict_proba(X_val_hybrid)
            val_probs[:, ep_idx] = val_p[:, 1] if val_p.shape[1] > 1 else val_p[:, 0]

            test_p = rf.predict_proba(X_test_hybrid)
            test_probs[:, ep_idx] = test_p[:, 1] if test_p.shape[1] > 1 else test_p[:, 0]

        return {
            "model_name": "ML+DL Hybrid Model",
            "val_probs": val_probs,
            "test_probs": test_probs,
            "y_val": self.y_val,
            "y_test": self.y_test,
            "mask_val": self.mask_val,
            "mask_test": self.mask_test,
            "mids_val": self.mids_val,
            "mids_test": self.mids_test,
        }

