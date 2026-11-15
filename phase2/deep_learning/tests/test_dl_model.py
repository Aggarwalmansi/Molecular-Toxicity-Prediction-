"""
test_dl_model.py
================
Automated test suite for Phase 2 Member 2: Deep Learning Model.
Verifies:
1. Model architecture dimensions and forward pass output shapes.
2. Latent bottleneck representation extraction (128-dim embeddings).
3. Masked loss invariance: Unobserved labels (mask=0) contribute zero gradient and zero loss.
4. NaN safety: Missing label inputs never corrupt gradients.
5. Checkpoint serialization and reproducible deserialization.
6. Real batch integration with Member 1 Tox21DataLoader.
"""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from phase2.data.src.data_loader import Tox21DataLoader
from phase2.deep_learning.src.model import MultiTaskToxicNet
from phase2.deep_learning.src.loss import MaskedWeightedBCEWithLogitsLoss


class TestMultiTaskToxicNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.batch_size = 16
        cls.input_dim = 2053
        cls.latent_dim = 128
        cls.num_endpoints = 12
        cls.model = MultiTaskToxicNet(
            input_dim=cls.input_dim,
            hidden_dims=[256, 128],
            latent_dim=cls.latent_dim,
            num_endpoints=cls.num_endpoints,
            dropout=0.2,
        )
        cls.model.eval()

    def test_01_forward_output_shapes(self):
        """Test logits and embeddings shapes match specifications."""
        x = torch.randn(self.batch_size, self.input_dim)
        logits = self.model(x)
        self.assertEqual(logits.shape, (self.batch_size, self.num_endpoints))

        z = self.model.get_embeddings(x)
        self.assertEqual(z.shape, (self.batch_size, self.latent_dim))

        logits2, z2 = self.model.forward_with_embeddings(x)
        self.assertEqual(logits2.shape, (self.batch_size, self.num_endpoints))
        self.assertEqual(z2.shape, (self.batch_size, self.latent_dim))

    def test_02_predict_proba_range(self):
        """Test sigmoid output probabilities are strictly in [0, 1]."""
        x = torch.randn(self.batch_size, self.input_dim)
        probs = self.model.predict_proba(x)
        self.assertTrue((probs >= 0.0).all())
        self.assertTrue((probs <= 1.0).all())

    def test_03_masked_loss_unobserved_invariance(self):
        """Unobserved targets (mask=0) MUST NOT alter the computed loss or gradients."""
        loss_fn = MaskedWeightedBCEWithLogitsLoss(
            pos_weight=torch.ones(self.num_endpoints) * 10.0
        )
        logits = torch.randn(4, self.num_endpoints, requires_grad=True)
        targets1 = torch.tensor([[1.0, 0.0] * 6, [0.0, 1.0] * 6, [0.0, 0.0] * 6, [1.0, 1.0] * 6])
        mask = torch.tensor([[1, 0] * 6, [0, 1] * 6, [1, 1] * 6, [0, 0] * 6], dtype=torch.float32)

        # targets2 differs ONLY where mask is 0
        targets2 = targets1.clone()
        targets2[mask == 0] = 1.0 - targets2[mask == 0]  # flip unobserved bits

        loss1 = loss_fn(logits, targets1, mask)
        loss2 = loss_fn(logits, targets2, mask)

        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6,
                               msg="Loss changed when unobserved labels were modified!")

        # Verify gradients are identical
        loss1.backward(retain_graph=True)
        grad1 = logits.grad.clone()
        logits.grad.zero_()

        loss2.backward()
        grad2 = logits.grad.clone()
        self.assertTrue(torch.allclose(grad1, grad2, atol=1e-6),
                        "Gradients differed when unobserved targets were modified!")

    def test_04_zero_gradient_for_masked_positions(self):
        """Gradient on logits where entire column is masked must be 0."""
        loss_fn = MaskedWeightedBCEWithLogitsLoss()
        logits = torch.randn(4, self.num_endpoints, requires_grad=True)
        targets = torch.zeros(4, self.num_endpoints)
        mask = torch.ones(4, self.num_endpoints)
        mask[:, 0] = 0.0  # endpoint 0 completely unobserved

        loss = loss_fn(logits, targets, mask)
        loss.backward()

        self.assertTrue((logits.grad[:, 0] == 0.0).all(),
                        "Unobserved endpoint received non-zero gradient!")

    def test_05_checkpoint_save_and_load(self):
        """Verify model weights can be saved and deserialized deterministically."""
        x = torch.randn(5, self.input_dim)
        orig_preds = self.model.predict_proba(x)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pt"
            torch.save({"model_state_dict": self.model.state_dict()}, ckpt_path)

            loaded_model = MultiTaskToxicNet(
                input_dim=self.input_dim,
                hidden_dims=[256, 128],
                latent_dim=self.latent_dim,
                num_endpoints=self.num_endpoints,
            )
            ckpt = torch.load(ckpt_path, weights_only=True)
            loaded_model.load_state_dict(ckpt["model_state_dict"])
            loaded_model.eval()

            loaded_preds = loaded_model.predict_proba(x)
            self.assertTrue(torch.allclose(orig_preds, loaded_preds, atol=1e-6))

    def test_06_real_batch_integration(self):
        """Test with real Member 1 data loader batch."""
        loader = Tox21DataLoader()
        train_loader, _, _ = loader.get_pytorch_dataloaders(
            feature_set="combined", batch_size=32, shuffle_train=False
        )
        pos_weights = loader.get_class_weights(as_tensor=True)
        loss_fn = MaskedWeightedBCEWithLogitsLoss(pos_weight=pos_weights)

        X_b, y_b, m_b = next(iter(train_loader))
        logits = self.model(X_b)
        loss = loss_fn(logits, y_b, m_b)

        self.assertFalse(torch.isnan(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_07_embeddings_artifacts_integrity(self):
        """Verify generated embeddings exist and match split counts (6259 train / 782 val / 782 test)."""
        import pandas as pd
        base_dir = Path(__file__).resolve().parent.parent
        emb_dir = base_dir / "embeddings"

        train_p = emb_dir / "train_embeddings.parquet"
        val_p = emb_dir / "val_embeddings.parquet"
        test_p = emb_dir / "test_embeddings.parquet"

        self.assertTrue(train_p.exists(), "train_embeddings.parquet missing")
        self.assertTrue(val_p.exists(), "val_embeddings.parquet missing")
        self.assertTrue(test_p.exists(), "test_embeddings.parquet missing")

        df_tr = pd.read_parquet(train_p)
        df_va = pd.read_parquet(val_p)
        df_te = pd.read_parquet(test_p)

        self.assertEqual(len(df_tr), 6259)
        self.assertEqual(len(df_va), 782)
        self.assertEqual(len(df_te), 782)
        self.assertEqual(df_tr.shape[1], 129)  # 128 embeddings + 1 molecule_id

    def test_08_checkpoint_artifact_loads(self):
        """Verify saved production checkpoint best_multitask_model.pt loads correctly."""
        base_dir = Path(__file__).resolve().parent.parent
        ckpt_path = base_dir / "checkpoints" / "best_multitask_model.pt"
        self.assertTrue(ckpt_path.exists(), "best_multitask_model.pt missing")

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", ckpt)
        self.assertIn("val_macro_roc_auc", ckpt)
        self.assertGreater(ckpt["val_macro_roc_auc"], 0.70)

    def test_09_test_evaluation_report_completeness(self):
        """Verify test_evaluation.csv contains all 12 endpoints and MACRO AVERAGE row."""
        import pandas as pd
        base_dir = Path(__file__).resolve().parent.parent
        csv_path = base_dir / "reports" / "test_evaluation.csv"
        self.assertTrue(csv_path.exists(), "test_evaluation.csv missing")

        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 13)  # 12 endpoints + 1 MACRO AVERAGE row
        self.assertIn("MACRO AVERAGE", df["Endpoint"].values)
        self.assertGreater(df.loc[df["Endpoint"] == "MACRO AVERAGE", "ROC-AUC"].values[0], 0.65)

    def test_10_plots_exist(self):
        """Verify generated diagnostic plots exist."""
        base_dir = Path(__file__).resolve().parent.parent
        plot_dir = base_dir / "reports" / "plots"
        self.assertTrue((plot_dir / "loss_curve.png").exists())
        self.assertTrue((plot_dir / "val_metrics_curve.png").exists())
        self.assertTrue((plot_dir / "per_endpoint_roc_pr.png").exists())


if __name__ == "__main__":
    unittest.main()
