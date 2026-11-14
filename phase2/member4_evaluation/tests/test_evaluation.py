"""
test_evaluation.py
==================
Automated Unit and Integration Test Suite for Phase 2 Member 4: Evaluation & Ablation Studies.
Verifies data invariants, metric computations, observation masking, threshold optimization,
single-task architecture, and configuration integrity.
"""

import json
import unittest
from pathlib import Path

import numpy as np
import torch

from phase2.member4_evaluation.src.data_utils import Tox21DataLoaderWrapper, set_seed
from phase2.member4_evaluation.src.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    bootstrap_metric_ci,
    evaluate_multitask_predictions,
)
from phase2.member4_evaluation.src.single_task_model import SingleTaskToxicNet
from phase2.member2_deep_learning_model.src.loss import MaskedWeightedBCEWithLogitsLoss


class TestMember4Evaluation(unittest.TestCase):
    """Test suite for Member 4 deliverables."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parents[1]
        cls.wrapper = Tox21DataLoaderWrapper()

    def test_01_split_and_feature_invariants(self):
        """Test 1: Check Split B frozen split counts and feature dimensions."""
        loader = self.wrapper.loader
        self.assertEqual(len(loader.get_split_indices("train")), 6259)
        self.assertEqual(len(loader.get_split_indices("val")), 782)
        self.assertEqual(len(loader.get_split_indices("test")), 782)

        X_train, mids_tr = loader.load_features("combined", split="train")
        self.assertEqual(X_train.shape, (6259, 2053))
        self.assertEqual(len(mids_tr), 6259)

    def test_02_masked_loss_unobserved_invariance(self):
        """Test 2: Modifying NaN/unobserved labels must not change masked loss."""
        criterion = MaskedWeightedBCEWithLogitsLoss()
        logits = torch.randn(10, 12)
        targets1 = torch.zeros(10, 12)
        mask = torch.zeros(10, 12)
        mask[:5, :] = 1.0  # Only rows 0-4 observed

        # targets2 differs ONLY where mask is 0
        targets2 = targets1.clone()
        targets2[mask == 0] = 1.0

        loss1 = criterion(logits, targets1, mask)
        loss2 = criterion(logits, targets2, mask)
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=5)

    def test_03_classification_metrics_correctness(self):
        """Test 3: Compute classification metrics on known ground truth."""
        y_true = np.array([1, 1, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1, 0.4, 0.3])
        m = compute_classification_metrics(y_true, y_prob, threshold=0.5)

        self.assertEqual(m["TP"], 2)
        self.assertEqual(m["FP"], 0)
        self.assertEqual(m["TN"], 4)
        self.assertEqual(m["FN"], 0)
        self.assertEqual(m["Precision"], 1.0)
        self.assertEqual(m["Recall"], 1.0)
        self.assertEqual(m["F1-Score"], 1.0)
        self.assertEqual(m["ROC-AUC"], 1.0)

    def test_04_threshold_optimization_f1(self):
        """Test 4: Verify threshold optimization finds threshold that maximizes F1."""
        y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        y_prob = np.array([0.45, 0.40, 0.1, 0.05, 0.2, 0.15, 0.0, 0.05])
        # Default 0.5 gives 0 TP and F1=0
        m_def = compute_classification_metrics(y_true, y_prob, threshold=0.5)
        self.assertEqual(m_def["F1-Score"], 0.0)

        best_th, best_f1 = find_optimal_threshold(y_true, y_prob, metric="f1")
        self.assertGreater(best_f1, 0.5)
        self.assertLess(best_th, 0.5)

    def test_05_bootstrap_ci_bounds(self):
        """Test 5: Verify non-parametric bootstrap confidence interval ordering."""
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1, 0.6, 0.4, 0.3, 0.2])
        mean_v, lower, upper = bootstrap_metric_ci(y_true, y_prob, metric="ROC-AUC", n_bootstraps=100)
        self.assertTrue(lower <= mean_v <= upper)

    def test_06_single_task_model_forward_shape(self):
        """Test 6: Verify SingleTaskToxicNet returns shape (B, 1)."""
        model = SingleTaskToxicNet(input_dim=2053, hidden_dims=[512, 256], latent_dim=128)
        x = torch.randn(8, 2053)
        out = model(x)
        self.assertEqual(out.shape, (8, 1))

    def test_07_multitask_eval_structure(self):
        """Test 7: Verify evaluate_multitask_predictions output columns and macro row."""
        probs = np.random.uniform(0, 1, size=(50, 12))
        labels = np.random.randint(0, 2, size=(50, 12)).astype(float)
        mask = np.ones((50, 12), dtype=bool)
        df = evaluate_multitask_predictions(probs, labels, mask, self.wrapper.target_assays)

        self.assertEqual(len(df), 13)  # 12 endpoints + 1 MACRO AVERAGE
        self.assertEqual(df.iloc[-1]["Endpoint"], "MACRO AVERAGE")
        self.assertIn("ROC-AUC", df.columns)
        self.assertIn("PR-AUC", df.columns)
        self.assertIn("F1-Score", df.columns)

    def test_08_configs_valid_json(self):
        """Test 8: Verify all 5 JSON configs in configs/ parse successfully."""
        config_files = [
            "evaluation_config.json",
            "seed_config.json",
            "dropout_config.json",
            "class_weight_config.json",
            "single_task_config.json",
        ]
        for cf in config_files:
            cpath = self.base_dir / "configs" / cf
            self.assertTrue(cpath.exists(), f"Missing config: {cf}")
            with open(cpath, "r") as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)

    def test_09_single_task_data_extraction(self):
        """Test 9: Verify extraction of observed single-task data."""
        X_obs, y_obs, mids_obs = self.wrapper.get_single_task_data("NR-AR", split="test")
        self.assertEqual(len(X_obs), len(y_obs))
        self.assertEqual(len(X_obs), len(mids_obs))
        self.assertEqual(X_obs.shape[1], 2053)
        self.assertEqual(y_obs.shape[1], 1)

    def test_10_directory_structure_completeness(self):
        """Test 10: Verify all designated Member 4 folders exist."""
        subdirs = [
            "configs", "src", "predictions/dropout", "predictions/class_weight",
            "predictions/multitask_single_task", "checkpoints/dropout",
            "checkpoints/class_weight", "checkpoints/single_task", "results",
            "plots/dropout", "plots/class_weight", "plots/multitask_single_task",
            "plots/model_comparison", "plots/error_analysis", "reports", "tests"
        ]
        for sub in subdirs:
            p = self.base_dir / sub
            self.assertTrue(p.exists(), f"Directory missing: {sub}")


if __name__ == "__main__":
    unittest.main()
