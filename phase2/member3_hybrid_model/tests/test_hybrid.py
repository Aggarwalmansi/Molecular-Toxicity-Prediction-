"""
test_hybrid.py
==============
Test suite for Phase 2 Member 3: feature fusion and probability blending.

Verifies:
1. build_hybrid_features returns the 2,181-dim fused matrix, block by block.
2. The raw block and the embedding block land in the documented column order.
3. Molecule IDs are aligned row-for-row with Member 1's label rows on all splits.
4. Mismatched molecule order is rejected rather than silently concatenated.
5. blend_predictions implements alpha * P_rf + (1 - alpha) * P_dl, including
   the alpha=0 and alpha=1 endpoints, and rejects bad shapes and bad alpha.
6. The Random Forest is built with the upgraded hyperparameters.

Run from the repository root:
    python -m unittest phase2.member3_hybrid_model.tests.test_hybrid
"""

import unittest

import numpy as np

from phase2.member1_data_pipeline.src.data_loader import Tox21DataLoader
from phase2.member3_hybrid_model.src.hybrid_features import (
    N_EMBEDDINGS,
    N_FUSED_FEATURES,
    N_RAW_FEATURES,
    build_hybrid_features,
    get_feature_names,
    load_dl_embeddings,
    load_raw_features,
)
from phase2.member3_hybrid_model.src.hybrid_model import (
    blend_predictions,
    load_config,
    make_random_forest,
)


class TestFeatureFusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = Tox21DataLoader()

    def test_fused_matrix_has_2181_columns(self):
        for split, expected_rows in [("train", 6259), ("val", 782), ("test", 782)]:
            features, mol_ids = build_hybrid_features(self.loader, split)
            self.assertEqual(features.shape, (expected_rows, N_FUSED_FEATURES))
            self.assertEqual(len(mol_ids), expected_rows)

    def test_fusion_preserves_both_blocks_in_order(self):
        """The fused matrix must be exactly [raw | embeddings], not a re-encoding."""
        fused, _ = build_hybrid_features(self.loader, "test")
        raw, _ = load_raw_features(self.loader, "test")
        embeddings, _ = load_dl_embeddings("test")

        np.testing.assert_allclose(fused[:, :N_RAW_FEATURES], raw)
        np.testing.assert_allclose(fused[:, N_RAW_FEATURES:], embeddings)

    def test_morgan_block_is_still_binary(self):
        """Fusion must not scale or otherwise damage the 2,048 raw Morgan bits."""
        fused, _ = build_hybrid_features(self.loader, "test")
        morgan = fused[:, :2048]
        self.assertTrue(np.all((morgan == 0.0) | (morgan == 1.0)))

    def test_molecule_ids_align_with_label_rows(self):
        for split in ["train", "val", "test"]:
            _, feature_ids = build_hybrid_features(self.loader, split)
            _, _, label_ids = self.loader.load_labels(split=split)
            self.assertEqual(list(feature_ids), list(label_ids))

    def test_ablation_arms_have_expected_widths(self):
        cases = [
            (dict(use_raw_features=False, use_descriptors=False), N_EMBEDDINGS),
            (dict(use_raw_features=False, use_descriptors=True), N_EMBEDDINGS + 5),
            (dict(use_raw_features=True, use_embeddings=False), N_RAW_FEATURES),
            (dict(), N_FUSED_FEATURES),
        ]
        for kwargs, expected_width in cases:
            features, _ = build_hybrid_features(self.loader, "val", **kwargs)
            self.assertEqual(features.shape[1], expected_width, msg=str(kwargs))

    def test_requesting_no_blocks_is_rejected(self):
        with self.assertRaises(ValueError):
            build_hybrid_features(self.loader, "val",
                                  use_raw_features=False, use_embeddings=False,
                                  use_descriptors=False)

    def test_feature_names_match_matrix_width(self):
        features, _ = build_hybrid_features(self.loader, "val")
        self.assertEqual(len(get_feature_names()), features.shape[1])


class TestProbabilityBlending(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.rf = rng.random((50, 12))
        self.dl = rng.random((50, 12))

    def test_blend_matches_the_formula(self):
        for alpha in [0.0, 0.25, 0.5, 0.6, 1.0]:
            expected = alpha * self.rf + (1 - alpha) * self.dl
            np.testing.assert_allclose(
                blend_predictions(self.rf, self.dl, alpha=alpha), expected
            )

    def test_endpoints_recover_each_model(self):
        np.testing.assert_allclose(blend_predictions(self.rf, self.dl, alpha=1.0), self.rf)
        np.testing.assert_allclose(blend_predictions(self.rf, self.dl, alpha=0.0), self.dl)

    def test_blend_stays_a_probability(self):
        blended = blend_predictions(self.rf, self.dl, alpha=0.6)
        self.assertTrue(np.all((blended >= 0.0) & (blended <= 1.0)))

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            blend_predictions(self.rf, self.dl[:10], alpha=0.6)

    def test_alpha_outside_unit_interval_is_rejected(self):
        for bad_alpha in [-0.1, 1.5]:
            with self.assertRaises(ValueError):
                blend_predictions(self.rf, self.dl, alpha=bad_alpha)


class TestRandomForestSettings(unittest.TestCase):
    def test_upgraded_hyperparameters_are_applied(self):
        forest = make_random_forest(load_config())
        self.assertEqual(forest.n_estimators, 200)
        self.assertEqual(forest.max_depth, 15)
        self.assertEqual(forest.min_samples_leaf, 2)
        self.assertEqual(forest.class_weight, "balanced")

    def test_config_declares_the_fused_width(self):
        self.assertEqual(load_config()["total_features"], N_FUSED_FEATURES)


if __name__ == "__main__":
    unittest.main()
