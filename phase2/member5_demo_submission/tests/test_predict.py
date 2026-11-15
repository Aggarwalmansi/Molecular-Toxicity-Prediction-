"""Integration checks against the team's actual artifacts and saved models."""
import unittest
import numpy as np
import pandas as pd
import torch
from phase2.member5_demo_submission.predict import (
    ARTIFACTS, DL_DIR, ENDPOINTS, featurize, load_neural_model, predict,
)

class PredictionTests(unittest.TestCase):
    def test_invalid_input(self):
        for value in ['', '   ', 'not-a-molecule', '[', '.']:
            with self.subTest(smiles=value), self.assertRaises(ValueError):
                featurize(value)

    def test_feature_parity(self):
        metadata = pd.read_parquet(ARTIFACTS / 'labels_and_splits.parquet')
        features = pd.read_parquet(ARTIFACTS / 'X_combined.parquet').set_index('mol_id')
        for row in metadata.iloc[[0, 100, 1000]].itertuples():
            _, actual, _ = featurize(row.smiles)
            np.testing.assert_allclose(actual[0], features.loc[row.mol_id].to_numpy(), atol=1e-5)

    def test_embedding_parity(self):
        saved = pd.read_parquet(DL_DIR.parents[0] / 'member5_demo_submission/models/checkpoint_test_embeddings.parquet')
        metadata = pd.read_parquet(ARTIFACTS / 'labels_and_splits.parquet').set_index('mol_id')
        row = saved.iloc[0]
        _, features, _ = featurize(metadata.loc[row.molecule_id, 'smiles'])
        with torch.inference_mode():
            embedding = load_neural_model().extract_latent(torch.from_numpy(features)).numpy()[0]
        expected = row.drop(labels=['molecule_id']).to_numpy(dtype=float)
        np.testing.assert_allclose(embedding, expected, atol=1e-4, rtol=1e-4)

    def test_predictions(self):
        first = predict('CCO')
        second = predict('CCO')
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(set(first.endpoint), set(ENDPOINTS))
        self.assertEqual(set(first.model), {'Multitask DL', 'Hybrid', 'Random Forest'})
        self.assertEqual(len(first), 36)
        self.assertTrue(first.activity_score.between(0, 1).all())
        with self.assertRaises(ValueError):
            predict('CCO', threshold=1.1)

if __name__ == '__main__':
    unittest.main()
