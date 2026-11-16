"""Export a consistent demo bundle without modifying historical team artifacts."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from phase2.member3_hybrid_model.src.hybrid_model import make_random_forest, load_config
from .predict import ARTIFACTS, DL_DIR, ENDPOINTS, HYBRID_DIR, ML_DIR, load_neural_model


def main():
    config = load_config()
    labels = pd.read_parquet(ARTIFACTS / 'labels_and_splits.parquet')
    features = pd.read_parquet(ARTIFACTS / 'X_combined.parquet').set_index('mol_id').loc[labels.mol_id].to_numpy(dtype=np.float64)
    model = load_neural_model(verify=False)
    with torch.inference_mode():
        chunks = [model.forward_with_embeddings(torch.from_numpy(x.astype(np.float32))) for x in np.array_split(features, 32)]
    probabilities = torch.sigmoid(torch.cat([x[0] for x in chunks])).numpy()
    embeddings = torch.cat([x[1] for x in chunks]).numpy()
    hybrid = np.concatenate([features, embeddings], axis=1)
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    test = labels.split.eq('test').to_numpy()
    exported = pd.DataFrame(embeddings[test], columns=[f'emb_{i:03}' for i in range(128)])
    exported.insert(0, 'molecule_id', labels.loc[test, 'mol_id'].to_numpy())
    exported.to_parquet(ML_DIR / 'checkpoint_test_embeddings.parquet', index=False)
    metrics = []
    for j, endpoint in enumerate(ENDPOINTS):
        train = labels.split.eq('train') & labels[endpoint].notna()
        filename = f"rf_{endpoint.replace('-', '_')}.joblib"
        # Reuse the evaluated feature-fusion forest instead of refitting on
        # numerically regenerated embeddings (which may alter tree splits).
        forest = joblib.load(DL_DIR.parent / 'member3_hybrid_model/models' / filename)
        joblib.dump(forest, HYBRID_DIR / filename, compress=3)
        baseline = make_random_forest(config)
        baseline.fit(features[train], labels.loc[train, endpoint].astype(int))
        joblib.dump(baseline, ML_DIR / filename, compress=3)
        for split in ['val', 'test']:
            obs = labels.split.eq(split) & labels[endpoint].notna()
            y = labels.loc[obs, endpoint]
            scores = {'Multitask DL': probabilities[obs, j],
                      'Feature fusion': forest.predict_proba(hybrid[obs])[:, 1],
                      'Stronger RF': baseline.predict_proba(features[obs])[:, 1]}
            scores['Probability blend'] = 0.80 * scores['Stronger RF'] + 0.20 * scores['Multitask DL']
            for name, score in scores.items():
                metrics.append(dict(endpoint=endpoint, model=name, split=split,
                                    roc_auc=round(float(roc_auc_score(y, score)), 4),
                                    average_precision=round(float(average_precision_score(y, score)), 4)))
        print('Prepared checkpoint-consistent hybrid:', endpoint)
    pd.DataFrame(metrics).to_csv(ML_DIR / 'inference_metrics.csv', index=False)
    checkpoint = DL_DIR / 'checkpoints/best_multitask_model.pt'
    metadata = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'sklearn_version': sklearn.__version__, 'torch_version': str(torch.__version__),
                'training_split': 'train', 'threshold': 0.5,
                'alpha': 0.80, 'feature_fusion_dim': 2181, 'rf_settings': config['random_forest'],
                'note': 'Current residual DL; 200-tree raw/fused RF; validation-selected 80/20 probability blend.'}
    (ML_DIR / 'bundle.json').write_text(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
