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
from .export_baseline import main as export_baseline
from .predict import ARTIFACTS, DL_DIR, ENDPOINTS, HYBRID_DIR, ML_DIR, load_neural_model


def main():
    export_baseline()
    labels = pd.read_parquet(ARTIFACTS / 'labels_and_splits.parquet')
    features = pd.read_parquet(ARTIFACTS / 'X_combined.parquet').set_index('mol_id').loc[labels.mol_id].to_numpy(dtype=np.float32)
    model = load_neural_model(verify=False)
    with torch.inference_mode():
        chunks = [model.forward_with_embeddings(torch.from_numpy(x)) for x in np.array_split(features, 32)]
    probabilities = torch.sigmoid(torch.cat([x[0] for x in chunks])).numpy()
    embeddings = torch.cat([x[1] for x in chunks]).numpy()
    hybrid = np.concatenate([embeddings, features[:, -5:]], axis=1)
    HYBRID_DIR.mkdir(parents=True, exist_ok=True)
    test = labels.split.eq('test').to_numpy()
    exported = pd.DataFrame(embeddings[test], columns=[f'emb_{i:03}' for i in range(128)])
    exported.insert(0, 'molecule_id', labels.loc[test, 'mol_id'].to_numpy())
    exported.to_parquet(ML_DIR / 'checkpoint_test_embeddings.parquet', index=False)
    metrics = []
    for j, endpoint in enumerate(ENDPOINTS):
        train = labels.split.eq('train') & labels[endpoint].notna()
        forest = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
        forest.fit(hybrid[train], labels.loc[train, endpoint])
        filename = f"rf_{endpoint.replace('-', '_')}.joblib"
        joblib.dump(forest, HYBRID_DIR / filename, compress=3)
        baseline = joblib.load(ML_DIR / filename)
        for split in ['val', 'test']:
            obs = labels.split.eq(split) & labels[endpoint].notna()
            y = labels.loc[obs, endpoint]
            scores = {'Multitask DL': probabilities[obs, j],
                      'Hybrid': forest.predict_proba(hybrid[obs])[:, 1],
                      'Random Forest': baseline.predict_proba(features[obs])[:, 1]}
            for name, score in scores.items():
                metrics.append(dict(endpoint=endpoint, model=name, split=split,
                                    roc_auc=roc_auc_score(y, score), average_precision=average_precision_score(y, score)))
        print('Prepared checkpoint-consistent hybrid:', endpoint)
    pd.DataFrame(metrics).to_csv(ML_DIR / 'inference_metrics.csv', index=False)
    checkpoint = DL_DIR / 'checkpoints/best_multitask_model.pt'
    metadata = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                'sklearn_version': sklearn.__version__, 'torch_version': str(torch.__version__),
                'training_split': 'train', 'threshold': 0.5,
                'note': 'Member 5 demo refit; original Member 3 hybrid artifacts are unchanged.'}
    (ML_DIR / 'bundle.json').write_text(json.dumps(metadata, indent=2))

if __name__ == '__main__':
    main()
