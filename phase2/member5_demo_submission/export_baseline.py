"""Save Phase 1 RF models for inference, using observed training labels only."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .predict import ARTIFACTS, ENDPOINTS, ML_DIR


def main():
    labels = pd.read_parquet(ARTIFACTS / 'labels_and_splits.parquet')
    features = pd.read_parquet(ARTIFACTS / 'X_combined.parquet').set_index('mol_id')
    features = features.loc[labels.mol_id].to_numpy()
    ML_DIR.mkdir(exist_ok=True)
    for endpoint in ENDPOINTS:
        observed = labels['split'].eq('train') & labels[endpoint].notna()
        model = RandomForestClassifier(n_estimators=100, max_depth=12,
                                       class_weight='balanced', random_state=42, n_jobs=-1)
        model.fit(features[observed], labels.loc[observed, endpoint])
        joblib.dump(model, ML_DIR / f"rf_{endpoint.replace('-', '_')}.joblib", compress=3)
        print(f'Saved {endpoint} ({observed.sum()} observed training molecules)')


if __name__ == '__main__':
    main()
