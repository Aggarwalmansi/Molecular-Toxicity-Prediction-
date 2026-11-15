"""
hybrid_model.py
===============
Trains one Random Forest per toxicity endpoint on the hybrid feature set.

Why one forest per endpoint instead of one multi-output model?
A molecule can have a label for SR-MMP but no label for NR-AR. Training each
endpoint separately lets us drop exactly the molecules whose label for THAT
endpoint is missing, and keep all the others. A single multi-output forest
would force us to either drop a molecule entirely or invent a label for it.

Missing labels are never treated as negatives. For each endpoint we keep only
the rows where Member 1's observation mask is True.
"""

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

TARGET_ASSAYS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]


def get_module_dir():
    """Returns the hybrid folder."""
    return Path(__file__).resolve().parent.parent


def load_config():
    """Reads the Random Forest settings from configs/hybrid_config.json."""
    config_file = get_module_dir() / "configs" / "hybrid_config.json"
    with open(config_file, "r") as f:
        return json.load(f)


def make_random_forest(config):
    """
    Builds a Random Forest with the same settings as the Phase 1 baseline.

    Keeping the forest identical to Phase 1 matters: when we later compare the
    standalone Random Forest against the hybrid, the only thing that changed
    is the input features, not the learning algorithm.
    """
    settings = config["random_forest"]
    return RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        max_depth=settings["max_depth"],
        class_weight=settings["class_weight"],
        random_state=settings["random_state"],
        n_jobs=settings["n_jobs"],
    )


def select_observed_rows(features, labels, mask, endpoint_index):
    """
    Keeps only the molecules whose label for this endpoint was actually measured.

    Returns:
        x: features for the observed molecules
        y: their labels as integers (0 or 1)
        observed_positions: which rows were kept, so predictions can be mapped back
    """
    observed = mask[:, endpoint_index]
    observed_positions = np.where(observed)[0]
    x = features[observed_positions]
    y = labels[observed_positions, endpoint_index].astype(int)
    return x, y, observed_positions


def train_one_endpoint(features, labels, mask, endpoint_index, config):
    """Trains a single Random Forest for one endpoint and returns it with its training counts."""
    x_train, y_train, _ = select_observed_rows(features, labels, mask, endpoint_index)

    forest = make_random_forest(config)
    forest.fit(x_train, y_train)

    counts = {
        "observed": len(y_train),
        "positives": int((y_train == 1).sum()),
        "negatives": int((y_train == 0).sum()),
    }
    return forest, counts


def train_all_endpoints(features, labels, mask, config, verbose=True):
    """
    Trains all 12 forests.

    Returns a dictionary mapping endpoint name -> fitted forest.
    """
    forests = {}
    for endpoint_index, endpoint in enumerate(TARGET_ASSAYS):
        forest, counts = train_one_endpoint(features, labels, mask, endpoint_index, config)
        forests[endpoint] = forest
        if verbose:
            print(
                f"  {endpoint:16s} trained on {counts['observed']:5d} molecules "
                f"({counts['positives']:4d} active, {counts['negatives']:5d} inactive)"
            )
    return forests


def predict_probabilities(forests, features):
    """
    Predicts the probability of being toxic for every molecule and every endpoint.

    Returns an array of shape (n_molecules, 12). Every molecule gets a
    prediction for every endpoint, even where the true label is missing. The
    evaluation step is what filters those out using the mask.
    """
    n_molecules = features.shape[0]
    probabilities = np.zeros((n_molecules, len(TARGET_ASSAYS)))

    for endpoint_index, endpoint in enumerate(TARGET_ASSAYS):
        forest = forests[endpoint]
        # predict_proba returns [prob_inactive, prob_active]; column 1 is what we want
        probabilities[:, endpoint_index] = forest.predict_proba(features)[:, 1]

    return probabilities


def save_forests(forests, output_dir=None, compress=3):
    """
    Saves one .joblib file per endpoint so the models can be reused without retraining.

    compress=3 shrinks the 12 files from about 9 MB to about 2 MB. Loading a
    compressed model still takes well under a second, so there is no practical
    cost to keeping them small in the repository.
    """
    if output_dir is None:
        output_dir = get_module_dir() / "models"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for endpoint in forests:
        safe_name = endpoint.replace("-", "_")
        joblib.dump(forests[endpoint], output_dir / f"rf_{safe_name}.joblib", compress=compress)

    return output_dir


def load_forests(models_dir=None):
    """Loads the 12 saved forests back from disk."""
    if models_dir is None:
        models_dir = get_module_dir() / "models"
    models_dir = Path(models_dir)

    forests = {}
    for endpoint in TARGET_ASSAYS:
        safe_name = endpoint.replace("-", "_")
        model_file = models_dir / f"rf_{safe_name}.joblib"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing saved model: {model_file}")
        forests[endpoint] = joblib.load(model_file)

    return forests
