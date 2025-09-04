from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
except Exception:  # optional dependency
    XGBClassifier = None  # type: ignore

def default_models(random_state: int = 42, n_jobs: int = 40) -> dict:
    """
    Return a dict of preconfigured baseline classifiers:
      - Random Forest
      - XGBoost (if available)
      - MLP (ANN)
    """
    models = {
        "Random Forest": RandomForestClassifier(
            max_depth=3, max_features="log2", min_samples_leaf=1,
            min_samples_split=8, n_estimators=270,
            n_jobs=n_jobs, random_state=random_state
        ),
        "ANN": MLPClassifier(
            hidden_layer_sizes=(50,), activation="tanh", alpha=0.0070,
            learning_rate="invscaling", solver="sgd",
            max_iter=1000, random_state=random_state
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            learning_rate=0.01, max_depth=7, colsample_bytree=0.73, gamma=1.96,
            min_child_weight=8, subsample=0.71, n_estimators=150,
            random_state=random_state, eval_metric="logloss", n_jobs=n_jobs,
        )
    return models
