"""
ExplainX AI — Model utilities
Trains and persists scikit-learn models locally. No cloud calls, no external APIs.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer, fetch_california_housing, make_classification

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

ALGORITHMS = {
    "RandomForestClassifier": RandomForestClassifier,
    "RandomForestRegressor": RandomForestRegressor,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "LogisticRegression": LogisticRegression,
}


def load_builtin_dataset(dataset_name: str):
    """Loads a local, offline dataset (bundled with scikit-learn — no downloads)."""
    if dataset_name == "breast_cancer":
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        return X, y, "classification"
    elif dataset_name == "california_housing":
        try:
            data = fetch_california_housing(as_frame=True)
            X, y = data.data, data.target
        except Exception:
            # Fallback fully-synthetic regression set if housing fetch is blocked (no internet)
            rng = np.random.RandomState(42)
            X = pd.DataFrame(rng.rand(1000, 6), columns=[f"feature_{i}" for i in range(6)])
            y = pd.Series(X.sum(axis=1) * 10 + rng.randn(1000))
        return X, y, "regression"
    elif dataset_name == "synthetic_credit_risk":
        # Fully synthetic, generated locally — simulates a loan-approval dataset
        # including a "protected" demographic-like attribute for fairness testing.
        X, y = make_classification(
            n_samples=2000, n_features=8, n_informative=5, n_redundant=1,
            weights=[0.7, 0.3], random_state=42,
        )
        cols = ["income", "credit_score", "debt_ratio", "employment_years",
                "loan_amount", "age", "num_accounts", "protected_group"]
        X = pd.DataFrame(X, columns=cols)
        # Rescale into human-readable ranges
        X["income"] = (X["income"] * 20000 + 50000).round(0)
        X["credit_score"] = (X["credit_score"] * 100 + 650).round(0)
        X["age"] = (X["age"] * 15 + 35).round(0)
        X["protected_group"] = (X["protected_group"] > X["protected_group"].median()).astype(int)
        y = pd.Series(y, name="loan_approved")
        return X, y, "classification"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def train_model(model_name: str, algorithm: str, dataset_name: str, test_size: float = 0.2):
    """Trains a model locally and saves it to disk. Returns metrics + metadata."""
    X, y, task_type = load_builtin_dataset(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model_cls = ALGORITHMS[algorithm]
    if algorithm == "LogisticRegression":
        model = model_cls(max_iter=2000)
    else:
        model = model_cls(n_estimators=200, random_state=42) if "n_estimators" in model_cls().get_params() else model_cls(random_state=42)

    model.fit(X_train, y_train)

    metrics = {}
    if task_type == "classification":
        preds = model.predict(X_test)
        metrics["accuracy"] = float(accuracy_score(y_test, preds))
        metrics["f1_score"] = float(f1_score(y_test, preds, average="weighted"))
        metrics["rmse"] = None
        metrics["r2_score"] = None
    else:
        preds = model.predict(X_test)
        metrics["accuracy"] = None
        metrics["f1_score"] = None
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))
        metrics["r2_score"] = float(r2_score(y_test, preds))

    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump({"model": model, "X_test": X_test, "y_test": y_test, "X_train": X_train}, model_path)

    return {
        "model": model,
        "task_type": task_type,
        "feature_names": list(X.columns),
        "target_name": y.name if hasattr(y, "name") else "target",
        "n_features": X.shape[1],
        "n_samples_trained": X_train.shape[0],
        "model_path": model_path,
        **metrics,
    }


def load_model_bundle(model_name: str):
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found for '{model_name}'")
    return joblib.load(model_path)
