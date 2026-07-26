"""
ExplainX AI — Explainability engine
Wraps SHAP (global + local) and LIME (local) explanations. Runs entirely
in-process on local compute — no calls to any external explanation service.
"""
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


class ExplainabilityEngine:
    def __init__(self, model, X_train: pd.DataFrame, feature_names: list, task_type: str):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.task_type = task_type
        self._shap_explainer = None

    # ---------------------------------------------------------------- SHAP
    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            try:
                self._shap_explainer = shap.TreeExplainer(self.model)
            except Exception:
                # Fallback for models SHAP can't introspect directly (e.g. LogisticRegression)
                background = shap.sample(self.X_train, min(100, len(self.X_train)), random_state=42)
                predict_fn = (
                    self.model.predict_proba if self.task_type == "classification" else self.model.predict
                )
                self._shap_explainer = shap.KernelExplainer(predict_fn, background)
        return self._shap_explainer

    def global_feature_importance(self, X_sample: pd.DataFrame, max_samples: int = 200):
        """Mean |SHAP value| per feature across a sample — the model's overall behavior."""
        X_sample = X_sample.sample(min(max_samples, len(X_sample)), random_state=42) if len(X_sample) > max_samples else X_sample
        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):  # multiclass returns list of arrays
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:  # (samples, features, classes)
            shap_values = shap_values[:, :, 1]

        importance = np.abs(shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def local_shap_explanation(self, instance: pd.DataFrame):
        """Per-feature SHAP contribution for a single prediction."""
        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(instance)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        contributions = shap_values[0]
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1] if len(np.atleast_1d(base_value)) > 1 else np.atleast_1d(base_value)[0]

        df = pd.DataFrame({
            "feature": self.feature_names,
            "value": instance.iloc[0].values,
            "shap_contribution": contributions,
        }).sort_values("shap_contribution", key=np.abs, ascending=False).reset_index(drop=True)

        return df, float(base_value)

    # ---------------------------------------------------------------- LIME
    def lime_explanation(self, instance: pd.DataFrame, num_features: int = 10):
        mode = "classification" if self.task_type == "classification" else "regression"
        explainer = LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            mode=mode,
            discretize_continuous=True,
            random_state=42,
        )
        predict_fn = self.model.predict_proba if mode == "classification" else self.model.predict
        exp = explainer.explain_instance(
            instance.iloc[0].values, predict_fn, num_features=num_features
        )
        return exp.as_list()

    # -------------------------------------------------------- Prediction
    def predict(self, instance: pd.DataFrame):
        if self.task_type == "classification":
            proba = self.model.predict_proba(instance)[0]
            pred_class = int(np.argmax(proba))
            return {"prediction": pred_class, "probability": float(proba[pred_class]), "all_probabilities": proba.tolist()}
        else:
            pred = float(self.model.predict(instance)[0])
            return {"prediction": pred, "probability": None, "all_probabilities": None}
