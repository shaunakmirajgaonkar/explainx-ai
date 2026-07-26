"""
ExplainX AI — Backend API
Fully local FastAPI service. No outbound network calls at runtime — all
model training, explanation, and fairness computation happens in-process.
"""
from typing import Optional, List
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from database import init_db, get_db, ModelRegistry, ExplanationLog, FairnessAudit
from model_utils import train_model, load_model_bundle, load_builtin_dataset
from explainer import ExplainabilityEngine
from fairness import evaluate_fairness, overall_fairness_verdict

app = FastAPI(title="ExplainX AI", description="Local Explainable AI Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

_engines_cache = {}


def get_engine(model_name: str, db: Session):
    if model_name in _engines_cache:
        return _engines_cache[model_name]
    record = db.query(ModelRegistry).filter(ModelRegistry.name == model_name).first()
    if not record:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    record_data = {
        "name": record.name,
        "feature_names": list(record.feature_names),
        "task_type": record.task_type,
    }

    bundle = load_model_bundle(model_name)
    engine = ExplainabilityEngine(
        model=bundle["model"], X_train=bundle["X_train"],
        feature_names=record_data["feature_names"], task_type=record_data["task_type"],
    )
    _engines_cache[model_name] = {"engine": engine, "bundle": bundle, "record": record_data}
    return _engines_cache[model_name]


# ----------------------------------------------------------------- Schemas
class TrainRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    algorithm: str
    dataset_name: str
    test_size: float = 0.2


class PredictRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    features: dict


class FairnessRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    protected_attribute: str
    privileged_value: float
    unprivileged_value: float


# ----------------------------------------------------------------- Routes
@app.get("/")
def root():
    return {"status": "ExplainX AI backend running locally", "version": "1.0.0"}


@app.get("/datasets")
def list_datasets():
    return {
        "datasets": [
            {"name": "breast_cancer", "task": "classification", "description": "Tumor diagnosis (built-in, offline)"},
            {"name": "california_housing", "task": "regression", "description": "Housing price regression"},
            {"name": "synthetic_credit_risk", "task": "classification", "description": "Synthetic loan approval with a protected attribute for bias testing"},
        ]
    }


@app.get("/algorithms")
def list_algorithms():
    return {"algorithms": ["RandomForestClassifier", "RandomForestRegressor", "GradientBoostingClassifier", "LogisticRegression"]}


@app.post("/train")
def train(req: TrainRequest, db: Session = Depends(get_db)):
    existing = db.query(ModelRegistry).filter(ModelRegistry.name == req.model_name).first()
    if existing:
        raise HTTPException(status_code=400, detail="A model with this name already exists")

    result = train_model(req.model_name, req.algorithm, req.dataset_name, req.test_size)

    record = ModelRegistry(
        name=req.model_name,
        task_type=result["task_type"],
        algorithm=req.algorithm,
        feature_names=result["feature_names"],
        target_name=result["target_name"],
        n_features=result["n_features"],
        n_samples_trained=result["n_samples_trained"],
        accuracy=result["accuracy"],
        f1_score=result["f1_score"],
        rmse=result["rmse"],
        r2_score=result["r2_score"],
        model_path=result["model_path"],
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    _engines_cache.pop(req.model_name, None)

    return {
        "message": f"Model '{req.model_name}' trained successfully",
        "task_type": result["task_type"],
        "metrics": {
            "accuracy": result["accuracy"], "f1_score": result["f1_score"],
            "rmse": result["rmse"], "r2_score": result["r2_score"],
        },
        "feature_names": result["feature_names"],
    }


@app.get("/models")
def list_models(db: Session = Depends(get_db)):
    records = db.query(ModelRegistry).all()
    return [{
        "name": r.name, "task_type": r.task_type, "algorithm": r.algorithm,
        "feature_names": r.feature_names, "n_features": r.n_features,
        "accuracy": r.accuracy, "f1_score": r.f1_score, "rmse": r.rmse, "r2_score": r.r2_score,
        "created_at": r.created_at.isoformat(),
    } for r in records]


@app.get("/models/{model_name}/sample")
def get_sample_data(model_name: str, n: int = 5, db: Session = Depends(get_db)):
    ctx = get_engine(model_name, db)
    X_test = ctx["bundle"]["X_test"]
    return X_test.head(n).to_dict(orient="records")


@app.post("/predict")
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    ctx = get_engine(req.model_name, db)
    engine, record = ctx["engine"], ctx["record"]
    instance = pd.DataFrame([req.features])[record["feature_names"]]
    result = engine.predict(instance)
    return result


@app.post("/explain/global")
def explain_global(model_name: str, db: Session = Depends(get_db)):
    ctx = get_engine(model_name, db)
    engine, bundle, record = ctx["engine"], ctx["bundle"], ctx["record"]
    importance_df = engine.global_feature_importance(bundle["X_test"])

    log = ExplanationLog(
        model_name=model_name, explanation_type="global_shap",
        top_features=importance_df.head(10).to_dict(orient="records"),
    )
    db.add(log)
    db.commit()

    return {"feature_importance": importance_df.to_dict(orient="records")}


@app.post("/explain/local")
def explain_local(req: PredictRequest, method: str = "shap", db: Session = Depends(get_db)):
    ctx = get_engine(req.model_name, db)
    engine, record = ctx["engine"], ctx["record"]
    instance = pd.DataFrame([req.features])[record["feature_names"]]
    prediction = engine.predict(instance)

    if method == "lime":
        explanation = engine.lime_explanation(instance)
        result = {"method": "lime", "explanation": [{"feature": f, "contribution": c} for f, c in explanation]}
    else:
        df, base_value = engine.local_shap_explanation(instance)
        result = {"method": "shap", "base_value": base_value, "explanation": df.to_dict(orient="records")}

    log = ExplanationLog(
        model_name=req.model_name, explanation_type=f"local_{method}",
        input_data=req.features, prediction=prediction.get("prediction"),
        top_features=result["explanation"][:5],
    )
    db.add(log)
    db.commit()

    result["prediction"] = prediction
    return result


@app.post("/fairness/evaluate")
def fairness_evaluate(req: FairnessRequest, db: Session = Depends(get_db)):
    ctx = get_engine(req.model_name, db)
    bundle, record = ctx["bundle"], ctx["record"]

    if req.protected_attribute not in record["feature_names"]:
        raise HTTPException(status_code=400, detail=f"'{req.protected_attribute}' is not a feature of this model")

    X_test, y_test = bundle["X_test"], bundle["y_test"]
    y_pred = bundle["model"].predict(X_test)
    protected_col = X_test[req.protected_attribute].values

    results = evaluate_fairness(
        y_true=np.asarray(y_test), y_pred=np.asarray(y_pred),
        protected_attr=protected_col,
        privileged_value=req.privileged_value, unprivileged_value=req.unprivileged_value,
    )
    verdict = overall_fairness_verdict(results)

    for metric_name, metric_data in results.items():
        if isinstance(metric_data, dict) and "status" in metric_data:
            audit = FairnessAudit(
                model_name=req.model_name, protected_attribute=req.protected_attribute,
                metric_name=metric_name, metric_value=metric_data.get("value") or 0.0,
                threshold_passed=metric_data["status"], details=metric_data,
            )
            db.add(audit)
    db.commit()

    return {"overall_verdict": verdict, "metrics": results}


@app.get("/audit/explanations/{model_name}")
def get_explanation_audit(model_name: str, limit: int = 50, db: Session = Depends(get_db)):
    logs = (db.query(ExplanationLog)
            .filter(ExplanationLog.model_name == model_name)
            .order_by(ExplanationLog.created_at.desc()).limit(limit).all())
    return [{
        "id": l.id, "type": l.explanation_type, "prediction": l.prediction,
        "top_features": l.top_features, "created_at": l.created_at.isoformat(),
    } for l in logs]


@app.get("/audit/fairness/{model_name}")
def get_fairness_audit(model_name: str, db: Session = Depends(get_db)):
    logs = (db.query(FairnessAudit)
            .filter(FairnessAudit.model_name == model_name)
            .order_by(FairnessAudit.created_at.desc()).all())
    return [{
        "protected_attribute": l.protected_attribute, "metric_name": l.metric_name,
        "metric_value": l.metric_value, "status": l.threshold_passed,
        "created_at": l.created_at.isoformat(),
    } for l in logs]


@app.delete("/models/{model_name}")
def delete_model(model_name: str, db: Session = Depends(get_db)):
    record = db.query(ModelRegistry).filter(ModelRegistry.name == model_name).first()
    if not record:
        raise HTTPException(status_code=404, detail="Model not found")
    db.delete(record)
    db.commit()
    _engines_cache.pop(model_name, None)
    return {"message": f"Model '{model_name}' deleted"}
