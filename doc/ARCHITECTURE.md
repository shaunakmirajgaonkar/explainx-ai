# Architecture

ExplainX AI is a two-tier local application: a FastAPI backend that owns all
ML logic and persistence, and a Streamlit dashboard that is a thin client
over the backend's REST API.

```
┌─────────────────────────┐        HTTP (localhost only)        ┌──────────────────────────┐
│   Streamlit Dashboard    │ ───────────────────────────────────▶│    FastAPI Backend       │
│   (dashboard/app.py)     │◀─────────────────────────────────── │    (backend/main.py)     │
└─────────────────────────┘                                      └──────────────┬───────────┘
                                                                                  │
                                        ┌─────────────────────────────────────────┼───────────────────────────────┐
                                        │                                         │                               │
                                        ▼                                         ▼                               ▼
                            ┌───────────────────────┐               ┌───────────────────────┐       ┌───────────────────────┐
                            │   model_utils.py       │               │   explainer.py        │       │   fairness.py          │
                            │   scikit-learn training│               │   SHAP + LIME engine   │       │   bias/fairness metrics│
                            └───────────┬───────────┘               └───────────┬───────────┘       └───────────────────────┘
                                        │                                         │
                                        ▼                                         ▼
                            ┌───────────────────────┐               ┌───────────────────────┐
                            │   models/*.joblib      │               │   database.py          │
                            │   trained model bundles│               │   SQLAlchemy + SQLite  │
                            └───────────────────────┘               └───────────────────────┘
```

## Components

### `backend/main.py`
The FastAPI application. Owns all routes (`/train`, `/predict`,
`/explain/global`, `/explain/local`, `/fairness/evaluate`, audit endpoints).
Maintains an in-process engine cache keyed by model name, storing plain
snapshot values (not live SQLAlchemy ORM objects) to avoid session-lifecycle
bugs across requests.

### `backend/database.py`
SQLAlchemy models backed by a local SQLite file (`explainx.db`):
- `ModelRegistry` — every trained model's metadata (algorithm, features, metrics)
- `ExplanationLog` — audit trail of every SHAP/LIME explanation generated
- `FairnessAudit` — audit trail of every fairness evaluation run

### `backend/model_utils.py`
Trains and persists scikit-learn models (`RandomForestClassifier`,
`RandomForestRegressor`, `GradientBoostingClassifier`, `LogisticRegression`)
against one of three built-in datasets, saving the fitted model plus train/test
splits to `models/*.joblib` via `joblib`.

### `backend/explainer.py`
Wraps SHAP (`TreeExplainer` with `KernelExplainer` fallback for non-tree
models) and LIME (`LimeTabularExplainer`) to produce:
- Global feature importance (mean |SHAP value| across a sample)
- Local per-instance explanations (SHAP contributions or LIME weights)

### `backend/fairness.py`
Computes group-fairness metrics comparing a "privileged" vs. "unprivileged"
group on a chosen protected attribute: demographic parity difference,
disparate impact ratio (four-fifths rule), equal opportunity difference, and
predictive parity difference. All NaN/inf values (e.g. from an empty group)
are sanitized to `None` before being returned as JSON.

### `dashboard/app.py`
A Streamlit application with a light, professional theme. Talks to the
backend exclusively over `http://127.0.0.1:8000` — no other network calls.
Pages: Overview, Train a Model, Global Explanations, Local Explanations
(What-If), Bias & Fairness Audit, Explanation Audit Log.

## Data flow

1. **Train**: Dashboard → `POST /train` → `model_utils.train_model()` →
   model saved to `models/*.joblib` → metadata saved to `ModelRegistry` (SQLite)
2. **Explain**: Dashboard → `POST /explain/global` or `/explain/local` →
   `get_engine()` loads the cached/joblib model → `ExplainabilityEngine`
   computes SHAP/LIME → result logged to `ExplanationLog` → returned to dashboard
3. **Audit fairness**: Dashboard → `POST /fairness/evaluate` → predictions
   generated on the held-out test set → `fairness.evaluate_fairness()`
   computes metrics → result logged to `FairnessAudit` → returned to dashboard

## Design principles

- **Local-first**: no external API calls at inference or training time.
- **Stateless engine, stateful storage**: the in-memory engine cache is a
  performance optimization only; all durable state lives in SQLite and on disk.
- **Auditability**: every explanation and fairness check is logged, so the
  platform can support compliance and traceability requirements out of the box.
