# ExplainX AI

A production-ready **Explainable AI (XAI)** platform for understanding, validating, and
trusting machine learning models — global & local interpretability, feature importance,
bias detection, fairness evaluation, and model debugging, all through an interactive
dashboard running **entirely on local infrastructure**.

No cloud APIs. No external services. No data leaves your machine.

## Architecture

```
explainx-ai/
├── backend/
│   ├── main.py          # FastAPI app — all REST endpoints
│   ├── database.py      # SQLAlchemy models (SQLite, local file)
│   ├── model_utils.py   # Training / loading scikit-learn models
│   ├── explainer.py      # SHAP + LIME explainability engine
│   └── fairness.py       # Bias & fairness metrics
├── dashboard/
│   └── app.py            # Streamlit dashboard (talks to localhost API)
├── models/                # Trained model artifacts (.joblib)
├── requirements.txt
├── run_backend.sh
└── run_dashboard.sh
```

## Stack

- **FastAPI** — local REST API, no external network calls at runtime
- **SQLAlchemy + SQLite** — model registry, explanation audit log, fairness audit log
- **scikit-learn** — RandomForest, GradientBoosting, LogisticRegression
- **SHAP + LIME** — global and local model explanations
- **Streamlit + Plotly** — interactive dashboard UI

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Terminal 1
./run_backend.sh

# Terminal 2
./run_dashboard.sh
```

Then open **http://127.0.0.1:8501** in your browser.

## What you can do

1. **Train a Model** — pick a built-in dataset (breast cancer diagnosis, California
   housing, or a synthetic loan-approval dataset with a protected attribute) and an
   algorithm, and train it locally in seconds.
2. **Global Explanations** — see mean |SHAP value| feature importance across the
   whole test set to understand overall model behavior.
3. **Local Explanations (What-If)** — tweak input values for a single instance and
   see exactly which features pushed the prediction up or down, via SHAP or LIME.
4. **Bias & Fairness Audit** — pick any feature as a "protected attribute" and get
   demographic parity, disparate impact ratio (four-fifths rule), equal opportunity,
   and predictive parity, with PASS/WARN/FAIL verdicts.
5. **Explanation Audit Log** — every explanation and fairness check is logged to
   SQLite for compliance and traceability.

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/datasets` | List built-in datasets |
| GET | `/algorithms` | List supported algorithms |
| POST | `/train` | Train a new model |
| GET | `/models` | List registered models |
| POST | `/predict` | Get a prediction for a single instance |
| POST | `/explain/global` | Global SHAP feature importance |
| POST | `/explain/local` | Local SHAP or LIME explanation |
| POST | `/fairness/evaluate` | Run a fairness audit |
| GET | `/audit/explanations/{model}` | Explanation history |
| GET | `/audit/fairness/{model}` | Fairness audit history |

## Notes

- All data and models are stored locally: `backend/explainx.db` (SQLite) and
  `models/*.joblib`.
- The `california_housing` dataset falls back to a fully synthetic regression
  set if fetched without an internet connection (SPPU labs / offline machines).
- Add your own datasets by extending `load_builtin_dataset()` in
  `backend/model_utils.py`.
