# Changelog

All notable changes to ExplainX AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-26

### Added
- Initial release of ExplainX AI — a fully local Explainable AI platform.
- FastAPI backend with SQLite-backed model registry, explanation audit log,
  and fairness audit log.
- Three built-in datasets: breast cancer (classification), California housing
  (regression, with an offline synthetic fallback), and a synthetic loan
  approval dataset with a protected attribute for bias testing.
- Model training endpoint supporting RandomForest, GradientBoosting, and
  Logistic Regression algorithms.
- Global explainability via SHAP (`TreeExplainer` / `KernelExplainer` fallback).
- Local (per-instance) explainability via SHAP and LIME.
- Bias & fairness evaluation: demographic parity, disparate impact ratio
  (four-fifths rule), equal opportunity, and predictive parity, each with a
  PASS/WARN/FAIL verdict.
- Streamlit dashboard with a light, professional theme: Overview, Train a
  Model, Global Explanations, Local Explanations (What-If), Bias & Fairness
  Audit, and Explanation Audit Log pages.
- Full local audit trail — every explanation and fairness check is logged to
  SQLite for compliance and traceability.

### Fixed
- Resolved a `DetachedInstanceError` where cached SQLAlchemy model records
  became invalid across requests; the engine cache now stores plain snapshot
  values instead of live ORM objects.
- Resolved a `ValueError: Out of range float values are not JSON compliant: nan`
  in the fairness evaluation endpoint by sanitizing `NaN`/`inf` values to
  `None` before serialization.
