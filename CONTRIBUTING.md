# Contributing to ExplainX AI

Thanks for your interest in contributing! This project is a fully local
Explainable AI platform, and contributions that keep it dependency-light and
100% local are especially welcome.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run the backend and dashboard locally (see `README.md` / `run instructions.md`).
4. Create a feature branch: `git checkout -b feature/my-improvement`.

## Development guidelines

- **Keep it local.** No external API calls, telemetry, or cloud dependencies
  should be introduced at runtime.
- **Follow the existing structure.** Backend logic lives in `backend/`
  (`main.py` for routes, `model_utils.py` for training, `explainer.py` for
  SHAP/LIME, `fairness.py` for bias metrics, `database.py` for persistence).
  Dashboard code lives in `dashboard/app.py`.
- **Add datasets via `load_builtin_dataset()`** in `backend/model_utils.py`
  rather than hardcoding new training logic elsewhere.
- **Write clear commit messages** describing the "why," not just the "what."
- **Test your changes locally** — train a model, run global/local
  explanations, and run a fairness audit end-to-end before opening a PR.

## Submitting changes

1. Push your branch and open a pull request against `main`.
2. Describe what you changed and why, and include any relevant screenshots
   of the dashboard if the UI changed.
3. Be responsive to review feedback — small, focused PRs are easiest to
   review and merge quickly.

## Reporting bugs

Please open an issue with:
- Steps to reproduce
- Expected vs. actual behavior
- Your Python version and OS
- The full traceback from the backend terminal, if applicable

## Code of Conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Please be
respectful and constructive in all interactions.
