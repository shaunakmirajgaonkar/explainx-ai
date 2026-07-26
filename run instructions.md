# Run Instructions

## Prerequisites

- Python 3.10–3.12 recommended (see note below for very new Python versions)
- macOS, Linux, or Windows with a terminal

## 1. Set up a virtual environment

```bash
cd explainx-ai
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note for very new Python versions (e.g. 3.13+):** some pinned package
> versions may not have prebuilt wheels yet and will try to compile from
> source, which can fail (e.g. matplotlib/freetype build errors). If this
> happens, relax the exact version pins in `requirements.txt` to unpinned
> package names and re-run `pip install -r requirements.txt` — pip will
> select versions with prebuilt wheels for your Python version.

## 3. Start the backend (Terminal 1)

```bash
cd backend
python3 -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Wait for:
```
INFO:     Application startup complete.
```

Leave this terminal running.

## 4. Start the dashboard (Terminal 2 — new tab)

```bash
cd explainx-ai
source venv/bin/activate
python3 -m streamlit run dashboard/app.py --server.port 8501
```

Open **http://127.0.0.1:8501** in your browser.

## 5. Using the dashboard

1. **Train a Model** — pick a dataset and algorithm, give it a name, click Train.
2. **Global Explanations** — select your trained model, click Compute.
3. **Local Explanations (What-If)** — adjust feature values, choose SHAP or
   LIME, click Explain.
4. **Bias & Fairness Audit** — for the synthetic credit-risk dataset, use
   `protected_group` as the protected attribute, `0` as privileged, `1` as
   unprivileged.
5. **Explanation Audit Log** — review everything logged to local SQLite.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'sqlalchemy'` (or similar)** — your
  venv isn't active, or a system/Homebrew Python is shadowing it. Run
  `source venv/bin/activate` and confirm with `which python3` that it points
  inside `venv/bin/`. Always launch with `python3 -m uvicorn ...` /
  `python3 -m streamlit ...` rather than the bare `uvicorn`/`streamlit`
  command, to guarantee the venv's interpreter is used.
- **`error: externally-managed-environment`** — you're trying to `pip
  install` outside a virtual environment on a Homebrew-managed Python. Use
  the venv steps above instead of installing system-wide.
- **`Address already in use` / `Port 8501 is not available`** — a previous
  session is still running. Find and stop it:
  ```bash
  lsof -ti:8000 | xargs kill -9   # for the backend
  lsof -ti:8501 | xargs kill -9   # for the dashboard
  ```
- **`Internal Server Error` in the dashboard** — check the backend terminal
  for the full Python traceback; it will show the exact failing line.
