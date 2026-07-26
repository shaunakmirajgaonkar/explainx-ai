"""
ExplainX AI — Interactive Dashboard
100% local Streamlit UI talking to the local FastAPI backend (localhost only).
"""
import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="ExplainX AI",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------- Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

    :root {
        --ex-bg: #f7f8fa;
        --ex-panel: #ffffff;
        --ex-panel-2: #f3f4f7;
        --ex-border: #e3e6ec;
        --ex-accent: #4f46e5;
        --ex-accent-2: #0891b2;
        --ex-good: #059669;
        --ex-warn: #b45309;
        --ex-bad: #dc2626;
        --ex-text: #1a1f2b;
        --ex-muted: #6b7280;
    }

    .stApp { background: var(--ex-bg); }

    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }

    h1, h2, h3 { color: var(--ex-text) !important; font-weight: 700 !important; letter-spacing: -0.02em; }

    p, span, label, div { color: var(--ex-text); }

    .ex-hero {
        background: linear-gradient(135deg, rgba(79,70,229,0.06), rgba(8,145,178,0.05));
        border: 1px solid var(--ex-border);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(16,24,40,0.04);
    }
    .ex-hero h1 { font-size: 2rem; margin: 0 0 4px 0; color: var(--ex-text) !important; }
    .ex-hero p { color: var(--ex-muted); margin: 0; font-size: 0.95rem; }

    .ex-card {
        background: var(--ex-panel);
        border: 1px solid var(--ex-border);
        border-radius: 14px;
        padding: 20px 22px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px rgba(16,24,40,0.03);
    }

    .ex-metric {
        background: var(--ex-panel);
        border: 1px solid var(--ex-border);
        border-radius: 12px;
        padding: 16px 18px;
        text-align: left;
        box-shadow: 0 1px 2px rgba(16,24,40,0.03);
    }
    .ex-metric .label { color: var(--ex-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }
    .ex-metric .value { color: var(--ex-text); font-size: 1.6rem; font-weight: 700; margin-top: 2px; }

    .ex-badge { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.75rem; font-weight: 700; letter-spacing: 0.03em; }
    .ex-badge.pass { background: rgba(5,150,105,0.1); color: var(--ex-good); border: 1px solid rgba(5,150,105,0.25); }
    .ex-badge.warn { background: rgba(180,83,9,0.1); color: var(--ex-warn); border: 1px solid rgba(180,83,9,0.25); }
    .ex-badge.fail { background: rgba(220,38,38,0.1); color: var(--ex-bad); border: 1px solid rgba(220,38,38,0.25); }

    section[data-testid="stSidebar"] { background: var(--ex-panel); border-right: 1px solid var(--ex-border); }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    section[data-testid="stSidebar"] * { color: var(--ex-text) !important; }

    .stButton > button {
        background: linear-gradient(135deg, var(--ex-accent), #4338ca);
        color: white !important; border: none; border-radius: 10px; font-weight: 600;
        padding: 0.5rem 1.2rem; transition: all 0.15s ease;
        box-shadow: 0 1px 2px rgba(16,24,40,0.08);
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(79,70,229,0.25); }
    .stButton > button p { color: white !important; }

    div[data-testid="stDataFrame"] { border: 1px solid var(--ex-border); border-radius: 10px; overflow: hidden; }

    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        background: var(--ex-panel) !important; border: 1px solid var(--ex-border) !important; color: var(--ex-text) !important;
    }

    .ex-footer-note { color: var(--ex-muted); font-size: 0.8rem; text-align: center; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)


def api_get(path, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Backend not reachable: {e}\n\nStart it with: `uvicorn backend.main:app --reload`")
        return None


def api_post(path, json=None, params=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json, params=params, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {getattr(e.response, 'text', str(e))}")
        return None


def badge(status: str) -> str:
    cls = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}.get(status, "warn")
    return f'<span class="ex-badge {cls}">{status}</span>'


# ----------------------------------------------------------------- Sidebar
with st.sidebar:
    st.markdown("### ◆ ExplainX AI")
    st.caption("Local Explainable AI Platform")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Overview", "Train a Model", "Global Explanations", "Local Explanations (What-If)",
         "Bias & Fairness Audit", "Explanation Audit Log"],
        label_visibility="collapsed",
    )
    st.divider()
    models_resp = api_get("/models")
    model_names = [m["name"] for m in models_resp] if models_resp else []
    st.caption(f"{len(model_names)} model(s) registered locally")
    st.markdown(
        '<div class="ex-footer-note">Runs 100% on local infrastructure.<br>No external API calls at inference time.</div>',
        unsafe_allow_html=True,
    )

st.markdown("""
<div class="ex-hero">
    <h1>ExplainX AI</h1>
    <p>Production-ready explainability for your ML models — global & local interpretability, bias detection, fairness evaluation, and full audit trails. Everything runs on your machine.</p>
</div>
""", unsafe_allow_html=True)

# ================================================================== OVERVIEW
if page == "Overview":
    if not models_resp:
        st.info("No models registered yet. Head to **Train a Model** to get started.")
    else:
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f'<div class="ex-metric"><div class="label">Registered Models</div><div class="value">{len(models_resp)}</div></div>', unsafe_allow_html=True)
        with cols[1]:
            clf = sum(1 for m in models_resp if m["task_type"] == "classification")
            st.markdown(f'<div class="ex-metric"><div class="label">Classification Models</div><div class="value">{clf}</div></div>', unsafe_allow_html=True)
        with cols[2]:
            reg = sum(1 for m in models_resp if m["task_type"] == "regression")
            st.markdown(f'<div class="ex-metric"><div class="label">Regression Models</div><div class="value">{reg}</div></div>', unsafe_allow_html=True)
        with cols[3]:
            avg_acc = [m["accuracy"] for m in models_resp if m["accuracy"]]
            val = f"{(sum(avg_acc)/len(avg_acc))*100:.1f}%" if avg_acc else "—"
            st.markdown(f'<div class="ex-metric"><div class="label">Avg. Accuracy</div><div class="value">{val}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Registered Models")
        df = pd.DataFrame(models_resp)
        display_cols = ["name", "algorithm", "task_type", "n_features", "accuracy", "f1_score", "rmse", "r2_score", "created_at"]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

# ================================================================== TRAIN
elif page == "Train a Model":
    st.markdown("#### Train a New Model")
    st.caption("Models train locally using scikit-learn — nothing leaves your machine.")

    datasets_resp = api_get("/datasets")
    algos_resp = api_get("/algorithms")

    with st.form("train_form"):
        c1, c2 = st.columns(2)
        with c1:
            model_name = st.text_input("Model name", placeholder="e.g. credit_risk_v1")
            dataset_name = st.selectbox(
                "Dataset",
                [d["name"] for d in datasets_resp["datasets"]] if datasets_resp else [],
                format_func=lambda x: next((f'{d["name"]} — {d["description"]}' for d in datasets_resp["datasets"] if d["name"] == x), x) if datasets_resp else x,
            )
        with c2:
            algorithm = st.selectbox("Algorithm", algos_resp["algorithms"] if algos_resp else [])
            test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

        submitted = st.form_submit_button("Train Model", use_container_width=True)

    if submitted:
        if not model_name:
            st.warning("Please provide a model name.")
        else:
            with st.spinner("Training model locally..."):
                result = api_post("/train", json={
                    "model_name": model_name, "algorithm": algorithm,
                    "dataset_name": dataset_name, "test_size": test_size,
                })
            if result:
                st.success(result["message"])
                m = result["metrics"]
                cols = st.columns(4)
                metric_pairs = [("Accuracy", m["accuracy"]), ("F1 Score", m["f1_score"]),
                                 ("RMSE", m["rmse"]), ("R² Score", m["r2_score"])]
                for col, (label, val) in zip(cols, metric_pairs):
                    display = f"{val:.4f}" if val is not None else "—"
                    col.markdown(f'<div class="ex-metric"><div class="label">{label}</div><div class="value">{display}</div></div>', unsafe_allow_html=True)

# ================================================================== GLOBAL EXPLANATIONS
elif page == "Global Explanations":
    st.markdown("#### Global Feature Importance")
    st.caption("Mean absolute SHAP value per feature — how much each feature drives predictions across the dataset.")

    if not model_names:
        st.info("Train a model first.")
    else:
        selected = st.selectbox("Select model", model_names)
        if st.button("Compute Global Explanation"):
            with st.spinner("Running SHAP analysis..."):
                result = api_post("/explain/global", params={"model_name": selected})
            if result:
                df = pd.DataFrame(result["feature_importance"])
                fig = px.bar(
                    df.head(15).sort_values("importance"), x="importance", y="feature", orientation="h",
                    title="Feature Importance (Mean |SHAP value|)",
                    color="importance", color_continuous_scale=["#c7d2fe", "#4f46e5", "#0891b2"],
                )
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#1a1f2b", height=500, coloraxis_showscale=False,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True, hide_index=True)

# ================================================================== LOCAL EXPLANATIONS
elif page == "Local Explanations (What-If)":
    st.markdown("#### Local Explanation — Single Prediction")
    st.caption("Explore why the model made a specific prediction for one instance. Adjust values and see the explanation update.")

    if not model_names:
        st.info("Train a model first.")
    else:
        selected = st.selectbox("Select model", model_names)
        record = next((m for m in models_resp if m["name"] == selected), None)

        sample = api_get(f"/models/{selected}/sample", params={"n": 1})
        if record and sample:
            st.markdown("**Adjust feature values:**")
            cols = st.columns(3)
            input_values = {}
            for i, feat in enumerate(record["feature_names"]):
                default_val = float(sample[0][feat])
                with cols[i % 3]:
                    input_values[feat] = st.number_input(feat, value=default_val, format="%.4f", key=f"local_{feat}")

            method = st.radio("Explanation method", ["shap", "lime"], horizontal=True)

            if st.button("Explain This Prediction", use_container_width=True):
                with st.spinner("Generating local explanation..."):
                    result = api_post("/explain/local", json={"model_name": selected, "features": input_values}, params={"method": method})
                if result:
                    pred = result["prediction"]
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        pred_display = f"Class {pred['prediction']}" if pred["probability"] is not None else f"{pred['prediction']:.4f}"
                        conf_display = f"{pred['probability']*100:.1f}% confidence" if pred["probability"] is not None else "Regression output"
                        st.markdown(f'<div class="ex-card"><div class="label" style="color:var(--ex-muted)">Prediction</div><h2>{pred_display}</h2><p style="color:var(--ex-muted)">{conf_display}</p></div>', unsafe_allow_html=True)

                    with c2:
                        exp_df = pd.DataFrame(result["explanation"])
                        val_col = "shap_contribution" if "shap_contribution" in exp_df.columns else "contribution"
                        exp_df = exp_df.sort_values(val_col, key=abs, ascending=True)
                        colors = ["#dc2626" if v < 0 else "#059669" for v in exp_df[val_col]]
                        fig = go.Figure(go.Bar(
                            x=exp_df[val_col], y=exp_df["feature"], orientation="h",
                            marker_color=colors,
                        ))
                        fig.update_layout(
                            title=f"{method.upper()} Contribution per Feature",
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                            font_color="#1a1f2b", height=400, margin=dict(l=10, r=10, t=50, b=10),
                        )
                        st.plotly_chart(fig, use_container_width=True)

# ================================================================== FAIRNESS
elif page == "Bias & Fairness Audit":
    st.markdown("#### Bias Detection & Fairness Evaluation")
    st.caption("Evaluate model fairness across a protected attribute using demographic parity, disparate impact, equal opportunity, and predictive parity.")

    if not model_names:
        st.info("Train a model first.")
    else:
        selected = st.selectbox("Select model", model_names)
        record = next((m for m in models_resp if m["name"] == selected), None)

        if record:
            c1, c2, c3 = st.columns(3)
            with c1:
                protected_attr = st.selectbox("Protected attribute", record["feature_names"])
            with c2:
                privileged_value = st.number_input("Privileged group value", value=0.0)
            with c3:
                unprivileged_value = st.number_input("Unprivileged group value", value=1.0)

            if st.button("Run Fairness Audit", use_container_width=True):
                with st.spinner("Computing fairness metrics..."):
                    result = api_post("/fairness/evaluate", json={
                        "model_name": selected, "protected_attribute": protected_attr,
                        "privileged_value": privileged_value, "unprivileged_value": unprivileged_value,
                    })
                if result:
                    verdict = result["overall_verdict"]
                    st.markdown(f'### Overall Verdict: {badge(verdict)}', unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)

                    metrics = result["metrics"]
                    metric_cards = [
                        ("Demographic Parity Difference", "demographic_parity_difference"),
                        ("Disparate Impact Ratio", "disparate_impact_ratio"),
                        ("Equal Opportunity Difference", "equal_opportunity_difference"),
                        ("Predictive Parity Difference", "predictive_parity_difference"),
                    ]
                    cols = st.columns(2)
                    for i, (label, key) in enumerate(metric_cards):
                        data = metrics[key]
                        val = data["value"]
                        val_display = f"{val:.4f}" if val is not None else "N/A"
                        status_badge = badge(data["status"])
                        with cols[i % 2]:
                            st.markdown(f'''
                                <div class="ex-card">
                                    <div style="display:flex; justify-content:space-between; align-items:center;">
                                        <span style="color:var(--ex-muted); font-size:0.85rem;">{label}</span>
                                        {status_badge}
                                    </div>
                                    <div style="font-size:1.5rem; font-weight:700; margin-top:6px;">{val_display}</div>
                                </div>
                            ''', unsafe_allow_html=True)

                    st.markdown("#### Group Sizes & Per-Group Accuracy")
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.json(metrics["group_sizes"])
                    with gc2:
                        st.json(metrics["accuracy_by_group"])

# ================================================================== AUDIT LOG
elif page == "Explanation Audit Log":
    st.markdown("#### Explanation Audit Trail")
    st.caption("Every explanation ever generated is logged locally in SQLite for compliance and traceability.")

    if not model_names:
        st.info("Train a model first.")
    else:
        selected = st.selectbox("Select model", model_names)
        logs = api_get(f"/audit/explanations/{selected}")
        if logs:
            df = pd.DataFrame(logs)
            st.dataframe(df[["id", "type", "prediction", "created_at"]], use_container_width=True, hide_index=True)
        else:
            st.info("No explanations logged yet for this model.")

        st.markdown("#### Fairness Audit History")
        fairness_logs = api_get(f"/audit/fairness/{selected}")
        if fairness_logs:
            fdf = pd.DataFrame(fairness_logs)
            st.dataframe(fdf, use_container_width=True, hide_index=True)
        else:
            st.info("No fairness audits logged yet for this model.")
