"""
F1 Pit-Window Decision Support — Streamlit app
================================================

Restructured from the prior repo's five loosely-related tabs
(``streamlit_app_enhanced.py``) into three views organized around a decision
workflow, per the re-scoping plan in ``docs/``:

1. **Race State** — current feature state → pit-window probability,
   phrased as "investigate," not "pit now" (this is a decision-support
   signal, not a command).
2. **Decision Policy** — the threshold/precision-recall tradeoff as an
   explicit, named policy choice, not a bare slider.
3. **Model Review** — calibration, feature importance, and this project's
   own stated limitations, in the app itself rather than only in a
   markdown file nobody opens.

Known limitation, stated here rather than hidden: this app scores
hand-entered/slider feature values, not a browsable table of real historical
laps by driver/session — that requires the data/repository.py snapshot
layer to be backed by a live, ingested database with lap identifiers
preserved end-to-end, which is not wired up yet (see MIGRATION_MAP.md,
Phase 3). Read every prediction in this app as "what the model would say for
a lap with these characteristics," not "what it said about a specific real
lap."
"""

import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import streamlit as st

from f1_pit_window.modeling.calibration import brier_score, reliability_curve
from f1_pit_window.modeling.evaluate import threshold_sweep
from f1_pit_window.modeling.inference import ModelArtifactValidationError, validate_model_artifacts
from f1_pit_window.modeling.train import DEFAULT_FEATURE_COLS

st.set_page_config(page_title="F1 Pit-Window Decision Support", page_icon="🏎️", layout="wide")

FEATURE_RANGES = {
    "degradation_rate": (-0.05, 0.20),
    "stint_age_squared": (0, 3600),
    "race_progress": (0.0, 1.0),
    "pace_delta": (-3.0, 3.0),
    "gap_closing_rate": (-2.0, 2.0),
    "relative_pace_delta": (-3.0, 3.0),
}
FEATURE_DESCRIPTIONS = {
    "degradation_rate": "OLS slope of lap time vs. stint age (s/lap)",
    "stint_age_squared": "Tyre age squared — non-linear degradation proxy (laps²)",
    "race_progress": "Current lap / total laps (0 = start, 1 = finish)",
    "pace_delta": "Lap time minus driver's own 5-lap rolling median (s)",
    "gap_closing_rate": "Avg. per-lap change in gap to leader over 3 laps; + = closing (s/lap)",
    "relative_pace_delta": "Lap time minus the field's median lap time at this lap (s)",
}


@st.cache_resource
def load_model_bundle():
    """
    Load trained model + scaler + saved metrics + held-out test arrays, and
    gate them through validate_model_artifacts() before anything else in
    this app touches them. A failure here means the app refuses to render,
    not that it renders with corrupted data.
    """
    with open("models/xgboost_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/metrics.pkl", "rb") as f:
        saved_metrics = pickle.load(f)

    X_test = np.load("models/X_test_scaled.npy")
    y_test = np.load("models/y_test.npy")
    y_proba = model.predict_proba(X_test)[:, 1]

    validate_model_artifacts(X_test, y_test, y_proba, saved_metrics)

    return {
        "model": model, "scaler": scaler, "saved_metrics": saved_metrics,
        "X_test": X_test, "y_test": y_test, "y_proba": y_proba,
        "feature_cols": saved_metrics.get("feature_cols", list(DEFAULT_FEATURE_COLS)),
        "threshold": saved_metrics.get("threshold", 0.60),
    }


try:
    bundle = load_model_bundle()
except ModelArtifactValidationError as exc:
    st.error(f"⚠️ Model artifacts failed validation — refusing to render: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Error loading model artifacts: {exc}")
    st.stop()


def view_race_state():
    """Render the Race State view: feature sliders → a live pit-window probability and action label."""
    st.header("🏁 Race State")
    st.caption(
        "Estimate whether a driver is entering a near-term pit window, given a lap's feature state. "
        "This is an analyst-assistance signal, not an autonomous strategy call."
    )

    feature_cols = bundle["feature_cols"]
    cols = st.columns(2)
    values = {}
    for i, feature in enumerate(feature_cols):
        lo, hi = FEATURE_RANGES.get(feature, (-1.0, 1.0))
        values[feature] = cols[i % 2].slider(
            feature, min_value=float(lo), max_value=float(hi), value=float((lo + hi) / 2),
            step=0.01, help=FEATURE_DESCRIPTIONS.get(feature, ""),
        )

    X = pd.DataFrame([values])[feature_cols].values
    X_scaled = bundle["scaler"].transform(X)
    proba = bundle["model"].predict_proba(X_scaled)[0, 1]
    threshold = bundle["threshold"]

    st.markdown("---")
    if proba < threshold * 0.7:
        label, color = "Unlikely near-term", "#2E7D32"
    elif proba < threshold:
        label, color = "Monitor", "#F9A825"
    else:
        label, color = "Investigate pit window", "#C62828"

    c1, c2 = st.columns([1, 2])
    c1.markdown(
        f'<div style="background:{color};padding:24px;text-align:center;color:white;border-radius:6px;">'
        f'<div style="font-size:40px;font-weight:800;">{proba:.0%}</div>'
        f'<div style="font-size:13px;letter-spacing:1px;margin-top:6px;">{label.upper()}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"**{label}** — pit-window probability **{proba:.1%}** against policy threshold τ={threshold:.2f}. "
        "Interpret alongside tactical context this model does not observe (traffic, fuel, opponent strategy) "
        "— see the Model Review tab for known limitations."
    )


def view_decision_policy():
    """Render the Decision Policy view: the threshold / precision-recall tradeoff as a named policy choice."""
    st.header("⚙️ Decision Policy")
    st.caption("The threshold is a policy choice, not a model output — pick it based on the cost of each error type.")

    sweep = threshold_sweep(bundle["y_test"], bundle["y_proba"], np.arange(0.1, 1.0, 0.05))
    col1, col2 = st.columns([3, 1])
    with col2:
        selected = st.slider("Threshold τ", 0.0, 1.0, float(bundle["threshold"]), 0.05)

    fig = go.Figure()
    for metric, color in [("Precision", "#1F4E79"), ("Recall", "#70AD47"), ("F1", "#C62828")]:
        fig.add_trace(go.Scatter(
            x=sweep["Threshold"], y=sweep[metric], mode="lines+markers", name=metric,
            line=dict(color=color, width=3),
        ))
    fig.add_vline(x=selected, line_dash="dash", annotation_text=f"τ={selected:.2f}")
    fig.update_layout(height=440, xaxis_title="Threshold", yaxis_title="Score", template="plotly_white")
    with col1:
        st.plotly_chart(fig, use_container_width=True)

    nearest = sweep.iloc[(sweep["Threshold"] - selected).abs().idxmin()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Precision", f"{nearest['Precision']:.1%}")
    m2.metric("Recall", f"{nearest['Recall']:.1%}")
    m3.metric("F1", f"{nearest['F1']:.3f}")

    st.markdown("---")
    st.subheader("Named decision styles")
    st.table(pd.DataFrame([
        {
            "Decision style": "Coverage-first", "Threshold behavior": "Lower (~0.4)",
            "Cost assumption": "Missing a real pit window costs more than a false alarm",
            "Intended use": "Flag most plausible pit windows for review",
        },
        {
            "Decision style": "Balanced", "Threshold behavior": "~0.5–0.6",
            "Cost assumption": "False positives and false negatives roughly equally costly",
            "Intended use": "General strategy review",
        },
        {
            "Decision style": "Intervention-first", "Threshold behavior": "Higher (~0.7+)",
            "Cost assumption": "A false alarm costs analyst attention that's expensive to spend",
            "Intended use": "Only surface strong signals",
        },
    ]))
    st.caption(
        f"This project's shipped threshold (τ={bundle['threshold']:.2f}) is a Balanced-to-Intervention-first "
        "policy choice, not a property the model discovered — see docs/decisions/ for the reasoning."
    )


def view_model_review():
    """Render the Model Review view: held-out metrics, calibration reliability curve, and feature importance."""
    st.header("🔬 Model Review")

    sm = bundle["saved_metrics"]
    st.subheader("Held-out evaluation")
    st.dataframe(pd.DataFrame([{
        "ROC-AUC": f"{sm['roc_auc']:.4f}", "F1": f"{sm['f1']:.4f}", "Recall": f"{sm['recall']:.4f}",
        "Precision": f"{sm['precision']:.4f}", "Threshold": f"{sm['threshold']:.2f}",
        "Train laps": f"{sm['train_size']:,}", "Test laps": f"{sm['test_size']:,}",
    }]), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Calibration")
    score = brier_score(bundle["y_test"], bundle["y_proba"])
    st.metric(
        "Brier score", f"{score:.4f}",
        help="0 = perfect; lower is better. Compare against this target's base-rate baseline, not 0.25.",
    )

    curve = reliability_curve(bundle["y_test"], bundle["y_proba"], n_bins=10)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=curve["mean_predicted"], y=curve["observed_rate"], mode="markers+lines", name="Model",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated", line=dict(dash="dash", color="gray"),
    ))
    fig.update_layout(
        height=400, xaxis_title="Mean predicted probability", yaxis_title="Observed positive rate",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Feature importance")
    importance = bundle["model"].feature_importances_
    feat_df = pd.DataFrame(
        {"Feature": bundle["feature_cols"], "Importance": importance}
    ).sort_values("Importance", ascending=False)
    fig2 = go.Figure(go.Bar(x=feat_df["Importance"], y=feat_df["Feature"], orientation="h"))
    fig2.update_layout(height=320, template="plotly_white", margin=dict(l=160))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Known limitations")
    st.markdown("""
- **Not a race-strategy optimizer.** This model estimates near-term pit-window likelihood from
  tire degradation and race state. It does not jointly optimize tire choice, fuel, traffic, or
  competitor response.
- **No tactical opponent signal.** `opponent_pit_window_signal` (whether nearby competitors have
  pitted or are likely to) is not implemented — see `docs/` for why it's a bigger lift than the
  other strategy-context features, not a missing afternoon of work.
- **Precision at the shipped threshold is real, not hidden.** Many false positives reflect a
  driver staying out for tactical reasons (undercut/overcut) that this feature set doesn't observe.
- **Trained on clean, dry-race laps only.** Safety-car, VSC, standing-start, and wet-weather laps
  were excluded — this model's behavior under those conditions is unvalidated, not just untested.
    """)


tab1, tab2, tab3 = st.tabs(["🏁 Race State", "⚙️ Decision Policy", "🔬 Model Review"])
with tab1:
    view_race_state()
with tab2:
    view_decision_policy()
with tab3:
    view_model_review()
