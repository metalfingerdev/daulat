"""Streamlit dashboard for explainable loan approval predictions.

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib

# Non-GUI backend: Streamlit runs this script off the main thread, and GUI
# backends (TkAgg, the Windows default) warn and hang there.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from streamlit import runtime

if __name__ == "__main__" and not runtime.exists():
    # Launched with plain `python app.py`: re-exec under `streamlit run`.
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from train_model import ARTIFACTS_PATH, MODEL_PATH, preprocess  # noqa: E402

FEATURE_LABELS = {
    "age": "Age (years)",
    "annual_income": "Annual income ($)",
    "employment_years": "Years employed",
    "loan_amount": "Loan amount ($)",
    "loan_term_months": "Loan term (months)",
    "credit_score": "Credit score",
    "debt_to_income": "Debt-to-income ratio",
    "num_credit_lines": "Open credit lines",
    "past_defaults": "Past defaults",
    "home_ownership": "Home ownership",
    "loan_purpose": "Loan purpose",
}

st.set_page_config(page_title="XAI Credit Risk Assessment", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or not ARTIFACTS_PATH.exists():
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(ARTIFACTS_PATH)


@st.cache_resource
def build_explainer(_artifacts):
    feature_names = _artifacts["feature_names"]
    categorical_names = {
        feature_names.index(col): values for col, values in _artifacts["categories"].items()
    }
    return LimeTabularExplainer(
        training_data=_artifacts["lime_training_data"],
        feature_names=[FEATURE_LABELS.get(f, f) for f in feature_names],
        class_names=_artifacts["class_names"],
        categorical_features=_artifacts["categorical_features"],
        categorical_names=categorical_names,
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )


def applicant_form(artifacts) -> pd.DataFrame:
    stats = artifacts["numeric_stats"]
    st.sidebar.header("Applicant details")

    def number(col, step=None, fmt=None):
        s = stats[col]
        if s["is_int"]:
            return st.sidebar.number_input(
                FEATURE_LABELS[col], int(s["min"]), int(s["max"]), int(s["median"]), step=step or 1
            )
        return st.sidebar.number_input(
            FEATURE_LABELS[col], float(s["min"]), float(s["max"]), float(s["median"]), step=step, format=fmt
        )

    values = {}
    st.sidebar.subheader("Personal")
    values["age"] = number("age")
    values["employment_years"] = number("employment_years", step=0.5, fmt="%.1f")
    values["home_ownership"] = st.sidebar.selectbox(
        FEATURE_LABELS["home_ownership"], artifacts["categories"]["home_ownership"]
    )

    st.sidebar.subheader("Financial")
    values["annual_income"] = number("annual_income", step=1000.0, fmt="%.0f")
    values["credit_score"] = st.sidebar.slider(FEATURE_LABELS["credit_score"], 300, 850, int(stats["credit_score"]["median"]))
    values["debt_to_income"] = st.sidebar.slider(
        FEATURE_LABELS["debt_to_income"], 0.0, 0.95, round(stats["debt_to_income"]["median"], 2), step=0.01
    )
    values["num_credit_lines"] = number("num_credit_lines")
    values["past_defaults"] = number("past_defaults")

    st.sidebar.subheader("Loan request")
    values["loan_amount"] = number("loan_amount", step=500.0, fmt="%.0f")
    values["loan_term_months"] = st.sidebar.selectbox(FEATURE_LABELS["loan_term_months"], [12, 24, 36, 48, 60], index=2)
    values["loan_purpose"] = st.sidebar.selectbox(
        FEATURE_LABELS["loan_purpose"],
        artifacts["categories"]["loan_purpose"],
        format_func=lambda v: v.replace("_", " ").title(),
    )

    return pd.DataFrame([values])[artifacts["feature_names"]]


def main():
    st.title("🏦 Explainable Credit Risk Assessment")
    st.caption("XGBoost loan approval prediction with LIME local explanations")

    model, artifacts = load_model()
    if model is None:
        st.error(
            "No trained model found. Run `python src/generate_data.py` and then "
            "`python src/train_model.py` first."
        )
        st.stop()

    raw = applicant_form(artifacts)
    encoded = preprocess(raw, artifacts["categories"])
    feature_names = artifacts["feature_names"]

    def predict_fn(x):
        return model.predict_proba(pd.DataFrame(x, columns=feature_names))

    proba_approved = float(predict_fn(encoded.to_numpy(dtype=float))[0, 1])
    approved = proba_approved >= 0.5

    col_decision, col_metrics = st.columns([2, 1])
    with col_decision:
        if approved:
            st.success("## ✅ APPROVED")
        else:
            st.error("## ❌ REJECTED")
        st.progress(proba_approved, text=f"Approval probability: {proba_approved:.1%}")
    with col_metrics:
        m = artifacts["metrics"]
        st.metric("Model accuracy (test)", f"{m['accuracy']:.1%}")
        st.metric("Model ROC AUC (test)", f"{m['roc_auc']:.3f}")

    st.divider()
    st.subheader("Why this decision? (LIME local explanation)")

    num_features = st.slider("Number of features to show", 3, len(feature_names), 8)
    explainer = build_explainer(artifacts)
    with st.spinner("Generating explanation..."):
        exp = explainer.explain_instance(
            encoded.to_numpy(dtype=float)[0],
            predict_fn,
            num_features=num_features,
            labels=(1,),
            num_samples=5000,
        )

    col_chart, col_table = st.columns([3, 2])
    with col_chart:
        fig = exp.as_pyplot_figure(label=1)
        fig.set_size_inches(8, max(3, 0.5 * num_features))
        ax = fig.axes[0]
        ax.set_title("Feature contributions toward approval")
        ax.set_xlabel("← pushes toward Rejected   |   pushes toward Approved →")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with col_table:
        contrib = pd.DataFrame(exp.as_list(label=1), columns=["Condition", "Weight"])
        contrib["Effect"] = contrib["Weight"].map(lambda w: "↑ Approve" if w > 0 else "↓ Reject")
        st.dataframe(
            contrib.style.format({"Weight": "{:+.3f}"}),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Positive weights increase the approval probability for this applicant; "
            "negative weights decrease it. Weights come from a local linear surrogate "
            "fitted around this applicant by LIME."
        )

    with st.expander("Full interactive LIME report"):
        st.iframe(exp.as_html(labels=(1,)), height=450)

    with st.expander("Applicant input"):
        st.dataframe(raw, hide_index=True, width="stretch")


if __name__ == "__main__":
    main()
