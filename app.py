from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data import load_dataset
from src.features import REQUIRED_INPUT_COLUMNS
from src.modeling import (
    global_feature_importance,
    local_shap_explanation,
    risk_tier,
    score_customers,
    train_and_evaluate,
)

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📉",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource(show_spinner="Training and validating churn models...")
def get_bundle():
    return train_and_evaluate(get_data())


def money(value: float) -> str:
    return f"${value:,.0f}"


def probability_gauge(probability: float) -> go.Figure:
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Predicted Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30]},
                    {"range": [30, 50]},
                    {"range": [50, 70]},
                    {"range": [70, 100]},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.8,
                    "value": probability * 100,
                },
            },
        )
    )


def clean_feature_label(name: str) -> str:
    label = name.replace("num__", "").replace("cat__", "")
    return label.replace("_", " ")


def input_form() -> pd.DataFrame:
    left, middle, right = st.columns(3)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with middle:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    with right:
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly = st.number_input("Monthly Charges", 0.0, 200.0, 75.0, 0.5)
        total_default = max(monthly * max(tenure, 1), monthly)
        total = st.number_input("Total Charges", 0.0, 10000.0, float(total_default), 1.0)

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone,
                "MultipleLines": multiple,
                "InternetService": internet,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device,
                "TechSupport": tech,
                "StreamingTV": tv,
                "StreamingMovies": movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly,
                "TotalCharges": total,
            }
        ]
    )


def overview_tab(frame: pd.DataFrame, bundle) -> None:
    churn_rate = frame["Churn"].eq("Yes").mean()
    scored = score_customers(bundle.model, frame.drop(columns=["Churn"]))
    high_risk = scored["RiskTier"].isin(["High", "Critical"]).sum()
    revenue_at_risk = scored["RevenueAtRisk"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{len(frame):,}")
    c2.metric("Historical Churn", f"{churn_rate:.1%}")
    c3.metric("High/Critical Risk", f"{high_risk:,}")
    c4.metric("Modeled Annual Revenue at Risk", money(revenue_at_risk))

    st.caption(
        "Revenue at risk is a prioritization estimate: churn probability × monthly charges × 12. "
        "It is not a realized financial-loss claim."
    )

    col1, col2 = st.columns(2)
    with col1:
        contract = (
            frame.assign(ChurnFlag=frame["Churn"].eq("Yes").astype(int))
            .groupby("Contract", as_index=False)["ChurnFlag"]
            .mean()
        )
        fig = px.bar(contract, x="Contract", y="ChurnFlag", title="Churn Rate by Contract")
        fig.update_yaxes(tickformat=".0%", title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        internet = (
            frame.assign(ChurnFlag=frame["Churn"].eq("Yes").astype(int))
            .groupby("InternetService", as_index=False)["ChurnFlag"]
            .mean()
        )
        fig = px.bar(internet, x="InternetService", y="ChurnFlag", title="Churn Rate by Internet Service")
        fig.update_yaxes(tickformat=".0%", title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        tenure_plot = frame.copy()
        tenure_plot["TenureGroup"] = pd.cut(
            tenure_plot["tenure"],
            [-0.1, 6, 12, 24, 48, np.inf],
            labels=["0-6", "7-12", "13-24", "25-48", "49+"],
        )
        tenure_rate = (
            tenure_plot.assign(ChurnFlag=tenure_plot["Churn"].eq("Yes").astype(int))
            .groupby("TenureGroup", observed=False, as_index=False)["ChurnFlag"]
            .mean()
        )
        fig = px.line(tenure_rate, x="TenureGroup", y="ChurnFlag", markers=True, title="Churn by Tenure")
        fig.update_yaxes(tickformat=".0%", title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        payment = (
            frame.assign(ChurnFlag=frame["Churn"].eq("Yes").astype(int))
            .groupby("PaymentMethod", as_index=False)["ChurnFlag"]
            .mean()
            .sort_values("ChurnFlag", ascending=False)
        )
        fig = px.bar(payment, x="ChurnFlag", y="PaymentMethod", orientation="h", title="Churn by Payment Method")
        fig.update_xaxes(tickformat=".0%", title="Churn rate")
        st.plotly_chart(fig, use_container_width=True)


def predictor_tab(bundle) -> None:
    st.subheader("Individual Customer Risk Scoring")
    customer = input_form()
    if st.button("Predict Churn", type="primary", use_container_width=True):
        probability = float(bundle.model.predict_proba(customer)[0, 1])
        tier = risk_tier(probability)
        annual_value = float(customer.loc[0, "MonthlyCharges"] * 12)
        revenue_at_risk = probability * annual_value

        gauge_col, metric_col = st.columns([2, 1])
        with gauge_col:
            st.plotly_chart(probability_gauge(probability), use_container_width=True)
        with metric_col:
            st.metric("Risk Tier", tier)
            st.metric("Annual Customer Value", money(annual_value))
            st.metric("Revenue at Risk", money(revenue_at_risk))
            st.metric("Decision Threshold", f"{bundle.threshold:.0%}")

        st.subheader("Why this customer is at risk")
        try:
            explanation = local_shap_explanation(bundle.model, customer)
            explanation["feature"] = explanation["feature"].map(clean_feature_label)
            explanation["direction"] = np.where(
                explanation["contribution"] >= 0, "Increases churn risk", "Reduces churn risk"
            )
            st.dataframe(
                explanation[["feature", "contribution", "direction"]],
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.info(f"Local SHAP explanation could not be rendered in this environment: {exc}")


def batch_tab(bundle) -> None:
    st.subheader("Batch Customer Scoring")
    st.write("Upload customer records using the same input schema as the Telco dataset (without the Churn column).")

    template = pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        "churn_scoring_template.csv",
        "text/csv",
    )

    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in batch.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return
        scored = score_customers(bundle.model, batch)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows Scored", f"{len(scored):,}")
        c2.metric("High/Critical Risk", f"{scored['RiskTier'].isin(['High', 'Critical']).sum():,}")
        c3.metric("Revenue at Risk", money(scored["RevenueAtRisk"].sum()))
        st.dataframe(scored.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "Download Prioritized Retention List",
            scored.to_csv(index=False).encode("utf-8"),
            "retention_priority_list.csv",
            "text/csv",
            type="primary",
        )


def model_tab(bundle) -> None:
    st.subheader("Model Validation & Explainability")
    metric_cols = st.columns(6)
    displayed = ["roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"]
    labels = ["ROC-AUC", "PR-AUC", "Recall", "Precision", "F1", "Accuracy"]
    for col, key, label in zip(metric_cols, displayed, labels):
        col.metric(label, f"{bundle.metrics[key]:.3f}")

    st.write(f"**Selected model:** {bundle.best_model_name}  |  **Operating threshold:** {bundle.threshold:.2f}")
    comparison = bundle.comparison.copy()
    comparison["cv_roc_auc_mean"] = comparison["cv_roc_auc_mean"].round(4)
    comparison["cv_roc_auc_std"] = comparison["cv_roc_auc_std"].round(4)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bundle.roc["fpr"], y=bundle.roc["tpr"], mode="lines", name="Model"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline"))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bundle.pr["recall"],
                y=bundle.pr["precision"],
                mode="lines",
                name="Precision-Recall",
            )
        )
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

    importance = global_feature_importance(bundle.model).head(15).copy()
    importance["feature"] = importance["feature"].map(clean_feature_label)
    fig = px.bar(
        importance.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        title="Top Global Model Features",
    )
    st.plotly_chart(fig, use_container_width=True)

    cm = np.asarray(bundle.metrics["confusion_matrix"])
    fig = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual"},
        x=["No churn", "Churn"],
        y=["No churn", "Churn"],
        title="Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)


st.title("📉 Customer Churn Intelligence & Retention Prioritization")
st.write(
    "Predict churn probability, understand churn drivers, quantify modeled revenue at risk, "
    "and prioritize retention actions from one application."
)

try:
    data = get_data()
    bundle = get_bundle()
except Exception as error:
    st.error(
        "The project could not load/train the model. Confirm internet access for the first dataset download "
        "or place the IBM Telco CSV at data/raw/telco_customer_churn.csv."
    )
    st.exception(error)
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["Executive Dashboard", "Customer Predictor", "Batch Scoring", "Model Insights"]
)
with tab1:
    overview_tab(data, bundle)
with tab2:
    predictor_tab(bundle)
with tab3:
    batch_tab(bundle)
with tab4:
    model_tab(bundle)
