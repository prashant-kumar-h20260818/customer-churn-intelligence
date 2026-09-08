from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

SERVICE_COLUMNS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

REQUIRED_INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "TotalServices",
    "AvgMonthlySpend",
    "NewCustomer",
    "HighMonthlyCharges",
    "AutomaticPayment",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
]


def validate_columns(frame: pd.DataFrame, required: Iterable[str] = REQUIRED_INPUT_COLUMNS) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create business-oriented churn features inside a sklearn Pipeline."""

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        frame = X.copy()
        validate_columns(frame)

        if "customerID" in frame.columns:
            frame = frame.drop(columns=["customerID"])

        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
        frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce")
        frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
        frame["SeniorCitizen"] = pd.to_numeric(frame["SeniorCitizen"], errors="coerce")

        service_flags = [frame[col].eq("Yes").astype(int) for col in SERVICE_COLUMNS]
        frame["TotalServices"] = np.sum(service_flags, axis=0)

        tenure_safe = frame["tenure"].replace(0, 1)
        frame["AvgMonthlySpend"] = frame["TotalCharges"] / tenure_safe
        frame["NewCustomer"] = (frame["tenure"] <= 6).astype(int)

        median_charge = frame["MonthlyCharges"].median()
        frame["HighMonthlyCharges"] = (frame["MonthlyCharges"] > median_charge).astype(int)
        frame["AutomaticPayment"] = frame["PaymentMethod"].str.contains(
            "automatic", case=False, na=False
        ).astype(int)

        frame["TenureGroup"] = pd.cut(
            frame["tenure"],
            bins=[-0.1, 6, 12, 24, 48, np.inf],
            labels=["0-6", "7-12", "13-24", "25-48", "49+"],
        ).astype(str)

        return frame
