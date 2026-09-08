import pandas as pd

from src.features import ChurnFeatureEngineer
from src.modeling import risk_tier


def example_customer():
    return pd.DataFrame(
        [{
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "No",
            "Dependents": "No",
            "tenure": 3,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 95.0,
            "TotalCharges": "285.0",
        }]
    )


def test_feature_engineering_creates_business_features():
    result = ChurnFeatureEngineer().fit_transform(example_customer())
    expected = {
        "TotalServices",
        "AvgMonthlySpend",
        "NewCustomer",
        "HighMonthlyCharges",
        "AutomaticPayment",
        "TenureGroup",
    }
    assert expected.issubset(result.columns)
    assert result.loc[0, "NewCustomer"] == 1


def test_risk_tiers():
    assert risk_tier(0.10) == "Low"
    assert risk_tier(0.40) == "Medium"
    assert risk_tier(0.60) == "High"
    assert risk_tier(0.90) == "Critical"
