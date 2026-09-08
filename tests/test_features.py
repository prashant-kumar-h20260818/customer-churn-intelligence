import pandas as pd

from src.features import ChurnFeatureEngineer
from src.modeling import risk_tier


def example_customer(monthly_charges=95.0):
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
            "MonthlyCharges": monthly_charges,
            "TotalCharges": str(monthly_charges * 3),
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


def test_high_charge_threshold_is_learned_during_fit():
    training = pd.concat([example_customer(50.0), example_customer(100.0)], ignore_index=True)
    engineer = ChurnFeatureEngineer().fit(training)
    scored = engineer.transform(example_customer(95.0))
    assert engineer.monthly_charge_median_ == 75.0
    assert scored.loc[0, "HighMonthlyCharges"] == 1


def test_risk_tiers():
    assert risk_tier(0.10) == "Low"
    assert risk_tier(0.40) == "Medium"
    assert risk_tier(0.60) == "High"
    assert risk_tier(0.90) == "Critical"
