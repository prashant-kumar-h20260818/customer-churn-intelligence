from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ChurnFeatureEngineer

RANDOM_STATE = 42


@dataclass
class TrainingBundle:
    model: Pipeline
    threshold: float
    best_model_name: str
    metrics: dict[str, Any]
    comparison: pd.DataFrame
    test_results: pd.DataFrame
    roc: dict[str, list[float]]
    pr: dict[str, list[float]]


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, Any]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=2,
        ),
    }


def build_pipeline(estimator: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", ChurnFeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("classifier", estimator),
        ]
    )


def _metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (probability >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
    }


def choose_threshold(y_true: pd.Series, probability: np.ndarray) -> float:
    """Choose an operating threshold using F2 to emphasize churn recall."""
    thresholds = np.arange(0.20, 0.81, 0.01)
    scores = [
        fbeta_score(y_true, probability >= threshold, beta=2, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def train_and_evaluate(
    frame: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = RANDOM_STATE,
) -> TrainingBundle:
    data = frame.copy().dropna(subset=["Churn"])
    y = data["Churn"].map({"Yes": 1, "No": 0})
    if y.isna().any():
        raise ValueError("Target 'Churn' must contain only Yes/No values.")
    X = data.drop(columns=["Churn"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    comparison_rows: list[dict[str, float | str]] = []
    model_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidate_models().items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        comparison_rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(scores.mean()),
                "cv_roc_auc_std": float(scores.std()),
            }
        )
        model_pipelines[name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "cv_roc_auc_mean", ascending=False
    ).reset_index(drop=True)
    best_model_name = str(comparison.iloc[0]["model"])
    selected = model_pipelines[best_model_name]

    oof_probability = cross_val_predict(
        clone(selected),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    threshold = choose_threshold(y_train, oof_probability)

    selected.fit(X_train, y_train)
    test_probability = selected.predict_proba(X_test)[:, 1]
    metrics = _metrics(y_test, test_probability, threshold)
    metrics["threshold"] = threshold
    metrics["test_customers"] = int(len(y_test))
    metrics["train_customers"] = int(len(y_train))
    metrics["dataset_customers"] = int(len(data))
    metrics["historical_churn_rate"] = float(y.mean())

    test_pred = (test_probability >= threshold).astype(int)
    test_results = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "probability": test_probability,
            "prediction": test_pred,
        },
        index=y_test.index,
    )

    fpr, tpr, _ = roc_curve(y_test, test_probability)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probability)
    metrics["confusion_matrix"] = confusion_matrix(y_test, test_pred).tolist()

    return TrainingBundle(
        model=selected,
        threshold=threshold,
        best_model_name=best_model_name,
        metrics=metrics,
        comparison=comparison,
        test_results=test_results,
        roc={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        pr={"precision": precision_curve.tolist(), "recall": recall_curve.tolist()},
    )


def risk_tier(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.50:
        return "Medium"
    if probability < 0.70:
        return "High"
    return "Critical"


def score_customers(model: Pipeline, frame: pd.DataFrame) -> pd.DataFrame:
    probability = model.predict_proba(frame)[:, 1]
    scored = frame.copy()
    scored["ChurnProbability"] = probability
    scored["RiskTier"] = [risk_tier(float(p)) for p in probability]
    monthly = pd.to_numeric(scored["MonthlyCharges"], errors="coerce").fillna(0)
    scored["AnnualCustomerValue"] = monthly * 12
    scored["RevenueAtRisk"] = scored["ChurnProbability"] * scored["AnnualCustomerValue"]
    scored["RetentionPriority"] = scored["RevenueAtRisk"]
    return scored.sort_values("RetentionPriority", ascending=False)


def transformed_feature_names(model: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def global_feature_importance(model: Pipeline) -> pd.DataFrame:
    estimator = model.named_steps["classifier"]
    names = transformed_feature_names(model)
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    result = pd.DataFrame({"feature": names, "importance": values})
    return result.sort_values("importance", ascending=False).reset_index(drop=True)


def local_shap_explanation(model: Pipeline, frame: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Return top local SHAP contributions for the positive churn class."""
    import shap

    engineered = model.named_steps["features"].transform(frame)
    transformed = model.named_steps["preprocessor"].transform(engineered)
    estimator = model.named_steps["classifier"]
    names = transformed_feature_names(model)

    try:
        explainer = shap.Explainer(estimator)
        values = explainer(transformed)
        shap_values = np.asarray(values.values)
        if shap_values.ndim == 3:
            class_index = 1 if shap_values.shape[2] > 1 else 0
            row_values = shap_values[0, :, class_index]
        elif shap_values.ndim == 2:
            row_values = shap_values[0]
        else:
            row_values = shap_values.reshape(-1)[: len(names)]
    except Exception:
        if hasattr(estimator, "coef_"):
            row_values = np.asarray(estimator.coef_[0]) * transformed[0]
        elif hasattr(estimator, "feature_importances_"):
            centered = transformed[0] - np.nanmean(transformed, axis=0)
            row_values = np.asarray(estimator.feature_importances_) * centered
        else:
            raise

    explanation = pd.DataFrame(
        {
            "feature": names,
            "contribution": row_values,
            "absolute_contribution": np.abs(row_values),
        }
    )
    return explanation.sort_values("absolute_contribution", ascending=False).head(top_n)


def save_bundle(bundle: TrainingBundle, directory: Path = Path("models")) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.model, directory / "churn_pipeline.joblib")
    payload = {
        "best_model": bundle.best_model_name,
        "threshold": bundle.threshold,
        "metrics": bundle.metrics,
        "comparison": bundle.comparison.to_dict(orient="records"),
        "roc": bundle.roc,
        "pr": bundle.pr,
    }
    (directory / "model_metrics.json").write_text(json.dumps(payload, indent=2))
