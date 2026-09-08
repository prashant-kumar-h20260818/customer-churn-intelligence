from __future__ import annotations

from pathlib import Path

from .data import load_dataset
from .modeling import global_feature_importance, save_bundle, score_customers, train_and_evaluate


def main() -> None:
    frame = load_dataset()
    bundle = train_and_evaluate(frame)
    save_bundle(bundle)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    score_customers(bundle.model, frame.drop(columns=["Churn"])).to_csv(
        "data/processed/scored_customers.csv", index=False
    )
    global_feature_importance(bundle.model).to_csv(
        "data/processed/feature_importance.csv", index=False
    )

    print(f"Best model: {bundle.best_model_name}")
    print(f"Threshold: {bundle.threshold:.2f}")
    for metric, value in bundle.metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
