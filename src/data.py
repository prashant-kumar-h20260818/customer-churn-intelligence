from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

DATA_URL = (
    "https://raw.githubusercontent.com/SaeidRostami/Customer_Churn/master/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
DEFAULT_DATA_PATH = Path("data/raw/telco_customer_churn.csv")


def download_dataset(path: Path = DEFAULT_DATA_PATH, url: str = DATA_URL) -> Path:
    """Download the IBM Telco churn sample from a public GitHub mirror."""
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path.write_bytes(response.content)
    return path


def load_dataset(path: Path = DEFAULT_DATA_PATH, download_if_missing: bool = True) -> pd.DataFrame:
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Dataset not found at {path}. Run `python -m src.data` to download it."
            )
        download_dataset(path)

    frame = pd.read_csv(path)
    if "Churn" not in frame.columns:
        raise ValueError("Expected target column 'Churn' in dataset.")
    return frame


if __name__ == "__main__":
    saved = download_dataset()
    print(f"Dataset saved to {saved}")
