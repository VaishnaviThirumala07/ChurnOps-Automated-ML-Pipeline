"""
Stage 1: Data Loading
Downloads the IBM Telco Customer Churn dataset and saves it to data/raw/.
Supports loading from a local path or a URL.
"""

import os
import sys
import yaml
import requests
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


# ── Load Params ────────────────────────────────────────────────────────────────
def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# ── Download Dataset ───────────────────────────────────────────────────────────
def download_dataset(url: str, save_path: str) -> None:
    """Download a CSV dataset from a URL."""
    logger.info(f"Downloading dataset from: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(response.content)
    logger.success(f"Dataset saved to: {save_path}")


# ── Validate Dataset ──────────────────────────────────────────────────────────
def validate_dataset(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the loaded dataset."""
    required_columns = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MonthlyCharges", "TotalCharges", "Churn"
    ]
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Churn distribution:\n{df['Churn'].value_counts(normalize=True).round(3)}")
    logger.info(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    logger.success("Dataset validation passed.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    params = load_params()
    raw_path = params["data"]["raw_data_path"]

    # IBM Telco Churn Dataset — hosted on GitHub (public mirror)
    DATA_URL = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
        "master/data/Telco-Customer-Churn.csv"
    )

    # If file already exists, skip download
    if Path(raw_path).exists():
        logger.info(f"Raw data already exists at {raw_path}. Skipping download.")
    else:
        download_dataset(DATA_URL, raw_path)

    # Load and validate
    df = pd.read_csv(raw_path)
    validate_dataset(df)

    logger.success(f"Data loading complete. {len(df)} records loaded.")
    return df


if __name__ == "__main__":
    logger.add("logs/data_load.log", rotation="1 MB")
    main()
