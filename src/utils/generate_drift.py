"""
Synthetic Drift Generator
Creates a drifted version of the dataset to simulate real-world data degradation.
Used to test the Evidently drift detection pipeline.
"""

import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def generate_drifted_data(df: pd.DataFrame, drift_intensity: float = 0.3) -> pd.DataFrame:
    """
    Apply synthetic drift to numeric columns and flip some categorical values.
    drift_intensity: 0.0 = no drift, 1.0 = extreme drift
    """
    drifted = df.copy()
    logger.info(f"Applying drift with intensity: {drift_intensity}")

    # 1. Shift MonthlyCharges upward (simulate price hike)
    drifted["MonthlyCharges"] = drifted["MonthlyCharges"] * (1 + drift_intensity * 0.5)

    # 2. Reduce tenure (simulate newer customer base)
    drifted["tenure"] = drifted["tenure"] * (1 - drift_intensity * 0.4)
    drifted["tenure"] = drifted["tenure"].clip(lower=0)

    # 3. Shift TotalCharges accordingly
    drifted["TotalCharges"] = drifted["TotalCharges"] * (1 + drift_intensity * 0.3)

    # 4. Flip Contract type to shorter (month-to-month) for ~30% of rows
    if "Contract" in drifted.columns:
        mask = np.random.rand(len(drifted)) < drift_intensity * 0.5
        drifted.loc[mask, "Contract"] = "Month-to-month"

    # 5. Add Gaussian noise to numeric features
    for col in ["MonthlyCharges", "tenure", "TotalCharges"]:
        noise = np.random.normal(0, drifted[col].std() * drift_intensity * 0.1, len(drifted))
        drifted[col] = drifted[col] + noise

    logger.success("Drift applied successfully.")
    return drifted


def main():
    params = load_params()
    raw_path = params["data"]["raw_data_path"]
    drifted_path = "data/raw/churn_drifted.csv"

    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    drifted_df = generate_drifted_data(df, drift_intensity=0.4)
    drifted_df.to_csv(drifted_path, index=False)
    logger.success(f"Drifted dataset saved -> {drifted_path}")
    logger.info(f"Original MonthlyCharges mean: {df['MonthlyCharges'].mean():.2f}")
    logger.info(f"Drifted  MonthlyCharges mean: {drifted_df['MonthlyCharges'].mean():.2f}")


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/drift_gen.log", rotation="1 MB")
    main()
