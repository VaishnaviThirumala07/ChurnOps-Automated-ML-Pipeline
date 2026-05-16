"""
Stage 2: Data Preprocessing
Cleans, encodes, and scales the raw churn dataset.
Produces train/test parquet files and a preprocessing pipeline artifact.
"""

import os
import sys
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ── Load Params ────────────────────────────────────────────────────────────────
def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# ── Fix Known Data Issues ─────────────────────────────────────────────────────
def fix_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle known data quality issues in the IBM Telco dataset."""
    logger.info("Fixing raw data issues...")

    # TotalCharges has spaces instead of NaN for new customers
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill NaN TotalCharges with MonthlyCharges (new customers, tenure=0)
    df["TotalCharges"].fillna(df["MonthlyCharges"], inplace=True)

    # Encode binary target: Yes -> 1, No -> 0
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    logger.info(f"Fixed data shape: {df.shape}")
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional business-relevant features."""
    logger.info("Engineering new features...")

    # Average monthly spend relative to tenure
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # High-value customer flag
    df["IsHighValue"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)

    # Number of services subscribed
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
        axis=1
    )

    # Advanced Interactions
    df["HasFiberOptic"] = (df["InternetService"] == "Fiber optic").astype(int)

    # 1. Lifetime Value (LTV) Projection
    df["LTV_Estimate"] = df["tenure"] * df["MonthlyCharges"]

    # 2. Bundle Intensity (Services / Cost)
    df["BundleValue"] = df["NumServices"] / (df["MonthlyCharges"] + 1)

    # 3. Security Bundle (High correlation with retention)
    extra_svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["SecurityBundleCount"] = df[extra_svc_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )
    
    # Tenure binning
    df["TenureGroup"] = pd.cut(
        df["tenure"], 
        bins=[0, 12, 24, 48, 72, 100], 
        labels=["New", "Junior", "Mid", "Senior", "Veteran"],
        include_lowest=True
    ).astype(str)

    logger.success("Feature engineering complete.")
    return df


# ── Build Preprocessor ────────────────────────────────────────────────────────
def build_preprocessor(params: dict) -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for numeric + categorical features."""
    cat_cols = params["preprocessing"]["categorical_columns"]
    num_cols = params["preprocessing"]["numerical_columns"] + [
        "AvgMonthlySpend", "IsHighValue", "NumServices", "LTV_Estimate", "BundleValue", "SecurityBundleCount"
    ]

    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    return preprocessor


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    params = load_params()
    raw_path   = params["data"]["raw_data_path"]
    train_path = params["data"]["train_path"]
    test_path  = params["data"]["test_path"]
    ref_path   = params["data"]["reference_path"]
    target_col = params["data"]["target_column"]
    test_size  = params["data"]["test_size"]
    seed       = params["base"]["random_seed"]
    drop_cols  = params["preprocessing"]["drop_columns"]

    # ── Load Raw Data ──────────────────────────────────────────────────────────
    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # ── Fix Data Issues ────────────────────────────────────────────────────────
    df = fix_raw_data(df)

    # ── Drop Non-Feature Columns ───────────────────────────────────────────────
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # ── Feature Engineering ────────────────────────────────────────────────────
    df = engineer_features(df)

    # ── Split Features & Target ────────────────────────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ── Train/Test Split ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    logger.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # ── Fit Preprocessor ───────────────────────────────────────────────────────
    preprocessor = build_preprocessor(params)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed  = preprocessor.transform(X_test)

    # Get output feature names
    cat_feature_names = preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(
        params["preprocessing"]["categorical_columns"]
    ).tolist()
    num_feature_names = params["preprocessing"]["numerical_columns"] + [
        "AvgMonthlySpend", "IsHighValue", "NumServices", "LTV_Estimate", "BundleValue", "SecurityBundleCount"
    ]
    all_feature_names = num_feature_names + cat_feature_names

    # ── Convert to DataFrames ──────────────────────────────────────────────────
    train_df = pd.DataFrame(X_train_processed, columns=all_feature_names)
    train_df[target_col] = y_train.values

    test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
    test_df[target_col] = y_test.values

    # ── Save Outputs ───────────────────────────────────────────────────────────
    Path(train_path).parent.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    logger.success(f"Saved train data -> {train_path}")
    logger.success(f"Saved test data  -> {test_path}")

    # Save reference dataset (first 1000 train rows, unprocessed, for Evidently)
    X_train.head(1000).assign(Churn=y_train.values[:1000]).to_parquet(ref_path, index=False)
    logger.success(f"Saved reference data -> {ref_path}")

    # ── Save Preprocessor Artifact ─────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.success("Saved preprocessor -> models/preprocessor.pkl")

    logger.info(f"Preprocessing complete. Feature count: {len(all_feature_names)}")


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/preprocess.log", rotation="1 MB")
    main()
