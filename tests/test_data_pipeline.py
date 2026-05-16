"""
Unit tests for Phase 1 & 2: Data Loading and Preprocessing
Run with: pytest tests/test_data_pipeline.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.data_load import validate_dataset, load_params
from src.stages.data_preprocess import fix_raw_data, engineer_features
from src.utils.generate_drift import generate_drifted_data


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_raw_df():
    """Minimal mock of the IBM Telco dataset."""
    return pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [1, 24, 60],
        "PhoneService": ["No", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL"],
        "OnlineSecurity": ["No", "Yes", "No"],
        "OnlineBackup": ["Yes", "No", "Yes"],
        "DeviceProtection": ["No", "Yes", "No"],
        "TechSupport": ["No", "No", "Yes"],
        "StreamingTV": ["No", "Yes", "No"],
        "StreamingMovies": ["No", "No", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 53.85],
        "TotalCharges": ["29.85", "1889.5", "3225.1"],
        "Churn": ["No", "No", "Yes"],
    })


# ── Tests: Data Loading ───────────────────────────────────────────────────────
class TestDataLoading:
    def test_validate_dataset_passes_valid_df(self, sample_raw_df):
        """Should pass validation without errors."""
        validate_dataset(sample_raw_df)

    def test_validate_dataset_fails_on_missing_column(self, sample_raw_df):
        """Should raise ValueError if required column is missing."""
        df_bad = sample_raw_df.drop(columns=["Churn"])
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_dataset(df_bad)

    def test_params_loaded_correctly(self):
        """params.yaml should be readable and contain expected keys."""
        params = load_params()
        assert "data" in params
        assert "train" in params
        assert "preprocessing" in params


# ── Tests: Preprocessing ─────────────────────────────────────────────────────
class TestPreprocessing:
    def test_fix_raw_data_encodes_churn(self, sample_raw_df):
        """Churn column should be numeric after fixing."""
        fixed = fix_raw_data(sample_raw_df.copy())
        assert fixed["Churn"].dtype in [np.int64, np.int32, float]

    def test_fix_raw_data_handles_total_charges(self, sample_raw_df):
        """TotalCharges with spaces should be converted to float."""
        df = sample_raw_df.copy()
        df.loc[0, "TotalCharges"] = " "  # Simulate issue
        fixed = fix_raw_data(df)
        assert fixed["TotalCharges"].dtype == float

    def test_feature_engineering_adds_new_columns(self, sample_raw_df):
        """Feature engineering should add 3 new columns."""
        fixed = fix_raw_data(sample_raw_df.copy())
        engineered = engineer_features(fixed)
        assert "AvgMonthlySpend" in engineered.columns
        assert "IsHighValue" in engineered.columns
        assert "NumServices" in engineered.columns

    def test_num_services_is_non_negative(self, sample_raw_df):
        """NumServices should never be negative."""
        fixed = fix_raw_data(sample_raw_df.copy())
        engineered = engineer_features(fixed)
        assert (engineered["NumServices"] >= 0).all()


# ── Tests: Drift Generator ────────────────────────────────────────────────────
class TestDriftGenerator:
    def test_drift_changes_monthly_charges(self, sample_raw_df):
        """Drifted MonthlyCharges should be higher than original."""
        drifted = generate_drifted_data(sample_raw_df.copy(), drift_intensity=0.5)
        assert drifted["MonthlyCharges"].mean() > sample_raw_df["MonthlyCharges"].mean()

    def test_drift_does_not_drop_columns(self, sample_raw_df):
        """Drift generator should not remove any columns."""
        drifted = generate_drifted_data(sample_raw_df.copy())
        assert set(drifted.columns) == set(sample_raw_df.columns)

    def test_zero_drift_preserves_data(self, sample_raw_df):
        """With zero drift intensity, data should be mostly unchanged."""
        drifted = generate_drifted_data(sample_raw_df.copy(), drift_intensity=0.0)
        pd.testing.assert_series_equal(
            drifted["MonthlyCharges"], sample_raw_df["MonthlyCharges"], check_exact=False
        )
