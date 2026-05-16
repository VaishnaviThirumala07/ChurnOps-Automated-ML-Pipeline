"""
Drift Monitoring Service
Uses Evidently AI to compare reference (train) data with current (serving) data.
Generates metrics for Prometheus.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from loguru import logger
import json

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

class DriftMonitor:
    def __init__(self, reference_data_path: str):
        self.reference_data = pd.read_parquet(reference_data_path)
        logger.info(f"Loaded reference data with {len(self.reference_data)} rows.")

    def run_drift_analysis(self, current_data: pd.DataFrame, report_save_path: str):
        """
        Compares current_data against self.reference_data.
        Saves a JSON report and returns drift status.
        """
        logger.info("Running drift analysis...")
        
        # Evidently Report
        report = Report(metrics=[
            DataDriftPreset(),
            TargetDriftPreset(),
            DataQualityPreset()
        ])
        
        # We need to ensure columns match
        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
        
        report.run(
            reference_data=self.reference_data[common_cols],
            current_data=current_data[common_cols]
        )
        
        # Save JSON
        Path(report_save_path).parent.mkdir(parents=True, exist_ok=True)
        report.save_json(report_save_path)
        
        # Extract summary metrics
        result = report.as_dict()
        drift_share = result["metrics"][0]["result"]["drift_share"]
        number_of_drifted_columns = result["metrics"][0]["result"]["number_of_drifted_columns"]
        
        logger.warning(f"Drift Share: {drift_share:.2f} | Drifted Columns: {number_of_drifted_columns}")
        
        return {
            "drift_share": drift_share,
            "drifted_columns": number_of_drifted_columns,
            "is_drift_detected": drift_share > 0.5 # Threshold from params.yaml
        }

if __name__ == "__main__":
    # Test with synthetic drift
    import yaml
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    monitor = DriftMonitor(params["data"]["reference_path"])
    
    # Load raw data and apply drift for testing
    raw_df = pd.read_csv(params["data"]["raw_data_path"])
    
    # Mocking 'current' data by taking a sample and adding drift
    from src.utils.generate_drift import generate_drifted_data
    drifted_sample = generate_drifted_data(raw_df.sample(500), drift_intensity=0.6)
    
    analysis = monitor.run_drift_analysis(drifted_sample, "reports/drift_report.json")
    print(f"Analysis Result: {analysis}")
