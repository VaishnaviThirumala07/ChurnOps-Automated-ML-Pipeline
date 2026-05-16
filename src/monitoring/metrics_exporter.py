"""
Prometheus Metrics Exporter
Exposes drift and model performance metrics to Prometheus.
"""

from prometheus_client import start_http_server, Gauge, Counter
import time
import json
import random
from pathlib import Path
from loguru import logger

# ── Define Metrics ────────────────────────────────────────────────────────────
DRIFT_SHARE = Gauge('ml_drift_share', 'Fraction of drifted features')
DRIFTED_COLUMNS = Gauge('ml_drifted_columns_count', 'Number of drifted columns')
PREDICTION_COUNT = Counter('ml_predictions_total', 'Total number of predictions made')
CHURN_PROB_AVG = Gauge('ml_churn_probability_avg', 'Average churn probability in last window')

def export_metrics(report_path: str):
    """Reads Evidently JSON report and updates Prometheus gauges."""
    if not Path(report_path).exists():
        logger.warning(f"Report not found at {report_path}")
        return

    with open(report_path, "r") as f:
        data = json.load(f)
        
    # Extract from Evidently JSON structure
    # Note: Exact path depends on Evidently version; this matches 0.4.x
    metrics = data["metrics"]
    drift_res = next(m["result"] for m in metrics if m["metric"] == "DatasetDriftMetric")
    
    DRIFT_SHARE.set(drift_res["drift_share"])
    DRIFTED_COLUMNS.set(drift_res["number_of_drifted_columns"])
    
    logger.info(f"Updated Prometheus metrics from {report_path}")

if __name__ == "__main__":
    # Start exporter on port 9000
    start_http_server(9000)
    logger.info("Prometheus exporter started on port 9000")
    
    while True:
        export_metrics("reports/drift_report.json")
        time.sleep(30)
