"""
Business Value & Threshold Optimizer Analysis
Calculates cost savings, optimal decision threshold, and net ROI under explicit business assumptions.
"""

import pandas as pd
import numpy as np
import joblib
import yaml
from loguru import logger
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def calculate_roi_at_threshold(y_true, y_prob, threshold, ltv=500, cost=50, save_rate=0.4):
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    
    revenue_saved = float(tp * save_rate * ltv)
    intervention_costs = float((tp + fp) * cost)
    net_profit = float(revenue_saved - intervention_costs)
    roi_percent = float((net_profit / intervention_costs * 100.0) if intervention_costs > 0 else 0.0)

    return {
        "threshold": float(round(threshold, 2)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "revenue_saved": round(revenue_saved, 2),
        "intervention_costs": round(intervention_costs, 2),
        "net_profit": round(net_profit, 2),
        "roi_percent": round(roi_percent, 1),
        "missed_churn_cost": round(float(fn * ltv), 2)
    }

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    biz = params.get("business", {"ltv": 500, "intervention_cost": 50, "save_rate": 0.40})
    ltv = biz.get("ltv", 500)
    cost = biz.get("intervention_cost", 50)
    save_rate = biz.get("save_rate", 0.40)
    
    test_df = pd.read_parquet(params["data"]["test_path"])
    y_test = test_df[params["data"]["target_column"]].values
    X_test = test_df.drop(columns=[params["data"]["target_column"]])
    
    model = joblib.load("models/model.pkl")
    probs = model.predict_proba(X_test)[:, 1]
    
    # Search threshold grid
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_t = 0.50
    best_roi = None
    max_net = -float("inf")
    
    for t in thresholds:
        r = calculate_roi_at_threshold(y_test, probs, t, ltv=ltv, cost=cost, save_rate=save_rate)
        if r["net_profit"] > max_net:
            max_net = r["net_profit"]
            best_t = t
            best_roi = r
            
    default_roi = calculate_roi_at_threshold(y_test, probs, 0.50, ltv=ltv, cost=cost, save_rate=save_rate)
    
    net_per_1k = (best_roi["net_profit"] / len(y_test)) * 1000.0

    output = {
        "assumptions": {
            "ltv": ltv,
            "intervention_cost": cost,
            "save_rate": save_rate,
            "disclaimer": "Under assumed LTV ($500), intervention cost ($50), and save rate (40%)"
        },
        "optimal_threshold": best_roi,
        "default_threshold": default_roi,
        "extrapolated_per_1000_customers": {
            "net_profit": round(net_per_1k, 2),
            "roi_percent": best_roi["roi_percent"]
        }
    }
    
    logger.info("=== BUSINESS ROI & THRESHOLD OPTIMIZER ANALYSIS ===")
    logger.info(f"Explicit Disclaimer: {output['assumptions']['disclaimer']}")
    logger.info(f"Optimal Threshold (t*): {best_t:.2f}")
    logger.info(f"Optimal Net Profit (Test Cohort): ${best_roi['net_profit']:,.2f} (ROI: {best_roi['roi_percent']}%)")
    logger.info(f"Extrapolated Net Profit / 1,000 Customers: ${net_per_1k:,.2f}")
    logger.info(f"Default (t=0.50) Net Profit: ${default_roi['net_profit']:,.2f} (ROI: {default_roi['roi_percent']}%)")
    
    with open("reports/business_roi.yaml", "w") as f:
        yaml.dump(output, f, default_flow_style=False)

if __name__ == "__main__":
    main()

