"""
Business Value Analysis
Calculates the estimated cost savings of the churn prediction model.
"""

import pandas as pd
import joblib
import yaml
# pyrefly: ignore [missing-import]
from loguru import logger
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def calculate_roi(y_true, y_prob, threshold=0.5):
    # Business assumptions
    avg_customer_value = 500  # $ revenue lost per churn
    intervention_cost = 50    # $ cost of promotion to keep customer
    success_rate = 0.4        # 40% of churners stay if given promotion
    
    # Calculate predictions based on threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Outcomes (Ensure numpy arrays for boolean indexing)
    tp = ((y_pred == 1) & (y_true == 1)).sum() 
    fp = ((y_pred == 1) & (y_true == 0)).sum() 
    fn = ((y_pred == 0) & (y_true == 1)).sum() 
    
    # Financials
    # Revenue saved = TP * success_rate * avg_customer_value
    # Total Cost   = (TP + FP) * intervention_cost
    # Potential Revenue Lost (missed churners) = FN * avg_customer_value
    revenue_saved = tp * success_rate * avg_customer_value
    costs = (tp + fp) * intervention_cost
    net_profit = revenue_saved - costs

    return {
        "revenue_saved": revenue_saved,
        "intervention_costs": costs,
        "net_profit": net_profit,
        "churners_missed_cost": fn * avg_customer_value
    }

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    test_df = pd.read_parquet(params["data"]["test_path"])
    y_test = test_df[params["data"]["target_column"]]
    X_test = test_df.drop(columns=[params["data"]["target_column"]])
    
    model = joblib.load("models/model.pkl")
    probs = model.predict_proba(X_test)[:, 1]
    
    roi = calculate_roi(y_test, probs)
    
    logger.info("=== BUSINESS ROI ANALYSIS ===")
    logger.info(f"Estimated Revenue Saved: ${roi['revenue_saved']:,.2f}")
    logger.info(f"Intervention Costs:      ${roi['intervention_costs']:,.2f}")
    logger.info(f"Net Profit:             ${roi['net_profit']:,.2f}")
    logger.info(f"Cost of Missed Churn:    ${roi['churners_missed_cost']:,.2f}")
    
    with open("reports/business_roi.yaml", "w") as f:
        yaml.dump(roi, f)

if __name__ == "__main__":
    main()
