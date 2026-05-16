"""
Threshold Optimizer for Business Profit
──────────────────────────────────────
Finds the decision threshold that maximizes Net Profit based on 
intervention costs and retention success rates.
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
import sys

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

def optimize_threshold():
    # 1. Load data and model
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    test_df = pd.read_parquet(params["data"]["test_path"])
    y_true = test_df[params["data"]["target_column"]].values
    X_test = test_df.drop(columns=[params["data"]["target_column"]])
    
    model = joblib.load("models/model.pkl")
    y_prob = model.predict_proba(X_test)[:, 1]

    # 2. Business parameters
    avg_customer_value = 500  # Revenue lost per churn
    intervention_cost = 50    # Cost of retention campaign
    success_rate = 0.4        # Success rate of retention

    # 3. Test thresholds
    thresholds = np.linspace(0.01, 0.99, 100)
    profits = []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        
        revenue_saved = tp * success_rate * avg_customer_value
        costs = (tp + fp) * intervention_cost
        net_profit = revenue_saved - costs
        profits.append(net_profit)

    # 4. Find best
    best_idx = np.argmax(profits)
    best_threshold = thresholds[best_idx]
    max_profit = profits[best_idx]

    logger.success(f"Optimal Business Threshold: {best_threshold:.2f}")
    logger.success(f"Maximum Predicted Profit: ${max_profit:,.2f}")

    # 5. Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, profits, color='#4F46E5', lw=2)
    plt.axvline(best_threshold, color='#EF4444', linestyle='--', label=f'Optimal Threshold: {best_threshold:.2f}')
    plt.title("Profit vs. Decision Threshold", fontsize=14)
    plt.xlabel("Probability Threshold")
    plt.ylabel("Expected Net Profit ($)")
    plt.grid(alpha=0.3)
    plt.legend()
    
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/threshold_profit_curve.png", dpi=120)
    logger.info("Saved profit curve -> reports/threshold_profit_curve.png")

    return best_threshold

if __name__ == "__main__":
    optimize_threshold()
