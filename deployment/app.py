"""
FastAPI Serving Application
Serves churn predictions using the Production model from MLflow / local artifacts.
"""

import os
import sys
import json
from pathlib import Path
import joblib
import pandas as pd
import xgboost  # Explicitly import for joblib unpickling
import sklearn  # Explicitly import for joblib unpickling
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import mlflow.pyfunc
from loguru import logger

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

app = FastAPI(title="ChurnOps Prediction API", version="1.1.0")

# ── Schemas ────────────────────────────────────────────────────────────────────
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction_default: int
    churn_prediction_optimal: int
    optimal_threshold: float
    recommended_action: str

# ── Global Model Load ──────────────────────────────────────────────────────────
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
METRICS_PATH = "reports/metrics.json"

model = None
preprocessor = None
optimal_threshold = 0.35

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, optimal_threshold
    try:
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics_data = json.load(f)
                optimal_threshold = metrics_data.get("optimal_threshold", 0.35)
                
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Connecting to MLflow at {tracking_uri}")
        
        model = mlflow.pyfunc.load_model("models:/ChurnModel/Production")
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        logger.success("Model loaded from MLflow Production stage.")
    except Exception as mlflow_e:
        logger.warning(f"MLflow load failed, falling back to local artifacts: {mlflow_e}")
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.success("Model and Preprocessor loaded from local artifacts.")

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy", "optimal_threshold": optimal_threshold}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_df = pd.DataFrame([data.dict()])
        
        service_cols = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        
        input_df["AvgMonthlySpend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
        input_df["IsHighValue"]     = (input_df["MonthlyCharges"] > 70).astype(int) 
        input_df["HasFiberOptic"]   = (input_df["InternetService"] == "Fiber optic").astype(int)
        
        input_df["NumServices"] = input_df[service_cols].apply(
            lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
            axis=1
        )
        
        input_df["LTV_Estimate"] = input_df["tenure"] * input_df["MonthlyCharges"]
        input_df["BundleValue"]  = input_df["NumServices"] / (input_df["MonthlyCharges"] + 1)
        
        extra_svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        input_df["SecurityBundleCount"] = input_df[extra_svc_cols].apply(
            lambda row: sum(1 for v in row if v == "Yes"), axis=1
        )
        
        bins = [0, 12, 24, 48, 72, 100]
        labels = ["New", "Junior", "Mid", "Senior", "Veteran"]
        input_df["TenureGroup"] = pd.cut(input_df["tenure"], bins=bins, labels=labels, include_lowest=True).astype(str)
        
        processed_data = preprocessor.transform(input_df)
        
        prob = float(model.predict_proba(processed_data)[0, 1])
        pred_def = int(prob >= 0.50)
        pred_opt = int(prob >= optimal_threshold)
        
        action = (
            "Immediate high-touch retention offer ($50 budget)" if prob >= optimal_threshold
            else "Standard monitoring — no promotional discount required"
        )
        
        return {
            "churn_probability": round(prob, 4),
            "churn_prediction_default": pred_def,
            "churn_prediction_optimal": pred_opt,
            "optimal_threshold": optimal_threshold,
            "recommended_action": action
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

