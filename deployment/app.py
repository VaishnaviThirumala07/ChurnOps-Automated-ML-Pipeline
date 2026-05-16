"""
FastAPI Serving Application
Serves churn predictions using the Production model from MLflow.
"""

import os
import sys
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

app = FastAPI(title="Churn Prediction API", version="1.0.0")

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
    churn_prediction: int

# ── Global Model Load ──────────────────────────────────────────────────────────
# In production, we'd load from MLflow Model Registry.
# For simplicity in local dev, we can fall back to the models/ folder.
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = None
preprocessor = None

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        # Load from MLflow Registry
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Connecting to MLflow at {tracking_uri}")
        
        # Try loading Production model
        model = mlflow.pyfunc.load_model("models:/ChurnModel/Production")
        
        # We still need the preprocessor (usually logged as an artifact in the same run)
        # For simplicity, we fallback to local if not found in mlflow artifacts
        if os.path.exists(PREPROCESSOR_PATH):
            preprocessor = joblib.load(PREPROCESSOR_PATH)
        
        logger.success("Model loaded from MLflow Production stage.")
    except Exception as mlflow_e:
        logger.warning(f"MLflow load failed, falling back to local: {mlflow_e}")
        # Local fallback
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.success("Model and Preprocessor loaded from local artifacts.")
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")

# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([data.dict()])
        
        # Feature Engineering (mimic Stage 2)
        input_df["AvgMonthlySpend"] = input_df["TotalCharges"] / (input_df["tenure"] + 1)
        input_df["IsHighValue"]     = (input_df["MonthlyCharges"] > 70).astype(int) 
        input_df["HasFiberOptic"]   = (input_df["InternetService"] == "Fiber optic").astype(int)
        
        input_df["NumServices"] = input_df[service_cols].apply(
            lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
            axis=1
        )
        
        # New high-impact features
        input_df["LTV_Estimate"] = input_df["tenure"] * input_df["MonthlyCharges"]
        input_df["BundleValue"]  = input_df["NumServices"] / (input_df["MonthlyCharges"] + 1)
        
        extra_svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        input_df["SecurityBundleCount"] = input_df[extra_svc_cols].apply(
            lambda row: sum(1 for v in row if v == "Yes"), axis=1
        )
        
        # Tenure Grouping
        bins = [0, 12, 24, 48, 72, 100]
        labels = ["New", "Junior", "Mid", "Senior", "Veteran"]
        input_df["TenureGroup"] = pd.cut(input_df["tenure"], bins=bins, labels=labels, include_lowest=True).astype(str)
        
        # Preprocess
        processed_data = preprocessor.transform(input_df)
        
        # Predict
        prob = float(model.predict_proba(processed_data)[0, 1])
        pred = int(model.predict(processed_data)[0])
        
        return {
            "churn_probability": prob,
            "churn_prediction": pred
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
