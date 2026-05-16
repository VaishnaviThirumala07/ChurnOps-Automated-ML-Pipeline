"""
Stage 3: Model Training
Trains an XGBoost model using processed data and logs results to MLflow.
Produces: models/model.pkl, MLflow run with metrics, confusion matrix, and ROC curve.
"""

import os
import sys
import io
import yaml
import joblib
import pandas as pd
import mlflow
import mlflow.xgboost
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI/servers
import matplotlib.pyplot as plt
import io

from pathlib import Path
from loguru import logger
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    classification_report,
)


# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ── Load Params ────────────────────────────────────────────────────────────────
def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


# ── Plot: Confusion Matrix ─────────────────────────────────────────────────────
def _save_confusion_matrix(y_true, y_pred, save_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {save_path}")


# ── Plot: ROC Curve ────────────────────────────────────────────────────────────
def _save_roc_curve(y_true, y_prob, roc_auc: float, save_path: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4F46E5", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Model", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved ROC curve → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    params = load_params()

    # Paths
    train_path = params["data"]["train_path"]
    test_path  = params["data"]["test_path"]
    model_name = params["train"]["model_name"]
    target_col = params["data"]["target_column"]

    # Hyperparameters
    n_estimators      = params["train"]["n_estimators"]
    max_depth         = params["train"]["max_depth"]
    learning_rate     = params["train"]["learning_rate"]
    subsample         = params["train"]["subsample"]
    colsample_bytree  = params["train"]["colsample_bytree"]
    scale_pos_weight  = params["train"]["scale_pos_weight"]
    seed              = params["base"]["random_seed"]

    # Load Data
    logger.info("Loading processed data...")
    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    # MLflow Setup — prefer env var so CI works without code changes
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Churn_Prediction")
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    with mlflow.start_run(run_name="XGBoost_Baseline"):
        logger.info("Starting model training...")

        # Initialize & Train
        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="auc",
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        roc_score = roc_auc_score(y_test, y_prob)
        f1        = f1_score(y_test, y_pred)
        acc       = accuracy_score(y_test, y_pred)

        logger.info(f"AUC: {roc_score:.4f} | F1: {f1:.4f} | ACC: {acc:.4f}")
        logger.info(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

        # MLflow — log params and scalar metrics
        mlflow.log_params(params["train"])
        mlflow.log_metric("auc",      roc_score)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)

        # MLflow — log visual artifacts
        Path("reports").mkdir(exist_ok=True)
        cm_path  = "reports/confusion_matrix.png"
        roc_path = "reports/roc_curve.png"

        _save_confusion_matrix(y_test, y_pred, cm_path)
        _save_roc_curve(y_test, y_prob, roc_score, roc_path)

        mlflow.log_artifact(cm_path,  artifact_path="evaluation")
        mlflow.log_artifact(roc_path, artifact_path="evaluation")

        # Log Model to Registry
        mlflow.xgboost.log_model(model, "model", registered_model_name=model_name)

        # Save Local Model (DVC-tracked artifact)
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/baseline_model.pkl")
        logger.success(
            f"Baseline model saved → models/baseline_model.pkl and logged to MLflow Registry as '{model_name}'"
        )


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/train.log", rotation="1 MB")
    main()
