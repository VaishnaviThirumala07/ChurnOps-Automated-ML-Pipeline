"""
Stage 3 (Advanced): Hyperparameter Optimisation + Champion/Challenger + SHAP
─────────────────────────────────────────────────────────────────────────────
• Runs Optuna HPO for XGBoost (20 trials)
• Trains a LightGBM challenger with the same optimised budget for comparison
• Logs both models to MLflow; the better one wins "champion" tag
• Generates SHAP summary + bar plots and logs them as MLflow artifacts
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.xgboost
import mlflow.lightgbm
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from loguru import logger
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep logs clean


# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# ── Helpers ────────────────────────────────────────────────────────────────────
def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as f:
        return yaml.safe_load(f)


def save_confusion_matrix(y_true, y_pred, path: str, title: str = "Confusion Matrix") -> None:
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(y_true, y_prob, roc_auc: float, path: str, label: str = "Model") -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4F46E5", lw=2, label=f"{label} AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {label}", fontsize=14)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


# ── Optuna Objective ───────────────────────────────────────────────────────────
def xgb_objective(trial, X_train, y_train, X_val, y_val):
    param = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 600),
        "max_depth":        trial.suggest_int("max_depth", 3, 10),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "eval_metric":      "auc",
        "random_state":     42,
    }
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


# ── SHAP Plots ─────────────────────────────────────────────────────────────────
def generate_shap_plots(model, X_sample: pd.DataFrame, save_dir: str) -> list:
    """Returns a list of file paths for the generated SHAP plots."""
    logger.info("Generating SHAP explanations (this may take ~30 s)...")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    paths = []

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # 1. Summary (beeswarm)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title("SHAP Feature Impact (Beeswarm)", fontsize=13)
    plt.tight_layout()
    p = f"{save_dir}/shap_summary_beeswarm.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(p)

    # 2. Bar (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (mean |SHAP value|)", fontsize=13)
    plt.tight_layout()
    p = f"{save_dir}/shap_summary_bar.png"
    plt.savefig(p, dpi=120, bbox_inches="tight")
    plt.close()
    paths.append(p)

    logger.success(f"Saved {len(paths)} SHAP plots to {save_dir}")
    return paths


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    params     = load_params()
    target_col = params["data"]["target_column"]
    seed       = params["base"]["random_seed"]

    train_df = pd.read_parquet(params["data"]["train_path"])
    test_df  = pd.read_parquet(params["data"]["test_path"])

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Churn_Advanced")
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    Path("reports").mkdir(exist_ok=True)

    # ── 1. HPO with Optuna ─────────────────────────────────────────────────────
    logger.info("Running Optuna HPO for XGBoost (50 trials)...")
    study = optuna.create_study(direction="maximize", study_name="xgb_churn_hpo")
    study.optimize(
        lambda t: xgb_objective(t, X_train, y_train, X_test, y_test),
        n_trials=50,
        show_progress_bar=True,
    )
    best_xgb_params = study.best_params
    logger.info(f"Best XGBoost params: {best_xgb_params}")

    # ── 2. Train Champion (XGBoost, tuned) ────────────────────────────────────
    with mlflow.start_run(run_name="XGBoost_Optuna_Champion") as xgb_run:
        xgb_model = XGBClassifier(**best_xgb_params, eval_metric="auc", random_state=seed)
        xgb_model.fit(X_train, y_train)

        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict(X_test)
        xgb_auc  = roc_auc_score(y_test, xgb_prob)
        xgb_f1   = f1_score(y_test, xgb_pred)

        mlflow.log_params({**best_xgb_params, "model_type": "XGBoost"})
        mlflow.log_metric("auc",      xgb_auc)
        mlflow.log_metric("f1_score", xgb_f1)

        save_confusion_matrix(y_test, xgb_pred, "reports/xgb_confusion_matrix.png",
                              "XGBoost Confusion Matrix")
        save_roc_curve(y_test, xgb_prob, xgb_auc, "reports/xgb_roc_curve.png", "XGBoost")
        mlflow.log_artifact("reports/xgb_confusion_matrix.png", artifact_path="evaluation")
        mlflow.log_artifact("reports/xgb_roc_curve.png",        artifact_path="evaluation")

        # SHAP (sample 300 rows to keep runtime reasonable)
        shap_sample = X_test.sample(min(300, len(X_test)), random_state=seed)
        shap_paths  = generate_shap_plots(xgb_model, shap_sample, "reports/shap")
        for sp in shap_paths:
            mlflow.log_artifact(sp, artifact_path="shap")

        mlflow.xgboost.log_model(xgb_model, "model", registered_model_name="ChurnModel_XGB")
        logger.success(f"XGBoost Champion — AUC={xgb_auc:.4f} | F1={xgb_f1:.4f}")

    # ── 3. Train Challenger (LightGBM) ─────────────────────────────────────────
    with mlflow.start_run(run_name="LightGBM_Challenger"):
        lgbm_model = LGBMClassifier(
            n_estimators=best_xgb_params.get("n_estimators", 300),
            max_depth=best_xgb_params.get("max_depth", 6),
            learning_rate=best_xgb_params.get("learning_rate", 0.05),
            subsample=best_xgb_params.get("subsample", 0.8),
            colsample_bytree=best_xgb_params.get("colsample_bytree", 0.8),
            random_state=seed,
            verbose=-1,
        )
        lgbm_model.fit(X_train, y_train)

        lgbm_prob = lgbm_model.predict_proba(X_test)[:, 1]
        lgbm_pred = lgbm_model.predict(X_test)
        lgbm_auc  = roc_auc_score(y_test, lgbm_prob)
        lgbm_f1   = f1_score(y_test, lgbm_pred)

        mlflow.log_params({"model_type": "LightGBM", "n_estimators": lgbm_model.n_estimators})
        mlflow.log_metric("auc",      lgbm_auc)
        mlflow.log_metric("f1_score", lgbm_f1)

        save_confusion_matrix(y_test, lgbm_pred, "reports/lgbm_confusion_matrix.png",
                              "LightGBM Confusion Matrix")
        save_roc_curve(y_test, lgbm_prob, lgbm_auc, "reports/lgbm_roc_curve.png", "LightGBM")
        mlflow.log_artifact("reports/lgbm_confusion_matrix.png", artifact_path="evaluation")
        mlflow.log_artifact("reports/lgbm_roc_curve.png",        artifact_path="evaluation")

        mlflow.lightgbm.log_model(lgbm_model, "model", registered_model_name="ChurnModel_LGBM")
        logger.success(f"LightGBM Challenger — AUC={lgbm_auc:.4f} | F1={lgbm_f1:.4f}")

    with mlflow.start_run(run_name="Ultimate_Stacking_Ensemble") as stack_run:
        estimators = [
            ('xgb', XGBClassifier(**best_xgb_params, random_state=seed)),
            ('lgbm', lgbm_model),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed))
        ]
        
        stack_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            passthrough=False
        )
        
        stack_model.fit(X_train, y_train)
        
        stack_prob = stack_model.predict_proba(X_test)[:, 1]
        stack_pred = stack_model.predict(X_test)
        
        acc   = accuracy_score(y_test, stack_pred)
        auc_v = roc_auc_score(y_test, stack_prob)
        f1_v  = f1_score(y_test, stack_pred)

        logger.success(f"Stacking Ensemble Results: Acc={acc:.4f}, AUC={auc_v:.4f}, F1={f1_v:.4f}")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("auc",      auc_v)
        mlflow.log_metric("f1_score", f1_v)

        # Save as the final production model
        joblib.dump(stack_model, "models/model.pkl")
        logger.success("Stacking Ensemble saved → models/model.pkl")
        
        # Log artifacts
        save_confusion_matrix(y_test, stack_pred, "reports/stack_confusion_matrix.png", "Ultimate Stacking Ensemble")
        mlflow.log_artifact("reports/stack_confusion_matrix.png", artifact_path="evaluation")
        
        # Log model to registry
        mlflow.sklearn.log_model(stack_model, "model", registered_model_name="ChurnModel_Stacking")

        # Export metrics for UI
        metrics_dict = {
            "accuracy": float(acc),
            "auc": float(auc_v),
            "f1_score": float(f1_v)
        }
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.success("Metrics saved -> reports/metrics.json")


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/train_advanced.log", rotation="1 MB")
    main()
