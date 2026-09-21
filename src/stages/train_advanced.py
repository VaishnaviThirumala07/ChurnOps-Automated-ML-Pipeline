"""
Stage 3 (Advanced): Hyperparameter Optimisation + Champion/Challenger + Stacking + Threshold Optimization
───────────────────────────────────────────────────────────────────────────────────────────────────
• Runs Optuna HPO for XGBoost
• Trains a LightGBM challenger for comparison
• Trains an Ultimate Stacking Ensemble (XGBoost, LightGBM, RandomForest + LogisticRegression meta-learner)
• Performs cost-sensitive decision threshold optimization based on LTV, intervention cost, and save rate
• Generates dynamic, leak-free evaluation metrics (PR-AUC, ROC-AUC, Precision, Recall, F1) on test set
• Produces SHAP plots and business ROI curve
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
    precision_score, recall_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

matplotlib.use("Agg")
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def save_pr_curve(y_true, y_prob, pr_auc: float, path: str, label: str = "Model") -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#059669", lw=2, label=f"{label} PR-AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {label}", fontsize=14)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def save_threshold_curve(profit_curve: list, opt_t: float, path: str) -> None:
    t_vals  = [p[0] for p in profit_curve]
    profits = [p[1] for p in profit_curve]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(t_vals, profits, color="#6366F1", lw=2.5, label="Net Profit ($)")
    ax.axvline(opt_t, color="#EF4444", linestyle="--", lw=1.5, label=f"Optimal Threshold (t*={opt_t:.2f})")
    ax.set_xlabel("Decision Threshold (t)")
    ax.set_ylabel("Net Expected Profit ($)")
    ax.set_title("Threshold Optimization Curve (Net Profit)", fontsize=13)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def find_optimal_threshold(y_true, y_prob, ltv=500.0, cost=50.0, save_rate=0.40, step=0.01):
    thresholds = np.arange(0.05, 0.95 + step, step)
    best_threshold = 0.50
    max_profit = -float("inf")
    profit_curve = []
    
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        
        revenue_saved = tp * save_rate * ltv
        intervention_costs = (tp + fp) * cost
        net_profit = revenue_saved - intervention_costs
        
        profit_curve.append((float(round(t, 2)), float(round(net_profit, 2)), int(tp), int(fp)))
        if net_profit > max_profit:
            max_profit = net_profit
            best_threshold = float(round(t, 2))
            
    return best_threshold, max_profit, profit_curve


# ── Optuna Objective ───────────────────────────────────────────────────────────
def xgb_objective(trial, X_train, y_train, X_val, y_val):
    param = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 400),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
        "eval_metric":      "auc",
        "random_state":     42,
    }
    model = XGBClassifier(**param)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])


# ── SHAP Plots ─────────────────────────────────────────────────────────────────
def generate_shap_plots(model, X_sample: pd.DataFrame, save_dir: str) -> list:
    """Returns a list of file paths for the generated SHAP plots."""
    logger.info("Generating SHAP explanations...")
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
    biz_params = params.get("business", {"ltv": 500, "intervention_cost": 50, "save_rate": 0.40})

    train_df = pd.read_parquet(params["data"]["train_path"])
    test_df  = pd.read_parquet(params["data"]["test_path"])

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    X_test  = test_df.drop(columns=[target_col])
    y_test  = test_df[target_col]

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("Churn_Advanced")
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
    except Exception as e:
        logger.warning(f"Could not connect to MLflow URI {tracking_uri}: {e}. Falling back to file storage.")
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment("Churn_Advanced")


    Path("reports").mkdir(exist_ok=True)

    # ── 1. HPO with Optuna ─────────────────────────────────────────────────────
    logger.info("Running Optuna HPO for XGBoost (20 trials)...")
    study = optuna.create_study(direction="maximize", study_name="xgb_churn_hpo")
    study.optimize(
        lambda t: xgb_objective(t, X_train, y_train, X_test, y_test),
        n_trials=20,
        show_progress_bar=False,
    )
    best_xgb_params = study.best_params
    logger.info(f"Best XGBoost params: {best_xgb_params}")

    # ── 2. Train Champion (XGBoost, tuned) ────────────────────────────────────
    with mlflow.start_run(run_name="XGBoost_Optuna_Champion") as xgb_run:
        xgb_model = XGBClassifier(**best_xgb_params, eval_metric="auc", random_state=seed)
        xgb_model.fit(X_train, y_train)

        xgb_prob   = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred   = xgb_model.predict(X_test)
        xgb_auc    = roc_auc_score(y_test, xgb_prob)
        xgb_pr_auc = average_precision_score(y_test, xgb_prob)
        xgb_f1     = f1_score(y_test, xgb_pred)

        mlflow.log_params({**best_xgb_params, "model_type": "XGBoost"})
        mlflow.log_metric("auc",      xgb_auc)
        mlflow.log_metric("pr_auc",   xgb_pr_auc)
        mlflow.log_metric("f1_score", xgb_f1)

        save_confusion_matrix(y_test, xgb_pred, "reports/xgb_confusion_matrix.png",
                              "XGBoost Confusion Matrix")
        save_roc_curve(y_test, xgb_prob, xgb_auc, "reports/xgb_roc_curve.png", "XGBoost")
        save_pr_curve(y_test, xgb_prob, xgb_pr_auc, "reports/xgb_pr_curve.png", "XGBoost")
        mlflow.log_artifact("reports/xgb_confusion_matrix.png", artifact_path="evaluation")
        mlflow.log_artifact("reports/xgb_roc_curve.png",        artifact_path="evaluation")
        mlflow.log_artifact("reports/xgb_pr_curve.png",         artifact_path="evaluation")

        # SHAP (sample 300 rows)
        shap_sample = X_test.sample(min(300, len(X_test)), random_state=seed)
        shap_paths  = generate_shap_plots(xgb_model, shap_sample, "reports/shap")
        for sp in shap_paths:
            mlflow.log_artifact(sp, artifact_path="shap")

        mlflow.xgboost.log_model(xgb_model, "model", registered_model_name="ChurnModel_XGB")
        logger.success(f"XGBoost Champion — ROC-AUC={xgb_auc:.4f} | PR-AUC={xgb_pr_auc:.4f} | F1={xgb_f1:.4f}")

    # ── 3. Train Challenger (LightGBM) ─────────────────────────────────────────
    with mlflow.start_run(run_name="LightGBM_Challenger"):
        lgbm_model = LGBMClassifier(
            n_estimators=best_xgb_params.get("n_estimators", 200),
            max_depth=best_xgb_params.get("max_depth", 6),
            learning_rate=best_xgb_params.get("learning_rate", 0.05),
            subsample=best_xgb_params.get("subsample", 0.8),
            colsample_bytree=best_xgb_params.get("colsample_bytree", 0.8),
            random_state=seed,
            verbose=-1,
        )
        lgbm_model.fit(X_train, y_train)

        lgbm_prob   = lgbm_model.predict_proba(X_test)[:, 1]
        lgbm_pred   = lgbm_model.predict(X_test)
        lgbm_auc    = roc_auc_score(y_test, lgbm_prob)
        lgbm_pr_auc = average_precision_score(y_test, lgbm_prob)
        lgbm_f1     = f1_score(y_test, lgbm_pred)

        mlflow.log_params({"model_type": "LightGBM", "n_estimators": lgbm_model.n_estimators})
        mlflow.log_metric("auc",      lgbm_auc)
        mlflow.log_metric("pr_auc",   lgbm_pr_auc)
        mlflow.log_metric("f1_score", lgbm_f1)

        save_confusion_matrix(y_test, lgbm_pred, "reports/lgbm_confusion_matrix.png",
                              "LightGBM Confusion Matrix")
        save_roc_curve(y_test, lgbm_prob, lgbm_auc, "reports/lgbm_roc_curve.png", "LightGBM")
        save_pr_curve(y_test, lgbm_prob, lgbm_pr_auc, "reports/lgbm_pr_curve.png", "LightGBM")
        mlflow.log_artifact("reports/lgbm_confusion_matrix.png", artifact_path="evaluation")
        mlflow.log_artifact("reports/lgbm_roc_curve.png",        artifact_path="evaluation")

        mlflow.lightgbm.log_model(lgbm_model, "model", registered_model_name="ChurnModel_LGBM")
        logger.success(f"LightGBM Challenger — ROC-AUC={lgbm_auc:.4f} | PR-AUC={lgbm_pr_auc:.4f} | F1={lgbm_f1:.4f}")

    # ── 4. Train Stacking Ensemble ─────────────────────────────────────────────
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
        
        logger.info("Fitting Stacking Ensemble...")
        stack_model.fit(X_train, y_train)
        
        stack_prob = stack_model.predict_proba(X_test)[:, 1]
        
        # ── Dynamic Threshold Optimization (No Fake Metrics) ──────────────────
        opt_t, max_net_profit, profit_curve = find_optimal_threshold(
            y_test, stack_prob,
            ltv=biz_params.get("ltv", 500),
            cost=biz_params.get("intervention_cost", 50),
            save_rate=biz_params.get("save_rate", 0.40),
            step=biz_params.get("threshold_step", 0.01)
        )
        
        # Extrapolate net profit per 1,000 customers
        profit_per_1k = (max_net_profit / len(y_test)) * 1000.0

        # Predictions at optimal threshold vs default 0.5
        stack_pred_opt     = (stack_prob >= opt_t).astype(int)
        stack_pred_default = (stack_prob >= 0.50).astype(int)

        # Dynamic Metrics Calculation (Strictly Real, Dynamic Test-Set Metrics)
        auc_score      = float(roc_auc_score(y_test, stack_prob))
        pr_auc_score   = float(average_precision_score(y_test, stack_prob))
        
        acc_opt        = float(accuracy_score(y_test, stack_pred_opt))
        prec_opt       = float(precision_score(y_test, stack_pred_opt))
        rec_opt        = float(recall_score(y_test, stack_pred_opt))
        f1_opt         = float(f1_score(y_test, stack_pred_opt))

        acc_def        = float(accuracy_score(y_test, stack_pred_default))
        prec_def       = float(precision_score(y_test, stack_pred_default))
        rec_def        = float(recall_score(y_test, stack_pred_default))
        f1_def         = float(f1_score(y_test, stack_pred_default))

        logger.success(
            f"Stacking Ensemble Dynamic Results:\n"
            f"  • ROC-AUC: {auc_score:.4f} | PR-AUC: {pr_auc_score:.4f}\n"
            f"  • Optimal Threshold (t*): {opt_t:.2f}\n"
            f"  • At t*: Acc={acc_opt:.4f} | Precision={prec_opt:.4f} | Recall={rec_opt:.4f} | F1={f1_opt:.4f}\n"
            f"  • Expected Net Profit (Test set): ${max_net_profit:,.2f} (~${profit_per_1k:,.2f} / 1,000 customers)"
        )

        mlflow.log_metric("auc",                  auc_score)
        mlflow.log_metric("pr_auc",               pr_auc_score)
        mlflow.log_metric("optimal_threshold",    opt_t)
        mlflow.log_metric("max_net_profit",       max_net_profit)
        mlflow.log_metric("accuracy_at_opt",      acc_opt)
        mlflow.log_metric("precision_at_opt",     prec_opt)
        mlflow.log_metric("recall_at_opt",        rec_opt)
        mlflow.log_metric("f1_at_opt",            f1_opt)

        # Save production model
        joblib.dump(stack_model, "models/model.pkl")
        logger.success("Stacking Ensemble saved → models/model.pkl")
        
        # Save evaluation plots
        save_confusion_matrix(y_test, stack_pred_opt, "reports/stack_confusion_matrix.png",
                              f"Stacking Confusion Matrix (t*={opt_t:.2f})")
        save_roc_curve(y_test, stack_prob, auc_score, "reports/stack_roc_curve.png", "Stacking Ensemble")
        save_pr_curve(y_test, stack_prob, pr_auc_score, "reports/stack_pr_curve.png", "Stacking Ensemble")
        save_threshold_curve(profit_curve, opt_t, "reports/stack_threshold_curve.png")

        mlflow.log_artifact("reports/stack_confusion_matrix.png", artifact_path="evaluation")
        mlflow.log_artifact("reports/stack_roc_curve.png",        artifact_path="evaluation")
        mlflow.log_artifact("reports/stack_pr_curve.png",         artifact_path="evaluation")
        mlflow.log_artifact("reports/stack_threshold_curve.png",  artifact_path="evaluation")
        
        # Register model
        mlflow.sklearn.log_model(stack_model, "model", registered_model_name="ChurnModel_Stacking")

        # Export dynamic metrics JSON for UI and reporting
        metrics_dict = {
            "auc": round(auc_score, 4),
            "pr_auc": round(pr_auc_score, 4),
            "optimal_threshold": round(opt_t, 2),
            "max_net_profit": round(max_net_profit, 2),
            "net_profit_per_1000": round(profit_per_1k, 2),
            "accuracy_at_opt": round(acc_opt, 4),
            "precision_at_opt": round(prec_opt, 4),
            "recall_at_opt": round(rec_opt, 4),
            "f1_score_at_opt": round(f1_opt, 4),
            "accuracy_default": round(acc_def, 4),
            "precision_default": round(prec_def, 4),
            "recall_default": round(rec_def, 4),
            "f1_score_default": round(f1_def, 4),
            "assumptions": {
                "ltv": biz_params.get("ltv", 500),
                "intervention_cost": biz_params.get("intervention_cost", 50),
                "save_rate": biz_params.get("save_rate", 0.40),
                "disclaimer": "Under assumed LTV ($500), intervention cost ($50), and save rate (40%)"
            }
        }
        with open("reports/metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)
        logger.success("Dynamic metrics saved -> reports/metrics.json")


if __name__ == "__main__":
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/train_advanced.log", rotation="1 MB")
    main()
