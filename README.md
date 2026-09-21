# 📡 ChurnOps: Automated MLOps Pipeline for Churn Prediction

[![ML Pipeline](https://github.com/your-username/DS_Project1/actions/workflows/retrain.yml/badge.svg)](https://github.com/your-username/DS_Project1/actions)
[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![MLflow](https://img.shields.io/badge/MLflow-2.10-blue?logo=mlflow)](https://mlflow.org)
[![DVC](https://img.shields.io/badge/DVC-3.47-purple)](https://dvc.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Production-grade MLOps system** for telecom customer churn prediction on the IBM Telco dataset (7,043 rows). Features leak-free evaluation, Stacking Ensembles, Optuna HPO, SHAP explainability, threshold optimization, REST API serving, Evidently AI drift monitoring, and GitHub Actions automated retraining.

---

## 📊 Results & Leak-Free Evaluation Protocol

> [!NOTE]
> **Evaluation Integrity**: Stratified 80/20 train/test split is applied prior to any feature transformation or scaling. Test set remains un-oversampled and untouched during training to prevent data leakage. Given the ~26.5% class imbalance, model performance is evaluated using **PR-AUC (Precision-Recall AUC)** alongside **ROC-AUC** and **Recall@t***.

| Metric | Baseline XGBoost | Stacking Ensemble (Default t=0.50) | **Stacking Ensemble (Optimal t*=0.20)** |
| :--- | :--- | :--- | :--- |
| **ROC-AUC** | 0.848 | 0.846 | **0.846** |
| **PR-AUC (Imbalanced)** | 0.658 | 0.660 | **0.660** |
| **Recall (Churners Caught)** | 52.4% | 46.8% | **81.3%** |
| **Precision** | 63.8% | 68.2% | **51.5%** |
| **F1-Score** | 0.575 | 0.555 | **0.631** |

---

## 💰 Business ROI & Threshold Optimization

Rather than using an arbitrary `0.50` decision threshold, this system employs a **cost-sensitive threshold optimizer** $t^*$ that maximizes expected monetary profit on customer retention campaigns:

$$\text{Net Profit}(t) = \text{TP}(t) \times (S \times V - C) - \text{FP}(t) \times C$$

### Explicit ROI Assumptions:
* **Customer Lifetime Value ($V$)**: $500
* **Intervention Campaign Cost ($C$)**: $50
* **Retention Campaign Save Rate ($S$)**: 40% (40% of contacted churners are successfully retained)

### Financial Impact (per 1,000 Customers)
> *Under assumed LTV ($500), intervention cost ($50), and save rate (40%)*

| Metric | Default Threshold (t=0.50) | **Optimal Threshold (t*=0.20)** |
| :--- | :--- | :--- |
| **Targeted Customers** | 182 | **420** |
| **Churners Retained (TP × 40%)** | ~49 customers ($24,500 value) | **~86 customers ($43,000 value)** |
| **Campaign Cost** | $9,100 | **$21,000** |
| **Net Profit** | **~$15,400** | **~$22,214 (+44% gain)** |
| **Net Campaign ROI** | 169.2% | **105.8%** |

---


## 🏗️ Architecture

```mermaid
graph TD
    DS[IBM Telco Dataset] --> DL[Stage 1: Data Load]
    DL --> DV[Stage 1.5: Pandera Validation]
    DV --> PP[Stage 2: Preprocessing + Feature Engineering]
    PP --> TR[Stage 3: XGBoost Training]
    PP --> TA[Stage 4: Optuna HPO + Champion/Challenger + SHAP]

    TR --> MLR[(MLflow Registry)]
    TA --> MLR

    MLR --> API[FastAPI Serving :8000]
    MLR --> UI[Streamlit Demo UI]

    API -->|Live Predictions| ED[Evidently AI Drift Detection]
    ED -->|Metrics| PM[Prometheus :9090]
    PM --> GF[Grafana :3000]

    PM -->|drift_share > 0.5| DT[Drift Trigger Service]
    DT -->|repository_dispatch| GHA[GitHub Actions CI/CD]
    GHA --> TR

    style MLR fill:#1e3a5f,color:#fff
    style GHA fill:#2d1b69,color:#fff
    style DT fill:#7f1d1d,color:#fff
```

---

---


## ⚡ Quick Start

### 1. Setup Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Copy and fill in environment variables
```bash
cp .env.example .env
# Edit .env with your MLflow URI, GitHub token, etc.
```

### 3. Run the DVC Pipeline (Reproducible)
```bash
dvc repro
```
Or run stages individually:
```bash
python src/stages/data_load.py
python src/stages/data_validate.py
python src/stages/data_preprocess.py
python src/stages/train.py
```

### 4. Advanced Training (Optuna HPO + SHAP + Champion/Challenger)
```bash
python src/stages/train_advanced.py
```

### 5. Start the Monitoring Stack
```bash
docker-compose up -d
```
| Service | URL |
|---------|-----|
| **FastAPI** | http://localhost:8000 |
| **API Docs** | http://localhost:8000/docs |
| **MLflow UI** | http://localhost:5000 |
| **Prometheus** | http://localhost:9090 |
| **Grafana** | http://localhost:3000 |

### 6. Launch the Streamlit Demo
```bash
streamlit run deployment/streamlit_app.py
```

### 7. Simulate Drift & Trigger Retraining
```bash
# Generate drifted data
python src/utils/generate_drift.py

# Run drift analysis
python src/monitoring/drift_service.py

# Start retraining trigger (polls Prometheus every 60s)
export GITHUB_TOKEN=<your-token>
export GITHUB_REPO=<owner/repo>
python src/utils/drift_trigger.py
```

---

## 🔬 ML Pipeline Details

### Feature Engineering
| Feature | Description |
|---------|-------------|
| `AvgMonthlySpend` | `TotalCharges / (tenure + 1)` — spend velocity |
| `IsHighValue` | Binary flag: `MonthlyCharges > median` |
| `NumServices` | Count of active subscriptions (0–9) |

### Hyperparameter Optimisation (Optuna — 20 trials)
Tuned parameters: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`

### SHAP Explainability
- **Beeswarm plot**: Feature impact direction and magnitude per prediction
- **Bar plot**: Global mean |SHAP value| feature ranking
- Both logged as MLflow artifacts for every advanced training run

---

## 🧪 Tests
```bash
pytest tests/ -v
```
| Test Class | Coverage |
|------------|----------|
| `TestDataLoading` | Schema validation, missing columns, params loading |
| `TestPreprocessing` | Target encoding, TotalCharges fix, feature engineering |
| `TestDriftGenerator` | Drift application, column preservation, zero-drift case |

---

## 🔄 CI/CD Flow

1. **Push to `main`** → GitHub Actions triggers `ML Training Pipeline`
2. **DVC pull** fetches cached data/models from remote (DagsHub)
3. Pipeline runs: `data_load → preprocess → train`
4. `promote_model.py` transitions the new version to **Production** in MLflow Registry
5. Docker image built and pushed to DockerHub
6. Training reports (confusion matrix, ROC) uploaded as GitHub workflow artifacts

**Drift-triggered retraining:**
- `drift_service.py` (Evidently) detects feature drift > 0.5
- `drift_trigger.py` calls GitHub API `repository_dispatch` with `event_type: drift_retrain`
- Same pipeline re-runs automatically, producing a new registered model version

---

## 🔧 Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `MLFLOW_TRACKING_URI` | DagsHub or remote MLflow URI |
| `DAGSHUB_USER_TOKEN` | For `dvc pull` in CI |
| `DOCKER_USERNAME` | DockerHub username |
| `DOCKER_PASSWORD` | DockerHub access token |
| `GITHUB_TOKEN` | Auto-available in Actions (for dispatches) |

---

---

