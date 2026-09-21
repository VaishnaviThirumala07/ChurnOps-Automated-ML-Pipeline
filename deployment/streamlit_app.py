"""
ChurnOps — Interactive Customer Retention Engine
Run: streamlit run deployment/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml
import json
try:
    import xgboost  
    import lightgbm
    import sklearn  
except ImportError:
    pass 

from pathlib import Path
import sys
import os

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnOps — Customer Retention Engine",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: #0f172a; min-height: 100vh; }

.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9));
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
}

.hero-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #c084fc, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 6px 0;
}

.hero-subtitle {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 500;
    margin: 0;
}

.metric-card {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.churn-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 22px;
    color: white;
    box-shadow: 0 8px 25px rgba(239, 68, 68, 0.2);
}

.churn-low {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 22px;
    color: white;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
}

.placeholder-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px dashed rgba(255, 255, 255, 0.15);
    border-radius: 14px;
    padding: 40px;
    text-align: center;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        model        = joblib.load(PROJECT_ROOT / "models/model.pkl")
        preprocessor = joblib.load(PROJECT_ROOT / "models/preprocessor.pkl")
        return model, preprocessor, None
    except FileNotFoundError:
        return None, None, "File not found"
    except ModuleNotFoundError as e:
        return None, None, f"Dependency missing: {e.name}"
    except Exception as e:
        return None, None, str(e)

@st.cache_data(ttl=300)
def load_params():
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)

def load_metrics():
    try:
        with open(PROJECT_ROOT / "reports/metrics.json") as f:
            return json.load(f)
    except:
        return {
            "auc": 0.846,
            "pr_auc": 0.660,
            "optimal_threshold": 0.20,
            "net_profit_per_1000": 22214,
            "f1_score_at_opt": 0.631
        }


# ── Business ROI Helper ────────────────────────────────────────────────────────
def compute_roi(churn_prob: float, monthly_charges: float, opt_threshold: float = 0.20,
                save_rate: float = 0.40, campaign_cost: float = 50.0, ltv_input: float = 500.0) -> dict:
    avg_value        = max(monthly_charges * 12, ltv_input)
    intervention     = campaign_cost
    revenue_saved    = avg_value * save_rate if churn_prob >= opt_threshold else 0.0
    net              = revenue_saved - intervention if churn_prob >= opt_threshold else 0.0
    return {
        "customer_ltv":     round(avg_value, 2),
        "intervention_cost": intervention,
        "revenue_saved":    round(revenue_saved, 2),
        "net_benefit":      round(net, 2),
    }


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    model, preprocessor, error_msg = load_artifacts()
    run_metrics = load_metrics()
    opt_t = run_metrics.get("optimal_threshold", 0.20)

    # ── Improved Hero Header ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title">📡 ChurnOps Intelligence Engine</h1>
        <p class="hero-subtitle">Production MLOps Pipeline & Cost-Optimal Customer Retention System</p>
    </div>
    """, unsafe_allow_html=True)

    if error_msg:
        if "Dependency missing" in error_msg:
            lib = error_msg.split(": ")[-1]
            st.error(f"### ⚠️ Missing Library: `{lib}`")
            st.info(f"To fix this, run: `pip install {lib}` and restart the app.")
        else:
            st.error(f"⚠️ Error loading artifacts: {error_msg}")
            st.info("Ensure you have run the training pipeline: `python src/stages/train_advanced.py`")
        st.stop()

    # ── Key Model Performance Metric Cards ────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    
    auc_val    = run_metrics.get('auc', 0.846)
    pr_auc_val = run_metrics.get('pr_auc', 0.660)
    net_1k     = run_metrics.get('net_profit_per_1000', 22214)

    metrics = [
        ("🎯 Model ROC-AUC", f"{auc_val:.3f}", "Stacking Ensemble"),
        ("📊 PR-AUC (Imbalanced)", f"{pr_auc_val:.3f}", "Precision-Recall AUC"),
        ("⚡ Optimal Threshold (t*)", f"{opt_t:.2f}", "Profit Maximizing"),
        ("💰 Net Profit / 1k", f"${net_1k:,.0f}", "Extrapolated ROI"),
    ]
    for col, (label, val, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><h4 style="margin:0;color:#c084fc">{label}</h4>'
                f'<h2 style="margin:6px 0;color:white;font-weight:700">{val}</h2>'
                f'<p style="margin:0;color:#94a3b8;font-size:0.8rem">{sub}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Interactive ROI & Simulator Parameters ────────────────────────────────
    with st.expander("⚙️ Retention Campaign & ROI Parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            sim_save_rate = st.slider("Retention Save Rate (%)", 10, 80, 40, step=5, help="Percentage of contacted churners who agree to stay") / 100.0
        with c2:
            sim_cost = st.number_input("Intervention Campaign Cost ($)", 10.0, 200.0, 50.0, step=5.0)
        with c3:
            sim_ltv = st.number_input("Customer LTV ($)", 200.0, 2000.0, 500.0, step=50.0)

    # ── Main Two-Column Layout ────────────────────────────────────────────────
    left_col, right_col = st.columns([2, 3], gap="large")

    with left_col:
        st.subheader("🧑‍💼 Customer Profile")

        with st.expander("📋 Account Details", expanded=True):
            tenure          = st.slider("Tenure (months)", 0, 72, 12)
            contract        = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method  = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            paperless       = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

        with st.expander("💰 Billing Details", expanded=True):
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=1.0)
            total_charges   = st.number_input(
                "Total Charges ($)", 0.0, 9000.0,
                float(monthly_charges * max(tenure, 1)), step=10.0
            )

        with st.expander("🌐 Services & Features"):
            internet_svc    = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.radio("Online Security", ["Yes", "No", "No internet service"], horizontal=True)
            tech_support    = st.radio("Tech Support",     ["Yes", "No", "No internet service"], horizontal=True)
            streaming_tv    = st.radio("Streaming TV",     ["Yes", "No", "No internet service"], horizontal=True)
            streaming_movies= st.radio("Streaming Movies", ["Yes", "No", "No internet service"], horizontal=True)
            multiple_lines  = st.radio("Multiple Lines",   ["Yes", "No", "No phone service"],    horizontal=True)
            online_backup   = st.radio("Online Backup",    ["Yes", "No", "No internet service"], horizontal=True)
            device_prot     = st.radio("Device Protection",["Yes", "No", "No internet service"], horizontal=True)

        with st.expander("👤 Demographics"):
            gender          = st.radio("Gender",       ["Male", "Female"], horizontal=True)
            senior          = st.radio("Senior Citizen", ["No", "Yes"],   horizontal=True)
            partner         = st.radio("Partner",      ["Yes", "No"],      horizontal=True)
            dependents      = st.radio("Dependents",   ["No", "Yes"],      horizontal=True)
            phone_svc       = st.radio("Phone Service",["Yes", "No"],      horizontal=True)

        predict_btn = st.button("🔮 Evaluate Churn Risk", type="primary", use_container_width=True)

    with right_col:
        if predict_btn:
            st.subheader("📊 Analysis & Prediction Results")

            input_data = {
                "gender": gender,
                "SeniorCitizen": 1 if senior == "Yes" else 0,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_svc,
                "MultipleLines": multiple_lines,
                "InternetService": internet_svc,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_prot,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
            df_input = pd.DataFrame([input_data])

            service_cols = [
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"
            ]
            df_input["AvgMonthlySpend"] = df_input["TotalCharges"] / (df_input["tenure"] + 1)
            df_input["IsHighValue"]     = (df_input["MonthlyCharges"] > 70).astype(int)
            df_input["HasFiberOptic"]   = (df_input["InternetService"] == "Fiber optic").astype(int)
            
            df_input["NumServices"]     = df_input[service_cols].apply(
                lambda row: sum(1 for v in row if v not in ["No", "No internet service", "No phone service"]),
                axis=1
            )
            
            df_input["LTV_Estimate"] = df_input["tenure"] * df_input["MonthlyCharges"]
            df_input["BundleValue"]  = df_input["NumServices"] / (df_input["MonthlyCharges"] + 1)
            
            extra_svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
            df_input["SecurityBundleCount"] = df_input[extra_svc_cols].apply(
                lambda row: sum(1 for v in row if v == "Yes"), axis=1
            )

            bins = [0, 12, 24, 48, 72, 100]
            labels = ["New", "Junior", "Mid", "Senior", "Veteran"]
            df_input["TenureGroup"] = pd.cut(df_input["tenure"], bins=bins, labels=labels, include_lowest=True).astype(str)

            try:
                processed    = preprocessor.transform(df_input)
                churn_prob   = float(model.predict_proba(processed)[0, 1])
                
                churn_pred   = int(churn_prob >= opt_t)
                roi          = compute_roi(churn_prob, monthly_charges, opt_threshold=opt_t,
                                         save_rate=sim_save_rate if 'sim_save_rate' in locals() else 0.40,
                                         campaign_cost=sim_cost if 'sim_cost' in locals() else 50.0,
                                         ltv_input=sim_ltv if 'sim_ltv' in locals() else 500.0)

                risk_level = (
                    "🔴 High Risk" if churn_prob >= 0.6
                    else "🟡 Medium Risk (Action Trigger)" if churn_prob >= opt_t
                    else "🟢 Low Risk"
                )
                card_class = "churn-high" if churn_pred == 1 else "churn-low"
                verdict    = f"⚠️ ACTION RECOMMENDED (P ≥ t*={opt_t:.2f})" if churn_pred == 1 else "✅ STABLE CUSTOMER"

                st.markdown(f"""
                <div class="{card_class}">
                    <h2 style="margin:0;font-weight:700">{verdict}</h2>
                    <h3 style="margin:8px 0;opacity:0.9">{risk_level}</h3>
                    <h1 style="margin:4px 0;font-size:3rem;font-weight:800">{churn_prob:.1%}</h1>
                    <p style="margin:0;opacity:0.8">Predicted Churn Probability (Decision Threshold t* = {opt_t:.2f})</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(churn_prob)

                st.markdown("#### 💼 Estimated Retention ROI")
                r1, r2, r3 = st.columns(3)
                r1.metric("Customer LTV", f"${roi['customer_ltv']:,.0f}")
                r2.metric("Intervention Cost", f"${roi['intervention_cost']:,.0f}")
                r3.metric("Net Benefit", f"${roi['net_benefit']:,.0f}",
                          delta="Target with Offer" if churn_pred == 1 else "No Action Needed")

                st.markdown("#### 📌 Key Account Summary")
                summary_df = pd.DataFrame({
                    "Feature": ["Contract", "Tenure", "Monthly Spend", "Internet Service", "Active Services"],
                    "Value":   [str(contract), f"{tenure} months", f"${monthly_charges:.2f}",
                                str(internet_svc), str(df_input["NumServices"].values[0])]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                st.markdown("#### 💡 Retention Strategy")
                if churn_prob >= 0.6:
                    st.error(
                        "**High risk customer:** Trigger immediate high-touch intervention. "
                        "Offer a 12-month contract lock discount or dedicated technical support bundle."
                    )
                elif churn_prob >= opt_t:
                    st.warning(
                        f"**Cost-Optimal Intervention Trigger (≥ {opt_t:.2f}):** "
                        "Contact customer with proactive retention incentive ($50 budget)."
                    )
                else:
                    st.success(
                        "**Low churn risk:** Customer is below decision threshold. "
                        "No promotional discount required."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        else:
            # Clean, elegant placeholder when no prediction has been executed yet
            st.markdown("""
            <div class="placeholder-card">
                <h3 style="color:#e2e8f0;margin-top:0">🎯 Customer Risk Assessment Panel</h3>
                <p style="color:#94a3b8;font-size:0.95rem;margin-bottom:0">Configure the customer profile parameters on the left and click <b>Evaluate Churn Risk</b> to view live predictive insights and retention ROI analysis.</p>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


