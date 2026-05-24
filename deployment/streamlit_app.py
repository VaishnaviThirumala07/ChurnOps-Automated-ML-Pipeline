"""
Churn Prediction — Interactive Streamlit Demo
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
    page_title="Churn Prediction System",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

.metric-card {
    background: rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease;
}
.metric-card:hover { transform: translateY(-3px); }

.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.churn-high {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 12px;
    padding: 20px;
    color: white;
}
.churn-low {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 12px;
    padding: 20px;
    color: white;
}
.gauge-container {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 16px;
    border: 1px solid rgba(255,255,255,0.1);
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

@st.cache_data
def load_params():
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_metrics():
    try:
        with open(PROJECT_ROOT / "reports/metrics.json") as f:
            return json.load(f)
    except:
        return {"auc": 0.0, "accuracy": 0.0, "f1_score": 0.0}


# ── Business ROI Helper ────────────────────────────────────────────────────────
def compute_roi(churn_prob: float, monthly_charges: float) -> dict:
    avg_value        = monthly_charges * 12
    intervention     = 50.0
    success_rate     = 0.40
    revenue_saved    = avg_value * success_rate if churn_prob >= 0.5 else 0.0
    net              = revenue_saved - intervention if churn_prob >= 0.5 else 0.0
    return {
        "customer_ltv":     round(avg_value, 2),
        "intervention_cost": intervention,
        "revenue_saved":    round(revenue_saved, 2),
        "net_benefit":      round(net, 2),
    }


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    model, preprocessor, error_msg = load_artifacts()

    # ── Header ────────────────────────────────────────────────────────────────
    col_logo, col_title = st.columns([1, 9])
    with col_title:
        st.markdown('<p class="hero-title">🔮 Visionary Intelligence</p>', unsafe_allow_html=True)
        st.markdown(
            "**Enterprise Customer Retention Engine & Predictive Analytics**",
            help="Next-generation churn prediction using advanced ensemble stacking"
        )

    if error_msg:
        if "Dependency missing" in error_msg:
            lib = error_msg.split(": ")[-1]
            st.error(f"### ⚠️ Missing Library: `{lib}`")
            st.info(f"To fix this, run: `pip install {lib}` and restart the app.")
        else:
            st.error(f"⚠️ Error loading artifacts: {error_msg}")
            st.info("Ensure you have run the training pipeline: `python src/stages/train.py`")
        st.stop()

    st.divider()

    # ── System Metrics Row ────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    run_metrics = load_metrics()
    
    metrics = [
        ("🎯 Model AUC", f"{run_metrics.get('auc', 0):.2f}", "Ultimate Stacking Ensemble"),
        ("📈 Accuracy", f"{run_metrics.get('accuracy', 0)*100:.1f}%", "Ensemble Optimized"),
        ("🔍 Monitoring", "Active", "Continuous Drift Detection"),
        ("💼 ROI Focus", "High", "Business Value Analysis"),
    ]
    for col, (label, val, sub) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><h4 style="margin:0;color:#a78bfa">{label}</h4>'
                f'<h2 style="margin:6px 0;color:white">{val}</h2>'
                f'<p style="margin:0;color:#9ca3af;font-size:0.8rem">{sub}</p></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input Panel + Prediction ──────────────────────────────────────────────
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

        with st.expander("💰 Billing", expanded=True):
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, step=1.0)
            total_charges   = st.number_input(
                "Total Charges ($)", 0.0, 9000.0,
                float(monthly_charges * max(tenure, 1)), step=10.0
            )

        with st.expander("🌐 Services"):
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

        predict_btn = st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True)

    with right_col:
        st.subheader("📊 Prediction Results")

        if predict_btn:
            # Build input DataFrame
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

            # Feature engineering (mirrors preprocess stage)
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
            
            # New high-impact features
            df_input["LTV_Estimate"] = df_input["tenure"] * df_input["MonthlyCharges"]
            df_input["BundleValue"]  = df_input["NumServices"] / (df_input["MonthlyCharges"] + 1)
            
            extra_svc_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
            df_input["SecurityBundleCount"] = df_input[extra_svc_cols].apply(
                lambda row: sum(1 for v in row if v == "Yes"), axis=1
            )

            # Tenure binning (matching Stage 2)
            bins = [0, 12, 24, 48, 72, 100]
            labels = ["New", "Junior", "Mid", "Senior", "Veteran"]
            df_input["TenureGroup"] = pd.cut(df_input["tenure"], bins=bins, labels=labels, include_lowest=True).astype(str)

            try:
                processed    = preprocessor.transform(df_input)
                churn_prob   = float(model.predict_proba(processed)[0, 1])
                churn_pred   = int(model.predict(processed)[0])
                roi          = compute_roi(churn_prob, monthly_charges)

                # ── Prediction card ───────────────────────────────────────────
                risk_level = (
                    "🔴 High Risk" if churn_prob >= 0.7
                    else "🟡 Medium Risk" if churn_prob >= 0.4
                    else "🟢 Low Risk"
                )
                card_class = "churn-high" if churn_pred == 1 else "churn-low"
                verdict    = "⚠️ LIKELY TO CHURN" if churn_pred == 1 else "✅ LIKELY TO STAY"

                st.markdown(f"""
                <div class="{card_class}">
                    <h2 style="margin:0">{verdict}</h2>
                    <h3 style="margin:8px 0">{risk_level}</h3>
                    <h1 style="margin:4px 0;font-size:2.8rem">{churn_prob:.1%}</h1>
                    <p style="margin:0;opacity:0.8">Churn Probability</p>
                </div>
                """, unsafe_allow_html=True)

                # ── Probability gauge ─────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Churn Probability**")
                st.progress(churn_prob)

                # ── Business ROI ──────────────────────────────────────────────
                st.markdown("#### 💼 Business Impact Analysis")
                r1, r2, r3 = st.columns(3)
                r1.metric("Customer LTV (annual)", f"${roi['customer_ltv']:,.0f}")
                r2.metric("Intervention Cost",     f"${roi['intervention_cost']:,.0f}")
                r3.metric("Net Benefit (if acted)", f"${roi['net_benefit']:,.0f}",
                          delta="Save customer" if churn_pred == 1 else "No action needed")

                # ── Feature summary ───────────────────────────────────────────
                st.markdown("#### 📌 Key Input Summary")
                summary_df = pd.DataFrame({
                    "Feature": ["Contract", "Tenure", "Monthly $", "Internet", "Num Services"],
                    "Value":   [str(contract), f"{tenure} months", f"${monthly_charges:.2f}",
                                str(internet_svc), str(df_input["NumServices"].values[0])]
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # ── Recommendation ────────────────────────────────────────────
                st.markdown("#### 💡 Recommended Action")
                if churn_prob >= 0.7:
                    st.error(
                        "**Immediate action required.** Offer a personalised retention package. "
                        "Consider a contract upgrade discount or loyalty reward."
                    )
                elif churn_prob >= 0.4:
                    st.warning(
                        "**Monitor closely.** Proactively reach out with a satisfaction check-in "
                        "and highlight service value."
                    )
                else:
                    st.success(
                        "**Customer is stable.** No intervention needed. "
                        "Consider upselling additional services."
                    )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

        else:
            st.info("👈 Fill in the customer profile and click **Predict Churn Risk**")

            st.markdown("#### 💡 How it works")
            st.markdown("""
            This dashboard uses a machine learning model to analyze customer behavior patterns and predict the probability of churn. 
            By entering customer details on the left, you can get a real-time risk assessment and a calculated estimate of the potential 
            financial impact of retaining that customer.
            """)


if __name__ == "__main__":
    main()
