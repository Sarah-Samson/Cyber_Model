import os
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime
import streamlit as st

# Agno and Google Imports
from agno.agent import Agent
from agno.models.google import Gemini
from scipy import stats
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Cyber Insurance Debugger", page_icon="🛡️", layout="wide")
st.title("🛡️ Cyber Insurance Pricing Engine - Diagnostics Mode")

# =====================================================================
# PHASE 1: CACHED DATA LOADING & TRAINING (Prevents Streamlit Rerun Loop)
# =====================================================================

@st.cache_data
def load_and_train_models():
    """Loads raw data once and extracts the exact regression coefficients."""
    try:
        incidents = pd.read_csv("https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/incidents_master_cleaned.csv")
        financial = pd.read_csv("https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/financial_impact_cleaned.csv")
        
        # 1. Frequency Model Training
        company_freq = incidents.groupby("company_name").agg({
            "incident_id": "count",
            "company_revenue_usd": "first",
            "employee_count": "first",
            "is_public_company": "first",
        }).reset_index().dropna()

        X_freq = company_freq[["company_revenue_usd", "employee_count", "is_public_company"]].copy()
        y_freq = company_freq["incident_id"]
        X_freq["log_revenue"] = np.log1p(X_freq["company_revenue_usd"])
        X_freq["log_employees"] = np.log1p(X_freq["employee_count"])
        X_freq["revenue_tier"] = pd.cut(X_freq["company_revenue_usd"], bins=[0, 1e9, 10e9, 100e9, np.inf], labels=[0, 1, 2, 3]).astype(int)
        
        freq_cols = ["log_revenue", "log_employees", "is_public_company", "revenue_tier"]
        X_train_f, _, y_train_f, _ = train_test_split(X_freq[freq_cols].fillna(0), y_freq, test_size=0.2, random_state=42)
        
        freq_model = PoissonRegressor(alpha=0.1292, max_iter=1000)
        freq_model.fit(X_train_f, y_train_f)
        
        freq_coefs = {
            "intercept": float(freq_model.intercept_),
            "log_revenue": float(freq_model.coef_[0]),
            "log_employees": float(freq_model.coef_[1]),
            "is_public": float(freq_model.coef_[2])
        }

        # 2. Severity Model Training
        merged = incidents.merge(financial, on="incident_id", how="inner")
        log_losses = np.log(merged["total_loss_usd"].values)
        mu, sigma = float(log_losses.mean()), float(log_losses.std())

        X_sev = merged[["company_revenue_usd", "employee_count", "is_public_company"]].copy()
        X_sev["log_revenue"] = np.log1p(X_sev["company_revenue_usd"])
        X_sev["log_employees"] = np.log1p(X_sev["employee_count"])
        X_sev["log_records"] = np.log1p(1000000)
        
        sev_cols = ["log_revenue", "log_employees", "is_public_company", "log_records"]
        X_train_s, _, y_train_s, _ = train_test_split(X_sev[sev_cols].fillna(0), np.log(merged["total_loss_usd"]), test_size=0.2, random_state=42)
        
        sev_model = Ridge(alpha=1.0)
        sev_model.fit(X_train_s, y_train_s)
        
        sev_coefs = {
            "intercept": float(sev_model.intercept_),
            "log_revenue": float(sev_model.coef_[0]),
            "log_employees": float(sev_model.coef_[1]),
            "is_public": float(sev_model.coef_[2]),
            "log_records": float(sev_model.coef_[3]),
        }

        return freq_coefs, sev_coefs, {"mu": mu, "sigma": sigma}, True
    except Exception as e:
        st.error(f"Failed to initialize models natively: {e}")
        return {}, {}, {}, False

# Trigger background load
FREQ_COEFFICIENTS, SEVER_COEFFICIENTS, LOGNORM_PARAMS, DATA_OK = load_and_train_models()

# Show coefficients in sidebar for sanity validation
if DATA_OK:
    st.sidebar.success("✅ Models Trained & Cached Safely")
    st.sidebar.json({"Freq Intercept": FREQ_COEFFICIENTS["intercept"], "Sev Intercept": SEVER_COEFFICIENTS["intercept"]})

# =====================================================================
# PHASE 2: CORE ACTUARIAL CALCULATIONS
# =====================================================================

INDUSTRY_RELATIVITIES = {"51": 1.231, "52": 1.181, "44-45": 1.264, "92": 1.221, "31-33": 1.023}
LOADING_MULTIPLIER = 1.58

def premium_quotation_tool(
    company_name: str,
    company_revenue_usd: float,
    employee_count: int,
    industry_code: str,
    is_public_company: bool,
    data_records_at_risk: int = 1000000,
) -> str:
    """The central calculation method with explicit input typing validation."""
    try:
        # TYPE ENFORCEMENT GUARDRAIL: Forces strings out of LLM payloads into numbers
        rev = float(company_revenue_usd)
        emp = int(employee_count)
        records = int(data_records_at_risk)
        
        # Explicit Boolean Conversion Guardrail
        if isinstance(is_public_company, str):
            pub = is_public_company.lower() in ['true', 'yes', '1']
        else:
            pub = bool(is_public_company)

        # 1. Math Evaluation
        log_rev = np.log1p(rev)
        log_emp = np.log1p(emp)
        pub_num = 1 if pub else 0

        # Freq Logic
        log_lambda = FREQ_COEFFICIENTS["intercept"] + (FREQ_COEFFICIENTS["log_revenue"] * log_rev) + (FREQ_COEFFICIENTS["log_employees"] * log_emp) + (FREQ_COEFFICIENTS["is_public"] * pub_num)
        freq = np.exp(log_lambda)

        # Sev Logic
        log_loss = SEVER_COEFFICIENTS["intercept"] + (SEVER_COEFFICIENTS["log_revenue"] * log_rev) + (SEVER_COEFFICIENTS["log_employees"] * log_emp) + (SEVER_COEFFICIENTS["is_public"] * pub_num) + (SEVER_COEFFICIENTS["log_records"] * np.log1p(records))
        sev = np.exp(log_loss)

        # 2. Premium Compilation
        pure_premium = freq * sev
        base_loaded_premium = pure_premium * LOADING_MULTIPLIER
        ind_rel = INDUSTRY_RELATIVITIES.get(str(industry_code), 1.0)
        final_premium = base_loaded_premium * ind_rel

        # Return explicit trace payload back to the chat interface
        diagnostics = {
            "SYSTEM_INPUT_RECEIVED": {
                "parsed_revenue": rev,
                "parsed_employees": emp,
                "parsed_is_public": pub,
                "raw_rev_type_from_llm": str(type(company_revenue_usd)),
                "raw_public_type_from_llm": str(type(is_public_company))
            },
            "MATHEMATICAL_TRACE": {
                "calculated_frequency": round(freq, 4),
                "calculated_severity": round(sev, 2),
                "pure_premium": round(pure_premium, 2),
                "final_premium_calculated": round(final_premium, 2)
            }
        }
        return json.dumps(diagnostics, indent=2)

    except Exception as e:
        return json.dumps({"calculation_error": str(e)}, indent=2)

# =====================================================================
# PHASE 3: AGENT STRUCTURING & CHAT ENVIRONMENT
# =====================================================================

if "insurance_agent_messages" not in st.session_state:
    st.session_state.insurance_agent_messages = [{"role": "assistant", "content": "Diagnostics Engine online. Request a quote to check parameters."}]

# Persist Agent Session State so it doesn't drop weights on interaction
if "active_agent" not in st.session_state:
    st.session_state.active_agent = Agent(
        name="Cyber Premium Quotation Agent",
        model=Gemini(id="gemini-2.5-flash"), # Safe standard baseline
        tools=[premium_quotation_tool],
        instructions="""You are an underwriting validator. When a user requests a premium quote, you MUST call premium_quotation_tool immediately. Ensure you pass numbers natively without strings if possible.""",
        markdown=True
    )

for msg in st.session_state.insurance_agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Enter test profile..."):
    st.session_state.insurance_agent_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            agent_response = st.session_state.active_agent.run(user_query)
            output_text = agent_response.content if hasattr(agent_response, 'content') else str(agent_response)
            response_placeholder.markdown(output_text)
            st.session_state.insurance_agent_messages.append({"role": "assistant", "content": output_text})
        except Exception as e:
            response_placeholder.error(f"Engine Fault: {e}")
