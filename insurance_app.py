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
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --- UI Configuration ---
st.set_page_config(
    page_title="Cyber Insurance Pricing Engine", 
    page_icon="🛡️", 
    layout="wide"
)

# Custom Styling (Black and Gold Theme)
st.markdown("""
    <style>
    .reportview-container { background: #111111; color: #FFFFFF; }
    .main-header { color: #D4AF37; font-size: 32px; font-weight: bold; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🛡️ Cyber Insurance Pricing Engine & Risk Validator</div>', unsafe_allow_html=True)

# --- Constant Mappings ---
INDUSTRY_CODES = {
    "51": "Technology",
    "52": "Finance & Insurance",
    "44-45": "Retail Trade",
    "92": "Public Administration / Telecom",
    "31-33": "Industrial Manufacturing"
}
INDUSTRY_RELATIVITIES = {"51": 1.231, "52": 1.181, "44-45": 1.264, "92": 1.221, "31-33": 1.023}
LOADING_MULTIPLIER = 1.58

# =====================================================================
# PHASE 1: DATA PIPELINE & MODEL CACHING
# =====================================================================

@st.cache_data
def initialize_and_train_models():
    """Loads operational records and fits Generalized Linear Regression models natively."""
    try:
        incidents = pd.read_csv("https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/incidents_master_cleaned.csv")
        financial = pd.read_csv("https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/financial_impact_cleaned.csv")
        
        # 1. Frequency Model
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

        # 2. Severity Model
        merged = incidents.merge(financial, on="incident_id", how="inner")
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

        return freq_coefs, sev_coefs, True
    except Exception as e:
        st.error(f"Actuarial Core Calibration Error: {e}")
        return {}, {}, False

# Run global background pipeline configuration
FREQ_COEFFICIENTS, SEVER_COEFFICIENTS, DATA_OPERATIONAL = initialize_and_train_models()

# =====================================================================
# PHASE 2: MATHEMATICAL PREDICTIONS & SUB-ROUTINES
# =====================================================================

def predict_frequency(revenue: float, employees: int, is_public: bool) -> dict:
    log_rev = np.log1p(revenue)
    log_emp = np.log1p(employees)
    pub_val = 1 if is_public else 0
    
    log_lambda = (
        FREQ_COEFFICIENTS["intercept"] + 
        (FREQ_COEFFICIENTS["log_revenue"] * log_rev) + 
        (FREQ_COEFFICIENTS["log_employees"] * log_emp) + 
        (FREQ_COEFFICIENTS["is_public"] * pub_val)
    )
    predicted_freq = np.exp(log_lambda)
    risk_score = min(100.0, max(0.0, (predicted_freq / 2.5) * 100))
    return {"predicted_frequency": float(predicted_freq), "risk_score": float(risk_score)}

def predict_severity(revenue: float, employees: int, is_public: bool, records: int) -> dict:
    log_rev = np.log1p(revenue)
    log_emp = np.log1p(employees)
    log_rec = np.log1p(records)
    pub_val = 1 if is_public else 0
    
    log_loss = (
        SEVER_COEFFICIENTS["intercept"] + 
        (SEVER_COEFFICIENTS["log_revenue"] * log_rev) + 
        (SEVER_COEFFICIENTS["log_employees"] * log_emp) + 
        (SEVER_COEFFICIENTS["is_public"] * pub_val) + 
        (SEVER_COEFFICIENTS["log_records"] * log_rec)
    )
    return {"expected_severity": float(np.exp(log_loss))}

def calculate_pure_premium(freq: float, sev: float) -> dict:
    pure_premium = freq * sev
    final_premium = pure_premium * LOADING_MULTIPLIER
    return {
        "pure_premium": pure_premium,
        "final_premium": final_premium,
        "loading_components": {
            "acquisition": final_premium * 0.20,
            "admin": final_premium * 0.10,
            "profit": final_premium * 0.15,
            "uncertainty": final_premium * 0.08,
            "reinsurance": final_premium * 0.05
        },
        "total_loading": final_premium - pure_premium
    }

# =====================================================================
# PHASE 3: AGENT ANALYTICS HELPER TOOLS
# =====================================================================

def explain_coverage_tiers(tier_name: str) -> str:
    """Provides legal policy definitions for requested coverage levels."""
    definitions = {
        "tier 1": "Primary Coverage (Tier 1) mitigates external catastrophic events: Ransomware response, large-scale data leaks, and direct upstream supply-chain software compromises.",
        "tier 2": "Secondary Coverage (Tier 2) monitors operational disruptions: Layer 7 DDoS attacks, internal Trojan infections, automated keyloggers, and persistence-based Advanced Persistent Threats (APTs)."
    }
    return definitions.get(tier_name.lower().strip(), "Specified coverage tier parameters are unmapped.")

def compare_coverage_costs(base_premium: float) -> str:
    """Applies actuarial split metrics to compute tier cost profiles."""
    try:
        premium = float(base_premium)
        comparison = {
            "tier_1_cost": f"${(premium * 0.625):,.2f}",
            "tier_2_cost": f"${(premium * 0.375):,.2f}",
            "combined_total": f"${premium:,.2f}"
        }
        return json.dumps(comparison, indent=2)
    except Exception as e:
        return f"Comparison breakdown engine error: {e}"

# =====================================================================
# PHASE 4: CENTRAL UNDERWRITING CALCULATION ENGINE
# =====================================================================

def premium_quotation_tool(
    company_name: str,
    company_revenue_usd: float,
    employee_count: int,
    industry_code: str,
    is_public_company: bool,
    data_records_at_risk: int = 1000000,
) -> str:
    """Computes final quotes while dynamically blocking shorthand parameter hallucinations."""
    try:
        # --- INPUT TYPE & UNIT GUARDRAIL ---
        raw_revenue = float(company_revenue_usd)
        parsed_employees = int(employee_count)
        parsed_records = int(data_records_at_risk)
        
        # Unit scaling safety verification
        if raw_revenue < 1000000:
            if parsed_employees > 500 or parsed_records > 100000:
                raw_revenue *= 1_000_000_000  # Automatically scales up Billions
            else:
                raw_revenue *= 1_000_000      # Automatically scales up Millions

        if isinstance(is_public_company, str):
            parsed_is_public = is_public_company.lower() in ['true', 'yes', '1', 'public']
        else:
            parsed_is_public = bool(is_public_company)

        # Execution of predictive sub-routines
        freq_res = predict_frequency(raw_revenue, parsed_employees, parsed_is_public)
        sev_res = predict_severity(raw_revenue, parsed_employees, parsed_is_public, parsed_records)
        
        ind_rel = INDUSTRY_RELATIVITIES.get(str(industry_code), 1.0)
        prem_res = calculate_pure_premium(freq_res["predicted_frequency"], sev_res["expected_severity"])
        
        adjusted_premium = prem_res["final_premium"] * ind_rel
        
        t1 = adjusted_premium * 0.625
        t2 = adjusted_premium * 0.375

        quotation = {
            "quotation_id": f"QUOTE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "company_profile": {
                "name": company_name,
                "revenue_usd": f"${raw_revenue:,.0f}",
                "employees": parsed_employees,
                "industry": INDUSTRY_CODES.get(str(industry_code), "Unknown Segment"),
                "corporate_status": "Public" if parsed_is_public else "Private",
            },
            "actuarial_metrics": {
                "predicted_annual_frequency": round(freq_res["predicted_frequency"], 4),
                "composite_risk_score": f"{freq_res['risk_score']:.1f}/100",
                "expected_incident_severity": f"${sev_res['expected_severity']:,.2f}",
                "pure_premium_baseline": f"${prem_res['pure_premium']:,.2f}"
            },
            "adjustments": {
                "industry_relativity_applied": f"{ind_rel:.3f}x",
                "total_annual_combined_premium": f"${adjusted_premium:,.2f}"
            },
            "coverage_structures": {
                "tier_1_primary": {
                    "scope": "Ransomware, Data Breaches, Supply Chain Compromises",
                    "annual_premium": f"${t1:,.2f}"
                },
                "tier_2_secondary": {
                    "scope": "DDoS Mitigations, Trojans, Malware, APT System Damages",
                    "annual_premium": f"${t2:,.2f}"
                },
                "combined_total": f"${adjusted_premium:,.2f}"
            }
        }
        return json.dumps(quotation, indent=2)
    except Exception as e:
        return json.dumps({"underwriting_engine_fault": str(e)}, indent=2)

# =====================================================================
# PHASE 5: DEFINITIVE AGENT INITIALIZATION & AGNO WORKFLOW
# =====================================================================

def create_quotation_agent() -> Agent:
    """Generates the primary pricing agent bound to explicit tool structures."""
    return Agent(
        name="Cyber Insurance Pricing Agent",
        model=Gemini(id="gemini-2.5-flash"),
        tools=[premium_quotation_tool, explain_coverage_tiers, compare_coverage_costs],
        instructions="""You are an expert underwriter executing tasks within an insurance ecosystem.
        
        CRITICAL PROCESSING INTERFACE RULES:
        1. Always map conversational revenue phrases into precise numerical float representations for 'premium_quotation_tool':
           - '150 Billion' -> 150000000000.0
           - '25 Million' -> 25000000.0
           - Do NOT pass shorthand single digits like 150 or 25. Doing so breaks the GLM pricing weights.
           
        2. Translate sector strings to the target system codes:
           - Tech/Technology -> '51'
           - Finance/Banking/Insurance -> '52'
           - Retail -> '44-45'
           - Telecom/Public Admin -> '92'
           - Manufacturing -> '31-33'
           
        3. Present final calculated outputs using clean Markdown tables featuring Tier 1, Tier 2, and Combined coverage amounts.""",
        markdown=True
    )

# --- Chat Interface Stateful Orchestration ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Welcome to the Cyber Insurance Pricing Engine. Please provide company profiles to begin risk underwriting."}
    ]

if "pricing_agent" not in st.session_state:
    st.session_state.pricing_agent = create_quotation_agent()

# Render historic interactions
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

# Process current conversation framework
if prompt_input := st.chat_input("Enter company risk profile details..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
        ui_placeholder = st.empty()
        try:
            agent_output = st.session_state.pricing_agent.run(prompt_input)
            response_payload = agent_output.content if hasattr(agent_output, 'content') else str(agent_output)
            ui_placeholder.markdown(response_payload)
            st.session_state.chat_history.append({"role": "assistant", "content": response_payload})
        except Exception as runtime_err:
            ui_placeholder.error(f"Agent Framework Connection Exception: {runtime_err}")
