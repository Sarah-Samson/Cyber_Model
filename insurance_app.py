import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
from typing import Dict, Tuple, Any
from datetime import datetime
import streamlit as st

# Agno and Google Imports
from agno.agent import Agent
from agno.models.google import Gemini
from sklearn.linear_model import Ridge, PoissonRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# --- UI Layout Initialization ---
st.set_page_config(
    page_title="Cyber & AI Risk Insurance Pricing",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Cyber & AI Risk Insurance Pricing Engine")
st.markdown("Dynamic actuarial pricing using AI-powered Frequency-Severity models with absolute data-type safety.")

# --- Constant Operational Parameters ---
LOADING_FACTORS = {"acquisition": 0.20, "admin": 0.10, "profit": 0.15, "uncertainty": 0.08, "reinsurance": 0.05}
TOTAL_LOADING = sum(LOADING_FACTORS.values())
LOADING_MULTIPLIER = 1 + TOTAL_LOADING

INDUSTRY_RELATIVITIES = {"51": 1.231, "52": 1.181, "44-45": 1.264, "92": 1.221, "31-33": 1.023}
INDUSTRY_CODES = {
    "31-33": "Industrial Manufacturing",
    "44-45": "Retail",
    "51": "Technology",
    "52": "Finance & Insurance",
    "92": "Telecom"
}

# =====================================================================
# PHASE 1: CORE ACTUARIAL DATA & ENGINE CALIBRATION (CACHED)
# =====================================================================
@st.cache_data
def calibrate_actuarial_core() -> Tuple[dict, dict, dict]:
    """Loads operational datasets natively and extracts linear modeling coefficients safely."""
    try:
        incidents = pd.read_csv("https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/incidents_master_cleaned.csv")
        financial = pd.read_csv("https://raw.githubusercontent.com/rajat4186/Cyber-Risk-Premium-Pricing-Agentic-AI-Project/refs/heads/main/data/financial_impact_cleaned.csv")
        
        # Frequency Pipeline Tuning
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

        # Severity Pipeline Tuning
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
        
        lognorm_params = {"mu": float(np.log(merged["total_loss_usd"]).mean()), "sigma": float(np.log(merged["total_loss_usd"]).std())}

        return freq_coefs, sev_coefs, lognorm_params
    except Exception as e:
        st.error(f"Critical Backend Training Fault: {e}")
        return {}, {}, {}

# Run background operations
FREQ_COEFFICIENTS, SEVER_COEFFICIENTS, LOGNORM_PARAMS = calibrate_actuarial_core()

# =====================================================================
# PHASE 2: MATHEMATICAL PREDICTIONS & SUB-ROUTINES
# =====================================================================
def predict_frequency(revenue: float, employees: int, is_public: bool) -> dict:
    log_rev = np.log1p(revenue)
    log_emp = np.log1p(employees)
    pub_val = 1 if is_public else 0
    log_lambda = FREQ_COEFFICIENTS["intercept"] + (FREQ_COEFFICIENTS["log_revenue"] * log_rev) + (FREQ_COEFFICIENTS["log_employees"] * log_emp) + (FREQ_COEFFICIENTS["is_public"] * pub_val)
    predicted_freq = np.exp(log_lambda)
    return {"predicted_frequency": float(predicted_freq), "risk_score": float(min(100.0, (predicted_freq / 2.5) * 100))}

def predict_severity(revenue: float, employees: int, is_public: bool, records: int) -> dict:
    log_rev = np.log1p(revenue)
    log_emp = np.log1p(employees)
    log_rec = np.log1p(records)
    pub_val = 1 if is_public else 0
    log_loss = SEVER_COEFFICIENTS["intercept"] + (SEVER_COEFFICIENTS["log_revenue"] * log_rev) + (SEVER_COEFFICIENTS["log_employees"] * log_emp) + (SEVER_COEFFICIENTS["is_public"] * pub_val) + (SEVER_COEFFICIENTS["log_records"] * log_rec)
    return {"expected_severity": float(np.exp(log_loss))}

def calculate_pure_premium(freq: float, sev: float) -> dict:
    pure_premium = freq * sev
    final_premium = pure_premium * LOADING_MULTIPLIER
    return {
        "pure_premium": pure_premium,
        "final_premium": final_premium,
        "loading_components": {k: final_premium * v for k, v in LOADING_FACTORS.items()},
        "total_loading": final_premium - pure_premium
    }

# =====================================================================
# PHASE 3: COMPREHENSIVE UNDERWRITING SYSTEM TOOLS
# =====================================================================
def premium_quotation_tool(
    company_name: str,
    company_revenue_usd: float,
    employee_count: int,
    industry_code: str,
    is_public_company: bool,
    data_records_at_risk: int = 1000000,
) -> str:
    """Computes premium metrics while using a robust guardrail to fix single-digit parsing issues."""
    try:
        # --- ACTUARIAL UNIT GUARDRAIL ---
        raw_revenue = float(company_revenue_usd)
        parsed_employees = int(employee_count)
        parsed_records = int(data_records_at_risk)
        
        # Catch and upscale if the model clips shorthand parameters down to simple integers
        if raw_revenue < 1000000:
            if parsed_employees > 500 or parsed_records > 100000:
                raw_revenue *= 1_000_000_000
            else:
                raw_revenue *= 1_000_000

        if isinstance(is_public_company, str):
            parsed_is_public = is_public_company.lower() in ['true', 'yes', '1', 'public']
        else:
            parsed_is_public = bool(is_public_company)

        # Run Actuarial Matrix Processors
        freq_res = predict_frequency(raw_revenue, parsed_employees, parsed_is_public)
        sev_res = predict_severity(raw_revenue, parsed_employees, parsed_is_public, parsed_records)
        
        ind_rel = INDUSTRY_RELATIVITIES.get(str(industry_code), 1.0)
        prem_res = calculate_pure_premium(freq_res["predicted_frequency"], sev_res["expected_severity"])
        
        adjusted_premium = prem_res["final_premium"] * ind_rel
        t1 = adjusted_premium * 1.00
        t2 = adjusted_premium * 0.60

        quotation = {
            "quotation_id": f"QUOTE-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "company": {
                "name": company_name,
                "revenue_usd": f"${raw_revenue:,.0f}",
                "employees": parsed_employees,
                "industry": INDUSTRY_CODES.get(str(industry_code), "Unknown"),
                "status": "Public" if parsed_is_public else "Private",
            },
            "actuarial_metrics": {
                "predicted_frequency_EXTRACTED": round(freq_res["predicted_frequency"], 4),
                "risk_score": round(freq_res["risk_score"], 1),
                "expected_severity_EXTRACTED": f"${sev_res['expected_severity']:,.0f}",
                "pure_premium": f"${prem_res['pure_premium']:,.0f}",
            },
            "adjustments": {
                "base_final_premium": f"${prem_res['final_premium']:,.0f}",
                "industry_adjustment_factor": f"{ind_rel:.3f}x",
                "final_adjusted_premium": f"${adjusted_premium:,.0f}",
            },
            "coverage_tier_1_primary": {
                "name": "Primary Coverage (Tier 1)",
                "covers": ["Ransomware", "Data Breaches", "Supply Chain"],
                "annual_premium": f"${t1:,.0f}",
            },
            "coverage_tier_2_secondary": {
                "name": "Secondary Coverage (Tier 2)",
                "covers": ["DDoS", "Malware", "Trojans", "Backdoors", "APTs"],
                "annual_premium": f"${t2:,.0f}",
            },
            "coverage_combined": {
                "name": "Complete Coverage (Tier 1 + Tier 2)",
                "annual_premium": f"${(t1 + t2):,.0f}",
                "recommended_for": "Enterprise, Finance, Healthcare companies",
            },
        }
        return json.dumps(quotation, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)}, indent=2)

def explain_coverage_tiers(tier_id: int) -> str:
    """Legal framework policy descriptions mapping tool."""
    info = {
        1: "Tier 1 covers Ransomware events, Forensics responses, Core recovery pipelines, and Data Breach leaks.",
        2: "Tier 2 covers Advanced network layer DDoS mitigation, internal Backdoors, Trojans, and APT system damages."
    }
    return info.get(int(tier_id), "Invalid selection choice.")

def compare_coverage_costs(tier1_premium: float) -> str:
    """Computes basic financial tier comparisons."""
    p = float(tier1_premium)
    return json.dumps({"tier_1_only": f"${p:,.0f}", "tier_2_only": f"${(p*0.6):,.0f}", "combined": f"${(p*1.6):,.0f}"}, indent=2)

# =====================================================================
# PHASE 4: AGENT WORKFLOW DEFINITIONS & SCHEMAS
# =====================================================================
def create_quotation_agent() -> Agent:
    """Produces the localized engine orchestration instance with rigorous parsing directives."""
    return Agent(
        name="Cyber Risk Premium Quotation Agent",
        model=Gemini(id="gemini-2.5-flash"),
        tools=[premium_quotation_tool, explain_coverage_tiers, compare_coverage_costs],
        instructions="""You are an expert underwriting advisory instance.
        
        CRITICAL PROCESSING INTERFACE RULES:
        1. Always translate conversational language regarding revenue parameters into absolute raw float representations for 'premium_quotation_tool':
           - '150 Billion' -> pass exactly as 150000000000.0
           - '20 Million' -> pass exactly as 20000000.0
           - NEVER allow simple scalar terms like 150 or 20 to proceed. Doing so breaks structural formula metrics.
           
        2. Cleanly map sectors to targeting string codes:
           Tech/Technology -> '51', Finance/Banking/Insurance -> '52', Retail -> '44-45'.
           
        3. Break calculations out nicely into pristine Markdown tables featuring Tier 1, Tier 2, and combined premium amounts.""",
        markdown=True,
    )

# =====================================================================
# PHASE 5: LIVE RUNTIME INTERFACE (STREAMLIT ORCHESTRATION)
# =====================================================================
if "insurance_agent_messages" not in st.session_state:
    st.session_state.insurance_agent_messages = [
        {"role": "assistant", "content": "Welcome to the Cyber Insurance Pricing Engine. Please provide company details to begin risk underwriting."}
    ]

if "quotation_agent" not in st.session_state:
    st.session_state.quotation_agent = create_quotation_agent()

# Display Session History Elements
for msg in st.session_state.insurance_agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Collect and process current iterations
if user_query := st.chat_input("Describe your corporate structure or request premium matrices..."):
    st.session_state.insurance_agent_messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        try:
            agent_response = st.session_state.quotation_agent.run(user_query)
            output_text = agent_response.content if hasattr(agent_response, 'content') else str(agent_response)
            response_placeholder.markdown(output_text)
            st.session_state.insurance_agent_messages.append({"role": "assistant", "content": output_text})
        except Exception as e:
            response_placeholder.error(f"Processing Error Encountered: {e}")
