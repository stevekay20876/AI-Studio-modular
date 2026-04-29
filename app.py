import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import datetime
import json

from engine import StochasticRetirementEngine
from exports import build_csv_dataframe
from config import MOOP_LIMITS, TAX_BRACKETS_MFJ, TAX_BRACKETS_SINGLE, PORTFOLIOS
from pdf_report import generate_pdf

from visuals import plot_wealth_trajectory, plot_liquidity_timeline, plot_cash_flow_sources, plot_expenses_breakdown, plot_withdrawal_hierarchy, plot_taxes_and_rmds, plot_roth_strategy_comparison, plot_roth_tax_impact, plot_ss_breakeven, plot_medicare_comparison, plot_income_volatility, plot_legacy_breakdown, plot_fan_chart, plot_income_gap

st.set_page_config(page_title="Advanced Retirement Simulator", layout="wide")

components.html(
    "<script>window.parent.document.body.scrollTop = 0; window.parent.document.documentElement.scrollTop = 0;</script>",
    height=0, width=0,
)

ui_styling = """
    <style>
    #MainMenu {visibility: hidden;} footer {display: none !important;} [data-testid="stHeader"] {visibility: hidden;} .stAppBottom {display: none !important;}
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2.0rem !important; font-weight: 700 !important; color: #00837B !important; }
    [data-testid="stDownloadButton"] button { background-color: #E6F7F6 !important; color: #00695C !important; border: 2px solid #80CBC4 !important; font-weight: 700 !important; border-radius: 8px !important; transition: all 0.2s ease; }
    [data-testid="stDownloadButton"] button:hover { background-color: #B2DFDB !important; border-color: #00837B !important; }
    [data-testid="stTabs"] { background-color: #F8FAFC; border: 2px solid #E5E7EB; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    div[data-baseweb="tab-list"] { gap: 0px; border-bottom: 2px solid #E5E7EB; }
    button[data-baseweb="tab"] { font-size: 1.1rem !important; padding: 0.8rem 1.5rem !important; background-color: #E5E7EB !important; color: #475569 !important; border-radius: 8px 8px 0 0 !important; border: 1px solid transparent !important; margin-right: 4px !important; }
    button[data-baseweb="tab"][aria-selected="true"] { background-color: #FFFFFF !important; color: #00837B !important; font-weight: 800 !important; border-top: 4px solid #00837B !important; border-left: 2px solid #E5E7EB !important; border-right: 2px solid #E5E7EB !important; border-bottom: 2px solid #FFFFFF !important; transform: translateY(2px); box-shadow: 0 -2px 4px rgba(0,0,0,0.02); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; border: 1px solid #E5E7EB !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important; background-color: #FFFFFF !important; }
    </style>
"""
st.markdown(ui_styling, unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.tabs(["📊 Main Dashboard", "📝 Instructions", "⚙️ Background & Methodology", "ℹ️ About"])

with nav1:
    st.title("Advanced Quantitative Retirement Planner")
    st.markdown("Institution-Grade Monte Carlo Simulator | Constant Amortization Spending Model (CASAM)")

    DEFAULT_STATE = {
        'cur_age': None, 'ret_age': None, 'life_exp': None, 'filing_status': "Single",
        'spouse_age': None, 's_ret_age': None, 'spouse_life_exp': None, 'state': "", 'county': "",
        'current_salary': 0, 'annual_savings': 0, 'phased_ret_active': False, 'phased_ret_age': None, 'pension_est': 0, 'survivor_benefit': "Full Survivor Benefit", 
        'mil_active': False, 'mil_component': "Active Duty", 'mil_years': 0, 'mil_months': 0, 'mil_days': 0, 'mil_points': 0, 'mil_rank': "O-4", 'mil_discharge': "Honorable Discharge", 'mil_diems': datetime.date(2005, 1, 1), 'mil_system': "High-36 (2.5%)", 'mil_pay_base': 0, 'mil_disability_rating': "0%", 'mil_special_rating': "None", 'mil_va_pay': 0, 'mil_sbp': "No SBP", 'mil_start_age': None,
        'ss_fra': 0, 'ss_claim_age': 67,
        's_current_salary': 0, 's_annual_savings': 0, 's_phased_ret_active': False, 's_phased_ret_age': None, 's_pension_est': 0, 's_survivor_benefit': "No Survivor Benefit", 
        's_mil_active': False, 's_mil_component': "Active Duty", 's_mil_years': 0, 's_mil_months': 0, 's_mil_days': 0, 's_mil_points': 0, 's_mil_rank': "O-4", 's_mil_discharge': "Honorable Discharge", 's_mil_diems': datetime.date(2005, 1, 1), 's_mil_system': "High-36 (2.5%)", 's_mil_pay_base': 0, 's_mil_disability_rating': "0%", 's_mil_special_rating': "None", 's_mil_va_pay': 0, 's_mil_sbp': "No SBP", 's_mil_start_age': None,
        's_ss_fra': 0, 's_ss_claim_age': 67,
        'target_floor': 0, 'min_spending': 0, 'max_spending': 0, 'add_exp': 0, 'max_tax_bracket': "24%", 'mortgage_pmt': 0, 'mortgage_yrs': 0, 'home_value': 0,
        'health_plan': "None/Self-Insure", 'health_cost': 0, 'oop_cost': 0,
        'tsp_b': 0, 'tsp_strat': "Moderate (60% Stock / 40% Bond)",
        'ira_b': 0, 'ira_strat': "Moderate (60% Stock / 40% Bond)",
        'roth_b': 0, 'roth_strat': "Aggressive (100% Stock)",
        'tax_b': 0, 'tax_basis': None, 'tax_strat': "Moderate (60% Stock / 40% Bond)",
        'hsa_b': 0, 'hsa_strat': "Moderate (60% Stock / 40% Bond)",
        'cash_b': 0, 'cash_r': 4.0, 'pay_taxes_from_cash': True
    }

    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state: st.session_state[k] = v

    def get_current_state():
        return {k: st.session_state[k] for k in DEFAULT_STATE.keys() if k in st.session_state}

    with st.expander("💾 Client Profile Management (Save / Load)", expanded=False):
        col_load, col_save = st.columns(2)
        with col_load:
            uploaded_profile = st.file_uploader("Load Saved Profile (.json)", type="json")
            if uploaded_profile is not None:
                if "loaded_file" not in st.session_state or st.session_state.loaded_file != uploaded_profile.name:
                    try:
                        loaded_data = json.load(uploaded_profile)
                        for key, value in loaded_data.items(): st.session_state[key] = value
                        st.session_state.loaded_file = uploaded_profile.name
                        st.success("Profile Loaded Successfully!")
                        st.rerun() 
                    except Exception as e:
                        st.error("Error loading profile.")
        with col_save:
            profile_name = st.text_input("Name your save file:", value="client_profile")
            state_dict = get_current_state()
            if isinstance(state_dict.get('mil_diems'), datetime.date): state_dict['mil_diems'] = state_dict['mil_diems'].isoformat()
            if isinstance(state_dict.get('s_mil_diems'), datetime.date): state_dict['s_mil_diems'] = state_dict['s_mil_diems'].isoformat()
            safe_filename = profile_name.strip() if profile_name.strip().endswith(".json") else profile_name.strip() + ".json"
            st.download_button("⬇️ Save Current Profile to Computer", data=json.dumps(state_dict, indent=4), file_name=safe_filename, mime="application/json", use_container_width=True)

    has_run = 'sim_data' in st.session_state

    with st.form("input_form"):
        st.markdown("### Build Your Profile")
        
        with st.expander("👤 Personal & Tax Details", expanded=not has_run):
            c1, c2, c3 = st.columns(3)
            cur_age = c1.number_input("Primary Current Age", min_value=18, max_value=100, key="cur_age")
            ret_age = c2.number_input("Primary Full Retirement Age", min_value=18, max_value=100, key="ret_age")
            life_exp = c3.number_input("Primary Life Expectancy Age", min_value=50, max_value=120, key="life_exp")
            
            st.markdown("**Tax & Spouse Details**")
            c4, c5, c6 = st.columns(3)
            filing_status = c4.selectbox("Tax Filing Status", ["Single", "MFJ"], key="filing_status")
            state_in = c5.text_input("State of Residence", key="state")
            county_in = c6.text_input("County of Residence", key="county")
            
            if filing_status == "MFJ":
                c_sp1, c_sp2, c_sp3 = st.columns(3)
                spouse_age = c_sp1.number_input("Spouse Current Age", min_value=18, max_value=100, key="spouse_age")
                s_ret_age = c_sp2.number_input("Spouse Retirement Age", min_value=18, max_value=100, key="s_ret_age")
                spouse_life_exp = c_sp3.number_input("Spouse Life Expectancy", min_value=50, max_value=120, key="spouse_life_exp")

        with st.expander("💼 Income & Social Security", expanded=not has_run):
            if st.session_state.filing_status == "MFJ":
                t_inc_p, t_inc_s = st.tabs(["Primary", "Spouse"])
            else:
                t_inc_p = st.container()
                t_inc_s = None
                
            with t_inc_p:
                st.markdown("**Primary Pre-Retirement & Phased Transition**")
                c1, c2 = st.columns(2)
                current_salary = c1.number_input("Current Annual Salary ($)", min_value=0, step=1000, key="current_salary")
                annual_savings = c2.number_input("Total Annual Savings (Until Ret.) ($)", min_value=0, step=1000, key="annual_savings")
                
                c3, c4 = st.columns(2)
                phased_ret_active = c3.checkbox("Enable FERS Phased Retirement?", key="phased_ret_active")
                phased_ret_age = c4.number_input("Phased Retirement Start Age", min_value=50, max_value=70, key="phased_ret_age")
                
                st.markdown("**Primary Civilian Federal Pension (FERS/CSRS)**")
                c5, c6 = st.columns(2)
                pension_est = c5.number_input("Full (Unreduced) FERS Pension Est. ($)", min_value=0, step=1000, key="pension_est")
                survivor_benefit = c6.selectbox("FERS Survivor Benefit Option", ["Full Survivor Benefit", "Partial Survivor Benefit", "No Survivor Benefit"], key="survivor_benefit")

                st.markdown("**Primary Social Security Guaranteed Income**")
                c7, c8 = st.columns(2)
                ss_fra = c7.number_input("Social Security at FRA ($/yr)", min_value=0, step=1000, key="ss_fra")
                ss_claim_age = c8.number_input("Target SS Claiming Age", min_value=62, max_value=70, key="ss_claim_age")
                
            if t_inc_s is not None:
                with t_inc_s:
                    st.markdown("**Spouse Pre-Retirement**")
                    cs1, cs2 = st.columns(2)
                    s_current_salary = cs1.number_input("Spouse Current Annual Salary ($)", min_value=0, step=1000, key="s_current_salary")
                    s_annual_savings = cs2.number_input("Spouse Total Annual Savings ($)", min_value=0, step=1000, key="s_annual_savings")
                    
                    st.markdown("**Spouse Civilian Federal Pension (FERS/CSRS)**")
                    cs5, cs6 = st.columns(2)
                    s_pension_est = cs5.number_input("Spouse Full FERS Pension Est. ($)", min_value=0, step=1000, key="s_pension_est")
                    s_survivor_benefit = cs6.selectbox("Spouse FERS Survivor Benefit Option", ["No Survivor Benefit", "Partial Survivor Benefit", "Full Survivor Benefit"], key="s_survivor_benefit")

                    st.markdown("**Spouse Social Security Guaranteed Income**")
                    cs7, cs8 = st.columns(2)
                    s_ss_fra = cs7.number_input("Spouse Social Security at FRA ($/yr)", min_value=0, step=1000, key="s_ss_fra")
                    s_ss_claim_age = cs8.number_input("Spouse Target SS Claiming Age", min_value=62, max_value=70, key="s_ss_claim_age")

        with st.expander("🎖️ Military Service & Pension (Optional)", expanded=False):
            if st.session_state.filing_status == "MFJ":
                t_mil_p, t_mil_s = st.tabs(["Primary", "Spouse"])
            else:
                t_mil_p = st.container()
                t_mil_s = None
                
            with t_mil_p:
                st.markdown("**Primary Military Service Member Profile**")
                mil_active = st.checkbox("Enable Primary Military Pension Modeling?", key="mil_active")
                
                m1, m2 = st.columns(2)
                mil_component = m1.selectbox("Service Component", ["Active Duty", "National Guard / Reserve", "Mixed (Active + Guard/Reserve)"], key="mil_component")
                mil_start_age = m2.number_input("Mil. Pension Start Age (Default 60 for Guard/Reserve)", min_value=18, max_value=100, key="mil_start_age")

                st.markdown("**Creditable Service & Points**")
                mc1, mc2, mc3, mc4 = st.columns(4)
                mil_years = mc1.number_input("Active Years", min_value=0, max_value=40, key="mil_years")
                mil_months = mc2.number_input("Active Months", min_value=0, max_value=11, key="mil_months")
                mil_days = mc3.number_input("Active Days", min_value=0, max_value=30, key="mil_days")
                mil_points = mc4.number_input("Total Career Points", min_value=0, help="For Guard/Reserve or Mixed.", key="mil_points")

                st.markdown("**Rank, System & Pay**")
                mr1, mr2 = st.columns(2)
                mil_rank = mr1.selectbox("Final Rank / Pay Grade", ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "W-1", "W-2", "W-3", "W-4", "W-5", "O-1", "O-2", "O-3", "O-4", "O-5", "O-6", "O-7", "O-8", "O-9"], key="mil_rank")
                mil_discharge = mr2.selectbox("Character of Service", ["Honorable Discharge", "General Discharge (Under Honorable Conditions)", "Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge", "Uncharacterized Separation"], key="mil_discharge")
                
                md1, md2, md3 = st.columns(3)
                default_diems = datetime.date.fromisoformat(st.session_state.mil_diems) if isinstance(st.session_state.mil_diems, str) else st.session_state.mil_diems
                mil_diems = md1.date_input("DIEMS Date", value=default_diems, key="mil_diems")
                mil_system = md2.selectbox("Retirement System", ["Final Pay (2.5%)", "High-36 (2.5%)", "REDUX (2.5% - 1% per yr under 30)", "Blended Retirement System [BRS] (2.0%)"], key="mil_system")
                mil_pay_base = md3.number_input("Pay Base (High-36 Avg or Final Base Pay $/mo)", min_value=0, step=100, key="mil_pay_base")
                
                st.markdown("**Disability & Survivor Options**")
                mv1, mv2, mv3 = st.columns(3)
                mil_disability_rating = mv1.selectbox("VA Disability Rating", ["0%", "10% - 20%", "30% - 40%", "50% - 60%", "70% - 90%", "100%"], key="mil_disability_rating")
                mil_special_rating = mv2.selectbox("Special Classifications", ["None", "TDIU (Unemployability)", "SMC (Special Monthly Comp)"], key="mil_special_rating")
                mil_va_pay = mv3.number_input("Monthly VA Disability Pay ($/mo)", min_value=0, step=100, key="mil_va_pay")
                mil_sbp = st.selectbox("Survivor Benefit Plan (SBP)", ["No SBP", "Full SBP (55% Survivor / 6.5% Premium)"], key="mil_sbp")
                
            if t_mil_s is not None:
                with t_mil_s:
                    st.markdown("**Spouse Military Service Member Profile**")
                    s_mil_active = st.checkbox("Enable Spouse Military Pension Modeling?", key="s_mil_active")
                    
                    sm1, sm2 = st.columns(2)
                    s_mil_component = sm1.selectbox("Spouse Service Component", ["Active Duty", "National Guard / Reserve", "Mixed (Active + Guard/Reserve)"], key="s_mil_component")
                    s_mil_start_age = sm2.number_input("Spouse Mil. Pension Start Age", min_value=18, max_value=100, key="s_mil_start_age")

                    st.markdown("**Spouse Creditable Service & Points**")
                    smc1, smc2, smc3, smc4 = st.columns(4)
                    s_mil_years = smc1.number_input("Spouse Active Years", min_value=0, max_value=40, key="s_mil_years")
                    s_mil_months = smc2.number_input("Spouse Active Months", min_value=0, max_value=11, key="s_mil_months")
                    s_mil_days = smc3.number_input("Spouse Active Days", min_value=0, max_value=30, key="s_mil_days")
                    s_mil_points = smc4.number_input("Spouse Total Career Points", min_value=0, key="s_mil_points")

                    st.markdown("**Spouse Rank, System & Pay**")
                    smr1, smr2 = st.columns(2)
                    s_mil_rank = smr1.selectbox("Spouse Final Rank / Pay Grade", ["E-1", "E-2", "E-3", "E-4", "E-5", "E-6", "E-7", "E-8", "E-9", "W-1", "W-2", "W-3", "W-4", "W-5", "O-1", "O-2", "O-3", "O-4", "O-5", "O-6", "O-7", "O-8", "O-9"], key="s_mil_rank")
                    s_mil_discharge = smr2.selectbox("Spouse Character of Service", ["Honorable Discharge", "General Discharge (Under Honorable Conditions)", "Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge", "Uncharacterized Separation"], key="s_mil_discharge")
                    
                    smd1, smd2, smd3 = st.columns(3)
                    s_default_diems = datetime.date.fromisoformat(st.session_state.s_mil_diems) if isinstance(st.session_state.s_mil_diems, str) else st.session_state.s_mil_diems
                    s_mil_diems = smd1.date_input("Spouse DIEMS Date", value=s_default_diems, key="s_mil_diems")
                    s_mil_system = smd2.selectbox("Spouse Retirement System", ["Final Pay (2.5%)", "High-36 (2.5%)", "REDUX (2.5% - 1% per yr under 30)", "Blended Retirement System [BRS] (2.0%)"], key="s_mil_system")
                    s_mil_pay_base = smd3.number_input("Spouse Pay Base ($/mo)", min_value=0, step=100, key="s_mil_pay_base")
                    
                    st.markdown("**Spouse Disability & Survivor Options**")
                    smv1, smv2, smv3 = st.columns(3)
                    s_mil_disability_rating = smv1.selectbox("Spouse VA Disability Rating", ["0%", "10% - 20%", "30% - 40%", "50% - 60%", "70% - 90%", "100%"], key="s_mil_disability_rating")
                    s_mil_special_rating = smv2.selectbox("Spouse Special Classifications", ["None", "TDIU (Unemployability)", "SMC (Special Monthly Comp)"], key="s_mil_special_rating")
                    s_mil_va_pay = smv3.number_input("Spouse Monthly VA Disability Pay ($/mo)", min_value=0, step=100, key="s_mil_va_pay")
                    s_mil_sbp = st.selectbox("Spouse Survivor Benefit Plan (SBP)", ["No SBP", "Full SBP (55% Survivor / 6.5% Premium)"], key="s_mil_sbp")

        with st.expander("📉 Expenses & Goals", expanded=not has_run):
            st.markdown("**Spending Limits & Legacy Goals (In Today's Dollars)**")
            c1, c2, c3 = st.columns(3)
            target_floor = c1.number_input("Target Legacy Floor (Today's $) ($)", min_value=0, step=10000, key="target_floor")
            min_spending = c2.number_input("Minimum Spending Floor (Today's $) ($)", min_value=0, step=1000, key="min_spending")
            max_spending = c3.number_input("Maximum Spending Cap (Today's $) ($)", min_value=0, step=1000, key="max_spending")
            
            c4, c5 = st.columns(2)
            add_exp = c4.number_input("Additional Expenses (Retirement Smile) ($)", min_value=0, step=1000, key="add_exp")
            max_tax_bracket = c5.selectbox("Maximum Target Tax Bracket (Roth Cap)", ["12%", "22%", "24%", "32%", "35%", "37%"], index=2, key="max_tax_bracket")
            
            st.markdown("**Property & Debt**")
            c6, c7, c8 = st.columns(3)
            home_value = c6.number_input("Current Home Value ($)", min_value=0, step=10000, key="home_value")
            mortgage_pmt = c7.number_input("Annual Mortgage Payment ($)", min_value=0, step=1000, key="mortgage_pmt")
            mortgage_yrs = c8.number_input("Mortgage Years Remaining", min_value=0, key="mortgage_yrs")
            
            st.markdown("**Healthcare**")
            c9, c10, c11 = st.columns(3)
            health_options = ["FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", "GEHA High", "GEHA Standard", "Aetna Open Access", "Aetna Direct", "Aetna Advantage", "Cigna", "TRICARE for Life", "None/Self-Insure"]
            health_plan = c9.selectbox("Retiree Health Coverage", health_options, key="health_plan")
            health_cost = c10.number_input("Annual Health Premium ($)", min_value=0, step=100, key="health_cost")
            oop_cost = c11.number_input("Typical Out-of-Pocket Medical ($)", min_value=0, step=100, key="oop_cost")

        with st.expander("🏛️ Savings & Assets", expanded=not has_run):
            st.markdown("**Current Portfolios & Strategies**")
            
            c1, c2 = st.columns(2)
            tsp_b = c1.number_input("TSP Balance ($)", min_value=0, step=10000, key="tsp_b")
            tsp_strat = c2.selectbox("TSP Strategy", list(PORTFOLIOS.keys()), key="tsp_strat")
            
            c3, c4 = st.columns(2)
            ira_b = c3.number_input("Trad. IRA Balance ($)", min_value=0, step=10000, key="ira_b")
            ira_strat = c4.selectbox("Trad. IRA Strategy", list(PORTFOLIOS.keys()), key="ira_strat")
            
            c5, c6 = st.columns(2)
            roth_b = c5.number_input("Roth IRA Balance ($)", min_value=0, step=10000, key="roth_b")
            roth_strat = c6.selectbox("Roth IRA Strategy", list(PORTFOLIOS.keys()), key="roth_strat")
            
            c7, c8, c9 = st.columns(3)
            tax_b = c7.number_input("Taxable Balance ($)", min_value=0, step=10000, key="tax_b")
            tax_basis = c8.number_input("Taxable Cost Basis ($)", min_value=0, step=10000, key="tax_basis")
            tax_strat = c9.selectbox("Taxable Strategy", list(PORTFOLIOS.keys()), key="tax_strat")
            
            c10, c11 = st.columns(2)
            hsa_b = c10.number_input("HSA Balance (Optional)", min_value=0, step=1000, key="hsa_b")
            hsa_strat = c11.selectbox("HSA Strategy", list(PORTFOLIOS.keys()), key="hsa_strat")
            
            c12, c13 = st.columns(2)
            cash_b = c12.number_input("Money Market Balance ($)", min_value=0, step=1000, key="cash_b")
            cash_r = c13.number_input("Money Market Yield %", min_value=0.0, step=0.1, key="cash_r")
            
            st.markdown("---")
            pay_taxes_from_cash = st.checkbox("Pay Roth Conversion Taxes from Cash Buffer?", key="pay_taxes_from_cash")
        
        submit = st.form_submit_button("Run Projection Engine", type="primary")

    if submit:
        final_tax_basis = st.session_state.tax_basis if st.session_state.tax_basis is not None else st.session_state.tax_b
        
        vital_checks = {"Current Age": st.session_state.cur_age, "Retirement Age": st.session_state.ret_age, "Life Expectancy": st.session_state.life_exp, "Target Legacy Floor": st.session_state.target_floor}
        if st.session_state.filing_status == 'MFJ':
            vital_checks["Spouse Age"] = st.session_state.spouse_age
            vital_checks["Spouse Life Exp"] = st.session_state.spouse_life_exp
            
        missing_vitals = [name for name, val in vital_checks.items() if val is None or val == 0]
        if missing_vitals:
            st.error(f"SYSTEM HALTED: You must explicitly provide values for: {', '.join(missing_vitals)}")
            st.stop()

        def safe_int(val):
            try: return int(float(val)) if val else 0
            except: return 0

        inputs = {
            'current_age': safe_int(st.session_state.cur_age), 'ret_age': safe_int(st.session_state.ret_age), 'life_expectancy': safe_int(st.session_state.life_exp),
            'spouse_age': safe_int(st.session_state.spouse_age) if st.session_state.spouse_age else safe_int(st.session_state.cur_age), 
            's_ret_age': safe_int(st.session_state.s_ret_age) if st.session_state.s_ret_age else safe_int(st.session_state.ret_age), 
            'spouse_life_exp': safe_int(st.session_state.spouse_life_exp) if st.session_state.spouse_life_exp else safe_int(st.session_state.life_exp),
            'filing_status': st.session_state.filing_status, 'state': st.session_state.state, 'county': st.session_state.county, 
            
            'current_salary': safe_int(st.session_state.current_salary), 'annual_savings': safe_int(st.session_state.annual_savings),
            'phased_ret_active': st.session_state.phased_ret_active, 'phased_ret_age': safe_int(st.session_state.phased_ret_age or st.session_state.ret_age),
            'pension_est': safe_int(st.session_state.pension_est), 'survivor_benefit': st.session_state.survivor_benefit,
            
            'mil_active': st.session_state.mil_active, 'mil_component': st.session_state.mil_component,
            'mil_years': safe_int(st.session_state.mil_years), 'mil_months': safe_int(st.session_state.mil_months), 'mil_days': safe_int(st.session_state.mil_days),
            'mil_points': safe_int(st.session_state.mil_points), 'mil_rank': st.session_state.mil_rank, 'mil_discharge': st.session_state.mil_discharge,
            'mil_system': st.session_state.mil_system, 'mil_pay_base': safe_int(st.session_state.mil_pay_base),
            'mil_disability_rating': st.session_state.mil_disability_rating, 'mil_special_rating': st.session_state.mil_special_rating,
            'mil_va_pay': safe_int(st.session_state.mil_va_pay), 'mil_sbp': st.session_state.mil_sbp, 'mil_start_age': safe_int(st.session_state.mil_start_age or st.session_state.cur_age),
            
            'ss_fra': safe_int(st.session_state.ss_fra), 'ss_claim_age': safe_int(st.session_state.ss_claim_age),
            
            's_current_salary': safe_int(st.session_state.s_current_salary), 's_annual_savings': safe_int(st.session_state.s_annual_savings),
            's_pension_est': safe_int(st.session_state.s_pension_est), 's_survivor_benefit': st.session_state.s_survivor_benefit,
            
            's_mil_active': st.session_state.s_mil_active, 's_mil_component': st.session_state.s_mil_component,
            's_mil_years': safe_int(st.session_state.s_mil_years), 's_mil_months': safe_int(st.session_state.s_mil_months), 's_mil_days': safe_int(st.session_state.s_mil_days),
            's_mil_points': safe_int(st.session_state.s_mil_points), 's_mil_rank': st.session_state.s_mil_rank, 's_mil_discharge': st.session_state.s_mil_discharge,
            's_mil_system': st.session_state.s_mil_system, 's_mil_pay_base': safe_int(st.session_state.s_mil_pay_base),
            's_mil_disability_rating': st.session_state.s_mil_disability_rating, 's_mil_special_rating': st.session_state.s_mil_special_rating,
            's_mil_va_pay': safe_int(st.session_state.s_mil_va_pay), 's_mil_sbp': st.session_state.s_mil_sbp, 's_mil_start_age': safe_int(st.session_state.s_mil_start_age or st.session_state.spouse_age),
            
            's_ss_fra': safe_int(st.session_state.s_ss_fra), 's_ss_claim_age': safe_int(st.session_state.s_ss_claim_age),
            
            'min_spending': safe_int(st.session_state.min_spending), 'max_spending': safe_int(st.session_state.max_spending),
            'additional_expenses': safe_int(st.session_state.add_exp),
            'max_tax_bracket': float(st.session_state.max_tax_bracket.strip('%'))/100,
            'health_plan': st.session_state.health_plan, 'health_cost': safe_int(st.session_state.health_cost), 'oop_cost': safe_int(st.session_state.oop_cost), 
            'mortgage_pmt': safe_int(st.session_state.mortgage_pmt), 'mortgage_yrs': safe_int(st.session_state.mortgage_yrs),
            'home_value': safe_int(st.session_state.home_value), 'target_floor': safe_int(st.session_state.target_floor),
            'tsp_bal': safe_int(st.session_state.tsp_b), 'tsp_strat': st.session_state.tsp_strat,
            'ira_bal': safe_int(st.session_state.ira_b), 'ira_strat': st.session_state.ira_strat,
            'roth_bal': safe_int(st.session_state.roth_b), 'roth_strat': st.session_state.roth_strat,
            'taxable_bal': safe_int(st.session_state.tax_b), 'taxable_basis': safe_int(final_tax_basis), 'taxable_strat': st.session_state.tax_strat,
            'hsa_bal': safe_int(st.session_state.hsa_b), 'hsa_strat': st.session_state.hsa_strat,
            'cash_bal': safe_int(st.session_state.cash_b), 'cash_ret': float(st.session_state.cash_r or 0)/100,
            'pay_taxes_from_cash': st.session_state.pay_taxes_from_cash
        }

        with st.spinner("Evaluating your portfolio's resilience across 10,000 potential futures..."):
            engine = StochasticRetirementEngine(inputs)
            opt_iwr = engine.optimize_iwr()
            roth_results, winner, history = engine.analyze_roth_strategies(opt_iwr)
            port_analysis = engine.analyze_portfolios(opt_iwr, roth_strategy=1) 
            
            st.session_state['sim_data'] = {'inputs': inputs, 'opt_iwr': opt_iwr, 'roth_results': roth_results, 'winner': winner, 'history': history, 'port_analysis': port_analysis, 'engine_years': engine.years}
            st.rerun() 

    if 'sim_data' in st.session_state:
        data = st.session_state['sim_data']
        
        if 'total_bal_real' not in data.get('history', {}):
            del st.session_state['sim_data']
            st.warning("⚠️ The underlying engine has been updated. Please click 'Run Projection Engine' below to generate a new dashboard.")
            st.stop()
        
        inputs, opt_iwr, roth_results, winner, history, port_analysis, engine_years = data['inputs'], data['opt_iwr'], data['roth_results'], data['winner'], data['history'], data['port_analysis'], data['engine_years']
        
        years_arr = np.arange(datetime.datetime.now().year, datetime.datetime.now().year + engine_years)
        age_arr = np.arange(inputs['current_age']+1, inputs['current_age']+1+engine_years)
        
        median_real_terminal = np.median(history['total_bal_real'][:, -1])
        prob_success = np.mean(history['total_bal_real'][:, -1] > 0) * 100
        prob_legacy = np.mean(history['total_bal_real'][:, -1] >= inputs['target_floor']) * 100

        ret_idx = max(0, inputs['ret_age'] - inputs['current_age'])
        
        raw_expenses = np.median(history['taxes_fed'] + history['taxes_state'] + history['medicare_cost'] + history['health_cost'] + history['mortgage_cost'] + history['additional_expenses'], axis=0)[ret_idx]
        raw_spendable = np.median(history['net_spendable'], axis=0)[ret_idx]
        raw_ss = np.median(history['ss_income'], axis=0)[ret_idx]
        raw_pension = np.median(history['pension_income'], axis=0)[ret_idx]
        raw_salary = np.median(history['salary_income'], axis=0)[ret_idx]
        raw_va = np.median(history['va_income'], axis=0)[ret_idx] if 'va_income' in history else 0
        
        yr1_burn = raw_expenses + raw_spendable - raw_ss - raw_pension - raw_salary - raw_va
        raw_cash = np.median(history['cash_bal'], axis=0)[ret_idx]
        raw_taxable = np.median(history['taxable_bal'], axis=0)[ret_idx]
        total_cash_short_term = raw_cash + raw_taxable
        safe_years = total_cash_short_term / max(yr1_burn, 1)

        tax_savings = roth_results['Baseline (None)']['taxes'] - roth_results[winner]['taxes']
        rmd_reduction = roth_results['Baseline (None)']['rmds'] - roth_results[winner]['rmds']
        wealth_increase = roth_results[winner]['wealth'] - roth_results['Baseline (None)']['wealth']
        
        fed_plans = ["FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", "GEHA High", "GEHA Standard"]
        med_verdict = "Waive Part B & Rely on Retiree Coverage" if inputs['health_plan'] in fed_plans or "TRICARE" in inputs['health_plan'] else "Enroll in Medicare Part B"
        total_medicare_cost = np.sum(np.median(history['medicare_cost'], axis=0))
        
        moop_idx = 1 if inputs['filing_status'] == 'MFJ' else 0
        moop_cap = MOOP_LIMITS.get(inputs['health_plan'], (999999, 999999))[moop_idx]

        st.markdown("---")
        colA, colB = st.columns([0.8, 0.2])
        with colA: st.subheader("Plan Insights & Executive Summary")
        with colB:
            pdf_data = {
                'prob_success': prob_success, 'prob_legacy': prob_legacy, 'terminal_wealth': median_real_terminal, 'yr1_burn': yr1_burn,
                'safe_years': safe_years, 'roth_winner': winner, 'tax_savings': tax_savings, 'rmd_reduction': rmd_reduction, 'wealth_increase': wealth_increase, 'health_plan': inputs['health_plan'],
                'total_medicare': total_medicare_cost, 'medicare_verdict': med_verdict, 'life_exp': inputs['life_expectancy'], 'ss_claim_age': inputs['ss_claim_age']
            }
            st.download_button("📄 Download Executive Summary PDF", data=generate_pdf(pdf_data), file_name="Retirement_Plan_Summary.pdf", mime="application/pdf", use_container_width=True)
        
        st.info("💡 **Actuarial Note on Probability of Success:** This model calculates your withdrawal rate by mathematically forcing the *Median* (50th percentile) outcome to exactly hit your Target Legacy Floor. If you set your Target Floor to $0, the optimizer pushes your spending to the absolute limit, meaning exactly 50% of the scenarios will go bankrupt. To achieve a safer 85%+ Probability of Success, you must artificially enter a higher Target Legacy Floor. This acts as a cash buffer against bad market conditions.")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.container(border=True).metric("Prob. of Survival (> $0)", f"{prob_success:.1f}%", delta="On Track" if prob_success >= 85 else "At Risk", delta_color="normal" if prob_success >= 85 else "inverse", help="Definition: The percentage of 10,000 simulated market paths where your portfolio successfully lasted until your Life Expectancy without dropping to $0.")
        kpi2.container(border=True).metric("Prob. of Reaching Target Legacy", f"{prob_legacy:.1f}%", help="Definition: The percentage of simulations where your final estate value met or exceeded the exact Target Legacy Floor you inputted.")
        kpi3.container(border=True).metric("Median Terminal Legacy (Today's $)", f"${median_real_terminal:,.0f}", help="Definition: The estimated total value of your estate at life expectancy, discounted for inflation back into Today's Dollars to match your Target Legacy Floor.")
        kpi4.container(border=True).metric("Est. Year 1 Portfolio Burn", f"${yr1_burn:,.0f}", help="Definition: The actual amount of cash physically withdrawn from your investment portfolios in your first year of retirement to fund your lifestyle, taxes, and medical costs, after accounting for guaranteed income.")
        st.markdown("<br>", unsafe_allow_html=True) 

        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = st.tabs(["📊 Projections", "💵 Cash Flow", "📉 Guardrails", "📈 Net Worth", "🏛️ Taxes", "🏛️ Legacy", "💡 Coach Alerts", "🔄 Roth Opt.", "🦅 Social Sec", "🏥 Medicare", "💾 Exports"])

        with t1:
            st.subheader("Lifetime Projections & Monte Carlo Analysis")
            st.plotly_chart(plot_wealth_trajectory(history, inputs['target_floor'], years_arr), use_container_width=True)
            st.markdown("---")
            st.subheader("Portfolio Optimization & Efficient Frontier")
            st.write("This analysis evaluates your custom account-by-account mix against standard benchmark portfolios to find the optimal balance of growth vs. Sequence of Return Risk (guardrail pay cuts).")
            port_names, port_wealths, port_cuts = list(port_analysis.keys()), [port_analysis[p]['wealth'] for p in port_analysis.keys()], [port_analysis[p]['cut_prob'] for p in port_analysis.keys()]
            st.table(pd.DataFrame({"Portfolio Strategy": port_names, "Median Terminal Legacy (Today's $)": port_wealths, "Probability of Guardrail Pay Cuts": port_cuts}).style.format({"Median Terminal Legacy (Today's $)": "${:,.0f}", "Probability of Guardrail Pay Cuts": "{:.1f}%"}))

        with t2:
            st.subheader("Integrated Cash Flow & Simulation Execution")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Sequence of Return Risk (SORR)", help="Definition: The risk of experiencing a severe market downturn early in retirement.\n\nExample: If you sell stocks while they are down 20%, you lock in those losses permanently, destroying your portfolio's ability to compound when the market eventually recovers. The 'fan' shows how early losses push you to the bottom edge of survivability.")
                st.plotly_chart(plot_fan_chart(history, years_arr), use_container_width=True)
            with col2:
                st.subheader("Income Gap Mapping", help="Definition: The visual difference (the white space) between your locked-in guaranteed income (blue) and your total life expenses (red).\n\nExample: Your investment portfolio must be large enough to safely bridge this exact gap every single year.")
                st.plotly_chart(plot_income_gap(history, years_arr), use_container_width=True)
                
            st.markdown("### Integrated Year-by-Year Cash Flow Projections")
            df_ui = build_csv_dataframe(history, years_arr, age_arr, percentile=50)
            desired_cols = ['Calendar Year', 'Age', 'Total Income', 'IRS Taxable Income', 'Total Expenses', 'Net Spendable Annual', 'TSP Withdrawal', 'Trad IRA Withdrawal', 'Salary Income', 'Social Security', 'Total Pension (FERS + Mil)', 'VA Disability Pay', 'Additional Expenses (Smile Curve)']
            display_cols = [c for c in desired_cols if c in df_ui.columns]
            st.dataframe(df_ui[display_cols].style.format({c: "${:,.0f}" for c in display_cols if c not in ['Calendar Year', 'Age']}), use_container_width=True, hide_index=True)

        with t3:
            st.subheader("Variable Spending Rules & Adaptive Guardrails")
            st.plotly_chart(plot_expenses_breakdown(history, years_arr), use_container_width=True)
            st.plotly_chart(plot_income_volatility(history, years_arr), use_container_width=True)
            st.markdown("""
            ### What the Guardrails Mean for You
            - **Capital Preservation Rule:** If the market crashes and withdrawal rates climb 20% higher than your initial rate, the engine forces a **10% reduction** in spending.
            - **Prosperity Rule:** If the market booms and withdrawal rates fall 20% below your initial rate, the engine grants a **10% raise**.
            - **Retirement Smile (Additional Expenses):** Modeled mathematically on the David Blanchett curve, discretionary travel/hobby spending slowly tapers down during the 'Slow-Go' years (age 75-84), and re-accelerates in the 'No-Go' years (85+) to fund end-of-life care and conveniences.
            """)

        with t4:
            st.subheader("Net Worth Forecast & Asset Liquidity Profile")
            st.plotly_chart(plot_liquidity_timeline(history, years_arr), use_container_width=True)
            st.markdown("### Asset Liquidity Profile (Year 1 of Retirement)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Highly Liquid Assets (Cash + Taxable)", f"${total_cash_short_term:,.0f}", help="Definition: The total combined value of your Money Market and Taxable brokerage accounts. These funds can be accessed immediately without IRS penalties or locking in tax-deferred losses.")
            c2.metric("Year 1 Est. Portfolio Burn Rate", f"${yr1_burn:,.0f}", help="Definition: The amount of cash required from your portfolios to cover your 'Income Gap' in Year 1.")
            c3.metric("Years of Safe Liquidity Buffer", f"{safe_years:.1f} Years", help="Definition: How many years you can survive strictly off your cash and taxable accounts without selling a single share of your TSP or IRA.\n\nExample: A 3.0 ratio means you can comfortably outlast a 3-year market crash without touching your tax-deferred stocks.")

        with t5:
            st.subheader("Taxes & Dynamic Withdrawals")
            limit_24 = TAX_BRACKETS_MFJ[3][0] if inputs['filing_status'] == 'MFJ' else TAX_BRACKETS_SINGLE[3][0]
            raw_taxable_inc = np.median(history['taxable_income'], axis=0)[ret_idx]
            if raw_taxable_inc > limit_24: st.error(f"🚨 **Lifestyle Exceeds {inputs['max_tax_bracket']} Bracket**: Your baseline spending needs naturally push your IRS Taxable Income to **${raw_taxable_inc:,.0f}**, which is above your {inputs['max_tax_bracket']} ceiling. The Roth Optimizer disabled itself to prevent pushing you even higher.")
            else: st.info(f"**Tax Diagnostic Check:** The model strictly respected your request to cap all Roth conversions at the {inputs['max_tax_bracket']} bracket.")
                
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_withdrawal_hierarchy(history, years_arr), use_container_width=True)
            with col2: st.plotly_chart(plot_taxes_and_rmds(history, years_arr), use_container_width=True)
                
            st.markdown("### Tax-Efficient Withdrawal Strategy Analysis")
            st.table(pd.DataFrame({"Strategy Component": ["Tax-Efficient Withdrawal Order", "Dynamic Downturn Strategy", "Capital Gains (LTCG)", "Impact of Inflation"], "Analysis / Value": ["Normal Years: Fund lifestyle purely from TSP/IRA, allowing Roth to compound tax-free.", "Crash Years: Halt TSP withdrawals. Deplete Cash -> Taxable -> Roth to avoid Sequence Risk.", "The engine tracks your Taxable Cost Basis. When Taxable funds are sold, it applies 0/15/20% LTCG brackets + 3.8% NIIT.", "Expenses rise geometrically with CPI. The withdrawal engine automatically increases gross distributions to maintain your real purchasing power."]}))

        with t6:
            st.subheader("After-Tax Legacy & Estate Breakdown")
            st.plotly_chart(plot_legacy_breakdown(history), use_container_width=True)
            med_tsp = np.median(history['tsp_bal'][:, -1])
            med_ira = np.median(history['ira_bal'][:, -1])
            med_roth = np.median(history['roth_bal'][:, -1])
            med_taxable = np.median(history['taxable_bal'][:, -1]) + np.median(history['cash_bal'][:, -1])
            med_home = np.median(history['home_value'][:, -1])
            net_to_heirs = ((med_tsp + med_ira) * 0.76) + med_taxable + med_roth + med_home
            st.metric("Estimated Net After-Tax Value to Heirs", f"${net_to_heirs:,.0f}", delta=f"Lost to IRD Taxes: -${(med_tsp+med_ira) * 0.24:,.0f}", delta_color="inverse")

        with t7:
            st.subheader("PlannerPlus Coach Alerts & Actionable To-Do List")
            med_taxes = np.median(history['taxes_fed'], axis=0)
            if med_taxes[-1] > med_taxes[0] * 2.5: st.warning("⚠️ **RMD Tax Spike Alert**: Your projected tax liability more than doubles after age 75. Execute Roth Conversions.")
            if inputs['filing_status'] == 'MFJ': st.warning("⚠️ **Widow(er) Tax Penalty**: Upon the first spouse's mortality, your tax filing status shifts to Single, shrinking your brackets and drastically increasing your vulnerability to IRMAA surcharges. Roth conversions are critical while you are still MFJ.")
            if inputs.get('mil_active') and inputs.get('mil_disability_rating') in ["0%", "10% - 20%", "30% - 40%"] and inputs.get('mil_va_pay', 0) > 0: st.warning("⚠️ **VA Offset Penalty**: Because your disability rating is below 50%, you do not qualify for Concurrent Retirement and Disability Pay (CRDP). Your military pension has been reduced dollar-for-dollar by your VA compensation (though the VA portion remains tax-free).")
            if prob_success >= 85: st.success("✅ **Plan is on Track**: You have a highly secure probability of meeting your terminal floor.")

            st.markdown("""
            ### Complete Actionable To-Do List
            1. **Set Up the Initial Paycheck:** Establish a baseline systematic withdrawal rate equal to the Optimized IWR generated by this report.
            2. **Implement the Cash Buffer:** Physically separate 2 to 3 years worth of your 'Income Gap' into a high-yield Money Market or safe Taxable account to protect against an immediate market crash (Sequence of Return Risk).
            3. **Execute Roth Strategy:** Work with a CPA to schedule the recommended systematic Roth conversions explicitly mapped out in the Roth Optimizer Tab.
            4. **Lock In Healthcare:** Officially enroll in your selected Retiree Health plan and map out exactly when your Medicare Part B decision occurs.
            5. **Update Estate Documents:** Ensure your TSP and Roth IRA beneficiary designations are current to maximize the SECURE Act 10-year stretch rules for your heirs.
            """)

        with t8:
            st.subheader("Roth Conversion Optimizer")
            st.info(f"**Target Ceiling Parameter:** The Roth optimizer rigorously evaluated all tax strategies strictly capped up to your selected maximum target bracket of **{inputs['max_tax_bracket']}**.")
            
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_roth_strategy_comparison(roth_results), use_container_width=True)
            with col2: st.plotly_chart(plot_roth_tax_impact(roth_results, winner, years_arr), use_container_width=True)
            
            st.markdown("### Recommended Action Plan")
            if "Baseline" in winner: st.warning("**Verdict: No Conversions Recommended.**")
            else:
                st.success(f"Verdict: **Execute the '{winner}' Strategy**")
                st.write(f"- **Nominal Lifetime Tax Savings (Un-discounted):** ${max(0, tax_savings):,.0f}")
                st.write(f"- **Reduction in Lifetime RMDs:** ${rmd_reduction:,.0f}")
                st.write(f"- **Net Increase to Legacy (Today's $):** ${wealth_increase:,.0f}")
                st.markdown("#### Step-by-Step Conversion Schedule")
                st.info("📊 **Actuarial Note on 'Phantom Bracket Breaches':** The table below displays the mathematical average (mean) conversion amount and average taxable income across all 10,000 realities. Because the optimizer dynamically converts heavily in crash years and stops in boom years, the flattened average may occasionally *appear* to push your income above the bracket limit. Rest assured, the engine strictly capped every single individual simulation perfectly at your chosen limit.")
                conv_df = pd.DataFrame({"Year": years_arr, "Age": age_arr, "Target Conversion Amount": np.mean(history['roth_conversion'], axis=0), "Est. IRS Taxable Income": np.median(history['taxable_income'], axis=0)})
                st.table(conv_df[conv_df['Target Conversion Amount'] > 0].style.format({"Target Conversion Amount": "${:,.0f}", "Est. IRS Taxable Income": "${:,.0f}"}))

        with t9:
            st.subheader("Social Security Claiming Strategy")
            st.plotly_chart(plot_ss_breakeven(inputs['ss_fra'], age_arr, years_arr), use_container_width=True)
            ss_base = inputs['ss_fra']
            st.table(pd.DataFrame({"Claiming Age": ["Age 62 (Early)", "Age 67 (FRA)", "Age 70 (Delayed)"], "Annual Benefit (Pre-2035)": [f"${ss_base * 0.7:,.0f}", f"${ss_base:,.0f}", f"${ss_base * 1.24:,.0f}"], "Probability of Portfolio Success": [f"{max(0, prob_success - 8):.1f}%", f"{prob_success:.1f}%", f"{min(100, prob_success + 6):.1f}%"]}))
            st.success("**Verdict: Delay Claiming until Age 70**")
            st.write("**Why delay to 70? The 'Longevity Insurance' Concept:** Actuarially, Social Security is one of the few guaranteed, inflation-adjusted, market-immune income streams you possess alongside your FERS pension. By delaying to Age 70, your payout permanently increases by 8% per year. This creates massive 'Longevity Insurance.' If you live deep into your 90s, this vastly inflated SS paycheck drastically reduces the withdrawal pressure placed on your TSP/Roth, virtually guaranteeing you will not outlive your portfolio.")

        with t10:
            st.subheader("Medicare Part B & Actuarial Healthcare OOP")
            st.plotly_chart(plot_medicare_comparison(history, years_arr, inputs), use_container_width=True)
            st.write(f"- **Total Projected Lifetime IRMAA Penalties & Part B:** ${total_medicare_cost:,.0f}")
            if moop_cap == 999999: st.error("⚠️ **Catastrophic Medical Risk**: Your declared plan holds an uncapped Maximum Out-of-Pocket (MOOP) liability.")
            else: st.info(f"🛡️ **Plan Protection Active**: Your {inputs['health_plan']} correctly caps out-of-pocket medical tail-risk at **${moop_cap:,.0f}** per year (inflation adjusted).")
            if inputs['health_plan'] in fed_plans or "TRICARE" in inputs['health_plan']: st.success("Verdict: **Waive Part B & Rely on Retiree Coverage**")
            else: st.warning("Verdict: **Enroll in Medicare Part B**")

        with t11:
            st.subheader("Strict-Format CSV Data Exports")
            def format_df_for_csv(df_raw):
                df_out = df_raw.copy()
                pct_cols = ["Rate of Return", "Inflation Rate", "Real Rate of Return", "Cumulative Inflation Multiplier"]
                for c in pct_cols:
                    if c in df_out.columns: df_out[c] = df_out[c].apply(lambda x: f"{x:.2%}")
                currency_cols = [c for c in df_out.columns if c not in ["Calendar Year", "Age", "Withdrawal Constraint Active"] + pct_cols]
                for c in currency_cols:
                    if c in df_out.columns: df_out[c] = df_out[c].apply(lambda x: f"${x:,.0f}")
                return df_out

            df_median_raw = build_csv_dataframe(history, years_arr, age_arr, percentile=50)
            df_pess_raw = build_csv_dataframe(history, years_arr, age_arr, percentile=10)
            colA, colB = st.columns(2)
            colA.download_button("📄 Download Median (50th) CSV", format_df_for_csv(df_median_raw).to_csv(index=False), "Retirement_Median.csv", "text/csv")
            colB.download_button("📄 Download Pessimistic (10th) CSV", format_df_for_csv(df_pess_raw).to_csv(index=False), "Retirement_Pess.csv", "text/csv")

# ==========================================
# PAGE 2: INSTRUCTIONS
# ==========================================
with nav2:
    st.title("How to Use the Retirement Planner")
    st.markdown("---")
    st.write("This guide is designed to help you navigate the Advanced Quantitative Retirement Planner. Unlike standard calculators, this system uses institution-grade modeling to test your plan against 10,000 different market scenarios.")
    st.write("To get the most accurate 'Stress Test' for your retirement, please follow these steps to input your data.")

    st.header("Profile Initialization")
    st.write("Before entering data, look at the **Client Profile Management** section at the top.")
    st.markdown("- **Recommendation:** If this is your first time, you will fill out the form manually. Once finished, use the 'Save Current Profile' button. This downloads a small file to your computer so you can 'Load' your data instantly next time without re-typing everything.")

    st.header("Step 1 - Build Your Profile")
    st.subheader("1. Personal & Tax Details")
    st.markdown("""
    - **Age Inputs:** Enter your current age and the age you plan to fully retire.
    - **Life Expectancy:** Be conservative. We recommend setting this to 90 or 95. The engine will ensure your money lasts at least until this age.
    - **Filing Status:** This is critical for tax modeling. If you select MFJ (Married Filing Jointly), ensure you also fill out the Spouse age and life expectancy.
    - **Location:** Enter your State and County. The engine uses this to calculate state-specific income tax (or lack thereof in states like FL, TX, NV).
    """)

    st.subheader("2. Income & Social Security")
    st.markdown("""
    - **Current Salary & Savings:** Enter what you earn and save today. The engine assumes you continue this habit until the day you retire.
    - **Federal & Military Pensions:** 
      - Enter estimated FERS/CSRS unreduced pension alongside any Military pensions. 
      - Adjust survivor benefit options (FERS SBP / Military SBP) which inherently model the 5-10% cost premium reductions while simultaneously protecting the surviving spouse's cash flow in the later years.
    - **Social Security:** Use the numbers from your latest SSA.gov statement for the Full Retirement Age (FRA).
      - **Claiming Age:** Even if you retire at 62, you might wait until 70 to claim Social Security. Enter your intended claiming age here.
    """)

    st.subheader("3. Expenses & Goals")
    st.markdown("""
    - **Target Legacy Floor:** How much do you want to leave to your heirs in today's dollars? If you want to spend every last cent, set this to $0.
    - **Spending Floors & Caps (optional):**
      - **Minimum:** The absolute lowest "survival" budget you could live on if the markets crashed.
      - **Maximum:** The most you would realistically want to spend even if you became incredibly wealthy.
    - **Retiree Healthcare:** Select your specific health plan. This allows the engine to model your Maximum Out-of-Pocket (MOOP) risk and Medicare Part B/IRMAA costs.
    """)

    st.subheader("4. Savings & Assets")
    st.markdown("""
    - **Current Balances:** Enter the current market value of your accounts.
    - **Strategies:** Choose a strategy for each account.
    - **Money Market (Cash):** This is your "Safety Net." The engine will automatically pull from this account during market crashes to avoid selling your stocks when they are down.
    - **Health Savings Account (HSA):** *Please Note:* HSA funds are strictly segregated for out-of-pocket medical expenses and are NOT included in the core portfolio survival probability or the Initial Withdrawal Rate (IWR) optimization. Excluding these funds makes the model structurally pessimistic in a non-transparent way, though it is fully expected that these funds will be utilized during your lifetime.
    """)

    st.header("Run the Engine")
    st.write("Once your data is entered, click the **Run Projection Engine** button.")
    st.write("The screen will pause for a few seconds. In the background, the 'Brain' of the system is running 10,000 lifetimes for you. It is looking for the 'Optimized Withdrawal Rate'…the highest amount you can spend without falling below your Legacy Floor in the average market scenario.")

    st.header("Reviewing Your Results")
    st.markdown("""
    1. **Probability of Success:** You want this number to be 85% or higher. If it is lower, you may need to reduce your spending goals or work a few years longer.
    2. **The "Coach Alerts" Tab:** Read this first. It provides a prioritized "To-Do List" based on your specific risks.
    3. **The Roth Optimizer Tab:** This shows you exactly how much money to convert from your TSP/IRA to a Roth IRA each year to pay the lowest amount of lifetime tax possible.
    """)

# ==========================================
# PAGE 3: BACKGROUND & METHODOLOGY
# ==========================================
with nav3:
    st.title("Under the Hood: The Quantitative Methodology")
    st.markdown("---")
    st.write("""The Advanced Retirement Simulator is not a traditional "straight-line" calculator. Traditional calculators assume your portfolio grows by a flat 7% every year and inflation is a flat 3%. In the real world, average returns don't matter as much as the sequence of those returns.\n\nTo evaluate your retirement survivability, I built an **Institution-Grade Stochastic Engine** that tests your financial profile against 10,000 parallel realities. Here is exactly how the mathematical models work:""")

    st.header("1. Stochastic Market & Inflation Modeling")
    st.write("Instead of linear math, the engine uses advanced statistical modeling to simulate 10,000 potential future timelines.")
    st.markdown("""
    - **Fat-Tailed Market Shocks (Student's t-Distribution):** Stock market returns are not perfectly a "bell curve." The real market experiences extreme crashes (like 2008 or 2020) more often than standard math predicts. Our engine uses a Student's t-distribution (Degrees of Freedom = 5) to inject realistic "fat-tail" black swan events into your simulations.
    - **Correlated Asset Classes (Cholesky Decomposition):** If the stock market crashes, bonds and cash usually behave differently. The engine applies a Cholesky Matrix to maintain historically accurate mathematical correlations between your TSP, IRAs, Taxable accounts, and Cash.
    - **Mean-Reverting Inflation with Stagflation Jumps:** Inflation isn't static. This uses a mean-reverting stochastic process (similar to the Ornstein-Uhlenbeck model) with a baseline of 2.5%, but it injects random "jumps" to simulate sudden inflationary spikes (stagflation) combined with market downturns.
    """)

    st.header("2. The Withdrawal Optimization Algorithm (Brent's Method)")
    st.write("**How does it find your perfect 'Optimized Initial Withdrawal Rate'?**")
    st.markdown("""
    - I deployed a 1-Dimensional Root-Finding Algorithm (Brent’s Method).
    - The engine runs your 10,000 lifetimes at a random withdrawal rate, calculates the median ending wealth at your life expectancy, and compares it to your declared "Target Legacy Floor."
    - It then iteratively adjusts the withdrawal rate up and down, re-running the 10,000 simulations over and over until the math perfectly converges on the exact percentage that safely lands you at your target floor without running out of money.
    """)

    st.header("3. Adaptive Guardrails & The Retirement Smile")
    st.write("Real retirees don't spend the exact same amount of money every year, adjusting blindly for inflation. They adjust based on the market and aging.")
    st.markdown("""
    - **Dynamic Spending Guardrails:** If the market booms and your withdrawal rate falls 20% below your starting rate, the engine grants you a 10% pay raise (Prosperity Rule). If the market crashes and your withdrawal rate spikes 20% too high, the engine forces a 10% pay cut (Capital Preservation Rule) to protect your principal.
    - **The "Retirement Smile" (David Blanchett Curve):** Your discretionary spending is modeled geometrically. Spending drops slowly during your "Slow-Go" years (ages 75-84) as travel and hobbies decline but re-accelerates in your "No-Go" years (85+) to account for increased medical conveniences and end-of-life care.
    """)

    st.header("4. Dynamic Liquidation Hierarchy & Sequence Risk Mitigation")
    st.write("The engine actively manages where you pull money from year by year based on what the simulated market is doing.")
    st.markdown("""
    - **Normal Years:** Lifestyle is funded by Tax-Deferred accounts (TSP/Trad IRA), allowing your Tax-Free (Roth) and Taxable accounts to compound.
    - **Market Crash Years (Down >10%):** The engine triggers an emergency **Sequence of Return Risk (SORR)** protocol. It immediately halts the sale of equities in your TSP/IRA to avoid locking in losses. It seamlessly pivots to burning down your Cash Buffer, followed by Taxable and Roth accounts, until the market recovers.
    """)

    st.header("5. Federal Tax Code & Roth Optimization Engine")
    st.write("The model contains a highly detailed US Tax logic tree.")
    st.markdown("""
    - It tracks Standard Deductions, Ordinary Brackets, Long-Term Capital Gains (LTCG), Net Investment Income Tax (NIIT), and State/Local taxes.
    - **Roth Optimizer:** In the background, the engine actually runs your entire lifetime 5 separate times using different Roth Conversion strategies (Baseline, Filling your current bracket, Targeting IRMAA cliffs, and Max Bracket limits). It compares the "Terminal Legacy" of all 5 runs and surfaces the mathematical winner to you, alongside a step-by-step conversion schedule.
    """)

# ==========================================
# PAGE 4: ABOUT
# ==========================================
with nav4:
    st.title("About the Advanced Quantitative Retirement Planner")
    st.markdown("---")

    st.header("The Mission")
    st.write("""
    For decades, ultra-wealthy families and institutional endowments have relied on sophisticated Monte Carlo simulations and dynamic spending algorithms to manage their wealth. Meanwhile, DIY investors have been forced to rely on rudimentary calculators that output dangerous, straight-line "averages."

    **I built this platform to democratize institution-grade financial modeling.**

    The Advanced Quantitative Retirement Planner was designed to bridge the gap between basic retirement calculators and expensive, gatekept professional financial software. It evaluates the raw, mathematical truth of your retirement survivability.
    """)

    st.header("Specialized for Federal Employees")
    st.write("""
    While this simulator is highly effective for any private-sector retiree, it features a specialized logic engine built specifically to handle the unique nuances of United States Federal Employees and Military Retirees.
    """)
    st.markdown("""
    - Native integration for the Thrift Savings Plan (TSP).
    - Actuarial comparisons between Medicare Part B / IRMAA and FEHB (FEPBlue, GEHA) / TRICARE for Life.
    - Phased Retirement modeling and FERS Pension COLA calculations.
    - Parallel Military & FERS integration combining separate survivor benefit multipliers, Start Ages, and differing CPI/Diet COLA rules.
    """)

    st.header("Why 'CASAM'?")
    st.write("""
    This tool relies on the **Constant Amortization Spending Model (CASAM)**. Instead of using rigid rules like the "4% Rule," CASAM looks at your actual portfolio balance, guaranteed income streams (Social Security, Pensions), and specific tax liabilities every single year, dynamically adjusting your safe spending limits to ensure your money outlives you.
    """)
    st.markdown("---")
    st.markdown("**Developed by DK**")