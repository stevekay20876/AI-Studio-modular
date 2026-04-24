import streamlit as st
import numpy as np
import pandas as pd
import datetime

from engine import StochasticRetirementEngine
from exports import build_csv_dataframe
from config import MOOP_LIMITS, TAX_BRACKETS_MFJ, TAX_BRACKETS_SINGLE
from visuals import (
    plot_wealth_trajectory, plot_liquidity_timeline, plot_cash_flow_sources,
    plot_expenses_breakdown, plot_withdrawal_hierarchy, plot_taxes_and_rmds,
    plot_roth_strategy_comparison, plot_roth_tax_impact, plot_ss_breakeven,
    plot_medicare_comparison, plot_income_volatility, plot_legacy_breakdown,
    plot_fan_chart, plot_income_gap
)

st.set_page_config(page_title="Advanced Retirement Simulator", layout="wide")

st.title("Advanced Quantitative Retirement Planner")
st.markdown("Institution-Grade Monte Carlo Simulator | Constant Amortization Spending Model (CASAM)")

with st.sidebar.form("input_form"):
    st.header("Client Parameters")
    
    # Primary Client Fields
    c1, c2 = st.columns(2)
    cur_age = c1.number_input("Current Age", min_value=18, max_value=100, value=None)
    ret_age = c2.number_input("Full Retirement Age", min_value=18, max_value=100, value=None)
    life_exp = st.number_input("Life Expectancy Age", min_value=50, max_value=120, value=None)
    
    # Filing Status & Spouse Fields
    filing_status = st.selectbox("Tax Filing Status", ["Single", "MFJ"])
    st.write("---")
    st.markdown("**Spouse Details (If MFJ)**")
    c_sp1, c_sp2 = st.columns(2)
    spouse_age = c_sp1.number_input("Spouse Current Age", min_value=18, max_value=100, value=None)
    spouse_life_exp = c_sp2.number_input("Spouse Life Expectancy", min_value=50, max_value=120, value=None)
    st.write("---")
    
    c3, c4 = st.columns(2)
    state = c3.text_input("State of Residence")
    county = c4.text_input("County of Residence")
    
    st.subheader("Pre-Retirement & Phased Transition")
    current_salary = st.number_input("Current Annual Salary ($)", min_value=0, value=None)
    annual_savings = st.number_input("Total Annual Savings (Until Ret.) ($)", min_value=0, value=None)
    phased_ret_active = st.checkbox("Enable FERS Phased Retirement?")
    phased_ret_age = st.number_input("Phased Retirement Start Age", min_value=50, max_value=70, value=None)
    
    st.subheader("Federal Details & Guaranteed Income")
    pension_est = st.number_input("Full Pension Estimate at Retirement ($)", min_value=0, value=None)
    ss_fra = st.number_input("Social Security at FRA ($/yr)", min_value=0, value=None)
    ss_claim_age = st.number_input("Target SS Claiming Age", min_value=62, max_value=70, value=67)
    
    st.subheader("Expenses & Health")
    max_spending = st.number_input("Maximum Annual Spending Cap (Optional) ($)", min_value=0, value=None)
    mortgage_pmt = st.number_input("Annual Mortgage Payment ($)", min_value=0, value=None)
    mortgage_yrs = st.number_input("Mortgage Years Remaining", min_value=0, value=None)
    home_value = st.number_input("Current Home Value ($)", min_value=0, value=None)
    
    health_options = [
        "FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", 
        "GEHA High", "GEHA Standard", "Aetna Open Access", 
        "Aetna Direct", "Aetna Advantage", "Cigna", "TRICARE for Life", "None/Self-Insure"
    ]
    health_plan = st.selectbox("Retiree Health Coverage", health_options)
    health_cost = st.number_input("Current Annual Health Premium ($)", min_value=0, value=None)
    oop_cost = st.number_input("Today's Typical Out-of-Pocket Medical ($)", min_value=0, value=None)
    
    target_floor = st.number_input("Target Legacy (Floor) at Life Exp ($)", min_value=0, value=None)
    
    st.subheader("Current Portfolios (Balance / Expected Return % / Volatility %)")
    tsp_b = st.number_input("TSP / 401(k) Balance", min_value=0, value=None)
    tsp_r = st.number_input("TSP Return %", value=None)
    tsp_v = st.number_input("TSP Volatility %", value=None)
    
    roth_b = st.number_input("Roth IRA Balance", min_value=0, value=None)
    roth_r = st.number_input("Roth Return %", value=None)
    roth_v = st.number_input("Roth Volatility %", value=None)
    
    tax_b = st.number_input("Taxable Balance", min_value=0, value=None)
    tax_basis = st.number_input("Taxable Cost Basis ($)", min_value=0, value=tax_b)
    tax_r = st.number_input("Taxable Return %", value=None)
    tax_v = st.number_input("Taxable Volatility %", value=None)
    
    cash_b = st.number_input("Money Market Balance", min_value=0, value=None)
    cash_r = st.number_input("Money Market Yield %", value=None)
    
    pay_taxes_from_cash = st.checkbox("Pay Roth Conversion Taxes from Cash Buffer?", value=True)
    
    hsa_b = st.number_input("HSA Balance (Optional)", min_value=0, value=None)
    hsa_r = st.number_input("HSA Return % (Optional)", value=None)
    hsa_v = st.number_input("HSA Volatility % (Optional)", value=None)
    
    submit = st.form_submit_button("Run Optimization Engine")

if submit:
    vital_checks = {"Current Age": cur_age, "Retirement Age": ret_age, "Life Expectancy": life_exp, "Target Legacy Floor": target_floor}
    
    # Dynamically require spouse inputs only if MFJ is selected
    if filing_status == 'MFJ':
        vital_checks["Spouse Age"] = spouse_age
        vital_checks["Spouse Life Exp"] = spouse_life_exp
        
    missing_vitals = [name for name, val in vital_checks.items() if val is None]
    
    if missing_vitals:
        st.error(f"SYSTEM HALTED: You must explicitly provide values for: {', '.join(missing_vitals)}")
        st.stop()

    def safe_float(val, is_vol=False):
        v = float(val or 0.0)
        if is_vol and v == 0.0:
            return 0.0001 
        return v

    inputs = {
        'current_age': int(cur_age), 'ret_age': int(ret_age), 'life_expectancy': int(life_exp),
        'spouse_age': int(spouse_age) if spouse_age else int(cur_age), 
        'spouse_life_exp': int(spouse_life_exp) if spouse_life_exp else int(life_exp),
        'filing_status': filing_status, 'state': state, 'county': county, 
        'current_salary': safe_float(current_salary), 'annual_savings': safe_float(annual_savings),
        'phased_ret_active': phased_ret_active, 'phased_ret_age': int(phased_ret_age or ret_age),
        'pension_est': safe_float(pension_est), 'ss_fra': safe_float(ss_fra), 'ss_claim_age': int(ss_claim_age),
        'max_spending': safe_float(max_spending),
        'health_plan': health_plan, 'health_cost': safe_float(health_cost),
        'oop_cost': safe_float(oop_cost), 'mortgage_pmt': safe_float(mortgage_pmt), 'mortgage_yrs': int(mortgage_yrs or 0),
        'home_value': safe_float(home_value), 'target_floor': safe_float(target_floor),
        'tsp_bal': safe_float(tsp_b), 'tsp_ret': safe_float(tsp_r)/100, 'tsp_vol': safe_float(tsp_v, True)/100,
        'roth_bal': safe_float(roth_b), 'roth_ret': safe_float(roth_r)/100, 'roth_vol': safe_float(roth_v, True)/100,
        'taxable_bal': safe_float(tax_b), 'taxable_basis': safe_float(tax_basis), 'taxable_ret': safe_float(tax_r)/100, 'taxable_vol': safe_float(tax_v, True)/100,
        'hsa_bal': safe_float(hsa_b), 'hsa_ret': safe_float(hsa_r)/100, 'hsa_vol': safe_float(hsa_v, True)/100,
        'cash_bal': safe_float(cash_b), 'cash_ret': safe_float(cash_r)/100,
        'pay_taxes_from_cash': pay_taxes_from_cash
    }

    with st.spinner("Executing 10,000 Iteration Monte Carlo & Brent Optimization..."):
        engine = StochasticRetirementEngine(inputs)
        opt_iwr = engine.optimize_iwr()
        roth_results = engine.analyze_roth_strategies(opt_iwr)
        
        winner = max(roth_results, key=lambda key: roth_results[key]['wealth'])
        history = roth_results[winner]['hist']
    
    st.success(f"Simulatio