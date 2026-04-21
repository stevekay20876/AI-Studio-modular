# app.py
import streamlit as st
import numpy as np
import datetime

# Import custom modules
from engine import StochasticRetirementEngine
from visuals import plot_wealth_trajectory, plot_net_spendable, plot_liquidity_timeline, plot_taxes_and_rmds
from exports import build_csv_dataframe

st.set_page_config(page_title="Advanced Retirement Simulator", layout="wide")

st.title("Advanced Quantitative Retirement Planner")
st.markdown("Institution-Grade Monte Carlo Simulator | Constant Amortization Spending Model (CASAM)")

with st.sidebar.form("input_form"):
    st.header("Client Parameters")
    c1, c2 = st.columns(2)
    cur_age = c1.number_input("Current Age", min_value=18, max_value=100, value=None)
    ret_age = c2.number_input("Retirement Age", min_value=18, max_value=100, value=None)
    life_exp = st.number_input("Life Expectancy Age", min_value=50, max_value=120, value=None)
    
    filing_status = st.selectbox("Tax Filing Status", ["Single", "MFJ"])
    state = st.text_input("State of Residence")
    
    st.subheader("Federal Details & Income")
    pension_est = st.number_input("Annual Pension Estimate ($)", min_value=0, value=None)
    ss_fra = st.number_input("Social Security at FRA ($/yr)", min_value=0, value=None)
    
    st.subheader("Expenses & Health")
    health_plan = st.selectbox("Retiree Health Coverage", ["FEHB FEPBlue Basic", "Private ACA", "None/Self-Insure"])
    health_cost = st.number_input("Annual Health Premium ($)", min_value=0, value=None)
    target_floor = st.number_input("Target Estate Floor at Life Exp ($)", min_value=0, value=None)
    
    st.subheader("Current Portfolios (Balance / Expected Return % / Volatility %)")
    tsp_b = st.number_input("TSP / 401(k) Balance", min_value=0, value=None)
    tsp_r = st.number_input("TSP Return %", value=None)
    tsp_v = st.number_input("TSP Volatility %", value=None)
    roth_b = st.number_input("Roth IRA Balance", min_value=0, value=None)
    roth_r = st.number_input("Roth Return %", value=None)
    roth_v = st.number_input("Roth Volatility %", value=None)
    tax_b = st.number_input("Taxable Balance", min_value=0, value=None)
    tax_r = st.number_input("Taxable Return %", value=None)
    tax_v = st.number_input("Taxable Volatility %", value=None)
    cash_b = st.number_input("Money Market Balance", min_value=0, value=None)
    cash_r = st.number_input("Money Market Yield %", value=None)
    hsa_b = st.number_input("HSA Balance (Optional)", min_value=0, value=0)
    hsa_r = st.number_input("HSA Return % (Optional)", value=0.0)
    hsa_v = st.number_input("HSA Volatility % (Optional)", value=0.0)
    
    submit = st.form_submit_button("Run Optimization Engine")

if submit:
    inputs_check = [cur_age, ret_age, life_exp, target_floor, tsp_b, tsp_r, tsp_v, roth_b, roth_r, roth_v, tax_b, tax_r, tax_v, cash_b, cash_r]
    if any(i is None for i in inputs_check):
        st.error("SYSTEM HALTED: All core numerical parameters must be explicitly provided.")
        st.stop()

    inputs = {
        'current_age': int(cur_age), 'ret_age': int(ret_age), 'life_expectancy': int(life_exp),
        'filing_status': filing_status, 'state': state, 'pension_est': float(pension_est or 0),
        'ss_fra': float(ss_fra or 0), 'health_plan': health_plan, 'health_cost': float(health_cost or 0),
        'target_floor': float(target_floor),
        'tsp_bal': float(tsp_b), 'tsp_ret': float(tsp_r)/100, 'tsp_vol': float(tsp_v)/100,
        'roth_bal': float(roth_b), 'roth_ret': float(roth_r)/100, 'roth_vol': float(roth_v)/100,
        'taxable_bal': float(tax_b), 'taxable_ret': float(tax_r)/100, 'taxable_vol': float(tax_v)/100,
        'hsa_bal': float(hsa_b), 'hsa_ret': float(hsa_r)/100, 'hsa_vol': float(hsa_v)/100,
        'cash_bal': float(cash_b), 'cash_ret': float(cash_r)/100
    }

    with st.spinner("Executing 10,000 Iteration Monte Carlo & Brent Optimization..."):
        engine = StochasticRetirementEngine(inputs)
        opt_iwr = engine.optimize_iwr()
        history = engine.run_mc(opt_iwr)
    
    st.success(f"Simulation Complete. Optimized Initial Withdrawal Rate: **{opt_iwr*100:.2f}%**")

    # Shared Array Setup
    years_arr = np.arange(datetime.datetime.now().year, datetime.datetime.now().year + engine.years)
    age_arr = np.arange(inputs['current_age']+1, inputs['current_age']+1+engine.years)

    # UI Tabs
    t1, t2, t3, t4, t5 = st.tabs(["📊 Lifetime Projections", "💵 Cash Flow", "📈 Net Worth", "🏛️ Taxes", "🔄 CSV Export"])

    with t1:
        st.plotly_chart(plot_wealth_trajectory(history, inputs['target_floor'], years_arr), use_container_width=True)
    with t2:
        st.plotly_chart(plot_net_spendable(history, years_arr), use_container_width=True)
    with t3:
        st.plotly_chart(plot_liquidity_timeline(history, years_arr), use_container_width=True)
    with t4:
        st.plotly_chart(plot_taxes_and_rmds(history, years_arr), use_container_width=True)
    
    with t5:
        st.subheader("Data Exports")
        df_median = build_csv_dataframe(history, years_arr, age_arr, percentile=50)
        df_pess = build_csv_dataframe(history, years_arr, age_arr, percentile=10)

        colA, colB = st.columns(2)
        colA.download_button("📄 Download Median (50th) CSV", df_median.to_csv(index=False), "Retirement_Median.csv", "text/csv")
        colB.download_button("📄 Download Pessimistic (10th) CSV", df_pess.to_csv(index=False), "Retirement_Pess.csv", "text/csv")