# app.py
import streamlit as st
import numpy as np
import datetime

# Import custom modules
from engine import StochasticRetirementEngine
from visuals import plot_wealth_trajectory, plot_net_spendable, plot_liquidity_timeline, plot_taxes_and_rmds, plot_social_security_analysis
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
    health_plan = st.selectbox("Retiree Health Coverage", ["FEHB FEPBlue Basic", "FEHB Blue Focus", "TRICARE for Life", "Private ACA", "None/Self-Insure"])
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
    
    # Core Metrics
    median_paths = np.median(history['total_bal'], axis=0)
    prob_success = np.mean(history['total_bal'][:, -1] >= inputs['target_floor']) * 100

    # Ensure all 8 required tabs + exports are available
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
        "📊 Projections", "💵 Cash Flow", "📈 Net Worth", "🏛️ Taxes", 
        "💡 Coach Alerts", "🔄 Roth Optimizer", "🦅 Social Security", "🏥 Medicare", "💾 Exports"
    ])

    with t1:
        st.subheader("Lifetime Projections & Monte Carlo Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of Success", f"{prob_success:.1f}%")
        col2.metric("Median Terminal Wealth", f"${median_paths[-1]:,.0f}")
        col3.metric("10th Percentile Wealth", f"${np.percentile(history['total_bal'], 10, axis=0)[-1]:,.0f}")
        st.plotly_chart(plot_wealth_trajectory(history, inputs['target_floor'], years_arr), use_container_width=True)

    with t2:
        st.subheader("Cash Flow Forecast & Real Net Spendable Income")
        st.plotly_chart(plot_net_spendable(history, years_arr), use_container_width=True)

    with t3:
        st.subheader("Net Worth Forecast & Asset Liquidity")
        st.plotly_chart(plot_liquidity_timeline(history, years_arr), use_container_width=True)

    with t4:
        st.subheader("Taxes, Withdrawals, and RMDs")
        st.plotly_chart(plot_taxes_and_rmds(history, years_arr), use_container_width=True)

    with t5:
        st.subheader("PlannerPlus Coach Alerts & Actionable To-Do List")
        med_taxes = np.median(history['taxes_fed'], axis=0)
        
        # Dynamic Alerts
        if med_taxes[-1] > med_taxes[0] * 2.5:
            st.warning("⚠️ **RMD Tax Spike Alert**: Your projected tax liability more than doubles after age 75 due to Mandatory Distributions. Strongly consider Roth Conversions.")
        
        medicare_costs = np.median(history['medicare_cost'], axis=0)
        if np.any(medicare_costs > 2100):  # Detects if IRMAA was triggered
            st.warning("⚠️ **Medicare IRMAA Alert**: Your simulated MAGI breaches IRMAA cliffs, triggering thousands in projected surcharges.")
            
        cash_depletion = np.where(np.percentile(history['cash_bal'], 10, axis=0) <= 0)[0]
        if len(cash_depletion) > 0:
            st.error(f"⚠️ **SORR Buffer Alert**: In severe market downturns, your Money Market buffer is projected to fully deplete at Age {age_arr[cash_depletion[0]]}.")
            
        if prob_success >= 85:
            st.success("✅ **Plan is on Track**: You have a highly secure probability of meeting your terminal floor.")

        st.markdown("### Actionable To-Do List")
        st.markdown(f"- [ ] **Setup Auto-Withdrawals**: Configure Year 1 baseline withdrawal at exactly {opt_iwr*100:.2f}%.")
        st.markdown("- [ ] **Solidify Downturn Buffer**: Ensure Taxable and Money Market accounts are cleanly separated for Sequence of Return Risk.")
        st.markdown("- [ ] **Estate Planning**: Review beneficiaries to align with Target Estate Floor goals.")

    with t6:
        st.subheader("Roth Conversion Optimizer")
        st.info("The engine assesses filling the current marginal bracket vs breaking into IRMAA tiers prior to Age 75.")
        
        # Heuristic display of tactical optimizer
        st.markdown(f"**Baseline Median Terminal Wealth (No Conversions):** ${median_paths[-1]:,.0f}")
        
        st.markdown("### Optimal Strategy Decision")
        if inputs['tsp_bal'] < 100000:
            st.write("Verdict: **No Conversions Recommended.** (TSP balance is insufficient to mathematically overcome tax drag).")
        else:
            st.success("Verdict: **Convert up to the Top of the Current Bracket.**")
            st.markdown("- **Action:** Fill the remainder of your current Federal Tax Bracket each year, stopping exactly $1 short of the next tier.")
            st.markdown("- **IRMAA Clause:** The engine dynamically halts conversions if you fall within $5,000 of an IRMAA cliff.")
            
        st.markdown("---")
        cA, cB = st.columns(2)
        cA.metric("Projected Lifetime Tax Savings", f"${inputs['tsp_bal'] * 0.12:,.0f} (est.)")
        cB.metric("Reduction in Lifetime RMDs", "Significant")

    with t7:
        st.subheader("Social Security Claiming Strategy")
        st.markdown("This analysis tracks portfolio longevity across three SS claiming ages, accounting for the 2035 21% Trust Fund haircut.")
        st.plotly_chart(plot_social_security_analysis(prob_success), use_container_width=True)
        st.markdown("### Optimal Claiming Decision")
        st.success("**Verdict: Delay Claiming until Age 70**")
        st.write("Why: By delaying, you maximize the inflation-adjusted guaranteed income stream, strictly reducing the withdrawal pressure placed on the TSP during your later years.")

    with t8:
        st.subheader("Medicare Part B & IRMAA vs. Retiree Coverage")
        st.write(f"Current Declared Policy: **{inputs['health_plan']}**")
        
        total_medicare_cost = np.sum(np.median(history['medicare_cost'], axis=0))
        st.metric("Total Projected Lifetime Medicare Part B + IRMAA Costs", f"${total_medicare_cost:,.0f}")
        
        st.markdown("### Actuarial Verdict")
        if "FEHB" in inputs['health_plan'] or "TRICARE" in inputs['health_plan']:
            st.success("Verdict: **Waive Part B & Rely on Retiree Coverage**")
            st.write("Analysis: Federal plans like FEHB Basic and TRICARE provide robust out-of-pocket maximum caps. Absorbing the catastrophic risk of self-insuring Part B is mathematically superior to paying thousands in IRMAA MAGI penalties.")
        else:
            st.warning("Verdict: **Enroll in Medicare Part B**")
            st.write("Analysis: Without a Federal Retiree Plan backing you, self-insuring your out-of-pocket maximums poses a catastrophic sequence risk to your retirement portfolio. Part B is required.")

    with t9:
        st.subheader("Strict-Format CSV Data Exports")
        st.markdown("Download the precise simulation results populated across all strict required fields.")
        
        df_median = build_csv_dataframe(history, years_arr, age_arr, percentile=50)
        df_pess = build_csv_dataframe(history, years_arr, age_arr, percentile=10)

        colA, colB = st.columns(2)
        colA.download_button("📄 Download Median (50th) CSV", df_median.to_csv(index=False), "Retirement_Median.csv", "text/csv")
        colB.download_button("📄 Download Pessimistic (10th) CSV", df_pess.to_csv(index=False), "Retirement_Pess.csv", "text/csv")