import streamlit as st
import numpy as np
import pandas as pd
import datetime

from engine import StochasticRetirementEngine
from exports import build_csv_dataframe
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
    c1, c2 = st.columns(2)
    cur_age = c1.number_input("Current Age", min_value=18, max_value=100, value=None)
    ret_age = c2.number_input("Full Retirement Age", min_value=18, max_value=100, value=None)
    life_exp = st.number_input("Life Expectancy Age", min_value=50, max_value=120, value=None)
    
    c3, c4 = st.columns(2)
    state = c3.text_input("State of Residence")
    county = c4.text_input("County of Residence")
    filing_status = st.selectbox("Tax Filing Status", ["Single", "MFJ"])
    
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
    tax_r = st.number_input("Taxable Return %", value=None)
    tax_v = st.number_input("Taxable Volatility %", value=None)
    
    cash_b = st.number_input("Money Market Balance", min_value=0, value=None)
    cash_r = st.number_input("Money Market Yield %", value=None)
    
    hsa_b = st.number_input("HSA Balance (Optional)", min_value=0, value=None)
    hsa_r = st.number_input("HSA Return % (Optional)", value=None)
    hsa_v = st.number_input("HSA Volatility % (Optional)", value=None)
    
    submit = st.form_submit_button("Run Optimization Engine")

if submit:
    vital_checks = {"Current Age": cur_age, "Retirement Age": ret_age, "Life Expectancy": life_exp, "Target Legacy Floor": target_floor}
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
        'filing_status': filing_status, 'state': state, 'county': county, 
        'current_salary': safe_float(current_salary), 'annual_savings': safe_float(annual_savings),
        'phased_ret_active': phased_ret_active, 'phased_ret_age': int(phased_ret_age or ret_age),
        'pension_est': safe_float(pension_est), 'ss_fra': safe_float(ss_fra), 'ss_claim_age': int(ss_claim_age),
        'health_plan': health_plan, 'health_cost': safe_float(health_cost),
        'oop_cost': safe_float(oop_cost), 'mortgage_pmt': safe_float(mortgage_pmt), 'mortgage_yrs': int(mortgage_yrs or 0),
        'home_value': safe_float(home_value), 'target_floor': safe_float(target_floor),
        'tsp_bal': safe_float(tsp_b), 'tsp_ret': safe_float(tsp_r)/100, 'tsp_vol': safe_float(tsp_v, True)/100,
        'roth_bal': safe_float(roth_b), 'roth_ret': safe_float(roth_r)/100, 'roth_vol': safe_float(roth_v, True)/100,
        'taxable_bal': safe_float(tax_b), 'taxable_ret': safe_float(tax_r)/100, 'taxable_vol': safe_float(tax_v, True)/100,
        'hsa_bal': safe_float(hsa_b), 'hsa_ret': safe_float(hsa_r)/100, 'hsa_vol': safe_float(hsa_v, True)/100,
        'cash_bal': safe_float(cash_b), 'cash_ret': safe_float(cash_r)/100
    }

    with st.spinner("Executing 10,000 Iteration Monte Carlo & Brent Optimization..."):
        engine = StochasticRetirementEngine(inputs)
        opt_iwr = engine.optimize_iwr()
        roth_results = engine.analyze_roth_strategies(opt_iwr)
        
        winner = max(roth_results, key=lambda key: roth_results[key]['wealth'])
        history = roth_results[winner]['hist']
    
    st.success(f"Simulation Complete. Optimized Initial Portfolio Withdrawal Rate: **{opt_iwr*100:.2f}%**")

    years_arr = np.arange(datetime.datetime.now().year, datetime.datetime.now().year + engine.years)
    age_arr = np.arange(inputs['current_age']+1, inputs['current_age']+1+engine.years)
    median_paths = np.median(history['total_bal'], axis=0)
    prob_success = np.mean(history['total_bal'][:, -1] >= inputs['target_floor']) * 100
    df_median = build_csv_dataframe(history, years_arr, age_arr, percentile=50)

    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = st.tabs([
        "📊 Projections", "💵 Cash Flow", "📉 Guardrails", "📈 Net Worth", "🏛️ Taxes", 
        "🏛️ Legacy", "💡 Coach Alerts", "🔄 Roth Opt.", "🦅 Social Sec", "🏥 Medicare", "💾 Exports"
    ])

    with t1:
        st.subheader("Lifetime Projections & Monte Carlo Analysis")
        col1, col2, col3 = st.columns(3)
        col1.metric("Probability of Success", f"{prob_success:.1f}%")
        col2.metric("Median Terminal Wealth", f"${median_paths[-1]:,.0f}")
        col3.metric("10th Percentile Wealth", f"${np.percentile(history['total_bal'], 10, axis=0)[-1]:,.0f}")
        st.plotly_chart(plot_wealth_trajectory(history, inputs['target_floor'], years_arr), use_container_width=True)

    with t2:
        st.subheader("Integrated Cash Flow & Simulation Execution")
        st.info(f"**How the Model Reaches the Target Legacy:** The mathematical engine utilizes a 1-Dimensional Root-Finding Algorithm (Brent's Method). It iteratively executes 10,000 parallel market simulations, adjusting your exact Initial Withdrawal Rate (IWR) up and down until it successfully forces the Median (50th Percentile) Terminal Wealth to land exactly at your declared Target Legacy Floor of **${inputs['target_floor']:,.0f}** at Life Expectancy.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_fan_chart(history, years_arr), use_container_width=True)
        with col2:
            st.plotly_chart(plot_income_gap(history, years_arr), use_container_width=True)
            
        st.markdown("### Integrated Year-by-Year Cash Flow Projections")
        display_cols = ['Calendar Year', 'Age', 'Total Income', 'IRS Taxable Income', 'Total Expenses', 'Net Spendable Annual', 'Annual 401(k)/TSP Withdrawal', 'Salary Income', 'Social Security', 'Pension']
        st.dataframe(df_median[display_cols].style.format({"Total Income": "${:,.0f}", "IRS Taxable Income": "${:,.0f}", "Total Expenses": "${:,.0f}", "Net Spendable Annual": "${:,.0f}", "Annual 401(k)/TSP Withdrawal": "${:,.0f}", "Salary Income": "${:,.0f}", "Social Security": "${:,.0f}", "Pension": "${:,.0f}"}), use_container_width=True)

    with t3:
        st.subheader("Variable Spending Rules & Adaptive Guardrails")
        st.plotly_chart(plot_income_volatility(history, years_arr), use_container_width=True)

    with t4:
        st.subheader("Net Worth Forecast & Asset Liquidity Profile")
        st.plotly_chart(plot_liquidity_timeline(history, years_arr), use_container_width=True)
        
        st.markdown("### Asset Liquidity Profile (Year 1)")
        total_cash_short_term = inputs['cash_bal'] + inputs['taxable_bal']
        yr1_portfolio_burn = df_median['Total Expenses'][0] + df_median['Net Spendable Annual'][0] - df_median['Social Security'][0] - df_median['Pension'][0] - df_median['Salary Income'][0]
        safe_years = total_cash_short_term / max(yr1_portfolio_burn, 1)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Highly Liquid Assets (Cash + Taxable)", f"${total_cash_short_term:,.0f}")
        c2.metric("Year 1 Est. Portfolio Burn Rate", f"${yr1_portfolio_burn:,.0f}")
        c3.metric("Years of Safe Liquidity Buffer", f"{safe_years:.1f} Years")

    with t5:
        st.subheader("Taxes & Dynamic Withdrawals")
        limit_24 = 394600 if filing_status == 'MFJ' else 197300
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_withdrawal_hierarchy(history, years_arr), use_container_width=True)
        with col2:
            st.plotly_chart(plot_taxes_and_rmds(history, years_arr), use_container_width=True)

    with t6:
        st.subheader("After-Tax Legacy & Estate Breakdown")
        st.plotly_chart(plot_legacy_breakdown(history), use_container_width=True)
        med_tsp = np.median(history['tsp_bal'][:, -1])
        med_roth = np.median(history['roth_bal'][:, -1])
        med_taxable = np.median(history['taxable_bal'][:, -1]) + np.median(history['cash_bal'][:, -1])
        med_home = np.median(history['home_value'][:, -1])
        net_to_heirs = (med_tsp * 0.76) + med_taxable + med_roth + med_home
        st.metric("Estimated Net After-Tax Value to Heirs", f"${net_to_heirs:,.0f}", delta=f"Lost to IRD Taxes: -${med_tsp * 0.24:,.0f}", delta_color="inverse")

    with t7:
        st.subheader("PlannerPlus Coach Alerts & Actionable To-Do List")
        med_taxes = np.median(history['taxes_fed'], axis=0)
        
        if med_taxes[-1] > med_taxes[0] * 2.5:
            st.warning("⚠️ **RMD Tax Spike Alert**: Your projected tax liability more than doubles after age 75. Execute Roth Conversions.")
        medicare_costs = np.median(history['medicare_cost'], axis=0)
        if np.any(medicare_costs > 2100):  
            st.warning("⚠️ **Medicare IRMAA Alert**: Your simulated MAGI breaches IRMAA cliffs, triggering thousands in projected surcharges.")
        cash_depletion = np.where(np.percentile(history['cash_bal'], 10, axis=0) <= 0)[0]
        if len(cash_depletion) > 0:
            st.error(f"⚠️ **SORR Buffer Alert**: In severe downturns, your Money Market buffer is projected to fully deplete at Age {age_arr[cash_depletion[0]]}.")
        if prob_success >= 85:
            st.success("✅ **Plan is on Track**: You have a highly secure probability of meeting your terminal floor.")

    with t8:
        st.subheader("Roth Conversion Optimizer")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_roth_strategy_comparison(roth_results), use_container_width=True)
        with col2:
            st.plotly_chart(plot_roth_tax_impact(roth_results, years_arr), use_container_width=True)
            
        tax_savings = roth_results['Baseline (None)']['taxes'] - roth_results[winner]['taxes']
        rmd_reduction = roth_results['Baseline (None)']['rmds'] - roth_results[winner]['rmds']
        wealth_increase = roth_results[winner]['wealth'] - roth_results['Baseline (None)']['wealth']
        
        st.markdown("### Recommended Action Plan")
        if "Baseline" in winner:
            st.warning("**Verdict: No Conversions Recommended.**")
        else:
            st.success(f"Verdict: **Execute the '{winner}' Strategy**")
            
            st.write(f"- **Real Lifetime Tax Savings:** ${max(0, tax_savings):,.0f}")
            st.write(f"- **Reduction in Lifetime RMDs:** ${rmd_reduction:,.0f}")
            st.write(f"- **Net Increase to Legacy:** ${wealth_increase:,.0f}")
            
            st.markdown("#### Step-by-Step Conversion Schedule")
            roth_amts = np.median(roth_results[winner]['hist']['roth_conversion'], axis=0)
            conv_df = pd.DataFrame({"Year": years_arr, "Age": age_arr, "Target Conversion Amount": roth_amts, "Est. IRS Taxable Income": np.median(roth_results[winner]['hist']['taxable_income'], axis=0)})
            st.table(conv_df[conv_df['Target Conversion Amount'] > 0].style.format({"Target Conversion Amount": "${:,.0f}", "Est. IRS Taxable Income": "${:,.0f}"}))

    with t9:
        st.subheader("Social Security Claiming Strategy")
        st.plotly_chart(plot_ss_breakeven(inputs['ss_fra'], age_arr), use_container_width=True)

    with t10:
        st.subheader("Medicare Part B & Actuarial Healthcare OOP")
        st.plotly_chart(plot_medicare_comparison(history, years_arr, inputs), use_container_width=True)
        
        total_medicare_cost = np.sum(np.median(history['medicare_cost'], axis=0))
        st.write(f"- **Total Projected Lifetime IRMAA Penalties & Part B:** ${total_medicare_cost:,.0f}")
        
        moop_cap = MOOP_LIMITS.get(health_plan, (999999, 999999))[1 if filing_status == 'MFJ' else 0]
        if moop_cap == 999999:
            st.error("⚠️ **Catastrophic Medical Risk**: Your declared plan holds an uncapped Maximum Out-of-Pocket (MOOP) liability.")
        else:
            st.info(f"🛡️ **Plan Protection Active**: Your {health_plan} correctly caps out-of-pocket medical tail-risk at **${moop_cap:,.0f}** per year (inflation adjusted).")

    with t11:
        st.subheader("Strict-Format CSV Data Exports")
        df_pess = build_csv_dataframe(history, years_arr, age_arr, percentile=10)
        colA, colB = st.columns(2)
        colA.download_button("📄 Download Median (50th) CSV", df_median.to_csv(index=False), "Retirement_Median.csv", "text/csv")
        colB.download_button("📄 Download Pessimistic (10th) CSV", df_pess.to_csv(index=False), "Retirement_Pess.csv", "text/csv")