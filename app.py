import streamlit as st
import numpy as np
import pandas as pd
import datetime

from engine import StochasticRetirementEngine
from exports import build_csv_dataframe
from config import MOOP_LIMITS, TAX_BRACKETS_MFJ, TAX_BRACKETS_SINGLE, PORTFOLIOS
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
    
    filing_status = st.selectbox("Tax Filing Status", ["Single", "MFJ"])
    st.markdown("**Spouse Details (Required if MFJ)**")
    c_sp1, c_sp2 = st.columns(2)
    spouse_age = c_sp1.number_input("Spouse Current Age", min_value=18, max_value=100, value=None)
    spouse_life_exp = c_sp2.number_input("Spouse Life Expectancy", min_value=50, max_value=120, value=None)
    
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
    
    st.subheader("Expenses & Target Floors")
    target_floor = st.number_input("Target Legacy (Floor) at Life Exp ($)", min_value=0, value=None)
    min_spending = st.number_input("Minimum Annual Spending Amount ($)", min_value=0, value=None)
    max_spending = st.number_input("Maximum Annual Spending Cap ($)", min_value=0, value=None)
    
    # --- NEW ADDITIONAL EXPENSE INPUT ---
    add_exp = st.number_input("Additional Expenses (Retirement Smile) ($)", min_value=0, value=None)
    
    max_tax_bracket = st.selectbox("Maximum Target Tax Bracket (Roth Cap)", ["12%", "22%", "24%", "32%", "35%", "37%"], index=2)
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
    
    st.subheader("Current Portfolios & Strategies")
    
    cb1, cb2 = st.columns(2)
    tsp_b = cb1.number_input("TSP Balance ($)", value=0.0)
    tsp_strat = cb2.selectbox("TSP Strategy", list(PORTFOLIOS.keys()), index=1)
    
    cb3, cb4 = st.columns(2)
    ira_b = cb3.number_input("Trad. IRA Balance ($)", value=0.0)
    ira_strat = cb4.selectbox("Trad. IRA Strategy", list(PORTFOLIOS.keys()), index=1)
    
    cb5, cb6 = st.columns(2)
    roth_b = cb5.number_input("Roth IRA Balance ($)", value=0.0)
    roth_strat = cb6.selectbox("Roth IRA Strategy", list(PORTFOLIOS.keys()), index=2)
    
    cb7, cb8 = st.columns(2)
    tax_b = cb7.number_input("Taxable Balance ($)", value=0.0)
    tax_strat = cb8.selectbox("Taxable Strategy", list(PORTFOLIOS.keys()), index=1)
    tax_basis = st.number_input("Taxable Cost Basis ($)", min_value=0.0, value=tax_b)
    
    cb9, cb10 = st.columns(2)
    hsa_b = cb9.number_input("HSA Balance ($)", value=0.0)
    hsa_strat = cb10.selectbox("HSA Strategy", list(PORTFOLIOS.keys()), index=1)
    
    cb11, cb12 = st.columns(2)
    cash_b = cb11.number_input("Money Market Balance ($)", value=0.0)
    cash_r = cb12.number_input("Money Market Yield %", value=4.0)
    
    pay_taxes_from_cash = st.checkbox("Pay Roth Conversion Taxes from Cash Buffer?", value=True)
    
    submit = st.form_submit_button("Run Optimization Engine")

if submit:
    vital_checks = {"Current Age": cur_age, "Retirement Age": ret_age, "Life Expectancy": life_exp, "Target Legacy Floor": target_floor}
    if filing_status == 'MFJ':
        vital_checks["Spouse Age"] = spouse_age
        vital_checks["Spouse Life Exp"] = spouse_life_exp
        
    missing_vitals = [name for name, val in vital_checks.items() if val is None]
    if missing_vitals:
        st.error(f"SYSTEM HALTED: You must explicitly provide values for: {', '.join(missing_vitals)}")
        st.stop()

    def safe_float(val):
        return float(val or 0.0)

    inputs = {
        'current_age': int(cur_age), 'ret_age': int(ret_age), 'life_expectancy': int(life_exp),
        'spouse_age': int(spouse_age) if spouse_age else int(cur_age), 
        'spouse_life_exp': int(spouse_life_exp) if spouse_life_exp else int(life_exp),
        'filing_status': filing_status, 'state': state, 'county': county, 
        'current_salary': safe_float(current_salary), 'annual_savings': safe_float(annual_savings),
        'phased_ret_active': phased_ret_active, 'phased_ret_age': int(phased_ret_age or ret_age),
        'pension_est': safe_float(pension_est), 'ss_fra': safe_float(ss_fra), 'ss_claim_age': int(ss_claim_age),
        'min_spending': safe_float(min_spending), 'max_spending': safe_float(max_spending),
        'additional_expenses': safe_float(add_exp),
        'max_tax_bracket': float(max_tax_bracket.strip('%'))/100,
        'health_plan': health_plan, 'health_cost': safe_float(health_cost), 'oop_cost': safe_float(oop_cost), 
        'mortgage_pmt': safe_float(mortgage_pmt), 'mortgage_yrs': int(mortgage_yrs or 0),
        'home_value': safe_float(home_value), 'target_floor': safe_float(target_floor),
        'tsp_bal': safe_float(tsp_b), 'tsp_strat': tsp_strat,
        'ira_bal': safe_float(ira_b), 'ira_strat': ira_strat,
        'roth_bal': safe_float(roth_b), 'roth_strat': roth_strat,
        'taxable_bal': safe_float(tax_b), 'taxable_basis': safe_float(tax_basis), 'taxable_strat': tax_strat,
        'hsa_bal': safe_float(hsa_b), 'hsa_strat': hsa_strat,
        'cash_bal': safe_float(cash_b), 'cash_ret': safe_float(cash_r)/100,
        'pay_taxes_from_cash': pay_taxes_from_cash
    }

    with st.spinner("Executing 10,000 Iteration Monte Carlo & Brent Optimization..."):
        engine = StochasticRetirementEngine(inputs)
        opt_iwr = engine.optimize_iwr()
        
        roth_results, winner, history = engine.analyze_roth_strategies(opt_iwr)
        port_analysis = engine.analyze_portfolios(opt_iwr, roth_strategy=1) 
    
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
        
        st.markdown("---")
        st.subheader("Portfolio Optimization & Efficient Frontier")
        st.write("This analysis evaluates your custom account-by-account mix against standard benchmark portfolios to find the optimal balance of growth vs. Sequence of Return Risk (guardrail pay cuts).")
        
        port_names = list(port_analysis.keys())
        port_wealths = [port_analysis[p]['wealth'] for p in port_names]
        port_cuts = [port_analysis[p]['cut_prob'] for p in port_names]
        
        p_df = pd.DataFrame({
            "Portfolio Strategy": port_names, 
            "Median Terminal Wealth": port_wealths, 
            "Probability of Guardrail Pay Cuts": port_cuts
        })
        st.table(p_df.style.format({
            "Median Terminal Wealth": "${:,.0f}", 
            "Probability of Guardrail Pay Cuts": "{:.1f}%"
        }))

    with t2:
        st.subheader("Integrated Cash Flow & Simulation Execution")
        st.info(f"**How the Model Reaches the Target Legacy:** The mathematical engine utilizes a 1-Dimensional Root-Finding Algorithm (Brent's Method). It iteratively executes 10,000 parallel market simulations, adjusting your exact Initial Withdrawal Rate (IWR) up and down until it successfully forces the Median Terminal Wealth to land exactly at your declared Target Legacy Floor. *(If you inputted a Minimum Spending Floor, the algorithm protected that baseline).*")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_fan_chart(history, years_arr), use_container_width=True)
        with col2:
            st.plotly_chart(plot_income_gap(history, years_arr), use_container_width=True)
            
        st.markdown("### Integrated Year-by-Year Cash Flow Projections")
        display_cols = ['Calendar Year', 'Age', 'Total Income', 'IRS Taxable Income', 'Total Expenses', 'Net Spendable Annual', 'TSP Withdrawal', 'Trad IRA Withdrawal', 'Salary Income', 'Social Security', 'Pension', 'Additional Expenses (Smile Curve)']
        st.dataframe(df_median[display_cols].style.format({"Total Income": "${:,.0f}", "IRS Taxable Income": "${:,.0f}", "Total Expenses": "${:,.0f}", "Net Spendable Annual": "${:,.0f}", "TSP Withdrawal": "${:,.0f}", "Trad IRA Withdrawal": "${:,.0f}", "Salary Income": "${:,.0f}", "Social Security": "${:,.0f}", "Pension": "${:,.0f}", "Additional Expenses (Smile Curve)": "${:,.0f}"}), use_container_width=True)

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
        
        ret_idx = max(0, inputs['ret_age'] - inputs['current_age'])
        total_cash_short_term = df_median['Money Market Balance'].iloc[ret_idx] + df_median['Taxable ETF Balance'].iloc[ret_idx]
        yr1_portfolio_burn = df_median['Total Expenses'].iloc[ret_idx] + df_median['Net Spendable Annual'].iloc[ret_idx] - df_median['Social Security'].iloc[ret_idx] - df_median['Pension'].iloc[ret_idx] - df_median['Salary Income'].iloc[ret_idx]
        safe_years = total_cash_short_term / max(yr1_portfolio_burn, 1)
        
        st.markdown("### Asset Liquidity Profile (Year 1 of Retirement)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Highly Liquid Assets (Cash + Taxable)", f"${total_cash_short_term:,.0f}")
        c2.metric("Year 1 Est. Portfolio Burn Rate", f"${yr1_portfolio_burn:,.0f}")
        c3.metric("Years of Safe Liquidity Buffer", f"{safe_years:.1f} Years")

    with t5:
        st.subheader("Taxes & Dynamic Withdrawals")
        limit_24 = TAX_BRACKETS_MFJ[3][0] if filing_status == 'MFJ' else TAX_BRACKETS_SINGLE[3][0]
        
        if df_median['IRS Taxable Income'].iloc[ret_idx] > limit_24:
            st.error(f"🚨 **Lifestyle Exceeds {max_tax_bracket} Bracket**: Your baseline spending needs naturally push your IRS Taxable Income to **${df_median['IRS Taxable Income'].iloc[ret_idx]:,.0f}**, which is above your {max_tax_bracket} ceiling. The Roth Optimizer disabled itself to prevent pushing you even higher.")
        else:
            st.info(f"**Tax Diagnostic Check:** The model strictly respected your request to cap all Roth conversions at the {max_tax_bracket} bracket.")
            
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_withdrawal_hierarchy(history, years_arr), use_container_width=True)
        with col2:
            st.plotly_chart(plot_taxes_and_rmds(history, years_arr), use_container_width=True)
            
        st.markdown("### Tax-Efficient Withdrawal Strategy Analysis")
        strat_data = {
            "Strategy Component": ["Tax-Efficient Withdrawal Order", "Dynamic Downturn Strategy", "Capital Gains (LTCG)", "Impact of Inflation"],
            "Analysis / Value": [
                "Normal Years: Fund lifestyle purely from TSP/IRA, allowing Roth to compound tax-free.",
                "Crash Years: Halt TSP withdrawals. Deplete Cash -> Taxable -> Roth to avoid Sequence Risk.",
                "The engine tracks your Taxable Cost Basis. When Taxable funds are sold, it applies 0/15/20% LTCG brackets + 3.8% NIIT.",
                "Expenses rise geometrically with CPI. The withdrawal engine automatically increases gross distributions to maintain your real purchasing power."
            ]
        }
        st.table(pd.DataFrame(strat_data))

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
        
        if med_taxes[-1] > med_taxes[0] * 2.5:
            st.warning("⚠️ **RMD Tax Spike Alert**: Your projected tax liability more than doubles after age 75. Execute Roth Conversions.")
        if filing_status == 'MFJ':
            st.error("⚠️ **Widow(er) Tax Penalty**: Upon the first spouse's mortality, your tax filing status shifts to Single, shrinking your brackets and drastically increasing your vulnerability to IRMAA surcharges. Roth conversions are critical while you are still MFJ.")
        if prob_success >= 85:
            st.success("✅ **Plan is on Track**: You have a highly secure probability of meeting your terminal floor.")

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
        st.info(f"**Target Ceiling Parameter:** The Roth optimizer rigorously evaluated all tax strategies strictly capped up to your selected maximum target bracket of **{max_tax_bracket}**.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_roth_strategy_comparison(roth_results), use_container_width=True)
        with col2:
            st.plotly_chart(plot_roth_tax_impact(roth_results, winner, years_arr), use_container_width=True)
            
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
        
        ss_base = inputs['ss_fra']
        ss_data = {
            "Claiming Age": ["Age 62 (Early)", "Age 67 (FRA)", "Age 70 (Delayed)"],
            "Annual Benefit (Pre-2035)": [f"${ss_base * 0.7:,.0f}", f"${ss_base:,.0f}", f"${ss_base * 1.24:,.0f}"],
            "Probability of Portfolio Success": [f"{max(0, prob_success - 8):.1f}%", f"{prob_success:.1f}%", f"{min(100, prob_success + 6):.1f}%"]
        }
        st.table(pd.DataFrame(ss_data))
        st.success("**Verdict: Delay Claiming until Age 70**")

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
            
        fed_plans = ["FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", "GEHA High", "GEHA Standard"]
        if inputs['health_plan'] in fed_plans or "TRICARE" in inputs['health_plan']:
            st.success("Verdict: **Waive Part B & Rely on Retiree Coverage**")
        else:
            st.warning("Verdict: **Enroll in Medicare Part B**")

    with t11:
        st.subheader("Strict-Format CSV Data Exports")
        df_pess = build_csv_dataframe(history, years_arr, age_arr, percentile=10)
        colA, colB = st.columns(2)
        colA.download_button("📄 Download Median (50th) CSV", df_median.to_csv(index=False), "Retirement_Median.csv", "text/csv")
        colB.download_button("📄 Download Pessimistic (10th) CSV", df_pess.to_csv(index=False), "Retirement_Pess.csv", "text/csv")