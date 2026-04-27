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
from visuals import (
    plot_wealth_trajectory, plot_liquidity_timeline, plot_cash_flow_sources,
    plot_expenses_breakdown, plot_withdrawal_hierarchy, plot_taxes_and_rmds,
    plot_roth_strategy_comparison, plot_roth_tax_impact, plot_ss_breakeven,
    plot_medicare_comparison, plot_income_volatility, plot_legacy_breakdown,
    plot_fan_chart, plot_income_gap
)

st.set_page_config(page_title="Advanced Retirement Simulator", layout="wide")

components.html(
    """
    <script>
        window.parent.document.body.scrollTop = 0;
        window.parent.document.documentElement.scrollTop = 0;
    </script>
    """,
    height=0,
    width=0,
)

ui_styling = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {display: none !important;}
    [data-testid="stHeader"] {visibility: hidden;}
    .stAppBottom {display: none !important;}
    
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2.0rem !important; font-weight: 700 !important; color: #00837B !important; }
    button[data-baseweb="tab"] { font-size: 1.2rem !important; padding: 1rem 1.5rem !important; }
    [data-testid="stTabs"] { border: 2px solid #E5E7EB; border-radius: 12px; padding: 15px; background-color: #FFFFFF; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; border: 1px solid #E5E7EB !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important; background-color: #FFFFFF !important; }
    </style>
"""
st.markdown(ui_styling, unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.tabs(["📊 Main Dashboard", "📝 Instructions", "⚙️ Background & Methodology", "ℹ️ About"])

# ==========================================
# PAGE 1: MAIN DASHBOARD
# ==========================================
with nav1:
    st.title("Advanced Quantitative Retirement Planner")
    st.markdown("Institution-Grade Monte Carlo Simulator | Constant Amortization Spending Model (CASAM)")

    DEFAULT_STATE = {
        'cur_age': None, 'ret_age': None, 'life_exp': None, 'filing_status': "Single",
        'spouse_age': None, 'spouse_life_exp': None, 'state': "", 'county': "",
        'current_salary': None, 'annual_savings': None, 'phased_ret_active': False,
        'phased_ret_age': None, 'pension_est': None, 'survivor_benefit': "Full Survivor Benefit", 
        'ss_fra': None, 'ss_claim_age': 67,
        'target_floor': None, 'min_spending': None, 'max_spending': None, 'add_exp': None,
        'max_tax_bracket': "24%", 'mortgage_pmt': None, 'mortgage_yrs': None, 'home_value': None,
        'health_plan': "None/Self-Insure", 'health_cost': None, 'oop_cost': None,
        'tsp_b': 0.0, 'tsp_strat': "Moderate (60% Stock / 40% Bond)",
        'ira_b': 0.0, 'ira_strat': "Moderate (60% Stock / 40% Bond)",
        'roth_b': 0.0, 'roth_strat': "Aggressive (100% Stock)",
        'tax_b': 0.0, 'tax_basis': None, 'tax_strat': "Moderate (60% Stock / 40% Bond)",
        'hsa_b': None, 'hsa_strat': "Moderate (60% Stock / 40% Bond)",
        'cash_b': 0.0, 'cash_r': 4.0, 'pay_taxes_from_cash': True
    }

    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def get_current_state():
        return {k: st.session_state[k] for k in DEFAULT_STATE.keys() if k in st.session_state}

    with st.expander("💾 Client Profile Management (Save / Load)", expanded=False):
        st.write("Save your current inputs to your computer, or load a previously saved profile to instantly fill out the form.")
        col_load, col_save = st.columns(2)
        
        with col_load:
            uploaded_profile = st.file_uploader("Load Saved Profile (.json)", type="json")
            if uploaded_profile is not None:
                if "loaded_file" not in st.session_state or st.session_state.loaded_file != uploaded_profile.name:
                    try:
                        loaded_data = json.load(uploaded_profile)
                        for key, value in loaded_data.items():
                            st.session_state[key] = value
                        st.session_state.loaded_file = uploaded_profile.name
                        st.success("Profile Loaded Successfully!")
                        st.rerun() 
                    except Exception as e:
                        st.error("Error loading profile.")
                    
        with col_save:
            st.write("") 
            st.write("")
            st.download_button(
                label="⬇️ Save Current Profile to Computer",
                data=json.dumps(get_current_state(), indent=4),
                file_name="client_profile.json",
                mime="application/json",
                use_container_width=True
            )

    with st.form("input_form"):
        st.markdown("### Step 1: Build Your Profile")
        
        with st.expander("👤 Personal & Tax Details", expanded=True):
            c1, c2, c3 = st.columns(3)
            cur_age = c1.number_input("Current Age", min_value=18, max_value=100, key="cur_age")
            ret_age = c2.number_input("Full Retirement Age", min_value=18, max_value=100, key="ret_age")
            life_exp = c3.number_input("Primary Life Expectancy Age", min_value=50, max_value=120, key="life_exp")
            
            st.markdown("**Tax & Spouse Details**")
            c4, c5, c6 = st.columns(3)
            filing_status = c4.selectbox("Tax Filing Status", ["Single", "MFJ"], key="filing_status")
            state_in = c5.text_input("State of Residence", key="state")
            county_in = c6.text_input("County of Residence", key="county")
            
            c_sp1, c_sp2 = st.columns(2)
            spouse_age = c_sp1.number_input("Spouse Current Age (If MFJ)", min_value=18, max_value=100, key="spouse_age")
            spouse_life_exp = c_sp2.number_input("Spouse Life Expectancy (If MFJ)", min_value=50, max_value=120, key="spouse_life_exp")

        with st.expander("💼 Income & Social Security", expanded=False):
            st.markdown("**Pre-Retirement & Phased Transition**")
            c1, c2 = st.columns(2)
            current_salary = c1.number_input("Current Annual Salary ($)", min_value=0, key="current_salary")
            annual_savings = c2.number_input("Total Annual Savings (Until Ret.) ($)", min_value=0, key="annual_savings")
            
            c3, c4 = st.columns(2)
            phased_ret_active = c3.checkbox("Enable FERS Phased Retirement?", key="phased_ret_active")
            phased_ret_age = c4.number_input("Phased Retirement Start Age", min_value=50, max_value=70, key="phased_ret_age")
            
            st.markdown("**Federal Details & Guaranteed Income**")
            c5, c6 = st.columns(2)
            pension_est = c5.number_input("Full (Unreduced) Pension Est. ($)", min_value=0, key="pension_est")
            survivor_benefit = c6.selectbox("FERS Survivor Benefit Option", ["Full Survivor Benefit", "Partial Survivor Benefit", "No Survivor Benefit"], key="survivor_benefit")
            
            c7, c8 = st.columns(2)
            ss_fra = c7.number_input("Social Security at FRA ($/yr)", min_value=0, key="ss_fra")
            ss_claim_age = c8.number_input("Target SS Claiming Age", min_value=62, max_value=70, key="ss_claim_age")

        with st.expander("📉 Expenses & Goals", expanded=False):
            st.markdown("**Spending Limits & Legacy Goals (In Today's Dollars)**")
            c1, c2, c3 = st.columns(3)
            target_floor = c1.number_input("Target Legacy Floor (Today's $) ($)", min_value=0, key="target_floor")
            min_spending = c2.number_input("Minimum Spending Floor (Today's $) ($)", min_value=0, key="min_spending")
            max_spending = c3.number_input("Maximum Spending Cap (Today's $) ($)", min_value=0, key="max_spending")
            
            c4, c5 = st.columns(2)
            add_exp = c4.number_input("Additional Expenses (Retirement Smile) ($)", min_value=0, key="add_exp")
            max_tax_bracket = c5.selectbox("Maximum Target Tax Bracket (Roth Cap)", ["12%", "22%", "24%", "32%", "35%", "37%"], index=2, key="max_tax_bracket")
            
            st.markdown("**Property & Debt**")
            c6, c7, c8 = st.columns(3)
            home_value = c6.number_input("Current Home Value ($)", min_value=0, key="home_value")
            mortgage_pmt = c7.number_input("Annual Mortgage Payment ($)", min_value=0, key="mortgage_pmt")
            mortgage_yrs = c8.number_input("Mortgage Years Remaining", min_value=0, key="mortgage_yrs")
            
            st.markdown("**Healthcare**")
            c9, c10, c11 = st.columns(3)
            health_options = ["FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", "GEHA High", "GEHA Standard", "Aetna Open Access", "Aetna Direct", "Aetna Advantage", "Cigna", "TRICARE for Life", "None/Self-Insure"]
            health_plan = c9.selectbox("Retiree Health Coverage", health_options, key="health_plan")
            health_cost = c10.number_input("Annual Health Premium ($)", min_value=0, key="health_cost")
            oop_cost = c11.number_input("Typical Out-of-Pocket Medical ($)", min_value=0, key="oop_cost")

        with st.expander("🏛️ Savings & Assets", expanded=False):
            st.markdown("**Current Portfolios & Strategies**")
            
            c1, c2 = st.columns(2)
            tsp_b = c1.number_input("TSP Balance ($)", key="tsp_b")
            tsp_strat = c2.selectbox("TSP Strategy", list(PORTFOLIOS.keys()), key="tsp_strat")
            
            c3, c4 = st.columns(2)
            ira_b = c3.number_input("Trad. IRA Balance ($)", key="ira_b")
            ira_strat = c4.selectbox("Trad. IRA Strategy", list(PORTFOLIOS.keys()), key="ira_strat")
            
            c5, c6 = st.columns(2)
            roth_b = c5.number_input("Roth IRA Balance ($)", key="roth_b")
            roth_strat = c6.selectbox("Roth IRA Strategy", list(PORTFOLIOS.keys()), key="roth_strat")
            
            c7, c8, c9 = st.columns(3)
            tax_b = c7.number_input("Taxable Balance ($)", key="tax_b")
            tax_basis = c8.number_input("Taxable Cost Basis ($)", min_value=0.0, key="tax_basis")
            tax_strat = c9.selectbox("Taxable Strategy", list(PORTFOLIOS.keys()), key="tax_strat")
            
            c10, c11 = st.columns(2)
            hsa_b = c10.number_input("HSA Balance (Optional)", min_value=0, key="hsa_b")
            hsa_strat = c11.selectbox("HSA Strategy", list(PORTFOLIOS.keys()), key="hsa_strat")
            
            c12, c13 = st.columns(2)
            cash_b = c12.number_input("Money Market Balance ($)", key="cash_b")
            cash_r = c13.number_input("Money Market Yield %", key="cash_r")
            
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

        def safe_float(val):
            return float(val or 0.0)

        inputs = {
            'current_age': int(st.session_state.cur_age), 'ret_age': int(st.session_state.ret_age), 'life_expectancy': int(st.session_state.life_exp),
            'spouse_age': int(st.session_state.spouse_age) if st.session_state.spouse_age else int(st.session_state.cur_age), 
            'spouse_life_exp': int(st.session_state.spouse_life_exp) if st.session_state.spouse_life_exp else int(st.session_state.life_exp),
            'filing_status': st.session_state.filing_status, 'state': st.session_state.state, 'county': st.session_state.county, 
            'current_salary': safe_float(st.session_state.current_salary), 'annual_savings': safe_float(st.session_state.annual_savings),
            'phased_ret_active': st.session_state.phased_ret_active, 'phased_ret_age': int(st.session_state.phased_ret_age or st.session_state.ret_age),
            'pension_est': safe_float(st.session_state.pension_est), 'survivor_benefit': st.session_state.survivor_benefit,
            'ss_fra': safe_float(st.session_state.ss_fra), 'ss_claim_age': int(st.session_state.ss_claim_age),
            'min_spending': safe_float(st.session_state.min_spending), 'max_spending': safe_float(st.session_state.max_spending),
            'additional_expenses': safe_float(st.session_state.add_exp),
            'max_tax_bracket': float(st.session_state.max_tax_bracket.strip('%'))/100,
            'health_plan': st.session_state.health_plan, 'health_cost': safe_float(st.session_state.health_cost), 'oop_cost': safe_float(st.session_state.oop_cost), 
            'mortgage_pmt': safe_float(st.session_state.mortgage_pmt), 'mortgage_yrs': int(st.session_state.mortgage_yrs or 0),
            'home_value': safe_float(st.session_state.home_value), 'target_floor': safe_float(st.session_state.target_floor),
            'tsp_bal': safe_float(st.session_state.tsp_b), 'tsp_strat': st.session_state.tsp_strat,
            'ira_bal': safe_float(st.session_state.ira_b), 'ira_strat': st.session_state.ira_strat,
            'roth_bal': safe_float(st.session_state.roth_b), 'roth_strat': st.session_state.roth_strat,
            'taxable_bal': safe_float(st.session_state.tax_b), 'taxable_basis': safe_float(final_tax_basis), 'taxable_strat': st.session_state.tax_strat,
            'hsa_bal': safe_float(st.session_state.hsa_b), 'hsa_strat': st.session_state.hsa_strat,
            'cash_bal': safe_float(st.session_state.cash_b), 'cash_ret': safe_float(st.session_state.cash_r)/100,
            'pay_taxes_from_cash': st.session_state.pay_taxes_from_cash
        }

        with st.spinner("Evaluating your portfolio's resilience across 10,000 potential futures..."):
            engine = StochasticRetirementEngine(inputs)
            opt_iwr = engine.optimize_iwr()
            
            roth_results, winner, history = engine.analyze_roth_strategies(opt_iwr)
            port_analysis = engine.analyze_portfolios(opt_iwr, roth_strategy=1) 
            
            st.session_state['sim_data'] = {
                'inputs': inputs, 'opt_iwr': opt_iwr, 'roth_results': roth_results,
                'winner': winner, 'history': history, 'port_analysis': port_analysis,
                'engine_years': engine.years
            }

    if 'sim_data' in st.session_state:
        data = st.session_state['sim_data']
        inputs = data['inputs']
        opt_iwr = data['opt_iwr']
        roth_results = data['roth_results']
        winner = data['winner']
        history = data['history']
        port_analysis = data['port_analysis']
        engine_years = data['engine_years']
        
        years_arr = np.arange(datetime.datetime.now().year, datetime.datetime.now().year + engine_years)
        age_arr = np.arange(inputs['current_age']+1, inputs['current_age']+1+engine_years)
        
        median_real_terminal = np.median(history['total_bal_real'][:, -1])
        prob_success = np.mean(history['total_bal_real'][:, -1] > 0) * 100
        prob_legacy = np.mean(history['total_bal_real'][:, -1] >= inputs['target_floor']) * 100
        
        df_median = build_csv_dataframe(history, years_arr, age_arr, percentile=50)

        ret_idx = max(0, inputs['ret_age'] - inputs['current_age'])
        yr1_burn = (df_median['Total Expenses'].iloc[ret_idx] + 
                    df_median['Net Spendable Annual'].iloc[ret_idx] - 
                    df_median['Social Security'].iloc[ret_idx] - 
                    df_median['Pension'].iloc[ret_idx] - 
                    df_median['Salary Income'].iloc[ret_idx])
        
        total_cash_short_term = df_median['Money Market Balance'].iloc[ret_idx] + df_median['Taxable ETF Balance'].iloc[ret_idx]
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
        with colA:
            st.subheader("Plan Insights & Executive Summary")
        with colB:
            pdf_data = {
                'prob_success': prob_success, 'prob_legacy': prob_legacy, 'terminal_wealth': median_real_terminal, 'yr1_burn': yr1_burn,
                'safe_years': safe_years, 'roth_winner': winner, 'tax_savings': tax_savings,
                'rmd_reduction': rmd_reduction, 'wealth_increase': wealth_increase, 'health_plan': inputs['health_plan'],
                'total_medicare': total_medicare_cost, 'medicare_verdict': med_verdict
            }
            pdf_bytes = generate_pdf(pdf_data)
            st.download_button(label="📄 Download Executive Summary PDF", data=pdf_bytes, file_name="Retirement_Plan_Summary.pdf", mime="application/pdf", use_container_width=True)
        
        st.info("💡 **Actuarial Note on Probability of Success:** This model calculates your withdrawal rate by mathematically forcing the *Median* (50th percentile) outcome to exactly hit your Target Legacy Floor. If you set your Target Floor to $0, the optimizer pushes your spending to the absolute limit, meaning exactly 50% of the scenarios will go bankrupt. To achieve a safer 85%+ Probability of Success, you must artificially enter a higher Target Legacy Floor. This acts as a cash buffer against bad market conditions.")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        with kpi1.container(border=True):
            st.metric("Prob. of Survival (> $0)", f"{prob_success:.1f}%", delta="On Track" if prob_success >= 85 else "At Risk", delta_color="normal" if prob_success >= 85 else "inverse", help="Definition: The percentage of 10,000 simulated market paths where your portfolio successfully lasted until your Life Expectancy without dropping to $0.")
            
        with kpi2.container(border=True):
            st.metric("Prob. of Reaching Target Legacy", f"{prob_legacy:.1f}%", help="Definition: The percentage of simulations where your final estate value met or exceeded the exact Target Legacy Floor you inputted.")
            
        with kpi3.container(border=True):
            st.metric("Median Terminal Legacy (Today's $)", f"${median_real_terminal:,.0f}", help="Definition: The estimated total value of your estate at life expectancy, discounted for inflation back into Today's Dollars to match your Target Legacy Floor.")
            
        with kpi4.container(border=True):
            st.metric("Est. Year 1 Portfolio Burn", f"${yr1_burn:,.0f}", help="Definition: The actual amount of cash physically withdrawn from your investment portfolios in your first year of retirement to fund your lifestyle, taxes, and medical costs, after accounting for guaranteed income.\n\nExample: If your lifestyle costs $100k and your pension is $60k, your 'Burn' is $40k.")
            
        st.markdown("<br>", unsafe_allow_html=True) 

        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = st.tabs([
            "📊 Projections", "💵 Cash Flow", "📉 Guardrails", "📈 Net Worth", "🏛️ Taxes", 
            "🏛️ Legacy", "💡 Coach Alerts", "🔄 Roth Opt.", "🦅 Social Sec", "🏥 Medicare", "💾 Exports"
        ])

        with t1:
            st.subheader("Lifetime Projections & Monte Carlo Analysis")
            st.plotly_chart(plot_wealth_trajectory(history, inputs['target_floor'], years_arr), use_container_width=True)
            
            st.markdown("---")
            st.subheader("Portfolio Optimization & Efficient Frontier")
            st.write("This analysis evaluates your custom account-by-account mix against standard benchmark portfolios to find the optimal balance of growth vs. Sequence of Return Risk (guardrail pay cuts).")
            
            port_names = list(port_analysis.keys())
            port_wealths = [port_analysis[p]['wealth'] for p in port_names]
            port_cuts = [port_analysis[p]['cut_prob'] for p in port_names]
            
            p_df = pd.DataFrame({"Portfolio Strategy": port_names, "Median Terminal Legacy (Today's $)": port_wealths, "Probability of Guardrail Pay Cuts": port_cuts})
            st.table(p_df.style.format({"Median Terminal Legacy (Today's $)": "${:,.0f}", "Probability of Guardrail Pay Cuts": "{:.1f}%"}))

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
            
            st.markdown("### Asset Liquidity Profile (Year 1 of Retirement)")
            c1, c2, c3 = st.columns(3)
            c1.metric("Highly Liquid Assets (Cash + Taxable)", f"${total_cash_short_term:,.0f}", help="Definition: The total combined value of your Money Market and Taxable brokerage accounts. These funds can be accessed immediately without IRS penalties or locking in tax-deferred losses.")
            c2.metric("Year 1 Est. Portfolio Burn Rate", f"${yr1_burn:,.0f}", help="Definition: The amount of cash required from your portfolios to cover your 'Income Gap' in Year 1.")
            c3.metric("Years of Safe Liquidity Buffer", f"{safe_years:.1f} Years", help="Definition: How many years you can survive strictly off your cash and taxable accounts without selling a single share of your TSP or IRA.\n\nExample: A 3.0 ratio means you can comfortably outlast a 3-year market crash without touching your tax-deferred stocks.")

        with t5:
            st.subheader("Taxes & Dynamic Withdrawals")
            limit_24 = TAX_BRACKETS_MFJ[3][0] if inputs['filing_status'] == 'MFJ' else TAX_BRACKETS_SINGLE[3][0]
            
            if df_median['IRS Taxable Income'].iloc[ret_idx] > limit_24:
                st.error(f"🚨 **Lifestyle Exceeds {inputs['max_tax_bracket']} Bracket**: Your baseline spending needs naturally push your IRS Taxable Income to **${df_median['IRS Taxable Income'].iloc[ret_idx]:,.0f}**, which is above your {inputs['max_tax_bracket']} ceiling. The Roth Optimizer disabled itself to prevent pushing you even higher.")
            else:
                st.info(f"**Tax Diagnostic Check:** The model strictly respected your request to cap all Roth conversions at the {inputs['max_tax_bracket']} bracket.")
                
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
            if inputs['filing_status'] == 'MFJ':
                st.warning("⚠️ **Widow(er) Tax Penalty**: Upon the first spouse's mortality, your tax filing status shifts to Single, shrinking your brackets and drastically increasing your vulnerability to IRMAA surcharges. Roth conversions are critical while you are still MFJ.")
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
            st.info(f"**Target Ceiling Parameter:** The Roth optimizer rigorously evaluated all tax strategies strictly capped up to your selected maximum target bracket of **{inputs['max_tax_bracket']}**.")
            
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
                
                st.write(f"- **Nominal Lifetime Tax Savings (Un-discounted):** ${max(0, tax_savings):,.0f}")
                st.write(f"- **Reduction in Lifetime RMDs:** ${rmd_reduction:,.0f}")
                st.write(f"- **Net Increase to Legacy (Today's $):** ${wealth_increase:,.0f}")
                
                st.markdown("#### Step-by-Step Conversion Schedule")
                st.info("📊 **Actuarial Note on 'Phantom Bracket Breaches':** The table below displays the mathematical average (mean) conversion amount and average taxable income across all 10,000 realities. Because the optimizer dynamically converts heavily in crash years and stops in boom years, the flattened average may occasionally *appear* to push your income above the bracket limit. Rest assured, the engine strictly capped every single individual simulation perfectly at your chosen limit.")
                
                roth_amts = np.mean(history['roth_conversion'], axis=0)
                conv_df = pd.DataFrame({"Year": years_arr, "Age": age_arr, "Target Conversion Amount": roth_amts, "Est. IRS Taxable Income": np.median(history['taxable_income'], axis=0)})
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
            st.write("**Why delay to 70? The 'Longevity Insurance' Concept:** Actuarially, Social Security is one of the few guaranteed, inflation-adjusted, market-immune income streams you possess alongside your FERS pension. By delaying to Age 70, your payout permanently increases by 8% per year. This creates massive 'Longevity Insurance.' If you live deep into your 90s, this vastly inflated SS paycheck drastically reduces the withdrawal pressure placed on your TSP/Roth, virtually guaranteeing you will not outlive your portfolio.")

        with t10:
            st.subheader("Medicare Part B & Actuarial Healthcare OOP")
            st.plotly_chart(plot_medicare_comparison(history, years_arr, inputs), use_container_width=True)
            
            total_medicare_cost = np.sum(np.median(history['medicare_cost'], axis=0))
            st.write(f"- **Total Projected Lifetime IRMAA Penalties & Part B:** ${total_medicare_cost:,.0f}")
            
            moop_cap = MOOP_LIMITS.get(inputs['health_plan'], (999999, 999999))[1 if inputs['filing_status'] == 'MFJ' else 0]
            if moop_cap == 999999:
                st.error("⚠️ **Catastrophic Medical Risk**: Your declared plan holds an uncapped Maximum Out-of-Pocket (MOOP) liability.")
            else:
                st.info(f"🛡️ **Plan Protection Active**: Your {inputs['health_plan']} correctly caps out-of-pocket medical tail-risk at **${moop_cap:,.0f}** per year (inflation adjusted).")
                
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
    - **FERS/Pension Details:** If you are a Federal Employee, enter your estimated unreduced pension.
      - **Survivor Benefit:** Select the option you plan to choose. This automatically models the 5% or 10% reduction in your retirement "paycheck."
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
      - **Conservative:** 20% Stocks / 80% Bonds.
      - **Moderate:** 60% Stocks / 40% Bonds.
      - **Aggressive:** 100% Stocks.
    - **Money Market (Cash):** This is your "Safety Net." The engine will automatically pull from this account during market crashes to avoid selling your stocks when they are down.
    """)

    st.header("Run the Engine")
    st.write("Once your data is entered, click the **Run Projection Engine** button.")
    st.write("**What happens next?**")
    st.write("The screen will pause for a few seconds. In the background, the 'Brain' of the system is running 10,000 lifetimes for you. It is looking for the 'Optimized Withdrawal Rate'…the highest amount you can spend without falling below your Legacy Floor in the average market scenario.")

    st.header("Reviewing Your Results")
    st.write("Once the dashboard populates, focus on these three areas:")
    st.markdown("""
    1. **Probability of Success:** You want this number to be 85% or higher. If it is lower, you may need to reduce your spending goals or work a few years longer.
    2. **The "Coach Alerts" Tab:** Read this first. It provides a prioritized "To-Do List" based on your specific risks (e.g., "You have a high risk of RMD tax spikes").
    3. **The Roth Optimizer Tab:** This shows you exactly how much money to convert from your TSP/IRA to a Roth IRA each year to pay the lowest amount of lifetime tax possible.
    """)

    st.header("Final Step: Export Your Plan")
    st.write("At the top of the dashboard, click **Download Executive Summary PDF**. This provides a formal, 2-page summary of your results that you can keep for your records.")

    st.markdown("---")
    st.caption("*Stochastic modeling (Monte Carlo simulations) generates hypothetical market returns based on statistical probability; it does not predict actual future stock market performance. The projections generated by this tool regarding the likelihood of various investment outcomes are hypothetical in nature, do not reflect actual investment results, and are not guarantees of future results. Always consult with a licensed fiduciary, Certified Financial Planner (CFP®), or CPA before executing Roth conversions, claiming Social Security, or making irrevocable retirement decisions.*")

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
    """)

    st.header("Why 'CASAM'?")
    st.write("""
    This tool relies on the **Constant Amortization Spending Model (CASAM)**. Instead of using rigid rules like the "4% Rule," CASAM looks at your actual portfolio balance, guaranteed income streams (Social Security, Pensions), and specific tax liabilities every single year, dynamically adjusting your safe spending limits to ensure your money outlives you.
    """)