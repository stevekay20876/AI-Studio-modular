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
components.html("<script>window.parent.document.body.scrollTop=0; window.parent.document.documentElement.scrollTop=0;</script>", height=0, width=0)

st.markdown("""<style>#MainMenu {visibility: hidden;} footer {display: none !important;} [data-testid="stHeader"] {visibility: hidden;} .stAppBottom {display: none !important;} .block-container { padding-top: 2rem; padding-bottom: 2rem; } [data-testid="stMetricValue"] { font-size: 2.0rem !important; font-weight: 700 !important; color: #00837B !important; } [data-testid="stDownloadButton"] button {background-color: #E6F7F6 !important; color: #00695C !important; border: 2px solid #80CBC4 !important; font-weight: 700 !important; border-radius: 8px !important; transition: all 0.2s ease;} [data-testid="stDownloadButton"] button:hover {background-color: #B2DFDB !important; border-color: #00837B !important;} [data-testid="stTabs"] { background-color: #F8FAFC; border: 2px solid #E5E7EB; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); } div[data-baseweb="tab-list"] {gap: 0px; border-bottom: 2px solid #E5E7EB;} button[data-baseweb="tab"] { font-size: 1.1rem !important; padding: 0.8rem 1.5rem !important; background-color: #E5E7EB !important; color: #475569 !important; border-radius: 8px 8px 0 0 !important; border: 1px solid transparent !important; margin-right: 4px !important;} button[data-baseweb="tab"][aria-selected="true"] { background-color: #FFFFFF !important; color: #00837B !important; font-weight: 800 !important; border-top: 4px solid #00837B !important; border-left: 2px solid #E5E7EB !important; border-right: 2px solid #E5E7EB !important; border-bottom: 2px solid #FFFFFF !important; transform: translateY(2px); box-shadow: 0 -2px 4px rgba(0,0,0,0.02);} [data-testid="stVerticalBlockBorderWrapper"] { border-radius: 12px !important; border: 1px solid #E5E7EB !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03) !important; background-color: #FFFFFF !important; }</style>""", unsafe_allow_html=True)

nav1, nav2, nav3, nav4 = st.tabs(["📊 Main Dashboard", "📝 Instructions", "⚙️ Background & Methodology", "ℹ️ About"])

with nav1:
    st.title("Advanced Quantitative Retirement Planner")
    st.markdown("Institution-Grade Monte Carlo Simulator | Constant Amortization Spending Model (CASAM)")

    DEFAULT_STATE = {'cur_age': None, 'ret_age': None, 'life_exp': None, 'filing_status': "Single", 'spouse_age': None, 'spouse_life_exp': None, 'state': "", 'county': "", 'current_salary': None, 'annual_savings': None, 'phased_ret_active': False, 'phased_ret_age': None, 'pension_est': None, 'survivor_benefit': "Full Survivor Benefit", 'mil_pension_est': 0.0, 'mil_pension_start_age': None, 'mil_survivor_benefit': "No SBP", 'ss_fra': None, 'ss_claim_age': 67, 'target_floor': None, 'min_spending': None, 'max_spending': None, 'add_exp': None, 'max_tax_bracket': "24%", 'mortgage_pmt': None, 'mortgage_yrs': None, 'home_value': None, 'health_plan': "None/Self-Insure", 'health_cost': None, 'oop_cost': None, 'tsp_b': 0.0, 'tsp_strat': "Moderate (60% Stock / 40% Bond)", 'ira_b': 0.0, 'ira_strat': "Moderate (60% Stock / 40% Bond)", 'roth_b': 0.0, 'roth_strat': "Aggressive (100% Stock)", 'tax_b': 0.0, 'tax_basis': None, 'tax_strat': "Moderate (60% Stock / 40% Bond)", 'hsa_b': None, 'hsa_strat': "Moderate (60% Stock / 40% Bond)", 'cash_b': 0.0, 'cash_r': 4.0, 'pay_taxes_from_cash': True}
    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def get_current_state():
        return {k: st.session_state[k] for k in DEFAULT_STATE.keys() if k in st.session_state}

    with st.expander("💾 Client Profile Management (Save / Load)", expanded=False):
        col_load, col_save = st.columns(2)
        with col_load:
            uploaded_profile = st.file_uploader("Load Saved Profile (.json)", type="json")
            if uploaded_profile is not None:
                try:
                    loaded_data = json.load(uploaded_profile)
                    for key, value in loaded_data.items():
                        st.session_state[key] = value
                    st.success("Profile Loaded Successfully!")
                    st.rerun() 
                except Exception as e:
                    st.error("Error loading profile.")
        with col_save:
            st.write("") 
            st.write("")
            st.download_button(label="⬇️ Save Current Profile to Computer", data=json.dumps(get_current_state(), indent=4), file_name="client_profile.json", mime="application/json", use_container_width=True)

    has_run = 'sim_data' in st.session_state

    with st.form("input_form"):
        with st.expander("👤 Personal & Tax Details", expanded=not has_run):
            c1, c2, c3 = st.columns(3)
            cur_age = c1.number_input("Current Age", min_value=18, max_value=100, key="cur_age")
            ret_age = c2.number_input("Full Retirement Age", min_value=18, max_value=100, key="ret_age")
            life_exp = c3.number_input("Primary Life Expectancy Age", min_value=50, max_value=120, key="life_exp")
            c4, c5, c6 = st.columns(3)
            filing_status = c4.selectbox("Tax Filing Status", ["Single", "MFJ"], key="filing_status")
            state_in = c5.text_input("State of Residence", key="state")
            county_in = c6.text_input("County of Residence", key="county")
            c_sp1, c_sp2 = st.columns(2)
            spouse_age = c_sp1.number_input("Spouse Current Age (If MFJ)", min_value=18, max_value=100, key="spouse_age")
            spouse_life_exp = c_sp2.number_input("Spouse Life Expectancy (If MFJ)", min_value=50, max_value=120, key="spouse_life_exp")

        with st.expander("💼 Income & Social Security", expanded=not has_run):
            c1, c2 = st.columns(2)
            current_salary = c1.number_input("Current Annual Salary ($)", min_value=0, key="current_salary")
            annual_savings = c2.number_input("Total Annual Savings (Until Ret.) ($)", min_value=0, key="annual_savings")
            c3, c4 = st.columns(2)
            phased_ret_active = c3.checkbox("Enable FERS Phased Retirement?", key="phased_ret_active")
            phased_ret_age = c4.number_input("Phased Retirement Start Age", min_value=50, max_value=70, key="phased_ret_age")
            c5, c6 = st.columns(2)
            pension_est = c5.number_input("Full (Unreduced) FERS Pension Est. ($)", min_value=0, key="pension_est")
            survivor_benefit = c6.selectbox("FERS Survivor Benefit Option", ["Full Survivor Benefit", "Partial Survivor Benefit", "No Survivor Benefit"], key="survivor_benefit")
            c_mil1, c_mil2, c_mil3 = st.columns(3)
            mil_pension_est = c_mil1.number_input("Annual Mil. Pension Est. ($)", min_value=0, key="mil_pension_est")
            mil_pension_start_age = c_mil2.number_input("Mil. Pension Start Age", min_value=18, max_value=100, key="mil_pension_start_age")
            mil_survivor_benefit = c_mil3.selectbox("Survivor Benefit Plan (SBP)", ["No SBP", "Full SBP (55% Survivor / 6.5% Premium)"], key="mil_survivor_benefit")
            c7, c8 = st.columns(2)
            ss_fra = c7.number_input("Social Security at FRA ($/yr)", min_value=0, key="ss_fra")
            ss_claim_age = c8.number_input("Target SS Claiming Age", min_value=62, max_value=70, key="ss_claim_age")

        with st.expander("📉 Expenses & Goals", expanded=not has_run):
            c1, c2, c3 = st.columns(3)
            target_floor = c1.number_input("Target Legacy Floor (Today's $) ($)", min_value=0, key="target_floor")
            min_spending = c2.number_input("Minimum Spending Floor (Today's $) ($)", min_value=0, key="min_spending")
            max_spending = c3.number_input("Maximum Spending Cap (Today's $) ($)", min_value=0, key="max_spending")
            c4, c5 = st.columns(2)
            add_exp = c4.number_input("Additional Expenses (Retirement Smile) ($)", min_value=0, key="add_exp")
            max_tax_bracket = c5.selectbox("Maximum Target Tax Bracket (Roth Cap)", ["12%", "22%", "24%", "32%", "35%", "37%"], index=2, key="max_tax_bracket")
            c6, c7, c8 = st.columns(3)
            home_value = c6.number_input("Current Home Value ($)", min_value=0, key="home_value")
            mortgage_pmt = c7.number_input("Annual Mortgage Payment ($)", min_value=0, key="mortgage_pmt")
            mortgage_yrs = c8.number_input("Mortgage Years Remaining", min_value=0, key="mortgage_yrs")
            c9, c10, c11 = st.columns(3)
            health_plan = c9.selectbox("Retiree Health Coverage", ["FEHB FEPBlue Basic", "FEPBlue Standard", "FEPBlue Focus", "GEHA High", "GEHA Standard", "Aetna Open Access", "Aetna Direct", "Aetna Advantage", "Cigna", "TRICARE for Life", "None/Self-Insure"], key="health_plan")
            health_cost = c10.number_input("Annual Health Premium ($)", min_value=0, key="health_cost")
            oop_cost = c11.number_input("Typical Out-of-Pocket Medical ($)", min_value=0, key="oop_cost")

        with st.expander("🏛️ Savings & Assets", expanded=not has_run):
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
            pay_taxes_from_cash = st.checkbox("Pay Roth Conversion Taxes from Cash Buffer?", key="pay_taxes_from_cash")
        
        submit = st.form_submit_button("Run Projection Engine", type="primary")

    if submit:
        final_tax_basis = st.session_state.tax_basis if st.session_state.tax_basis is not None else st.session_state.tax_b
        missing_vitals = [name for name, val in {"Current Age": st.session_state.cur_age, "Retirement Age": st.session_state.ret_age, "Life Expectancy": st.session_state.life_exp, "Target Legacy Floor": st.session_state.target_floor}.items() if val is None or val == 0]
        if missing_vitals:
            st.error(f"SYSTEM HALTED: You must explicitly provide values for: {', '.join(missing_vitals)}")
            st.stop()

        def safe_float(val): return float(val or 0.0)

        inputs = {
            'current_age': int(st.session_state.cur_age), 'ret_age': int(st.session_state.ret_age), 'life_expectancy': int(st.session_state.life_exp),
            'spouse_age': int(st.session_state.spouse_age) if st.session_state.spouse_age else int(st.session_state.cur_age), 
            'spouse_life_exp': int(st.session_state.spouse_life_exp) if st.session_state.spouse_life_exp else int(st.session_state.life_exp),
            'filing_status': st.session_state.filing_status, 'state': st.session_state.state, 'county': st.session_state.county, 
            'current_salary': safe_float(st.session_state.current_salary), 'annual_savings': safe_float(st.session_state.annual_savings),
            'phased_ret_active': st.session_state.phased_ret_active, 'phased_ret_age': int(st.session_state.phased_ret_age or st.session_state.ret_age),
            'pension_est': safe_float(st.session_state.pension_est), 'survivor_benefit': st.session_state.survivor_benefit,
            'mil_pension_est': safe_float(st.session_state.mil_pension_est), 'mil_pension_start_age': int(st.session_state.mil_pension_start_age or st.session_state.cur_age), 'mil_survivor_benefit': st.session_state.mil_survivor_benefit,
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
            st.session_state['sim_data'] = {'inputs': inputs, 'opt_iwr': opt_iwr, 'roth_results': roth_results, 'winner': winner, 'history': history, 'port_analysis': port_analysis, 'engine_years': engine.years}
            st.rerun()

    if 'sim_data' in st.session_state:
        data = st.session_state['sim_data']
        inputs, opt_iwr, roth_results, winner, history, port_analysis, engine_years = data['inputs'], data['opt_iwr'], data['roth_results'], data['winner'], data['history'], data['port_analysis'], data['engine_years']
        
        years_arr = np.arange(datetime.datetime.now().year, datetime.datetime.now().year + engine_years)
        age_arr = np.arange(inputs['current_age']+1, inputs['current_age']+1+engine_years)
        
        median_real_terminal = np.median(history['total_bal_real'][:, -1])
        prob_success = np.mean(history['total_bal_real'][:, -1] > 0) * 100
        prob_legacy = np.mean(history['total_bal_real'][:, -1] >= inputs['target_floor']) * 100

        ret_idx = max(0, inputs['ret_age'] - inputs['current_age'])
        yr1_burn = np.median(history['taxes_fed'] + history['taxes_state'] + history['medicare_cost'] + history['health_cost'] + history['mortgage_cost'] + history['additional_expenses'], axis=0)[ret_idx] + np.median(history['net_spendable'], axis=0)[ret_idx] - np.median(history['ss_income'], axis=0)[ret_idx] - np.median(history['pension_income'], axis=0)[ret_idx] - np.median(history['salary_income'], axis=0)[ret_idx]
        
        total_cash_short_term = np.median(history['cash_bal'], axis=0)[ret_idx] + np.median(history['taxable_bal'], axis=0)[ret_idx]
        safe_years = total_cash_short_term / max(yr1_burn, 1)

        tax_savings = roth_results['Baseline (None)']['taxes'] - roth_results[winner]['taxes']
        rmd_reduction = roth_results['Baseline (None)']['rmds'] - roth_results[winner]['rmds']
        wealth_increase = roth_results[winner]['wealth'] - roth_results['Baseline (None)']['wealth']
        total_medicare_cost = np.sum(np.median(history['medicare_cost'], axis=0))
        
        colA, colB = st.columns([0.8, 0.2])
        with colA: st.subheader("Plan Insights & Executive Summary")
        with colB: st.download_button(label="📄 Download Executive Summary PDF", data=generate_pdf({'prob_success': prob_success, 'prob_legacy': prob_legacy, 'terminal_wealth': median_real_terminal, 'yr1_burn': yr1_burn, 'safe_years': safe_years, 'roth_winner': winner, 'tax_savings': tax_savings, 'rmd_reduction': rmd_reduction, 'wealth_increase': wealth_increase, 'health_plan': inputs['health_plan'], 'total_medicare': total_medicare_cost, 'medicare_verdict': "Waive Part B" if "FEHB" in inputs['health_plan'] or "TRICARE" in inputs['health_plan'] else "Enroll in Medicare Part B"}), file_name="Retirement_Plan_Summary.pdf", mime="application/pdf", use_container_width=True)
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.container(border=True).metric("Prob. of Survival (> $0)", f"{prob_success:.1f}%", delta="On Track" if prob_success >= 85 else "At Risk", delta_color="normal" if prob_success >= 85 else "inverse")
        kpi2.container(border=True).metric("Prob. of Reaching Target Legacy", f"{prob_legacy:.1f}%")
        kpi3.container(border=True).metric("Median Terminal Legacy (Today's $)", f"${median_real_terminal:,.0f}")
        kpi4.container(border=True).metric("Est. Year 1 Portfolio Burn", f"${yr1_burn:,.0f}")

        t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11 = st.tabs(["📊 Projections", "💵 Cash Flow", "📉 Guardrails", "📈 Net Worth", "🏛️ Taxes", "🏛️ Legacy", "💡 Coach Alerts", "🔄 Roth Opt.", "🦅 Social Sec", "🏥 Medicare", "💾 Exports"])

        with t1:
            st.plotly_chart(plot_wealth_trajectory(history, inputs['target_floor'], years_arr), use_container_width=True)
            st.table(pd.DataFrame({"Portfolio Strategy": list(port_analysis.keys()), "Median Terminal Legacy (Today's $)": [port_analysis[p]['wealth'] for p in port_analysis.keys()], "Probability of Guardrail Pay Cuts": [port_analysis[p]['cut_prob'] for p in port_analysis.keys()]}).style.format({"Median Terminal Legacy (Today's $)": "${:,.0f}", "Probability of Guardrail Pay Cuts": "{:.1f}%"}))

        with t2:
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_fan_chart(history, years_arr), use_container_width=True)
            with col2: st.plotly_chart(plot_income_gap(history, years_arr), use_container_width=True)
            df_ui = build_csv_dataframe(history, years_arr, age_arr, percentile=50)
            display_cols = [c for c in ['Calendar Year', 'Age', 'Total Income', 'IRS Taxable Income', 'Total Expenses', 'Net Spendable Annual', 'TSP Withdrawal', 'Trad IRA Withdrawal', 'Salary Income', 'Social Security', 'Pension', 'Additional Expenses (Smile Curve)'] if c in df_ui.columns]
            st.dataframe(df_ui[display_cols].style.format({c: "${:,.0f}" for c in display_cols if c not in ['Calendar Year', 'Age']}), use_container_width=True, hide_index=True)

        with t3:
            st.plotly_chart(plot_expenses_breakdown(history, years_arr), use_container_width=True)
            st.plotly_chart(plot_income_volatility(history, years_arr), use_container_width=True)

        with t4:
            st.plotly_chart(plot_liquidity_timeline(history, years_arr), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Highly Liquid Assets (Cash + Taxable)", f"${total_cash_short_term:,.0f}")
            c2.metric("Year 1 Est. Portfolio Burn Rate", f"${yr1_burn:,.0f}")
            c3.metric("Years of Safe Liquidity Buffer", f"{safe_years:.1f} Years")

        with t5:
            if np.median(history['taxable_income'], axis=0)[ret_idx] > (TAX_BRACKETS_MFJ[3][0] if inputs['filing_status'] == 'MFJ' else TAX_BRACKETS_SINGLE[3][0]):
                st.error(f"🚨 Lifestyle Exceeds {inputs['max_tax_bracket']} Bracket")
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_withdrawal_hierarchy(history, years_arr), use_container_width=True)
            with col2: st.plotly_chart(plot_taxes_and_rmds(history, years_arr), use_container_width=True)

        with t6:
            st.plotly_chart(plot_legacy_breakdown(history), use_container_width=True)

        with t7:
            if np.median(history['taxes_fed'], axis=0)[-1] > np.median(history['taxes_fed'], axis=0)[0] * 2.5: st.warning("⚠️ RMD Tax Spike Alert")
            if inputs['filing_status'] == 'MFJ': st.warning("⚠️ Widow(er) Tax Penalty")

        with t8:
            col1, col2 = st.columns(2)
            with col1: st.plotly_chart(plot_roth_strategy_comparison(roth_results), use_container_width=True)
            with col2: st.plotly_chart(plot_roth_tax_impact(roth_results, winner, years_arr), use_container_width=True)
            st.write(f"Verdict: Execute {winner}")
            conv_df = pd.DataFrame({"Year": years_arr, "Age": age_arr, "Target Conversion Amount": np.mean(history['roth_conversion'], axis=0), "Est. IRS Taxable Income": np.median(history['taxable_income'], axis=0)})
            st.table(conv_df[conv_df['Target Conversion Amount'] > 0].style.format({"Target Conversion Amount": "${:,.0f}", "Est. IRS Taxable Income": "${:,.0f}"}))

        with t9:
            st.plotly_chart(plot_ss_breakeven(inputs['ss_fra'], age_arr), use_container_width=True)

        with t10:
            st.plotly_chart(plot_medicare_comparison(history, years_arr, inputs), use_container_width=True)

        with t11:
            def format_df_for_csv(df_raw):
                df_out = df_raw.copy()
                pct_cols = ["Rate of Return", "Inflation Rate", "Real Rate of Return", "Cumulative Inflation Multiplier"]
                for c in pct_cols:
                    if c in df_out.columns: df_out[c] = df_out[c].apply(lambda x: f"{x:.2%}")
                currency_cols = [c for c in df_out.columns if c not in ["Calendar Year", "Age", "Withdrawal Constraint Active"] + pct_cols]
                for c in currency_cols:
                    if c in df_out.columns: df_out[c] = df_out[c].apply(lambda x: f"${x:,.0f}")
                return df_out

            colA, colB = st.columns(2)
            colA.download_button("📄 Download Median CSV", format_df_for_csv(build_csv_dataframe(history, years_arr, age_arr, 50)).to_csv(index=False), "Retirement_Median.csv", "text/csv")
            colB.download_button("📄 Download Pessimistic CSV", format_df_for_csv(build_csv_dataframe(history, years_arr, age_arr, 10)).to_csv(index=False), "Retirement_Pess.csv", "text/csv")

with nav2:
    st.title("Instructions")
with nav3:
    st.title("Methodology")
with nav4:
    st.title("About")