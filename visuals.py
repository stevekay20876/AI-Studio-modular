# visuals.py
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_wealth_trajectory(history, target_floor, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['total_bal'], axis=0), mode='lines', name='Median (50th)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=years_arr, y=np.percentile(history['total_bal'], 90, axis=0), mode='lines', name='Optimistic (90th)', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=years_arr, y=np.percentile(history['total_bal'], 10, axis=0), mode='lines', name='Pessimistic (10th)', line=dict(color='red', dash='dash')))
    fig.add_hline(y=target_floor, line_dash="dot", line_color="black", annotation_text="Target Legacy Floor")
    fig.update_layout(title="Stochastic Portfolio Projections", xaxis_title="Year", yaxis_title="Balance ($)", hovermode="x unified")
    return fig

def plot_fan_chart(history, years_arr):
    p10 = np.percentile(history['total_bal'], 10, axis=0)
    p25 = np.percentile(history['total_bal'], 25, axis=0)
    p50 = np.median(history['total_bal'], axis=0)
    p75 = np.percentile(history['total_bal'], 75, axis=0)
    p90 = np.percentile(history['total_bal'], 90, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=p90, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=years_arr, y=p10, mode='lines', fill='tonexty', fillcolor='rgba(173, 216, 230, 0.2)', line=dict(width=0), name='10th-90th Percentile Range'))
    fig.add_trace(go.Scatter(x=years_arr, y=p75, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=years_arr, y=p25, mode='lines', fill='tonexty', fillcolor='rgba(70, 130, 180, 0.4)', line=dict(width=0), name='25th-75th Percentile Range'))
    fig.add_trace(go.Scatter(x=years_arr, y=p50, mode='lines', name='Median Path (50th)', line=dict(color='darkblue', width=3)))
    fig.update_layout(title="Sequence of Return Risk (Hurricane/Fan Chart)", xaxis_title="Year", yaxis_title="Portfolio Wealth ($)", hovermode="x unified")
    return fig

def plot_liquidity_timeline(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['home_value'], axis=0), mode='lines', stackgroup='one', name='Home Value', fillcolor='lightgray', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['tsp_bal'], axis=0), mode='lines', stackgroup='one', name='TSP'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['ira_bal'], axis=0), mode='lines', stackgroup='one', name='Trad IRA'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxable_bal'], axis=0), mode='lines', stackgroup='one', name='Taxable'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['roth_bal'], axis=0), mode='lines', stackgroup='one', name='Roth IRA'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['cash_bal'], axis=0), mode='lines', stackgroup='one', name='Cash Buffer'))
    fig.update_layout(title="Total Net Worth Forecast", xaxis_title="Year", yaxis_title="Balance ($)", hovermode="x unified")
    return fig

def plot_cash_flow_sources(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['salary_income'], axis=0), name="Salary", marker_color='purple'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['ss_income'], axis=0), name="Social Security", marker_color='#1f77b4'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['pension_income'], axis=0), name="Pension", marker_color='#ff7f0e'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['tsp_withdrawal'], axis=0), name="TSP Withdrawal", marker_color='#2ca02c'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['ira_withdrawal'], axis=0), name="IRA Withdrawal", marker_color='#98df8a'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['roth_withdrawal'], axis=0), name="Roth Withdrawal", marker_color='green'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['taxable_withdrawal'], axis=0), name="Taxable Withdrawal", marker_color='orange'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['cash_withdrawal'], axis=0), name="Cash Withdrawal", marker_color='gray'))
    
    total_need = np.median(history['net_spendable'] + history['taxes_fed'] + history['taxes_state'] + history['health_cost'] + history['medicare_cost'] + history['mortgage_cost'] + history['additional_expenses'], axis=0)
    fig.add_trace(go.Scatter(x=years_arr, y=total_need, mode='lines', name='Total Spending Need', line=dict(color='black', width=2)))
    fig.update_layout(barmode='stack', title="Income Sources vs Total Spending Need", xaxis_title="Year", yaxis_title="Amount ($)")
    return fig

def plot_income_gap(history, years_arr):
    guaranteed_income = np.median(history['salary_income'] + history['ss_income'] + history['pension_income'], axis=0)
    total_expenses = np.median(history['taxes_fed'] + history['health_cost'] + history['medicare_cost'] + history['mortgage_cost'] + history['additional_expenses'] + history['net_spendable'], axis=0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=total_expenses, mode='lines', name='Total Expenses / Lifestyle', line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=years_arr, y=guaranteed_income, mode='lines', fill='tozeroy', name='Guaranteed Base', line=dict(color='blue')))
    fig.update_layout(title="Income Gap Mapping", xaxis_title="Year", yaxis_title="Amount ($)")
    return fig

def plot_expenses_breakdown(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['taxes_fed'], axis=0), name="Federal/State Taxes", marker_color='crimson'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['medicare_cost'], axis=0), name="Medicare + IRMAA", marker_color='orange'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['health_cost'], axis=0), name="Health / OOP", marker_color='purple'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['mortgage_cost'], axis=0), name="Mortgage Payment", marker_color='brown'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['additional_expenses'], axis=0), name="Additional Expenses (Smile)", marker_color='pink'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['net_spendable'], axis=0), name="Discretionary Lifestyle", marker_color='teal'))
    fig.update_layout(barmode='stack', title="Itemized Core Expenses", xaxis_title="Year", yaxis_title="Amount ($)")
    return fig

def plot_income_volatility(history, years_arr):
    med_spend = np.median(history['net_spendable'], axis=0)
    high_spend = np.percentile(history['net_spendable'], 90, axis=0)
    low_spend = np.percentile(history['net_spendable'], 10, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=high_spend, mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=years_arr, y=low_spend, mode='lines', fill='tonexty', fillcolor='rgba(0,128,128,0.2)', line=dict(width=0), name='Spending Range (10th-90th)'))
    fig.add_trace(go.Scatter(x=years_arr, y=med_spend, mode='lines', name='Median Spending', line=dict(color='teal', width=3)))
    fig.update_layout(title="Variable Spending: Guardrail Adaptive Cash Flows", xaxis_title="Year", yaxis_title="Net Spendable Income ($)", hovermode="x unified")
    return fig

def plot_withdrawal_hierarchy(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['tsp_withdrawal'], axis=0), name="TSP Withdrawal", marker_color='#2ca02c'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['ira_withdrawal'], axis=0), name="Trad IRA Withdrawal", marker_color='#98df8a'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['roth_withdrawal'], axis=0), name="Roth Withdrawal", marker_color='green'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['taxable_withdrawal'], axis=0), name="Taxable Withdrawal", marker_color='orange'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['cash_withdrawal'], axis=0), name="Cash Withdrawal", marker_color='gray'))
    fig.add_trace(go.Bar(x=years_arr, y=np.median(history['extra_rmd'], axis=0), name="Reinvested RMD", marker_color='darkgray'))
    fig.update_layout(barmode='stack', title="Dynamic Account Liquidation Hierarchy", xaxis_title="Year", yaxis_title="Amount ($)")
    return fig

def plot_taxes_and_rmds(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxes_fed'], axis=0), mode='lines', fill='tozeroy', name='Federal Taxes', line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['rmds'], axis=0), mode='lines', name='RMD Volume', line=dict(color='orange', width=3)))
    fig.update_layout(title="Tax Trajectory vs Mandatory RMD Cliffs", xaxis_title="Year", yaxis_title="Amount ($)", hovermode="x unified")
    return fig

def plot_legacy_breakdown(history):
    tsp_term = np.median(history['tsp_bal'][:, -1])
    ira_term = np.median(history['ira_bal'][:, -1])
    roth_term = np.median(history['roth_bal'][:, -1])
    taxable_term = np.median(history['taxable_bal'][:, -1])
    cash_term = np.median(history['cash_bal'][:, -1])
    home_term = np.median(history['home_value'][:, -1])
    
    labels = ['TSP (Tax-Deferred)', 'Trad IRA (Tax-Deferred)', 'Roth IRA (Tax-Free)', 'Taxable Investments', 'Cash/MM', 'Real Estate']
    values = [max(0, tsp_term), max(0, ira_term), max(0, roth_term), max(0, taxable_term), max(0, cash_term), max(0, home_term)]
    colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#ff7f0e', '#d62728', 'gray']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker=dict(colors=colors))])
    fig.update_layout(title="Terminal Estate Composition (At Life Expectancy)")
    return fig

def plot_roth_strategy_comparison(roth_results):
    strategies = list(roth_results.keys())
    wealths = [roth_results[s]['wealth'] for s in strategies]
    winner_idx = np.argmax(wealths)
    colors = ['gray'] * len(strategies)
    colors[winner_idx] = 'green'
    
    fig = px.bar(x=wealths, y=strategies, orientation='h', title="Strategic Scenario Comparison (Median Terminal Wealth)")
    fig.update_traces(marker_color=colors)
    fig.update_layout(xaxis_title="Median Terminal Wealth ($)", yaxis_title="")
    return fig

def plot_roth_tax_impact(roth_results, winner, years_arr):
    fig = go.Figure()
    base_tax = roth_results['Baseline (None)']['tax_path']
    opt_tax = roth_results[winner]['tax_path']
    
    fig.add_trace(go.Scatter(x=years_arr, y=base_tax, mode='lines', fill='tozeroy', name='Baseline Lifetime Taxes', line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=years_arr, y=opt_tax, mode='lines', fill='tozeroy', name=f'Optimal ({winner}) Taxes', line=dict(color='green')))
    fig.update_layout(title="Lifetime Tax Liability Comparison", xaxis_title="Year", yaxis_title="Taxes Paid ($)")
    return fig

def plot_ss_breakeven(ss_fra, age_arr):
    early = ss_fra * 0.7 * 0.79
    fra = ss_fra * 1.0 * 0.79
    delayed = ss_fra * 1.24 * 0.79
    
    cum_early = np.cumsum([early if age >= 62 else 0 for age in age_arr])
    cum_fra = np.cumsum([fra if age >= 67 else 0 for age in age_arr])
    cum_delayed = np.cumsum([delayed if age >= 70 else 0 for age in age_arr])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=age_arr, y=cum_early, mode='lines', name='Claim at 62'))
    fig.add_trace(go.Scatter(x=age_arr, y=cum_fra, mode='lines', name='Claim at FRA (67)'))
    fig.add_trace(go.Scatter(x=age_arr, y=cum_delayed, mode='lines', name='Claim at 70'))
    fig.update_layout(title="Cumulative Guaranteed Income Breakeven Analysis", xaxis_title="Age", yaxis_title="Cumulative Income ($)")
    return fig

def plot_medicare_comparison(history, years_arr, inputs):
    fig = go.Figure()
    medicare_irmaa = np.median(history['medicare_cost'], axis=0)
    health_prem = np.median(history['health_cost'], axis=0)
    
    fig.add_trace(go.Scatter(x=years_arr, y=health_prem, mode='lines', stackgroup='one', name=f"{inputs['health_plan']} Premium + OOP"))
    fig.add_trace(go.Scatter(x=years_arr, y=medicare_irmaa, mode='lines', stackgroup='one', name="Medicare Part B + IRMAA"))
    fig.update_layout(title="Lifetime Healthcare Cost Comparison", xaxis_title="Year", yaxis_title="Annual Cost ($)")
    return fig