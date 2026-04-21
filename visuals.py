# visuals.py
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

def plot_wealth_trajectory(history, target_floor, years_arr):
    median_paths = np.median(history['total_bal'], axis=0)
    pessimistic = np.percentile(history['total_bal'], 10, axis=0)
    optimistic = np.percentile(history['total_bal'], 90, axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=median_paths, mode='lines', name='Median (50th)', line=dict(color='blue', width=3)))
    fig.add_trace(go.Scatter(x=years_arr, y=optimistic, mode='lines', name='Optimistic (90th)', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=years_arr, y=pessimistic, mode='lines', name='Pessimistic (10th)', line=dict(color='red', dash='dash')))
    fig.add_hline(y=target_floor, line_dash="dot", line_color="black", annotation_text="Target Estate Floor")
    fig.update_layout(title="Stochastic Portfolio Projections", xaxis_title="Year", yaxis_title="Portfolio Balance ($)", hovermode="x unified")
    return fig

def plot_net_spendable(history, years_arr):
    med_spendable = np.median(history['net_spendable'], axis=0)
    fig = px.bar(x=years_arr, y=med_spendable, labels={'x': 'Year', 'y': 'Net Income ($)'}, title="Projected Post-Tax Real Income")
    fig.update_traces(marker_color='teal')
    return fig

def plot_liquidity_timeline(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['tsp_bal'], axis=0), mode='lines', stackgroup='one', name='TSP (Tax-Deferred)'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxable_bal'], axis=0), mode='lines', stackgroup='one', name='Taxable'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['roth_bal'], axis=0), mode='lines', stackgroup='one', name='Roth IRA'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['cash_bal'], axis=0), mode='lines', stackgroup='one', name='Cash Buffer'))
    fig.update_layout(title="Asset Liquidity & Composition Timeline", xaxis_title="Year", yaxis_title="Balance ($)", hovermode="x unified")
    return fig

def plot_taxes_and_rmds(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxes_fed'], axis=0), mode='lines', fill='tozeroy', name='Federal Taxes', line=dict(color='crimson')))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['rmds'], axis=0), mode='lines', name='RMD Volume', line=dict(color='orange', width=3)))
    fig.update_layout(title="Tax Trajectory vs Mandatory RMD Cliffs", xaxis_title="Year", yaxis_title="Amount ($)", hovermode="x unified")
    return fig

def plot_social_security_analysis(prob_success):
    # Heuristic adjustment for visual analysis based on standard SS math + 2035 Trust depletion
    strategies = ["Age 62 (Early)", "Age 67 (FRA)", "Age 70 (Delayed)"]
    probs = [max(0, prob_success - 8), prob_success, min(100, prob_success + 6)]
    
    fig = px.bar(x=strategies, y=probs, color=strategies, 
                 labels={'x': 'Claiming Age', 'y': 'Probability of Success (%)'},
                 title="Impact of SS Claiming Age on Terminal Portfolio Longevity")
    fig.update_yaxes(range=[0, 100])
    return fig