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
    fig.update_layout(title="Stochastic Portfolio Projections", xaxis_title="Year", yaxis_title="Portfolio Balance ($)")
    return fig

def plot_net_spendable(history, years_arr):
    med_spendable = np.median(history['net_spendable'], axis=0)
    fig = px.bar(x=years_arr, y=med_spendable, labels={'x': 'Year', 'y': 'Net Income ($)'}, title="Projected Post-Tax Real Income")
    return fig

def plot_liquidity_timeline(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['tsp_bal'], axis=0), mode='lines', stackgroup='one', name='TSP (Tax-Deferred)'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxable_bal'], axis=0), mode='lines', stackgroup='one', name='Taxable'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['roth_bal'], axis=0), mode='lines', stackgroup='one', name='Roth IRA'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['cash_bal'], axis=0), mode='lines', stackgroup='one', name='Cash Buffer'))
    fig.update_layout(title="Asset Liquidity & Composition")
    return fig

def plot_taxes_and_rmds(history, years_arr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['taxes_fed'], axis=0), mode='lines', fill='tozeroy', name='Federal Taxes'))
    fig.add_trace(go.Scatter(x=years_arr, y=np.median(history['rmds'], axis=0), mode='lines', name='RMD Volume', line=dict(color='orange', width=2)))
    fig.update_layout(title="Tax Trajectory vs RMD Cliffs")
    return fig