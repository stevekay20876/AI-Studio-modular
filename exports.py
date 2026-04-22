# exports.py
import numpy as np
import pandas as pd

def build_csv_dataframe(history, years_arr, age_arr, percentile=50):
    data = {
        "Calendar Year": years_arr,
        "Age": age_arr,
        "Rate of Return": np.percentile(history['port_return'], percentile, axis=0),
        "Inflation Rate": np.percentile(history['inflation'], percentile, axis=0),
        "Real Rate of Return": np.percentile(history['real_return'], percentile, axis=0),
        "Taxable ETF Balance": np.percentile(history['taxable_bal'], percentile, axis=0),
        "Roth IRA Balance": np.percentile(history['roth_bal'], percentile, axis=0),
        "HSA Balance": np.percentile(history['hsa_bal'], percentile, axis=0),
        "Money Market Balance": np.percentile(history['cash_bal'], percentile, axis=0),
        "Annual 401(k)/TSP Withdrawal": np.percentile(history['tsp_withdrawal'], percentile, axis=0),
        "Pension": np.percentile(history['pension_income'], percentile, axis=0),
        "Social Security": np.percentile(history['ss_income'], percentile, axis=0),
        "RMD Amount": np.percentile(history['rmds'], percentile, axis=0),
        "Extra RMD Amount": np.percentile(history['extra_rmd'], percentile, axis=0),
        "Roth Conversion Amount": np.percentile(history['roth_conversion'], percentile, axis=0),
        "Federal Taxes": np.percentile(history['taxes_fed'], percentile, axis=0),
        "State Taxes": np.percentile(history['taxes_state'], percentile, axis=0),
        "Medicare Cost": np.percentile(history['medicare_cost'], percentile, axis=0),
        "Health Insurance Cost": np.percentile(history['health_cost'], percentile, axis=0),
        "Total Income": np.percentile(history['net_spendable'] + history['taxes_fed'] + history['taxes_state'] + history['medicare_cost'] + history['health_cost'], percentile, axis=0),
        "Total Expenses": np.percentile(
            history['taxes_fed'] + history['taxes_state'] + history['medicare_cost'] + history['health_cost'] + history['mortgage_cost'], percentile, axis=0),
        "Net Spendable Annual": np.percentile(history['net_spendable'], percentile, axis=0),
        "Net Monthly": np.percentile(history['net_spendable'], percentile, axis=0) / 12,
        "Ending 401(k)/TSP Balance": np.percentile(history['tsp_bal'], percentile, axis=0),
        "Ending Total Balance (excluding HSA)": np.percentile(history['total_bal'], percentile, axis=0),
        "Withdrawal Constraint Active": ["Yes" if flag > 0 else "No" for flag in np.percentile(history['constraint_active'], percentile, axis=0)]
    }
    return pd.DataFrame(data)