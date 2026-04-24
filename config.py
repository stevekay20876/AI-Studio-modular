# config.py
import numpy as np

# --- 2025 IRS TAX BRACKETS & LIMITS ---
STD_DED_SINGLE = 15000
STD_DED_MFJ = 30000

TAX_BRACKETS_SINGLE = [
    (11925, 0.10), (48475, 0.12), (103350, 0.22), 
    (197300, 0.24), (250525, 0.32), (626350, 0.35), (np.inf, 0.37)
]
TAX_BRACKETS_MFJ = [
    (23850, 0.10), (96950, 0.12), (206700, 0.22), 
    (394600, 0.24), (501050, 0.32), (751600, 0.35), (np.inf, 0.37)
]

# --- LONG-TERM CAPITAL GAINS (LTCG) BRACKETS ---
LTCG_BRACKETS_SINGLE = [(47025, 0.0), (518900, 0.15), (np.inf, 0.20)]
LTCG_BRACKETS_MFJ = [(94050, 0.0), (583750, 0.15), (np.inf, 0.20)]

# NIIT (Net Investment Income Tax) Thresholds
NIIT_THRESHOLD_SINGLE = 200000
NIIT_THRESHOLD_MFJ = 250000

# 2024/2025 Estimated IRMAA Thresholds
MEDICARE_PART_B_BASE = 2096.40
IRMAA_BRACKETS_SINGLE = [(103000, 0), (129000, 838.8), (161000, 2101.2), (193000, 3362.4), (499999, 4624.8), (np.inf, 5043.6)]
IRMAA_BRACKETS_MFJ = [(206000, 0), (258000, 838.8), (322000, 2101.2), (386000, 3362.4), (749999, 4624.8), (np.inf, 5043.6)]

# --- STATE RETIREMENT EXEMPTIONS ---
# States that fully exempt Retirement Income (TSP, Pensions, SS) or have NO income tax
RETIREMENT_TAX_FREE_STATES = ["FL", "TX", "NV", "WA", "SD", "WY", "AK", "TN", "NH", "IL", "PA", "MS"]

# --- STATUTORY HEALTH INSURANCE MOOP LIMITS (Single, MFJ) ---
MOOP_LIMITS = {
    "FEHB FEPBlue Basic": (6500, 13000), "FEPBlue Standard": (6000, 12000), "FEPBlue Focus": (8500, 17000),
    "GEHA High": (6000, 12000), "GEHA Standard": (7000, 14000), "Aetna Open Access": (5000, 10000),
    "Aetna Direct": (5000, 10000), "Aetna Advantage": (5000, 10000), "Cigna": (6500, 13000),
    "TRICARE for Life": (3000, 3000), "None/Self-Insure": (999999, 999999)
}

# --- IRS UNIFORM LIFETIME TABLE (RMD Divisors) ---
IRS_RMD_DIVISORS = {
    75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1, 80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8,
    85: 16.0, 86: 15.2, 87: 14.4, 88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5,
    95: 8.9, 96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2, 104: 4.9, 
    105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4, 112: 3.3, 113: 3.1, 114: 3.0,
    115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0
}

# --- PRE-SET ACTUARIAL PORTFOLIOS ---
PORTFOLIOS = {
    "Conservative (20% Stock / 80% Bond)": {"ret": 0.045, "vol": 0.06},
    "Moderate (60% Stock / 40% Bond)": {"ret": 0.070, "vol": 0.10},
    "Aggressive (100% Stock)": {"ret": 0.095, "vol": 0.15}
}