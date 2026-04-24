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

# 2024/2025 Estimated IRMAA Thresholds
MEDICARE_PART_B_BASE = 2096.40
IRMAA_BRACKETS_SINGLE = [(103000, 0), (129000, 838.8), (161000, 2101.2), (193000, 3362.4), (499999, 4624.8), (np.inf, 5043.6)]
IRMAA_BRACKETS_MFJ = [(206000, 0), (258000, 838.8), (322000, 2101.2), (386000, 3362.4), (749999, 4624.8), (np.inf, 5043.6)]

# --- STATUTORY HEALTH INSURANCE MOOP LIMITS (Single, MFJ/Family) ---
# Used to cap catastrophic Out-Of-Pocket risk in the simulation
MOOP_LIMITS = {
    "FEHB FEPBlue Basic": (6500, 13000),
    "FEPBlue Standard": (6000, 12000),
    "FEPBlue Focus": (8500, 17000),
    "GEHA High": (6000, 12000),
    "GEHA Standard": (7000, 14000),
    "Aetna Open Access": (5000, 10000),
    "Aetna Direct": (5000, 10000),
    "Aetna Advantage": (5000, 10000),
    "Cigna": (6500, 13000),
    "TRICARE for Life": (3000, 3000),
    "None/Self-Insure": (999999, 999999) # Uncapped Risk
}