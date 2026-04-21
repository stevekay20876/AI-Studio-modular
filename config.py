# config.py
import numpy as np

# Tax Standard Deductions
STD_DED_SINGLE = 14600
STD_DED_MFJ = 29200

# Federal Tax Brackets (2024 limits)
TAX_BRACKETS_SINGLE = [
    (11600, 0.10), (47150, 0.12), (100525, 0.22), 
    (191950, 0.24), (243725, 0.32), (609350, 0.35), (np.inf, 0.37)
]
TAX_BRACKETS_MFJ = [
    (23200, 0.10), (94300, 0.12), (201050, 0.22), 
    (383900, 0.24), (487450, 0.32), (731200, 0.35), (np.inf, 0.37)
]

# Medicare Part B Base Premium (Annualized)
MEDICARE_PART_B_BASE = 2096.40 

# IRMAA MAGI Brackets & Surcharges (Annualized)
IRMAA_BRACKETS_SINGLE = [
    (103000, 0), (129000, 838.8), (161000, 2101.2), 
    (193000, 3362.4), (499999, 4624.8), (np.inf, 5043.6)
]
IRMAA_BRACKETS_MFJ = [
    (206000, 0), (258000, 838.8), (322000, 2101.2), 
    (386000, 3362.4), (749999, 4624.8), (np.inf, 5043.6)
]