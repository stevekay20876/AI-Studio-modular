# engine.py
import numpy as np
import scipy.optimize as optimize
from scipy.stats import t
import datetime

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


class StochasticRetirementEngine:
    def __init__(self, inputs):
        self.inputs = inputs
        self.iterations = 10000
        self.years = max(1, inputs['life_expectancy'] - inputs['current_age'])
        self.n_assets = 5 
        self.setup_covariance_matrix()
        
    def setup_covariance_matrix(self):
        corr = np.array([
            [1.00, -0.15, -0.15, -0.15, -0.15],
            [-0.15, 1.00,  0.85,  0.85,  0.85],
            [-0.15, 0.85,  1.00,  0.85,  0.85],
            [-0.15, 0.85,  0.85,  1.00,  0.85],
            [-0.15, 0.85,  0.85,  0.85,  1.00]
        ])
        vols = np.array([0.012, self.inputs['tsp_vol'], self.inputs['roth_vol'], self.inputs['taxable_vol'], self.inputs['hsa_vol']])
        cov = np.outer(vols, vols) * corr
        self.L = np.linalg.cholesky(cov)

    def generate_stochastic_paths(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        shocks = t.rvs(df=5, size=(self.iterations, self.years, self.n_assets))
        correlated_shocks = np.einsum('ij,kyj->kyi', self.L, shocks)
        
        drifts = np.array([
            0.03,
            self.inputs['tsp_ret'] - (self.inputs['tsp_vol']**2)/2,
            self.inputs['roth_ret'] - (self.inputs['roth_vol']**2)/2,
            self.inputs['taxable_ret'] - (self.inputs['taxable_vol']**2)/2,
            self.inputs['hsa_ret'] - (self.inputs['hsa_vol']**2)/2
        ])
        
        returns = np.exp(drifts + correlated_shocks) - 1
        inf_paths = np.zeros((self.iterations, self.years))
        inf_base = 0.025   
        kappa = 0.5        
        jump_prob = 0.05   
        current_inf = np.full(self.iterations, inf_base)
        
        for yr in range(self.years):
            dW = correlated_shocks[:, yr, 0]
            jumps = np.where(np.random.rand(self.iterations) < jump_prob, np.random.normal(0.04, 0.01, self.iterations), 0)
            current_inf = current_inf + kappa * (inf_base - current_inf) + dW + jumps
            inf_paths[:, yr] = np.clip(current_inf, -0.01, 0.12) 
            stagflation_shock = np.where(jumps > 0, -0.10, 0)
            returns[:, yr, 1:] += stagflation_shock[:, None]
            
        return returns, inf_paths

    def run_mc(self, iwr, seed=None, roth_strategy=0):
        returns, inf_paths = self.generate_stochastic_paths(seed=seed)
        cash_ret = self.inputs['cash_ret']
        
        tsp = np.full(self.iterations, self.inputs['tsp_bal'])
        roth = np.full(self.iterations, self.inputs['roth_bal'])
        taxable = np.full(self.iterations, self.inputs['taxable_bal'])
        hsa = np.full(self.iterations, self.inputs['hsa_bal'])
        cash = np.full(self.iterations, self.inputs['cash_bal'])
        
        base_pension = np.full(self.iterations, self.inputs['pension_est'])
        base_ss = np.full(self.iterations, self.inputs['ss_fra'])
        
        # Base Withdrawal is calculated off the initial balance, but won't trigger until ret_age
        base_withdrawal = (tsp[0] + roth[0] + taxable[0] + cash[0]) * iwr
        scheduled_withdrawal = np.full(self.iterations, base_withdrawal)
        
        history = {
            'total_bal': np.zeros((self.iterations, self.years)),
            'tsp_bal': np.zeros((self.iterations, self.years)),
            'roth_bal': np.zeros((self.iterations, self.years)),
            'taxable_bal': np.zeros((self.iterations, self.years)),
            'cash_bal': np.zeros((self.iterations, self.years)),
            'hsa_bal': np.zeros((self.iterations, self.years)),
            'tsp_withdrawal': np.zeros((self.iterations, self.years)),
            'roth_withdrawal': np.zeros((self.iterations, self.years)),    
            'taxable_withdrawal': np.zeros((self.iterations, self.years)), 
            'cash_withdrawal': np.zeros((self.iterations, self.years)),    
            'rmds': np.zeros((self.iterations, self.years)),
            'extra_rmd': np.zeros((self.iterations, self.years)),
            'taxes_fed': np.zeros((self.iterations, self.years)),
            'taxes_state': np.zeros((self.iterations, self.years)),
            'taxable_income': np.zeros((self.iterations, self.years)), 
            'magi': np.zeros((self.iterations, self.years)),           
            'medicare_cost': np.zeros((self.iterations, self.years)),
            'health_cost': np.zeros((self.iterations, self.years)),
            'mortgage_cost': np.zeros((self.iterations, self.years)),
            'net_spendable': np.zeros((self.iterations, self.years)),
            'port_return': np.zeros((self.iterations, self.years)),
            'real_return': np.zeros((self.iterations, self.years)),
            'inflation': inf_paths,
            'constraint_active': np.zeros((self.iterations, self.years)),
            'ss_income': np.zeros((self.iterations, self.years)),
            'pension_income': np.zeros((self.iterations, self.years)),
            'roth_conversion': np.zeros((self.iterations, self.years)),
        }

        age = self.inputs['current_age']
        current_year = datetime.datetime.now().year
        ret_age = self.inputs['ret_age']
        
        base_health_premium = self.inputs.get('health_cost', 0.0)
        base_oop_cost = self.inputs.get('oop_cost', 0.0)
        mortgage_pmt = self.inputs.get('mortgage_pmt', 0.0)
        mortgage_yrs = self.inputs.get('mortgage_yrs', 0)
        health_plan = self.inputs.get('health_plan', "None/Self-Insure")
        
        deduction = STD_DED_MFJ if self.inputs['filing_status'] == 'MFJ' else STD_DED_SINGLE
        brackets = TAX_BRACKETS_MFJ if self.inputs['filing_status'] == 'MFJ' else TAX_BRACKETS_SINGLE
        irmaa_brackets = IRMAA_BRACKETS_MFJ if self.inputs['filing_status'] == 'MFJ' else IRMAA_BRACKETS_SINGLE

        state_str = self.inputs.get('state', '').strip().upper()
        county_str = self.inputs.get('county', '').strip().upper()
        
        tax_free_states = ["TX", "FL", "NV", "WA", "SD", "WY", "AK", "TN", "NH"]
        if state_str in tax_free_states:
            state_tax_rate = 0.0
        else:
            state_tax_rate = 0.045 if state_str != "" else 0.0
            
        if county_str != "" and state_str in ["MD", "IN", "PA", "OH", "NY", "MARYLAND", "INDIANA", "PENNSYLVANIA", "OHIO", "NEW YORK"]:
            local_tax_rate = 0.025 
        elif county_str != "":
            local_tax_rate = 0.010 
        else:
            local_tax_rate = 0.0
            
        combined_state_local_rate = state_tax_rate + local_tax_rate

        for yr in range(self.years):
            age += 1
            current_year += 1
            
            # --- 1. APPLY RETURNS ---
            prev_total_port = tsp + roth + taxable + cash
            tsp *= (1 + returns[:, yr, 1])
            roth *= (1 + returns[:, yr, 2])
            taxable *= (1 + returns[:, yr, 3])
            hsa *= (1 + returns[:, yr, 4])
            cash *= (1 + cash_ret)
            
            current_total_port = tsp + roth + taxable + cash
            history['port_return'][:, yr] = (current_total_port - prev_total_port) / np.maximum(prev_total_port, 1)
            history['real_return'][:, yr] = history['port_return'][:, yr] - inf_paths[:, yr]
            
            # --- 2. COLA ADJUSTMENTS ---
            if yr > 0:
                cpi = np.maximum(0, inf_paths[:, yr]) 
                base_ss *= (1 + cpi)
                fers_cola = np.where(cpi <= 0.02, cpi, np.where(cpi <= 0.03, 0.02, cpi - 0.01))
                fers_cola = np.where(age >= 62, fers_cola, 0.0) 
                base_pension *= (1 + fers_cola)

            pension = np.where(age >= ret_age, base_pension, 0)
            ss_haircut = 0.79 if current_year >= 2035 else 1.0
            ss = np.where(age >= 67, base_ss * ss_haircut, 0)
            history['pension_income'][:, yr] = pension
            history['ss_income'][:, yr] = ss
            
            # --- 3. ACCUMULATION VS DECUMULATION LOGIC ---
            w_needed = np.zeros(self.iterations)
            constraint_flag = np.zeros(self.iterations)
            
            # Only activate withdrawals and guardrails IF retired
            if age >= ret_age:
                if yr > 0:
                    port_ret = history['port_return'][:, yr-1]
                    inf_adj = np.where(port_ret < 0, 0, inf_paths[:, yr])
                    scheduled_withdrawal *= (1 + inf_adj)
                    
                    cwr = scheduled_withdrawal / current_total_port
                    ceiling_hit = cwr > iwr * 1.2
                    scheduled_withdrawal = np.where(ceiling_hit, scheduled_withdrawal * 0.9, scheduled_withdrawal)
                    constraint_flag = np.where(ceiling_hit, 1, constraint_flag)
                    
                    scheduled_withdrawal = np.where(cwr < iwr * 0.8, scheduled_withdrawal * 1.1, scheduled_withdrawal)
                    
                    sorr_trigger = current_total_port <= prev_total_port * 0.9
                    scheduled_withdrawal = np.where(sorr_trigger, scheduled_withdrawal * 0.9, scheduled_withdrawal)
                    constraint_flag = np.where(sorr_trigger, 1, constraint_flag)
                    
                w_needed = scheduled_withdrawal.copy()
            else:
                # If still working, scheduled withdrawal inflates silently but is NOT withdrawn
                if yr > 0:
                    cpi_adj = np.maximum(0, inf_paths[:, yr])
                    scheduled_withdrawal *= (1 + cpi_adj)

            history['constraint_active'][:, yr] = constraint_flag
            
            # --- 4. REQUIRED MINIMUM DISTRIBUTIONS (RMDs) ---
            rmd_rate = 0.036 if age >= 75 else 0.0
            rmds = tsp * rmd_rate
            history['rmds'][:, yr] = rmds
            tsp -= rmds
            
            w_remaining = np.maximum(0, w_needed - rmds)
            excess_rmd = np.maximum(0, rmds - w_needed)
            history['extra_rmd'][:, yr] = excess_rmd
            
            # --- 5. WITHDRAWAL HIERARCHY (ONLY EXECUTES IF RETIRED OR IF RMD IS REQUIRED) ---
            w_tsp = np.zeros(self.iterations)
            w_cash = np.zeros(self.iterations)
            w_taxable = np.zeros(self.iterations)
            w_roth = np.zeros(self.iterations)
            
            if age >= ret_age:
                tsp_prior_ret = returns[:, yr-1, 1] if yr > 0 else np.zeros(self.iterations)
                downturn = tsp_prior_ret <= -0.10
                
                w_tsp_norm = np.where(~downturn, np.minimum(tsp, w_remaining), 0)
                tsp -= w_tsp_norm
                w_remaining -= w_tsp_norm
                
                w_cash_down = np.where(downturn, np.minimum(cash, w_remaining), 0)
                cash -= w_cash_down
                w_remaining -= w_cash_down
                
                w_tax_down = np.where(downturn, np.minimum(taxable, w_remaining), 0)
                taxable -= w_tax_down
                w_remaining -= w_tax_down
                
                w_roth_down = np.where(downturn, np.minimum(roth, w_remaining), 0)
                roth -= w_roth_down
                w_remaining -= w_roth_down
                
                # Universal Depletion Fallback
                w_tax_fb = np.minimum(taxable, w_remaining)
                taxable -= w_tax_fb
                w_remaining -= w_tax_fb
                
                w_cash_fb = np.minimum(cash, w_remaining)
                cash -= w_cash_fb
                w_remaining -= w_cash_fb
                
                w_roth_fb = np.minimum(roth, w_remaining)
                roth -= w_roth_fb
                w_remaining -= w_roth_fb
                
                w_tsp_fb = np.minimum(tsp, w_remaining)
                tsp -= w_tsp_fb
                w_remaining -= w_tsp_fb
                
                w_tsp += (w_tsp_norm + w_tsp_fb)
                w_cash += (w_cash_down + w_cash_fb)
                w_taxable += (w_tax_down + w_tax_fb)
                w_roth += (w_roth_down + w_roth_fb)
            
            actual_portfolio_withdrawal = w_tsp + w_cash + w_taxable + w_roth + rmds - excess_rmd
            
            history['tsp_withdrawal'][:, yr] = w_tsp
            history['roth_withdrawal'][:, yr] = w_roth
            history['taxable_withdrawal'][:, yr] = w_taxable
            history['cash_withdrawal'][:, yr] = w_cash
            
            # Reinvest excess RMD back into taxable regardless of retirement status
            taxable += excess_rmd
            
            # --- 6. BASE TAXES ---
            gross_income = rmds + w_tsp + pension + (ss * 0.85)
            magi = gross_income.copy() 
            taxable_income = np.maximum(0, gross_income - deduction) 
            
            base_tax_fed = np.zeros(self.iterations)
            for i in range(len(brackets)):
                prev_limit = brackets[i-1][0] if i > 0 else 0
                limit, rate = brackets[i]
                base_tax_fed += np.clip(taxable_income - prev_limit, 0, limit - prev_limit) * rate
                
            base_tax_state_local = taxable_income * combined_state_local_rate
            
            # --- 7. ROTH CONVERSIONS ---
            total_tax_fed = base_tax_fed.copy()
            total_tax_state = base_tax_state_local.copy()
            conv_amt = np.zeros(self.iterations)
            
            final_taxable_income = taxable_income.copy()
            final_magi = magi.copy()
            
            if roth_strategy > 0 and age < 75:
                space = np.zeros(self.iterations)
                
                if roth_strategy in [1, 4]: 
                    for limit, rate in brackets:
                        mask = (taxable_income < limit) & (space == 0)
                        space[mask] = limit - taxable_income[mask] - 1
                    space = np.where(space > 1e6, 0, space)
                    
                    if roth_strategy == 1:
                        for irmaa_limit, surcharge in irmaa_brackets:
                            crosses_cliff = (magi < irmaa_limit) & ((magi + space) >= irmaa_limit)
                            space = np.where(crosses_cliff, irmaa_limit - magi - 1, space)
                        
                elif roth_strategy == 2: 
                    irmaa_tier_1 = irmaa_brackets[0][0]
                    space = np.maximum(0, irmaa_tier_1 - magi - 1)
                    
                elif roth_strategy == 3:
                    irmaa_tier_2 = irmaa_brackets[1][0]
                    space = np.maximum(0, irmaa_tier_2 - magi - 1)
                
                if roth_strategy in [1, 2, 3]:
                    limit_24_pct = brackets[3][0] 
                    max_allowable_space = np.maximum(0, limit_24_pct - taxable_income - 1)
                    space = np.minimum(space, max_allowable_space)
                elif roth_strategy == 4:
                    limit_32_pct = brackets[4][0] 
                    max_allowable_space = np.maximum(0, limit_32_pct - taxable_income - 1)
                    space = np.minimum(space, max_allowable_space)
                
                conv_amt = np.minimum(space, tsp)
                final_taxable_income = taxable_income + conv_amt
                final_magi = magi + conv_amt
                
                new_tax_fed = np.zeros(self.iterations)
                for i in range(len(brackets)):
                    prev_limit = brackets[i-1][0] if i > 0 else 0
                    limit, rate = brackets[i]
                    new_tax_fed += np.clip(final_taxable_income - prev_limit, 0, limit - prev_limit) * rate
                
                extra_tax_fed = new_tax_fed - base_tax_fed
                extra_tax_state = conv_amt * combined_state_local_rate
                extra_tax_total = extra_tax_fed + extra_tax_state
                
                w_tax_cash = np.minimum(cash, extra_tax_total)
                cash -= w_tax_cash
                rem_tax = extra_tax_total - w_tax_cash
                w_tax_taxable = np.minimum(taxable, rem_tax)
                taxable -= w_tax_taxable
                rem_tax -= w_tax_taxable
                
                net_to_roth = conv_amt - rem_tax 
                tsp -= conv_amt
                roth += net_to_roth
                
                total_tax_fed = new_tax_fed
                total_tax_state = base_tax_state_local + extra_tax_state
            
            history['roth_conversion'][:, yr] = conv_amt
            history['taxes_fed'][:, yr] = total_tax_fed
            history['taxes_state'][:, yr] = total_tax_state
            history['taxable_income'][:, yr] = final_taxable_income
            history['magi'][:, yr] = final_magi
            
            # --- 8. HEALTHCARE & HSA ---
            base_health_premium *= (1 + inf_paths[:, yr])
            inflated_oop = base_oop_cost * (1 + inf_paths[:, yr])
            
            medicare_cost = np.zeros(self.iterations)
            if age >= 65 and "FEHB" not in health_plan and "TRICARE" not in health_plan:
                medicare_cost += MEDICARE_PART_B_BASE
                for i in range(len(irmaa_brackets)):
                    limit, surcharge = irmaa_brackets[i]
                    medicare_cost = np.where(final_magi > (irmaa_brackets[i-1][0] if i>0 else 0), MEDICARE_PART_B_BASE + surcharge, medicare_cost)
            history['medicare_cost'][:, yr] = medicare_cost

            w_hsa = np.minimum(hsa, inflated_oop)
            hsa -= w_hsa
            oop_remainder = inflated_oop - w_hsa
            
            history['health_cost'][:, yr] = base_health_premium + oop_remainder
            
            # --- 9. MORTGAGE ---
            current_mortgage = np.full(self.iterations, mortgage_pmt if yr < mortgage_yrs else 0.0)
            history['mortgage_cost'][:, yr] = current_mortgage
            
            # --- 10. WRAP UP BALANCES & SPENDABLE ---
            history['total_bal'][:, yr] = tsp + roth + taxable + cash
            history['tsp_bal'][:, yr] = tsp
            history['roth_bal'][:, yr] = roth
            history['taxable_bal'][:, yr] = taxable
            history['cash_bal'][:, yr] = cash
            history['hsa_bal'][:, yr] = hsa
            
            # Only record net spendable (lifestyle withdrawal) if actually retired
            if age >= ret_age:
                history['net_spendable'][:, yr] = (actual_portfolio_withdrawal + pension + ss 
                                                   - base_tax_fed - base_tax_state_local 
                                                   - medicare_cost - history['health_cost'][:, yr] 
                                                   - current_mortgage)
            else:
                history['net_spendable'][:, yr] = 0.0
            
        return history

    def objective_function(self, iwr_test):
        history = self.run_mc(iwr_test, seed=42, roth_strategy=0)
        median_path = np.median(history['total_bal'], axis=0)
        target_floor = self.inputs.get('target_floor', 0.0)
        
        # Premature Depletion Penalty: Forces optimizer to reject unsustainable pa