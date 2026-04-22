# engine.py
import numpy as np
import scipy.optimize as optimize
from scipy.stats import t
import datetime
from config import *

class StochasticRetirementEngine:
    def __init__(self, inputs):
        self.inputs = inputs
        self.iterations = 10000
        self.years = max(1, inputs['life_expectancy'] - inputs['current_age'])
        self.n_assets = 5 # Inflation, TSP, Roth, Taxable, HSA
        self.setup_covariance_matrix()
        
    def setup_covariance_matrix(self):
        corr = np.array([
            [1.00, -0.15, -0.15, -0.15, -0.15],
            [-0.15, 1.00,  0.85,  0.85,  0.85],
            [-0.15, 0.85,  1.00,  0.85,  0.85],
            [-0.15, 0.85,  0.85,  1.00,  0.85],
            [-0.15, 0.85,  0.85,  0.85,  1.00]
        ])
        vols = np.array([
            0.012, # <-- CHANGED from 0.02 to match historical inflation volatility
            self.inputs['tsp_vol'], 
            self.inputs['roth_vol'], 
            self.inputs['taxable_vol'], 
            self.inputs['hsa_vol']
        ])
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
# Ornstein-Uhlenbeck Inflation with Historically Calibrated Jump-Diffusion
        inf_paths = np.zeros((self.iterations, self.years))
        inf_base = 0.025   # Base inflation target of 2.5%
        kappa = 0.5        # Faster mean-reversion (simulating Fed intervention)
        jump_prob = 0.05   # 5% chance of a severe inflation shock (e.g., 1970s, 2022)
        current_inf = np.full(self.iterations, inf_base)
        
        for yr in range(self.years):
            dW = correlated_shocks[:, yr, 0]
            # 5% chance of an inflation spike averaging 4%
            jumps = np.where(np.random.rand(self.iterations) < jump_prob, np.random.normal(0.04, 0.01, self.iterations), 0)
            
            # Calculate new inflation
            current_inf = current_inf + kappa * (inf_base - current_inf) + dW + jumps
            inf_paths[:, yr] = np.clip(current_inf, -0.01, 0.12) # Bounded between -1% and 12%
            
            # Stagflation impact: if jump occurs, apply shock to equity returns
            stagflation_shock = np.where(jumps > 0, -0.10, 0)
            returns[:, yr, 1:] += stagflation_shock[:, None]
            
        return returns, inf_paths

    def run_mc(self, iwr, seed=None):
        returns, inf_paths = self.generate_stochastic_paths(seed=seed)
        cash_ret = self.inputs['cash_ret']
        
        # Balance Init
        tsp = np.full(self.iterations, self.inputs['tsp_bal'])
        roth = np.full(self.iterations, self.inputs['roth_bal'])
        taxable = np.full(self.iterations, self.inputs['taxable_bal'])
        hsa = np.full(self.iterations, self.inputs['hsa_bal'])
        cash = np.full(self.iterations, self.inputs['cash_bal'])
        
        # Income Bases Init (To track compounding COLAs)
        base_pension = np.full(self.iterations, self.inputs['pension_est'])
        base_ss = np.full(self.iterations, self.inputs['ss_fra'])
        
        base_withdrawal = (tsp[0] + roth[0] + taxable[0] + cash[0]) * iwr
        scheduled_withdrawal = np.full(self.iterations, base_withdrawal)
        
        history = {
            'total_bal': np.zeros((self.iterations, self.years)),
            'tsp_bal': np.zeros((self.iterations, self.years)),
            'roth_bal': np.zeros((self.iterations, self.years)),
            'taxable_bal': np.zeros((self.iterations, self.years)),
            'hsa_bal': np.zeros((self.iterations, self.years)),
            'cash_bal': np.zeros((self.iterations, self.years)),
            'tsp_withdrawal': np.zeros((self.iterations, self.years)),
            'rmds': np.zeros((self.iterations, self.years)),
            'extra_rmd': np.zeros((self.iterations, self.years)),
            'taxes_fed': np.zeros((self.iterations, self.years)),
            'taxes_state': np.zeros((self.iterations, self.years)),
            'medicare_cost': np.zeros((self.iterations, self.years)),
            'health_cost': np.zeros((self.iterations, self.years)),
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
        health_premium = self.inputs['health_cost']
        state_tax_rate = 0.045 if self.inputs['state'].strip() != "" else 0.0

        for yr in range(self.years):
            age += 1
            current_year += 1
            
            # Apply Returns
            prev_total_port = tsp + roth + taxable + cash
            tsp *= (1 + returns[:, yr, 1])
            roth *= (1 + returns[:, yr, 2])
            taxable *= (1 + returns[:, yr, 3])
            hsa *= (1 + returns[:, yr, 4])
            cash *= (1 + cash_ret)
            
            current_total_port = tsp + roth + taxable + cash
            history['port_return'][:, yr] = (current_total_port - prev_total_port) / np.maximum(prev_total_port, 1)
            history['real_return'][:, yr] = history['port_return'][:, yr] - inf_paths[:, yr]
            
            # --- COLA MATH APPLIED HERE ---
            if yr > 0:
                # CPI cannot be negative for federal COLAs
                cpi = np.maximum(0, inf_paths[:, yr]) 
                
                # 1. Social Security gets Full CPI
                base_ss *= (1 + cpi)
                
                # 2. FERS Diet COLA Logic
                # If CPI <= 2%, get full CPI. If 2-3%, get 2%. If >3%, get CPI - 1%.
                fers_cola = np.where(cpi <= 0.02, cpi, 
                                     np.where(cpi <= 0.03, 0.02, cpi - 0.01))
                
                # FERS COLA statutorily begins at Age 62
                fers_cola = np.where(age >= 62, fers_cola, 0.0) 
                
                base_pension *= (1 + fers_cola)

            # Generate Year's Actual Income Streams
            pension = np.where(age >= self.inputs['ret_age'], base_pension, 0)
            ss_haircut = 0.79 if current_year >= 2035 else 1.0
            ss = np.where(age >= 67, base_ss * ss_haircut, 0)
            
            history['pension_income'][:, yr] = pension
            history['ss_income'][:, yr] = ss
            
            # Guardrails (Guyton-Klinger & SORR)
            constraint_flag = np.zeros(self.iterations)
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

            history['constraint_active'][:, yr] = constraint_flag
            w_needed = scheduled_withdrawal.copy()
            
            # RMDs
            rmd_rate = 0.036 if age >= 75 else 0.0
            rmds = tsp * rmd_rate
            history['rmds'][:, yr] = rmds
            tsp -= rmds
            
            w_remaining = np.maximum(0, w_needed - rmds)
            excess_rmd = np.maximum(0, rmds - w_needed)
            history['extra_rmd'][:, yr] = excess_rmd
            
            # Withdrawals & Hierarchy
            tsp_prior_ret = returns[:, yr-1, 1] if yr > 0 else np.zeros(self.iterations)
            downturn = tsp_prior_ret <= -0.10
            
            w_tsp = np.where(~downturn, np.minimum(tsp, w_remaining), 0)
            tsp -= w_tsp
            w_remaining -= w_tsp
            
            w_cash = np.where(downturn, np.minimum(cash, w_remaining), 0)
            cash -= w_cash
            w_remaining -= w_cash
            
            w_taxable = np.where(downturn, np.minimum(taxable, w_remaining), 0)
            taxable -= w_taxable
            w_remaining -= w_taxable
            
            w_roth = np.where(downturn, np.minimum(roth, w_remaining), 0)
            roth -= w_roth
            w_remaining -= w_roth
            
            w_tsp_fallback = np.where(downturn & (w_remaining > 0), np.minimum(tsp, w_remaining), 0)
            tsp -= w_tsp_fallback
            history['tsp_withdrawal'][:, yr] = w_tsp + w_tsp_fallback
            
            taxable += excess_rmd
            
            # Taxes
            taxable_income = rmds + w_tsp + w_tsp_fallback + pension + (ss * 0.85)
            deduction = STD_DED_MFJ if self.inputs['filing_status'] == 'MFJ' else STD_DED_SINGLE
            brackets = TAX_BRACKETS_MFJ if self.inputs['filing_status'] == 'MFJ' else TAX_BRACKETS_SINGLE
            irmaa_brackets = IRMAA_BRACKETS_MFJ if self.inputs['filing_status'] == 'MFJ' else IRMAA_BRACKETS_SINGLE
            
            agi = np.maximum(0, taxable_income - deduction)
            
            tax_fed = np.zeros(self.iterations)
            for i in range(len(brackets)):
                prev_limit = brackets[i-1][0] if i > 0 else 0
                limit, rate = brackets[i]
                tax_fed += np.clip(agi - prev_limit, 0, limit - prev_limit) * rate
            history['taxes_fed'][:, yr] = tax_fed
            
            tax_state = agi * state_tax_rate
            history['taxes_state'][:, yr] = tax_state
            
            health_premium *= (1 + inf_paths[:, yr])
            history['health_cost'][:, yr] = health_premium
            
            medicare_cost = np.zeros(self.iterations)
            if age >= 65 and "FEHB" not in self.inputs['health_plan'] and "TRICARE" not in self.inputs['health_plan']:
                medicare_cost += MEDICARE_PART_B_BASE
                for i in range(len(irmaa_brackets)):
                    limit, surcharge = irmaa_brackets[i]
                    medicare_cost = np.where(agi > (irmaa_brackets[i-1][0] if i>0 else 0), MEDICARE_PART_B_BASE + surcharge, medicare_cost)
            history['medicare_cost'][:, yr] = medicare_cost
            
            history['total_bal'][:, yr] = tsp + roth + taxable + cash
            history['tsp_bal'][:, yr] = tsp
            history['roth_bal'][:, yr] = roth
            history['taxable_bal'][:, yr] = taxable
            history['cash_bal'][:, yr] = cash
            history['hsa_bal'][:, yr] = hsa
            
            history['net_spendable'][:, yr] = scheduled_withdrawal + pension + ss - tax_fed - tax_state - medicare_cost - health_premium
            
        return history

    def objective_function(self, iwr_test):
        history = self.run_mc(iwr_test, seed=42)
        return np.median(history['total_bal'][:, -1]) - self.inputs['target_floor']

    def optimize_iwr(self):
        try:
            return optimize.brentq(self.objective_function, a=0.01, b=0.15, xtol=1e-4, maxiter=15)
        except ValueError:
            return 0.04