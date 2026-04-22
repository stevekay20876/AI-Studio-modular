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
        # Base correlation matrix
        corr = np.array([
            [1.00, -0.15, -0.15, -0.15, -0.15],
            [-0.15, 1.00,  0.85,  0.85,  0.85],
            [-0.15, 0.85,  1.00,  0.85,  0.85],
            [-0.15, 0.85,  0.85,  1.00,  0.85],
            [-0.15, 0.85,  0.85,  0.85,  1.00]
        ])
        # Volatilities (Inflation heavily calibrated to 1.2% historical volatility)
        vols = np.array([
            0.012, 
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
            
        # Heavy-tailed Student's t-shocks
        shocks = t.rvs(df=5, size=(self.iterations, self.years, self.n_assets))
        correlated_shocks = np.einsum('ij,kyj->kyi', self.L, shocks)
        
        # Drift calculation (Geometric Mean adjustment)
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

    def run_mc(self, iwr, seed=None, roth_strategy=0):
        returns, inf_paths = self.generate_stochastic_paths(seed=seed)
        cash_ret = self.inputs['cash_ret']
        
        # Balance Init
        tsp = np.full(self.iterations, self.inputs['tsp_bal'])
        roth = np.full(self.iterations, self.inputs['roth_bal'])
        taxable = np.full(self.iterations, self.inputs['taxable_bal'])
        hsa = np.full(self.iterations, self.inputs['hsa_bal'])
        cash = np.full(self.iterations, self.inputs['cash_bal'])
        
        # Income Bases Init (To track compounding COLAs over time)
        base_pension = np.full(self.iterations, self.inputs['pension_est'])
        base_ss = np.full(self.iterations, self.inputs['ss_fra'])
        
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
            'rmds': np.zeros((self.iterations, self.years)),
            'extra_rmd': np.zeros((self.iterations, self.years)),
            'taxes_fed': np.zeros((self.iterations, self.years)),
            'taxes_state': np.zeros((self.iterations, self.years)),
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
        health_premium = self.inputs['health_cost']
        state_tax_rate = 0.045 if self.inputs['state'].strip() != "" else 0.0
        
        deduction = STD_DED_MFJ if self.inputs['filing_status'] == 'MFJ' else STD_DED_SINGLE
        brackets = TAX_BRACKETS_MFJ if self.inputs['filing_status'] == 'MFJ' else TAX_BRACKETS_SINGLE
        irmaa_brackets = IRMAA_BRACKETS_MFJ if self.inputs['filing_status'] == 'MFJ' else IRMAA_BRACKETS_SINGLE

        for yr in range(self.years):
            age += 1
            current_year += 1
            
            # 1. Apply Returns
            prev_total_port = tsp + roth + taxable + cash
            tsp *= (1 + returns[:, yr, 1])
            roth *= (1 + returns[:, yr, 2])
            taxable *= (1 + returns[:, yr, 3])
            hsa *= (1 + returns[:, yr, 4])
            cash *= (1 + cash_ret)
            
            current_total_port = tsp + roth + taxable + cash
            history['port_return'][:, yr] = (current_total_port - prev_total_port) / np.maximum(prev_total_port, 1)
            history['real_return'][:, yr] = history['port_return'][:, yr] - inf_paths[:, yr]
            
            # 2. COLA MATH
            if yr > 0:
                cpi = np.maximum(0, inf_paths[:, yr]) # CPI cannot be negative for federal COLAs
                
                # Social Security full CPI
                base_ss *= (1 + cpi)
                
                # FERS Diet COLA Logic (Full if <=2%, 2% if 2-3%, CPI-1% if >3%)
                fers_cola = np.where(cpi <= 0.02, cpi, np.where(cpi <= 0.03, 0.02, cpi - 0.01))
                fers_cola = np.where(age >= 62, fers_cola, 0.0) # Begins at 62
                base_pension *= (1 + fers_cola)

            pension = np.where(age >= self.inputs['ret_age'], base_pension, 0)
            ss_haircut = 0.79 if current_year >= 2035 else 1.0
            ss = np.where(age >= 67, base_ss * ss_haircut, 0)
            history['pension_income'][:, yr] = pension
            history['ss_income'][:, yr] = ss
            
            # 3. Guardrails (Guyton-Klinger & SORR)
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
            
            # 4. RMDs
            rmd_rate = 0.036 if age >= 75 else 0.0
            rmds = tsp * rmd_rate
            history['rmds'][:, yr] = rmds
            tsp -= rmds
            
            w_remaining = np.maximum(0, w_needed - rmds)
            excess_rmd = np.maximum(0, rmds - w_needed)
            history['extra_rmd'][:, yr] = excess_rmd
            
            # 5. Withdrawals & Liquidation Hierarchy (SORR)
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
            
            # 6. Base Taxes
            taxable_income = rmds + w_tsp + w_tsp_fallback + pension + (ss * 0.85)
            agi = np.maximum(0, taxable_income - deduction)
            base_tax_fed = np.zeros(self.iterations)
            for i in range(len(brackets)):
                prev_limit = brackets[i-1][0] if i > 0 else 0
                limit, rate = brackets[i]
                base_tax_fed += np.clip(agi - prev_limit, 0, limit - prev_limit) * rate
                
            tax_state = agi * state_tax_rate
            history['taxes_state'][:, yr] = tax_state
            
            # 7. REAL ROTH CONVERSION LOGIC
            total_tax_fed = base_tax_fed.copy()
            conv_amt = np.zeros(self.iterations)
            
            if roth_strategy > 0 and age < 75:
                space = np.zeros(self.iterations)
                if roth_strategy == 1: # Fill Current Bracket
                    for limit, rate in brackets:
                        mask = (agi < limit) & (space == 0)
                        space[mask] = limit - agi[mask] - 1
                elif roth_strategy == 2: # Fill to IRMAA Tier 1
                    irmaa_limit = irmaa_brackets[0][0]
                    space = np.maximum(0, irmaa_limit - agi - 1)
                
                conv_amt = np.minimum(space, tsp)
                
                # Calculate Extra Tax
                new_agi = agi + conv_amt
                new_tax_fed = np.zeros(self.iterations)
                for i in range(len(brackets)):
                    prev_limit = brackets[i-1][0] if i > 0 else 0
                    limit, rate = brackets[i]
                    new_tax_fed += np.clip(new_agi - prev_limit, 0, limit - prev_limit) * rate
                
                extra_tax = new_tax_fed - base_tax_fed
                
                # Pay conversion tax from outside assets first to maximize tax-free growth
                w_tax_cash = np.minimum(cash, extra_tax)
                cash -= w_tax_cash
                rem_tax = extra_tax - w_tax_cash
                
                w_tax_taxable = np.minimum(taxable, rem_tax)
                taxable -= w_tax_taxable
                rem_tax -= w_tax_taxable
                
                # Move funds
                net_to_roth = conv_amt - rem_tax # Tax withheld if cash/taxable run out
                tsp -= conv_amt
                roth += net_to_roth
                total_tax_fed = new_tax_fed
            
            history['roth_conversion'][:, yr] = conv_amt
            history['taxes_fed'][:, yr] = total_tax_fed
            
            # 8. Medicare & Health Costs
            health_premium *= (1 + inf_paths[:, yr])
            history['health_cost'][:, yr] = health_premium
            medicare_cost = np.zeros(self.iterations)
            if age >= 65 and "FEHB" not in self.inputs['health_plan'] and "TRICARE" not in self.inputs['health_plan']:
                medicare_cost += MEDICARE_PART_B_BASE
                for i in range(len(irmaa_brackets)):
                    limit, surcharge = irmaa_brackets[i]
                    medicare_cost = np.where(new_agi if roth_strategy > 0 else agi > (irmaa_brackets[i-1][0] if i>0 else 0), MEDICARE_PART_B_BASE + surcharge, medicare_cost)
            history['medicare_cost'][:, yr] = medicare_cost
            
            # 9. Mortgage Logic
            current_mortgage = np.full(self.iterations, self.inputs['mortgage_pmt'] if yr < self.inputs['mortgage_yrs'] else 0.0)
            history['mortgage_cost'][:, yr] = current_mortgage
            
            # Record Ending Balances
            history['total_bal'][:, yr] = tsp + roth + taxable + cash
            history['tsp_bal'][:, yr] = tsp
            history['roth_bal'][:, yr] = roth
            history['taxable_bal'][:, yr] = taxable
            history['cash_bal'][:, yr] = cash
            history['hsa_bal'][:, yr] = hsa
            
            # 10. Net Spendable (Discretionary Lifestyle)
            # Uses base_tax_fed so lifestyle doesn't drop just to pay discretionary conversion taxes
            history['net_spendable'][:, yr] = (scheduled_withdrawal + pension + ss 
                                               - base_tax_fed - tax_state 
                                               - medicare_cost - health_premium 
                                               - current_mortgage)
            
        return history

    def objective_function(self, iwr_test):
        # Deterministic solver runs on strict seed, baseline strategy
        history = self.run_mc(iwr_test, seed=42, roth_strategy=0)
        return np.median(history['total_bal'][:, -1]) - self.inputs['target_floor']

    def optimize_iwr(self):
        try:
            return optimize.brentq(self.objective_function, a=0.01, b=0.15, xtol=1e-4, maxiter=15)
        except ValueError:
            return 0.04
            
    def analyze_roth_strategies(self, opt_iwr):
        # Runs the MC engine iteratively strictly on the same seed to ensure fair exact comparison
        hist_base = self.run_mc(opt_iwr, seed=42, roth_strategy=0)
        hist_bracket = self.run_mc(opt_iwr, seed=42, roth_strategy=1)
        hist_irmaa = self.run_mc(opt_iwr, seed=42, roth_strategy=2)
        
        results = {
            'Baseline': {'wealth': np.median(hist_base['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_base['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_base['rmds'], axis=0)), 'hist': hist_base},
            'Current Bracket': {'wealth': np.median(hist_bracket['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_bracket['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_bracket['rmds'], axis=0)), 'hist': hist_bracket},
            'IRMAA Tier 1': {'wealth': np.median(hist_irmaa['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_irmaa['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_irmaa['rmds'], axis=0)), 'hist': hist_irmaa}
        }
        return results