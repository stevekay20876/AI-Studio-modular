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
        home_value = np.full(self.iterations, self.inputs.get('home_value', 0.0))
        
        base_pension = np.full(self.iterations, self.inputs['pension_est'])
        
        ss_claim_age = self.inputs.get('ss_claim_age', 67)
        months_early = max(0, (67 - ss_claim_age) * 12)
        months_late = max(0, (ss_claim_age - 67) * 12)
        reduction = (min(36, months_early) * (5/900)) + (max(0, months_early - 36) * (5/1200))
        increase = months_late * (8/1200)
        ss_modifier = 1.0 - reduction + increase
        base_ss = np.full(self.iterations, self.inputs['ss_fra'] * ss_modifier)
        
        scheduled_withdrawal = np.zeros(self.iterations)
        
        history = {
            'total_bal': np.zeros((self.iterations, self.years)),
            'tsp_bal': np.zeros((self.iterations, self.years)),
            'roth_bal': np.zeros((self.iterations, self.years)),
            'taxable_bal': np.zeros((self.iterations, self.years)),
            'cash_bal': np.zeros((self.iterations, self.years)),
            'hsa_bal': np.zeros((self.iterations, self.years)),
            'home_value': np.zeros((self.iterations, self.years)),
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
            'salary_income': np.zeros((self.iterations, self.years)),
            'port_return': np.zeros((self.iterations, self.years)),
            'real_return': np.zeros((self.iterations, self.years)),
            'inflation': inf_paths,
            'constraint_active': np.zeros((self.iterations, self.years)),
            'ss_income': np.zeros((self.iterations, self.years)),
            'pension_income': np.zeros((self.iterations, self.years)),
            'roth_conversion': np.zeros((self.iterations, self.years)),
        }

        age = self.inputs['current_age']
        ret_age = self.inputs['ret_age']
        current_year = datetime.datetime.now().year
        
        filing_status = self.inputs['filing_status']
        base_salary = self.inputs.get('current_salary', 0.0)
        phased_ret_active = self.inputs.get('phased_ret_active', False)
        phased_age = self.inputs.get('phased_ret_age', ret_age)
        
        base_health_premium = self.inputs.get('health_cost', 0.0)
        base_oop_cost = self.inputs.get('oop_cost', 0.0)
        health_plan = self.inputs.get('health_plan', "None/Self-Insure")
        moop_idx = 1 if filing_status == 'MFJ' else 0
        base_moop = MOOP_LIMITS.get(health_plan, (999999, 999999))[moop_idx]
        
        mortgage_pmt = self.inputs.get('mortgage_pmt', 0.0)
        mortgage_yrs = self.inputs.get('mortgage_yrs', 0)
        annual_savings = self.inputs.get('annual_savings', 0.0)
        
        deduction = STD_DED_MFJ if filing_status == 'MFJ' else STD_DED_SINGLE
        brackets = TAX_BRACKETS_MFJ if filing_status == 'MFJ' else TAX_BRACKETS_SINGLE
        irmaa_brackets = IRMAA_BRACKETS_MFJ if filing_status == 'MFJ' else IRMAA_BRACKETS_SINGLE

        state_str = self.inputs.get('state', '').strip().upper()
        county_str = self.inputs.get('county', '').strip().upper()
        tax_free_states = ["TX", "FL", "NV", "WA", "SD", "WY", "AK", "TN", "NH"]
        state_tax_rate = 0.0 if state_str in tax_free_states else (0.045 if state_str != "" else 0.0)
        local_tax_rate = 0.025 if county_str != "" and state_str in ["MD", "IN", "PA", "OH", "NY"] else (0.010 if county_str != "" else 0.0)
        combined_state_local_rate = state_tax_rate + local_tax_rate

        cum_inf = np.ones(self.iterations)

        for yr in range(self.years):
            age += 1
            current_year += 1
            
            if yr > 0:
                cum_inf *= (1 + np.maximum(0, inf_paths[:, yr]))
            
            # --- HOME VALUE APPRECIATION ---
            home_value *= (1 + inf_paths[:, yr]) # Home grows with standard inflation
            history['home_value'][:, yr] = home_value
            
            # --- SALARY & PHASED RETIREMENT LOGIC ---
            inflated_salary = base_salary * cum_inf
            current_salary_income = np.zeros(self.iterations)
            current_pension = np.zeros(self.iterations)
            
            if age < ret_age:
                # Accumulation Phase
                tsp += (annual_savings * 0.70)
                taxable += (annual_savings * 0.30)
                
                if phased_ret_active and age >= phased_age:
                    current_salary_income = inflated_salary * 0.50
                    # FERS COLA applied to pension base
                    fers_cola = np.where(inf_paths[:, yr] <= 0.02, inf_paths[:, yr], np.where(inf_paths[:, yr] <= 0.03, 0.02, inf_paths[:, yr] - 0.01))
                    base_pension *= (1 + np.maximum(0, fers_cola))
                    current_pension = base_pension * 0.50
                else:
                    current_salary_income = inflated_salary
                    current_pension = np.zeros(self.iterations)
            else:
                # Full Retirement Phase
                current_salary_income = np.zeros(self.iterations)
                fers_cola = np.where(inf_paths[:, yr] <= 0.02, inf_paths[:, yr], np.where(inf_paths[:, yr] <= 0.03, 0.02, inf_paths[:, yr] - 0.01))
                base_pension *= (1 + np.maximum(0, fers_cola))
                current_pension = base_pension
                
            history['salary_income'][:, yr] = current_salary_income
            history['pension_income'][:, yr] = current_pension
            
            prev_total_port = tsp + roth + taxable + cash
            tsp *= (1 + returns[:, yr, 1])
            roth *= (1 + returns[:, yr, 2])
            taxable *= (1 + returns[:, yr, 3])
            hsa *= (1 + returns[:, yr, 4])
            cash *= (1 + cash_ret)
            
            current_total_port = tsp + roth + taxable + cash
            history['port_return'][:, yr] = (current_total_port - prev_total_port) / np.maximum(prev_total_port, 1)
            history['real_return'][:, yr] = history['port_return'][:, yr] - inf_paths[:, yr]
            
            ss_haircut = 0.79 if current_year >= 2035 else 1.0
            base_ss *= (1 + np.maximum(0, inf_paths[:, yr]))
            ss = np.where(age >= ss_claim_age, base_ss * ss_haircut, 0)
            history['ss_income'][:, yr] = ss
            
            w_needed = np.zeros(self.iterations)
            constraint_flag = np.zeros(self.iterations)
            
            if age == ret_age:
                scheduled_withdrawal = current_total_port * iwr
            
            if age >= ret_age:
                if yr > 0 and age > ret_age:
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

            history['constraint_active'][:, yr] = constraint_flag
            
            rmd_divisor = np.maximum(1.9, 24.6 - (age - 75) * 0.8)
            rmd_rate = np.where(age >= 75, 1.0 / rmd_divisor, 0.0)
            rmds = tsp * rmd_rate
            history['rmds'][:, yr] = rmds
            tsp -= rmds
            
            w_remaining = np.maximum(0, w_needed - rmds)
            excess_rmd = np.maximum(0, rmds - w_needed)
            history['extra_rmd'][:, yr] = excess_rmd
            
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
            
            taxable += excess_rmd
            
            gross_income = rmds + w_tsp + current_pension + (ss * 0.85) + current_salary_income
            magi = gross_income.copy() 
            taxable_income = np.maximum(0, gross_income - deduction) 
            
            base_tax_fed = np.zeros(self.iterations)
            for i in range(len(brackets)):
                prev_limit = brackets[i-1][0] if i > 0 else 0
                limit, rate = brackets[i]
                base_tax_fed += np.clip(taxable_income - prev_limit, 0, limit - prev_limit) * rate
                
            base_tax_state_local = taxable_income * combined_state_local_rate
            
            total_tax_fed = base_tax_fed.copy()
            total_tax_state = base_tax_state_local.copy()
            conv_amt = np.zeros(self.iterations)
            
            final_taxable_income = taxable_income.copy()
            final_magi = magi.copy()
            
            if roth_strategy > 0 and age >= ret_age and age < 75:
                space = np.zeros(self.iterations)
                limit_24_pct = brackets[3][0] 
                
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
                
                # --- STRICT CEILING LIMITS ---
                if roth_strategy in [1, 2, 3]:
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
            
            # --- ACTUARIAL HEALTHCARE OOP MODEL ---
            current_health_premium = base_health_premium * cum_inf
            
            # Age-Morbidity factor: Medical costs naturally rise ~2.5% per year as you age
            age_morbidity = 1.025 ** max(0, age - self.inputs['current_age'])
            # Medical inflation generally runs 1.5x standard CPI
            med_cpi_cum = np.prod(1 + (np.maximum(0, inf_paths[:, :yr+1]) * 1.5), axis=1) if yr > 0 else np.ones(self.iterations)
            
            raw_oop = base_oop_cost * med_cpi_cum * age_morbidity
            current_moop = base_moop * cum_inf
            inflated_oop = np.minimum(raw_oop, current_moop)
            
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
            
            history['health_cost'][:, yr] = current_health_premium + oop_remainder
            
            current_mortgage = np.full(self.iterations, mortgage_pmt if yr < mortgage_yrs else 0.0)
            history['mortgage_cost'][:, yr] = current_mortgage
            
            history['total_bal'][:, yr] = tsp + roth + taxable + cash
            history['tsp_bal'][:, yr] = tsp
            history['roth_bal'][:, yr] = roth
            history['taxable_bal'][:, yr] = taxable
            history['cash_bal'][:, yr] = cash
            history['hsa_bal'][:, yr] = hsa
            
            history['net_spendable'][:, yr] = (actual_portfolio_withdrawal + current_pension + ss + current_salary_income
                                               - total_tax_fed - total_tax_state 
                                               - medicare_cost - history['health_cost'][:, yr] 
                                               - current_mortgage)
            
        return history

    def objective_function(self, iwr_test):
        history = self.run_mc(iwr_test, seed=42, roth_strategy=0)
        median_path = np.median(history['total_bal'], axis=0)
        target_floor = self.inputs.get('target_floor', 0.0)
        
        if np.any(median_path <= 0):
            depletion_year = np.argmax(median_path <= 0)
            years_failed_early = self.years - depletion_year
            penalty = -1000000 * years_failed_early 
            return penalty - target_floor
            
        return median_path[-1] - target_floor

    def optimize_iwr(self):
        try:
            return optimize.brentq(self.objective_function, a=0.01, b=0.12, xtol=1e-4, maxiter=20)
        except ValueError:
            return 0.04
            
    def analyze_roth_strategies(self, opt_iwr):
        hist_base = self.run_mc(opt_iwr, seed=42, roth_strategy=0)
        hist_bracket = self.run_mc(opt_iwr, seed=42, roth_strategy=1)
        hist_irmaa1 = self.run_mc(opt_iwr, seed=42, roth_strategy=2)
        hist_irmaa2 = self.run_mc(opt_iwr, seed=42, roth_strategy=3)
        hist_aggressive = self.run_mc(opt_iwr, seed=42, roth_strategy=4) 
        
        results = {
            'Baseline (None)': {'wealth': np.median(hist_base['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_base['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_base['rmds'], axis=0)), 'hist': hist_base},
            'Current Bracket (Capped at 24%)': {'wealth': np.median(hist_bracket['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_bracket['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_bracket['rmds'], axis=0)), 'hist': hist_bracket},
            'Target IRMAA Tier 1 (Capped at 24%)': {'wealth': np.median(hist_irmaa1['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_irmaa1['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_irmaa1['rmds'], axis=0)), 'hist': hist_irmaa1},
            'Target IRMAA Tier 2 (Capped at 24%)': {'wealth': np.median(hist_irmaa2['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_irmaa2['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_irmaa2['rmds'], axis=0)), 'hist': hist_irmaa2},
            'Aggressive 32% Bracket Fill': {'wealth': np.median(hist_aggressive['total_bal'][:, -1]), 'taxes': np.sum(np.median(hist_aggressive['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist_aggressive['rmds'], axis=0)), 'hist': hist_aggressive}
        }
        return results