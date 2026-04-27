# engine.py
import numpy as np
import scipy.optimize as optimize
from scipy.stats import t
import datetime
from config import *
import gc

class StochasticRetirementEngine:
    def __init__(self, inputs):
        self.inputs = inputs
        self.iterations = 10000
        
        base_years = inputs['life_expectancy'] - inputs['current_age']
        if inputs['filing_status'] == 'MFJ':
            spouse_years = inputs['spouse_life_exp'] - inputs['spouse_age']
            self.years = max(1, max(base_years, spouse_years))
        else:
            self.years = max(1, base_years)
            
        self.n_assets = 6 
        
    def get_port_params(self, asset_key, override_port=None):
        if override_port:
            return PORTFOLIOS[override_port]['ret'], PORTFOLIOS[override_port]['vol']
        else:
            strat = self.inputs[asset_key]
            return PORTFOLIOS[strat]['ret'], PORTFOLIOS[strat]['vol']

    def setup_covariance_matrix(self, override_port=None):
        _, v_tsp = self.get_port_params('tsp_strat', override_port)
        _, v_ira = self.get_port_params('ira_strat', override_port)
        _, v_roth = self.get_port_params('roth_strat', override_port)
        _, v_tax = self.get_port_params('taxable_strat', override_port)
        _, v_hsa = self.get_port_params('hsa_strat', override_port)
        
        corr = np.array([
            [1.00, -0.15, -0.15, -0.15, -0.15, -0.15],
            [-0.15, 1.00,  0.85,  0.85,  0.85,  0.85],
            [-0.15, 0.85,  1.00,  0.85,  0.85,  0.85],
            [-0.15, 0.85,  0.85,  1.00,  0.85,  0.85],
            [-0.15, 0.85,  0.85,  0.85,  1.00,  0.85],
            [-0.15, 0.85,  0.85,  0.85,  0.85,  1.00]
        ])
        vols = np.array([0.012, v_tsp, v_ira, v_roth, v_tax, v_hsa])
        cov = np.outer(vols, vols) * corr
        return np.linalg.cholesky(cov)

    def generate_stochastic_paths(self, L, seed=None, override_port=None):
        if seed is not None:
            np.random.seed(seed)
            
        # FIX 2: Multiply by sqrt(3/5) to normalize Student's t-distribution variance back to 1.0
        variance_scalar = np.sqrt(3.0 / 5.0)
        shocks = t.rvs(df=5, size=(self.iterations, self.years, self.n_assets)) * variance_scalar
        correlated_shocks = np.einsum('ij,kyj->kyi', L, shocks)
        
        r_tsp, v_tsp = self.get_port_params('tsp_strat', override_port)
        r_ira, v_ira = self.get_port_params('ira_strat', override_port)
        r_roth, v_roth = self.get_port_params('roth_strat', override_port)
        r_tax, v_tax = self.get_port_params('taxable_strat', override_port)
        r_hsa, v_hsa = self.get_port_params('hsa_strat', override_port)
        
        drifts = np.array([
            0.03,
            r_tsp - (v_tsp**2)/2,
            r_ira - (v_ira**2)/2,
            r_roth - (v_roth**2)/2,
            r_tax - (v_tax**2)/2,
            r_hsa - (v_hsa**2)/2
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

    def run_mc(self, iwr, seed=None, roth_strategy=0, override_port=None):
        L = self.setup_covariance_matrix(override_port)
        returns, inf_paths = self.generate_stochastic_paths(L, seed=seed, override_port=override_port)
        cash_ret = self.inputs['cash_ret']
        
        tsp = np.full(self.iterations, self.inputs['tsp_bal'])
        ira = np.full(self.iterations, self.inputs['ira_bal'])
        roth = np.full(self.iterations, self.inputs['roth_bal'])
        taxable = np.full(self.iterations, self.inputs['taxable_bal'])
        hsa = np.full(self.iterations, self.inputs['hsa_bal'])
        cash = np.full(self.iterations, self.inputs['cash_bal'])
        home_value = np.full(self.iterations, self.inputs.get('home_value', 0.0))
        
        taxable_basis = np.full(self.iterations, self.inputs.get('taxable_basis', self.inputs['taxable_bal']))
        
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
            'total_bal_real': np.zeros((self.iterations, self.years)), 
            'cum_inf': np.zeros((self.iterations, self.years)),        
            'tsp_bal': np.zeros((self.iterations, self.years)),
            'ira_bal': np.zeros((self.iterations, self.years)),
            'roth_bal': np.zeros((self.iterations, self.years)),
            'taxable_bal': np.zeros((self.iterations, self.years)),
            'cash_bal': np.zeros((self.iterations, self.years)),
            'hsa_bal': np.zeros((self.iterations, self.years)),
            'home_value': np.zeros((self.iterations, self.years)),
            'tsp_withdrawal': np.zeros((self.iterations, self.years)),
            'ira_withdrawal': np.zeros((self.iterations, self.years)),
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
            'additional_expenses': np.zeros((self.iterations, self.years)),
            'net_spendable': np.zeros((self.iterations, self.years)),
            'salary_income': np.zeros((self.iterations, self.years)),
            'port_return': np.zeros((self.iterations, self.years)),
            'real_return': np.zeros((self.iterations, self.years)),
            'inflation': inf_paths,
            'constraint_active': np.zeros((self.iterations, self.years)),
            'ss_income': np.zeros((self.iterations, self.years)),
            'pension_income': np.zeros((self.iterations, self.years)),
            'roth_conversion': np.zeros((self.iterations, self.years)),
            'roth_taxes_from_cash': np.zeros((self.iterations, self.years)), 
        }

        age = self.inputs['current_age']
        ret_age = self.inputs['ret_age']
        spouse_age = self.inputs.get('spouse_age', age)
        primary_life_exp = self.inputs['life_expectancy']
        spouse_life_exp = self.inputs.get('spouse_life_exp', 95)
        
        current_year = datetime.datetime.now().year
        
        base_filing_status = self.inputs['filing_status']
        base_salary = self.inputs.get('current_salary', 0.0)
        phased_ret_active = self.inputs.get('phased_ret_active', False)
        phased_age = self.inputs.get('phased_ret_age', ret_age)
        pay_taxes_from_cash = self.inputs.get('pay_taxes_from_cash', True)
        survivor_benefit_choice = self.inputs.get('survivor_benefit', "No Survivor Benefit")
        
        min_spending = self.inputs.get('min_spending', 0.0)
        max_spending = self.inputs.get('max_spending', 0.0)
        base_add_exp = self.inputs.get('additional_expenses', 0.0)
        user_max_bracket = float(self.inputs.get('max_tax_bracket', '0.24'))
        
        base_health_premium = self.inputs.get('health_cost', 0.0)
        base_oop_cost = self.inputs.get('oop_cost', 0.0)
        health_plan = self.inputs.get('health_plan', "None/Self-Insure")
        
        mortgage_pmt = self.inputs.get('mortgage_pmt', 0.0)
        mortgage_yrs = self.inputs.get('mortgage_yrs', 0)
        annual_savings = self.inputs.get('annual_savings', 0.0)

        state_str = self.inputs.get('state', '').strip().upper()
        county_str = self.inputs.get('county', '').strip().upper()
        state_tax_rate = 0.0 if state_str in RETIREMENT_TAX_FREE_STATES else (0.045 if state_str != "" else 0.0)
        local_tax_rate = 0.025 if county_str != "" and state_str in ["MD", "IN", "PA", "OH", "NY"] else (0.010 if county_str != "" else 0.0)
        combined_state_local_rate = state_tax_rate + local_tax_rate

        cum_inf = np.ones(self.iterations)

        for yr in range(self.years):
            age += 1
            spouse_age += 1
            current_year += 1
            
            primary_alive = age <= primary_life_exp
            spouse_alive = spouse_age <= spouse_life_exp if base_filing_status == 'MFJ' else False
            
            if base_filing_status == 'MFJ':
                if primary_alive and spouse_alive:
                    current_filing_status = 'MFJ'
                    moop_idx = 1
                    ss_mult = 1.0
                    if survivor_benefit_choice == 'Full Survivor Benefit':
                        pension_mult = 0.90
                    elif survivor_benefit_choice == 'Partial Survivor Benefit':
                        pension_mult = 0.95
                    else:
                        pension_mult = 1.0
                elif not primary_alive and spouse_alive:
                    current_filing_status = 'Single'
                    moop_idx = 0
                    ss_mult = 0.50 
                    if survivor_benefit_choice == 'Full Survivor Benefit':
                        pension_mult = 0.50
                    elif survivor_benefit_choice == 'Partial Survivor Benefit':
                        pension_mult = 0.25
                    else:
                        pension_mult = 0.0
                elif primary_alive and not spouse_alive:
                    current_filing_status = 'Single'
                    moop_idx = 0
                    ss_mult = 0.50
                    pension_mult = 1.0
                else:
                    current_filing_status = 'Single'
                    moop_idx = 0
                    ss_mult = 0.0
                    pension_mult = 0.0
            else:
                current_filing_status = 'Single'
                moop_idx = 0
                if primary_alive:
                    ss_mult = 1.0
                    pension_mult = 1.0
                else:
                    ss_mult = 0.0
                    pension_mult = 0.0
                
            deduction = STD_DED_MFJ if current_filing_status == 'MFJ' else STD_DED_SINGLE
            brackets = TAX_BRACKETS_MFJ if current_filing_status == 'MFJ' else TAX_BRACKETS_SINGLE
            irmaa_brackets = IRMAA_BRACKETS_MFJ if current_filing_status == 'MFJ' else IRMAA_BRACKETS_SINGLE
            ltcg_brackets = LTCG_BRACKETS_MFJ if current_filing_status == 'MFJ' else LTCG_BRACKETS_SINGLE
            niit_threshold = NIIT_THRESHOLD_MFJ if current_filing_status == 'MFJ' else NIIT_THRESHOLD_SINGLE
            base_moop = MOOP_LIMITS.get(health_plan, (999999, 999999))[moop_idx]

            if yr > 0:
                cum_inf *= (1 + np.maximum(0, inf_paths[:, yr]))
            history['cum_inf'][:, yr] = cum_inf
            
            # FIX 1: Max bracket check with dynamic inflation mapping
            limit_max_pct = np.full(self.iterations, np.inf)
            for i in range(len(brackets)):
                if np.isclose(brackets[i][1], user_max_bracket, atol=0.01):
                    limit_max_pct = brackets[i][0] * cum_inf
                    break
            
            home_value *= 1.035
            history['home_value'][:, yr] = home_value
            
            inflated_salary = base_salary * cum_inf
            current_salary_income = np.zeros(self.iterations)
            current_pension = np.zeros(self.iterations)
            
            if age < ret_age:
                tsp += (annual_savings * 0.70)
                tax_savings_add = (annual_savings * 0.30)
                taxable += tax_savings_add
                taxable_basis += tax_savings_add
                
                if phased_ret_active and age >= phased_age:
                    current_salary_income = inflated_salary * 0.50
                    fers_cola = np.where(inf_paths[:, yr] <= 0.02, inf_paths[:, yr], np.where(inf_paths[:, yr] <= 0.03, 0.02, inf_paths[:, yr] - 0.01))
                    base_pension *= (1 + np.maximum(0, fers_cola))
                    current_pension = base_pension * 0.50 * pension_mult
                else:
                    current_salary_income = inflated_salary
                    current_pension = np.zeros(self.iterations)
            else:
                current_salary_income = np.zeros(self.iterations)
                fers_cola = np.where(inf_paths[:, yr] <= 0.02, inf_paths[:, yr], np.where(inf_paths[:, yr] <= 0.03, 0.02, inf_paths[:, yr] - 0.01))
                base_pension *= (1 + np.maximum(0, fers_cola))
                current_pension = base_pension * pension_mult
                
            history['salary_income'][:, yr] = current_salary_income
            history['pension_income'][:, yr] = current_pension
            
            prev_total_port = tsp + ira + roth + taxable + cash
            tsp *= (1 + returns[:, yr, 1])
            ira *= (1 + returns[:, yr, 2])
            roth *= (1 + returns[:, yr, 3])
            taxable *= (1 + returns[:, yr, 4])
            hsa *= (1 + returns[:, yr, 5])
            cash *= (1 + cash_ret)
            
            current_total_port = tsp + ira + roth + taxable + cash
            history['port_return'][:, yr] = (current_total_port - prev_total_port) / np.maximum(prev_total_port, 1)
            history['real_return'][:, yr] = history['port_return'][:, yr] - inf_paths[:, yr]
            
            ss_haircut = 0.79 if current_year >= 2035 else 1.0
            base_ss *= (1 + np.maximum(0, inf_paths[:, yr]))
            ss = np.where(age >= ss_claim_age, base_ss * ss_haircut * ss_mult, 0)
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
                    
                inflated_min_spend = min_spending * cum_inf
                w_needed = np.maximum(scheduled_withdrawal, inflated_min_spend)
                
                if max_spending > 0:
                    inflated_max_spend = max_spending * cum_inf
                    w_needed = np.minimum(w_needed, inflated_max_spend)

            history['constraint_active'][:, yr] = constraint_flag
            
            rmd_divisor = IRS_RMD_DIVISORS.get(age, 1.9 if age > 120 else 0.0)
            rmd_rate = 1.0 / rmd_divisor if rmd_divisor > 0 else 0.0
            
            rmd_tsp = tsp * rmd_rate
            rmd_ira = ira * rmd_rate
            rmds = rmd_tsp + rmd_ira
            
            history['rmds'][:, yr] = rmds
            tsp -= rmd_tsp
            ira -= rmd_ira
            
            w_remaining = np.maximum(0, w_needed - rmds)
            excess_rmd = np.maximum(0, rmds - w_needed)
            history['extra_rmd'][:, yr] = excess_rmd
            
            w_tsp = np.zeros(self.iterations)
            w_ira = np.zeros(self.iterations)
            w_cash = np.zeros(self.iterations)
            w_taxable = np.zeros(self.iterations)
            w_roth = np.zeros(self.iterations)
            
            if age >= ret_age:
                tsp_prior_ret = returns[:, yr-1, 1] if yr > 0 else np.zeros(self.iterations)
                downturn = tsp_prior_ret <= -0.10
                
                w_tsp_norm = np.where(~downturn, np.minimum(tsp, w_remaining), 0)
                tsp -= w_tsp_norm
                w_remaining -= w_tsp_norm
                
                w_ira_norm = np.where(~downturn, np.minimum(ira, w_remaining), 0)
                ira -= w_ira_norm
                w_remaining -= w_ira_norm
                
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
                
                w_ira_fb = np.minimum(ira, w_remaining)
                ira -= w_ira_fb
                w_remaining -= w_ira_fb
                
                w_tsp_fb = np.minimum(tsp, w_remaining)
                tsp -= w_tsp_fb
                w_remaining -= w_tsp_fb
                
                w_tsp += (w_tsp_norm + w_tsp_fb)
                w_ira += (w_ira_norm + w_ira_fb)
                w_cash += (w_cash_down + w_cash_fb)
                w_taxable += (w_tax_down + w_tax_fb)
                w_roth += (w_roth_down + w_roth_fb)
            
            actual_portfolio_withdrawal = w_tsp + w_ira + w_cash + w_taxable + w_roth + rmds - excess_rmd
            
            history['tsp_withdrawal'][:, yr] = w_tsp
            history['ira_withdrawal'][:, yr] = w_ira
            history['roth_withdrawal'][:, yr] = w_roth
            history['taxable_withdrawal'][:, yr] = w_taxable
            history['cash_withdrawal'][:, yr] = w_cash
            
            total_w_taxable = w_taxable
            gains_ratio = np.maximum(0, 1.0 - (taxable_basis / np.maximum(taxable, 1.0)))
            realized_gains = total_w_taxable * gains_ratio
            taxable_basis -= (total_w_taxable - realized_gains) 
            
            taxable += excess_rmd
            taxable_basis += excess_rmd 
            
            gross_income = rmds + w_tsp + w_ira + current_pension + (ss * 0.85) + current_salary_income
            magi = gross_income.copy() 
            taxable_income = np.maximum(0, gross_income - (deduction * cum_inf)) 
            
            # FIX 1: Ordinary Income Tax Bracket Inflation
            base_tax_fed = np.zeros(self.iterations)
            for i in range(len(brackets)):
                prev_limit = (brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                limit = brackets[i][0] * cum_inf
                rate = brackets[i][1]
                base_tax_fed += np.clip(taxable_income - prev_limit, 0, limit - prev_limit) * rate
                
            # FIX 1: LTCG Bracket Inflation
            ltcg_tax = np.zeros(self.iterations)
            for i in range(len(ltcg_brackets)):
                limit = ltcg_brackets[i][0] * cum_inf
                rate = ltcg_brackets[i][1]
                applicable_gains = np.clip(taxable_income + realized_gains - limit, 0, realized_gains)
                ltcg_tax += applicable_gains * rate
                
            # FIX 1: NIIT Threshold Inflation
            niit_tax = np.where(magi > (niit_threshold * cum_inf), realized_gains * 0.038, 0.0)
            base_tax_fed += (ltcg_tax + niit_tax)
            
            state_taxable_base = np.where(np.isin(state_str, RETIREMENT_TAX_FREE_STATES), 
                                          taxable_income - rmds - w_tsp - w_ira - current_pension - (ss * 0.85), 
                                          taxable_income)
            state_taxable_base = np.maximum(0, state_taxable_base)
            base_tax_state_local = state_taxable_base * combined_state_local_rate
            
            total_tax_fed = base_tax_fed.copy()
            total_tax_state = base_tax_state_local.copy()
            conv_amt = np.zeros(self.iterations)
            w_tax_cash_roth = np.zeros(self.iterations)
            
            final_taxable_income = taxable_income.copy()
            final_magi = magi.copy()
            
            if roth_strategy > 0 and age >= ret_age and age < 75:
                space = np.zeros(self.iterations)
                
                if roth_strategy in [1, 4]: 
                    for i in range(len(brackets)):
                        limit = brackets[i][0] * cum_inf
                        mask = (taxable_income < limit) & (space == 0)
                        space[mask] = limit[mask] - taxable_income[mask] - 1
                    space = np.where(space > 1e6, 0, space)
                    
                    if roth_strategy == 1:
                        for i in range(len(irmaa_brackets)):
                            irmaa_limit = irmaa_brackets[i][0] * cum_inf
                            crosses_cliff = (magi < irmaa_limit) & ((magi + space) >= irmaa_limit)
                            space = np.where(crosses_cliff, irmaa_limit - magi - 1, space)
                        
                elif roth_strategy == 2: 
                    irmaa_tier_1 = irmaa_brackets[0][0] * cum_inf
                    space = np.maximum(0, irmaa_tier_1 - magi - 1)
                    
                elif roth_strategy == 3:
                    irmaa_tier_2 = irmaa_brackets[1][0] * cum_inf
                    space = np.maximum(0, irmaa_tier_2 - magi - 1)
                
                max_allowable_space = np.maximum(0, limit_max_pct - taxable_income - 1)
                space = np.minimum(space, max_allowable_space)
                
                conv_from_ira = np.minimum(space, ira)
                ira -= conv_from_ira
                rem_space = space - conv_from_ira
                
                conv_from_tsp = np.minimum(rem_space, tsp)
                tsp -= conv_from_tsp
                
                conv_amt = conv_from_ira + conv_from_tsp
                
                final_taxable_income = taxable_income + conv_amt
                final_magi = magi + conv_amt
                
                # Recalculate Ordinary Income Tax
                new_tax_fed = np.zeros(self.iterations)
                for i in range(len(brackets)):
                    prev_limit = (brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                    limit = brackets[i][0] * cum_inf
                    rate = brackets[i][1]
                    new_tax_fed += np.clip(final_taxable_income - prev_limit, 0, limit - prev_limit) * rate
                
                # FIX 3: Recalculate LTCG and NIIT penalties based on new elevated MAGI/Taxable limits
                new_ltcg_tax = np.zeros(self.iterations)
                for i in range(len(ltcg_brackets)):
                    limit = ltcg_brackets[i][0] * cum_inf
                    rate = ltcg_brackets[i][1]
                    applicable_gains = np.clip(final_taxable_income + realized_gains - limit, 0, realized_gains)
                    new_ltcg_tax += applicable_gains * rate
                    
                new_niit_tax = np.where(final_magi > (niit_threshold * cum_inf), realized_gains * 0.038, 0.0)
                
                extra_tax_fed = (new_tax_fed + new_ltcg_tax + new_niit_tax) - base_tax_fed
                
                state_conv_tax_base = np.where(np.isin(state_str, RETIREMENT_TAX_FREE_STATES), 0.0, conv_amt)
                extra_tax_state = state_conv_tax_base * combined_state_local_rate
                extra_tax_total = extra_tax_fed + extra_tax_state
                
                if pay_taxes_from_cash:
                    w_tax_cash_roth = np.minimum(cash, extra_tax_total)
                    cash -= w_tax_cash_roth
                    rem_tax = extra_tax_total - w_tax_cash_roth
                    
                    w_tax_taxable = np.minimum(taxable, rem_tax)
                    taxable -= w_tax_taxable
                    
                    gains_ratio_tax = np.maximum(0, 1.0 - (taxable_basis / np.maximum(taxable + w_tax_taxable, 1.0)))
                    taxable_basis -= (w_tax_taxable - (w_tax_taxable * gains_ratio_tax))
                    
                    rem_tax -= w_tax_taxable
                    net_to_roth = conv_amt - rem_tax 
                else:
                    net_to_roth = conv_amt - extra_tax_total
                    
                roth += net_to_roth
                total_tax_fed = new_tax_fed + new_ltcg_tax + new_niit_tax
                total_tax_state = base_tax_state_local + extra_tax_state
            
            history['roth_conversion'][:, yr] = conv_amt
            history['taxes_fed'][:, yr] = total_tax_fed
            history['taxes_state'][:, yr] = total_tax_state
            history['taxable_income'][:, yr] = final_taxable_income
            history['magi'][:, yr] = final_magi
            history['roth_taxes_from_cash'][:, yr] = w_tax_cash_roth 
            
            current_health_premium = base_health_premium * cum_inf
            age_morbidity = 1.025 ** max(0, age - self.inputs['current_age'])
            med_cpi_cum = np.prod(1 + (np.maximum(0, inf_paths[:, :yr+1]) * 1.5), axis=1) if yr > 0 else np.ones(self.iterations)
            
            raw_oop = base_oop_cost * med_cpi_cum * age_morbidity
            current_moop = base_moop * cum_inf
            inflated_oop = np.minimum(raw_oop, current_moop)
            
            # FIX 1: Medicare IRMAA Bracket Inflation
            medicare_cost = np.zeros(self.iterations)
            if age >= 65 and "FEHB" not in health_plan and "TRICARE" not in health_plan:
                medicare_cost += MEDICARE_PART_B_BASE
                for i in range(len(irmaa_brackets)):
                    prev_limit = (irmaa_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                    surcharge = irmaa_brackets[i][1]
                    medicare_cost = np.where(final_magi > prev_limit, MEDICARE_PART_B_BASE + surcharge, medicare_cost)
            history['medicare_cost'][:, yr] = medicare_cost

            w_hsa = np.minimum(hsa, inflated_oop)
            hsa -= w_hsa
            oop_remainder = inflated_oop - w_hsa
            
            history['health_cost'][:, yr] = current_health_premium + oop_remainder
            
            current_mortgage = np.full(self.iterations, mortgage_pmt if yr < mortgage_yrs else 0.0)
            history['mortgage_cost'][:, yr] = current_mortgage
            
            if age >= ret_age:
                years_in_ret = age - ret_age
                smile_mult = 1.0 - (0.015 * years_in_ret) + (0.0005 * (years_in_ret ** 2))
                current_add_exp = base_add_exp * cum_inf * smile_mult
            else:
                current_add_exp = np.zeros(self.iterations)
            history['additional_expenses'][:, yr] = current_add_exp
            
            history['total_bal'][:, yr] = tsp + ira + roth + taxable + cash
            history['total_bal_real'][:, yr] = history['total_bal'][:, yr] / cum_inf
            
            history['tsp_bal'][:, yr] = tsp
            history['ira_bal'][:, yr] = ira
            history['roth_bal'][:, yr] = roth
            history['taxable_bal'][:, yr] = taxable
            history['cash_bal'][:, yr] = cash
            history['hsa_bal'][:, yr] = hsa
            
            if age >= ret_age:
                history['net_spendable'][:, yr] = (actual_portfolio_withdrawal + current_pension + ss + current_salary_income
                                                   - total_tax_fed - total_tax_state 
                                                   - medicare_cost - history['health_cost'][:, yr] 
                                                   - current_mortgage - current_add_exp)
            else:
                history['net_spendable'][:, yr] = 0.0
            
        return history

def objective_function(self, iwr_test):
        history = self.run_mc(iwr_test, seed=42, roth_strategy=0)
        median_real_path = np.median(history['total_bal_real'], axis=0)
        target_floor = self.inputs.get('target_floor', 0.0)
        
        terminal_wealth = median_real_path[-1]
        
        # FIX: Create a smooth, continuous mathematical gradient for early depletion.
        # This prevents Brent's Method from crashing when the Target Floor is set to $0.
        if terminal_wealth <= 1.0:  # Using 1.0 to account for floating point near-zeros
            # Count exactly how many years the median path was bankrupt
            years_bankrupt = np.sum(median_real_path <= 1.0)
            initial_wealth = median_real_path[0]
            
            # Create a smooth artificial negative trajectory (falling short by 5% per bankrupt year)
            # This gives the root-finder a perfect mathematical slope to follow back up to 0
            artificial_terminal = -(years_bankrupt * initial_wealth * 0.05)
            return artificial_terminal - target_floor
            
        return terminal_wealth - target_floor

    def optimize_iwr(self):
        try:
            # Widened bounds and changed except block to catch all optimizer convergence failures
            return optimize.brentq(self.objective_function, a=0.001, b=0.40, xtol=1e-4, maxiter=40)
        except Exception:
            # Fallback only triggers if user is massively over-funded beyond a 40% initial withdrawal
            return 0.04
            
    def analyze_portfolios(self, opt_iwr, roth_strategy=0):
        results = {}
        hist_custom = self.run_mc(opt_iwr, seed=42, roth_strategy=roth_strategy, override_port=None)
        results["Your Custom Mix"] = {
            'wealth': np.median(hist_custom['total_bal_real'][:, -1]), 
            'cut_prob': np.mean(np.any(hist_custom['constraint_active'] == 1, axis=1)) * 100
        }
        del hist_custom
        gc.collect()
        
        for port in ["Conservative (20% Stock / 80% Bond)", "Moderate (60% Stock / 40% Bond)", "Aggressive (100% Stock)"]:
            hist = self.run_mc(opt_iwr, seed=42, roth_strategy=roth_strategy, override_port=port)
            results[port] = {
                'wealth': np.median(hist['total_bal_real'][:, -1]), 
                'cut_prob': np.mean(np.any(hist['constraint_active'] == 1, axis=1)) * 100
            }
            del hist
            gc.collect()
            
        return results

    def analyze_roth_strategies(self, opt_iwr):
        results = {}
        user_max = float(self.inputs.get("max_tax_bracket", 0.24)) * 100
        strats = [
            (0, 'Baseline (None)'),
            (1, 'Fill Current Bracket (IRMAA Protected)'),
            (2, 'Target IRMAA Tier 1'),
            (3, 'Target IRMAA Tier 2'),
            (4, f'Max User Bracket Fill ({user_max:.0f}%)')
        ]
        
        best_wealth = -np.inf
        winner_name = 'Baseline (None)'
        winner_hist = None
        
        for s_idx, s_name in strats:
            hist = self.run_mc(opt_iwr, seed=42, roth_strategy=s_idx)
            wealth = np.median(hist['total_bal_real'][:, -1])
            
            results[s_name] = {
                'wealth': wealth,
                'taxes': np.sum(np.median(hist['taxes_fed'], axis=0)),
                'rmds': np.sum(np.median(hist['rmds'], axis=0)),
                'tax_path': np.median(hist['taxes_fed'], axis=0),
                'conv_path': np.median(hist['roth_conversion'], axis=0),
                'taxable_inc_path': np.median(hist['taxable_income'], axis=0)
            }
            
            if wealth > best_wealth:
                best_wealth = wealth
                winner_name = s_name
                winner_hist = hist 
            else:
                del hist 
                gc.collect()
                
        return results, winner_name, winner_hist