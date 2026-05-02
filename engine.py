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
        
        # Calculate derived integer ages for logic thresholds
        current_year = datetime.datetime.now().year
        self.ret_year = int(self.inputs['ret_date'].split("-")[0])
        self.ret_month = int(self.inputs['ret_date'].split("-")[1])
        
        self.s_ret_year = int(self.inputs['s_ret_date'].split("-")[0])
        self.s_ret_month = int(self.inputs['s_ret_date'].split("-")[1])

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
        vols = np.array([0.020, v_tsp, v_ira, v_roth, v_tax, v_hsa])
        cov = np.outer(vols, vols) * corr
        return np.linalg.cholesky(cov)

    def generate_stochastic_paths(self, L, seed=None, override_port=None):
        if seed is not None: np.random.seed(seed)
        variance_scalar = np.sqrt(3.0 / 5.0)
        shocks = t.rvs(df=5, size=(self.iterations, self.years, self.n_assets)) * variance_scalar
        correlated_shocks = np.einsum('ij,kyj->kyi', L, shocks)
        
        r_tsp, v_tsp = self.get_port_params('tsp_strat', override_port)
        r_ira, v_ira = self.get_port_params('ira_strat', override_port)
        r_roth, v_roth = self.get_port_params('roth_strat', override_port)
        r_tax, v_tax = self.get_port_params('taxable_strat', override_port)
        r_hsa, v_hsa = self.get_port_params('hsa_strat', override_port)
        
        drifts = np.array([0.03, r_tsp - (v_tsp**2)/2, r_ira - (v_ira**2)/2, r_roth - (v_roth**2)/2, r_tax - (v_tax**2)/2, r_hsa - (v_hsa**2)/2])
        returns = np.exp(drifts + correlated_shocks) - 1
        inf_paths = np.zeros((self.iterations, self.years))
        
        inf_base, kappa, jump_prob, dt = 0.021, 0.25, 0.04, 1.0            
        sqrt_dt = np.sqrt(dt)
        current_inf = np.full(self.iterations, inf_base)
        
        for yr in range(self.years):
            dW = correlated_shocks[:, yr, 0] * sqrt_dt
            jumps = np.where(np.random.rand(self.iterations) < jump_prob, np.random.uniform(0.03, 0.05, self.iterations), 0)
            current_inf = current_inf + kappa * (inf_base - current_inf) * dt + dW + jumps
            inf_paths[:, yr] = np.clip(current_inf, -0.01, 0.15) 
            stagflation_shock = np.where(inf_paths[:, yr] > 0.05, -1.5 * (inf_paths[:, yr] - 0.05), 0)
            returns[:, yr, 1:] += stagflation_shock[:, None]
            
        return returns, inf_paths

    def run_mc(self, iwr, seed=None, roth_strategy=0, override_port=None):
        L = self.setup_covariance_matrix(override_port)
        returns, inf_paths = self.generate_stochastic_paths(L, seed=seed, override_port=override_port)
        cash_ret = float(self.inputs.get('cash_ret', 0.04))
        
        tsp = np.full(self.iterations, float(self.inputs['tsp_bal']))
        ira = np.full(self.iterations, float(self.inputs['ira_bal']))
        roth = np.full(self.iterations, float(self.inputs['roth_bal']))
        taxable = np.full(self.iterations, float(self.inputs['taxable_bal']))
        hsa = np.full(self.iterations, float(self.inputs['hsa_bal']))
        cash = np.full(self.iterations, float(self.inputs['cash_bal']))
        home_value = np.full(self.iterations, float(self.inputs.get('home_value', 0.0)))
        taxable_basis = np.full(self.iterations, float(self.inputs.get('taxable_basis', self.inputs['taxable_bal'])))
        
        p_base_salary = float(self.inputs.get('current_salary', 0))
        s_base_salary = float(self.inputs.get('s_current_salary', 0))
        p_max_tsp = self.inputs.get('p_max_tsp', False)
        p_tsp_c = float(self.inputs.get('p_tsp_contrib', 0))
        p_tax_c = float(self.inputs.get('p_taxable_contrib', 0))
        p_roth_c = float(self.inputs.get('p_roth_contrib', 0))
        p_cash_c = float(self.inputs.get('p_cash_contrib', 0))
        p_hsa_c = float(self.inputs.get('p_hsa_contrib', 0))

        s_max_tsp = self.inputs.get('s_max_tsp', False)
        s_tsp_c = float(self.inputs.get('s_tsp_contrib', 0))
        s_tax_c = float(self.inputs.get('s_taxable_contrib', 0))
        s_roth_c = float(self.inputs.get('s_roth_contrib', 0))
        s_cash_c = float(self.inputs.get('s_cash_contrib', 0))
        s_hsa_c = float(self.inputs.get('s_hsa_contrib', 0))

        p_base_pension = np.full(self.iterations, float(self.inputs.get('pension_est', 0)))
        s_base_pension = np.full(self.iterations, float(self.inputs.get('s_pension_est', 0)))
        
        p_surv_choice = self.inputs.get('survivor_benefit', 'No Survivor Benefit')
        if self.inputs.get('pension_type', 'FERS') == "FERS":
            p_pension_mult = 0.90 if p_surv_choice == 'Full Survivor Benefit' else (0.95 if p_surv_choice == 'Partial Survivor Benefit' else 1.0)
            p_fers_survivor_mult = 0.50 if p_surv_choice == 'Full Survivor Benefit' else (0.25 if p_surv_choice == 'Partial Survivor Benefit' else 0.0)
        else:
            p_pension_mult = 0.85 if "100%" in p_surv_choice else (0.925 if "50%" in p_surv_choice else (0.965 if "Present Value" in p_surv_choice else 1.0))
            p_fers_survivor_mult = 1.0 if "100%" in p_surv_choice else (0.50 if "50%" in p_surv_choice else 0.0)

        s_surv_choice = self.inputs.get('s_survivor_benefit', 'No Survivor Benefit')
        if self.inputs.get('s_pension_type', 'FERS') == "FERS":
            s_pension_mult = 0.90 if s_surv_choice == 'Full Survivor Benefit' else (0.95 if s_surv_choice == 'Partial Survivor Benefit' else 1.0)
            s_fers_survivor_mult = 0.50 if s_surv_choice == 'Full Survivor Benefit' else (0.25 if s_surv_choice == 'Partial Survivor Benefit' else 0.0)
        else:
            s_pension_mult = 0.85 if "100%" in s_surv_choice else (0.925 if "50%" in s_surv_choice else (0.965 if "Present Value" in s_surv_choice else 1.0))
            s_fers_survivor_mult = 1.0 if "100%" in s_surv_choice else (0.50 if "50%" in s_surv_choice else 0.0)

        p_mil_active = self.inputs.get('mil_active', False)
        p_base_mil_gross = np.zeros(self.iterations)
        p_base_va = np.zeros(self.iterations)
        p_crdp = False
        p_mil_sbp = False
        p_mil_start_age = self.inputs.get('mil_start_age', 60)
        
        if p_mil_active:
            if self.inputs['mil_discharge'] not in ["Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge"]:
                if self.inputs['mil_component'] in ["National Guard / Reserve", "Mixed (Active + Guard/Reserve)"]:
                    eq_years = self.inputs['mil_points'] / 360.0
                else:
                    eq_years = self.inputs['mil_years'] + (self.inputs['mil_months'] / 12.0) + (self.inputs['mil_days'] / 360.0)
                sys = self.inputs['mil_system']
                mult = eq_years * 0.02 if "BRS" in sys else ((eq_years * 0.025) - max(0, 30 - eq_years) * 0.01 if "REDUX" in sys else eq_years * 0.025)
                p_base_mil_gross = np.full(self.iterations, self.inputs['mil_pay_base'] * mult * 12)
                p_mil_sbp = "Full SBP" in self.inputs['mil_sbp']
                p_base_va = np.full(self.iterations, self.inputs['mil_va_pay'] * 12)
                p_crdp = self.inputs['mil_disability_rating'] in ["50% - 60%", "70% - 90%", "100%"] or self.inputs['mil_special_rating'] in ["TDIU (Unemployability)", "SMC (Special Monthly Comp)"]

        s_mil_active = self.inputs.get('s_mil_active', False)
        s_base_mil_gross = np.zeros(self.iterations)
        s_base_va = np.zeros(self.iterations)
        s_crdp = False
        s_mil_sbp = False
        s_mil_start_age = self.inputs.get('s_mil_start_age', 60)
        
        if s_mil_active:
            if self.inputs['s_mil_discharge'] not in ["Other Than Honorable (OTH) Discharge", "Bad Conduct Discharge (BCD)", "Dishonorable Discharge"]:
                if self.inputs['s_mil_component'] in ["National Guard / Reserve", "Mixed (Active + Guard/Reserve)"]:
                    s_eq_years = self.inputs['s_mil_points'] / 360.0
                else:
                    s_eq_years = self.inputs['s_mil_years'] + (self.inputs['s_mil_months'] / 12.0) + (self.inputs['s_mil_days'] / 360.0)
                s_sys = self.inputs['s_mil_system']
                s_mult = s_eq_years * 0.02 if "BRS" in s_sys else ((s_eq_years * 0.025) - max(0, 30 - s_eq_years) * 0.01 if "REDUX" in s_sys else s_eq_years * 0.025)
                s_base_mil_gross = np.full(self.iterations, self.inputs['s_mil_pay_base'] * s_mult * 12)
                s_mil_sbp = "Full SBP" in self.inputs['s_mil_sbp']
                s_base_va = np.full(self.iterations, self.inputs['s_mil_va_pay'] * 12)
                s_crdp = self.inputs['s_mil_disability_rating'] in ["50% - 60%", "70% - 90%", "100%"] or self.inputs['s_mil_special_rating'] in ["TDIU (Unemployability)", "SMC (Special Monthly Comp)"]

        p_ss_claim = self.inputs.get('ss_claim_age', 67)
        p_months_early, p_months_late = max(0, (67 - p_ss_claim) * 12), max(0, (p_ss_claim - 67) * 12)
        p_ss_modifier = 1.0 - ((min(36, p_months_early) * (5/900)) + (max(0, p_months_early - 36) * (5/1200))) + (p_months_late * (8/1200))
        p_base_ss = np.full(self.iterations, float(self.inputs.get('ss_fra', 0)))
        
        s_ss_claim = self.inputs.get('s_ss_claim_age', 67)
        s_months_early, s_months_late = max(0, (67 - s_ss_claim) * 12), max(0, (s_ss_claim - 67) * 12)
        s_ss_modifier = 1.0 - ((min(36, s_months_early) * (5/900)) + (max(0, s_months_early - 36) * (5/1200))) + (s_months_late * (8/1200))
        s_base_ss = np.full(self.iterations, float(self.inputs.get('s_ss_fra', 0)))
        
        history = {
            'total_bal': np.zeros((self.iterations, self.years)), 'total_bal_real': np.zeros((self.iterations, self.years)), 
            'cum_inf': np.zeros((self.iterations, self.years)), 'tsp_bal': np.zeros((self.iterations, self.years)),
            'ira_bal': np.zeros((self.iterations, self.years)), 'roth_bal': np.zeros((self.iterations, self.years)),
            'taxable_bal': np.zeros((self.iterations, self.years)), 'cash_bal': np.zeros((self.iterations, self.years)),
            'hsa_bal': np.zeros((self.iterations, self.years)), 'home_value': np.zeros((self.iterations, self.years)),
            'tsp_withdrawal': np.zeros((self.iterations, self.years)), 'ira_withdrawal': np.zeros((self.iterations, self.years)),
            'roth_withdrawal': np.zeros((self.iterations, self.years)), 'taxable_withdrawal': np.zeros((self.iterations, self.years)), 
            'cash_withdrawal': np.zeros((self.iterations, self.years)), 'rmds': np.zeros((self.iterations, self.years)),
            'extra_rmd': np.zeros((self.iterations, self.years)), 'taxes_fed': np.zeros((self.iterations, self.years)),
            'taxes_state': np.zeros((self.iterations, self.years)), 'taxable_income': np.zeros((self.iterations, self.years)), 
            'magi': np.zeros((self.iterations, self.years)), 'medicare_cost': np.zeros((self.iterations, self.years)),
            'health_cost': np.zeros((self.iterations, self.years)), 'mortgage_cost': np.zeros((self.iterations, self.years)),
            'additional_expenses': np.zeros((self.iterations, self.years)), 'net_spendable': np.zeros((self.iterations, self.years)),
            'salary_income': np.zeros((self.iterations, self.years)), 'port_return': np.zeros((self.iterations, self.years)),
            'real_return': np.zeros((self.iterations, self.years)), 'inflation': inf_paths,
            'constraint_active': np.zeros((self.iterations, self.years)), 'ss_income': np.zeros((self.iterations, self.years)),
            'pension_income': np.zeros((self.iterations, self.years)), 'va_income': np.zeros((self.iterations, self.years)),
            'roth_conversion': np.zeros((self.iterations, self.years)), 'roth_taxes_from_cash': np.zeros((self.iterations, self.years)), 
            'income_gap': np.zeros((self.iterations, self.years)), 'guaranteed_income': np.zeros((self.iterations, self.years)),
            'tax_paid': np.zeros((self.iterations, self.years))
        }

        age = self.inputs['current_age']
        spouse_age = self.inputs.get('spouse_age', age)
        primary_life_exp = self.inputs['life_expectancy']
        spouse_life_exp = self.inputs.get('spouse_life_exp', 95)
        current_year = datetime.datetime.now().year
        
        base_filing_status = self.inputs['filing_status']
        pay_taxes_from_cash = self.inputs.get('pay_taxes_from_cash', True)
        min_spending = float(self.inputs.get('min_spending', 0))
        max_spending = float(self.inputs.get('max_spending', 0))
        base_add_exp = float(self.inputs.get('additional_expenses', 0))
        user_max_bracket = float(self.inputs.get('max_tax_bracket', '0.24'))
        base_health_premium = float(self.inputs.get('health_cost', 0))
        base_oop_cost = float(self.inputs.get('oop_cost', 0))
        health_plan = self.inputs.get('health_plan', "None/Self-Insure")
        mortgage_pmt = float(self.inputs.get('mortgage_pmt', 0))
        mortgage_yrs = int(self.inputs.get('mortgage_yrs', 0))
        
        has_40_quarters = self.inputs.get('has_40_quarters', True)
        has_dependent_children = self.inputs.get('has_dependent_children', False)
        wants_dental_vision = self.inputs.get('wants_dental_vision', False)

        state_str = self.inputs.get('state', '').strip().upper()
        county_str = self.inputs.get('county', '').strip().upper()
        state_tax_rate = STATE_TAX_RATES.get(state_str, 0.045) if state_str not in RETIREMENT_TAX_FREE_STATES else 0.0
        local_tax_rate = 0.025 if county_str != "" and state_str in ["MD", "IN", "PA", "OH", "NY"] else (0.010 if county_str != "" else 0.0)
        combined_state_local_rate = state_tax_rate + local_tax_rate
        cum_inf = np.ones(self.iterations)
        
        target_lifestyle = np.zeros(self.iterations)
        ref_draw = np.zeros(self.iterations)

        for yr in range(self.years):
            age += 1
            spouse_age += 1
            current_year += 1
            
            p_work_frac = 1.0 if current_year < self.ret_year else (self.ret_month / 12.0 if current_year == self.ret_year else 0.0)
            p_ret_frac = 1.0 - p_work_frac
            
            s_work_frac = 1.0 if current_year < self.s_ret_year else (self.s_ret_month / 12.0 if current_year == self.s_ret_year else 0.0)
            s_ret_frac = 1.0 - s_work_frac
            
            primary_alive = age <= primary_life_exp
            spouse_alive = spouse_age <= spouse_life_exp if base_filing_status == 'MFJ' else False
            
            if base_filing_status == 'MFJ':
                if primary_alive and spouse_alive:
                    current_filing_status, moop_idx = 'MFJ', 1
                elif not primary_alive and spouse_alive:
                    current_filing_status, moop_idx = 'Single', 0
                elif primary_alive and not spouse_alive:
                    current_filing_status, moop_idx = 'Single', 0
                else:
                    current_filing_status, moop_idx = 'Single', 0
            else:
                current_filing_status, moop_idx = 'Single', 0
                
            base_deduction = STD_DED_MFJ if current_filing_status == 'MFJ' else STD_DED_SINGLE
            extra_ded_primary = EXTRA_DED_65_SINGLE if current_filing_status == 'Single' and age >= 65 else 0
            extra_ded_primary_mfj = EXTRA_DED_65_MFJ_PER_PERSON if current_filing_status == 'MFJ' and age >= 65 else 0
            extra_ded_spouse = EXTRA_DED_65_MFJ_PER_PERSON if current_filing_status == 'MFJ' and spouse_age >= 65 else 0
            
            deduction = base_deduction + extra_ded_primary + extra_ded_primary_mfj + extra_ded_spouse
            
            brackets = TAX_BRACKETS_MFJ if current_filing_status == 'MFJ' else TAX_BRACKETS_SINGLE
            irmaa_brackets = IRMAA_BRACKETS_MFJ if current_filing_status == 'MFJ' else IRMAA_BRACKETS_SINGLE
            ltcg_brackets = LTCG_BRACKETS_MFJ if current_filing_status == 'MFJ' else LTCG_BRACKETS_SINGLE
            niit_threshold = NIIT_THRESHOLD_MFJ if current_filing_status == 'MFJ' else NIIT_THRESHOLD_SINGLE
            base_moop = MOOP_LIMITS.get(health_plan, (999999, 999999))[moop_idx]

            if yr > 0: cum_inf *= (1 + np.maximum(0, inf_paths[:, yr]))
            history['cum_inf'][:, yr] = cum_inf
            
            limit_max_pct = np.full(self.iterations, np.inf)
            for i in range(len(brackets)):
                if np.isclose(brackets[i][1], user_max_bracket, atol=0.01):
                    limit_max_pct = brackets[i][0] * cum_inf
                    break
            
            home_value *= 1.03
            
            yr_salary = np.zeros(self.iterations)
            yr_pension = np.zeros(self.iterations)
            yr_va = np.zeros(self.iterations)
            yr_ss = np.zeros(self.iterations)
            yr_savings = np.zeros(self.iterations)

            inf_floor = np.maximum(0, inf_paths[:, yr])
            
            if self.inputs.get('pension_type', 'FERS') == 'FERS':
                p_cola = np.where(inf_floor <= 0.02, inf_floor, np.where(inf_floor <= 0.03, 0.02, inf_floor - 0.01))
                p_cola = np.where(age >= 62, p_cola, 0.0) 
            else:
                p_cola = np.minimum(inf_floor, 0.03)

            if self.inputs.get('s_pension_type', 'FERS') == 'FERS':
                s_cola = np.where(inf_floor <= 0.02, inf_floor, np.where(inf_floor <= 0.03, 0.02, inf_floor - 0.01))
                s_cola = np.where(spouse_age >= 62, s_cola, 0.0)
            else:
                s_cola = np.minimum(inf_floor, 0.03)
                
            p_base_pension *= (1 + p_cola)
            s_base_pension *= (1 + s_cola)
            
            p_base_mil_gross *= (1 + inf_floor)
            s_base_mil_gross *= (1 + inf_floor)
            p_base_va *= (1 + inf_floor)
            s_base_va *= (1 + inf_floor)
            p_base_ss *= (1 + inf_floor)
            s_base_ss *= (1 + inf_floor)

            p_salary_inf = p_base_salary * cum_inf
            s_salary_inf = s_base_salary * cum_inf
            ss_haircut = 0.79 if current_year >= 2035 else 1.0
            
            p_active_ss = p_base_ss * p_ss_modifier * ss_haircut
            
            if primary_alive:
                if p_work_frac > 0:
                    yr_salary += p_salary_inf * p_work_frac
                    if p_max_tsp:
                        if age < 50: p_tsp_c_yr = 24500 * cum_inf
                        elif 50 <= age <= 59: p_tsp_c_yr = 32500 * cum_inf
                        elif 60 <= age <= 63: p_tsp_c_yr = 35750 * cum_inf
                        else: p_tsp_c_yr = 32500 * cum_inf
                    else: p_tsp_c_yr = p_tsp_c * cum_inf
                    tsp += p_tsp_c_yr * p_work_frac
                    taxable += p_tax_c * cum_inf * p_work_frac
                    taxable_basis += p_tax_c * cum_inf * p_work_frac
                    roth += p_roth_c * cum_inf * p_work_frac
                    cash += p_cash_c * cum_inf * p_work_frac
                    hsa += p_hsa_c * cum_inf * p_work_frac

                if p_ret_frac > 0:
                    yr_pension += p_base_pension * p_pension_mult * p_ret_frac
                    
                if age >= p_mil_start_age:
                    offset = p_base_va if not p_crdp else np.zeros(self.iterations)
                    taxable_mil = np.maximum(0, p_base_mil_gross - offset)
                    sbp_cost = p_base_mil_gross * 0.065 if (p_mil_sbp and spouse_alive) else np.zeros(self.iterations)
                    yr_pension += np.maximum(0, taxable_mil - sbp_cost)
                    yr_va += p_base_va
            else:
                if spouse_alive:
                    yr_pension += p_base_pension * p_fers_survivor_mult
                    if p_mil_sbp: yr_pension += p_base_mil_gross * 0.55

            s_active_ss = s_base_ss * s_ss_modifier * ss_haircut
            if spouse_alive:
                if s_work_frac > 0:
                    yr_salary += s_salary_inf * s_work_frac
                    if s_max_tsp:
                        if spouse_age < 50: s_tsp_c_yr = 24500 * cum_inf
                        elif 50 <= spouse_age <= 59: s_tsp_c_yr = 32500 * cum_inf
                        elif 60 <= spouse_age <= 63: s_tsp_c_yr = 35750 * cum_inf
                        else: s_tsp_c_yr = 32500 * cum_inf
                    else: s_tsp_c_yr = s_tsp_c * cum_inf
                    tsp += s_tsp_c_yr * s_work_frac
                    taxable += s_tax_c * cum_inf * s_work_frac
                    taxable_basis += s_tax_c * cum_inf * s_work_frac
                    roth += s_roth_c * cum_inf * s_work_frac
                    cash += s_cash_c * cum_inf * s_work_frac
                    hsa += s_hsa_c * cum_inf * s_work_frac

                if s_ret_frac > 0:
                    yr_pension += s_base_pension * s_pension_mult * s_ret_frac
                    
                if spouse_age >= s_mil_start_age:
                    offset = s_base_va if not s_crdp else np.zeros(self.iterations)
                    taxable_mil = np.maximum(0, s_base_mil_gross - offset)
                    sbp_cost = s_base_mil_gross * 0.065 if (s_mil_sbp and primary_alive) else np.zeros(self.iterations)
                    yr_pension += np.maximum(0, taxable_mil - sbp_cost)
                    yr_va += s_base_va
            else:
                if primary_alive:
                    yr_pension += s_base_pension * s_fers_survivor_mult
                    if s_mil_sbp: yr_pension += s_base_mil_gross * 0.55

            p_ss_val = p_active_ss if age >= p_ss_claim else np.zeros(self.iterations)
            s_ss_val = s_active_ss if spouse_age >= s_ss_claim else np.zeros(self.iterations)

            if primary_alive and spouse_alive:
                yr_ss = p_ss_val + s_ss_val
            elif primary_alive and not spouse_alive:
                s_widow_limit = np.where(s_ss_claim < 67, np.maximum(s_active_ss, s_base_ss * 0.825 * ss_haircut), s_active_ss)
                p_surv_penalty = np.clip(1.0 - ((67 - age) * (0.285 / 7.0)), 0.715, 1.0)
                inherited_ss = s_widow_limit * p_surv_penalty
                yr_ss = np.maximum(p_ss_val, inherited_ss)
            elif not primary_alive and spouse_alive:
                p_widow_limit = np.where(p_ss_claim < 67, np.maximum(p_active_ss, p_base_ss * 0.825 * ss_haircut), p_active_ss)
                s_surv_penalty = np.clip(1.0 - ((67 - spouse_age) * (0.285 / 7.0)), 0.715, 1.0)
                inherited_ss = p_widow_limit * s_surv_penalty
                yr_ss = np.maximum(s_ss_val, inherited_ss)
            else:
                yr_ss = np.zeros(self.iterations)

            history['salary_income'][:, yr] = yr_salary
            history['pension_income'][:, yr] = yr_pension
            history['va_income'][:, yr] = yr_va
            history['ss_income'][:, yr] = yr_ss
            
            total_guaranteed = yr_pension + yr_va + yr_ss + yr_salary
            history['guaranteed_income'][:, yr] = total_guaranteed
            
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
            
            history['total_bal'][:, yr] = current_total_port + home_value
            history['total_bal_real'][:, yr] = (current_total_port + home_value) / cum_inf
            
            w_needed = np.zeros(self.iterations)
            constraint_flag = np.zeros(self.iterations)
            
            if current_year == self.ret_year or (yr == 0 and current_year > self.ret_year):
                total_guaranteed_full = (p_base_pension * p_pension_mult) + (p_base_va) + p_ss_val
                if spouse_alive and current_year >= self.s_ret_year:
                    total_guaranteed_full += (s_base_pension * s_pension_mult) + s_base_va + s_ss_val
                target_lifestyle = (current_total_port * iwr) + total_guaranteed_full
                ref_draw = current_total_port * iwr 
                
            if current_year >= self.ret_year:
                if yr > 0 and current_year > self.ret_year:
                    port_ret_prev = history['port_return'][:, yr-1]
                    inf_adj = np.where(port_ret_prev <= -0.10, 0, inf_paths[:, yr])
                    target_lifestyle *= (1 + inf_adj)
                    
                    w_needed = np.maximum(0, target_lifestyle - total_guaranteed)
                    ref_gap = ref_draw * cum_inf
                    
                    prosperity = w_needed < (ref_gap * 0.8)
                    target_lifestyle = np.where(prosperity, target_lifestyle * 1.05, target_lifestyle)
                    
                    preservation = w_needed > (ref_gap * 1.2)
                    target_lifestyle = np.where(preservation, target_lifestyle * 0.90, target_lifestyle)
                    constraint_flag = np.where(preservation, 1, constraint_flag)
                    
                    sweep_mask = port_ret_prev > 0.08
                    target_cash = w_needed * 2.0
                    cash_deficit = np.maximum(0, target_cash - cash)
                    excess_gains = np.maximum(0, port_ret_prev - 0.08) * prev_total_port
                    sweep_allowance = excess_gains * 0.50
                    actual_sweep = np.where(sweep_mask, np.minimum(sweep_allowance, cash_deficit), 0)
                    
                    sweep_from_taxable = np.minimum(actual_sweep, taxable)
                    taxable -= sweep_from_taxable
                    sweep_from_tsp = np.minimum(actual_sweep - sweep_from_taxable, tsp)
                    tsp -= sweep_from_tsp
                    cash += (sweep_from_taxable + sweep_from_tsp)

                inflated_min_spend = min_spending * cum_inf
                inflated_max_spend = max_spending * cum_inf if max_spending > 0 else np.full(self.iterations, np.inf)
                
                target_lifestyle = np.clip(target_lifestyle, inflated_min_spend + total_guaranteed, inflated_max_spend + total_guaranteed)
                w_needed = np.maximum(0, target_lifestyle - total_guaranteed)

            history['constraint_active'][:, yr] = constraint_flag
            history['income_gap'][:, yr] = w_needed
            
            rmd_divisor = IRS_RMD_DIVISORS.get(age, 1.9 if age > 120 else 0.0)
            s_rmd_divisor = IRS_RMD_DIVISORS.get(spouse_age, 1.9 if spouse_age > 120 else 0.0)
            
            rmd_tsp = tsp * (1.0 / rmd_divisor if rmd_divisor > 0 else 0.0)
            rmd_ira = ira * (1.0 / rmd_divisor if rmd_divisor > 0 else 0.0)
            rmds = rmd_tsp + rmd_ira
            history['rmds'][:, yr] = rmds
            
            tsp -= rmd_tsp
            ira -= rmd_ira
            
            w_remaining = np.maximum(0, w_needed - rmds)
            excess_rmd = np.maximum(0, rmds - w_needed)
            history['extra_rmd'][:, yr] = excess_rmd
            
            w_tsp, w_ira, w_cash, w_taxable, w_roth = np.zeros(self.iterations), np.zeros(self.iterations), np.zeros(self.iterations), np.zeros(self.iterations), np.zeros(self.iterations)
            
            if current_year >= self.ret_year:
                downturn_state = history['port_return'][:, yr-1] <= -0.10 if yr > 0 else np.zeros(self.iterations, dtype=bool)
                normal_state = ~downturn_state
                
                pi_base_pre = rmds + yr_pension + yr_salary + (0.5 * yr_ss)
                calc_ss_base_pre = 0.5 * np.clip(pi_base_pre - (32000 if current_filing_status == 'MFJ' else 25000), 0, (44000 if current_filing_status == 'MFJ' else 34000) - (32000 if current_filing_status == 'MFJ' else 25000)) + 0.85 * np.maximum(0, pi_base_pre - (44000 if current_filing_status == 'MFJ' else 34000))
                taxable_ss_base_pre = np.minimum(0.85 * yr_ss, calc_ss_base_pre)
                
                base_taxable_pre = np.maximum(0, rmds + yr_pension + taxable_ss_base_pre + yr_salary - (deduction * cum_inf))
                bracket_space = np.maximum(0, limit_max_pct - base_taxable_pre)
                
                pull_tsp_1 = np.where(normal_state, np.minimum(w_remaining, np.minimum(tsp, bracket_space)), 0)
                w_tsp += pull_tsp_1
                w_remaining -= pull_tsp_1
                tsp -= pull_tsp_1
                bracket_space -= pull_tsp_1
                
                pull_ira_1 = np.where(normal_state, np.minimum(w_remaining, np.minimum(ira, bracket_space)), 0)
                w_ira += pull_ira_1
                w_remaining -= pull_ira_1
                ira -= pull_ira_1
                
                pull_cash_1 = np.where(downturn_state, np.minimum(w_remaining, cash), 0)
                w_cash += pull_cash_1
                w_remaining -= pull_cash_1
                cash -= pull_cash_1
                
                pull_tax_1 = np.minimum(w_remaining, taxable)
                w_taxable += pull_tax_1
                w_remaining -= pull_tax_1
                taxable -= pull_tax_1
                
                pull_cash_2 = np.where(normal_state, np.minimum(w_remaining, cash), 0)
                w_cash += pull_cash_2
                w_remaining -= pull_cash_2
                cash -= pull_cash_2
                
                pull_roth_1 = np.minimum(w_remaining, roth)
                w_roth += pull_roth_1
                w_remaining -= pull_roth_1
                roth -= pull_roth_1
                
                pull_tsp_2 = np.minimum(w_remaining, tsp)
                w_tsp += pull_tsp_2
                w_remaining -= pull_tsp_2
                tsp -= pull_tsp_2
                
                pull_ira_2 = np.minimum(w_remaining, ira)
                w_ira += pull_ira_2
                w_remaining -= pull_ira_2
                ira -= pull_ira_2
            
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
            
            t1 = 32000 if current_filing_status == 'MFJ' else 25000
            t2 = 44000 if current_filing_status == 'MFJ' else 34000
            
            pi_base = rmds + w_tsp + w_ira + yr_pension + yr_salary + realized_gains + (0.5 * yr_ss)
            calc_ss_base = 0.5 * np.clip(pi_base - t1, 0, t2 - t1) + 0.85 * np.maximum(0, pi_base - t2)
            taxable_ss_base = np.minimum(0.85 * yr_ss, calc_ss_base)
            
            gross_income = rmds + w_tsp + w_ira + yr_pension + taxable_ss_base + yr_salary
            magi = gross_income.copy() 
            taxable_income = np.maximum(0, gross_income - (deduction * cum_inf)) 
            
            base_tax_fed = np.zeros(self.iterations)
            for i in range(len(brackets)):
                prev_limit = (brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                limit = brackets[i][0] * cum_inf
                taxable_in_bracket = np.clip(taxable_income, prev_limit, limit) - prev_limit
                base_tax_fed += np.maximum(0, taxable_in_bracket) * brackets[i][1]
                
            ltcg_tax = np.zeros(self.iterations)
            for i in range(len(ltcg_brackets)):
                prev_limit = (ltcg_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                limit = ltcg_brackets[i][0] * cum_inf
                
                gains_start = taxable_income
                gains_end = taxable_income + realized_gains
                overlap_start = np.maximum(gains_start, prev_limit)
                overlap_end = np.minimum(gains_end, limit)
                taxable_gains_in_bracket = np.maximum(0, overlap_end - overlap_start)
                
                ltcg_tax += taxable_gains_in_bracket * ltcg_brackets[i][1]
                
            niit_tax = np.where(magi > (niit_threshold * cum_inf), realized_gains * 0.038, 0.0)
            base_tax_fed += (ltcg_tax + niit_tax)
            
            if state_str in RETIREMENT_TAX_FREE_STATES:
                state_taxable_base = np.maximum(0, taxable_income - rmds - w_tsp - w_ira - yr_pension - taxable_ss_base)
            else:
                if state_str not in STATES_TAXING_SS:
                    state_taxable_base = np.maximum(0, taxable_income - taxable_ss_base)
                else:
                    state_taxable_base = taxable_income
                    
            if age >= 65 or spouse_age >= 65:
                if current_filing_status == 'MFJ':
                    state_exclusion = STATE_EXCLUSIONS_65_MFJ.get(state_str, 0.0) * cum_inf
                else:
                    state_exclusion = STATE_EXCLUSIONS_65_SINGLE.get(state_str, 0.0) * cum_inf
                
                ret_income = yr_pension + w_tsp + w_ira
                allowed_exclusion = np.minimum(state_exclusion, ret_income)
                state_taxable_base = np.maximum(0, state_taxable_base - allowed_exclusion)
                    
            base_tax_state_local = state_taxable_base * combined_state_local_rate
            
            total_tax_fed = base_tax_fed.copy()
            total_tax_state = base_tax_state_local.copy()
            conv_amt = np.zeros(self.iterations)
            w_tax_cash_roth = np.zeros(self.iterations)
            final_taxable_income = taxable_income.copy()
            final_magi = magi.copy()
            
            if roth_strategy > 0 and current_year >= self.ret_year and age < 75:
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
                    space = np.maximum(0, (irmaa_brackets[0][0] * cum_inf) - magi - 1)
                elif roth_strategy == 3:
                    space = np.maximum(0, (irmaa_brackets[1][0] * cum_inf) - magi - 1)
                
                space = np.minimum(space, np.maximum(0, limit_max_pct - taxable_income - 1))
                conv_from_ira = np.minimum(space, ira)
                ira -= conv_from_ira
                conv_from_tsp = np.minimum(space - conv_from_ira, tsp)
                tsp -= conv_from_tsp
                conv_amt = conv_from_ira + conv_from_tsp
                
                pi_conv = pi_base + conv_amt
                calc_ss_conv = 0.5 * np.clip(pi_conv - t1, 0, t2 - t1) + 0.85 * np.maximum(0, pi_conv - t2)
                taxable_ss_conv = np.minimum(0.85 * yr_ss, calc_ss_conv)
                
                final_gross_income = rmds + w_tsp + w_ira + yr_pension + taxable_ss_conv + yr_salary + conv_amt
                final_taxable_income = np.maximum(0, final_gross_income - (deduction * cum_inf))
                final_magi = final_gross_income.copy()
                
                new_tax_fed = np.zeros(self.iterations)
                for i in range(len(brackets)):
                    prev_limit = (brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                    limit = brackets[i][0] * cum_inf
                    taxable_in_bracket = np.clip(final_taxable_income, prev_limit, limit) - prev_limit
                    new_tax_fed += np.maximum(0, taxable_in_bracket) * brackets[i][1]
                
                new_ltcg_tax = np.zeros(self.iterations)
                for i in range(len(ltcg_brackets)):
                    prev_limit = (ltcg_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                    limit = ltcg_brackets[i][0] * cum_inf
                    
                    gains_start = final_taxable_income
                    gains_end = final_taxable_income + realized_gains
                    overlap_start = np.maximum(gains_start, prev_limit)
                    overlap_end = np.minimum(gains_end, limit)
                    taxable_gains_in_bracket = np.maximum(0, overlap_end - overlap_start)
                    
                    new_ltcg_tax += taxable_gains_in_bracket * ltcg_brackets[i][1]
                    
                new_niit_tax = np.where(final_magi > (niit_threshold * cum_inf), realized_gains * 0.038, 0.0)
                extra_tax_fed = (new_tax_fed + new_ltcg_tax + new_niit_tax) - base_tax_fed
                
                if state_str in RETIREMENT_TAX_FREE_STATES:
                    extra_state_taxable = np.zeros(self.iterations)
                else:
                    if state_str not in STATES_TAXING_SS:
                        extra_state_taxable = final_taxable_income - taxable_ss_conv - (taxable_income - taxable_ss_base)
                    else:
                        extra_state_taxable = final_taxable_income - taxable_income
                
                if age >= 65 or spouse_age >= 65:
                    if current_filing_status == 'MFJ':
                        state_exclusion = STATE_EXCLUSIONS_65_MFJ.get(state_str, 0.0) * cum_inf
                    else:
                        state_exclusion = STATE_EXCLUSIONS_65_SINGLE.get(state_str, 0.0) * cum_inf
                    
                    ret_income = yr_pension + w_tsp + w_ira + conv_amt
                    allowed_exclusion = np.minimum(state_exclusion, ret_income)
                    base_ret_income = yr_pension + w_tsp + w_ira
                    base_allowed_exclusion = np.minimum(state_exclusion, base_ret_income)
                    
                    extra_exclusion_benefit = allowed_exclusion - base_allowed_exclusion
                    extra_state_taxable = np.maximum(0, extra_state_taxable - extra_exclusion_benefit)

                extra_tax_state = extra_state_taxable * combined_state_local_rate
                extra_tax_total = extra_tax_fed + extra_tax_state
                
if pay_taxes_from_cash:
                    w_tax_cash_roth = np.minimum(cash, extra_tax_total)
                    cash -= w_tax_cash_roth
                    rem_tax = extra_tax_total - w_tax_cash_roth
                    w_tax_taxable = np.minimum(taxable, rem_tax)
                    taxable -= w_tax_taxable
                    gains_ratio_tax = np.maximum(0, 1.0 - (taxable_basis / np.maximum(taxable + w_tax_taxable, 1.0)))
                    taxable_basis -= (w_tax_taxable - (w_tax_taxable * gains_ratio_tax))
                    net_to_roth = np.maximum(0, conv_amt - (rem_tax - w_tax_taxable))
                else:
                    net_to_roth = np.maximum(0, conv_amt - extra_tax_total)
                    
                roth += net_to_roth
                total_tax_fed = new_tax_fed + new_ltcg_tax + new_niit_tax
                total_tax_state = base_tax_state_local + extra_tax_state
            
            history['roth_conversion'][:, yr] = conv_amt
            history['taxes_fed'][:, yr] = total_tax_fed
            history['taxes_state'][:, yr] = total_tax_state
            history['tax_paid'][:, yr] = total_tax_fed + total_tax_state
            history['taxable_income'][:, yr] = final_taxable_income
            history['magi'][:, yr] = final_magi
            history['roth_taxes_from_cash'][:, yr] = w_tax_cash_roth 
            
# Healthcare Base Costs
            base_p_health = float(self.inputs.get('p_health_cost', 0))
            base_s_health = float(self.inputs.get('s_health_cost', 0))
            MEDICARE_PART_A_BASE = 505.0 # Fixed missing variable

            age_morbidity = 1.025 ** max(0, age - self.inputs['current_age'])
            med_cpi_cum = np.prod(1 + (np.maximum(0, inf_paths[:, :yr+1]) * 1.5), axis=1) if yr > 0 else np.ones(self.iterations)
            raw_oop = base_oop_cost * med_cpi_cum * age_morbidity
            inflated_oop = np.minimum(raw_oop, base_moop * cum_inf)
            
            p_med_cost = np.zeros(self.iterations)
            s_med_cost = np.zeros(self.iterations)
            p_health_prem = base_p_health * cum_inf
            s_health_prem = base_s_health * cum_inf

            # ==========================================
            # PRIMARY MEDICARE vs ACA LOGIC
            # ==========================================
            intent_delay = 3 if self.inputs.get('intent_to_work_40_quarters', False) else 0
            primary_medicare_age = 65 + intent_delay

            if age >= primary_medicare_age:
                if health_plan in ["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"]:
                    if has_40_quarters or intent_delay > 0:
                        # RULE 1: Transition to Medicare (Or finally hit 40 quarters via intent)
                        p_med_cost += MEDICARE_PART_B_BASE * cum_inf
                        
                        # RULE 2: Apply 10% Part B Penalty per year delayed
                        if intent_delay > 0:
                            p_med_cost *= (1.0 + (0.10 * intent_delay))
                            
                        for i in range(len(irmaa_brackets)):
                            prev_limit = (irmaa_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                            p_med_cost = np.where(final_magi > prev_limit, p_med_cost + (irmaa_brackets[i][1] * cum_inf), p_med_cost)
                        
                        p_health_prem = np.zeros(self.iterations)
                        if wants_dental_vision: p_health_prem += 600 * cum_inf
                        if has_dependent_children: p_health_prem += (base_p_health * 0.5) * cum_inf
                    else:
                        # RULE 1b: No 40 Quarters -> Cost Benefit Analysis
                        aca_sub_cost = np.minimum(base_p_health * cum_inf, final_magi * 0.085)
                        med_tot_cost = (MEDICARE_PART_A_BASE + MEDICARE_PART_B_BASE) * cum_inf
                        
                        for i in range(len(irmaa_brackets)):
                            prev_limit = (irmaa_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                            med_tot_cost = np.where(final_magi > prev_limit, med_tot_cost + (irmaa_brackets[i][1] * cum_inf), med_tot_cost)

                        stay_on_aca = aca_sub_cost < med_tot_cost
                        p_med_cost = np.where(stay_on_aca, 0, med_tot_cost)
                        
                        # Fix Bug: Dependent children accounted for if switching to Medicare
                        dep_cost = (base_p_health * 0.5) * cum_inf if has_dependent_children else 0
                        den_cost = 600 * cum_inf if wants_dental_vision else 0
                        p_health_prem = np.where(stay_on_aca, aca_sub_cost, den_cost + dep_cost)
                
                elif "FEHB" not in health_plan and "TRICARE" not in health_plan:
                    # Standard Private Insurance transition
                    p_med_cost += MEDICARE_PART_B_BASE * cum_inf
                    for i in range(len(irmaa_brackets)):
                        prev_limit = (irmaa_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                        p_med_cost = np.where(final_magi > prev_limit, p_med_cost + (irmaa_brackets[i][1] * cum_inf), p_med_cost)

            # ==========================================
            # SPOUSE MEDICARE LOGIC (Independent Age Trigger)
            # ==========================================
            s_health_plan = self.inputs.get('s_health_plan', "None/Self-Insure")
            if spouse_alive and spouse_age >= 65:
                if s_health_plan in ["None/Self-Insure", "Affordable Care Act", "Spouse's Insurance"]:
                    if has_40_quarters:
                        s_med_cost += MEDICARE_PART_B_BASE * cum_inf
                        for i in range(len(irmaa_brackets)):
                            prev_limit = (irmaa_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                            s_med_cost = np.where(final_magi > prev_limit, s_med_cost + (irmaa_brackets[i][1] * cum_inf), s_med_cost)
                        
                        s_health_prem = np.zeros(self.iterations)
                        if wants_dental_vision: s_health_prem += 600 * cum_inf
                
                elif "FEHB" not in s_health_plan and "TRICARE" not in s_health_plan:
                    s_med_cost += MEDICARE_PART_B_BASE * cum_inf
                    for i in range(len(irmaa_brackets)):
                        prev_limit = (irmaa_brackets[i-1][0] * cum_inf) if i > 0 else np.zeros(self.iterations)
                        s_med_cost = np.where(final_magi > prev_limit, s_med_cost + (irmaa_brackets[i][1] * cum_inf), s_med_cost)

            history['medicare_cost'][:, yr] = p_med_cost + s_med_cost

            w_hsa = np.minimum(hsa, inflated_oop)
            hsa -= w_hsa
            history['health_cost'][:, yr] = (p_health_prem + s_health_prem) + (inflated_oop - w_hsa)
            
            current_mortgage = np.full(self.iterations, mortgage_pmt if yr < mortgage_yrs else 0.0)
            history['mortgage_cost'][:, yr] = current_mortgage
            
            if current_year >= self.ret_year:
                years_in_ret = current_year - self.ret_year
                smile_mult = 1.0 - (0.015 * years_in_ret) + (0.0005 * (years_in_ret ** 2))
                current_add_exp = base_add_exp * cum_inf * smile_mult
            else:
                current_add_exp = np.zeros(self.iterations)
            history['additional_expenses'][:, yr] = current_add_exp
            
            history['tsp_bal'][:, yr] = tsp
            history['ira_bal'][:, yr] = ira
            history['roth_bal'][:, yr] = roth
            history['taxable_bal'][:, yr] = taxable
            history['cash_bal'][:, yr] = cash
            history['hsa_bal'][:, yr] = hsa
            
            if current_year >= self.ret_year:
                total_deductions = total_tax_fed + total_tax_state + medicare_cost + history['health_cost'][:, yr] + current_mortgage + current_add_exp
                history['net_spendable'][:, yr] = actual_portfolio_withdrawal + yr_pension + yr_va + yr_ss + yr_salary - total_deductions
            else:
                history['net_spendable'][:, yr] = 0.0
            
        return history

    def objective_function(self, iwr_test):
        history = self.run_mc(iwr_test, seed=42, roth_strategy=0)
        median_real_path = np.median(history['total_bal_real'], axis=0)
        target_floor = self.inputs.get('target_floor', 0.0)
        terminal_wealth = median_real_path[-1]
        
        return terminal_wealth - target_floor

    def optimize_iwr(self):
        try:
            return optimize.brentq(self.objective_function, a=0.001, b=0.40, xtol=1e-4, maxiter=40)
        except Exception:
            return 0.04
            
    def analyze_portfolios(self, opt_iwr, roth_strategy=0):
        results = {}
        hist_custom = self.run_mc(opt_iwr, seed=42, roth_strategy=roth_strategy, override_port=None)
        results["Your Custom Mix"] = {'wealth': np.median(hist_custom['total_bal_real'][:, -1]), 'cut_prob': np.mean(np.any(hist_custom['constraint_active'] == 1, axis=1)) * 100}
        del hist_custom
        gc.collect()
        
        for port in ["Conservative (20% Stock / 80% Bond)", "Moderate (60% Stock / 40% Bond)", "Aggressive (100% Stock)"]:
            hist = self.run_mc(opt_iwr, seed=42, roth_strategy=roth_strategy, override_port=port)
            results[port] = {'wealth': np.median(hist['total_bal_real'][:, -1]), 'cut_prob': np.mean(np.any(hist['constraint_active'] == 1, axis=1)) * 100}
            del hist
            gc.collect()
            
        return results

    def analyze_roth_strategies(self, opt_iwr):
        user_max = float(self.inputs.get("max_tax_bracket", 0.24)) * 100
        strats = [(0, 'Baseline (None)'), (1, 'Fill Current Bracket (IRMAA Protected)'), (2, 'Target IRMAA Tier 1'), (3, 'Target IRMAA Tier 2'), (4, f'Max User Bracket Fill ({user_max:.0f}%)')]
        
        results, best_wealth, winner_name, winner_hist = {}, -np.inf, 'Baseline (None)', None
        
        for s_idx, s_name in strats:
            hist = self.run_mc(opt_iwr, seed=42, roth_strategy=s_idx)
            wealth = np.median(hist['total_bal_real'][:, -1])
            results[s_name] = {'wealth': wealth, 'taxes': np.sum(np.median(hist['taxes_fed'], axis=0)), 'rmds': np.sum(np.median(hist['rmds'], axis=0)), 'tax_path': np.median(hist['taxes_fed'], axis=0), 'conv_path': np.median(hist['roth_conversion'], axis=0), 'taxable_inc_path': np.median(hist['taxable_income'], axis=0)}
            
            if wealth > best_wealth:
                best_wealth, winner_name, winner_hist = wealth, s_name, hist 
            else:
                del hist 
                gc.collect()
                
        return results, winner_name, winner_hist