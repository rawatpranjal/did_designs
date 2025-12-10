import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_smoking

# Copied from 07_synthetic_did/main.py
def prepare_sdid_data(df):
    TREATMENT_YEAR = 1989
    TREATED_STATE = 'California'
    wide = df.pivot(index='year', columns='state', values='cigsale')
    pre_mask = wide.index < TREATMENT_YEAR
    post_mask = wide.index >= TREATMENT_YEAR
    control_states = [s for s in wide.columns if s != TREATED_STATE]

    Y_pre_treat = wide.loc[pre_mask, TREATED_STATE].values
    Y_post_treat = wide.loc[post_mask, TREATED_STATE].values
    Y_pre_ctrl = wide.loc[pre_mask, control_states].values
    Y_post_ctrl = wide.loc[post_mask, control_states].values

    return wide, Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, control_states

def solve_unit_weights_current(Y_pre_treat, Y_pre_ctrl, N_tr=1, T_post=1, zeta=None):
    """Current implementation in main.py"""
    T_pre, N_co = Y_pre_ctrl.shape

    if zeta is None:
        diff_ctrl = np.diff(Y_pre_ctrl, axis=0)
        sigma_sq = np.var(diff_ctrl)
        # Current incorrect formula
        zeta = (N_tr * T_post)**0.25 * sigma_sq

    omega_init = np.ones(N_co) / N_co

    def objective(omega):
        synth = Y_pre_ctrl @ omega
        alpha = np.mean(Y_pre_treat - synth)
        residuals = Y_pre_treat - (alpha + synth)
        # Current objective
        loss = np.sum(residuals ** 2) + (zeta * T_pre * np.sum(omega ** 2))
        return loss

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, None) for _ in range(N_co)]

    result = minimize(objective, omega_init, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-10})

    omega = result.x
    synth = Y_pre_ctrl @ omega
    alpha = np.mean(Y_pre_treat - synth)

    return omega, alpha

def solve_unit_weights_corrected(Y_pre_treat, Y_pre_ctrl, N_tr=1, T_post=1):
    """Corrected implementation based on Arkhangelsky et al. (2021)"""
    T_pre, N_co = Y_pre_ctrl.shape

    diff_ctrl = np.diff(Y_pre_ctrl, axis=0)
    sigma_sq = np.var(diff_ctrl)
    
    # Correct zeta calculation: (N_tr * T_post)^0.25 * sigma
    # But we need zeta^2 for the penalty term
    # zeta^2 = (N_tr * T_post)^0.5 * sigma^2
    
    zeta_sq = (N_tr * T_post)**0.5 * sigma_sq

    omega_init = np.ones(N_co) / N_co

    def objective(omega):
        synth = Y_pre_ctrl @ omega
        alpha = np.mean(Y_pre_treat - synth)
        residuals = Y_pre_treat - (alpha + synth)
        # Correct objective: sum(residuals^2) + zeta^2 * T_pre * ||omega||^2
        loss = np.sum(residuals ** 2) + (zeta_sq * T_pre * np.sum(omega ** 2))
        return loss

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, None) for _ in range(N_co)]

    result = minimize(objective, omega_init, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-10})

    omega = result.x
    synth = Y_pre_ctrl @ omega
    alpha = np.mean(Y_pre_treat - synth)

    return omega, alpha

def solve_time_weights(Y_pre_ctrl, Y_post_ctrl):
    """Find time weights (lambda)."""
    T_pre = Y_pre_ctrl.shape[0]
    target = Y_post_ctrl.mean(axis=0)
    features = Y_pre_ctrl.T
    lam_init = np.ones(T_pre) / T_pre

    def objective(lam):
        pred = features @ lam
        alpha = np.mean(target - pred)
        residuals = target - (alpha + pred)
        return np.sum(residuals ** 2)

    constraints = {'type': 'eq', 'fun': lambda l: np.sum(l) - 1}
    bounds = [(0, None) for _ in range(T_pre)]

    result = minimize(objective, lam_init, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-10})

    lam = result.x
    pred = features @ lam
    alpha = np.mean(target - pred)

    return lam, alpha

def calculate_sdid_tau(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, omega, lam):
    mu_11 = np.mean(Y_post_treat)
    mu_10 = np.sum(Y_pre_treat * lam)

    synth_post = Y_post_ctrl @ omega
    mu_01 = np.mean(synth_post)

    synth_pre = Y_pre_ctrl @ omega
    mu_00 = np.sum(synth_pre * lam)

    tau = (mu_11 - mu_10) - (mu_01 - mu_00)
    return tau

def main():
    df = load_smoking()
    wide, Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, control_states = prepare_sdid_data(df)
    
    N_tr = 1
    T_post = len(Y_post_treat)
    
    # 1. Current Implementation
    omega_curr, alpha_curr = solve_unit_weights_current(Y_pre_treat, Y_pre_ctrl, N_tr, T_post)
    lam_curr, _ = solve_time_weights(Y_pre_ctrl, Y_post_ctrl)
    tau_curr = calculate_sdid_tau(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, omega_curr, lam_curr)
    print(f"Current SDID Tau: {tau_curr:.2f}")
    
    # 2. Corrected Implementation
    omega_corr, alpha_corr = solve_unit_weights_corrected(Y_pre_treat, Y_pre_ctrl, N_tr, T_post)
    tau_corr = calculate_sdid_tau(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, omega_corr, lam_curr)
    print(f"Corrected SDID Tau: {tau_corr:.2f}")

    # 3. DIFP (SC with intercept, negligible regularization)
    def solve_unit_weights_difp(Y_pre_treat, Y_pre_ctrl):
        T_pre, N_co = Y_pre_ctrl.shape
        diff_ctrl = np.diff(Y_pre_ctrl, axis=0)
        sigma_sq = np.var(diff_ctrl)
        zeta_sq = 1e-6 * sigma_sq # Negligible
        
        omega_init = np.ones(N_co) / N_co
        
        def objective(omega):
            synth = Y_pre_ctrl @ omega
            alpha = np.mean(Y_pre_treat - synth)
            residuals = Y_pre_treat - (alpha + synth)
            loss = np.sum(residuals ** 2) + (zeta_sq * T_pre * np.sum(omega ** 2))
            return loss

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, None) for _ in range(N_co)]

        result = minimize(objective, omega_init, method='SLSQP',
                        bounds=bounds, constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-10})
        return result.x, np.mean(Y_pre_treat - (Y_pre_ctrl @ result.x))

    omega_difp, alpha_difp = solve_unit_weights_difp(Y_pre_treat, Y_pre_ctrl)
    synth_post_difp = (Y_post_ctrl @ omega_difp) + alpha_difp
    tau_difp = np.mean(Y_post_treat - synth_post_difp)
    print(f"DIFP Tau (intercept, negligible reg): {tau_difp:.2f}")
    
    # 4. SC (No intercept, negligible regularization)
    def solve_unit_weights_sc(Y_pre_treat, Y_pre_ctrl):
        T_pre, N_co = Y_pre_ctrl.shape
        diff_ctrl = np.diff(Y_pre_ctrl, axis=0)
        sigma_sq = np.var(diff_ctrl)
        zeta_sq = 1e-6 * sigma_sq # Negligible
        
        omega_init = np.ones(N_co) / N_co
        
        def objective(omega):
            synth = Y_pre_ctrl @ omega
            # No intercept
            residuals = Y_pre_treat - synth
            loss = np.sum(residuals ** 2) + (zeta_sq * T_pre * np.sum(omega ** 2))
            return loss

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, None) for _ in range(N_co)]

        result = minimize(objective, omega_init, method='SLSQP',
                        bounds=bounds, constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-10})
        return result.x

    omega_sc = solve_unit_weights_sc(Y_pre_treat, Y_pre_ctrl)
    synth_post_sc = Y_post_ctrl @ omega_sc
    tau_sc = np.mean(Y_post_treat - synth_post_sc)
    print(f"SC Tau (no intercept, negligible reg): {tau_sc:.2f}")

if __name__ == "__main__":
    main()
