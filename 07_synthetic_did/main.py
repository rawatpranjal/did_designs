"""
Module 07: Synthetic Difference-in-Differences

This script demonstrates:
1. Manual SDID: Unit weights + Time weights + Weighted DiD
2. Comparison with naive DiD and Synthetic Control
3. Verification against synthdid package

Dataset: California Proposition 99 (1988) - Tobacco Tax
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Optional synthdid package for verification
try:
    from synthdid.synthdid import Synthdid as sdid_pkg
    SYNTHDID_AVAILABLE = True
except ImportError:
    SYNTHDID_AVAILABLE = False

# Optional pysyncon package for classic SC
try:
    from pysyncon import Dataprep, Synth as SynthClassic
    PYSYNCON_AVAILABLE = True
except ImportError:
    PYSYNCON_AVAILABLE = False

from utils import load_smoking, COLORS

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Treatment configuration
TREATMENT_YEAR = 1989
TREATED_STATE = 'California'


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_sdid_data(df):
    """Prepare data matrices for SDID estimation."""
    wide = df.pivot(index='year', columns='state', values='cigsale')
    pre_mask = wide.index < TREATMENT_YEAR
    post_mask = wide.index >= TREATMENT_YEAR
    control_states = [s for s in wide.columns if s != TREATED_STATE]

    Y_pre_treat = wide.loc[pre_mask, TREATED_STATE].values
    Y_post_treat = wide.loc[post_mask, TREATED_STATE].values
    Y_pre_ctrl = wide.loc[pre_mask, control_states].values
    Y_post_ctrl = wide.loc[post_mask, control_states].values

    return wide, Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, control_states


# =============================================================================
# Weight Optimization
# =============================================================================

def solve_unit_weights(Y_pre_treat, Y_pre_ctrl, N_tr=1, T_post=1, zeta=None):
    """Find unit weights (omega)."""
    T_pre, N_co = Y_pre_ctrl.shape

    if zeta is None:
        diff_ctrl = np.diff(Y_pre_ctrl, axis=0)
        sigma_sq = np.var(diff_ctrl)
        # Correct formula per Arkhangelsky et al. (2021) Eq 2.2
        # Zeta is the regularization parameter, but the penalty term is zeta^2
        # zeta = (N_tr * T_post)**0.25 * sigma
        # penalty = zeta^2 * T_pre * ||omega||^2
        # Therefore, penalty coefficient = (N_tr * T_post)**0.5 * sigma^2
        zeta_sq = (N_tr * T_post)**0.5 * sigma_sq

    omega_init = np.ones(N_co) / N_co

    def objective(omega):
        synth = Y_pre_ctrl @ omega
        alpha = np.mean(Y_pre_treat - synth)
        residuals = Y_pre_treat - (alpha + synth)
        # Paper scales penalty by T_pre
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


# =============================================================================
# Estimators
# =============================================================================

def manual_sdid(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl):
    """Compute the SDID estimator."""
    N_tr = 1
    T_post = len(Y_post_treat)

    omega, alpha_omega = solve_unit_weights(Y_pre_treat, Y_pre_ctrl, N_tr, T_post)
    lam, alpha_lam = solve_time_weights(Y_pre_ctrl, Y_post_ctrl)

    mu_11 = np.mean(Y_post_treat)
    mu_10 = np.sum(Y_pre_treat * lam)

    synth_post = Y_post_ctrl @ omega
    mu_01 = np.mean(synth_post)

    synth_pre = Y_pre_ctrl @ omega
    mu_00 = np.sum(synth_pre * lam)

    tau = (mu_11 - mu_10) - (mu_01 - mu_00)

    return tau, omega, lam, alpha_omega


def naive_did(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl):
    return (np.mean(Y_post_treat) - np.mean(Y_pre_treat)) - \
           (np.mean(Y_post_ctrl) - np.mean(Y_pre_ctrl))


def synthetic_control(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl):
    # Standard SC using regularized weights for comparison
    omega, alpha = solve_unit_weights(Y_pre_treat, Y_pre_ctrl, 1, len(Y_post_treat))
    synth_post = (Y_post_ctrl @ omega) + alpha
    return np.mean(Y_post_treat - synth_post)


def synthdid_package_estimate(df):
    """Compute SDID using the synthdid package for verification."""
    if not SYNTHDID_AVAILABLE:
        return None

    # Prepare data in synthdid format
    df_pkg = df.copy()
    df_pkg['treated'] = ((df_pkg['state'] == TREATED_STATE) &
                         (df_pkg['year'] >= TREATMENT_YEAR)).astype(int)

    model = sdid_pkg(df_pkg, unit='state', time='year',
                     treatment='treated', outcome='cigsale')
    result = model.fit()
    return result.att


def sc_package_estimate(df):
    """Classic SC (Abadie et al. 2010) using pysyncon package."""
    if not PYSYNCON_AVAILABLE:
        return None

    states = [s for s in df['state'].unique() if s != TREATED_STATE]
    pre_years = list(range(1970, TREATMENT_YEAR))
    post_years = list(range(TREATMENT_YEAR, 2001))

    # Match each pre-treatment year outcome (Abadie et al. approach)
    special_predictors = [
        ('cigsale', [year], 'mean') for year in pre_years
    ]

    dataprep = Dataprep(
        foo=df,
        predictors=[],
        predictors_op='mean',
        special_predictors=special_predictors,
        time_predictors_prior=pre_years,
        dependent='cigsale',
        unit_variable='state',
        time_variable='year',
        treatment_identifier=TREATED_STATE,
        controls_identifier=states,
        time_optimize_ssr=pre_years
    )

    synth = SynthClassic()
    synth.fit(dataprep)

    # Compute SC effect manually from weights
    weights = synth.weights(threshold=0.001).to_dict()
    wide = df.pivot(index='year', columns='state', values='cigsale')
    synth_ca = sum(wide[state] * w for state, w in weights.items())

    post_mask = wide.index.isin(post_years)
    sc_effect = (wide.loc[post_mask, TREATED_STATE] - synth_ca.loc[post_mask]).mean()
    return sc_effect


# =============================================================================
# Dashboard Visualization
# =============================================================================

def create_sdid_dashboard(wide, omega, lam, alpha_omega, tau_sdid, control_states):
    """
    Creates a comprehensive 3-panel dashboard:
    1. Main Trends (The "Hero" Plot)
    2. Unit Weights (Geography of the counterfactual)
    3. Time Weights (Temporal relevance)
    """
    years = wide.index
    years_pre = years[years < TREATMENT_YEAR]

    # Calculate curves
    actual = wide[TREATED_STATE]

    # The Synthetic Control curve (Unit weighted + Intercept)
    synth_full = (wide[control_states].values @ omega) + alpha_omega

    # --- Setup Grid ---
    # plt.style.use('seaborn-v0_8-whitegrid') # Already set in utils
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 0.6])

    # --- Panel 1: The Trajectories (Top Span) ---
    ax1 = fig.add_subplot(gs[0, :])

    # Plot Actual
    ax1.plot(years, actual, color=COLORS['treat'], linewidth=3, label='California (Actual)')

    # Plot Counterfactual (Dashed for post-treatment to imply "what if")
    ax1.plot(years[years < TREATMENT_YEAR], synth_full[years < TREATMENT_YEAR],
             color=COLORS['control'], linewidth=2, label='Synthetic Counterfactual')
    ax1.plot(years[years >= TREATMENT_YEAR], synth_full[years >= TREATMENT_YEAR],
             color=COLORS['control'], linewidth=2, linestyle='--')

    # Shade the Effect Area
    ax1.fill_between(years[years >= TREATMENT_YEAR],
                     actual[years >= TREATMENT_YEAR],
                     synth_full[years >= TREATMENT_YEAR],
                     color=COLORS['treat'], alpha=0.1, label=f'Effect: {tau_sdid:.2f} packs')

    # Treatment Line
    ax1.axvline(TREATMENT_YEAR, color=COLORS['grid'], linestyle=':', linewidth=1.5)
    ax1.text(TREATMENT_YEAR + 0.5, 120, 'Prop 99\nPassed', fontsize=10, color=COLORS['text'])

    # Annotations
    ax1.set_ylabel('Cigarette Sales (Packs/Capita)', fontsize=12)
    ax1.set_title('Synthetic Difference-in-Differences: Impact of Prop 99', fontsize=16, fontweight='bold')
    ax1.legend(loc='lower left', frameon=True, framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Clean spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Panel 2: Unit Weights (Bottom Left) ---
    ax2 = fig.add_subplot(gs[1, 0])

    # Filter to meaningful weights
    mask = omega > 0.01
    w_states = np.array(control_states)[mask]
    w_vals = omega[mask]

    # Sort
    sort_idx = np.argsort(w_vals)
    w_states = w_states[sort_idx]
    w_vals = w_vals[sort_idx]

    bars = ax2.barh(w_states, w_vals, color=COLORS['control'], alpha=0.7)
    ax2.set_xlabel('Weight Contribution', fontsize=10)
    ax2.set_title('Which States make up "Synthetic California"?', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x', color=COLORS['grid'])
    
    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Value labels
    for i, v in enumerate(w_vals):
        ax2.text(v + 0.01, i, f'{v:.2f}', va='center', fontsize=9, color=COLORS['text'])

    # --- Panel 3: Time Weights (Bottom Right) ---
    ax3 = fig.add_subplot(gs[1, 1])

    # Plot weights against years
    bars3 = ax3.bar(years_pre, lam, color=COLORS['counterfactual'], alpha=0.5, width=0.8)

    # Highlight high weight years
    high_w_mask = lam > np.mean(lam) + 0.5*np.std(lam)
    for i, is_high in enumerate(high_w_mask):
        if is_high:
            bars3[i].set_color(COLORS['treat'])
            bars3[i].set_alpha(0.8)

    ax3.set_xlabel('Pre-Treatment Year', fontsize=10)
    ax3.set_ylabel('Weight', fontsize=10)
    ax3.set_title('Which Years Matter? (Time Weights $\lambda$)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # Clean spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Add text explaining time weights
    ax3.text(0.05, 0.85, "SDID emphasizes pre-treatment years\nthat look most like the post-period.",
             transform=ax3.transAxes, fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=COLORS['grid']))

    plt.tight_layout()
    return fig


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("Module 07: Synthetic Difference-in-Differences")
    print("California Proposition 99 (1988) - Tobacco Tax")
    print("=" * 70)

    # 1. Load
    print("\n[1] Loading Data...")
    df = load_smoking()
    wide, Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, control_states = prepare_sdid_data(df)

    years_pre = wide.index[wide.index < TREATMENT_YEAR].tolist()
    print(f"Pre-treatment: {len(Y_pre_treat)} years, Post-treatment: {len(Y_post_treat)} years")
    print(f"Control states: {len(control_states)}")

    # 2. Estimate
    print("\n[2] Computing Estimates...")
    tau_sdid, omega, lam, alpha_omega = manual_sdid(
        Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl
    )

    # 3. Compare
    tau_naive = naive_did(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl)
    tau_sc = synthetic_control(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl)

    print(f"\nMethod Comparison:")
    print("-" * 50)
    print(f"  {'Method':<20} {'Estimate':>10} {'Paper':>10}")
    print("-" * 50)
    print(f"  {'Naive DiD':<20} {tau_naive:>10.2f} {-27.3:>10.1f}")

    # Classic SC via pysyncon
    if PYSYNCON_AVAILABLE:
        tau_sc_pkg = sc_package_estimate(df)
        print(f"  {'SC (pysyncon)':<20} {tau_sc_pkg:>10.2f} {-19.6:>10.1f}")
    else:
        print(f"  {'SC (manual)':<20} {tau_sc:>10.2f} {-19.6:>10.1f}")

    print(f"  {'SDID (manual)':<20} {tau_sdid:>10.2f} {-15.6:>10.1f}")

    # SDID package verification
    if SYNTHDID_AVAILABLE:
        tau_pkg = synthdid_package_estimate(df)
        print(f"  {'SDID (synthdid)':<20} {tau_pkg:>10.2f} {-15.6:>10.1f}")
    print("-" * 50)
    print("Paper: Arkhangelsky et al. (2021) Table 1")

    # Show top unit weights
    print("\nTop 5 Unit Weights:")
    top_idx = np.argsort(-omega)[:5]
    for idx in top_idx:
        if omega[idx] > 0.001:
            print(f"  {control_states[idx]:<15}: {omega[idx]:.4f}")

    # 4. Dashboard
    print("\n[3] Generating Dashboard...")
    
    # --- Plot 1: Performance Contest (Comparison) ---
    fig_comp, ax_comp = plt.subplots(figsize=(10, 6))
    
    methods = ['Naive DiD', 'Synthetic Control', 'Synthetic DiD']
    estimates = [tau_naive, tau_sc, tau_sdid]
    # Colors: Gray for Naive, Blue for SC, Red for SDID
    colors = [COLORS['counterfactual'], COLORS['control'], COLORS['treat']]
    
    bars = ax_comp.bar(methods, estimates, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    ax_comp.axhline(0, color='black', linewidth=1)
    
    # Add values
    for bar, v in zip(bars, estimates):
        height = bar.get_height()
        offset = 1 if height > 0 else -2
        ax_comp.text(bar.get_x() + bar.get_width()/2, height + offset, f"{v:.1f}",
                 ha='center', fontweight='bold', color=COLORS['text'])
                 
    # Add Paper Benchmark lines
    paper_benchmarks = [-27.3, -19.6, -15.6] # From Arkhangelsky et al. Table 1
    for i, bench in enumerate(paper_benchmarks):
        ax_comp.plot([i-0.3, i+0.3], [bench, bench], color='black', linestyle='--', linewidth=2, 
                     label='Paper Benchmark' if i==0 else "")
                     
    ax_comp.set_title('The Performance Contest: Estimating the Effect of Prop 99', fontsize=16, fontweight='bold')
    ax_comp.set_ylabel('Estimated Effect (Packs per Capita)', fontsize=12)
    ax_comp.legend()
    ax_comp.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # Clean spines
    ax_comp.spines['top'].set_visible(False)
    ax_comp.spines['right'].set_visible(False)
    
    plt.tight_layout()
    fig_comp.savefig(FIGS_DIR / 'sdid_comparison.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'sdid_comparison.png'}")

    fig = create_sdid_dashboard(wide, omega, lam, alpha_omega, tau_sdid, control_states)
    fig.savefig(FIGS_DIR / 'sdid_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'sdid_dashboard.png'}")

    plt.close('all')
    print("\n" + "=" * 70)
    print("Module 07 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
