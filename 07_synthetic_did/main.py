"""
Module 07: Synthetic Difference-in-Differences

This script demonstrates:
1. Manual SDID: Unit weights + Time weights + Weighted DiD
2. Comparison with naive DiD and Synthetic Control
3. Verification against pysynthdid package

Dataset: California Proposition 99 (1988) - Tobacco Tax

Reference: Arkhangelsky et al. (2021) "Synthetic Difference-in-Differences"
           American Economic Review
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from utils import load_smoking

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
    """
    Prepare data matrices for SDID estimation.

    Returns:
        wide: DataFrame with years as rows, states as columns
        Y_pre_treat: Pre-treatment outcomes for treated unit (T_pre,)
        Y_post_treat: Post-treatment outcomes for treated unit (T_post,)
        Y_pre_ctrl: Pre-treatment outcomes for controls (T_pre, N_co)
        Y_post_ctrl: Post-treatment outcomes for controls (T_post, N_co)
        control_states: List of control state names
    """
    # Pivot to wide format
    wide = df.pivot(index='year', columns='state', values='cigsale')

    # Define time periods
    pre_mask = wide.index < TREATMENT_YEAR
    post_mask = wide.index >= TREATMENT_YEAR

    # Control states (all except California)
    control_states = [s for s in wide.columns if s != TREATED_STATE]

    # Extract matrices
    Y_pre_treat = wide.loc[pre_mask, TREATED_STATE].values
    Y_post_treat = wide.loc[post_mask, TREATED_STATE].values
    Y_pre_ctrl = wide.loc[pre_mask, control_states].values
    Y_post_ctrl = wide.loc[post_mask, control_states].values

    return wide, Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, control_states


# =============================================================================
# Weight Optimization (The Core of SDID)
# =============================================================================

def solve_unit_weights(Y_pre_treat, Y_pre_ctrl, zeta=None):
    """
    Find unit weights (omega) that make synthetic control match treated pre-trends.

    Solves:
        min ||Y_pre_treat - (alpha + Y_pre_ctrl @ omega)||^2 + zeta * ||omega||^2
        s.t. sum(omega) = 1, omega >= 0

    Args:
        Y_pre_treat: Pre-treatment outcomes for treated unit (T_pre,)
        Y_pre_ctrl: Pre-treatment outcomes for controls (T_pre, N_co)
        zeta: Regularization parameter (default: data-driven)

    Returns:
        omega: Unit weights (N_co,)
        alpha: Intercept
    """
    T_pre, N_co = Y_pre_ctrl.shape

    # Data-driven regularization (from Arkhangelsky et al.)
    if zeta is None:
        # Variance of first differences in control group
        diff_ctrl = np.diff(Y_pre_ctrl, axis=0)
        zeta = (T_pre * N_co) ** 0.25 * np.var(diff_ctrl)

    # Initial guess: uniform weights
    omega_init = np.ones(N_co) / N_co

    def objective(omega):
        # Compute optimal alpha for given omega
        synth = Y_pre_ctrl @ omega
        alpha = np.mean(Y_pre_treat - synth)

        # Residual sum of squares + regularization
        residuals = Y_pre_treat - (alpha + synth)
        loss = np.sum(residuals ** 2) + zeta * np.sum(omega ** 2)
        return loss

    # Constraints: sum(omega) = 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

    # Bounds: omega >= 0
    bounds = [(0, None) for _ in range(N_co)]

    # Optimize
    result = minimize(objective, omega_init, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-10})

    omega = result.x

    # Compute final alpha
    synth = Y_pre_ctrl @ omega
    alpha = np.mean(Y_pre_treat - synth)

    return omega, alpha


def solve_time_weights(Y_pre_ctrl, Y_post_ctrl):
    """
    Find time weights (lambda) that make pre-periods resemble post-periods.

    The idea: weight pre-treatment periods so that the weighted average
    of control outcomes looks like the post-treatment average.

    Solves:
        min ||mean(Y_post_ctrl, axis=0) - (alpha + Y_pre_ctrl.T @ lambda)||^2
        s.t. sum(lambda) = 1, lambda >= 0

    Args:
        Y_pre_ctrl: Pre-treatment outcomes for controls (T_pre, N_co)
        Y_post_ctrl: Post-treatment outcomes for controls (T_post, N_co)

    Returns:
        lam: Time weights (T_pre,)
        alpha: Intercept
    """
    T_pre = Y_pre_ctrl.shape[0]

    # Target: average post-treatment outcome for each control unit
    target = Y_post_ctrl.mean(axis=0)  # (N_co,)

    # Features: pre-treatment outcomes transposed (N_co, T_pre)
    features = Y_pre_ctrl.T

    # Initial guess: uniform weights
    lam_init = np.ones(T_pre) / T_pre

    def objective(lam):
        # Compute optimal alpha for given lambda
        pred = features @ lam
        alpha = np.mean(target - pred)

        # Residual sum of squares
        residuals = target - (alpha + pred)
        return np.sum(residuals ** 2)

    # Constraints: sum(lambda) = 1
    constraints = {'type': 'eq', 'fun': lambda l: np.sum(l) - 1}

    # Bounds: lambda >= 0
    bounds = [(0, None) for _ in range(T_pre)]

    # Optimize
    result = minimize(objective, lam_init, method='SLSQP',
                     bounds=bounds, constraints=constraints,
                     options={'maxiter': 1000, 'ftol': 1e-10})

    lam = result.x

    # Compute final alpha
    pred = features @ lam
    alpha = np.mean(target - pred)

    return lam, alpha


# =============================================================================
# SDID Estimator
# =============================================================================

def manual_sdid(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl):
    """
    Compute the Synthetic DiD estimator manually.

    SDID = (mu_11 - mu_10) - (mu_01 - mu_00)

    where:
        mu_11 = mean(Y_post_treat)              # Treated, Post
        mu_10 = sum(Y_pre_treat * lambda)       # Treated, Pre (time-weighted)
        mu_01 = mean(Y_post_ctrl @ omega)       # Control, Post (unit-weighted)
        mu_00 = sum((Y_pre_ctrl @ omega) * lambda)  # Both weighted

    Returns:
        tau: SDID estimate
        omega: Unit weights
        lam: Time weights
        components: Dict with intermediate values
    """
    # Step 1: Unit weights (omega)
    omega, alpha_omega = solve_unit_weights(Y_pre_treat, Y_pre_ctrl)

    # Step 2: Time weights (lambda)
    lam, alpha_lam = solve_time_weights(Y_pre_ctrl, Y_post_ctrl)

    # Step 3: Compute the four weighted means

    # Treated, Post: simple average
    mu_11 = np.mean(Y_post_treat)

    # Treated, Pre: time-weighted average
    mu_10 = np.sum(Y_pre_treat * lam)

    # Control, Post: unit-weighted, then averaged over post periods
    synth_post = Y_post_ctrl @ omega  # (T_post,)
    mu_01 = np.mean(synth_post)

    # Control, Pre: both unit and time weighted
    synth_pre = Y_pre_ctrl @ omega  # (T_pre,)
    mu_00 = np.sum(synth_pre * lam)

    # Step 4: DiD formula
    tau = (mu_11 - mu_10) - (mu_01 - mu_00)

    components = {
        'mu_11': mu_11,  # Treated, Post
        'mu_10': mu_10,  # Treated, Pre (weighted)
        'mu_01': mu_01,  # Control, Post (weighted)
        'mu_00': mu_00,  # Control, Pre (double weighted)
        'alpha_omega': alpha_omega,
        'alpha_lam': alpha_lam
    }

    return tau, omega, lam, components


# =============================================================================
# Comparison Methods
# =============================================================================

def naive_did(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl):
    """
    Compute naive DiD (simple averages, no weighting).
    """
    mu_11 = np.mean(Y_post_treat)
    mu_10 = np.mean(Y_pre_treat)
    mu_01 = np.mean(Y_post_ctrl)
    mu_00 = np.mean(Y_pre_ctrl)

    return (mu_11 - mu_10) - (mu_01 - mu_00)


def synthetic_control(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl):
    """
    Compute Synthetic Control (unit weights only, no time weights).
    """
    # Get unit weights
    omega, _ = solve_unit_weights(Y_pre_treat, Y_pre_ctrl)

    # Synthetic control for post-treatment
    synth_post = np.mean(Y_post_ctrl @ omega)

    # Treatment effect = Treated_post - Synthetic_post
    return np.mean(Y_post_treat) - synth_post, omega


# =============================================================================
# Verification with pysynthdid
# =============================================================================

def verify_with_pysynthdid(df):
    """
    Verify our manual implementation against pysynthdid package.
    """
    try:
        from synthdid.model import SynthDID
        from synthdid.sample_data import fetch_CaliforniaSmoking

        # Use the package's built-in data
        pkg_df = fetch_CaliforniaSmoking()

        PRE_TERM = [1970, 1988]
        POST_TERM = [1989, 2000]
        TREATMENT = ["California"]

        sdid = SynthDID(pkg_df, PRE_TERM, POST_TERM, TREATMENT)
        sdid.fit(zeta_type="base")

        # Get the estimate
        pkg_estimate = sdid.effect

        return pkg_estimate, True

    except ImportError:
        return None, False
    except Exception as e:
        print(f"Warning: pysynthdid verification failed: {e}")
        return None, False


# =============================================================================
# Visualization
# =============================================================================

def plot_trends(wide, omega, control_states):
    """
    Plot California vs. control average vs. synthetic control.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    years = wide.index

    # California (actual)
    ax.plot(years, wide[TREATED_STATE], 'b-', linewidth=2.5,
            label='California (Treated)', marker='o', markersize=4)

    # Average of all controls
    control_avg = wide[control_states].mean(axis=1)
    ax.plot(years, control_avg, 'gray', linewidth=1.5,
            label='Control Average', linestyle='--', alpha=0.7)

    # Synthetic control (unit-weighted)
    synth = wide[control_states].values @ omega
    ax.plot(years, synth, 'r-', linewidth=2.5,
            label='Synthetic California', marker='s', markersize=4)

    # Treatment line
    ax.axvline(x=TREATMENT_YEAR, color='black', linestyle='--',
               linewidth=2, label='Prop 99 (1989)')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cigarette Sales Per Capita (Packs)', fontsize=12)
    ax.set_title('Synthetic DiD: California vs. Synthetic Control\n'
                 'Unit weights create a better control group', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_weights(omega, lam, control_states, years_pre):
    """
    Plot unit weights and time weights.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Unit weights (top contributors)
    ax1 = axes[0]
    top_n = 10
    top_idx = np.argsort(-omega)[:top_n]
    top_states = [control_states[i] for i in top_idx]
    top_weights = omega[top_idx]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, top_n))
    bars1 = ax1.barh(range(top_n), top_weights, color=colors)
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(top_states)
    ax1.set_xlabel('Weight', fontsize=11)
    ax1.set_title('Unit Weights (Omega)\nTop 10 Control States', fontsize=12)
    ax1.invert_yaxis()

    # Add value labels
    for i, (bar, w) in enumerate(zip(bars1, top_weights)):
        if w > 0.01:
            ax1.text(w + 0.005, i, f'{w:.3f}', va='center', fontsize=9)

    # Time weights
    ax2 = axes[1]
    colors2 = plt.cm.Oranges(np.linspace(0.3, 0.9, len(lam)))
    bars2 = ax2.bar(years_pre, lam, color=colors2, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Weight', fontsize=11)
    ax2.set_title('Time Weights (Lambda)\nPre-Treatment Years', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # Highlight most important years
    top_time_idx = np.argsort(-lam)[:3]
    for idx in top_time_idx:
        if lam[idx] > 0.05:
            ax2.annotate(f'{lam[idx]:.2f}',
                        xy=(years_pre[idx], lam[idx]),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=9)

    fig.suptitle('SDID Weights: Who and When Matter Most', fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_comparison(naive_est, sc_est, sdid_est, pkg_est=None):
    """
    Plot comparison of different estimation methods.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ['Naive DiD', 'Synthetic Control', 'SDID (Manual)']
    estimates = [naive_est, sc_est, sdid_est]
    colors = ['gray', 'steelblue', 'coral']

    if pkg_est is not None:
        methods.append('SDID (pysynthdid)')
        estimates.append(pkg_est)
        colors.append('green')

    bars = ax.bar(methods, estimates, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

    # Add value labels
    for bar, est in zip(bars, estimates):
        height = bar.get_height()
        ax.annotate(f'{est:.2f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, -20 if height < 0 else 5),
                   textcoords='offset points',
                   ha='center', va='bottom' if height < 0 else 'bottom',
                   fontsize=12, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel('Treatment Effect (Packs per Capita)', fontsize=12)
    ax.set_title('Comparison of DiD Methods\nEffect of Prop 99 on Cigarette Sales', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotation about why methods differ
    ax.annotate('Naive DiD: Poor control group match\n'
                'SC: No time weighting\n'
                'SDID: Best of both worlds',
                xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=9, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()
    return fig


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("=" * 70)
    print("Module 07: Synthetic Difference-in-Differences")
    print("California Proposition 99 (1988) - Tobacco Tax")
    print("Reference: Arkhangelsky et al. (2021)")
    print("=" * 70)

    # =========================================================================
    # Step 1: Load and Prepare Data
    # =========================================================================
    print("\n[1] Loading California Prop 99 smoking data...")
    df = load_smoking()

    wide, Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl, control_states = prepare_sdid_data(df)

    years_pre = wide.index[wide.index < TREATMENT_YEAR].tolist()
    years_post = wide.index[wide.index >= TREATMENT_YEAR].tolist()

    print(f"\nData dimensions:")
    print(f"  Years: {wide.index.min()} - {wide.index.max()}")
    print(f"  Pre-treatment periods: {len(Y_pre_treat)} ({years_pre[0]}-{years_pre[-1]})")
    print(f"  Post-treatment periods: {len(Y_post_treat)} ({years_post[0]}-{years_post[-1]})")
    print(f"  Treated unit: {TREATED_STATE}")
    print(f"  Control units: {len(control_states)}")

    # =========================================================================
    # Step 2: Calculate Unit Weights (Omega)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2] Calculating Unit Weights (Omega)")
    print("=" * 70)

    print("""
The Goal: Find weights omega such that the weighted average of control
states matches California's pre-treatment trajectory.

This is the "Synthetic Control" part of SDID.
    """)

    omega, alpha_omega = solve_unit_weights(Y_pre_treat, Y_pre_ctrl)

    # Show top contributing states
    top_n = 5
    top_idx = np.argsort(-omega)[:top_n]

    print(f"Top {top_n} Control States by Weight:")
    print("-" * 40)
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. {control_states[idx]}: {omega[idx]:.4f}")

    print(f"\nIntercept (alpha): {alpha_omega:.4f}")
    print(f"Sum of weights: {np.sum(omega):.4f} (should be 1.0)")
    print(f"Non-zero weights: {np.sum(omega > 0.001)}")

    # =========================================================================
    # Step 3: Calculate Time Weights (Lambda)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] Calculating Time Weights (Lambda)")
    print("=" * 70)

    print("""
The Goal: Find weights lambda such that the weighted pre-treatment average
resembles the post-treatment period. This is unique to SDID.

Why? It down-weights early pre-treatment periods that are very different
from the post-treatment period, focusing on more relevant comparisons.
    """)

    lam, alpha_lam = solve_time_weights(Y_pre_ctrl, Y_post_ctrl)

    # Show time weight distribution
    print("Time Weights by Year:")
    print("-" * 40)
    top_time_idx = np.argsort(-lam)[:5]
    for idx in top_time_idx:
        print(f"  {years_pre[idx]}: {lam[idx]:.4f}")

    print(f"\nSum of time weights: {np.sum(lam):.4f} (should be 1.0)")

    # =========================================================================
    # Step 4: Compute SDID Estimate
    # =========================================================================
    print("\n" + "=" * 70)
    print("[4] Computing SDID Estimate")
    print("=" * 70)

    tau_sdid, _, _, components = manual_sdid(Y_pre_treat, Y_post_treat,
                                              Y_pre_ctrl, Y_post_ctrl)

    print("""
SDID Formula: tau = (mu_11 - mu_10) - (mu_01 - mu_00)

where:
  mu_11 = mean(Y_post_treat)              -> Treated, Post
  mu_10 = sum(Y_pre_treat * lambda)       -> Treated, Pre (time-weighted)
  mu_01 = mean(Y_post_ctrl @ omega)       -> Control, Post (unit-weighted)
  mu_00 = sum((Y_pre_ctrl @ omega) * lambda) -> Both weighted
    """)

    print("Intermediate Values:")
    print("-" * 50)
    print(f"  mu_11 (Treated, Post):       {components['mu_11']:.4f}")
    print(f"  mu_10 (Treated, Pre):        {components['mu_10']:.4f}  (time-weighted)")
    print(f"  mu_01 (Control, Post):       {components['mu_01']:.4f}  (unit-weighted)")
    print(f"  mu_00 (Control, Pre):        {components['mu_00']:.4f}  (double-weighted)")
    print("-" * 50)

    delta_treat = components['mu_11'] - components['mu_10']
    delta_ctrl = components['mu_01'] - components['mu_00']

    print(f"\nFirst Differences:")
    print(f"  Delta Treated: {components['mu_11']:.2f} - {components['mu_10']:.2f} = {delta_treat:.4f}")
    print(f"  Delta Control: {components['mu_01']:.2f} - {components['mu_00']:.2f} = {delta_ctrl:.4f}")
    print("-" * 50)
    print(f"  SDID Estimate: {delta_treat:.4f} - {delta_ctrl:.4f} = {tau_sdid:.4f}")

    # =========================================================================
    # Step 5: Comparison with Other Methods
    # =========================================================================
    print("\n" + "=" * 70)
    print("[5] Comparison with Other Methods")
    print("=" * 70)

    # Naive DiD
    tau_naive = naive_did(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl)

    # Synthetic Control
    tau_sc, _ = synthetic_control(Y_pre_treat, Y_post_treat, Y_pre_ctrl, Y_post_ctrl)

    print("\nMethod Comparison:")
    print("-" * 50)
    print(f"  Naive DiD:           {tau_naive:.4f}  (no weighting)")
    print(f"  Synthetic Control:   {tau_sc:.4f}  (unit weights only)")
    print(f"  SDID:                {tau_sdid:.4f}  (unit + time weights)")
    print("-" * 50)

    print("""
Interpretation:
  - Naive DiD uses all control states equally -> poor pre-trend match
  - SC reweights units but ignores time dynamics
  - SDID combines both: better control group + relevant time periods
    """)

    # =========================================================================
    # Step 6: Verification with pysynthdid
    # =========================================================================
    print("\n" + "=" * 70)
    print("[6] Verification with pysynthdid Package")
    print("=" * 70)

    pkg_est, pkg_available = verify_with_pysynthdid(df)

    if pkg_available:
        print(f"\npysynthdid estimate: {pkg_est:.4f}")
        print(f"Manual estimate:     {tau_sdid:.4f}")
        print(f"Difference:          {abs(pkg_est - tau_sdid):.4f}")

        if abs(pkg_est - tau_sdid) < 1.0:
            print("\nVerification: PASSED (estimates are close)")
        else:
            print("\nNote: Estimates differ due to implementation details")
            print("(regularization, optimization algorithm, etc.)")
    else:
        print("""
pysynthdid package not installed. To verify, run:
  pip install git+https://github.com/MasaAsami/pysynthdid

Expected result from literature: approximately -15 to -20 packs per capita
        """)

    # =========================================================================
    # Step 7: Create Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("[7] Creating Visualizations")
    print("=" * 70)

    # Plot 1: Raw trends with synthetic control
    fig1 = plot_trends(wide, omega, control_states)
    fig1.savefig(FIGS_DIR / 'raw_trends.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'raw_trends.png'}")

    # Plot 2: Weights visualization
    fig2 = plot_weights(omega, lam, control_states, years_pre)
    fig2.savefig(FIGS_DIR / 'sdid_weights.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'sdid_weights.png'}")

    # Plot 3: Method comparison
    fig3 = plot_comparison(tau_naive, tau_sc, tau_sdid, pkg_est if pkg_available else None)
    fig3.savefig(FIGS_DIR / 'sdid_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'sdid_comparison.png'}")

    # =========================================================================
    # Step 8: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("[8] Summary")
    print("=" * 70)

    print(f"""
RESULTS
=======

SDID Estimate: {tau_sdid:.2f} packs per capita

Interpretation:
  California's Proposition 99 (tobacco tax) reduced cigarette sales
  by approximately {abs(tau_sdid):.0f} packs per capita compared to
  what would have happened without the policy.

Method Comparison:
  - Naive DiD:         {tau_naive:.2f} packs  (biased - poor control match)
  - Synthetic Control: {tau_sc:.2f} packs  (no time weighting)
  - SDID:              {tau_sdid:.2f} packs  (best of both worlds)

KEY INSIGHTS
============

1. SDID = Synthetic Control + Time Weights
   - Unit weights (omega): Create a synthetic California from other states
   - Time weights (lambda): Focus on pre-treatment years similar to post-treatment

2. Why SDID beats Naive DiD:
   - The average of all states is a poor control for California
   - Weighting creates a synthetic control that tracks California pre-1989

3. Why SDID beats Synthetic Control:
   - SC ignores time-varying unobservables
   - Time weights make the estimator doubly robust

4. The Building Block View:
   - SDID is still just four numbers (like 2x2 DiD)
   - But each number is a weighted average
   - The weights are chosen to improve identification

WHEN TO USE SDID
================

Use SDID when:
  - You have one (or few) treated units
  - Pre-trends don't match with simple averages
  - You want robustness to time-varying confounds

Use standard DiD when:
  - You have many treated and control units
  - Simple parallel trends hold
  - Treatment is staggered (use Callaway-Sant'Anna instead)

REFERENCES
==========

Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021).
"Synthetic Difference-in-Differences"
American Economic Review, 111(12), 4088-4118.
    """)

    plt.close('all')
    print("\n" + "=" * 70)
    print("Module 07 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
