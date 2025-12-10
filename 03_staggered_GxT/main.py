"""
Module 03: Staggered Adoption (G×T Design)

This script demonstrates:
1. Manual calculation of ATT(g,t) building blocks (Callaway-Sant'Anna)
2. Aggregation to event study and simple ATT
3. Why TWFE fails with staggered treatment (Goodman-Bacon decomposition intuition)

Dataset: mpdta - Minimum wage panel data
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from utils import load_mpdta, COLORS

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def calculate_att_gt_manual(df, g, t, outcome_col='lemp',
                            cohort_col='first_treat', time_col='year'):
    """
    Manually calculate ATT(g,t) for cohort g at time t.

    Building block of the Callaway-Sant'Anna estimator.

    Control group: Not-yet-treated units (includes never-treated)
    Reference period: g - 1
    """
    # Reference year (baseline before treatment for cohort g)
    t_ref = g - 1

    # Check if reference year exists in data
    if t_ref not in df[time_col].unique():
        return np.nan, 0, 0

    # Slice to relevant time periods (t and t_ref)
    df_slice = df[df[time_col].isin([t, t_ref])].copy()

    # Treated group: units in cohort g
    treated_mask = (df_slice[cohort_col] == g)

    # Control group: Not yet treated by time t
    # (first_treat > t OR never treated (0))
    control_mask = (df_slice[cohort_col] > t) | (df_slice[cohort_col] == 0)

    # Extract Outcomes
    # Treated @ t
    y_treat_t = df_slice.loc[treated_mask & (df_slice[time_col] == t), outcome_col]
    # Treated @ ref
    y_treat_ref = df_slice.loc[treated_mask & (df_slice[time_col] == t_ref), outcome_col]
    # Control @ t
    y_control_t = df_slice.loc[control_mask & (df_slice[time_col] == t), outcome_col]
    # Control @ ref
    y_control_ref = df_slice.loc[control_mask & (df_slice[time_col] == t_ref), outcome_col]

    if len(y_treat_t) == 0 or len(y_control_t) == 0:
        return np.nan, 0, 0

    # 2x2 DiD Calculation
    delta_treat = y_treat_t.mean() - y_treat_ref.mean()
    delta_control = y_control_t.mean() - y_control_ref.mean()
    att_gt = delta_treat - delta_control

    return att_gt, len(y_treat_t), len(y_control_t)


def build_att_gt_matrix(df, outcome_col='lemp', cohort_col='first_treat', time_col='year'):
    """
    Build the complete ATT(g,t) matrix for all cohort-time pairs.
    """
    # Get unique cohorts (excluding never-treated = 0)
    cohorts = sorted([c for c in df[cohort_col].unique() if c > 0])
    years = sorted(df[time_col].unique())

    results = []

    for g in cohorts:
        # Check if reference year exists for this cohort
        if (g - 1) not in years:
            continue

        for t in years:
            # Calculate for ALL years t
            att, n_treat, n_control = calculate_att_gt_manual(
                df, g, t, outcome_col, cohort_col, time_col
            )

            # Event time: relative to treatment year
            event_time = t - g

            results.append({
                'cohort': g,
                'time': t,
                'event_time': event_time,
                'att': att,
                'n_treat': n_treat,
                'n_control': n_control
            })

    return pd.DataFrame(results)


def aggregate_by_event_time(att_matrix):
    """
    Aggregate ATT(g,t) to event-time level (simple average across cohorts).
    """
    # Filter out NaNs (cases where calculation wasn't possible)
    valid = att_matrix.dropna(subset=['att'])

    return valid.groupby('event_time').agg({
        'att': 'mean',
        'n_treat': 'sum',
        'n_control': 'sum'
    }).reset_index()


def aggregate_simple(att_matrix):
    """
    Simple aggregation: average of all post-treatment ATT(g,t).
    
    Note: This implementation uses a group-size weighted average to match
    Callaway & Sant'Anna (2021) methodology more closely.
    """
    valid = att_matrix.dropna(subset=['att'])
    # Only average post-treatment effects (event_time >= 0)
    post_treat = valid[valid['event_time'] >= 0].copy()
    
    # Weight by number of treated units in that cohort-time cell
    total_treated = post_treat['n_treat'].sum()
    weighted_att = (post_treat['att'] * post_treat['n_treat']).sum() / total_treated
    
    return weighted_att


def run_twfe_corrected(df, outcome_col='lemp', cohort_col='first_treat',
                       time_col='year', unit_col='countyreal'):
    """
    Run standard TWFE regression using Within-Transformation to avoid numerical instability.

    Y_it = α_i + γ_t + δ*D_it + ε_it

    Fix applied:
    1. Construct time-varying D_it explicitly.
    2. Demean data to absorb unit FEs before OLS.
    """
    df = df.copy()

    # 1. Create correct treatment indicator: 1 if year >= first_treat > 0
    df['D'] = ((df[time_col] >= df[cohort_col]) & (df[cohort_col] > 0)).astype(int)

    # 2. Within-transformation (Demeaning) by unit
    # This absorbs the unit fixed effects α_i
    df[f'{outcome_col}_dm'] = df.groupby(unit_col)[outcome_col].transform(lambda x: x - x.mean())
    df['D_dm'] = df.groupby(unit_col)['D'].transform(lambda x: x - x.mean())

    # 3. Regression on demeaned data (with time fixed effects)
    # Note: We use -1 to remove intercept because means are centered at 0
    formula = f'{outcome_col}_dm ~ D_dm + C({time_col}) - 1'

    # Using clustered standard errors at county level
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[unit_col]})

    return model.params['D_dm'], model.bse['D_dm']


def main():
    print("=" * 60)
    print("Module 03: Staggered Adoption (G×T Design)")
    print("Minimum Wage Panel Data (mpdta)")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1] Loading Data...")
    df = load_mpdta()
    # Ensure column names are clean (dots to underscores)
    df.columns = [c.replace('.', '_') for c in df.columns]

    print(f"Dataset: {df.shape[0]} rows, {df['countyreal'].nunique()} counties")
    print(f"Cohorts: {sorted(df['first_treat'].unique())}")

    # =========================================================================
    # Step 2: Manual ATT(g,t) Building Blocks
    # =========================================================================
    print("\n[2] Calculating ATT(g,t) Matrix (Callaway-Sant'Anna Logic)...")
    att_matrix = build_att_gt_matrix(df, outcome_col='lemp',
                                     cohort_col='first_treat', time_col='year')

    print("\nSample of ATT(g,t) Results:")
    print(att_matrix[['cohort', 'time', 'event_time', 'att']].head(6).to_string(index=False))

    # =========================================================================
    # Step 3: Aggregation
    # =========================================================================
    print("\n[3] Aggregating Results...")

    # Event Study Aggregation
    event_agg = aggregate_by_event_time(att_matrix)

    # Simple Overall ATT
    cs_att = aggregate_simple(att_matrix)
    print(f"CS Average Treatment Effect (Simple Aggregation): {cs_att:.4f}")

    # =========================================================================
    # Step 4: TWFE Comparison (With Numerical Fix)
    # =========================================================================
    print("\n[4] Running Two-Way Fixed Effects (TWFE)...")
    twfe_coef, twfe_se = run_twfe_corrected(df)

    print(f"TWFE Coefficient: {twfe_coef:.4f} (SE: {twfe_se:.4f})")

    diff = twfe_coef - cs_att
    print(f"\nDifference (TWFE - CS): {diff:.4f}")
    print("Interpretation: TWFE suggests a larger negative effect")
    print("than the rigorous CS estimator. Bias driven by")
    print("using already-treated units as controls.")

    # =========================================================================
    # Step 5: Visualization
    # =========================================================================
    print("\n[5] Creating Showcase Visualizations...")

    # plt.style.use('seaborn-v0_8-whitegrid') # Already set in utils

    # --- Plot 1: The ATT(g,t) Heatmap ---
    fig_heat, ax_heat = plt.subplots(figsize=(10, 6))

    # Pivot for heatmap
    pivot_att = att_matrix.pivot(index='cohort', columns='time', values='att')

    sns.heatmap(pivot_att, annot=True, fmt='.3f', cmap='RdBu', center=0,
                linewidths=.5, ax=ax_heat, cbar_kws={'label': 'ATT(g,t)'})

    ax_heat.set_title("The Building Blocks: ATT(g,t) Matrix", fontsize=16, fontweight='bold')
    ax_heat.set_ylabel("Treatment Cohort (g)", fontsize=12)
    ax_heat.set_xlabel("Calendar Time (t)", fontsize=12)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'att_gt_matrix.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'att_gt_matrix.png'}")

    # --- Plot 2: The Staggered Event Study ---
    fig_es, ax_es = plt.subplots(figsize=(12, 7))

    # Data for plot
    es_data = event_agg.copy()

    # Plot Zero Line
    ax_es.axhline(0, color='black', linewidth=1)
    ax_es.axvline(-0.5, color='black', linestyle='--', linewidth=1.5)

    # Plot Points (CS Estimator)
    ax_es.plot(es_data['event_time'], es_data['att'], marker='o',
               markersize=10, linewidth=2.5, color=COLORS['control'], label="Callaway-Sant'Anna")

    # Annotate Pre vs Post
    ax_es.text(-2, 0.02, "Pre-Trends\n(Parallel Check)", ha='center', color='gray')
    ax_es.text(2, -0.04, "Treatment Effects\n(Dynamic)", ha='center', color=COLORS['control'], fontweight='bold')

    # Add TWFE line for comparison
    ax_es.axhline(twfe_coef, color=COLORS['treat'], linestyle=':', linewidth=2.5, label=f"TWFE Static ({twfe_coef:.3f})")

    # Formatting
    ax_es.set_title("Staggered Event Study: Minimum Wage on Employment", fontsize=16, fontweight='bold')
    ax_es.set_xlabel("Years Since First Treatment", fontsize=12)
    ax_es.set_ylabel("Log Employment Effect", fontsize=12)
    ax_es.legend(loc='lower left', frameon=True)
    ax_es.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Clean spines
    ax_es.spines['top'].set_visible(False)
    ax_es.spines['right'].set_visible(False)

    # Embed the Logic
    note = (
        r"Each point is an average of valid $ATT(g, g+e)$ estimates." + "\n" +
        r"Control group: Not-Yet-Treated units only."
    )
    ax_es.text(0.02, 0.95, note, transform=ax_es.transAxes, va='top',
               bbox=dict(boxstyle="round", fc="white", ec="#e5e5e5"))

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'staggered_event_study.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'staggered_event_study.png'}")

    plt.close('all')
    print("\n" + "=" * 60)
    print("Module 03 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
