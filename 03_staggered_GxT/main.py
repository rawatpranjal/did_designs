"""
Module 03: Staggered Adoption (G×T Design)

This script demonstrates:
1. Manual calculation of ATT(g,t) building blocks
2. Aggregation to event study and simple ATT
3. Comparison with the differences package
4. Why TWFE fails with staggered treatment

Dataset: mpdta - Minimum wage panel data
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

from utils import load_mpdta

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def calculate_att_gt_manual(df, g, t, outcome_col='lemp',
                            cohort_col='first.treat', time_col='year'):
    """
    Manually calculate ATT(g,t) for cohort g at time t.

    Building block of the Callaway-Sant'Anna estimator.

    Control group: Not-yet-treated units (includes never-treated)
    Reference period: g - 1
    """
    # Reference year (baseline before treatment)
    t_ref = g - 1

    # Check if reference year exists
    if t_ref not in df[time_col].unique():
        return np.nan, 0, 0

    # Slice to relevant time periods
    df_slice = df[df[time_col].isin([t, t_ref])].copy()

    # Treated group: cohort g
    treated_mask = (df_slice[cohort_col] == g)

    # Control group: Not yet treated by time t
    # (first.treat > t OR never treated)
    never_treat_val = 0  # In mpdta, 0 means never treated
    control_mask = (df_slice[cohort_col] > t) | (df_slice[cohort_col] == never_treat_val)

    # Get outcome values
    # Treated at time t
    y_treat_t = df_slice.loc[treated_mask & (df_slice[time_col] == t), outcome_col]
    # Treated at reference
    y_treat_ref = df_slice.loc[treated_mask & (df_slice[time_col] == t_ref), outcome_col]
    # Control at time t
    y_control_t = df_slice.loc[control_mask & (df_slice[time_col] == t), outcome_col]
    # Control at reference
    y_control_ref = df_slice.loc[control_mask & (df_slice[time_col] == t_ref), outcome_col]

    if len(y_treat_t) == 0 or len(y_control_t) == 0:
        return np.nan, 0, 0

    # Calculate DiD
    delta_treat = y_treat_t.mean() - y_treat_ref.mean()
    delta_control = y_control_t.mean() - y_control_ref.mean()
    att_gt = delta_treat - delta_control

    return att_gt, len(y_treat_t), len(y_control_t)


def build_att_gt_matrix(df, outcome_col='lemp', cohort_col='first.treat', time_col='year'):
    """
    Build the complete ATT(g,t) matrix for all cohort-time pairs.

    Returns DataFrame with columns: cohort, time, att, n_treat, n_control
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
            # Calculate for ALL years t (pre and post)
            # Pre-periods (t < g) serve as placebo tests for parallel trends
            att, n_treat, n_control = calculate_att_gt_manual(
                df, g, t, outcome_col, cohort_col, time_col
            )
            event_time = t - g  # Relative time since treatment

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
    Aggregate ATT(g,t) to event-time level.

    For each event time e, average ATT across all cohorts:
        ATT(e) = mean of ATT(g, g+e) over all g
    """
    return att_matrix.groupby('event_time').agg({
        'att': 'mean',
        'n_treat': 'sum',
        'n_control': 'sum'
    }).reset_index()


def aggregate_simple(att_matrix):
    """
    Simple aggregation: average all ATT(g,t).
    """
    return att_matrix['att'].mean()


def aggregate_by_cohort(att_matrix):
    """
    Aggregate by cohort: average ATT over time for each cohort.
    """
    return att_matrix.groupby('cohort').agg({
        'att': 'mean',
        'n_treat': 'sum'
    }).reset_index()


def run_twfe(df, outcome_col='lemp', treat_col='treat', time_col='year'):
    """
    Run standard TWFE regression for comparison.

    Y_it = α_i + γ_t + δ*D_it + ε_it
    """
    # Create unit fixed effects
    formula = f'{outcome_col} ~ {treat_col} + C(countyreal) + C({time_col})'
    model = smf.ols(formula, data=df).fit()
    return model.params[treat_col], model.bse[treat_col]


def main():
    print("=" * 60)
    print("Module 03: Staggered Adoption (G×T Design)")
    print("Minimum Wage Panel Data (mpdta)")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load and Explore Data
    # =========================================================================
    print("\n[1] Loading mpdta data...")
    df = load_mpdta()

    # Rename column for easier access
    df.columns = [c.replace('.', '_') for c in df.columns]

    print(f"\nDataset shape: {df.shape}")
    print(f"Years: {sorted(df['year'].unique())}")
    print(f"Number of counties: {df['countyreal'].nunique()}")

    # Treatment cohorts
    print("\nTreatment Cohorts:")
    cohort_counts = df.groupby('first_treat')['countyreal'].nunique()
    for cohort, count in cohort_counts.items():
        label = "Never treated" if cohort == 0 else f"Treated in {int(cohort)}"
        print(f"  {label}: {count} counties")

    # =========================================================================
    # Step 2: Manual ATT(g,t) Calculation
    # =========================================================================
    print("\n" + "=" * 60)
    print("[2] Manual ATT(g,t) Calculation")
    print("=" * 60)

    print("\nBuilding ATT(g,t) matrix...")
    print("For each cohort g and ALL time periods t:")
    print("  (Pre-periods t < g test parallel trends)")
    print("  ATT(g,t) = [ΔY(cohort g)] - [ΔY(not-yet-treated)]")
    print("  Reference period: g - 1")
    print("  Control: Units with first_treat > t or never treated")

    att_matrix = build_att_gt_matrix(df, outcome_col='lemp',
                                     cohort_col='first_treat', time_col='year')

    print("\n" + "-" * 60)
    print("ATT(g,t) Matrix:")
    print("-" * 60)
    print(att_matrix.to_string(index=False))

    # =========================================================================
    # Step 3: Aggregation
    # =========================================================================
    print("\n" + "=" * 60)
    print("[3] Aggregation")
    print("=" * 60)

    # Simple average
    simple_att = aggregate_simple(att_matrix)
    print(f"\nSimple ATT (average of all ATT(g,t)): {simple_att:.4f}")

    # By event time
    event_agg = aggregate_by_event_time(att_matrix)
    print("\nAggregated by Event Time:")
    print(event_agg.to_string(index=False))

    # By cohort
    cohort_agg = aggregate_by_cohort(att_matrix)
    print("\nAggregated by Cohort:")
    print(cohort_agg.to_string(index=False))

    # =========================================================================
    # Step 4: TWFE Comparison (Demonstrates the Problem)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[4] TWFE Comparison (Why It Can Fail)")
    print("=" * 60)

    twfe_coef, twfe_se = run_twfe(df)

    print(f"\nTWFE coefficient: {twfe_coef:.4f} (SE: {twfe_se:.4f})")
    print(f"Callaway-Sant'Anna (simple): {simple_att:.4f}")
    print(f"\nDifference: {twfe_coef - simple_att:.4f}")

    print("""
Note: TWFE and CS can differ because:
1. TWFE uses already-treated as controls (bad comparisons)
2. TWFE implicitly weights cohorts by sample size
3. With homogeneous effects, they may be similar
4. With heterogeneous effects, TWFE can be severely biased
    """)

    # =========================================================================
    # Step 5: Use differences Package
    # =========================================================================
    print("\n" + "=" * 60)
    print("[5] Using 'differences' Package")
    print("=" * 60)

    try:
        from differences import ATTgt

        # Prepare data for differences package
        df_pkg = df.copy()
        df_pkg['cohort'] = df_pkg['first_treat'].replace(0, np.nan)  # Never-treated as NaN

        att_gt = ATTgt(data=df_pkg, cohort_name='cohort')
        results = att_gt.fit('lemp ~ lpop')

        print("\nATTgt Results from 'differences' package:")
        print(results)

        # Aggregate by event
        event_results = att_gt.aggregate('event')
        print("\nEvent-Time Aggregation:")
        print(event_results)

        # Simple aggregation
        simple_results = att_gt.aggregate('simple')
        print("\nSimple Aggregation:")
        print(simple_results)

    except ImportError:
        print("\n'differences' package not installed.")
        print("Install with: pip install differences")
        print("Showing manual calculations only.")
    except Exception as e:
        print(f"\n'differences' package encountered an error: {e}")
        print("This may be due to numpy/linearmodels version incompatibility.")
        print("Showing manual calculations only.")

    # =========================================================================
    # Step 6: Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("[6] Creating Visualizations")
    print("=" * 60)

    # Plot 1: Raw trends by cohort
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    for cohort in sorted(df['first_treat'].unique()):
        cohort_data = df[df['first_treat'] == cohort].groupby('year')['lemp'].mean()
        label = "Never Treated" if cohort == 0 else f"Cohort {int(cohort)}"
        style = '--' if cohort == 0 else '-'
        ax1.plot(cohort_data.index, cohort_data.values, style,
                 marker='o', linewidth=2, label=label)

    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Log Employment', fontsize=12)
    ax1.set_title('Raw Trends by Treatment Cohort', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(FIGS_DIR / 'raw_trends_by_cohort.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'raw_trends_by_cohort.png'}")

    # Plot 2: Event Study from Manual Calculation
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    x = event_agg['event_time']
    y = event_agg['att']

    ax2.scatter(x, y, color='blue', s=100, zorder=5)
    ax2.plot(x, y, color='blue', linewidth=2, alpha=0.7)

    # Reference lines
    ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax2.axvline(x=-0.5, color='red', linewidth=2, linestyle='--', label='Treatment')

    ax2.set_xlabel('Event Time (Years Since Treatment)', fontsize=12)
    ax2.set_ylabel('ATT', fontsize=12)
    ax2.set_title('Staggered DiD: Event Study\n(Manual Callaway-Sant\'Anna Implementation)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Annotate
    for i, row in event_agg.iterrows():
        ax2.annotate(f'{row["att"]:.3f}',
                    xy=(row['event_time'], row['att']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', fontsize=9)

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / 'staggered_event_study.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'staggered_event_study.png'}")

    # Plot 3: ATT(g,t) Heatmap
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    pivot = att_matrix.pivot(index='cohort', columns='time', values='att')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                ax=ax3, cbar_kws={'label': 'ATT(g,t)'})

    ax3.set_title('ATT(g,t) Matrix\n(Building Blocks of Staggered DiD)', fontsize=14)
    ax3.set_xlabel('Calendar Time (t)', fontsize=12)
    ax3.set_ylabel('Treatment Cohort (g)', fontsize=12)

    fig3.tight_layout()
    fig3.savefig(FIGS_DIR / 'att_gt_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'att_gt_matrix.png'}")

    # =========================================================================
    # Step 7: Interpretation
    # =========================================================================
    print("\n" + "=" * 60)
    print("[7] Interpretation")
    print("=" * 60)

    print(f"""
RESULTS SUMMARY
===============

ATT(g,t) Building Blocks:
    We calculated {len(att_matrix)} group-time treatment effects.
    Each ATT(g,t) is a 2×2 DiD comparing:
        - Treated: Cohort g
        - Control: Not-yet-treated units
        - Pre: Year g-1
        - Post: Year t

Aggregations:
    - Simple ATT: {simple_att:.4f}
      (Average effect across all cohort-time pairs)

    - By Event Time: See event study plot
      (How effects evolve after treatment)

    - By Cohort: See cohort summary
      (Which cohorts have larger effects)

TWFE vs. Callaway-Sant'Anna:
    - TWFE: {twfe_coef:.4f}
    - CS:   {simple_att:.4f}

KEY INSIGHT:
    The Callaway-Sant'Anna estimator avoids "bad comparisons"
    by never using already-treated units as controls.
    Each ATT(g,t) is constructed using only:
        - The specific cohort g
        - Units not yet treated by time t

    This is just a disciplined way of computing 2×2 DiDs!
    """)

    plt.close('all')
    print("\n" + "=" * 60)
    print("Module 03 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
