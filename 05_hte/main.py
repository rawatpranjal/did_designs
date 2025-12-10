"""
Module 05: Heterogeneous Treatment Effects (HTE)

This script demonstrates:
1. Heterogeneous Treatment Effects (split-sample DiD)
2. Testing for effect heterogeneity across subgroups
3. Using the differences package for HTE analysis
4. Basic Triple Difference (DDD) concept

Dataset: mpdta (Minimum Wage Panel) - Real data from `did` R package

For advanced Doubly Robust DDD with covariate adjustment, see Module 06.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_mpdta, calculate_2x2_did

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def did_2x2(df, outcome_col='y', treat_col='treated', post_col='post'):
    """
    Simple 2x2 DiD calculation.
    """
    results = calculate_2x2_did(df, outcome_col, treat_col, post_col, verbose=False)
    return results['att']


def calculate_hte(df, subgroup_col, outcome_col='lemp', treat_col='treated', post_col='post'):
    """
    Calculate Heterogeneous Treatment Effects by subgroup.

    Returns ATT for each unique value of subgroup_col.
    """
    results = []

    for subgroup_val in sorted(df[subgroup_col].unique()):
        df_sub = df[df[subgroup_col] == subgroup_val].copy()
        att = did_2x2(df_sub, outcome_col, treat_col, post_col)

        # Bootstrap SE
        n_boot = 200
        boot_atts = []
        for _ in range(n_boot):
            df_boot = df_sub.sample(n=len(df_sub), replace=True)
            try:
                boot_att = did_2x2(df_boot, outcome_col, treat_col, post_col)
                boot_atts.append(boot_att)
            except:
                continue
        se = np.std(boot_atts) if boot_atts else np.nan

        results.append({
            'subgroup': subgroup_val,
            'att': att,
            'se': se,
            'n': len(df_sub)
        })

    return pd.DataFrame(results)


def calculate_ddd(df, eligible_col='eligible', outcome_col='y',
                  treat_col='treated', post_col='post'):
    """
    Calculate Triple Difference (DDD).

    DDD = DiD(eligible) - DiD(placebo)
    """
    # DiD for eligible (targeted group)
    df_eligible = df[df[eligible_col] == 1]
    did_eligible = did_2x2(df_eligible, outcome_col, treat_col, post_col)

    # DiD for placebo (not targeted)
    df_placebo = df[df[eligible_col] == 0]
    did_placebo = did_2x2(df_placebo, outcome_col, treat_col, post_col)

    # DDD
    ddd = did_eligible - did_placebo

    # Bootstrap SE for DDD
    n_boot = 500
    boot_ddd = []
    for _ in range(n_boot):
        df_boot = df.sample(n=len(df), replace=True)
        did_e = did_2x2(df_boot[df_boot[eligible_col] == 1], outcome_col, treat_col, post_col)
        did_p = did_2x2(df_boot[df_boot[eligible_col] == 0], outcome_col, treat_col, post_col)
        boot_ddd.append(did_e - did_p)
    se_ddd = np.std(boot_ddd)

    return {
        'did_eligible': did_eligible,
        'did_placebo': did_placebo,
        'ddd': ddd,
        'se': se_ddd
    }


def load_hte_data():
    """
    Load mpdta data and create subgroups for HTE analysis.

    Creates a population size subgroup (high/low population counties).

    For simple 2x2 HTE analysis, we use a simplified setup:
    - Pre: 2003-2004 (before most treatments)
    - Post: 2006-2007 (after treatment starts for most cohorts)
    - Treated: Counties that were ever treated (first_treat > 0)
    - Control: Counties that were never treated (first_treat == 0)
    """
    df = load_mpdta()

    # Clean column names (replace dots with underscores)
    df.columns = [c.replace('.', '_') for c in df.columns]

    # Create population subgroup based on median lpop
    median_lpop = df['lpop'].median()
    df['high_pop'] = (df['lpop'] >= median_lpop).astype(int)

    # Create treatment indicator: ever treated vs never treated
    df['treated'] = (df['first_treat'] > 0).astype(int)

    # Use a global pre/post split for simple 2x2 HTE
    # Pre: 2003, Post: 2007 (comparing endpoints)
    df['post'] = (df['year'] == 2007).astype(int)

    # Filter to just pre and post years for simple 2x2
    df = df[df['year'].isin([2003, 2007])].copy()

    print(f"Loaded mpdta data: {len(df)} observations")
    print(f"  - Counties: {df['countyreal'].nunique()}")
    print(f"  - Years: {sorted(df['year'].unique())}")
    print(f"  - Treatment cohorts: {sorted([c for c in df['first_treat'].unique() if c > 0])}")

    return df


def main():
    print("=" * 60)
    print("Module 05: Heterogeneous Treatment Effects (HTE)")
    print("mpdta - Minimum Wage Panel Data")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1] Loading mpdta data...")

    df = load_hte_data()

    print("\nData structure:")
    print(f"  Treated counties (ever): {df[df['treated']==1]['countyreal'].nunique()}")
    print(f"  Never-treated counties: {df[df['treated']==0]['countyreal'].nunique()}")
    print(f"  High population counties: {(df['high_pop']==1).sum()} obs")
    print(f"  Low population counties: {(df['high_pop']==0).sum()} obs")

    print("""
Dataset: mpdta from the 'did' R package
  - 500 US counties, 2003-2007
  - Outcome: log employment (lemp)
  - Treatment: minimum wage increase
  - Subgroup: county population (lpop)
    """)

    # =========================================================================
    # Step 2: Simple DiD (Pooled)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[2] Simple DiD (Pooled)")
    print("=" * 60)

    # DiD for all data
    simple_did = did_2x2(df, outcome_col='lemp', treat_col='treated', post_col='post')
    print(f"\nSimple DiD (all counties): {simple_did:.4f}")
    print("This pools all counties regardless of characteristics.")

    # =========================================================================
    # Step 3: Heterogeneous Treatment Effects
    # =========================================================================
    print("\n" + "=" * 60)
    print("[3] Heterogeneous Treatment Effects (HTE)")
    print("=" * 60)

    print("\nSplit-sample DiD by county population:")

    hte_results = calculate_hte(df, 'high_pop', outcome_col='lemp',
                                treat_col='treated', post_col='post')

    print("\n" + "-" * 60)
    for _, row in hte_results.iterrows():
        group = "High Population" if row['subgroup'] == 1 else "Low Population"
        sig = "*" if abs(row['att']) > 1.96 * row['se'] else ""
        print(f"  {group}: ATT = {row['att']:.4f} (SE: {row['se']:.4f}) {sig}")
    print("-" * 60)

    print("""
Interpretation:
  - Compare effects for high vs low population counties
  - Differences may reflect labor market structure
  - Larger counties may have more competitive labor markets
    """)

    # =========================================================================
    # Step 4: Triple Difference (DDD)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[4] Triple Difference (DDD)")
    print("=" * 60)

    print("\nDDD = DiD(high_pop) - DiD(low_pop)")
    print("      = Compare treatment effects across subgroups")

    ddd_results = calculate_ddd(df, eligible_col='high_pop', outcome_col='lemp',
                                treat_col='treated', post_col='post')

    print(f"\n" + "-" * 60)
    print(f"DiD for High Population:  {ddd_results['did_eligible']:.4f}")
    print(f"DiD for Low Population:   {ddd_results['did_placebo']:.4f}")
    print(f"-" * 60)
    print(f"DDD (High - Low Pop):     {ddd_results['ddd']:.4f} (SE: {ddd_results['se']:.4f})")
    print("-" * 60)

    print(f"""
Interpretation:
  - DDD here measures the DIFFERENCE in treatment effects
    between high and low population counties
  - A non-zero DDD indicates heterogeneous effects by county size
    """)

    # =========================================================================
    # Step 5: Manual DDD Decomposition (8 Cells)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[5] DDD Decomposition (8 Cells)")
    print("=" * 60)

    # Calculate all 8 cell means
    cells = {}
    for treated in [0, 1]:
        for post in [0, 1]:
            for high_pop in [0, 1]:
                mask = (df['treated'] == treated) & (df['post'] == post) & (df['high_pop'] == high_pop)
                mean_y = df.loc[mask, 'lemp'].mean()
                key = f"T{treated}_P{post}_H{high_pop}"
                cells[key] = mean_y

    print("\n8-Cell Means Table:")
    print("-" * 70)
    print(f"{'Group':<20} {'Never Treated':<25} {'Ever Treated':<25}")
    print(f"{'':20} {'Pre':>10} {'Post':>10}   {'Pre':>10} {'Post':>10}")
    print("-" * 70)
    print(f"{'Low Population':<20} {cells['T0_P0_H0']:>10.4f} {cells['T0_P1_H0']:>10.4f}   "
          f"{cells['T1_P0_H0']:>10.4f} {cells['T1_P1_H0']:>10.4f}")
    print(f"{'High Population':<20} {cells['T0_P0_H1']:>10.4f} {cells['T0_P1_H1']:>10.4f}   "
          f"{cells['T1_P0_H1']:>10.4f} {cells['T1_P1_H1']:>10.4f}")
    print("-" * 70)

    # Manual calculation
    did_h_treat = (cells['T1_P1_H1'] - cells['T1_P0_H1'])
    did_h_ctrl = (cells['T0_P1_H1'] - cells['T0_P0_H1'])
    did_high_manual = did_h_treat - did_h_ctrl

    did_l_treat = (cells['T1_P1_H0'] - cells['T1_P0_H0'])
    did_l_ctrl = (cells['T0_P1_H0'] - cells['T0_P0_H0'])
    did_low_manual = did_l_treat - did_l_ctrl

    ddd_manual = did_high_manual - did_low_manual

    print(f"\nManual Calculation:")
    print(f"  DiD(High Pop) = ({cells['T1_P1_H1']:.4f} - {cells['T1_P0_H1']:.4f}) - "
          f"({cells['T0_P1_H1']:.4f} - {cells['T0_P0_H1']:.4f}) = {did_high_manual:.4f}")
    print(f"  DiD(Low Pop)  = ({cells['T1_P1_H0']:.4f} - {cells['T1_P0_H0']:.4f}) - "
          f"({cells['T0_P1_H0']:.4f} - {cells['T0_P0_H0']:.4f}) = {did_low_manual:.4f}")
    print(f"  DDD = {did_high_manual:.4f} - {did_low_manual:.4f} = {ddd_manual:.4f}")

    # =========================================================================
    # Step 6: Using differences Package (if available)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[6] Using 'differences' Package for HTE")
    print("=" * 60)

    try:
        from differences import ATTgt, simulate_data

        # Generate data with multiple samples (for HTE)
        panel_data = simulate_data(samples=2)

        att_gt = ATTgt(data=panel_data, cohort_name='cohort')
        results = att_gt.fit(formula='y', split_sample_by='samples')

        print("\nATT by sample (from 'differences' package):")
        print(results)

        # Aggregate
        simple_agg = att_gt.aggregate('simple')
        print("\nSimple aggregation by sample:")
        print(simple_agg)

    except ImportError:
        print("\n'differences' package not installed.")
        print("Install with: pip install differences")
        print("Showing manual calculations only.")
    except Exception as e:
        print(f"\n'differences' package encountered an error: {e}")
        print("This may be due to numpy/linearmodels version incompatibility.")
        print("Showing manual calculations only.")

    # =========================================================================
    # Step 7: Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("[7] Creating Visualizations")
    print("=" * 60)

    # Plot 1: Trends by Group
    fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (pop_val, title) in zip(axes, [(0, 'Low Population'), (1, 'High Population')]):
        df_sub = df[df['high_pop'] == pop_val]

        # Treated counties
        treat_means = df_sub[df_sub['treated'] == 1].groupby('year')['lemp'].mean()
        ax.plot(treat_means.index, treat_means.values, 'r-o', linewidth=2,
                markersize=8, label='Ever Treated')

        # Control counties
        ctrl_means = df_sub[df_sub['treated'] == 0].groupby('year')['lemp'].mean()
        ax.plot(ctrl_means.index, ctrl_means.values, 'b-o', linewidth=2,
                markersize=8, label='Never Treated')

        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Log Employment', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig1.suptitle('Trends by Population Size\n(HTE compares effects across county types)', fontsize=14, y=1.02)
    fig1.tight_layout()
    fig1.savefig(FIGS_DIR / 'hte_trends.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'hte_trends.png'}")

    # Plot 2: HTE Comparison Bar Chart
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    groups = ['Low Population', 'High Population', 'DDD\n(Difference)']
    values = [ddd_results['did_placebo'], ddd_results['did_eligible'], ddd_results['ddd']]
    colors = ['gray', 'steelblue', 'darkgreen']

    bars = ax2.bar(groups, values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

    ax2.axhline(y=0, color='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.annotate(f'{val:.4f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5 if height >= 0 else -15),
                    textcoords='offset points',
                    ha='center', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Treatment Effect', fontsize=12)
    ax2.set_title('HTE Decomposition\nDifference in effects between high and low population counties',
                  fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / 'hte_decomposition.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'hte_decomposition.png'}")

    # Plot 3: HTE Forest Plot
    fig3, ax3 = plt.subplots(figsize=(10, 5))

    y_pos = [1, 0]
    atts = [hte_results.loc[hte_results['subgroup']==1, 'att'].values[0],
            hte_results.loc[hte_results['subgroup']==0, 'att'].values[0]]
    ses = [hte_results.loc[hte_results['subgroup']==1, 'se'].values[0],
           hte_results.loc[hte_results['subgroup']==0, 'se'].values[0]]
    labels = ['High Population', 'Low Population']

    # Plot points and CIs
    for y, att, se, label in zip(y_pos, atts, ses, labels):
        ax3.errorbar(att, y, xerr=1.96*se, fmt='o', markersize=10,
                    capsize=5, linewidth=2, label=label)

    ax3.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax3.set_xlabel('Treatment Effect (ATT)', fontsize=12)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels)
    ax3.set_title('Heterogeneous Treatment Effects\n(Split-Sample DiD)', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')

    fig3.tight_layout()
    fig3.savefig(FIGS_DIR / 'hte_forest.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'hte_forest.png'}")

    # =========================================================================
    # Step 8: Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("[8] Summary")
    print("=" * 60)

    print(f"""
HETEROGENEOUS TREATMENT EFFECTS (HTE)
=====================================
High Population:  ATT = {hte_results.loc[hte_results['subgroup']==1, 'att'].values[0]:.4f}
Low Population:   ATT = {hte_results.loc[hte_results['subgroup']==0, 'att'].values[0]:.4f}

HTE reveals how treatment effects vary across county population sizes.

TRIPLE DIFFERENCE (DDD)
=======================
DiD (High Pop):     {ddd_results['did_eligible']:.4f}
DiD (Low Pop):      {ddd_results['did_placebo']:.4f}
DDD (Difference):   {ddd_results['ddd']:.4f}

The DDD measures whether treatment effects differ significantly
between high and low population counties.

KEY INSIGHTS
============
1. HTE: Run DiD separately by subgroup, compare effects
2. DDD: Difference between subgroup DiDs
3. Both methods are just combinations of 2×2 comparisons
4. Use HTE to understand effect heterogeneity

WHEN TO USE EACH
================
- HTE: When you want to know if effects vary by characteristics
- DDD: When you have a within-unit control group
- For Doubly Robust DDD with covariates, see Module 06
    """)

    plt.close('all')
    print("\n" + "=" * 60)
    print("Module 05 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
