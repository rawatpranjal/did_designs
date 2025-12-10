"""
Module 05: Heterogeneous Treatment Effects (HTE)

This script demonstrates:
1. Simple Pooled DiD (The "Average" Effect)
2. Heterogeneous Treatment Effects via Split-Sample DiD
3. Using the differences package for HTE analysis
4. Visualizing heterogeneity with Forest Plots

Dataset: mpdta (Minimum Wage Panel) - Real data from `did` R package
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_mpdta, calculate_2x2_did, COLORS

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def did_2x2(df, outcome_col='lemp', treat_col='treated', post_col='post'):
    """
    Simple 2x2 DiD calculation wrapper.
    """
    results = calculate_2x2_did(df, outcome_col, treat_col, post_col, verbose=False)
    return results['att']


def calculate_hte(df, subgroup_col, outcome_col='lemp', treat_col='treated', post_col='post'):
    """
    Calculate Heterogeneous Treatment Effects by subgroup.
    Returns a DataFrame with ATT, SE, and N for each subgroup.
    """
    results = []

    # Iterate through unique values of the subgroup column
    for subgroup_val in sorted(df[subgroup_col].unique()):
        # Split the sample
        df_sub = df[df[subgroup_col] == subgroup_val].copy()

        # Calculate ATT for this subgroup
        att = did_2x2(df_sub, outcome_col, treat_col, post_col)

        # Simple Bootstrap Standard Error
        n_boot = 200
        boot_atts = []
        for _ in range(n_boot):
            # Resample with replacement within the subgroup
            df_boot = df_sub.sample(n=len(df_sub), replace=True)
            try:
                b_att = did_2x2(df_boot, outcome_col, treat_col, post_col)
                boot_atts.append(b_att)
            except:
                continue

        se = np.std(boot_atts) if boot_atts else 0.0

        results.append({
            'subgroup': subgroup_val,
            'att': att,
            'se': se,
            'n': len(df_sub)
        })

    return pd.DataFrame(results)


def load_hte_data():
    """
    Load mpdta data and prepare for HTE analysis.

    We create a 'high_pop' indicator to serve as our subgroup.
    We want to see if the Minimum Wage effect differs by county size.
    """
    df = load_mpdta()
    df.columns = [c.replace('.', '_') for c in df.columns]

    # Create subgroup based on median population (High vs Low)
    median_lpop = df['lpop'].median()
    df['high_pop'] = (df['lpop'] >= median_lpop).astype(int)

    # Treatment: Ever Treated (Simplification for 2x2)
    df['treated'] = (df['first_treat'] > 0).astype(int)

    # Simplification: Compare 2003 (Pre) vs 2007 (Post)
    # This turns the panel into a 2-period 2x2 setup for clear demonstration
    df = df[df['year'].isin([2003, 2007])].copy()
    df['post'] = (df['year'] == 2007).astype(int)

    return df


def main():
    print("=" * 60)
    print("Module 05: Heterogeneous Treatment Effects (HTE)")
    print("mpdta - Minimum Wage Panel Data")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1] Loading Data...")
    df = load_hte_data()
    print(f"Data: {len(df)} rows (2003 & 2007 only)")
    print(f"Subgroups: High Pop vs Low Pop counties")

    # =========================================================================
    # Step 2: Simple Pooled DiD
    # =========================================================================
    print("\n[2] Simple Pooled DiD...")
    pooled_att = did_2x2(df)
    print(f"Pooled ATT: {pooled_att:.4f}")
    print("This estimates the average effect across ALL counties.")

    # =========================================================================
    # Step 3: Heterogeneous Treatment Effects (Manual)
    # =========================================================================
    print("\n[3] Heterogeneous Treatment Effects (Split-Sample)...")

    hte_results = calculate_hte(df, 'high_pop')

    # Map binary 0/1 back to labels for display
    hte_results['label'] = hte_results['subgroup'].map({0: 'Low Population', 1: 'High Population'})

    print("\nResults by Subgroup:")
    print("-" * 60)
    for _, row in hte_results.iterrows():
        sig = "*" if abs(row['att']) > 1.96 * row['se'] else ""
        print(f"{row['label']:<15}: ATT = {row['att']:>7.4f} (SE: {row['se']:.4f}) {sig}")
    print("-" * 60)

    # Calculate difference
    att_diff = hte_results.loc[1, 'att'] - hte_results.loc[0, 'att']
    print(f"\nDifference (High - Low): {att_diff:.4f}")
    print("Does the policy affect large counties differently than small ones?")

    # =========================================================================
    # Step 4: Using differences Package (if available)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[4] Using 'differences' Package for HTE")
    print("=" * 60)

    try:
        from differences import ATTgt

        # Reload full mpdta data (not filtered) for proper staggered DiD
        # This shows HTE works even in the complex staggered setting
        df_full = load_mpdta()
        df_full.columns = [c.replace('.', '_') for c in df_full.columns]

        # Create population subgroup
        median_lpop = df_full['lpop'].median()
        df_full['high_pop'] = (df_full['lpop'] >= median_lpop).astype(int)

        # Fix for differences package: NaN for never-treated + MultiIndex
        df_full['first_treat'] = np.where(df_full['first_treat'] == 0, np.nan, df_full['first_treat'])
        df_full = df_full.set_index(['countyreal', 'year'])

        # Use ATTgt with split_sample_by
        print("Running Callaway-Sant'Anna by subgroup...")
        att_gt = ATTgt(data=df_full, cohort_name='first_treat')
        results = att_gt.fit(formula='lemp', split_sample_by='high_pop')

        print("\nATT by population size (from 'differences' package):")
        print(results)

        # Aggregate to simple ATT per group
        simple_agg = att_gt.aggregate('simple')
        print("\nSimple aggregation by subgroup:")
        print(simple_agg)

    except ImportError:
        print("\n'differences' package not installed.")
        print("Install with: pip install differences")
    except Exception as e:
        print(f"\n'differences' package encountered an error: {e}")

    # =========================================================================
    # Step 5: Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("[5] Creating Showcase Visualizations")
    print("=" * 60)
    # plt.style.use('seaborn-v0_8-whitegrid') # Already set in utils

    # --- Plot 1: Trends by Group (Visualizing the Mechanism) ---
    fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (pop_val, title) in zip(axes, [(0, 'Low Population'), (1, 'High Population')]):
        df_sub = df[df['high_pop'] == pop_val]

        # Calculate means
        treat_pre = df_sub[(df_sub['treated'] == 1) & (df_sub['post'] == 0)]['lemp'].mean()
        treat_post = df_sub[(df_sub['treated'] == 1) & (df_sub['post'] == 1)]['lemp'].mean()
        ctrl_pre = df_sub[(df_sub['treated'] == 0) & (df_sub['post'] == 0)]['lemp'].mean()
        ctrl_post = df_sub[(df_sub['treated'] == 0) & (df_sub['post'] == 1)]['lemp'].mean()

        # Plot Treated
        ax.plot([0, 1], [treat_pre, treat_post], 'o-', color=COLORS['treat'], linewidth=3, label='Treated', markersize=10)

        # Plot Control
        ax.plot([0, 1], [ctrl_pre, ctrl_post], 'o--', color=COLORS['control'], linewidth=2, label='Control', markersize=8)

        # Plot Counterfactual
        cf = treat_pre + (ctrl_post - ctrl_pre)
        ax.plot([0, 1], [treat_pre, cf], 'o:', color=COLORS['counterfactual'], alpha=0.6, label='Counterfactual')

        # Annotate ATT
        att_val = hte_results[hte_results['subgroup'] == pop_val]['att'].values[0]
        ax.annotate(f"ATT = {att_val:.3f}", xy=(1.05, (treat_post + cf) / 2),
                    color=COLORS['treat'], fontweight='bold', fontsize=12)
        ax.annotate('', xy=(1.02, treat_post), xytext=(1.02, cf),
                    arrowprops=dict(arrowstyle='<->', color=COLORS['treat'], lw=2))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pre (2003)', 'Post (2007)'])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Log Employment")
        ax.legend()
        ax.grid(True, alpha=0.3, color=COLORS['grid'])
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig1.suptitle('Heterogeneous Trends: Treatment Effects by Population Size', fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'hte_trends.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'hte_trends.png'}")

    # --- Plot 2: Forest Plot (The Standard for HTE) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    y_pos = np.arange(len(hte_results))
    atts = hte_results['att']
    errors = 1.96 * hte_results['se']
    labels = hte_results['label']

    # Plot Overall Pooled Effect Line
    ax2.axvline(pooled_att, color=COLORS['counterfactual'], linestyle='--', linewidth=1, label=f'Pooled ATT ({pooled_att:.3f})')
    ax2.axvline(0, color='black', linewidth=1)

    # Plot Points with Error Bars
    ax2.errorbar(atts, y_pos, xerr=errors, fmt='o', markersize=12,
                 color=COLORS['treat'], ecolor='black', capsize=5, linewidth=2, label='Subgroup ATT')

    # Formatting
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=12)
    ax2.set_xlabel('Treatment Effect (Log Employment)', fontsize=12)
    ax2.set_title('Forest Plot: Heterogeneous Treatment Effects', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, color=COLORS['grid'])
    
    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add text for difference
    ax2.text(pooled_att, -0.8, "Do these intervals overlap?", ha='center', color=COLORS['text'], fontstyle='italic')

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'hte_forest.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'hte_forest.png'}")

    # --- Plot 3: Decomposition Bar (The Triple Difference) ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    # Data
    att_low = hte_results.loc[0, 'att']
    att_high = hte_results.loc[1, 'att']
    diff = att_high - att_low
    
    labels = ['Low Pop\n(ATT)', 'High Pop\n(ATT)', 'Difference\n(High - Low)']
    values = [att_low, att_high, diff]
    colors = [COLORS['control'], COLORS['treat'], COLORS['counterfactual']]
    
    bars = ax3.bar(labels, values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
    ax3.axhline(0, color='black', linewidth=1)
    
    # Add values
    for bar, v in zip(bars, values):
        height = bar.get_height()
        offset = 0.005 if height > 0 else -0.015
        ax3.text(bar.get_x() + bar.get_width()/2, height + offset, f"{v:.3f}",
                 ha='center', fontweight='bold', color=COLORS['text'])
                 
    ax3.set_title('Decomposition of Heterogeneity\n(Triple Difference = Difference in ATTs)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Treatment Effect', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # Clean spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Annotation
    ax3.annotate('If this is non-zero,\neffects are heterogeneous', 
                 xy=(2, diff), xytext=(2, diff + 0.05 if diff > 0 else diff - 0.05),
                 ha='center', arrowprops=dict(arrowstyle='->', color=COLORS['text']))

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'hte_decomposition.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'hte_decomposition.png'}")

    print("\n" + "=" * 60)
    print("Module 05 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
