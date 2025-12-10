"""
Module 01: The Canonical 2x2 Design

This script demonstrates that DiD is just "Four Numbers":
1. Manual calculation using group means
2. OLS regression with interaction term
3. Proof that they are numerically identical

Dataset: Card & Krueger (1994) - NJ/PA Minimum Wage Study
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

from utils import load_card_krueger

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("Module 01: The Canonical 2x2 Design")
    print("Card & Krueger (1994) - NJ/PA Minimum Wage Study")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load the Data
    # =========================================================================
    print("\n[1] Loading Card & Krueger data...")
    df = load_card_krueger()

    # Handle missing values explicitly (ensures manual calc matches regression)
    initial_len = len(df)
    df = df.dropna(subset=['fte', 'treated', 'post'])
    if initial_len - len(df) > 0:
        print(f"Dropped {initial_len - len(df)} rows with missing values.")

    print(f"\nDataset shape: {df.shape}")
    print(f"\nSample by group and time:")
    print(df.groupby(['treated', 'post']).size().unstack())

    # =========================================================================
    # Step 2: Manual DiD Calculation (The "Four Numbers" Approach)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[2] Manual DiD Calculation: The 'Four Numbers' Approach")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # THE CORE OF DiD: Four group means and some subtraction
    # -------------------------------------------------------------------------

    # Control Group (PA = 0)
    mu_control_pre  = df.loc[(df['treated']==0) & (df['post']==0), 'fte'].mean()
    mu_control_post = df.loc[(df['treated']==0) & (df['post']==1), 'fte'].mean()

    # Treatment Group (NJ = 1)
    mu_treat_pre    = df.loc[(df['treated']==1) & (df['post']==0), 'fte'].mean()
    mu_treat_post   = df.loc[(df['treated']==1) & (df['post']==1), 'fte'].mean()

    # First differences
    diff_control = mu_control_post - mu_control_pre  # Δ Control
    diff_treat   = mu_treat_post - mu_treat_pre      # Δ Treated

    # Difference-in-Differences
    att_manual = diff_treat - diff_control

    print(f"""
The Four Means:
    μ(Control, Pre)  = {mu_control_pre:.4f}   [PA, Feb 1992]
    μ(Control, Post) = {mu_control_post:.4f}   [PA, Nov 1992]
    μ(Treated, Pre)  = {mu_treat_pre:.4f}   [NJ, Feb 1992]
    μ(Treated, Post) = {mu_treat_post:.4f}   [NJ, Nov 1992]

First Differences:
    Δ Control (PA):  {mu_control_post:.4f} - {mu_control_pre:.4f} = {diff_control:+.4f}
    Δ Treated (NJ):  {mu_treat_post:.4f} - {mu_treat_pre:.4f} = {diff_treat:+.4f}

Difference-in-Differences:
    ATT = Δ Treated - Δ Control
        = {diff_treat:+.4f} - ({diff_control:+.4f})
        = {att_manual:+.4f}
    """)

    # =========================================================================
    # Step 3: OLS Regression
    # =========================================================================
    print("\n" + "=" * 60)
    print("[3] OLS Regression Verification")
    print("=" * 60)

    # Fit regression: Y = α + β*D + γ*Post + δ*(D×Post)
    # Using HC1 robust standard errors for heteroscedasticity
    model = smf.ols('fte ~ treated * post', data=df).fit(cov_type='HC1')

    print("\nRegression Results (HC1 robust SEs):")
    print("-" * 40)
    print(f"Intercept (α):        {model.params['Intercept']:.4f}")
    print(f"Treated (β):          {model.params['treated']:.4f}")
    print(f"Post (γ):             {model.params['post']:.4f}")
    print(f"Treated×Post (δ):     {model.params['treated:post']:.4f}")
    print("-" * 40)

    # =========================================================================
    # Step 4: Verify Numerical Equivalence
    # =========================================================================
    print("\n" + "=" * 60)
    print("[4] Verification: Manual = Regression")
    print("=" * 60)

    regression_att = model.params['treated:post']

    print(f"\nManual DiD ATT:     {att_manual:.6f}")
    print(f"Regression δ:       {regression_att:.6f}")
    print(f"Difference:         {abs(att_manual - regression_att):.10f}")

    if np.isclose(att_manual, regression_att, atol=1e-6):
        print("\n✓ VERIFIED: Manual calculation equals regression coefficient!")
    else:
        print("\n✗ WARNING: Values don't match (check for missing data)")

    # =========================================================================
    # Step 5: Interpret the Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("[5] Interpretation")
    print("=" * 60)

    print(f"""
The 2x2 DiD Table:
                        Pre (Feb 1992)    Post (Nov 1992)    Change
    Control (PA):       {mu_control_pre:.2f}             {mu_control_post:.2f}              {diff_control:+.2f}
    Treatment (NJ):     {mu_treat_pre:.2f}             {mu_treat_post:.2f}              {diff_treat:+.2f}
    ────────────────────────────────────────────────────────────────
    Diff-in-Diff:                                           {att_manual:+.2f}

Interpretation:
    The minimum wage increase in New Jersey was associated with a
    {att_manual:.2f} FTE increase in fast-food employment relative to
    what would have occurred under the Pennsylvania trend.

Key Assumption:
    Parallel trends - absent the minimum wage increase, NJ employment
    would have followed the same trajectory as PA employment.
    """)

    # =========================================================================
    # Step 6: Create Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("[6] Creating Visualizations")
    print("=" * 60)

    # Plot 1: The Classic 2x2 Diagram
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    times = [0, 1]
    time_labels_plot = ['Feb 1992\n(Pre)', 'Nov 1992\n(Post)']

    # Control group (PA)
    ax1.plot(times, [mu_control_pre, mu_control_post], 'b-o',
             linewidth=2, markersize=10, label='PA (Control)')

    # Treated group (NJ)
    ax1.plot(times, [mu_treat_pre, mu_treat_post], 'r-o',
             linewidth=2, markersize=10, label='NJ (Treated)')

    # Counterfactual: what would have happened to NJ under PA's trend
    counterfactual = mu_treat_pre + diff_control
    ax1.plot([0, 1], [mu_treat_pre, counterfactual], 'r--',
             linewidth=2, alpha=0.5, label='Counterfactual (NJ)')

    # Annotate the ATT with arrow
    mid_y = (mu_treat_post + counterfactual) / 2
    ax1.annotate('', xy=(1.02, mu_treat_post), xytext=(1.02, counterfactual),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax1.annotate(f'ATT = {att_manual:+.2f}', xy=(1.08, mid_y),
                fontsize=12, color='green', fontweight='bold')

    ax1.set_xticks(times)
    ax1.set_xticklabels(time_labels_plot)
    ax1.set_xlabel('Time Period', fontsize=12)
    ax1.set_ylabel('Full-Time Equivalent Employment', fontsize=12)
    ax1.set_title('The Canonical 2x2 DiD\nCard & Krueger (1994)', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.2, 1.4)

    fig1.tight_layout()
    fig1.savefig(FIGS_DIR / 'did_2x2.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'did_2x2.png'}")

    # Plot 2: Detailed comparison with means and CIs
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Calculate means and standard errors
    summary = df.groupby(['treated', 'post'])['fte'].agg(['mean', 'std', 'count'])
    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    for treat_val, color, label in [(0, 'blue', 'PA (Control)'), (1, 'red', 'NJ (Treated)')]:
        means = [summary.loc[(treat_val, t), 'mean'] for t in times]
        ses = [summary.loc[(treat_val, t), 'se'] for t in times]

        ax2.errorbar(times, means, yerr=[1.96*s for s in ses],
                    marker='o', capsize=5, linewidth=2,
                    color=color, label=label, markersize=10)

    # Add counterfactual
    ax2.plot([0, 1], [mu_treat_pre, counterfactual], 'r--',
             alpha=0.5, linewidth=2, label='Counterfactual')

    # Annotate ATT
    ax2.annotate('', xy=(1.02, mu_treat_post), xytext=(1.02, counterfactual),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax2.annotate(f'ATT = {att_manual:.2f}', xy=(1.08, mid_y),
                fontsize=12, color='green', fontweight='bold')

    ax2.set_xticks(times)
    ax2.set_xticklabels(time_labels_plot)
    ax2.set_xlabel('Time Period', fontsize=12)
    ax2.set_ylabel('Full-Time Equivalent Employment', fontsize=12)
    ax2.set_title('Card & Krueger (1994): Minimum Wage and Employment\nDifference-in-Differences Estimate', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.2, 1.4)

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / 'card_krueger_detailed.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'card_krueger_detailed.png'}")

    # =========================================================================
    # Step 7: Full Regression Output
    # =========================================================================
    print("\n" + "=" * 60)
    print("[7] Full Regression Output")
    print("=" * 60)
    print(model.summary())

    plt.close('all')
    print("\n" + "=" * 60)
    print("Module 01 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
