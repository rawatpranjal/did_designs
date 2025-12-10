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

from utils import load_card_krueger, COLORS

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
    # Step 6: The "Showcase" Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("[6] Creating The Ultimate DiD Visual")
    print("=" * 60)

    # Setup the plot style
    # plt.style.use('seaborn-v0_8-whitegrid') # Already set in utils
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- 1. Plot the Actual Data ---

    # Control Group (PA) - The Baseline
    ax.plot([0, 1], [mu_control_pre, mu_control_post], color=COLORS['control'],
            marker='o', markersize=10, linewidth=2.5, linestyle='-',
            label='Control (PA)')

    # Treatment Group (NJ) - The Actual Outcome
    ax.plot([0, 1], [mu_treat_pre, mu_treat_post], color=COLORS['treat'],
            marker='o', markersize=10, linewidth=3, linestyle='-',
            label='Treated (NJ)')

    # --- 2. Plot the Counterfactual (The "Ghost" Line) ---
    # What NJ would have looked like if it followed PA's trend
    cf_outcome = mu_treat_pre + diff_control

    ax.plot([0, 1], [mu_treat_pre, cf_outcome], color=COLORS['treat'],
            linestyle='--', linewidth=2, alpha=0.6,
            label='Counterfactual (Parallel Trends)')

    # Add a hollow circle at the counterfactual point
    ax.plot(1, cf_outcome, marker='o', markersize=10, markerfacecolor='white',
            markeredgecolor=COLORS['treat'], markeredgewidth=2)

    # --- 3. Annotate the "Effect" (The DiD) ---
    # Draw a vertical double-arrow
    ax.annotate('', xy=(1, mu_treat_post), xytext=(1, cf_outcome),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))

    # Text label for the effect
    mid_point = (mu_treat_post + cf_outcome) / 2
    ax.text(1.02, mid_point, f"ATT = {att_manual:+.2f}\n(Causal Effect)",
            color='green', va='center', fontweight='bold', fontsize=12)

    # --- 4. Annotate the Equation "In-Situ" ---
    equation = (
        r"$\hat{\delta}_{DiD} = (\bar{y}_{T,Post} - \bar{y}_{T,Pre}) - "
        r"(\bar{y}_{C,Post} - \bar{y}_{C,Pre})$"
    )
    ax.text(0.5, 0.05, equation, transform=ax.transAxes, ha='center',
            fontsize=14, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

    # --- 5. Formatting ---
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-Period\n(Feb 1992)', 'Post-Period\n(Nov 1992)'], fontsize=11)
    ax.set_ylabel('FTE Employment', fontsize=12)
    ax.set_title('The Canonical 2x2 Difference-in-Differences Design',
                 fontsize=16, fontweight='bold', pad=20)

    # Move legend to bottom
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add subtle trend annotation
    ax.annotate("Parallel Trends Assumption:\nNJ would have followed PA's slope",
                xy=(0.5, (mu_treat_pre + cf_outcome)/2), xytext=(0.2, mu_treat_pre - 1.5),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="gray"),
                color="gray", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / 'did_showcase.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'did_showcase.png'}")

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
