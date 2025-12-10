"""
Module 06: Triple Differences - The Robust Way

This script demonstrates:
1. Why naive DDD fails when covariate distributions differ
2. The Doubly Robust DDD estimator (Ortiz-Villavicencio & Sant'Anna 2025)
3. Visual comparison of bias between methods

Dataset: Simulated Maternity Mandates scenario with covariate imbalance
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# True treatment effect (for simulation)
TRUE_ATT = 5.0


def make_ddd_data(n=5000, seed=123):
    """
    Generate DDD data with covariate imbalance.

    Scenario: Maternity Mandates
    - Policy affects women (target) in treated states
    - Men serve as placebo group within states
    - Covariate X (education) affects trends
    - CRITICAL: X distribution differs between men and women in treated states

    This imbalance breaks naive DDD but robust DDD handles it.
    """
    np.random.seed(seed)

    # Generate cross-sectional data
    df = pd.DataFrame({
        'id': range(n),
        'state_treat': np.random.choice([0, 1], n),  # S: State treatment
        'is_female': np.random.choice([0, 1], n),    # Q: Target group (eligibility)
    })

    # Covariate X (e.g., Education level: 0=Low, 1=High)
    # CRITICAL: X is correlated with State AND Gender
    # Women in Treated States have higher education (the imbalance)
    prob_x = 0.3 + 0.35 * df['state_treat'] * df['is_female']
    df['X'] = np.random.binomial(1, prob_x)

    # Expand to panel (Pre/Post)
    df_pre = df.copy()
    df_pre['post'] = 0
    df_post = df.copy()
    df_post['post'] = 1
    df = pd.concat([df_pre, df_post]).reset_index(drop=True)

    # Generate outcome Y
    # Key: Trends depend on X (parallel trends only conditional on X)
    base = 10.0

    # Time trend depends on X
    trend = 2.0 * df['X'] * df['post']

    # Treatment effect: only for women in treated states, post-treatment
    treatment_effect = TRUE_ATT * df['state_treat'] * df['is_female'] * df['post']

    # Add noise
    noise = np.random.normal(0, 1, len(df))

    df['y'] = base + trend + treatment_effect + noise

    return df


def show_covariate_imbalance(df):
    """
    Display the covariate imbalance that breaks naive DDD.
    """
    # Get pre-period data only (to show baseline characteristics)
    df_pre = df[df['post'] == 0].copy()

    # Calculate mean X by group
    groups = df_pre.groupby(['state_treat', 'is_female'])['X'].mean().unstack()
    groups.index = ['Control State', 'Treated State']
    groups.columns = ['Men', 'Women']

    print("\nMean Covariate X (Education) by Group:")
    print("-" * 50)
    print(groups.round(3))
    print("-" * 50)

    # Calculate the imbalance
    imbalance = groups.loc['Treated State', 'Women'] - groups.loc['Treated State', 'Men']
    print(f"\nImbalance in Treated State: Women - Men = {imbalance:.3f}")
    print("(Women are more educated than Men in treated states)")

    return groups


def naive_ddd_ols(df):
    """
    Naive DDD using 3-way fixed effects OLS.

    This is the standard approach that fails with covariate imbalance.
    """
    # Standard triple interaction
    model = smf.ols("y ~ state_treat * is_female * post + X", data=df).fit()

    return model.params['state_treat:is_female:post']


def naive_ddd_manual(df):
    """
    Naive DDD calculated manually as DiD_women - DiD_men.

    This shows explicitly what the OLS is doing.
    """
    # DiD for women
    df_women = df[df['is_female'] == 1]
    did_women = (
        (df_women[(df_women['state_treat']==1) & (df_women['post']==1)]['y'].mean() -
         df_women[(df_women['state_treat']==1) & (df_women['post']==0)]['y'].mean()) -
        (df_women[(df_women['state_treat']==0) & (df_women['post']==1)]['y'].mean() -
         df_women[(df_women['state_treat']==0) & (df_women['post']==0)]['y'].mean())
    )

    # DiD for men
    df_men = df[df['is_female'] == 0]
    did_men = (
        (df_men[(df_men['state_treat']==1) & (df_men['post']==1)]['y'].mean() -
         df_men[(df_men['state_treat']==1) & (df_men['post']==0)]['y'].mean()) -
        (df_men[(df_men['state_treat']==0) & (df_men['post']==1)]['y'].mean() -
         df_men[(df_men['state_treat']==0) & (df_men['post']==0)]['y'].mean())
    )

    return did_women - did_men, did_women, did_men


def robust_ddd(df):
    """
    Doubly Robust DDD estimator (Ortiz-Villavicencio & Sant'Anna 2025).

    Key insight: Instead of 2 DiDs, we need 3 comparisons.
    All counterfactuals are evaluated at the TARGET group's covariate distribution.
    """
    # Calculate change in Y (Delta Y = Y_post - Y_pre)
    df_wide = df.pivot_table(index='id', columns='post', values='y').reset_index()
    df_wide.columns = ['id', 'y_pre', 'y_post']
    df_wide['dy'] = df_wide['y_post'] - df_wide['y_pre']

    # Merge back covariates (from pre-period)
    df_pre = df[df['post'] == 0][['id', 'state_treat', 'is_female', 'X']]
    df_wide = df_wide.merge(df_pre, on='id')

    # Define the 4 groups
    # S=state_treat, Q=is_female
    mask_s0_q0 = (df_wide['state_treat'] == 0) & (df_wide['is_female'] == 0)  # Control Men
    mask_s0_q1 = (df_wide['state_treat'] == 0) & (df_wide['is_female'] == 1)  # Control Women
    mask_s1_q0 = (df_wide['state_treat'] == 1) & (df_wide['is_female'] == 0)  # Treated Men
    mask_s1_q1 = (df_wide['state_treat'] == 1) & (df_wide['is_female'] == 1)  # Treated Women (TARGET)

    # Fit outcome regression models for 3 comparison groups
    # Model: E[Delta Y | X] for each group

    # Model 1: Control State, Men (S=0, Q=0) - Base trend
    model_s0_q0 = smf.ols("dy ~ X", data=df_wide[mask_s0_q0]).fit()

    # Model 2: Control State, Women (S=0, Q=1) - Gender trend
    model_s0_q1 = smf.ols("dy ~ X", data=df_wide[mask_s0_q1]).fit()

    # Model 3: Treated State, Men (S=1, Q=0) - State shock
    model_s1_q0 = smf.ols("dy ~ X", data=df_wide[mask_s1_q0]).fit()

    # Target group: Treated State, Women (S=1, Q=1)
    target_group = df_wide[mask_s1_q1]

    # Observed change for target group
    observed_change = target_group['dy'].mean()

    # Predict counterfactuals AT THE TARGET GROUP'S COVARIATE DISTRIBUTION
    # This is the key insight from Ortiz-Villavicencio & Sant'Anna
    cf_s0_q0 = model_s0_q0.predict(target_group).mean()  # Base trend
    cf_s0_q1 = model_s0_q1.predict(target_group).mean()  # Control Women trend
    cf_s1_q0 = model_s1_q0.predict(target_group).mean()  # Treated Men trend

    # DR-DDD Formula:
    # ATT = (Observed - CF_men_treated) - (CF_women_control - CF_men_control)
    #
    # Intuition:
    # (Observed - CF_men_treated) = Effect for women vs men in treated states
    # (CF_women_control - CF_men_control) = Gender gap in control states
    # Difference removes spurious gender differences

    att_ddd = (observed_change - cf_s1_q0) - (cf_s0_q1 - cf_s0_q0)

    return att_ddd, {
        'observed': observed_change,
        'cf_men_ctrl': cf_s0_q0,
        'cf_women_ctrl': cf_s0_q1,
        'cf_men_treat': cf_s1_q0
    }


def bootstrap_se(df, estimator_func, n_boot=500, seed=42):
    """
    Bootstrap standard errors for an estimator.
    """
    np.random.seed(seed)

    # Get unique IDs
    ids = df['id'].unique()
    n_ids = len(ids)

    boot_estimates = []
    for b in range(n_boot):
        # Sample IDs with replacement
        boot_ids = np.random.choice(ids, size=n_ids, replace=True)

        # Build bootstrap sample (keeping panel structure)
        boot_frames = []
        for j, i in enumerate(boot_ids):
            subset = df[df['id'] == i].copy()
            subset['id'] = j  # Re-assign ID
            boot_frames.append(subset)

        df_boot = pd.concat(boot_frames, ignore_index=True)

        try:
            result = estimator_func(df_boot)
            if isinstance(result, tuple):
                result = result[0]
            boot_estimates.append(result)
        except:
            continue

    return np.std(boot_estimates)


def main():
    print("=" * 70)
    print("Module 06: Triple Differences - The Robust Way")
    print("Doubly Robust DDD (Ortiz-Villavicencio & Sant'Anna 2025)")
    print("=" * 70)

    # =========================================================================
    # Step 1: Generate Data with Covariate Imbalance
    # =========================================================================
    print("\n[1] Generating DDD Data with Covariate Imbalance")
    print("=" * 70)

    df = make_ddd_data(n=5000)

    print(f"\nDataset shape: {df.shape}")
    print(f"Unique individuals: {df['id'].nunique()}")
    print(f"Time periods: Pre (0), Post (1)")

    print(f"""
Simulation Setup:
  - Policy: Maternity Mandates
  - Target group: Women in treated states
  - Placebo group: Men (within-state control)
  - Covariate X: Education (affects trends)
  - TRUE TREATMENT EFFECT: {TRUE_ATT}

  KEY: Women in treated states have HIGHER education than men.
       This creates covariate imbalance that biases naive DDD.
    """)

    # =========================================================================
    # Step 2: Show the Covariate Imbalance
    # =========================================================================
    print("\n[2] Covariate Imbalance Analysis")
    print("=" * 70)

    cov_table = show_covariate_imbalance(df)

    print("""
Why this matters:
  - Trends depend on education (high education → larger trend)
  - Women in treated states are more educated
  - Naive DDD uses men's covariate distribution to predict counterfactual
  - This underestimates the counterfactual trend for women → BIAS
    """)

    # =========================================================================
    # Step 3: Naive DDD (Shows the Bias)
    # =========================================================================
    print("\n[3] Naive DDD (Biased)")
    print("=" * 70)

    # OLS approach
    naive_ols = naive_ddd_ols(df)

    # Manual approach (for decomposition)
    naive_manual, did_women, did_men = naive_ddd_manual(df)

    print(f"\nNaive DDD Decomposition:")
    print("-" * 50)
    print(f"  DiD for Women:        {did_women:.4f}")
    print(f"  DiD for Men:          {did_men:.4f}")
    print("-" * 50)
    print(f"  Naive DDD (manual):   {naive_manual:.4f}")
    print(f"  Naive DDD (OLS):      {naive_ols:.4f}")
    print("-" * 50)
    print(f"  TRUE EFFECT:          {TRUE_ATT:.4f}")
    print(f"  BIAS:                 {naive_ols - TRUE_ATT:.4f} ({(naive_ols - TRUE_ATT)/TRUE_ATT*100:.1f}%)")

    print("""
Why is naive DDD biased?
  - DiD_men uses men's covariate distribution
  - But we want counterfactual for WOMEN (different distribution)
  - The education imbalance creates spurious differences in trends
    """)

    # =========================================================================
    # Step 4: Robust DDD (Corrects the Bias)
    # =========================================================================
    print("\n[4] Doubly Robust DDD (Ortiz-Villavicencio & Sant'Anna 2025)")
    print("=" * 70)

    robust_att, components = robust_ddd(df)

    print(f"\nDR-DDD Decomposition:")
    print("-" * 50)
    print("Counterfactuals evaluated at TARGET group's covariate distribution:")
    print(f"  Observed (Women, Treated):      {components['observed']:.4f}")
    print(f"  CF Men, Control (base trend):   {components['cf_men_ctrl']:.4f}")
    print(f"  CF Women, Control (gender gap): {components['cf_women_ctrl']:.4f}")
    print(f"  CF Men, Treated (state shock):  {components['cf_men_treat']:.4f}")
    print("-" * 50)

    # Show the formula
    term1 = components['observed'] - components['cf_men_treat']
    term2 = components['cf_women_ctrl'] - components['cf_men_ctrl']

    print(f"\nFormula: ATT = (Observed - CF_men_treat) - (CF_women_ctrl - CF_men_ctrl)")
    print(f"         ATT = ({components['observed']:.4f} - {components['cf_men_treat']:.4f}) - "
          f"({components['cf_women_ctrl']:.4f} - {components['cf_men_ctrl']:.4f})")
    print(f"         ATT = {term1:.4f} - {term2:.4f}")
    print(f"         ATT = {robust_att:.4f}")
    print("-" * 50)
    print(f"  TRUE EFFECT:          {TRUE_ATT:.4f}")
    print(f"  BIAS:                 {robust_att - TRUE_ATT:.4f} ({(robust_att - TRUE_ATT)/TRUE_ATT*100:.1f}%)")

    # =========================================================================
    # Step 5: Bootstrap Standard Errors
    # =========================================================================
    print("\n[5] Bootstrap Standard Errors")
    print("=" * 70)

    print("\nComputing bootstrap SEs (this may take a moment)...")

    se_naive = bootstrap_se(df, naive_ddd_ols, n_boot=200)
    se_robust = bootstrap_se(df, lambda x: robust_ddd(x)[0], n_boot=200)

    print(f"\nStandard Errors:")
    print("-" * 50)
    print(f"  Naive DDD:   {naive_ols:.4f} (SE: {se_naive:.4f})")
    print(f"  Robust DDD:  {robust_att:.4f} (SE: {se_robust:.4f})")
    print("-" * 50)

    # =========================================================================
    # Step 6: Comparison Visualization
    # =========================================================================
    print("\n[6] Creating Visualizations")
    print("=" * 70)

    # Plot 1: Bias Comparison Bar Chart
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    methods = ['True ATT', 'Naive DDD\n(3WFE)', 'Robust DDD\n(DR)']
    estimates = [TRUE_ATT, naive_ols, robust_att]
    errors = [0, se_naive, se_robust]
    colors = ['green', 'red', 'blue']

    bars = ax1.bar(methods, estimates, yerr=[1.96*e for e in errors],
                   color=colors, alpha=0.7, edgecolor='black', linewidth=2,
                   capsize=5)

    ax1.axhline(y=TRUE_ATT, color='green', linestyle='--', linewidth=2,
                label=f'True Effect = {TRUE_ATT}')

    # Add value labels
    for bar, est, err in zip(bars, estimates, errors):
        height = bar.get_height()
        label = f'{est:.2f}'
        if err > 0:
            label += f'\n(SE: {err:.2f})'
        ax1.annotate(label,
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

    ax1.set_ylabel('Treatment Effect Estimate', fontsize=12)
    ax1.set_title('Triple Differences: Naive vs. Robust\n'
                  'Naive DDD is biased when covariate distributions differ', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, max(estimates) * 1.4)

    fig1.tight_layout()
    fig1.savefig(FIGS_DIR / 'ddd_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'ddd_comparison.png'}")

    # Plot 2: Covariate Distribution
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    df_pre = df[df['post'] == 0]

    # Left: Treated State
    ax = axes[0]
    groups_treat = df_pre[df_pre['state_treat'] == 1].groupby('is_female')['X'].mean()
    ax.bar(['Men', 'Women'], [groups_treat[0], groups_treat[1]],
           color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax.set_title('Treated State\n(Imbalanced)', fontsize=12)
    ax.set_ylabel('Mean Education (X)', fontsize=11)
    ax.set_ylim(0, 1)
    for i, v in enumerate([groups_treat[0], groups_treat[1]]):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

    # Right: Control State
    ax = axes[1]
    groups_ctrl = df_pre[df_pre['state_treat'] == 0].groupby('is_female')['X'].mean()
    ax.bar(['Men', 'Women'], [groups_ctrl[0], groups_ctrl[1]],
           color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')
    ax.set_title('Control State\n(Balanced)', fontsize=12)
    ax.set_ylabel('Mean Education (X)', fontsize=11)
    ax.set_ylim(0, 1)
    for i, v in enumerate([groups_ctrl[0], groups_ctrl[1]]):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center', fontweight='bold')

    fig2.suptitle('Covariate Imbalance: Women in Treated States Are More Educated',
                  fontsize=14, y=1.02)
    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / 'covariate_imbalance.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'covariate_imbalance.png'}")

    # Plot 3: DDD Decomposition Waterfall
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    # Create waterfall-style chart
    labels = ['Observed\n(Women, Treat)', 'CF Men, Treat\n(removes policy)',
              'CF Women, Ctrl\n(gender trend)', 'CF Men, Ctrl\n(base trend)', 'DR-DDD']
    values = [components['observed'], components['cf_men_treat'],
              components['cf_women_ctrl'], components['cf_men_ctrl'], robust_att]
    colors = ['coral', 'steelblue', 'coral', 'steelblue', 'green']

    ax3.bar(labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    for i, (label, v) in enumerate(zip(labels, values)):
        ax3.text(i, v + 0.1, f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)

    ax3.axhline(y=TRUE_ATT, color='green', linestyle='--', linewidth=2,
                label=f'True ATT = {TRUE_ATT}')
    ax3.set_ylabel('Value', fontsize=12)
    ax3.set_title('DR-DDD Decomposition\n'
                  'All counterfactuals evaluated at target group\'s covariate distribution',
                  fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    fig3.tight_layout()
    fig3.savefig(FIGS_DIR / 'ddd_decomposition.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'ddd_decomposition.png'}")

    # =========================================================================
    # Step 7: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("[7] Summary")
    print("=" * 70)

    print(f"""
RESULTS COMPARISON
==================

                        Estimate    SE      Bias
                        --------    --      ----
True Effect:            {TRUE_ATT:.4f}      -       -
Naive DDD (3WFE):       {naive_ols:.4f}    {se_naive:.4f}   {naive_ols - TRUE_ATT:+.4f} ({(naive_ols - TRUE_ATT)/TRUE_ATT*100:+.1f}%)
Robust DDD (DR):        {robust_att:.4f}    {se_robust:.4f}   {robust_att - TRUE_ATT:+.4f} ({(robust_att - TRUE_ATT)/TRUE_ATT*100:+.1f}%)

KEY INSIGHTS
============

1. NAIVE DDD IS BIASED when covariate distributions differ between
   target (women) and placebo (men) groups.

2. THE BIAS ARISES because naive DDD uses the wrong covariate
   distribution when computing counterfactuals.

3. ROBUST DDD FIXES THIS by evaluating all counterfactual predictions
   at the target group's covariate distribution.

4. THE FORMULA requires 3 comparisons (not 2):
   ATT = (Observed - CF_men_treat) - (CF_women_ctrl - CF_men_ctrl)

WHEN TO USE ROBUST DDD
======================

Use Robust DDD when:
  - Covariate distributions differ between target and placebo groups
  - Trends depend on observable covariates
  - You want protection against model misspecification

Use Naive DDD when:
  - Target and placebo groups are balanced on covariates
  - Trends don't depend on covariates
  - Quick/simple analysis is sufficient

REFERENCE
=========
Ortiz-Villavicencio, A. & Sant'Anna, P. H. C. (2025)
"Doubly Robust Difference-in-Difference-in-Differences Estimators"
    """)

    plt.close('all')
    print("\n" + "=" * 70)
    print("Module 06 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
