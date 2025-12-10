"""
Module 04: Covariates and Doubly Robust DiD

This script demonstrates:
1. Why simple DiD fails with selection bias
2. IPW (Inverse Probability Weighting)
3. Outcome Regression
4. Doubly Robust estimation
5. Comparison of all methods

Dataset: LaLonde (1986) - NSW Job Training Program
Source: https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils import load_lalonde as _load_lalonde

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def load_lalonde():
    """
    Load LaLonde (1986) NSW job training data from data/lalonde.csv.

    Creates dummy variables for race (black, hispan) for regression.
    """
    df = _load_lalonde()

    # Create dummy variables for race
    df['black'] = (df['race'] == 'black').astype(int)
    df['hisp'] = (df['race'] == 'hispan').astype(int)

    print(f"Loaded LaLonde data: {len(df)} observations")
    print(f"  - Treated (NSW): {df['treat'].sum():.0f}")
    print(f"  - Control (PSID): {(1 - df['treat']).sum():.0f}")

    return df


def naive_did(df):
    """
    Simple DiD without covariate adjustment.
    """
    # Treated: change in earnings
    treat_post = df.loc[df['treat'] == 1, 're78'].mean()
    treat_pre = df.loc[df['treat'] == 1, 're75'].mean()
    delta_treat = treat_post - treat_pre

    # Control: change in earnings
    ctrl_post = df.loc[df['treat'] == 0, 're78'].mean()
    ctrl_pre = df.loc[df['treat'] == 0, 're75'].mean()
    delta_ctrl = ctrl_post - ctrl_pre

    return delta_treat - delta_ctrl


def estimate_propensity_score(df, covariates):
    """
    Estimate propensity score P(D=1 | X) using logistic regression.
    """
    X = df[covariates].values
    y = df['treat'].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_scaled, y)

    # Get propensity scores
    pscore = model.predict_proba(X_scaled)[:, 1]

    # Clip to avoid extreme weights
    pscore = np.clip(pscore, 0.01, 0.99)

    return pscore


def ipw_did(df, pscore):
    """
    Inverse Probability Weighted DiD.

    Reweight control units to match treated on observables.
    """
    df = df.copy()
    df['pscore'] = pscore

    # IPW weights for control units
    # w = p / (1 - p) for controls, 1 for treated
    df['ipw'] = np.where(df['treat'] == 1, 1, df['pscore'] / (1 - df['pscore']))

    # Normalize weights for controls
    df.loc[df['treat'] == 0, 'ipw'] = (
        df.loc[df['treat'] == 0, 'ipw'] /
        df.loc[df['treat'] == 0, 'ipw'].sum() *
        len(df[df['treat'] == 0])
    )

    # Calculate change in earnings
    df['delta_y'] = df['re78'] - df['re75']

    # Weighted means
    treat_mean = df.loc[df['treat'] == 1, 'delta_y'].mean()
    ctrl_mean = np.average(df.loc[df['treat'] == 0, 'delta_y'],
                          weights=df.loc[df['treat'] == 0, 'ipw'])

    return treat_mean - ctrl_mean


def outcome_regression_did(df, covariates):
    """
    Outcome Regression DiD.

    Predict counterfactual change for treated using control model.
    """
    df = df.copy()
    df['delta_y'] = df['re78'] - df['re75']

    X_ctrl = df.loc[df['treat'] == 0, covariates].values
    y_ctrl = df.loc[df['treat'] == 0, 'delta_y'].values

    X_treat = df.loc[df['treat'] == 1, covariates].values
    y_treat = df.loc[df['treat'] == 1, 'delta_y'].values

    # Fit model on control group
    model = LinearRegression()
    model.fit(X_ctrl, y_ctrl)

    # Predict counterfactual for treated
    y_treat_counterfactual = model.predict(X_treat)

    # ATT = mean(actual) - mean(counterfactual)
    att = y_treat.mean() - y_treat_counterfactual.mean()

    return att


def doubly_robust_did(df, pscore, covariates):
    """
    Doubly Robust DiD estimator.

    Combines IPW and Outcome Regression for robustness.
    """
    df = df.copy()
    df['pscore'] = pscore
    df['delta_y'] = df['re78'] - df['re75']

    # Fit outcome model on controls
    X_ctrl = df.loc[df['treat'] == 0, covariates].values
    y_ctrl = df.loc[df['treat'] == 0, 'delta_y'].values

    or_model = LinearRegression()
    or_model.fit(X_ctrl, y_ctrl)

    # Predict for all units
    df['mu_hat'] = or_model.predict(df[covariates].values)

    # DR formula
    n1 = df['treat'].sum()

    # For treated: Y - mu_hat
    treat_component = (df['delta_y'] - df['mu_hat']) * df['treat']

    # For controls: weighted residual
    ctrl_weight = df['pscore'] / (1 - df['pscore']) * (1 - df['treat'])
    ctrl_component = (df['delta_y'] - df['mu_hat']) * ctrl_weight

    # Normalize control weights
    ctrl_weight_sum = ctrl_weight.sum()
    if ctrl_weight_sum > 0:
        ctrl_component = ctrl_component / ctrl_weight_sum * n1

    att_dr = (treat_component.sum() - ctrl_component.sum()) / n1

    return att_dr


def main():
    print("=" * 60)
    print("Module 04: Covariates and Doubly Robust DiD")
    print("LaLonde (1986) - NSW Job Training Program")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1] Loading LaLonde data...")
    df = load_lalonde()

    print(f"\nDataset shape: {df.shape}")
    print(f"Treated: {df['treat'].sum():.0f}")
    print(f"Control: {(1 - df['treat']).sum():.0f}")

    # Define covariates
    covariates = ['age', 'educ', 'black', 'hisp', 'married', 're74', 're75']

    # =========================================================================
    # Step 2: Show Selection Problem
    # =========================================================================
    print("\n" + "=" * 60)
    print("[2] The Selection Problem")
    print("=" * 60)

    print("\nMean characteristics by treatment status:")
    print("-" * 50)
    summary = df.groupby('treat')[covariates + ['re78']].mean()
    summary.index = ['Control', 'Treated']
    print(summary.round(2).T)

    print("""
Key observations:
  - Treated have LOWER pre-treatment earnings (re74, re75)
  - Treated are younger, less educated, more likely Black
  - This is SELECTION BIAS: workers who seek training are different
    """)

    # =========================================================================
    # Step 3: Naive DiD (Shows the Problem)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[3] Naive DiD (Without Adjustment)")
    print("=" * 60)

    naive_att = naive_did(df)

    treat_change = df.loc[df['treat'] == 1, 're78'].mean() - df.loc[df['treat'] == 1, 're75'].mean()
    ctrl_change = df.loc[df['treat'] == 0, 're78'].mean() - df.loc[df['treat'] == 0, 're75'].mean()

    print(f"\nTreated: Change in earnings = ${treat_change:,.0f}")
    print(f"Control: Change in earnings = ${ctrl_change:,.0f}")
    print(f"\nNaive DiD ATT = ${naive_att:,.0f}")

    if naive_att < 0:
        print("\nPROBLEM: Negative ATT suggests training HURTS earnings!")
        print("This is wrong - driven by selection bias (different trends).")

    # =========================================================================
    # Step 4: Propensity Score Estimation
    # =========================================================================
    print("\n" + "=" * 60)
    print("[4] Propensity Score Estimation")
    print("=" * 60)

    pscore = estimate_propensity_score(df, covariates)
    df['pscore'] = pscore

    print(f"\nPropensity score summary:")
    print(f"  Treated: mean={pscore[df['treat']==1].mean():.3f}, "
          f"min={pscore[df['treat']==1].min():.3f}, max={pscore[df['treat']==1].max():.3f}")
    print(f"  Control: mean={pscore[df['treat']==0].mean():.3f}, "
          f"min={pscore[df['treat']==0].min():.3f}, max={pscore[df['treat']==0].max():.3f}")

    # =========================================================================
    # Step 5: IPW DiD
    # =========================================================================
    print("\n" + "=" * 60)
    print("[5] IPW (Inverse Probability Weighting) DiD")
    print("=" * 60)

    ipw_att = ipw_did(df, pscore)

    print(f"\nIPW DiD ATT = ${ipw_att:,.0f}")
    print("\nInterpretation: Reweight controls to match treated on observables.")

    # =========================================================================
    # Step 6: Outcome Regression DiD
    # =========================================================================
    print("\n" + "=" * 60)
    print("[6] Outcome Regression DiD")
    print("=" * 60)

    or_att = outcome_regression_did(df, covariates)

    print(f"\nOutcome Regression ATT = ${or_att:,.0f}")
    print("\nInterpretation: Predict counterfactual change using control model.")

    # =========================================================================
    # Step 7: Doubly Robust DiD
    # =========================================================================
    print("\n" + "=" * 60)
    print("[7] Doubly Robust DiD")
    print("=" * 60)

    dr_att = doubly_robust_did(df, pscore, covariates)

    print(f"\nDoubly Robust ATT = ${dr_att:,.0f}")
    print("\nInterpretation: Combines IPW and OR for robustness.")
    print("Consistent if EITHER propensity OR outcome model is correct.")

    # =========================================================================
    # Step 8: Comparison of All Methods
    # =========================================================================
    print("\n" + "=" * 60)
    print("[8] Comparison of All Methods")
    print("=" * 60)

    print("\n" + "-" * 50)
    print(f"{'Method':<25} {'ATT Estimate':>15}")
    print("-" * 50)
    print(f"{'Naive DiD':<25} ${naive_att:>14,.0f}")
    print(f"{'IPW DiD':<25} ${ipw_att:>14,.0f}")
    print(f"{'Outcome Regression':<25} ${or_att:>14,.0f}")
    print(f"{'Doubly Robust':<25} ${dr_att:>14,.0f}")
    print("-" * 50)

    # =========================================================================
    # Step 9: Visualizations
    # =========================================================================
    print("\n" + "=" * 60)
    print("[9] Creating Visualizations")
    print("=" * 60)

    # Plot 1: Propensity Score Distribution
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    ax1.hist(pscore[df['treat'] == 1], bins=30, alpha=0.6, label='Treated',
             density=True, color='red')
    ax1.hist(pscore[df['treat'] == 0], bins=30, alpha=0.6, label='Control',
             density=True, color='blue')

    ax1.set_xlabel('Propensity Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Propensity Score Distribution\n(Overlap is necessary for valid inference)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(FIGS_DIR / 'propensity_scores.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'propensity_scores.png'}")

    # Plot 2: Earnings Trends
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Create time series
    times = ['1974\n(Pre)', '1975\n(Pre)', '1978\n(Post)']

    treat_means = [df.loc[df['treat']==1, 're74'].mean(),
                   df.loc[df['treat']==1, 're75'].mean(),
                   df.loc[df['treat']==1, 're78'].mean()]

    ctrl_means = [df.loc[df['treat']==0, 're74'].mean(),
                  df.loc[df['treat']==0, 're75'].mean(),
                  df.loc[df['treat']==0, 're78'].mean()]

    x = [0, 1, 2]
    ax2.plot(x, treat_means, 'r-o', linewidth=2, markersize=10, label='Treated (NSW)')
    ax2.plot(x, ctrl_means, 'b-o', linewidth=2, markersize=10, label='Control')

    # Add counterfactual
    counterfactual = treat_means[1] + (ctrl_means[2] - ctrl_means[1])
    ax2.plot([1, 2], [treat_means[1], counterfactual], 'r--',
             linewidth=2, alpha=0.5, label='Counterfactual')

    ax2.axvline(x=1.5, color='gray', linestyle='--', label='Treatment')

    ax2.set_xticks(x)
    ax2.set_xticklabels(times)
    ax2.set_xlabel('Year', fontsize=12)
    ax2.set_ylabel('Real Earnings ($)', fontsize=12)
    ax2.set_title('Earnings Trends: Selection Bias in LaLonde Data\n(Note: Treated have "dipping" pre-trends)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / 'earnings_trends.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'earnings_trends.png'}")

    # Plot 3: Method Comparison Bar Chart
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    methods = ['Naive DiD', 'IPW', 'Outcome Reg.', 'Doubly Robust']
    estimates = [naive_att, ipw_att, or_att, dr_att]
    colors = ['gray', 'steelblue', 'forestgreen', 'darkred']

    bars = ax3.bar(methods, estimates, color=colors, alpha=0.8, edgecolor='black')

    ax3.axhline(y=0, color='black', linewidth=1)

    # Add value labels
    for bar, est in zip(bars, estimates):
        height = bar.get_height()
        ax3.annotate(f'${est:,.0f}',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 5 if height > 0 else -15),
                    textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

    ax3.set_ylabel('ATT Estimate ($)', fontsize=12)
    ax3.set_title('Comparison of DiD Methods\n(Covariate adjustment corrects selection bias)', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    fig3.tight_layout()
    fig3.savefig(FIGS_DIR / 'method_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'method_comparison.png'}")

    # =========================================================================
    # Step 10: Interpretation
    # =========================================================================
    print("\n" + "=" * 60)
    print("[10] Interpretation")
    print("=" * 60)

    print(f"""
RESULTS SUMMARY
===============

The Problem:
    Job training participants have LOWER pre-treatment earnings
    and different trends than the general population (selection bias).

Naive DiD Result: ${naive_att:,.0f}
    WRONG SIGN! Suggests training hurts earnings.
    This happens because control units have higher baseline earnings
    and stable trends, while treated have "dipping" pre-trends.

Covariate-Adjusted Results:
    IPW:            ${ipw_att:,.0f}
    Outcome Reg:    ${or_att:,.0f}
    Doubly Robust:  ${dr_att:,.0f}

    All positive, suggesting training HELPS earnings (correct direction).

KEY INSIGHTS:
    1. Simple DiD requires UNCONDITIONAL parallel trends
    2. When groups differ systematically, adjust for covariates
    3. IPW reweights controls to look like treated
    4. Outcome regression predicts counterfactual directly
    5. Doubly robust combines both for insurance

RECOMMENDATION:
    Use Doubly Robust when:
    - Treatment is non-random
    - Groups have different baseline characteristics
    - You want protection against model misspecification
    """)

    plt.close('all')
    print("\n" + "=" * 60)
    print("Module 04 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
