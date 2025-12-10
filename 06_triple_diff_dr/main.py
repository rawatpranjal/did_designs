"""
Module 06: Triple Differences - The Robust Way

This script demonstrates:
1. Why naive DDD can be biased when covariate distributions differ
2. The Outcome Regression DDD estimator (inspired by Ortiz-Villavicencio & Sant'Anna 2025)
3. Comparison using real-world policy data

Dataset: Meyer, Viscusi, & Durbin (1995) - Worker's Compensation
Source: Wooldridge package via Rdatasets

Policy: Kentucky raised the cap on worker's compensation benefits.
- Treated State (S=1): Kentucky
- Control State (S=0): Michigan
- Target Group (Q=1): High Earners (benefits increased)
- Placebo Group (Q=0): Low Earners (benefits unchanged)
- Outcome: Log duration of leave (ldurat)
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

from utils import load_injury, COLORS

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)


def show_covariate_imbalance(df):
    """
    Display the covariate imbalance between high and low earners.
    """
    print("\nCovariate Comparison: High Earners vs Low Earners")
    print("-" * 60)

    # Calculate means
    high = df[df['highearn'] == 1]
    low = df[df['highearn'] == 0]

    age_high, age_low = high['age'].mean(), low['age'].mean()
    marr_high, marr_low = high['married'].mean(), low['married'].mean()
    male_high, male_low = high['male'].mean(), low['male'].mean()

    print(f"{'Covariate':<15} {'High Earners':>15} {'Low Earners':>15} {'Difference':>15}")
    print("-" * 60)
    print(f"{'Age':<15} {age_high:>15.2f} {age_low:>15.2f} {age_high - age_low:>+15.2f}")
    print(f"{'Married (%)':<15} {marr_high*100:>15.1f} {marr_low*100:>15.1f} {(marr_high - marr_low)*100:>+15.1f}")
    print(f"{'Male (%)':<15} {male_high*100:>15.1f} {male_low*100:>15.1f} {(male_high - male_low)*100:>+15.1f}")
    print("-" * 60)

    print("\nKey Finding: High earners are OLDER, more MALE, and more MARRIED.")
    print("If injury duration trends depend on these factors (which they likely do),")
    print("naive DDD comparing High vs Low will be biased.")

    return {
        'age': [age_low, age_high],
        'married': [marr_low, marr_high],
        'male': [male_low, male_high]
    }


def naive_ddd_ols(df):
    """
    Naive DDD using 3-way fixed effects OLS.
    Model: ldurat ~ ky * highearn * afchnge + covariates
    """
    # Standard linear control
    model = smf.ols("ldurat ~ ky * highearn * afchnge + age + male + married", data=df).fit(cov_type='HC1')
    return model.params['ky:highearn:afchnge'], model.bse['ky:highearn:afchnge']


def robust_ddd(df):
    """
    Outcome Regression DDD estimator (Target-Adjusted).
    Based on Ortiz-Villavicencio & Sant'Anna (2025), Section 4.1.

    Key insight: Predict counterfactuals at the TARGET group's covariate distribution.

    1. Fit outcome model E[Y|X] for every group/time cell.
    2. Predict Y_hat for the TARGET group (KY High Post) using these models.
    3. Compute DDD using these adjusted means.
    """
    # Target group: Kentucky High Earners, Post-policy
    target_mask = (df['ky'] == 1) & (df['highearn'] == 1) & (df['afchnge'] == 1)
    target_data = df[target_mask].copy()

    # Dictionary to store adjusted means
    adj_means = {}

    # Iterate over the 8 cells (2 states * 2 groups * 2 periods)
    # The target cell is (1, 1, 1)
    for ky in [0, 1]:
        for high in [0, 1]:
            for post in [0, 1]:

                # Identify the specific cell
                mask = (df['ky'] == ky) & (df['highearn'] == high) & (df['afchnge'] == post)
                cell_data = df[mask]

                if len(cell_data) < 5:
                    adj_means[(ky, high, post)] = np.nan
                    continue

                # Fit model: E[Y | X] for THIS cell
                model = smf.ols("ldurat ~ age + male + married", data=cell_data).fit()

                # Predict: What if the TARGET group had the parameters of THIS cell?
                # This standardizes everything to the Target's covariate distribution
                pred = model.predict(target_data)

                adj_means[(ky, high, post)] = pred.mean()

    # Compute DDD using the adjusted means
    try:
        # Kentucky DiD (Adjusted)
        ky_high_diff = adj_means[(1, 1, 1)] - adj_means[(1, 1, 0)]
        ky_low_diff  = adj_means[(1, 0, 1)] - adj_means[(1, 0, 0)]
        ky_did = ky_high_diff - ky_low_diff

        # Michigan DiD (Adjusted)
        mi_high_diff = adj_means[(0, 1, 1)] - adj_means[(0, 1, 0)]
        mi_low_diff  = adj_means[(0, 0, 1)] - adj_means[(0, 0, 0)]
        mi_did = mi_high_diff - mi_low_diff

        ddd_robust = ky_did - mi_did

        components = {
            'ky_did': ky_did, 'mi_did': mi_did,
            'ky_high': ky_high_diff, 'ky_low': ky_low_diff,
            'mi_high': mi_high_diff, 'mi_low': mi_low_diff
        }
        return ddd_robust, components

    except KeyError:
        return np.nan, {}


def bootstrap_se(df, estimator_func, n_boot=200):
    """Simple bootstrap for SE."""
    boot_ests = []
    n = len(df)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        try:
            est, _ = estimator_func(df.iloc[idx])
            if not np.isnan(est):
                boot_ests.append(est)
        except:
            continue
    return np.std(boot_ests)


def main():
    print("=" * 70)
    print("Module 06: Triple Differences - The Robust Way")
    print("Meyer, Viscusi, & Durbin (1995) - Worker's Compensation")
    print("=" * 70)

    # 1. Load Data
    print("\n[1] Loading Data...")
    df = load_injury()
    # Ensure no missing values in covariates
    df = df.dropna(subset=['ldurat', 'age', 'male', 'married', 'ky', 'highearn', 'afchnge'])

    print(f"Dataset: {len(df)} observations")
    print("Structure: 2 States (KY, MI) x 2 Groups (High/Low Earn) x 2 Periods")

    # 2. Show Imbalance
    print("\n[2] Checking Covariate Imbalance...")
    imbalance_data = show_covariate_imbalance(df)

    # 3. Naive Estimation
    print("\n[3] Naive DDD (OLS with linear controls)...")
    naive_est, naive_se = naive_ddd_ols(df)
    print(f"Naive DDD:  {naive_est:.4f} (SE: {naive_se:.4f})")

    # 4. Robust Estimation
    print("\n[4] Robust DDD (Outcome Regression / Target-Adjusted)...")
    robust_est, components = robust_ddd(df)

    print("Bootstrapping SE (approx 5-10s)...")
    robust_se = bootstrap_se(df, robust_ddd, n_boot=200)

    print(f"Robust DDD: {robust_est:.4f} (SE: {robust_se:.4f})")

    # 5. Comparison
    print("\n[5] Comparison")
    print("-" * 50)
    print(f"{'Method':<20} {'Estimate':>10} {'SE':>10}")
    print("-" * 50)
    print(f"{'Naive (OLS)':<20} {naive_est:>10.4f} {naive_se:>10.4f}")
    print(f"{'Robust (OR)':<20} {robust_est:>10.4f} {robust_se:>10.4f}")
    print("-" * 50)

    diff = robust_est - naive_est
    print(f"Difference: {diff:.4f}")
    print("The difference represents the bias removed by properly adjusting for covariates.")

    # =========================================================================
    # Step 6: Visualizations
    # =========================================================================
    print("\n[6] Creating Showcase Visualizations...")
    # plt.style.use('seaborn-v0_8-whitegrid') # Already set in utils

    # --- Plot 1: Covariate Imbalance ---
    fig1, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Age
    axes[0].bar(['Low Earners', 'High Earners'], imbalance_data['age'],
                color=[COLORS['control'], COLORS['treat']], alpha=0.8)
    axes[0].set_title("Mean Age", fontsize=14)
    axes[0].set_ylabel("Years")
    axes[0].grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)

    # Married
    axes[1].bar(['Low Earners', 'High Earners'], [x*100 for x in imbalance_data['married']],
                color=[COLORS['control'], COLORS['treat']], alpha=0.8)
    axes[1].set_title("Married (%)", fontsize=14)
    axes[1].set_ylabel("Percent")
    axes[1].grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)

    fig1.suptitle("Why Naive DDD Fails: Systematic Covariate Imbalance", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'covariate_imbalance.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'covariate_imbalance.png'}")

    # --- Plot 2: Decomposition ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    labels = ['KY High\n(Target)', 'KY Low', 'MI High', 'MI Low']
    values = [components['ky_high'], components['ky_low'], components['mi_high'], components['mi_low']]
    # Red for KY, Blue for MI
    colors = [COLORS['treat'], '#ff9896', COLORS['control'], '#aec7e8']

    bars = ax2.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)
    ax2.axhline(0, color='black', linewidth=1)

    # Add values
    for bar, v in zip(bars, values):
        height = bar.get_height()
        offset = 0.005 if height > 0 else -0.015
        ax2.text(bar.get_x() + bar.get_width()/2, height + offset, f"{v:.3f}",
                 ha='center', fontweight='bold', color=COLORS['text'])

    ax2.set_title("Robust DDD Decomposition (Target-Adjusted Trends)", fontsize=16, fontweight='bold')
    ax2.set_ylabel("Change in Log Duration (Post - Pre)", fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Add DDD Calculation Annotation
    note = (
        f"Robust DDD = (KY High - KY Low) - (MI High - MI Low)\n"
        f"= ({values[0]:.3f} - {values[1]:.3f}) - ({values[2]:.3f} - {values[3]:.3f})\n"
        f"= {robust_est:.3f}"
    )
    ax2.text(0.02, 0.95, note, transform=ax2.transAxes, va='top',
             bbox=dict(boxstyle="round", fc="white", ec="#e5e5e5"), fontsize=11)

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'ddd_decomposition.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'ddd_decomposition.png'}")

    # --- Plot 3: Bias Comparison ---
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    
    methods = ['Naive DDD\n(OLS)', 'Robust DDD\n(Outcome Reg.)']
    estimates = [naive_est, robust_est]
    colors = [COLORS['control'], COLORS['treat']] # Blue for Naive, Red for Robust (Target)
    
    bars = ax3.bar(methods, estimates, color=colors, edgecolor='black', alpha=0.8, width=0.5)
    ax3.axhline(0, color='black', linewidth=1)
    
    # Add values
    for bar, v in zip(bars, estimates):
        height = bar.get_height()
        offset = 0.005 if height > 0 else -0.015
        ax3.text(bar.get_x() + bar.get_width()/2, height + offset, f"{v:.3f}",
                 ha='center', fontweight='bold', color=COLORS['text'])
                 
    ax3.set_title('Bias Comparison: Naive vs. Robust DDD', fontsize=16, fontweight='bold')
    ax3.set_ylabel('Estimated Effect', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y', color=COLORS['grid'])
    
    # Clean spines
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Annotation
    diff = robust_est - naive_est
    ax3.annotate(f'Difference due to\ncovariate adjustment:\n{diff:+.3f}', 
                 xy=(0.5, (naive_est + robust_est)/2), xytext=(0.5, max(estimates) + 0.05),
                 ha='center', arrowprops=dict(arrowstyle='-|>', color=COLORS['text']))

    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'ddd_comparison.png', dpi=300)
    print(f"Saved: {FIGS_DIR / 'ddd_comparison.png'}")

    plt.close('all')
    print("\n" + "=" * 70)
    print("Module 06 Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
