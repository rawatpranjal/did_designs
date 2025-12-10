"""
Module 02: Event Study (2×T Design)

This script demonstrates:
1. Manual event study by looping 2×2 DiD over time
2. Regression-based event study with year dummies
3. Pre-trends testing and dynamic effects visualization

Dataset: California Proposition 99 (1988) - Tobacco Tax
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_smoking, calculate_2x2_did, plot_event_study

# Output directory
FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Treatment year for California Prop 99
TREATMENT_YEAR = 1988
REFERENCE_YEAR = 1987  # Year before treatment


def manual_event_study(df, outcome_col='cigsale', treat_col='treated',
                       time_col='year', ref_year=REFERENCE_YEAR):
    """
    Calculate event study coefficients manually by looping 2×2 DiD.

    For each year t, computes:
        ATT(t) = [E(Y_t|D=1) - E(Y_ref|D=1)] - [E(Y_t|D=0) - E(Y_ref|D=0)]

    Returns DataFrame with event study results.
    """
    years = sorted(df[time_col].unique())
    results = []

    # Get reference year data
    df_ref = df[df[time_col] == ref_year].copy()

    for year in years:
        if year == ref_year:
            # Reference year is normalized to 0
            results.append({
                'year': year,
                'event_time': year - TREATMENT_YEAR,
                'att': 0.0,
                'se': 0.0,
                'n_treat': df[(df[treat_col] == 1) & (df[time_col] == year)].shape[0],
                'n_control': df[(df[treat_col] == 0) & (df[time_col] == year)].shape[0]
            })
            continue

        # Get current year data
        df_t = df[df[time_col] == year].copy()

        # Calculate means for the 2×2
        # Treated group
        y_treat_t = df_t.loc[df_t[treat_col] == 1, outcome_col].values
        y_treat_ref = df_ref.loc[df_ref[treat_col] == 1, outcome_col].values

        # Control group
        y_control_t = df_t.loc[df_t[treat_col] == 0, outcome_col].values
        y_control_ref = df_ref.loc[df_ref[treat_col] == 0, outcome_col].values

        if len(y_treat_t) == 0 or len(y_control_t) == 0:
            continue

        # Calculate DiD
        delta_treat = np.mean(y_treat_t) - np.mean(y_treat_ref)
        delta_control = np.mean(y_control_t) - np.mean(y_control_ref)
        att = delta_treat - delta_control

        # Simple standard error (pooled)
        # Note: This ignores Cov(Y_t, Y_ref) in panel data (conservative SEs).
        # The regression method handles this correctly via the variance-covariance matrix.
        n_treat = len(y_treat_t)
        n_control = len(y_control_t)
        var_treat = np.var(y_treat_t, ddof=1) / n_treat if n_treat > 1 else 0
        var_control = np.var(y_control_t, ddof=1) / n_control if n_control > 1 else 0
        se = np.sqrt(var_treat + var_control)

        results.append({
            'year': year,
            'event_time': year - TREATMENT_YEAR,
            'att': att,
            'se': se,
            'n_treat': n_treat,
            'n_control': n_control
        })

    return pd.DataFrame(results)


def regression_event_study(df, outcome_col='cigsale', treat_col='treated',
                           time_col='year', ref_year=REFERENCE_YEAR):
    """
    Calculate event study using regression with year dummies.

    Model: Y_it = α + Σ_t β_t * (D_i × 1[t]) + γ_t + ε_it

    The coefficients β_t are the event study estimates.
    """
    df = df.copy()

    # Create year dummies interacted with treatment
    years = sorted(df[time_col].unique())
    years_excl_ref = [y for y in years if y != ref_year]

    # Create interaction terms for each year (excluding reference)
    for year in years_excl_ref:
        df[f'treat_x_{year}'] = (df[treat_col] == 1) & (df[time_col] == year)
        df[f'treat_x_{year}'] = df[f'treat_x_{year}'].astype(int)

    # Build formula
    interactions = ' + '.join([f'treat_x_{y}' for y in years_excl_ref])
    formula = f'{outcome_col} ~ {interactions} + C({time_col}) + {treat_col}'

    # Fit model
    model = smf.ols(formula, data=df).fit()

    # Extract event study coefficients
    results = []
    for year in years:
        if year == ref_year:
            results.append({
                'year': year,
                'event_time': year - TREATMENT_YEAR,
                'att': 0.0,
                'se': 0.0
            })
        else:
            coef_name = f'treat_x_{year}'
            results.append({
                'year': year,
                'event_time': year - TREATMENT_YEAR,
                'att': model.params[coef_name],
                'se': model.bse[coef_name]
            })

    return pd.DataFrame(results), model


def main():
    print("=" * 60)
    print("Module 02: Event Study (2×T Design)")
    print("California Proposition 99 (1988) - Tobacco Tax")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load the Data
    # =========================================================================
    print("\n[1] Loading California Prop 99 smoking data...")
    df = load_smoking()

    print(f"\nDataset shape: {df.shape}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"States: {df['state'].nunique()}")
    print(f"Treatment group (California): {df[df['treated']==1]['state'].unique()}")

    # =========================================================================
    # Step 2: Visualize Raw Trends
    # =========================================================================
    print("\n[2] Visualizing raw trends...")

    fig1, ax1 = plt.subplots(figsize=(12, 6))

    # Plot California
    ca_data = df[df['treated'] == 1].groupby('year')['cigsale'].mean()
    ax1.plot(ca_data.index, ca_data.values, 'r-', linewidth=2.5,
             label='California (Treated)', marker='o', markersize=4)

    # Plot average of control states
    control_data = df[df['treated'] == 0].groupby('year')['cigsale'].mean()
    ax1.plot(control_data.index, control_data.values, 'b-', linewidth=2.5,
             label='Other States (Control)', marker='s', markersize=4)

    # Treatment line
    ax1.axvline(x=TREATMENT_YEAR, color='gray', linestyle='--',
                linewidth=2, label='Prop 99 (1988)')

    ax1.set_xlabel('Year', fontsize=12)
    ax1.set_ylabel('Cigarette Sales Per Capita (Packs)', fontsize=12)
    ax1.set_title('California vs. Other States: Raw Trends\nProposition 99 Tobacco Tax', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(FIGS_DIR / 'raw_trends.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'raw_trends.png'}")

    # =========================================================================
    # Step 3: Manual Event Study (Looping 2×2)
    # =========================================================================
    print("\n" + "=" * 60)
    print("[3] Manual Event Study: Looping 2×2 DiD Over Time")
    print("=" * 60)

    print(f"\nReference year: {REFERENCE_YEAR}")
    print(f"Treatment year: {TREATMENT_YEAR}")
    print("\nFor each year t, we calculate:")
    print("  ATT(t) = [Δ_CA(t vs ref)] - [Δ_Control(t vs ref)]")

    manual_results = manual_event_study(df)

    print("\n" + "-" * 60)
    print("Event Study Coefficients (Manual Calculation)")
    print("-" * 60)
    print(manual_results[['year', 'event_time', 'att', 'se']].to_string(index=False))

    # =========================================================================
    # Step 4: Regression Event Study
    # =========================================================================
    print("\n" + "=" * 60)
    print("[4] Regression Event Study")
    print("=" * 60)

    reg_results, model = regression_event_study(df)

    print("\nRegression model: Y ~ Σ(D × Year_t) + Year_FE + D")
    print(f"R-squared: {model.rsquared:.4f}")

    # =========================================================================
    # Step 5: Compare Manual vs Regression
    # =========================================================================
    print("\n" + "=" * 60)
    print("[5] Comparison: Manual vs Regression")
    print("=" * 60)

    comparison = manual_results[['year', 'att']].merge(
        reg_results[['year', 'att']],
        on='year',
        suffixes=('_manual', '_regression')
    )

    print("\n" + comparison.to_string(index=False))

    # Check correlation
    corr = comparison['att_manual'].corr(comparison['att_regression'])
    print(f"\nCorrelation between methods: {corr:.6f}")

    # =========================================================================
    # Step 6: Pre-Trends Test
    # =========================================================================
    print("\n" + "=" * 60)
    print("[6] Pre-Trends Test")
    print("=" * 60)

    pre_period = manual_results[manual_results['event_time'] < 0]
    print("\nPre-treatment coefficients (should be ~0 if parallel trends hold):")
    print(pre_period[['year', 'event_time', 'att', 'se']].to_string(index=False))

    # Test: are pre-period coefficients jointly zero?
    pre_att = pre_period['att'].values
    print(f"\nMean of pre-period ATTs: {np.mean(pre_att):.4f}")
    print(f"Std of pre-period ATTs: {np.std(pre_att):.4f}")

    # =========================================================================
    # Step 7: Create Event Study Plot
    # =========================================================================
    print("\n" + "=" * 60)
    print("[7] Creating Event Study Plot")
    print("=" * 60)

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    # Plot coefficients
    x = manual_results['event_time']
    y = manual_results['att']
    se = manual_results['se']

    # Point estimates
    ax2.scatter(x, y, color='blue', s=80, zorder=5)
    ax2.plot(x, y, color='blue', linewidth=1, alpha=0.5)

    # Confidence intervals (95%)
    ax2.fill_between(x, y - 1.96*se, y + 1.96*se,
                     color='blue', alpha=0.2, label='95% CI')

    # Reference lines
    ax2.axhline(y=0, color='black', linewidth=1, linestyle='-')
    ax2.axvline(x=-0.5, color='red', linewidth=2, linestyle='--',
                label='Treatment (1988)')

    # Labels
    ax2.set_xlabel('Years Relative to Prop 99', fontsize=12)
    ax2.set_ylabel('Treatment Effect (Packs per Capita)', fontsize=12)
    ax2.set_title('Event Study: California Proposition 99\nEffect on Cigarette Sales', fontsize=14)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)

    # Annotate pre and post
    ax2.annotate('Pre-treatment\n(test parallel trends)',
                xy=(-10, 5), fontsize=10, ha='center', color='gray')
    ax2.annotate('Post-treatment\n(treatment effects)',
                xy=(8, -30), fontsize=10, ha='center', color='gray')

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / 'event_study.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'event_study.png'}")

    # =========================================================================
    # Step 8: Interpretation
    # =========================================================================
    print("\n" + "=" * 60)
    print("[8] Interpretation")
    print("=" * 60)

    post_period = manual_results[manual_results['event_time'] >= 0]
    avg_post_effect = post_period['att'].mean()

    print(f"""
Event Study Results for California Prop 99:

PRE-TRENDS (Event time < 0):
    - Average pre-period ATT: {np.mean(pre_att):.2f} packs
    - Pre-period coefficients are {'close to zero' if abs(np.mean(pre_att)) < 5 else 'NOT close to zero'}
    - Parallel trends assumption: {'SUPPORTED' if abs(np.mean(pre_att)) < 5 else 'QUESTIONABLE'}

POST-TREATMENT EFFECTS (Event time >= 0):
    - Average post-period ATT: {avg_post_effect:.2f} packs
    - Effect direction: {'NEGATIVE (reduction in smoking)' if avg_post_effect < 0 else 'POSITIVE'}
    - Effect grows over time: {'YES' if post_period['att'].iloc[-1] < post_period['att'].iloc[0] else 'NO'}

INTERPRETATION:
    The tobacco tax reduced cigarette consumption in California by
    approximately {abs(avg_post_effect):.0f} packs per capita on average,
    with the effect growing stronger over time.

KEY INSIGHT:
    Each coefficient is a 2×2 DiD comparing that year to {REFERENCE_YEAR}.
    The event study is just a series of 2×2 comparisons.
    """)

    plt.close('all')
    print("\n" + "=" * 60)
    print("Module 02 Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
