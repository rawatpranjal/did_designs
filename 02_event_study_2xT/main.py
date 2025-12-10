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

from utils import load_smoking, calculate_2x2_did, plot_event_study, COLORS

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
    Uses Clustered Standard Errors (by state) for rigorous inference.

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

    # Fit model with CLUSTERED Standard Errors (State level)
    model = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['state']})

    # Extract event study coefficients
    results = []
    for year in years:
        if year == ref_year:
            results.append({
                'year': year,
                'event_time': year - TREATMENT_YEAR,
                'att': 0.0,
                'se': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0
            })
        else:
            coef_name = f'treat_x_{year}'
            results.append({
                'year': year,
                'event_time': year - TREATMENT_YEAR,
                'att': model.params[coef_name],
                'se': model.bse[coef_name],
                'ci_lower': model.conf_int().loc[coef_name][0],
                'ci_upper': model.conf_int().loc[coef_name][1]
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
    # Step 7: The "Showcase" Event Study Visualization
    # =========================================================================
    print("\n" + "=" * 60)
    print("[7] Creating The Ultimate Event Study Visual")
    print("=" * 60)

    # Use regression results (with clustered SEs)
    res = reg_results.copy()

    # plt.style.use('seaborn-v0_8-whitegrid') # Already set in utils
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    # Split into Pre and Post for styling
    pre = res[res['event_time'] < 0]
    post = res[res['event_time'] >= 0]
    ref = res[res['event_time'] == -1]  # Reference point

    # --- 1. Plot Confidence Intervals & Points ---

    # Pre-Period (Testing Zone): Gray color scheme
    ax2.errorbar(pre['event_time'], pre['att'],
                 yerr=1.96*pre['se'], fmt='o', color=COLORS['counterfactual'],
                 ecolor=COLORS['counterfactual'], elinewidth=2, capsize=0,
                 label='Pre-Trend Test', alpha=0.8, markersize=8)

    # Post-Period (Effect Zone): Red color scheme
    ax2.errorbar(post['event_time'], post['att'],
                 yerr=1.96*post['se'], fmt='o', color=COLORS['treat'],
                 ecolor=COLORS['treat'], elinewidth=2, capsize=0,
                 label='Treatment Effect', alpha=0.9, markersize=8)

    # Reference Year (Explicitly marked with black diamond)
    ax2.plot(ref['event_time'], ref['att'], marker='D', color='black',
             markersize=10, zorder=10, label='Reference (t=-1)')

    # --- 2. Structural Lines ---
    ax2.axhline(0, color='black', linewidth=1, linestyle='-')
    ax2.axvline(-0.5, color='black', linewidth=1.5, linestyle='--')

    # --- 3. Embed the Math ---
    equation = (
        r"$ATT_t = (\bar{y}_{T,t} - \bar{y}_{T,ref}) - (\bar{y}_{C,t} - \bar{y}_{C,ref})$"
    )
    ax2.text(0.02, 0.95, equation, transform=ax2.transAxes, ha='left', va='top',
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#e5e5e5", alpha=0.9))

    # --- 4. Annotations ---
    # Treatment label
    ax2.text(-0.7, ax2.get_ylim()[0]*0.85, "Treatment\nStarts (1988)",
             ha='right', va='bottom', fontsize=10, fontstyle='italic', color='black')

    # Pre-trend logic
    ax2.text(-10, 12, "Testing Zone:\nPre-trends ≈ 0",
             ha='center', fontsize=10, color=COLORS['counterfactual'], fontweight='bold')

    # Post-trend logic (positioned lower to avoid overlapping with data points)
    ax2.text(6, -35, "Result Zone:\nDynamic Treatment Effect",
             ha='center', fontsize=10, color=COLORS['treat'], fontweight='bold')

    # --- 5. Formatting ---
    ax2.set_xlabel("Years Relative to Proposition 99", fontsize=12)
    ax2.set_ylabel("Difference in Cigarette Sales (Packs/Capita)", fontsize=12)
    ax2.set_title("Event Study: The Effect of Prop 99 on Cigarette Sales",
                  fontsize=16, fontweight='bold', pad=15)

    ax2.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax2.set_xlim(res['event_time'].min() - 1, res['event_time'].max() + 1)

    # Clean spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig2.tight_layout()
    fig2.savefig(FIGS_DIR / 'event_study_showcase.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'event_study_showcase.png'}")

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
