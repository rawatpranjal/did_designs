"""
Shared utility functions for the Modern DiD Showcase.

This module provides common functions for:
- Loading datasets
- Manual DiD calculations
- Visualization helpers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get the root directory of the project
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_card_krueger():
    """
    Load the Card & Krueger (1994) minimum wage dataset.

    NJ raised minimum wage in April 1992 from $4.25 to $5.05.
    PA did not change minimum wage (control).

    Returns:
        pd.DataFrame with columns:
        - state: 0 = PA (control), 1 = NJ (treatment)
        - time: 0 = Feb 1992 (pre), 1 = Nov 1992 (post)
        - fte: Full-time equivalent employment
        - wage: Starting wage
    """
    df = pd.read_csv(DATA_DIR / "card.csv")

    # Create panel structure from the cross-sectional data
    # Wave 1: empft, emppt, nmgrs, wage_st (before)
    # Wave 2: empft2, emppt2, nmgrs2, wage_st2 (after)

    pre = df[['sheet', 'state', 'empft', 'emppt', 'nmgrs', 'wage_st']].copy()
    pre.columns = ['id', 'state', 'empft', 'emppt', 'nmgrs', 'wage']
    pre['time'] = 0

    post = df[['sheet', 'state', 'empft2', 'emppt2', 'nmgrs2', 'wage_st2']].copy()
    post.columns = ['id', 'state', 'empft', 'emppt', 'nmgrs', 'wage']
    post['time'] = 1

    panel = pd.concat([pre, post], ignore_index=True)

    # Calculate FTE: full-time + 0.5 * part-time + managers
    panel['fte'] = panel['empft'] + 0.5 * panel['emppt'] + panel['nmgrs']

    # State: 0 = PA (control), 1 = NJ (treatment)
    # In original data: state=0 is PA, state=1 is NJ
    panel['treated'] = panel['state']
    panel['post'] = panel['time']

    return panel[['id', 'state', 'time', 'treated', 'post', 'fte', 'wage']].dropna(subset=['fte'])


def load_smoking():
    """
    Load the California Proposition 99 smoking dataset.

    California passed Prop 99 (tobacco tax) in 1988.
    Other states serve as controls.

    Returns:
        pd.DataFrame with columns:
        - state: State name
        - year: Year (1970-2000)
        - cigsale: Cigarette sales per capita
        - treated: 1 if California, 0 otherwise
    """
    df = pd.read_csv(DATA_DIR / "smoking.csv")
    df['treated'] = (df['state'] == 'California').astype(int)
    return df


def load_medicaid():
    """
    Load the Medicaid expansion dataset.

    Staggered adoption of Medicaid expansion across US states.

    Returns:
        pd.DataFrame with columns:
        - stfips: State FIPS code
        - year: Year
        - dins: Change in insurance coverage
        - yexp2: Year of Medicaid expansion (cohort)
        - W: Population weight
    """
    df = pd.read_csv(DATA_DIR / "medicaid.csv")
    return df


def load_mpdta():
    """
    Load the minimum wage panel data (mpdta) from the did package.

    500 US counties, 2003-2007, teen employment and minimum wage policy.

    Returns:
        pd.DataFrame with columns:
        - year: Year
        - countyreal: County ID
        - lpop: Log population
        - lemp: Log employment
        - first.treat: Year of first treatment (cohort)
        - treat: Treatment indicator
    """
    df = pd.read_csv(DATA_DIR / "mpdta.csv")
    return df


def load_lalonde():
    """
    Load the LaLonde (1986) National Supported Work (NSW) dataset.

    Classic dataset for evaluating propensity score methods.
    Compares job training participants (NSW) to comparison group.

    Returns:
        pd.DataFrame with columns:
        - treat: 1 if in job training program, 0 otherwise
        - age: Age in years
        - educ: Years of education
        - race: black, hispan, or white
        - married: 1 if married
        - nodegree: 1 if no high school degree
        - re74: Real earnings in 1974 (pre-treatment)
        - re75: Real earnings in 1975 (pre-treatment)
        - re78: Real earnings in 1978 (outcome, post-treatment)
    """
    df = pd.read_csv(DATA_DIR / "lalonde.csv")
    # Drop the rownames column if present
    if 'rownames' in df.columns:
        df = df.drop(columns=['rownames'])
    return df


def load_castle():
    """
    Load the Castle Doctrine / Stand Your Ground dataset.

    State-level panel data (2000-2010) on crime rates and
    castle doctrine law implementation. Used for staggered DiD
    and triple difference analysis.

    Source: Cheng & Hoekstra (2013), via causaldata R package.

    Returns:
        pd.DataFrame with columns:
        - year: Year (2000-2010)
        - sid: State ID
        - post: 1 if castle doctrine law in effect
        - l_homicide: Log homicide rate (main outcome)
        - robbery, assault, burglary, etc.: Crime rates
        - lead1-lead9, lag0-lag5: Event study dummies
        - popwt: Population weight
    """
    df = pd.read_csv(DATA_DIR / "castle.csv")
    # Drop the rownames column if present
    if 'rownames' in df.columns:
        df = df.drop(columns=['rownames'])
    return df


# =============================================================================
# Manual DiD Calculation Functions
# =============================================================================

def calculate_2x2_did(df, outcome_col, treat_col, post_col, verbose=True):
    """
    Manually compute the 2x2 DiD estimator using group means.

    This is the "Four Numbers" approach:
    ATT = (E[Y|Treated,Post] - E[Y|Treated,Pre]) - (E[Y|Control,Post] - E[Y|Control,Pre])

    Args:
        df: DataFrame with outcome, treatment, and post indicators
        outcome_col: Name of outcome variable
        treat_col: Name of treatment group indicator (1=treated, 0=control)
        post_col: Name of post-period indicator (1=post, 0=pre)
        verbose: If True, print intermediate calculations

    Returns:
        dict with ATT and the four group means
    """
    # Calculate the four means
    mu_11 = df.loc[(df[treat_col] == 1) & (df[post_col] == 1), outcome_col].mean()
    mu_10 = df.loc[(df[treat_col] == 1) & (df[post_col] == 0), outcome_col].mean()
    mu_01 = df.loc[(df[treat_col] == 0) & (df[post_col] == 1), outcome_col].mean()
    mu_00 = df.loc[(df[treat_col] == 0) & (df[post_col] == 0), outcome_col].mean()

    # First differences
    delta_treated = mu_11 - mu_10
    delta_control = mu_01 - mu_00

    # Difference in Differences
    att = delta_treated - delta_control

    if verbose:
        print("=" * 50)
        print("Manual 2x2 DiD Calculation")
        print("=" * 50)
        print(f"\nGroup Means:")
        print(f"  E[Y | Treated, Post]  = {mu_11:.4f}")
        print(f"  E[Y | Treated, Pre]   = {mu_10:.4f}")
        print(f"  E[Y | Control, Post]  = {mu_01:.4f}")
        print(f"  E[Y | Control, Pre]   = {mu_00:.4f}")
        print(f"\nFirst Differences (Change over time):")
        print(f"  Treated: {mu_11:.4f} - {mu_10:.4f} = {delta_treated:.4f}")
        print(f"  Control: {mu_01:.4f} - {mu_00:.4f} = {delta_control:.4f}")
        print(f"\nDifference in Differences:")
        print(f"  ATT = {delta_treated:.4f} - {delta_control:.4f} = {att:.4f}")
        print("=" * 50)

    return {
        'att': att,
        'mu_11': mu_11, 'mu_10': mu_10,
        'mu_01': mu_01, 'mu_00': mu_00,
        'delta_treated': delta_treated,
        'delta_control': delta_control
    }


def calculate_att_gt(df, cohort_col, time_col, outcome_col, g, t, control_type='not_yet_treated'):
    """
    Calculate ATT(g,t) for a specific cohort g at time t.

    This is the building block for staggered DiD estimation.

    Args:
        df: Panel DataFrame
        cohort_col: Column with treatment cohort (year of first treatment)
        time_col: Column with calendar time
        outcome_col: Column with outcome variable
        g: Treatment cohort (year group was first treated)
        t: Calendar time period
        control_type: 'not_yet_treated' or 'never_treated'

    Returns:
        float: ATT(g,t) estimate
    """
    # Reference year is g-1 (year before treatment)
    t_ref = g - 1

    # Check that reference year exists in data
    if t_ref not in df[time_col].values:
        return np.nan

    # Slice data to just the two time periods we need
    df_slice = df[df[time_col].isin([t, t_ref])].copy()

    # Define treated group: cohort g
    treated_mask = (df_slice[cohort_col] == g)

    # Define control group based on control_type
    if control_type == 'not_yet_treated':
        # Units not yet treated by time t (includes never-treated)
        control_mask = (df_slice[cohort_col] > t) | (df_slice[cohort_col].isna())
    else:  # never_treated
        control_mask = df_slice[cohort_col].isna()

    # Calculate changes for each group
    try:
        delta_treated = (
            df_slice.loc[treated_mask & (df_slice[time_col] == t), outcome_col].mean() -
            df_slice.loc[treated_mask & (df_slice[time_col] == t_ref), outcome_col].mean()
        )

        delta_control = (
            df_slice.loc[control_mask & (df_slice[time_col] == t), outcome_col].mean() -
            df_slice.loc[control_mask & (df_slice[time_col] == t_ref), outcome_col].mean()
        )

        return delta_treated - delta_control
    except:
        return np.nan


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_did_trends(df, outcome_col, treat_col, time_col,
                    treatment_time=None, title=None, figsize=(10, 6)):
    """
    Create a parallel trends visualization for DiD.

    Args:
        df: DataFrame
        outcome_col: Name of outcome variable
        treat_col: Name of treatment indicator
        time_col: Name of time variable
        treatment_time: Time of treatment (for vertical line)
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate means by group and time
    means = df.groupby([time_col, treat_col])[outcome_col].mean().reset_index()

    # Plot each group
    for treat_val, label, color in [(0, 'Control', 'blue'), (1, 'Treated', 'red')]:
        group_data = means[means[treat_col] == treat_val]
        ax.plot(group_data[time_col], group_data[outcome_col],
                marker='o', label=label, color=color, linewidth=2)

    # Add treatment line
    if treatment_time is not None:
        ax.axvline(x=treatment_time, color='gray', linestyle='--',
                   label='Treatment', alpha=0.7)

    ax.set_xlabel('Time')
    ax.set_ylabel(outcome_col)
    ax.set_title(title or 'Difference-in-Differences: Parallel Trends')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_event_study(results_df, coef_col='att', se_col=None,
                     event_col='event_time', ci_level=0.95,
                     title=None, figsize=(10, 6)):
    """
    Create an event study plot.

    Args:
        results_df: DataFrame with event study results
        coef_col: Column with coefficient estimates
        se_col: Column with standard errors (optional)
        event_col: Column with event time (relative to treatment)
        ci_level: Confidence interval level
        title: Plot title
        figsize: Figure size tuple

    Returns:
        matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    results = results_df.sort_values(event_col)
    x = results[event_col]
    y = results[coef_col]

    # Plot coefficients
    ax.scatter(x, y, color='blue', s=80, zorder=5)
    ax.plot(x, y, color='blue', linewidth=1, alpha=0.5)

    # Add confidence intervals if standard errors provided
    if se_col is not None and se_col in results.columns:
        from scipy import stats
        z = stats.norm.ppf((1 + ci_level) / 2)
        se = results[se_col]
        ax.fill_between(x, y - z * se, y + z * se,
                       color='blue', alpha=0.2, label=f'{int(ci_level*100)}% CI')

    # Reference lines
    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')
    ax.axvline(x=-0.5, color='red', linewidth=1, linestyle='--',
               label='Treatment')

    ax.set_xlabel('Time Relative to Treatment')
    ax.set_ylabel('Treatment Effect')
    ax.set_title(title or 'Event Study')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_2x2_diagram(results, group_labels=('Control', 'Treated'),
                     time_labels=('Pre', 'Post'), figsize=(8, 6)):
    """
    Create a visual 2x2 DiD diagram.

    Args:
        results: dict from calculate_2x2_did
        group_labels: Labels for control and treatment groups
        time_labels: Labels for pre and post periods
        figsize: Figure size

    Returns:
        matplotlib Figure and Axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create the 2x2 structure
    times = [0, 1]

    # Control group
    control_y = [results['mu_00'], results['mu_01']]
    ax.plot(times, control_y, 'b-o', label=group_labels[0], linewidth=2, markersize=10)

    # Treated group (actual)
    treated_y = [results['mu_10'], results['mu_11']]
    ax.plot(times, treated_y, 'r-o', label=group_labels[1], linewidth=2, markersize=10)

    # Counterfactual (parallel trend)
    counterfactual = results['mu_10'] + results['delta_control']
    ax.plot([0, 1], [results['mu_10'], counterfactual], 'r--',
            alpha=0.5, linewidth=2, label='Counterfactual')

    # Annotate the ATT
    ax.annotate('', xy=(1, results['mu_11']), xytext=(1, counterfactual),
                arrowprops=dict(arrowstyle='<->', color='green', lw=2))
    ax.annotate(f'ATT = {results["att"]:.2f}',
                xy=(1.05, (results['mu_11'] + counterfactual) / 2),
                fontsize=12, color='green')

    ax.set_xticks(times)
    ax.set_xticklabels(time_labels)
    ax.set_xlabel('Time Period')
    ax.set_ylabel('Outcome')
    ax.set_title('Difference-in-Differences: 2x2 Design')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
