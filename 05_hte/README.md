# Module 05: Heterogeneous Treatment Effects and Triple Difference

> "HTE is just running DiD twice. DDD is the difference between two DiDs."

## 1. The Problem

### Part A: Heterogeneous Treatment Effects (HTE)

Does the treatment effect vary across subgroups?
- By demographic (men vs. women)
- By region (urban vs. rural)
- By baseline characteristics (high vs. low income)

### Part B: Triple Difference (DDD)

Standard DiD may be biased if there's a shock affecting all treated units. DDD adds a **within-state control group** to difference out this bias.

**Example:**
- Policy: Medicaid expansion
- Target: Low-income adults (eligible)
- Placebo: High-income adults (ineligible but same state)

If high-income adults in expansion states also improve, that's a state-level shock, not the policy.

## 2. The Target Parameters

### HTE

For subgroup $s$:
$$ATT_s = E[Y(1) - Y(0) | D=1, S=s]$$

Test: $ATT_{s=1} \neq ATT_{s=0}$?

### DDD

$$DDD = DiD_{\text{eligible}} - DiD_{\text{placebo}}$$

Expanded:
$$DDD = \underbrace{[(Y_{T,Post,E} - Y_{T,Pre,E}) - (Y_{C,Post,E} - Y_{C,Pre,E})]}_{\text{DiD for Eligible}}
     - \underbrace{[(Y_{T,Post,P} - Y_{T,Pre,P}) - (Y_{C,Post,P} - Y_{C,Pre,P})]}_{\text{DiD for Placebo}}$$

## 3. Assumptions

The formal identification assumptions for heterogeneous treatment effects in DiD:

1. **Subgroup Validity:** The characteristic defining the subgroup (e.g., Sex, Race) is exogenous and not affected by the treatment.
   - *Intuition:* The treatment doesn't change who belongs to which subgroup.

2. **Subgroup Parallel Trends:** Parallel trends holds *within* each subgroup.
   - *Intuition:* Treated Women would have followed the same trend as Control Women; Treated Men would have followed Control Men.

3. **Effect Stability (Sampling Stability):** The distribution of the subgroup does not change over time in a way that correlates with the outcome.
   - *Intuition:* The ratio of Men to Women doesn't suddenly shift in the treated state due to the policy (e.g., the workforce doesn't become 90% male because of the treatment).

## 4. The Data

We extend the Medicaid data with simulated income groups:

| Variable | Description |
|----------|-------------|
| state | State identifier |
| year | Year |
| treated | 1 if expansion state |
| post | 1 if post-expansion |
| eligible | 1 if low-income (targeted group) |
| y | Outcome (e.g., insurance coverage) |

## 5. The Solution (Forward-Engineering)

### HTE: Split-Sample DiD

```python
# Step 1: Split by subgroup
df_men = df[df['gender'] == 'male']
df_women = df[df['gender'] == 'female']

# Step 2: Run DiD for each
att_men = did_2x2(df_men)
att_women = did_2x2(df_women)

# Step 3: Compare
difference = att_men - att_women
```

### DDD: Difference of DiDs

```python
# Step 1: DiD for eligible (low income)
df_eligible = df[df['eligible'] == 1]
did_eligible = did_2x2(df_eligible)

# Step 2: DiD for placebo (high income)
df_placebo = df[df['eligible'] == 0]
did_placebo = did_2x2(df_placebo)

# Step 3: DDD
ddd = did_eligible - did_placebo
```

## 6. Results & Visualization

### HTE Results

| Subgroup | ATT | SE | Significant? |
|----------|-----|----|--------------|
| Low Income | +0.15 | 0.02 | Yes |
| High Income | +0.02 | 0.03 | No |

### DDD Decomposition

```
DiD (Eligible):   +0.15  (Policy + State shock)
DiD (Placebo):    +0.02  (State shock only)
─────────────────────────────────────────────
DDD:              +0.13  (Pure policy effect)
```

![HTE Comparison](figs/hte_comparison.png)

## 7. How to Run

```bash
cd 05_hte_and_ddd
python main.py
```

## 8. Key Takeaways

1. **HTE is simple:** Split sample, run DiD, compare
2. **DDD removes bias:** Differences out state-level confounders
3. **Placebo validity:** Placebo group must be unaffected by treatment
4. **Still uses 2×2:** Each component is still a 2×2 comparison

## 9. Using `differences` Package

```python
from differences import ATTgt

# HTE: Split by sample
att_gt = ATTgt(data=df, cohort_name='cohort')
att_gt.fit(formula='y', split_sample_by='income_group')
att_gt.aggregate('simple')

# DDD: Difference between groups
att_gt.aggregate('event', difference=['low_income', 'high_income'])
```

## Important Note: Covariate Adjustment

The naive DDD approach in this module assumes that the **target and placebo groups have similar covariate distributions**. When this assumption fails (e.g., women and men have different education levels), the naive DDD estimator can be biased.

For a **Doubly Robust DDD** approach that handles covariate imbalance, see **[Module 06: Triple Differences - The Robust Way](../06_triple_diff_dr/)**, which implements the Ortiz-Villavicencio & Sant'Anna (2025) estimator.

## References

- Gruber, J. (1994). The incidence of mandated maternity benefits. *American Economic Review*.
- Callaway, B., Goodman-Bacon, A., & Sant'Anna, P. H. (2021). Difference-in-differences with a continuous treatment.
- Ortiz-Villavicencio, A. & Sant'Anna, P. H. C. (2025). Doubly Robust Difference-in-Difference-in-Differences Estimators.
