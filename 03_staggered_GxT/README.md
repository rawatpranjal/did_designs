# Module 03: Staggered Adoption (G×T Design)

> "The key insight: never compare treated units to other treated units."

## 1. The Problem

Standard Two-Way Fixed Effects (TWFE) regression fails when:
- Different groups are treated at **different times** (staggered adoption)
- Treatment effects are **heterogeneous** across groups or time

**Why TWFE fails:** It implicitly uses already-treated units as controls for newly-treated units. This creates **negative weights** and biased estimates.

## 2. The Target Parameter

We want Group-Time Average Treatment Effects:

$$ATT(g,t) = E[Y_t(1) - Y_t(0) | G = g]$$

where:
- $g$ = treatment cohort (year group was first treated)
- $t$ = calendar time
- $G = g$ means "units first treated in year $g$"

**Building Block:** Each $ATT(g,t)$ is a 2×2 comparison:
- Treated: Cohort $g$
- Control: Not-yet-treated units at time $t$
- Pre: Year $g-1$ (baseline)
- Post: Year $t$

## 3. Assumptions

The formal identification assumptions for staggered DiD (Medicaid expansion):

1. **Conditional Parallel Trends (Base: Not-Yet-Treated):** The evolution of outcomes for a cohort treated in year $g$ is parallel to the evolution of units *not yet treated* by year $t$.
   - *Intuition:* The 2014 expansion states would have followed the same health outcome trends as the 2016 or Never-Treated states, had they not expanded in 2014.
   - *Note:* This is weaker than requiring parallel trends with *never-treated* units.

2. **No Anticipation:** Outcomes do not change prior to the specific expansion year $g$ for any cohort.
   - *Intuition:* States did not change health outcomes via other pre-expansion programs before their official expansion date.

3. **Limited Treatment Effect Heterogeneity (Standard TWFE only):** Treatment effects are constant across time and cohorts.
   - *Intuition:* For standard regression only—treatment effects must be the same for all groups and periods.
   - *Note:* The Callaway-Sant'Anna estimator relaxes this assumption, which is why we use it.

4. **Random Sampling:** The data is a random sample from a super-population.
   - *Intuition:* Required for the bootstrap/influence-function inference used in the CS estimator.

## 4. The Data

**Minimum Wage Panel Data (mpdta)** from the `did` package.

| Variable | Description |
|----------|-------------|
| countyreal | County identifier |
| year | Year (2003-2007) |
| lemp | Log employment |
| first.treat | Year of first treatment (cohort) |
| treat | Treatment indicator |

**Treatment Cohorts:**
- 2004: Counties treated in 2004
- 2006: Counties treated in 2006
- 2007: Counties treated in 2007
- Never: Counties never treated

## 5. The Solution (Forward-Engineering)

### Step 1: Identify Building Blocks

For each cohort $g$ and time $t \geq g$:

```
ATT(g,t) = [mean(Y_t | G=g) - mean(Y_{g-1} | G=g)]
         - [mean(Y_t | NotYetTreated) - mean(Y_{g-1} | NotYetTreated)]
```

### Step 2: Use Not-Yet-Treated Controls

**Critical:** Control group = units not yet treated by time $t$

This includes:
- Never-treated units
- Units treated after time $t$

This avoids the "bad comparison" problem of TWFE.

### Step 3: Aggregate to Summary Measures

Common aggregations:

| Aggregation | Formula | Interpretation |
|-------------|---------|----------------|
| Simple | $\frac{1}{|G|} \sum_{g,t} ATT(g,t)$ | Overall average effect |
| Event | $\frac{1}{|G|} \sum_g ATT(g, g+e)$ | Effect at event-time $e$ |
| Cohort | $\frac{1}{T_g} \sum_{t \geq g} ATT(g,t)$ | Average for cohort $g$ |

## 6. Results & Visualization

### The $ATT(g,t)$ Matrix

|        | 2004 | 2005 | 2006 | 2007 |
|--------|------|------|------|------|
| g=2004 | ATT  | ATT  | ATT  | ATT  |
| g=2006 |  .   |  .   | ATT  | ATT  |
| g=2007 |  .   |  .   |  .   | ATT  |

*Note: Cells where $t < g$ are pre-trends (placebo tests). Cells where $t \geq g$ are treatment effects.*

### Aggregated Event Study

![Staggered Event Study](figs/staggered_event_study.png)

## 7. How to Run

```bash
cd 03_staggered_GxT
python main.py
```

## 8. Key Takeaways

1. **TWFE is problematic:** With heterogeneous effects, TWFE gives biased estimates
2. **Building blocks:** $ATT(g,t)$ is just a 2×2 with the right control group
3. **Not-yet-treated:** Use units not yet treated as controls
4. **Aggregation matters:** Different summaries answer different questions

## 9. Why TWFE Fails (Goodman-Bacon Decomposition)

TWFE coefficient $\hat{\delta}$ is a weighted average of:
1. **Good comparisons:** Treated vs. never-treated
2. **Bad comparisons:** Newly-treated vs. already-treated

With heterogeneous effects, bad comparisons get **negative weights**, biasing $\hat{\delta}$ toward zero or even wrong sign.

---

## Appendix: Computation Results

| Method | Estimate | Notes |
|:-------|:---------|:------|
| Manual ATT(2004,2004) | -0.0194 | Cohort 2004, event time 0 |
| Manual ATT(2004,2005) | -0.0783 | Cohort 2004, event time 1 |
| CS Simple ATT (aggregated) | -0.0568 | Average of post-treatment ATT(g,t) |
| TWFE (demeaned) | -0.0365 | Biased toward zero |

**Verdict:** TWFE (-0.0365) differs from CS (-0.0568). TWFE biased due to bad comparisons (already-treated as controls).

## References

- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*. [[PDF]](../papers/Callaway_SantAnna_2021_DiD_Multiple_Periods.pdf)
- Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. (2025). A Practitioner's Guide to Difference-in-Differences. [[PDF]](../papers/Baker_etal_2025_Practitioners_Guide_DiD.pdf)
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*.
