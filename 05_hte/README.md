# Module 05: Heterogeneous Treatment Effects (HTE)

> "The average effect often hides the most interesting story. Dig deeper with Split-Sample DiD."

## 1. The Problem

Standard Difference-in-Differences estimates the **Average** Treatment Effect on the Treated (ATT). However, policies rarely affect everyone equally.

**Heterogeneity** asks:
- Does the policy work better for demographic A vs. demographic B?
- Is the effect stronger in urban areas vs. rural areas?
- Did the Minimum Wage hurt employment in small counties more than large counties?

Ignoring heterogeneity can mask important distributional consequences of a policy.

## 2. The Target Parameter

Instead of one single $\delta$, we want to estimate the ATT for specific subgroups $g$:

$$ATT_g = E[Y(1) - Y(0) | D=1, Group=g]$$

We define these subgroups based on baseline characteristics (covariates) determined *before* the treatment occurs.

## 3. The Solution: Split-Sample DiD

The most robust and transparent way to estimate HTE is simply to **split the sample** and run the DiD analysis separately for each subgroup.

1.  **Define Subgroups:** Split the data into $D_{High}$ and $D_{Low}$ based on a covariate $X$ (e.g., population size).
2.  **Estimate:** Run `calculate_2x2_did` separately for each dataframe.
3.  **Compare:** Calculate the difference $\Delta = \widehat{ATT}_{High} - \widehat{ATT}_{Low}$.
4.  **Test:** Check if the confidence intervals overlap or perform a formal t-test on the difference.

## 4. The Data (mpdta)

We use the Minimum Wage dataset again, but we stratify counties by **population size**:
*   **High Population:** Counties above median population.
*   **Low Population:** Counties below median population.

*Hypothesis:* Larger counties might have more resilient labor markets, showing smaller employment effects from minimum wage hikes.

## 5. Results & Visualization

### 1. Subgroup Trends
We plot the raw trends separately for High Population and Low Population counties.
*   **Visual Check:** Do the trends look parallel in the pre-period (2003-2004) for both groups?
*   **Mechanism:** This helps us see *why* the effect might differ.

![HTE Trends](figs/hte_trends.png)

### 2. HTE Forest Plot
This is the standard way to visualize heterogeneity. It shows the ATT for each subgroup relative to the overall pooled effect.
*   **Pooled ATT:** The dashed line shows the effect if we ignored heterogeneity.
*   **Subgroup ATTs:** The dots show the specific effects. Overlapping confidence intervals suggest no significant difference.

![HTE Forest Plot](figs/hte_forest.png)

### 3. Decomposition (Triple Difference)
We explicitly calculate the difference between the two subgroups ($ATT_{High} - ATT_{Low}$).
*   **Result:** The difference is very small (near zero).
*   **Conclusion:** We do not find evidence that the minimum wage affects large and small counties differently.

![HTE Decomposition](figs/hte_decomposition.png)

### Interpretation
*   **Pooled ATT:** The dashed line shows the effect if we ignored heterogeneity.
*   **Subgroup ATTs:** The green dots show the specific effects. If the error bars for the subgroups are far apart, there is evidence of heterogeneity.

| Subgroup | ATT | Interpretation |
|----------|-----|----------------|
| **Low Pop** | **-0.038** | Larger negative effect |
| **High Pop** | **-0.076** | Even larger negative effect |

*Note: In this specific 2003 vs 2007 cut of the data, High Pop counties actually show a larger decline.*

## 6. How to Run

```bash
cd 05_hte
python main.py
```

---

## Appendix: Computation Results

| Subgroup | ATT | SE | Notes |
|:---------|:----|:---|:------|
| Pooled (all counties) | -0.0385 | — | Average effect |
| Low Population | -0.0449 | 0.178 | Below median lpop |
| High Population | -0.0465 | 0.206 | Above median lpop |
| **Difference (High - Low)** | **-0.0016** | — | — |

**Verdict:** No significant heterogeneity by county size (CIs overlap substantially).

## References
*   Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.
*   Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. (2025). A Practitioner's Guide to Difference-in-Differences. [[PDF]](../papers/Baker_etal_2025_Practitioners_Guide_DiD.pdf)
