# Module 06: Triple Differences - The Robust Way

> "The 'folk wisdom' that DDD = DiD_target - DiD_placebo is incorrect when covariates matter."

## 1. The Problem

You want to study the effect of a policy on a specific group within treated states. You're worried that treated states might experience economic shocks affecting everyone, so you use a **placebo group** (unaffected by the policy) as a within-state control.

### The Naive Approach
Run a regression with a triple interaction: `Y ~ State × Post × Group + controls`.

### The Trap (Ortiz-Villavicencio & Sant'Anna 2025)
**If trends depend on covariates and the covariate distribution differs between target and placebo groups, the naive method fails.**

Why? The naive DDD integrates counterfactual trends over the **wrong population**. When you compute DiD for the placebo group, you're using *their* covariate distribution. But to construct a valid counterfactual for the target group, you need predictions **at the target group's covariate distribution**.

## 2. The Solution: Outcome Regression DDD

We use a "Target-Adjusted" estimator. Instead of 2 simple DiDs, we model the outcome for all groups and predict what their trends *would be* if they had the characteristics of the Target Group.

**Formula:**
$ \widehat{DDD}_{OR} = (\bar{Y}_{T} - \hat{Y}_{T}^{CF1}) - (\hat{Y}_{T}^{CF2} - \hat{Y}_{T}^{CF3}) $

Where all $\hat{Y}$ are predicted outcomes evaluated at the Target Group's covariate values ($X_{Target}$).

## 3. The Data

We use the **Meyer, Viscusi, & Durbin (1995)** worker's compensation dataset.
*   **Policy:** Kentucky raised the benefit cap (Treated State). Michigan did not (Control State).
*   **Target:** High Earners (benefits increased).
*   **Placebo:** Low Earners (benefits unchanged).

**The Imbalance:** High earners are systematically older (+4 years) and more likely to be married (+27 pp) than low earners.

## 4. Results

| Method | Estimate | Interpretation |
|--------|----------|----------------|
| **Naive DDD (OLS)** | **+0.036** | Small positive effect |
| **Robust DDD (OR)** | **-0.118** | Large negative effect |

**Conclusion:** Adjusting for the fact that high earners are older and married (characteristics associated with different injury duration trends) significantly changes the result. The naive estimator was biased upward.

## 5. Visualizations

### Covariate Imbalance
The root cause of the bias: The placebo group looks nothing like the target group.

![Imbalance](figs/covariate_imbalance.png)

### Robust Decomposition
This shows the trends *after* adjusting everyone to look like the Target Group.

![Decomposition](figs/ddd_decomposition.png)

## 6. How to Run

```bash
cd 06_triple_diff_dr
python main.py
```

---

## Appendix: Computation Results

| Method | Estimate | SE | Notes |
|:-------|:---------|:---|:------|
| Naive DDD (OLS) | +0.036 | 0.173 | 3-way FE with controls |
| Robust DDD (Target-Adj) | -0.118 | 0.205 | OR at target covariates |

**Verdict:** ~0.15 difference reflects covariate adjustment for age/marriage imbalance.

## References
*   **Ortiz-Villavicencio, A. & Sant'Anna, P. H. C. (2025).** Better Understanding Triple Differences Estimators. [[PDF]](../papers/Ortiz_SantAnna_2025_DR_DDD.pdf)
*   **Meyer, B. D., Viscusi, W. K., & Durbin, D. L. (1995).** Workers' Compensation and Injury Duration. *American Economic Review*.
