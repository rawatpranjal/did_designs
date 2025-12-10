# Module 04: Covariates and Doubly Robust DiD

> "When parallel trends fail unconditionally, fix the control group with propensity scores."

## 1. The Problem

The parallel trends assumption often fails because treated and control units are fundamentally different:
- Different baseline characteristics
- Different trends driven by those characteristics

**Example: LaLonde (1986) Job Training**
- Treated: Low-income workers who enrolled in job training
- Control: General population
- Problem: Workers who seek training have "dipping" earnings before enrollment (Ashenfelter dip)

Without adjustment, simple DiD gives **biased** (often wrong-sign) estimates.

## 2. The Target Parameter

The **Conditional** Average Treatment Effect on the Treated:

$$ATT = E[\tau | D=1] = E[Y_1(1) - Y_1(0) | D=1]$$

where we assume parallel trends holds **conditional on covariates X**:

$$E[Y_1(0) - Y_0(0) | D=1, X] = E[Y_1(0) - Y_0(0) | D=0, X]$$

## 3. Assumptions

The formal identification assumptions for doubly robust DiD (LaLonde job training):

1. **Conditional Parallel Trends:** Parallel trends holds **only** after conditioning on covariates $X$: $E[\Delta Y(0) | D=1, X] = E[\Delta Y(0) | D=0, X]$.
   - *Intuition:* Trainees and Non-Trainees have parallel earnings trends *only if* we compare people with the same age, education, and 1974 earnings.
   - *Note:* Comparing raw means creates bias; comparing workers with the same covariates removes bias.

2. **Strong Overlap (Common Support):** For every treated unit, there is a non-zero probability of being untreated. The Propensity Score $P(D=1|X)$ must be bounded away from 1.
   - *Intuition:* For every trainee, there exists a comparable non-trainee with similar covariate values.

3. **Covariate Exogeneity (No "Bad Controls"):** The covariates $X$ (e.g., education, pre-program earnings) are determined *before* the treatment and are not affected by the treatment itself.
   - *Intuition:* Pre-program earnings in 1974 were fixed before anyone knew about the training program.

4. **Double Robustness:** The estimator is consistent if **either** the Outcome Regression model (prediction of $Y$) **or** the Propensity Score model (prediction of $D$) is correctly specified.
   - *Intuition:* Both models don't need to be perfect—getting one right is sufficient for consistent estimation.

## 4. The Solution Methods

### Method 1: Inverse Probability Weighting (IPW)

1. Estimate propensity score: $e(X) = P(D=1 | X)$
2. Create weights: $w = \frac{e(X)}{1 - e(X)}$ for control units
3. Weight the control group to "look like" the treated group
4. Compute weighted DiD

### Method 2: Outcome Regression (OR)

1. Fit outcome model on control group: $E[Y | X, D=0]$
2. Predict counterfactual outcomes for treated units
3. DiD = Actual treated outcome - Predicted counterfactual

### Method 3: Doubly Robust (DR)

Combine IPW and OR:
- If propensity model is correct, DR is consistent
- If outcome model is correct, DR is consistent
- **Both** can be wrong, but DR protects you if **one** is right

$$\hat{ATT}_{DR} = \frac{1}{n_1} \sum_{i=1}^{n} \left[ D_i (\Delta Y_i - \hat{\mu}_0(X_i)) - \frac{\hat{e}(X_i)(1-D_i)}{1-\hat{e}(X_i)} (\Delta Y_i - \hat{\mu}_0(X_i)) \right]$$

where $\hat{e}(X_i)$ is the propensity score and $\hat{\mu}_0(X_i)$ is the predicted outcome change for controls.

## 5. The Data

**LaLonde (1986)** - National Supported Work Demonstration

Source: [MatchIt R package via Rdatasets](https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv)

| Variable | Description |
|----------|-------------|
| treat | 1 if in NSW job training program, 0 if PSID control |
| age | Age in years |
| educ | Years of education |
| race | "black", "hispan", or "white" |
| married | 1 if married |
| nodegree | 1 if no high school degree |
| re74, re75 | Real earnings 1974, 1975 (pre-treatment) |
| re78 | Real earnings 1978 (post-treatment) |

**Dataset size:** 614 observations (185 treated, 429 control)

**Key Feature:** Participants have declining earnings before treatment ("Ashenfelter dip" = selection bias).

## 6. Results & Visualization

### Propensity Score Distribution

![Propensity Scores](figs/propensity_scores.png)

Good overlap means we can find comparable control units.

### Comparison of Methods

| Method | ATT Estimate | Note |
|--------|--------------|------|
| Naive DiD | -$5,000 | Wrong sign! |
| IPW | +$1,500 | Reweighted controls |
| Outcome Regression | +$1,600 | Predicted counterfactual |
| Doubly Robust | +$1,550 | Protected estimate |

## 7. How to Run

```bash
cd 04_covariates_dr
python main.py
```

## 8. Key Takeaways

1. **Selection on observables:** If treatment is correlated with X, and X affects outcomes, simple DiD is biased
2. **Propensity scores:** Reweight controls to match treated on X
3. **Doubly robust:** Insurance against model misspecification
4. **Overlap matters:** Need control units with similar propensity scores as treated

## 9. When to Use Covariates

| Situation | Approach |
|-----------|----------|
| Randomized treatment | Simple DiD (Module 01) |
| Selection on observables | Covariates (this module) |
| Staggered + selection | CS with covariates |
| Selection on unobservables | Different methods needed |

## References

- Baker, A., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. (2025). A Practitioner's Guide to Difference-in-Differences. [[PDF]](../papers/Baker_etal_2025_Practitioners_Guide_DiD.pdf)
- Sant'Anna, P. H., & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *Journal of Econometrics*.
- LaLonde, R. J. (1986). Evaluating the econometric evaluations of training programs. *American Economic Review*.
