# Module 01: The Canonical 2x2 Design

> "Every complex DiD design is just a weighted average of simple 2x2 comparisons."

## 1. The Problem

We want to estimate the causal effect of a policy that affects one group at one specific time. This is the foundational case of Difference-in-Differences.

**Setting:** New Jersey raised its minimum wage from $4.25 to $5.05 in April 1992. Pennsylvania did not change its minimum wage. We want to know: did the minimum wage increase affect employment?

## 2. The Target Parameter

The Average Treatment Effect on the Treated (ATT) is:

$$\widehat{\text{ATT}} = \underbrace{(\bar{y}_{\text{NJ}, \text{Post}} - \bar{y}_{\text{NJ}, \text{Pre}})}_{\text{Change in Treated}} - \underbrace{(\bar{y}_{\text{PA}, \text{Post}} - \bar{y}_{\text{PA}, \text{Pre}})}_{\text{Change in Control}}$$

Where $\bar{y}$ represents the sample mean for that group and time period.

This is **"The Four Numbers"** approach. DiD is just the difference of two differences.

## 3. Assumptions

The formal identification assumptions for the canonical 2x2 DiD (Card & Krueger minimum wage study):

1. **Parallel Trends (Unconditional):** In the absence of the minimum wage increase, the average change in employment in New Jersey (Treatment) would have been equal to the average change in Pennsylvania (Control).
   - *Intuition:* Employment trends in NJ would have matched PA had the wage hike not occurred.

2. **No Anticipation:** Employers in New Jersey did not adjust employment levels *before* the law went into effect.
   - *Intuition:* NJ restaurants did not fire workers in Feb 1992 anticipating the Nov 1992 hike.

3. **SUTVA (Stable Unit Treatment Value Assumption):**
   - **No Interference:** The minimum wage hike in NJ did not affect employment in PA (e.g., no cross-border labor migration or general equilibrium price effects).
   - **Consistency:** The observed outcome for a treated unit is the potential outcome under treatment (no "hidden versions" of the treatment).
   - *Intuition:* Workers didn't flee PA to work in NJ, or vice versa.

4. **Exogeneity of Selection:** The decision to treat New Jersey was not determined by time-varying unobservables that also affect employment trends.
   - *Intuition:* NJ didn't raise wages *because* they forecasted an economic boom.

## 4. The Data

**Card & Krueger (1994)** - The classic minimum wage study.

| Variable | Description |
|----------|-------------|
| state | 0 = Pennsylvania (control), 1 = New Jersey (treatment) |
| time | 0 = February 1992 (pre), 1 = November 1992 (post) |
| fte | Full-time equivalent employment |

**Sample sizes:**
- Control (PA): ~70 restaurants observed twice
- Treatment (NJ): ~300 restaurants observed twice

## 5. The Solution (Forward-Engineering)

### Step 1: Calculate the Four Means

```
E[Y | Treated, Post]  = μ₁₁
E[Y | Treated, Pre]   = μ₁₀
E[Y | Control, Post]  = μ₀₁
E[Y | Control, Pre]   = μ₀₀
```

### Step 2: Compute First Differences

```
Change in Treated:  Δ_T = μ₁₁ - μ₁₀
Change in Control:  Δ_C = μ₀₁ - μ₀₀
```

### Step 3: Difference-in-Differences

```
ATT = Δ_T - Δ_C
```

### Step 4: Verify with Regression

The regression specification:

$$Y_{it} = \alpha + \beta D_i + \gamma Post_t + \delta (D_i \times Post_t) + \varepsilon_{it}$$

The coefficient $\delta$ is **numerically identical** to our manual calculation.

## 6. Results & Visualization

### The Counterfactual Wedge
This chart visualizes the core logic of DiD. The dashed red line represents the **counterfactual**: what would have happened to New Jersey if it had followed Pennsylvania's trend. The green arrow is the **ATT**.

![DiD Showcase](figs/did_showcase.png)

### The 2x2 Table

|           | Pre (Feb 1992) | Post (Nov 1992) | Change |
|-----------|----------------|-----------------|--------|
| Control (PA) | 23.33 | 21.17 | -2.16 |
| Treatment (NJ) | 20.44 | 21.03 | +0.59 |
| **Diff-in-Diff** | | | **+2.76** |

### Interpretation

The minimum wage increase in NJ was associated with a **2.75 FTE increase** in employment relative to what would have happened without the policy (the Pennsylvania trend).

This contradicts the standard economic prediction that minimum wage increases reduce employment.

![DiD Visualization](figs/did_2x2.png)

## 7. How to Run

```bash
cd 01_canonical_2x2
python main.py
```

## 8. Key Takeaways

1. **DiD is simple:** It's just four numbers and some subtraction
2. **Regression is a calculator:** The OLS coefficient equals the manual calculation
3. **The intuition:** We use the control group's change to construct the counterfactual for the treated group

---

## Appendix: Computation Results

| Method | Estimate | Notes |
|:-------|:---------|:------|
| Manual DiD ("Four Numbers") | +2.7536 | (21.03 - 20.44) - (21.17 - 23.33) |
| OLS Regression (δ) | +2.7536 | `fte ~ treated * post` |

**Verdict:** Exact match (diff = 0.0000). OLS is just a convenient calculator for the manual formula.

## References

- Card, D., & Krueger, A. B. (1994). Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania. *American Economic Review*, 84(4), 772-793. [[PDF]](../papers/Card_Krueger_1994_Minimum_Wage_Employment.pdf)
