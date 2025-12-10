# Module 07: Synthetic Difference-in-Differences

> "SDID is just weighted DiD—but the weights are chosen to make the parallel trends assumption more plausible."

## 1. The Problem

You want to evaluate a policy that affected **one unit** (e.g., California). Standard methods have limitations:

### Naive DiD
```
ATT = (CA_post - CA_pre) - (Avg_Other_post - Avg_Other_pre)
```
**Problem:** The average of other states may be a poor control for California. If pre-trends don't match, the parallel trends assumption fails.

### Synthetic Control (SC)
Reweights control units to match California's pre-treatment trajectory:
```
Synthetic_CA = 0.3 × Utah + 0.2 × Nevada + ...
```
**Problem:** SC ignores time-varying unobservables. It's not invariant to additive shifts in outcomes (fixed effects).

### The Solution: SDID

**Synthetic Difference-in-Differences** combines the strengths of both:
1. **Unit Weights (ω):** Like SC, reweight control units to match pre-trends
2. **Time Weights (λ):** NEW! Reweight pre-treatment periods to resemble post-treatment
3. **Double Robustness:** Combines two-way fixed effects with synthetic control weighting

## 2. The Math

The SDID estimator is:

$$\hat{\tau}_{sdid} = (\bar{Y}_{T,Post} - \bar{Y}^{\lambda}_{T,Pre}) - (\bar{Y}^{\omega}_{C,Post} - \bar{Y}^{\omega,\lambda}_{C,Pre})$$

where:
- $\bar{Y}_{T,Post}$ = Treated unit, post-treatment (simple mean)
- $\bar{Y}^{\lambda}_{T,Pre}$ = Treated unit, pre-treatment (time-weighted)
- $\bar{Y}^{\omega}_{C,Post}$ = Control units, post-treatment (unit-weighted)
- $\bar{Y}^{\omega,\lambda}_{C,Pre}$ = Control units, pre-treatment (doubly weighted)

### Unit Weights (ω)

Find ω such that the synthetic control matches the treated unit's pre-trend:

$$\min_{\omega} \|Y_{T,pre} - \omega_0 - Y_{C,pre}\omega\|^2 + \zeta^2 T_{pre} \|\omega\|^2_2$$

### Time Weights (λ)

Find λ such that weighted pre-periods resemble post-periods:

$$\min_{\lambda} \|\bar{Y}_{C,post} - \lambda_0 - Y^T_{C,pre}\lambda\|^2$$

## 3. The Data

We use the **California Proposition 99** dataset (same as Module 02):

| Variable | Description |
|----------|-------------|
| `state` | State name (California = treated) |
| `year` | Year (1970-2000) |
| `cigsale` | Cigarette sales per capita (packs) |

**Treatment:** Proposition 99 (1988) — 25-cent tobacco tax increase

## 4. Results

| Method | Estimate | Paper | Notes |
|--------|----------|-------|-------|
| Naive DiD | -27.35 | -27.3 | Biased (Control states differ from CA) |
| Synthetic Control | -19.45 | -19.6 | Classic SC (Abadie et al. 2010) |
| SDID | -15.60 | -15.6 | Unit + time weights (Arkhangelsky et al. 2021) |

Paper: Arkhangelsky et al. (2021) Table 1.

### Visualization

### 1. Synthetic vs. Real Trends (The "Hero" Plot)
*   **Blue Line:** Actual California cigarette sales.
*   **Red Line:** "Synthetic California" (a weighted average of other states).
*   **Pre-1989:** The red line tracks the blue line perfectly. This validates the method.
*   **Post-1989:** The gap represents the causal effect of Prop 99.

![SDID Dashboard](figs/sdid_dashboard.png)

### 2. The Performance Contest
Comparing the three major methods.
*   **Naive DiD:** Overestimates the effect (too negative) because control states were already declining faster than CA.
*   **Synthetic Control:** Improves the estimate.
*   **SDID:** Provides the most robust estimate by combining reweighting with fixed effects.

![SDID Comparison](figs/sdid_comparison.png)

## 5. Key Insights

1. **SDID is still a 2×2 DiD** — four numbers, just weighted differently.

2. **Unit weights create a better control group** — like synthetic control.

3. **Time weights focus on relevant periods** — down-weight early years that differ from post-treatment.

4. **Intercepts matter:** SDID allows for a "parallel shift" (α), whereas SC forces levels to match exactly. This makes SDID more robust to level differences.

## 6. How to Run

```bash
cd 07_synthetic_did
python main.py
```

---

## Appendix: Computation Results

| Method | Estimate | Paper | Package |
|:-------|:---------|:------|:--------|
| Naive DiD | -27.35 | -27.3 | Manual |
| Synthetic Control | -19.45 | -19.6 | pysyncon |
| SDID | -15.60 | -15.6 | synthdid |

**Verdict:** All estimates match Arkhangelsky et al. (2021) Table 1. SDID gives the most credible estimate by combining DiD's fixed effects with SC's pre-trend matching.

## References

- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S. (2021). "Synthetic Difference-in-Differences." *American Economic Review*.

- Abadie, A., Diamond, A., & Hainmueller, J. (2010). "Synthetic Control Methods for Comparative Case Studies." *Journal of the American Statistical Association*.
