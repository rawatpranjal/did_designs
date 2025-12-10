# Modern Difference-in-Differences

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Implementations of modern DiD estimators following Baker, Callaway, Cunningham, Goodman-Bacon, and Sant'Anna (2025).

## Modules

### 01. Canonical 2x2
Card & Krueger (1994) — NJ/PA Minimum Wage

**Estimator:**
$$\hat{\delta}_{2\times2} = (\bar{y}_{T,post} - \bar{y}_{T,pre}) - (\bar{y}_{C,post} - \bar{y}_{C,pre})$$

**Regression:**
$$Y_{it} = \alpha + \beta \text{Treat}_i + \gamma \text{Post}_t + \delta_{DiD} (\text{Treat}_i \times \text{Post}_t) + \varepsilon_{it}$$

**Estimand:**
$$ATT = \underbrace{(\mathbb{E}[Y_{2}|D=1] - \mathbb{E}[Y_{1}|D=1])}_{\text{Actual Path}} - \underbrace{(\mathbb{E}[Y_{2}|D=0] - \mathbb{E}[Y_{1}|D=0])}_{\text{Counterfactual Trend}}$$

| | | |
|:--:|:--:|:--:|
| ![](01_canonical_2x2/figs/did_2x2.png) | ![](01_canonical_2x2/figs/did_showcase.png) | ![](01_canonical_2x2/figs/card_krueger_did.png) |
| ![](01_canonical_2x2/figs/card_krueger_detailed.png) | | |

Static 2x2 design. Manual calculation vs. OLS verification.

[Code](01_canonical_2x2/) | [Notes](01_canonical_2x2/README.md)

---

### 02. Event Study (2×T)
California Prop 99 (1988) — Tobacco Tax

**Estimator:**
$$Y_{it} = \mu_i + \lambda_t + \sum_{k \neq -1} \delta_k \cdot \mathbf{1}\{t - T_i = k\} + \varepsilon_{it}$$

**Regression:**
$$Y_{it} = \mu_i + \lambda_t + \sum_{k \neq -1} \beta_k \cdot \mathbb{1}\{t - T_{treat} = k\} + \varepsilon_{it}$$

**Estimand:**
$$ATT(t) = \mathbb{E}[Y_{t} - Y_{g-1} \mid D=1] - \mathbb{E}[Y_{t} - Y_{g-1} \mid D=0]$$

| | | |
|:--:|:--:|:--:|
| ![](02_event_study_2xT/figs/event_study.png) | ![](02_event_study_2xT/figs/event_study_showcase.png) | ![](02_event_study_2xT/figs/raw_trends.png) |

Dynamic treatment effects with pre-trend testing. Reference period normalized to t = −1.

[Code](02_event_study_2xT/) | [Notes](02_event_study_2xT/README.md)

---

### 03. Staggered Adoption (G×T)
mpdta — US Counties Minimum Wage (2003-2007)

**Estimator (Callaway-Sant'Anna):**
$$ATT(g,t) = \mathbb{E}[Y_t - Y_{g-1} \mid G=g] - \mathbb{E}[Y_t - Y_{g-1} \mid C=NYT]$$

**Regression (Saturated):**
$$Y_{it} = \mu_i + \lambda_t + \sum_{g} \sum_{t} \tau_{g,t} \cdot \mathbb{1}\{G_i = g\} \cdot \mathbb{1}\{Period = t\} + \varepsilon_{it}$$

**Estimand:**
$$ATT(g,t) = \mathbb{E}[Y_{t} - Y_{g-1} \mid G=g] - \mathbb{E}[Y_{t} - Y_{g-1} \mid G \in \mathcal{C}_{NYT}]$$

| | | |
|:--:|:--:|:--:|
| ![](03_staggered_GxT/figs/staggered_event_study.png) | ![](03_staggered_GxT/figs/att_gt_matrix.png) | ![](03_staggered_GxT/figs/raw_trends_by_cohort.png) |

Callaway & Sant'Anna ATT(g,t) aggregation. Not-yet-treated as controls.

[Code](03_staggered_GxT/) | [Notes](03_staggered_GxT/README.md)

---

### 04. Doubly Robust DiD
LaLonde (1986) — NSW Job Training

**Estimator (Sant'Anna & Zhao 2020):**
$$\hat{\tau}^{DR} = \frac{1}{N_{tr}} \sum_{i} \left( \frac{D_i(Y_i - \hat{\mu}_{0,i})}{\hat{p}} - \frac{(1-D_i)\hat{p}(X_i)(Y_i - \hat{\mu}_{0,i})}{(1-\hat{p}(X_i))\hat{p}} \right)$$

**Regression (Weighted):**
$$\Delta Y_i = \alpha + \mathbf{X}_i'\beta + \varepsilon_i \quad \text{weighted by } w_i = \frac{D_i + (1-D_i)\hat{p}(X_i)}{1-\hat{p}(X_i)}$$

**Estimand:**
$$ATT = \mathbb{E}[\Delta Y \mid D=1] - \mathbb{E}\big[ \mathbb{E}[\Delta Y \mid X, D=0] \big| D=1 \big]$$

| | | |
|:--:|:--:|:--:|
| ![](04_covariates_dr/figs/method_comparison.png) | ![](04_covariates_dr/figs/propensity_scores.png) | ![](04_covariates_dr/figs/pscore_overlap.png) |
| ![](04_covariates_dr/figs/earnings_trends.png) | ![](04_covariates_dr/figs/earnings_trends_dip.png) | ![](04_covariates_dr/figs/results_comparison.png) |

IPW, outcome regression, and DR estimation for conditional parallel trends.

[Code](04_covariates_dr/) | [Notes](04_covariates_dr/README.md)

---

### 05. Heterogeneous Treatment Effects
mpdta — US Counties by Population Size

**Estimator:**
$$\widehat{ATT}(x) = (\bar{y}_{T,post|x} - \bar{y}_{T,pre|x}) - (\bar{y}_{C,post|x} - \bar{y}_{C,pre|x})$$

**Regression:**
$$Y_{it} = \mu_i + \lambda_t + \delta D_{it} + \eta (D_{it} \times \text{Subgroup}_i) + \varepsilon_{it}$$

**Estimand:**
$$ATT(x) = \mathbb{E}[Y_{post} - Y_{pre} \mid D=1, X=x] - \mathbb{E}[Y_{post} - Y_{pre} \mid D=0, X=x]$$

| | |
|:--:|:--:|
| ![](05_hte/figs/hte_forest.png) | ![](05_hte/figs/hte_trends.png) |

Split-sample DiD for subgroup analysis. Forest plot visualization of heterogeneity.

[Code](05_hte/) | [Notes](05_hte/README.md)

---

### 06. Robust Triple Differences
Meyer, Viscusi, & Durbin (1995) — Worker's Compensation

**Estimator (Target-Adjusted OR):**
$$\hat{\tau}^{DDD}_{OR} = \bar{Y}_{T,Target} - \frac{1}{N_{T,Target}} \sum_{i \in T,Target} \hat{\mu}_{0}(X_i)$$

**Regression (Cell-Specific):**
$$Y_{i} = \alpha + \mathbf{X}_i'\beta_{s,g,t} + \varepsilon_{i} \quad \text{for each state } s, \text{ group } g, \text{ time } t$$

**Estimand:**
$$ATT(g,t) = \mathbb{E}[\Delta Y_{Target}^{Treat}] - \mathbb{E}\big[ \mathbb{E}[\Delta Y \mid X, S \in \mathcal{C}] \mid S=g, Q=1 \big]$$

| | | |
|:--:|:--:|:--:|
| ![](06_triple_diff_dr/figs/ddd_comparison.png) | ![](06_triple_diff_dr/figs/covariate_imbalance.png) | ![](06_triple_diff_dr/figs/ddd_decomposition.png) |

DR-DDD following Ortiz-Villavicencio & Sant'Anna (2025). Corrects for covariate imbalance between target and placebo groups.

[Code](06_triple_diff_dr/) | [Notes](06_triple_diff_dr/README.md)

---

### 07. Synthetic DiD
California Prop 99 (1988) — Tobacco Tax

**Estimator:**
$$\hat{\tau}^{sdid} = (\bar{Y}_{T,post} - \sum_{t} \hat{\lambda}_t Y_{T,t}) - (\sum_{i} \hat{\omega}_i Y_{i,post} - \sum_{i,t} \hat{\omega}_i \hat{\lambda}_t Y_{i,t})$$

**Regression (Weighted TWFE):**
$$(\hat{\tau}, \hat{\alpha}, \hat{\beta}) = \operatorname*{argmin}_{\tau, \alpha, \beta} \sum_{i=1}^N \sum_{t=1}^T \hat{\omega}_i \hat{\lambda}_t \left( Y_{it} - \alpha_i - \beta_t - \tau D_{it} \right)^2$$

**Estimand:**
$$\hat{\tau}^{sdid} = \left( \bar{Y}_{1}^{post} - \sum_{t} \hat{\lambda}_t Y_{1,t} \right) - \left( \sum_{i} \hat{\omega}_i Y_{i}^{post} - \sum_{i,t} \hat{\omega}_i \hat{\lambda}_t Y_{i,t} \right)$$

| |
|:--:|
| ![](07_synthetic_did/figs/sdid_dashboard.png) |

SDID (Arkhangelsky et al. 2021). Unit and time weights for single treated unit settings.

[Code](07_synthetic_did/) | [Notes](07_synthetic_did/README.md)

---

## Usage

```bash
pip install -r requirements.txt
python 01_canonical_2x2/main.py
```

## Datasets

| Dataset | Source | Module |
|:--------|:-------|:-------|
| Card & Krueger | Card & Krueger (1994) | 01 |
| California Prop 99 | Abadie et al. (2010) | 02, 07 |
| mpdta | Callaway & Sant'Anna (2021) | 03 |
| LaLonde NSW | LaLonde (1986) | 04 |
| mpdta (HTE) | Callaway & Sant'Anna (2021) | 05 |
| Meyer et al. (1995) | Worker's Comp | 06 |
| Castle Doctrine | Cheng & Hoekstra (2013) | — |

## Computation Results

All estimates from running each module. Reported faithfully for objective comparison.

### Module 01: Canonical 2×2 (Card & Krueger)

| Method | Estimate | Notes |
|:-------|:---------|:------|
| Manual DiD ("Four Numbers") | 2.7536 | (21.03 - 23.33) - (20.44 - 20.44) |
| OLS Regression (δ coefficient) | 2.7536 | `fte ~ treated * post` |

**Verdict:** Exact match (diff = 0.0000)

---

### Module 02: Event Study (Prop 99)

| Method | Estimate | Notes |
|:-------|:---------|:------|
| Manual ATT(t=0, 1988) | -3.64 | First post-treatment year |
| Regression ATT(t=0) | -3.64 | Event study coefficient |
| Manual ATT(t=1, 1989) | -7.18 | |
| Regression ATT(t=1) | -7.18 | |
| Avg Post-Treatment ATT | -20.24 | Average over 1988-2000 |

**Verdict:** Manual = Regression (correlation 1.000)

---

### Module 03: Staggered DiD (mpdta)

| Method | Estimate | Notes |
|:-------|:---------|:------|
| Manual ATT(2004,2004) | -0.0194 | Cohort 2004, event time 0 |
| Manual ATT(2006,2006) | +0.0047 | Cohort 2006, event time 0 |
| Manual ATT(2007,2007) | -0.0261 | Cohort 2007, event time 0 |
| Manual CS Simple ATT | -0.0568 | Average of post-treatment ATT(g,t) |
| TWFE (within-transform) | -0.0365 | Demeaned to avoid singularity |

**Verdict:** TWFE (-0.0365) differs from CS (-0.0568). TWFE biased toward zero due to bad comparisons (already-treated as controls).

---

### Module 04: Doubly Robust DiD (LaLonde)

| Method | Estimate | Notes |
|:-------|:---------|:------|
| Naive DiD | $299 | No covariate adjustment |
| IPW DiD | $1,246 | Inverse probability weighting |
| Outcome Regression | $1,692 | Predict counterfactual |
| Doubly Robust | $1,261 | Combines IPW + OR |

**Verdict:** Naive severely biased (selection on observables). DR preferred.

---

### Module 05: Heterogeneous Treatment Effects (mpdta)

| Method | Estimate | SE | Notes |
|:-------|:---------|:---|:------|
| Pooled DiD | -0.0385 | — | All counties |
| DiD (High Population) | -0.0465 | 0.189 | Above median lpop |
| DiD (Low Population) | -0.0449 | 0.193 | Below median lpop |

**Verdict:** No significant heterogeneity by county size (CIs overlap substantially).

---

### Module 06: Robust Triple Diff (Meyer et al. 1995)

| Method | Estimate | SE | Notes |
|:-------|:---------|:---|:------|
| Naive DDD (OLS) | +0.036 | 0.173 | 3-way FE with controls |
| Robust DDD (Target-Adj) | -0.118 | 0.205 | OR at target covariates |

**Verdict:** ~0.15 difference reflects covariate adjustment for age/marriage imbalance.

---

## References

- Baker, Callaway, Cunningham, Goodman-Bacon, Sant'Anna (2025). A Practitioner's Guide to Difference-in-Differences.
- Callaway & Sant'Anna (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.
- Card & Krueger (1994). Minimum wages and employment. *AER*.
- Arkhangelsky, Athey, Hirshberg, Imbens, Wager (2021). Synthetic Difference-in-Differences. *AER*.
- Ortiz-Villavicencio & Sant'Anna (2025). Doubly Robust DDD Estimators.

## License

MIT (c) 2025 Pranjal Rawat
