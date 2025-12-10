# Bug Report: TWFE Numerical Instability in Module 03

**Date:** December 2025
**Status:** RESOLVED
**Severity:** Critical (produced nonsensical output)

---

## Executive Summary

The `run_twfe()` function in Module 03 produced a coefficient of **-25 billion** instead of the expected ~-0.03. The root cause was a data issue: the `treat` variable in the mpdta dataset is constant within each county, creating perfect multicollinearity with unit fixed effects.

| Metric | Before Fix | After Fix |
|--------|------------|-----------|
| TWFE coefficient | -25,287,832,262.91 | **-0.0365** |
| TWFE standard error | 49,746,281,220.43 | **0.0113** |
| CS estimate (reference) | -0.0234 | -0.0234 |

---

## 1. Problem Description

### Observed Behavior

```
TWFE coefficient: -25287832262.9097 (SE: 49746281220.4312)
Callaway-Sant'Anna (simple): -0.0234
```

The TWFE estimate was 10 orders of magnitude larger than the CS estimate and had the wrong sign relative to its standard error.

### Expected Behavior

TWFE and CS estimates should be in the same ballpark (within ~0.01-0.02 of each other), with TWFE potentially showing some bias due to heterogeneous treatment effects.

---

## 2. Root Cause Analysis

### The Data Structure

The mpdta dataset contains:
- **2,500 observations** (500 counties × 5 years)
- **4 treatment cohorts**: Never (309), 2004 (20), 2006 (40), 2007 (131)

### The Problem: `treat` is Time-Invariant

The `treat` column in mpdta indicates **"ever treated"** rather than **"currently treated"**:

```
County 8001: treat = [1, 1, 1, 1, 1]  ← Always 1 (ever treated)
County 8019: treat = [1, 1, 1, 1, 1]  ← Always 1 (ever treated)
County 1001: treat = [0, 0, 0, 0, 0]  ← Always 0 (never treated)
```

**Key diagnostic:**
```python
within_county_std = df.groupby('countyreal')['treat'].std()
print(within_county_std.mean())  # Output: 0.000000
```

### Why This Breaks TWFE

The Two-Way Fixed Effects model is:

$$Y_{it} = \alpha_i + \gamma_t + \delta D_{it} + \varepsilon_{it}$$

For $\delta$ to be identified, $D_{it}$ must vary **within units over time**. When `treat` is constant within each county:

1. `treat` is perfectly collinear with the county fixed effects $\alpha_i$
2. The design matrix becomes singular (or near-singular)
3. OLS produces garbage coefficients

---

## 3. The Solution

### Fix 1: Create Correct Treatment Indicator

The treatment indicator should be:

$$D_{it} = \mathbf{1}\{t \geq g_i\} \cdot \mathbf{1}\{g_i > 0\}$$

where $g_i$ is the first treatment year for unit $i$.

```python
df['D'] = ((df['year'] >= df['first_treat']) & (df['first_treat'] > 0)).astype(int)
```

**Result:**
- Original `treat`: 955 treated obs (constant within units)
- Corrected `D`: 291 treated obs (varies within 191 counties)

### Fix 2: Use Within-Transformation

With 500 county fixed effects, the standard OLS approach can have numerical instability. The within-transformation (demeaning) is mathematically equivalent but numerically stable:

```python
# Demean by county to absorb unit fixed effects
df['Y_dm'] = df.groupby('countyreal')['lemp'].transform(lambda x: x - x.mean())
df['D_dm'] = df.groupby('countyreal')['D'].transform(lambda x: x - x.mean())

# Regression on demeaned data
model = smf.ols('Y_dm ~ D_dm + C(year) - 1', data=df).fit()
```

---

## 4. Verification

After the fix:

```
TWFE coefficient: -0.0365 (SE: 0.0113)
Callaway-Sant'Anna (simple): -0.0234
Difference: -0.0131
```

The TWFE estimate is now:
- **Reasonable magnitude** (~0.04 vs ~0.02)
- **Same direction** (negative effect on employment)
- **Statistically significant** (t-stat ≈ 3.2)

The remaining difference (-0.0131) is the **expected TWFE bias** from:
1. Using already-treated units as controls
2. Implicit weighting by cohort size

This is exactly the pedagogical point Module 03 aims to illustrate.

---

## 5. Lessons Learned

### For Practitioners

1. **Always check within-unit variation** before running TWFE:
   ```python
   df.groupby('unit')['treatment'].std().mean()  # Should be > 0
   ```

2. **Distinguish "ever treated" from "currently treated"**:
   - `ever_treated = 1` if unit is ever treated (time-invariant)
   - `D_it = 1` if unit is treated at time t (time-varying)

3. **Use demeaning for large panels** to avoid numerical instability with many fixed effects.

### For This Codebase

The `treat` column in mpdta appears to be an "ever treated" indicator from the original R package. Future modules should:
- Create the time-varying indicator explicitly
- Document the treatment variable definition clearly

---

## 6. Files Modified

| File | Change |
|------|--------|
| `03_staggered_GxT/main.py` | Fixed `run_twfe()` to create correct treatment indicator and use within-transformation |

---

## 7. References

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data*, Ch. 10.
- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*.
- Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics*.
