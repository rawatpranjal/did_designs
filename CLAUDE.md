# CLAUDE.md

Instructions for Claude Code when working in this repository.

## Repository Overview

Modern Difference-in-Differences implementations following Baker, Callaway, Cunningham, Goodman-Bacon, and Sant'Anna (2025).

## Commands

```bash
pip install -r requirements.txt
python 01_canonical_2x2/main.py
```

## Structure

```
did_designs/
├── 01_canonical_2x2/     # Static 2x2 DiD
├── 02_event_study_2xT/   # Event study with pre-trends
├── 03_staggered_GxT/     # Callaway-Sant'Anna
├── 04_covariates_dr/     # Doubly robust DiD
├── 05_hte/               # Heterogeneous effects, DDD
├── 06_triple_diff_dr/    # Robust DDD with covariates
├── 07_synthetic_did/     # SDID
├── data/                 # CSV datasets
├── papers/               # Reference PDFs
├── utils.py              # Shared loaders and functions
└── archive/              # Old exploratory code
```

**Module structure:**
```
XX_module_name/
├── README.md    # Theory, assumptions, math
├── main.py      # Implementation
└── figs/        # Generated figures
```

## Dataset Registry

| Module | Design | Dataset | Loader |
|--------|--------|---------|--------|
| 01 | 2x2 | Card & Krueger | `load_card_krueger()` |
| 02 | Event Study | California Prop 99 | `load_smoking()` |
| 03 | Staggered | mpdta | `load_mpdta()` |
| 04 | Covariates/DR | LaLonde | `load_lalonde()` |
| 05 | HTE/DDD | Medicaid | `load_medicaid()` |
| 06 | Robust DDD | Simulated | Generated in module |
| 07 | SDID | California Prop 99 | `load_smoking()` |

Additional datasets: `load_castle()`, plus raw CSVs in `data/`.

## Coding Standards

- Use `pandas`, `numpy`, `statsmodels`, `matplotlib`, `seaborn`
- Robust standard errors: `cov_type='HC1'` or `'HC3'`
- Handle NaN: `df.dropna(subset=[...])`
- Seed simulations: `np.random.seed(42)`
- Save figures: `fig.savefig(FIGS_DIR / 'name.png', dpi=150, bbox_inches='tight')`

**Import pattern:**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_card_krueger
```

## Module Checklist

- [ ] Manual calculation matches package result
- [ ] Assumptions listed in README
- [ ] Robust standard errors used
- [ ] Figures saved to `figs/`
