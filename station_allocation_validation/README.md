# Station allocation and 20-min module-demand validation

This companion code reconstructs vehicle-level delivered power at ten charging stations from session boundaries, request trajectories, and published terminal/cabinet ratings, and then scores station-level module demand at a 20-min decision interval. I do not read measured delivered power, internal SOC, or module occupancy when forming a prediction. Those series are used only after the allocation rule is frozen.

All ten stations follow operator-published rules. S01 shares cabinet power equally by request. S02 reserves one module for every active vehicle and then fills residual capacity in arrival order. S03–S10 use dynamic demand: the request is first limited by the terminal rating, then shared inside the cabinet, and finally projected onto billed energy.

## Main results

**20-min station-level module demand** (main-text metric)  
The module MAE is 0.62 (95% CI 0.44–0.80). Relative to actual operating module demand the error is 5.44%, with a normalized bias of −2.74%.

```text
python scripts/score_20min.py
```

Scoring inputs are the frozen windows in `data/module_20min/windows/` (`predicted_modules`, `actual_modules`). Vehicle-minute reconstructed power is in `data/module_20min/minute/`. Session boundaries used to generate requests are in `data/module_20min/orders/`. `tests/test_20min_score.py` checks the three summary figures above.

**Cabinet-minute equivalent occupancy** (allocation confirmation)  
Numerical parameters are selected on the validation split only; confirmation is scored once. The power MAE down-weights the first three minutes of each session (weight 0.25). Occupancy is cabinet-minute `sum(power) / module rating`. The station-equal occupancy MAE is 0.17 modules (95% CI 0.14–0.19), or 4.94% of observed occupancy, with a normalized bias of −0.57%. Per-station confirmation tables are in [`reports/report.md`](reports/report.md).

```text
python -m pytest tests -q
python scripts/run_validation.py
```

## Data

GitHub provides only the processed files needed to reproduce the scores above: `data/stations/` and `data/module_20min/`. Those files already contain the terminal and cabinet ratings required for scoring. **Internal station layout source files are not included.**

Anonymized high-resolution charging-process curves and the vehicle-series actual/demand library will be released on Figshare (DOI to be added). If those files are present locally as described in `data/README.md`, run `python scripts/verify_data.py`. That check reads only `data/` inside this project.

## Layout

```text
station_allocation/     published-rule allocation, energy projection, 20-min scoring
scripts/                data checks, allocation validation, 20-min scoring
data/stations/          minute panels for allocation confirmation
data/module_20min/      20-min windows, minute predictions, order boundaries
tests/
reports/
```
