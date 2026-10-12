# Station allocation validation

This repository contains the ten-station files needed to reproduce two related scores.

1. **20-min station-level module demand** (main-text number): MAE 0.62 modules (95% CI 0.44–0.80); relative error 5.44%; normalized bias −2.74%.
2. **Cabinet-minute equivalent occupancy** (allocation confirmation): MAE 0.17 modules (95% CI 0.14–0.19); relative error 4.94%; normalized bias −0.57%.

Measured delivered power is not used to form a prediction. It is used only after allocation is frozen.

S01 and S02 follow operator-declared rules. S01 shares cabinet power equally by request. S02 reserves one module for every active vehicle and then fills residual capacity in arrival order. S03–S10 use dynamic-demand allocation: clip the request at the gun rating, share the cabinet pool, and project the trajectory onto billed energy.

## Reproduce the 0.62 module result

Scoring inputs are the frozen 20-min windows in `data/module_20min/windows/` (`predicted_modules`, `actual_modules`). Minute-level reconstructed power is in `data/module_20min/minute/`. Session-boundary inputs used to generate the request trajectories are in `data/module_20min/orders/`.

```text
python scripts/score_20min.py
```

This writes [`reports/module_20min.md`](reports/module_20min.md). The test `tests/test_20min_score.py` checks that the station-equal summary rounds to 0.62, 5.44%, and −2.74%.

## Allocation confirmation

Numerical parameters are selected on the validation split only. Confirmation is scored once. The power MAE down-weights the first three minutes of each session (weight 0.25). Occupancy is `sum(power) / module rating` at cabinet-minute resolution.

| Station | Rule | Sessions | Power MAE (kW) | Occupancy MAE | Relative occupancy MAE |
|---|---|---:|---:|---:|---:|
| S01 | equal share | 102 | 1.459 | 0.126 | 4.34% |
| S02 | one-module floor, then FCFS | 81 | 1.944 | 0.165 | 6.08% |
| S03 | dynamic demand | 123 | 1.784 | 0.211 | 3.61% |
| S04 | dynamic demand | 195 | 2.179 | 0.169 | 4.56% |
| S05 | dynamic demand | 178 | 1.493 | 0.178 | 3.85% |
| S06 | dynamic demand | 145 | 2.633 | 0.202 | 7.12% |
| S07 | dynamic demand | 59 | 1.592 | 0.135 | 4.99% |
| S08 | dynamic demand | 453 | 1.944 | 0.174 | 4.63% |
| S09 | dynamic demand | 509 | 1.337 | 0.099 | 3.52% |
| S10 | dynamic demand | 329 | 2.756 | 0.206 | 6.74% |

Full tables: [`reports/report.md`](reports/report.md), [`reports/metrics.csv`](reports/metrics.csv).

```text
python -m pytest tests -q
python scripts/run_validation.py
```

## Layout

```text
station_allocation/     allocation, energy projection, 20-min scoring
scripts/                prepare panels, run allocation, score 20-min
data/stations/          minute panels for allocation confirmation
data/module_20min/      20-min windows, minute predictions, order inputs
data/station_01_02_rule_calibration/
data/hierarchical_resource_stations_02_08/
tests/
reports/
```

Raw operator curve files are not in this repository. The 20-min windows already contain the predicted and measured integer module demand used for the 0.62 result.
