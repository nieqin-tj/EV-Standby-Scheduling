# Ten-station 20-min module demand

Primary aggregation is mean over each 20-min cabinet window, then ceiling to integer modules and summing to the station. Scores are station-equal.

Station-level module-demand MAE = **0.62** modules (95% CI 0.44–0.80). Relative MAE = 5.44%. Normalized bias = -2.74%.

| Station | Windows | MAE | RMSE | Bias | Actual mean | Relative MAE | Normalized bias |
|---|---:|---:|---:|---:|---:|---:|---:|
| S01 | 114 | 0.263 | 0.577 | -0.053 | 5.482 | 4.80% | -0.96% |
| S02 | 128 | 0.344 | 0.685 | -0.141 | 6.227 | 5.52% | -2.26% |
| S03 | 135 | 0.393 | 0.765 | -0.289 | 6.830 | 5.75% | -4.23% |
| S04 | 124 | 0.621 | 1.059 | -0.395 | 13.903 | 4.47% | -2.84% |
| S05 | 132 | 0.674 | 1.026 | -0.250 | 13.068 | 5.16% | -1.91% |
| S06 | 127 | 0.559 | 1.046 | -0.307 | 9.465 | 5.91% | -3.24% |
| S07 | 125 | 0.680 | 1.203 | -0.456 | 7.152 | 9.51% | -6.38% |
| S08 | 109 | 1.110 | 1.505 | -0.578 | 32.073 | 3.46% | -1.80% |
| S09 | 87 | 0.759 | 1.174 | -0.138 | 14.391 | 5.27% | -0.96% |
| S10 | 116 | 0.793 | 1.426 | -0.483 | 17.267 | 4.59% | -2.80% |
