# Ten-station allocation validation

Numerical parameters were selected on the validation split only. Confirmation metrics below were computed once after that choice was frozen.
S01 and S02 use operator-declared discrete rules. S03–S10 use dynamic demand allocation with energy projection.
The primary power metric down-weights the first three minutes of each session (weight 0.25).

| Station | Family | Sessions | Power MAE (kW) | Module MAE | Relative module MAE | Bias (kW) |
|---|---|---:|---:|---:|---:|---:|
| S01 | declared_rule | 102 | 1.459 | 0.126 | 4.34% | +0.113 |
| S02 | declared_rule | 81 | 1.944 | 0.165 | 6.08% | -0.099 |
| S03 | dynamic_demand | 123 | 1.784 | 0.211 | 3.61% | -0.197 |
| S04 | dynamic_demand | 195 | 2.179 | 0.169 | 4.56% | -0.142 |
| S05 | dynamic_demand | 178 | 1.493 | 0.178 | 3.85% | -0.154 |
| S06 | dynamic_demand | 145 | 2.633 | 0.202 | 7.12% | -1.176 |
| S07 | dynamic_demand | 59 | 1.592 | 0.135 | 4.99% | -0.344 |
| S08 | dynamic_demand | 453 | 1.944 | 0.174 | 4.63% | -0.075 |
| S09 | dynamic_demand | 509 | 1.337 | 0.099 | 3.52% | -0.084 |
| S10 | dynamic_demand | 329 | 2.756 | 0.206 | 6.74% | -0.105 |

Quality-consistent sessions have at least five points, do not cross a split boundary, and keep order energy within 1 kWh of the measured-power integral. That filter affects scoring membership only.
