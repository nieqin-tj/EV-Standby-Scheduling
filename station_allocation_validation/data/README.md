# Data for ten-station allocation and 20-min module demand

Scoring scripts in this project read only files under `data/`. Internal station layout source files are not distributed.

## In this GitHub tree

- `stations/`: minute panels used by `scripts/run_validation.py`.
- `module_20min/`: frozen 20-min windows, minute predictions, and session-boundary inputs for `scripts/score_20min.py`.

## On Figshare (not in GitHub)

- High-resolution charging-process curves, anonymized as S01–S10.
- Vehicle-series actual and demand curves with `mapping.csv`.

Reproduce the 0.62-module result without those files:

```text
python scripts/score_20min.py
```
