Scoring files for the ten-station 20-min module-demand result.

- `windows/`: station 20-min predicted and measured integer module demand (primary scoring input).
- `minute/`: vehicle-minute reconstructed power after allocation (`prediction_kw`).
- `orders/`: session-boundary inputs used to generate the request trajectories.
Measured curve files are not included. Run `python scripts/score_20min.py` to recompute the station-equal MAE.
