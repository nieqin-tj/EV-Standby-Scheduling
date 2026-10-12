"""Pure scoring for the paper's cabinet-first integer 20-minute demand.

Inputs must already be a coverage-audited complete cabinet-minute panel.
Missing rows are errors, not implicit zero demand. Explicit idle rows are allowed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def module_demand_20min(panel: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    if aggregation not in {"mean", "max"}:
        raise ValueError("aggregation must be mean or max")
    required = {"station_id", "cabinet_id", "timestamp", "predicted_kw", "actual_kw", "module_power_kw"}
    if not required.issubset(panel):
        raise ValueError(f"Missing columns: {sorted(required-set(panel))}")
    frame = panel.copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    if (frame.timestamp != frame.timestamp.dt.floor("min")).any():
        raise ValueError("timestamps must be exact minute boundaries")
    if frame.duplicated(["station_id", "cabinet_id", "timestamp"]).any():
        raise ValueError("Duplicate cabinet-minute")
    numeric = ["predicted_kw", "actual_kw", "module_power_kw"]
    if not np.isfinite(frame[numeric].to_numpy(float)).all() or (frame[numeric[:2]] < 0).any().any() or (frame.module_power_kw <= 0).any():
        raise ValueError("Nonfinite/negative power or nonpositive module rating")
    frame["window"] = frame.timestamp.dt.floor("20min")
    grouped = frame.groupby(["station_id", "cabinet_id", "window"])
    if (grouped.size() != 20).any():
        raise ValueError("Incomplete 20-minute cabinet window; audit coverage instead of filling zeros")
    if (grouped.module_power_kw.nunique() != 1).any():
        raise ValueError("Changing module rating within a window")
    cabinet = grouped.agg(predicted_kw=("predicted_kw", aggregation), actual_kw=("actual_kw", aggregation), module_power_kw=("module_power_kw", "first")).reset_index()
    # Snap only floating-point noise at an integer boundary, not genuine excess power.
    for name in ("predicted", "actual"):
        ratio = cabinet[f"{name}_kw"] / cabinet.module_power_kw
        nearest = np.rint(ratio)
        ratio = np.where(np.abs(ratio-nearest) <= 1e-10, nearest, ratio)
        cabinet[f"{name}_modules"] = np.ceil(ratio).astype(int)
    # All declared cabinets must have the same set of scored windows.
    for _, station in cabinet.groupby("station_id"):
        counts = station.groupby("window").cabinet_id.nunique()
        if (counts != station.cabinet_id.nunique()).any():
            raise ValueError("Incomplete cabinet coverage across station windows")
    station = cabinet.groupby(["station_id", "window"], as_index=False)[["predicted_modules", "actual_modules"]].sum()
    station["aggregation"] = aggregation
    return station


def score_module_demand(windows: pd.DataFrame) -> dict:
    if windows.empty:
        raise ValueError("Empty scoring cohort")
    error = windows.predicted_modules - windows.actual_modules
    mean_actual = float(windows.actual_modules.mean())
    return {"windows": len(windows), "mae_modules": float(error.abs().mean()),
            "rmse_modules": float(np.sqrt((error**2).mean())), "bias_modules": float(error.mean()),
            "actual_mean_modules": mean_actual,
            "relative_mae_pct": float(error.abs().mean()/mean_actual*100) if mean_actual > 0 else None}
