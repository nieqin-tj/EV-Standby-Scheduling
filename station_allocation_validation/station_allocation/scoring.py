"""Validation and confirmation scoring. Actual power is used only here."""
from __future__ import annotations

import numpy as np
import pandas as pd


def session_quality(frame: pd.DataFrame) -> pd.Series:
    quality = {}
    for vid, group in frame.groupby("vehicle_id"):
        group = group.sort_values("timestamp")
        dt = np.clip(np.diff(group.timestamp.astype("int64").to_numpy()) / 60e9, 0, 1)
        actual = group.actual_kw.to_numpy()
        energy = float(np.sum((actual[:-1] + actual[1:]) / 2 * dt) / 60) if len(group) > 1 else 0.0
        quality[vid] = bool(
            len(group) >= 5
            and group.split.nunique() == 1
            and abs(energy - float(group.energy_kwh.iloc[0])) <= 1
        )
    return frame.vehicle_id.map(quality)


def startup_weights(frame: pd.DataFrame) -> pd.Series:
    weights = pd.Series(1.0, index=frame.index)
    for _, group in frame.groupby("vehicle_id", sort=False):
        group = group.sort_values("timestamp")
        minute = (group.timestamp - group.timestamp.iloc[0]).dt.total_seconds().to_numpy() / 60.0
        weights.loc[group.index] = np.where(minute < 3, 0.25, 1.0)
    return weights


def score_power(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "n_points": 0, "n_sessions": 0, "point_mae_kw": None,
            "startup_weighted_point_mae_kw": None, "session_equal_mae_kw": None,
            "rmse_kw": None, "bias_kw": None, "p90_absolute_error_kw": None,
        }
    err = (frame.prediction_kw - frame.actual_kw).abs()
    weights = startup_weights(frame)
    session = frame.assign(e=err).groupby("vehicle_id").e.mean()
    return {
        "n_points": int(len(frame)),
        "n_sessions": int(frame.vehicle_id.nunique()),
        "point_mae_kw": float(err.mean()),
        "startup_weighted_point_mae_kw": float((err * weights).sum() / weights.sum()),
        "session_equal_mae_kw": float(session.mean()),
        "rmse_kw": float(np.sqrt((err ** 2).mean())),
        "bias_kw": float((frame.prediction_kw - frame.actual_kw).mean()),
        "p90_absolute_error_kw": float(np.quantile(err, 0.9)),
    }


def cabinet_minute_occupancy(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["cabinet_id", "timestamp"]
    grouped = frame.groupby(keys, sort=False)
    cabinet = grouped.agg(
        predicted_kw=("prediction_kw", "sum"),
        actual_kw=("actual_kw", "sum"),
        module_power_kw=("module_power_kw", "first"),
        module_count=("module_count", "first"),
        cabinet_cap_kw=("cabinet_cap_kw", "first"),
        all_quality_pass=("quality_pass", "all"),
    ).reset_index()
    cabinet["predicted_modules"] = cabinet.predicted_kw / cabinet.module_power_kw
    cabinet["actual_modules"] = cabinet.actual_kw / cabinet.module_power_kw
    cabinet["module_error"] = cabinet.predicted_modules - cabinet.actual_modules
    return cabinet


def score_modules(cabinet: pd.DataFrame) -> dict:
    if cabinet.empty:
        return {"cabinet_minutes": 0, "mae_modules": None, "relative_mae_pct": None, "bias_modules": None}
    err = cabinet.module_error.abs()
    mean_actual = float(cabinet.actual_modules.mean())
    return {
        "cabinet_minutes": int(len(cabinet)),
        "mae_modules": float(err.mean()),
        "rmse_modules": float(np.sqrt((cabinet.module_error ** 2).mean())),
        "bias_modules": float(cabinet.module_error.mean()),
        "actual_mean_modules": mean_actual,
        "relative_mae_pct": float(err.mean() / mean_actual * 100) if mean_actual > 0 else None,
    }
