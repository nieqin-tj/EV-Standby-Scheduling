"""Deterministic hierarchical resource projection with row-specific limits."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = frozenset(
    {
        "timestamp",
        "vehicle_id",
        "cabinet_id",
        "demand_kw",
        "base_prediction_kw",
        "energy_kwh",
        "gun_cap_kw",
        "cabinet_cap_kw",
    }
)
FORBIDDEN_TOKENS = ("actual", "measured", "truth", "internal_soc", "peak_kw")


@dataclass(frozen=True)
class VariableResourceConfig:
    station_trigger_kw: float | None = None
    station_cap_kw: float | None = None
    capped_demand_blend: float = 0.75
    projection_iterations: int = 1200
    energy_tolerance_kw_min: float = 1e-3


def _allocate(upper: np.ndarray, weights: np.ndarray, total: float) -> np.ndarray:
    upper = np.maximum(np.asarray(upper, dtype=float), 0.0)
    total = min(max(float(total), 0.0), float(upper.sum()))
    if upper.sum() <= total + 1e-12:
        return upper.copy()
    weights = np.maximum(np.asarray(weights, dtype=float), 1e-12)
    low, high = 0.0, float(np.max(upper / weights))
    for _ in range(55):
        scale = (low + high) / 2.0
        if np.minimum(upper, weights * scale).sum() < total:
            low = scale
        else:
            high = scale
    return np.minimum(upper, weights * ((low + high) / 2.0))


def _group_sum(values: np.ndarray, codes: np.ndarray, groups: int) -> np.ndarray:
    return np.bincount(codes, weights=values, minlength=groups)


def _trapezoid_weights(frame: pd.DataFrame, session_code: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    positions = pd.Series(np.arange(len(frame))).groupby(session_code, sort=False)
    for _, group_positions in positions:
        index = group_positions.to_numpy(dtype=int)
        if len(index) < 2:
            continue
        minute = frame.iloc[index]["timestamp"].astype("int64").to_numpy() / 60e9
        delta = np.clip(np.diff(minute), 0.0, 1.0)
        weights[index[:-1]] += delta / 2.0
        weights[index[1:]] += delta / 2.0
    return weights


def _validate(frame: pd.DataFrame) -> None:
    unknown = sorted(set(frame.columns) - REQUIRED_COLUMNS)
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    forbidden = [name for name in frame.columns if any(token in name.lower() for token in FORBIDDEN_TOKENS)]
    if unknown or missing or forbidden:
        raise ValueError(f"invalid resource inputs: missing={missing}, unknown={unknown}, forbidden={forbidden}")


def reconstruct_variable_resource(
    boundary: pd.DataFrame,
    config: VariableResourceConfig,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    _validate(boundary)
    frame = boundary.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values(["timestamp", "cabinet_id", "vehicle_id"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("boundary input is empty")

    numeric_columns = [
        "demand_kw",
        "base_prediction_kw",
        "energy_kwh",
        "gun_cap_kw",
        "cabinet_cap_kw",
    ]
    if frame[numeric_columns].isna().any().any():
        raise ValueError("numeric boundary inputs must not contain missing values")
    if (frame[["energy_kwh", "gun_cap_kw", "cabinet_cap_kw"]] <= 0).any().any():
        raise ValueError("energy and physical caps must be positive")

    session_category = pd.Categorical(frame["vehicle_id"])
    time_category = pd.Categorical(frame["timestamp"])
    cabinet_time_key = frame["timestamp"].astype(str) + "|" + frame["cabinet_id"].astype(str)
    cabinet_time_category = pd.Categorical(cabinet_time_key)
    session_code = session_category.codes.astype(int)
    time_code = time_category.codes.astype(int)
    cabinet_time_code = cabinet_time_category.codes.astype(int)
    n_sessions = len(session_category.categories)
    n_times = len(time_category.categories)
    n_cabinet_times = len(cabinet_time_category.categories)

    demand = np.maximum(frame["demand_kw"].to_numpy(float), 0.0)
    base = np.maximum(frame["base_prediction_kw"].to_numpy(float), 0.0)
    gun_cap = frame["gun_cap_kw"].to_numpy(float)
    upper = np.minimum(demand, gun_cap)
    cabinet_cap_by_group = (
        frame.assign(_cabinet_time_code=cabinet_time_code)
        .groupby("_cabinet_time_code", sort=True)["cabinet_cap_kw"]
        .first()
        .to_numpy(float)
    )
    cabinet_cap_nunique = (
        frame.assign(_cabinet_time_code=cabinet_time_code)
        .groupby("_cabinet_time_code")["cabinet_cap_kw"]
        .nunique()
    )
    if (cabinet_cap_nunique > 1).any():
        raise ValueError("cabinet cap is inconsistent within a cabinet-minute")

    energy_weight = _trapezoid_weights(frame, session_code)
    target_kw_min = (
        frame.groupby("vehicle_id", sort=True)["energy_kwh"]
        .first()
        .reindex(session_category.categories)
        .to_numpy(float)
        * 60.0
    )
    individual_support = _group_sum(upper * energy_weight, session_code, n_sessions)
    individually_feasible = individual_support + config.energy_tolerance_kw_min >= target_kw_min

    blend = float(config.capped_demand_blend)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("capped_demand_blend must be within [0, 1]")
    allocation_weight = (1.0 - blend) * demand + blend * upper
    prediction = np.minimum(base, upper)
    station_reference = _group_sum(base, time_code, n_times)
    cabinet = frame["cabinet_id"].astype(str).to_numpy()

    if config.station_cap_kw is not None and config.station_trigger_kw is not None:
        for _, positions_value in frame.groupby("timestamp", sort=True).indices.items():
            positions = np.asarray(positions_value, dtype=int)
            time_index = int(time_code[positions[0]])
            if station_reference[time_index] < config.station_trigger_kw:
                continue
            local_cabinet = cabinet[positions]
            cabinet_ids = sorted(set(local_cabinet))
            local_positions = [np.flatnonzero(local_cabinet == cabinet_id) for cabinet_id in cabinet_ids]
            cabinet_upper = np.array(
                [
                    min(
                        float(frame.iloc[positions[local][0]]["cabinet_cap_kw"]),
                        float(upper[positions[local]].sum()),
                    )
                    for local in local_positions
                ]
            )
            cabinet_weight = np.array(
                [float(allocation_weight[positions[local]].sum()) for local in local_positions]
            )
            cabinet_budget = _allocate(cabinet_upper, cabinet_weight, float(config.station_cap_kw))
            for local, budget in zip(local_positions, cabinet_budget):
                row_index = positions[local]
                prediction[row_index] = _allocate(upper[row_index], allocation_weight[row_index], float(budget))

    prediction = np.maximum(np.minimum(prediction, upper), 1e-12)
    station_cap = np.full(n_times, np.inf, dtype=float)
    if config.station_cap_kw is not None and config.station_trigger_kw is not None:
        station_cap[station_reference >= config.station_trigger_kw] = float(config.station_cap_kw)

    max_energy_error = np.inf
    for iteration in range(config.projection_iterations):
        session_total = _group_sum(prediction * energy_weight, session_code, n_sessions)
        factor = np.divide(target_kw_min, session_total, out=np.ones_like(target_kw_min), where=session_total > 1e-12)
        factor[~individually_feasible] = np.maximum(factor[~individually_feasible], 1.0)
        prediction = np.minimum(prediction * factor[session_code], upper)

        cabinet_total = _group_sum(prediction, cabinet_time_code, n_cabinet_times)
        cabinet_factor = np.minimum(
            1.0,
            np.divide(
                cabinet_cap_by_group,
                cabinet_total,
                out=np.ones_like(cabinet_total),
                where=cabinet_total > 1e-12,
            ),
        )
        prediction *= cabinet_factor[cabinet_time_code]

        station_total = _group_sum(prediction, time_code, n_times)
        station_factor = np.minimum(
            1.0,
            np.divide(station_cap, station_total, out=np.ones_like(station_total), where=station_total > 1e-12),
        )
        prediction *= station_factor[time_code]

        if iteration % 20 == 19 or iteration == config.projection_iterations - 1:
            session_total = _group_sum(prediction * energy_weight, session_code, n_sessions)
            feasible_error = np.abs(session_total[individually_feasible] - target_kw_min[individually_feasible])
            max_energy_error = float(feasible_error.max()) if len(feasible_error) else 0.0
            if max_energy_error <= config.energy_tolerance_kw_min:
                break

    session_total = _group_sum(prediction * energy_weight, session_code, n_sessions)
    session_error = np.abs(session_total - target_kw_min)
    final_feasible = individually_feasible & (session_error <= max(config.energy_tolerance_kw_min, 0.06))
    cabinet_total = _group_sum(prediction, cabinet_time_code, n_cabinet_times)
    station_total = _group_sum(prediction, time_code, n_times)

    output = frame[["timestamp", "vehicle_id", "cabinet_id", "demand_kw", "energy_kwh"]].copy()
    output["prediction_kw"] = prediction
    output["energy_weight_min"] = energy_weight
    output["session_energy_feasible"] = final_feasible[session_code]
    output["session_energy_error_kwh"] = session_error[session_code] / 60.0
    audit: dict[str, float | int | bool] = {
        "n_points": len(frame),
        "n_sessions": n_sessions,
        "n_energy_infeasible_sessions": int((~final_feasible).sum()),
        "iterations": iteration + 1,
        "max_energy_error_kw_min": float(session_error.max()) if len(session_error) else 0.0,
        "max_gun_ratio": float(np.max(np.divide(prediction, gun_cap, out=np.zeros_like(prediction), where=gun_cap > 0))),
        "max_cabinet_ratio": float(np.max(np.divide(cabinet_total, cabinet_cap_by_group))),
        "max_station_power_kw": float(station_total.max()),
        "gun_violation_count": int((prediction > gun_cap + 1e-8).sum()),
        "cabinet_violation_count": int((cabinet_total > cabinet_cap_by_group + 1e-8).sum()),
        "station_violation_count": int((station_total > station_cap + 1e-8).sum()),
        "negative_power_count": int((prediction < -1e-12).sum()),
        "physical_pass": bool(
            (prediction >= -1e-12).all()
            and (prediction <= gun_cap + 1e-8).all()
            and (cabinet_total <= cabinet_cap_by_group + 1e-8).all()
            and (station_total <= station_cap + 1e-8).all()
        ),
    }
    return output, audit
