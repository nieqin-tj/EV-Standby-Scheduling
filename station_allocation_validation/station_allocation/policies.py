"""Station-specific deterministic module plans used before energy projection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationPolicyParameters:
    policy: str
    granularity_threshold_kw: float
    delivered_power_factor: float
    policy_blend: float
    startup_factor: float = 1.0
    startup_minutes: int = 0


def requested_modules(demand_kw: float, module_kw: float, maximum: int, threshold_kw: float) -> int:
    if demand_kw <= 0.4:
        return 0
    full = min(int((demand_kw - 0.4) // module_kw), maximum)
    if demand_kw - full * module_kw >= threshold_kw and full < maximum:
        full += 1
    return max(full, 1)


def _equal_integer(requests: np.ndarray, pool: int, order: np.ndarray) -> np.ndarray:
    result = np.zeros_like(requests)
    while pool > 0:
        eligible = [index for index in order if result[index] < requests[index]]
        if not eligible:
            break
        for index in eligible:
            if pool <= 0:
                break
            result[index] += 1
            pool -= 1
    return result


def _minimum_one_fcfs(requests: np.ndarray, pool: int, order: np.ndarray) -> np.ndarray:
    result = np.zeros_like(requests)
    for index in order:
        if pool <= 0:
            break
        if requests[index] > 0:
            result[index] = 1
            pool -= 1
    for index in order:
        addition = min(int(requests[index] - result[index]), pool)
        result[index] += addition
        pool -= addition
        if pool <= 0:
            break
    return result


def policy_base_prediction(frame: pd.DataFrame, parameters: StationPolicyParameters) -> pd.DataFrame:
    required = {"timestamp", "vehicle_id", "cabinet_id", "demand_kw", "gun_cap_kw", "module_count", "module_power_kw"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing policy inputs: {missing}")
    data = frame.sort_values(["timestamp", "cabinet_id", "vehicle_id"]).copy()
    starts = data.groupby("vehicle_id")["timestamp"].min()
    session_minute = data.groupby("vehicle_id").cumcount().to_numpy()
    base = np.zeros(len(data), dtype=float)
    for _, positions_value in data.groupby(["cabinet_id", "timestamp"], sort=True).indices.items():
        positions = np.asarray(positions_value, dtype=int)
        group = data.iloc[positions]
        module_kw = float(group["module_power_kw"].iloc[0])
        pool = int(group["module_count"].iloc[0])
        caps = np.minimum(group["gun_cap_kw"].to_numpy(float), group["demand_kw"].to_numpy(float))
        max_modules = np.maximum(np.floor(group["gun_cap_kw"].to_numpy(float) / module_kw + 1e-9).astype(int), 1)
        requests = np.array([
            requested_modules(value, module_kw, int(maximum), parameters.granularity_threshold_kw)
            for value, maximum in zip(caps, max_modules)
        ])
        arrival = starts.reindex(group["vehicle_id"]).astype("int64").to_numpy()
        order = np.lexsort((group["vehicle_id"].astype(str).to_numpy(), arrival))
        if parameters.policy == "equal_share_by_request":
            modules = _equal_integer(requests, pool, order)
        elif parameters.policy == "fcfs_minimum_one_module":
            modules = _minimum_one_fcfs(requests, pool, order)
        else:
            raise ValueError(f"unsupported operational policy: {parameters.policy}")
        policy_power = np.minimum(caps, modules * module_kw * parameters.delivered_power_factor)
        unconstrained = caps
        mixed = parameters.policy_blend * policy_power + (1.0 - parameters.policy_blend) * unconstrained
        if parameters.startup_minutes > 0:
            startup = session_minute[positions] < parameters.startup_minutes
            mixed = np.where(startup, mixed * parameters.startup_factor, mixed)
        base[positions] = mixed
    output = data[["timestamp", "vehicle_id"]].copy()
    output["policy_base_prediction_kw"] = base
    output["policy_upper_kw"] = base
    return output
