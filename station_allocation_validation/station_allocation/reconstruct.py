"""Station reconstruction from demand, order energy, and physical caps."""
from __future__ import annotations

import numpy as np
import pandas as pd

from station_allocation.config import StationConfiguration, get_station_configuration
from station_allocation.policies import StationPolicyParameters, policy_base_prediction
from station_allocation.resource import VariableResourceConfig, reconstruct_variable_resource

BOUNDARY = ["timestamp", "vehicle_id", "cabinet_id", "demand_kw", "base_prediction_kw", "energy_kwh", "gun_cap_kw", "cabinet_cap_kw"]


def reconstruct_station(boundary: pd.DataFrame, station_id: str | StationConfiguration, projection_iterations: int = 1200) -> tuple[pd.DataFrame, dict]:
    station = station_id if isinstance(station_id, StationConfiguration) else get_station_configuration(station_id)
    if station.policy in {"equal_share_by_request", "fcfs_minimum_one_module"}:
        parameters = StationPolicyParameters(
            station.policy, station.granularity_threshold_kw, station.delivered_power_factor,
            station.policy_blend, station.startup_factor, station.startup_minutes,
        )
        planned = policy_base_prediction(boundary, parameters)
        source = boundary.merge(planned, on=["timestamp", "vehicle_id"], validate="one_to_one")
        source["demand_kw"] = source["policy_upper_kw"]
        source["base_prediction_kw"] = source["policy_base_prediction_kw"]
        prediction, audit = reconstruct_variable_resource(
            source[BOUNDARY],
            VariableResourceConfig(capped_demand_blend=1.0, projection_iterations=projection_iterations),
        )
    else:
        source = boundary.copy()
        source["base_prediction_kw"] = np.minimum(source["demand_kw"], source["gun_cap_kw"])
        prediction, audit = reconstruct_variable_resource(
            source[BOUNDARY],
            VariableResourceConfig(
                station_trigger_kw=station.station_trigger_kw,
                station_cap_kw=station.station_cap_kw,
                capped_demand_blend=station.capped_demand_blend,
                projection_iterations=projection_iterations,
            ),
        )
    result = boundary.merge(prediction[["timestamp", "vehicle_id", "prediction_kw"]], on=["timestamp", "vehicle_id"], validate="one_to_one")
    result["allocated_modules_equivalent"] = result["prediction_kw"] / result["module_power_kw"]
    result["controller_power_kw"] = result.groupby(["timestamp", "cabinet_id"])["prediction_kw"].transform("sum")
    result["station_power_kw"] = result.groupby("timestamp")["prediction_kw"].transform("sum")
    result["station_id"] = station.station_id
    return result, {"station_id": station.station_id, "configuration": station.__dict__, "physical_audit": audit}
