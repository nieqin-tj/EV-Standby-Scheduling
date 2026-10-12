"""Station rule families used in this validation.

S01 and S02 follow operator-declared discrete rules. S03–S10 share dynamic
demand allocation with cabinet energy projection. Numerical values listed here
are search defaults; run_validation.py reselects them on the validation split.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationConfiguration:
    station_id: str
    policy: str
    family: str
    granularity_threshold_kw: float = 1.0
    delivered_power_factor: float = 1.0
    policy_blend: float = 1.0
    startup_factor: float = 1.0
    startup_minutes: int = 0
    capped_demand_blend: float = 0.75
    station_trigger_kw: float | None = None
    station_cap_kw: float | None = None


STATION_CONFIGURATIONS = {
    "S01": StationConfiguration("S01", "equal_share_by_request", "declared_rule", policy_blend=0.50),
    "S02": StationConfiguration(
        "S02", "fcfs_minimum_one_module", "declared_rule",
        delivered_power_factor=0.92, policy_blend=0.25, startup_factor=0.80, startup_minutes=2,
    ),
    "S03": StationConfiguration("S03", "dynamic_demand", "dynamic_demand", capped_demand_blend=0.0, station_trigger_kw=580.0, station_cap_kw=680.0),
    "S04": StationConfiguration("S04", "dynamic_demand", "dynamic_demand", capped_demand_blend=0.75),
    "S05": StationConfiguration("S05", "dynamic_demand", "dynamic_demand", capped_demand_blend=0.0, station_trigger_kw=900.0, station_cap_kw=900.0),
    "S06": StationConfiguration("S06", "dynamic_demand", "dynamic_demand", capped_demand_blend=0.75),
    "S07": StationConfiguration("S07", "dynamic_demand", "dynamic_demand", capped_demand_blend=0.75),
    "S08": StationConfiguration("S08", "dynamic_demand", "dynamic_demand", capped_demand_blend=0.75),
    "S09": StationConfiguration("S09", "dynamic_demand", "dynamic_demand", capped_demand_blend=1.0),
    "S10": StationConfiguration("S10", "dynamic_demand", "dynamic_demand", capped_demand_blend=0.75, station_trigger_kw=1300.0, station_cap_kw=1340.0),
}


def get_station_configuration(station_id: str) -> StationConfiguration:
    try:
        return STATION_CONFIGURATIONS[station_id.upper()]
    except KeyError as error:
        allowed = ", ".join(STATION_CONFIGURATIONS)
        raise ValueError(f"unknown station_id={station_id!r}; allowed: {allowed}") from error
