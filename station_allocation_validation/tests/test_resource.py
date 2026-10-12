from __future__ import annotations

import pandas as pd
import pytest

from station_allocation.resource import (
    VariableResourceConfig,
    reconstruct_variable_resource,
)


def _frame() -> pd.DataFrame:
    rows = []
    for minute in range(6):
        for vehicle, cabinet, demand, energy, gun, cabinet_cap in (
            ("A", "C1", 100.0, 8.0, 120.0, 150.0),
            ("B", "C1", 100.0, 8.0, 120.0, 150.0),
            ("C", "C2", 80.0, 6.0, 80.0, 120.0),
        ):
            rows.append(
                {
                    "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=minute),
                    "vehicle_id": vehicle,
                    "cabinet_id": cabinet,
                    "demand_kw": demand,
                    "base_prediction_kw": demand,
                    "energy_kwh": energy,
                    "gun_cap_kw": gun,
                    "cabinet_cap_kw": cabinet_cap,
                }
            )
    return pd.DataFrame(rows)


def test_variable_caps_and_determinism() -> None:
    frame = _frame()
    config = VariableResourceConfig(station_trigger_kw=200.0, station_cap_kw=220.0)
    first, audit = reconstruct_variable_resource(frame, config)
    second, _ = reconstruct_variable_resource(frame.sample(frac=1.0, random_state=3), config)
    first = first.sort_values(["timestamp", "vehicle_id"]).reset_index(drop=True)
    second = second.sort_values(["timestamp", "vehicle_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(first, second)
    assert audit["physical_pass"]
    assert audit["gun_violation_count"] == 0
    assert audit["cabinet_violation_count"] == 0
    assert audit["station_violation_count"] == 0


def test_actual_power_is_rejected() -> None:
    frame = _frame()
    frame["actual_kw"] = 1.0
    with pytest.raises(ValueError, match="invalid resource inputs"):
        reconstruct_variable_resource(frame, VariableResourceConfig())
