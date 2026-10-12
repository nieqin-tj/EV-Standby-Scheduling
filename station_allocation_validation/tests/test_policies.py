import numpy as np
import pandas as pd

from station_allocation.policies import StationPolicyParameters, policy_base_prediction


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2025-01-01 00:01"] * 3), "vehicle_id": ["a", "b", "c"],
        "cabinet_id": ["x"] * 3, "demand_kw": [80.0] * 3, "gun_cap_kw": [120.0] * 3,
        "module_count": [5] * 3, "module_power_kw": [20.0] * 3,
    })


def test_station2_guarantees_one_then_prioritizes_earliest() -> None:
    out = policy_base_prediction(_frame(), StationPolicyParameters("fcfs_minimum_one_module", 3.0, 1.0, 1.0))
    np.testing.assert_allclose(out["policy_base_prediction_kw"], [60.0, 20.0, 20.0])


def test_station1_equal_integer_distribution() -> None:
    out = policy_base_prediction(_frame(), StationPolicyParameters("equal_share_by_request", 3.0, 1.0, 1.0))
    np.testing.assert_allclose(out["policy_base_prediction_kw"], [40.0, 40.0, 20.0])


def test_startup_factor_only_limits_early_points() -> None:
    frame = pd.concat([_frame().iloc[[0]], _frame().iloc[[0]].assign(timestamp=pd.Timestamp("2025-01-01 00:02"))])
    out = policy_base_prediction(
        frame, StationPolicyParameters("equal_share_by_request", 3.0, 1.0, 1.0, 0.5, 1)
    )
    np.testing.assert_allclose(out["policy_base_prediction_kw"], [40.0, 80.0])
