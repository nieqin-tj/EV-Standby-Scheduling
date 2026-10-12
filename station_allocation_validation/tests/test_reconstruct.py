import pandas as pd

from station_allocation.config import get_station_configuration
from station_allocation.reconstruct import reconstruct_station


def test_unknown_station_is_rejected():
    try:
        get_station_configuration("S99")
    except ValueError as error:
        assert "S99" in str(error)
    else:
        raise AssertionError("expected unknown station to fail")


def test_s01_and_s04_families_run_on_a_tiny_panel():
    rows = []
    for minute in range(5):
        rows.append({
            "timestamp": pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=minute),
            "vehicle_id": "A", "cabinet_id": "C", "terminal_name": "T",
            "demand_kw": 80.0, "energy_kwh": 5.0, "gun_cap_kw": 120.0,
            "cabinet_cap_kw": 120.0, "module_count": 6, "module_power_kw": 20.0,
        })
    frame = pd.DataFrame(rows)
    for sid in ("S01", "S04"):
        pred, audit = reconstruct_station(frame, sid, projection_iterations=40)
        assert audit["physical_audit"]["physical_pass"]
        assert pred.prediction_kw.between(0, 120).all()
