"""Load minute panels. Actual power is retained for scoring only."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "stations"

PANEL_COLUMNS = [
    "station_id", "timestamp", "vehicle_id", "cabinet_id", "terminal_name",
    "demand_kw", "actual_kw", "energy_kwh", "gun_cap_kw", "cabinet_cap_kw",
    "module_count", "module_power_kw", "split",
]


def panel_path(station_id: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DATA) / f"{station_id.lower()}_panel.parquet"


def load_panel(station_id: str, data_dir: Path | None = None) -> pd.DataFrame:
    path = panel_path(station_id, data_dir)
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    if "energy_kwh" not in frame.columns and "order_energy_kwh" in frame.columns:
        frame["energy_kwh"] = frame["order_energy_kwh"]
    missing = [name for name in PANEL_COLUMNS if name not in frame.columns]
    if "terminal_name" in missing:
        frame["terminal_name"] = ""
        missing.remove("terminal_name")
    if missing:
        raise ValueError(f"{station_id} panel missing columns: {missing}")
    if frame.duplicated(["vehicle_id", "timestamp"]).any():
        raise ValueError(f"{station_id}: duplicate vehicle-minute rows")
    return frame.sort_values(["timestamp", "cabinet_id", "vehicle_id"]).reset_index(drop=True)


def list_stations(data_dir: Path | None = None) -> list[str]:
    folder = data_dir or DATA
    return sorted(path.stem.split("_")[0].upper() for path in folder.glob("s*_panel.parquet"))
