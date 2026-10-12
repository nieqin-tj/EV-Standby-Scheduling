"""Assemble the ten-station minute panels used by the validation run.

All inputs are local copies under data/. This script does not read operator
extracts outside the repository.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "stations"
COLUMNS = [
    "station_id", "timestamp", "vehicle_id", "cabinet_id", "terminal_name",
    "demand_kw", "actual_kw", "energy_kwh", "gun_cap_kw", "cabinet_cap_kw",
    "module_count", "module_power_kw", "split",
]
SOURCES = {
    "S01": ROOT / "data/station_01_02_rule_calibration/s01_panel.parquet",
    "S02": ROOT / "data/station_01_02_rule_calibration/s02_panel.parquet",
    "S03": ROOT / "data/hierarchical_resource_stations_02_08/s03_panel.parquet",
    "S04": ROOT / "data/hierarchical_resource_stations_02_08/s04_panel.parquet",
    "S05": ROOT / "data/hierarchical_resource_stations_02_08/s05_panel.parquet",
    "S06": ROOT / "data/stations/s06_panel.parquet",
    "S07": ROOT / "data/hierarchical_resource_stations_02_08/s07_panel.parquet",
    "S08": ROOT / "data/hierarchical_resource_stations_02_08/s08_panel.parquet",
    "S09": ROOT / "data/stations/s09_panel.parquet",
    "S10": ROOT / "data/stations/s10_panel.parquet",
}


def _keep(frame: pd.DataFrame, station_id: str) -> pd.DataFrame:
    data = frame.copy()
    data["station_id"] = station_id
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    if "energy_kwh" not in data.columns:
        data["energy_kwh"] = data["order_energy_kwh"]
    if "terminal_name" not in data.columns:
        data["terminal_name"] = ""
    out = data[COLUMNS].copy()
    out["demand_kw"] = pd.to_numeric(out.demand_kw, errors="raise").clip(lower=0)
    out["actual_kw"] = pd.to_numeric(out.actual_kw, errors="raise").clip(lower=0)
    return out.sort_values(["timestamp", "vehicle_id"]).reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"panels": {}}
    for sid, path in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(path)
        frame = _keep(pd.read_parquet(path), sid)
        dest = OUT / f"{sid.lower()}_panel.parquet"
        frame.to_parquet(dest, index=False)
        counts = frame.split.value_counts().to_dict()
        manifest["panels"][sid] = {
            "rows": int(len(frame)),
            "sessions": int(frame.vehicle_id.nunique()),
            "source": str(path.relative_to(ROOT)).replace("\\", "/"),
            "split_rows": {str(k): int(v) for k, v in counts.items()},
            "start": str(frame.timestamp.min()),
            "end": str(frame.timestamp.max()),
        }
        print(f"{sid}: {len(frame)} rows, {frame.vehicle_id.nunique()} sessions", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
