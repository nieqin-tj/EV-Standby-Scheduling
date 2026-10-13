"""Check that scoring and panels run from local data/ only."""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STATIONS = [f"S{i:02d}" for i in range(1, 11)]
ADDRESS = re.compile(r"路|号充电站|上海|浦东|奉贤")


def _ok(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print("ok:", message, flush=True)


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from station_allocation.io import load_panel
    from scripts.score_20min import load_windows, station_row, summarize

    for sid in STATIONS:
        path = DATA / "stations" / f"{sid.lower()}_panel.parquet"
        _ok(path.exists(), f"panel {sid}")
        frame = load_panel(sid)
        _ok(len(frame) > 0 and set(["demand_kw", "actual_kw", "split"]).issubset(frame.columns), f"panel columns {sid}")

        raw = DATA / "station_raw" / sid
        _ok(raw.is_dir(), f"station_raw {sid}")
        _ok(not ADDRESS.search(raw.name), f"no address in {raw.name}")
        _ok((raw / "orders.xlsx").exists() and (raw / "layout.xlsx").exists(), f"orders/layout {sid}")
        n_csv = len(list((raw / "curves").glob("S*.csv")))
        _ok(n_csv > 0, f"{sid} curves={n_csv}")

        for agg in ("mean", "max"):
            _ok((DATA / "module_20min" / "windows" / f"{sid.lower()}_test_20min_{agg}.csv").exists(), f"20min {sid} {agg}")

    mapping = DATA / "vehicle_curves" / "mapping.csv"
    _ok(mapping.exists(), "vehicle mapping")
    text = mapping.read_text(encoding="utf-8", errors="ignore")
    _ok("D:\\" not in text and "/Users/" not in text, "mapping has no local absolute paths")
    n_actual = len(list((DATA / "vehicle_curves" / "actual_curves").rglob("*.csv")))
    n_demand = len(list((DATA / "vehicle_curves" / "demand_curves").rglob("*.csv")))
    _ok(n_actual > 100 and n_demand > 100, f"vehicle curves actual={n_actual} demand={n_demand}")

    for manifest in ("actual_manifest.csv", "demand_manifest.csv"):
        body = (DATA / "vehicle_curves" / manifest).read_text(encoding="utf-8", errors="ignore")
        _ok("D:\\" not in body and "C:\\Users\\" not in body, f"{manifest} has no drive paths")

    rows = [station_row(sid, load_windows("mean")[sid]) for sid in STATIONS]
    summary = summarize(rows)
    _ok(round(summary["mae_modules"], 2) == 0.62, f"20-min MAE {summary['mae_modules']:.4f}")
    _ok(round(summary["relative_mae_pct"], 2) == 5.44, f"relative {summary['relative_mae_pct']:.4f}")
    _ok(round(summary["normalized_bias_pct"], 2) == -2.74, f"bias {summary['normalized_bias_pct']:.4f}")

    result = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
    _ok(result.returncode == 0, "pytest")
    print("DATA VERIFIED", flush=True)


if __name__ == "__main__":
    main()
