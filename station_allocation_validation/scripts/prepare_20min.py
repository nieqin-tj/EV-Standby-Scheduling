"""Copy frozen 20-min windows, minute predictions, and order inputs into data/module_20min.

Looks first in this repository. If a sibling station_sequence report tree exists, it is
used only to refresh local copies. Terminal names are dropped from minute predictions.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "station_sequence" / "reports"
SRC_NINE = SIBLING / "paper_20min_temporal_experiment"
SRC_S06 = SIBLING / "s06_paper_20min"
OUT = ROOT / "data" / "module_20min"
STATIONS = [f"S{i:02d}" for i in range(1, 11)]


def _src(sid: str) -> Path:
    return SRC_S06 if sid == "S06" else SRC_NINE


def main() -> None:
    windows = OUT / "windows"
    orders = OUT / "orders"
    minute = OUT / "minute"
    for folder in (windows, orders, minute):
        folder.mkdir(parents=True, exist_ok=True)
    for sid in STATIONS:
        src = _src(sid)
        stem = sid.lower()
        for agg in ("mean", "max"):
            name = f"{stem}_test_20min_{agg}.csv"
            origin = src / name
            dest = windows / name
            if origin.exists():
                shutil.copy2(origin, dest)
            elif not dest.exists():
                raise FileNotFoundError(name)
        order_name = f"{stem}_allowed_order_inputs.parquet"
        order_origin = src / order_name
        if order_origin.exists():
            shutil.copy2(order_origin, orders / order_name)
        elif not (orders / order_name).exists():
            raise FileNotFoundError(order_name)
        pred_name = f"{stem}_final_minute_predictions.parquet"
        pred_origin = src / pred_name
        pred_dest = minute / pred_name
        if pred_origin.exists():
            frame = pd.read_parquet(pred_origin)
            frame = frame.drop(columns=["terminal_name"], errors="ignore")
            frame.to_parquet(pred_dest, index=False)
        elif not pred_dest.exists():
            raise FileNotFoundError(pred_name)
        print(sid, "ok", flush=True)
    (OUT / "README.md").write_text(
        "Scoring files for the ten-station 20-min module-demand result.\n\n"
        "- `windows/`: station 20-min predicted and measured integer module demand (primary scoring input).\n"
        "- `minute/`: vehicle-minute reconstructed power after allocation (`prediction_kw`).\n"
        "- `orders/`: session-boundary inputs used to generate the request trajectories.\n"
        "Measured curve files are not included. Run `python scripts/score_20min.py` to recompute the station-equal MAE.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
