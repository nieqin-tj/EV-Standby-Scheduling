"""Verify frozen 20-min files already stored under data/module_20min."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "module_20min"
STATIONS = [f"S{i:02d}" for i in range(1, 11)]


def main() -> None:
    missing = []
    for sid in STATIONS:
        stem = sid.lower()
        for relative in (
            OUT / "windows" / f"{stem}_test_20min_mean.csv",
            OUT / "windows" / f"{stem}_test_20min_max.csv",
            OUT / "orders" / f"{stem}_allowed_order_inputs.parquet",
            OUT / "minute" / f"{stem}_final_minute_predictions.parquet",
        ):
            if not relative.exists():
                missing.append(str(relative.relative_to(ROOT)))
    if missing:
        raise SystemExit("missing 20-min files:\n" + "\n".join(missing))
    print("20-min files present for S01–S10", flush=True)


if __name__ == "__main__":
    main()
