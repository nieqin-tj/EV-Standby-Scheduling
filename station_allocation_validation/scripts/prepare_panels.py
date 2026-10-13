"""Verify the ten-station minute panels already stored under data/stations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from station_allocation.io import load_panel

OUT = ROOT / "data" / "stations"
STATIONS = [f"S{i:02d}" for i in range(1, 11)]


def main() -> None:
    manifest = {"panels": {}}
    for sid in STATIONS:
        frame = load_panel(sid)
        counts = frame.split.value_counts().to_dict()
        manifest["panels"][sid] = {
            "rows": int(len(frame)),
            "sessions": int(frame.vehicle_id.nunique()),
            "source": f"data/stations/{sid.lower()}_panel.parquet",
            "split_rows": {str(k): int(v) for k, v in counts.items()},
            "start": str(frame.timestamp.min()),
            "end": str(frame.timestamp.max()),
        }
        print(f"{sid}: {len(frame)} rows, {frame.vehicle_id.nunique()} sessions", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
