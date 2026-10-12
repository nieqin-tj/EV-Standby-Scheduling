"""Select numerical parameters on validation days, then score confirmation once."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from station_allocation.config import STATION_CONFIGURATIONS, StationConfiguration
from station_allocation.io import load_panel
from station_allocation.reconstruct import reconstruct_station
from station_allocation.scoring import cabinet_minute_occupancy, score_modules, score_power, session_quality

OUT = ROOT / "reports"
STATIONS = [f"S{i:02d}" for i in range(1, 11)]


def candidates(station: StationConfiguration) -> list[StationConfiguration]:
    if station.family == "declared_rule":
        if station.station_id == "S01":
            return [replace(station, policy_blend=b) for b in (0.5, 1.0)]
        return [
            replace(station, policy_blend=b, delivered_power_factor=f, startup_factor=s, startup_minutes=m)
            for b in (0.25, 0.5, 1.0)
            for f in (0.92, 1.0)
            for s, m in ((1.0, 0), (0.8, 2))
        ]
    options = [replace(station, capped_demand_blend=b, station_trigger_kw=None, station_cap_kw=None) for b in (0.0, 0.75, 1.0)]
    if station.station_trigger_kw is not None:
        options.append(station)
    return options


def reconstruct(frame: pd.DataFrame, station: StationConfiguration, projection_iterations: int = 1200) -> tuple[pd.DataFrame, dict]:
    prediction, audit = reconstruct_station(frame, station, projection_iterations=projection_iterations)
    out = frame.drop(columns=["prediction_kw"], errors="ignore").merge(
        prediction[["timestamp", "vehicle_id", "prediction_kw"]],
        on=["timestamp", "vehicle_id"],
        validate="one_to_one",
    )
    return out, audit


def select(frame: pd.DataFrame, station: StationConfiguration) -> tuple[StationConfiguration, list[dict]]:
    quality = frame.split.eq("validation") & frame.quality_pass
    rows, best, best_mae = [], None, np.inf
    for cfg in candidates(station):
        pred, audit = reconstruct(frame, cfg, projection_iterations=300)
        metrics = score_power(pred.loc[quality])
        mae = metrics["startup_weighted_point_mae_kw"]
        rows.append({"config": asdict(cfg), "validation_startup_weighted_mae_kw": mae, "physical_pass": audit["physical_audit"]["physical_pass"]})
        print(f"  {station.station_id} candidate mae={mae:.4f} policy={cfg.policy} blend={cfg.policy_blend if cfg.family=='declared_rule' else cfg.capped_demand_blend}", flush=True)
        if mae is not None and mae < best_mae:
            best, best_mae = cfg, mae
    if best is None:
        raise RuntimeError(f"{station.station_id}: no validation candidate")
    return best, rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records, selection = [], {}
    for sid in STATIONS:
        station = STATION_CONFIGURATIONS[sid]
        frame = load_panel(sid)
        frame["quality_pass"] = session_quality(frame)
        print(f"{sid}: {len(frame)} points, family={station.family}", flush=True)
        winner, rows = select(frame, station)
        pred, audit = reconstruct(frame, winner)
        pred.to_parquet(OUT / f"{sid.lower()}_predictions.parquet", index=False)
        selection[sid] = {"winner": asdict(winner), "candidates": rows, "physical_audit": audit}
        for split in ("validation", "confirmation"):
            for population, mask in (
                ("all_eligible", pred.split.eq(split)),
                ("quality_consistent", pred.split.eq(split) & pred.quality_pass),
            ):
                power = score_power(pred.loc[mask])
                modules = score_modules(cabinet_minute_occupancy(pred.loc[mask]))
                records.append({"station_id": sid, "family": winner.family, "split": split, "population": population, **power, **modules})
    pd.DataFrame(records).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "selection.json").write_text(json.dumps(selection, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    confirm = [r for r in records if r["split"] == "confirmation" and r["population"] == "quality_consistent"]
    lines = [
        "# Ten-station allocation validation",
        "",
        "Numerical parameters were selected on the validation split only. Confirmation metrics below were computed once after that choice was frozen.",
        "S01 and S02 use operator-declared discrete rules. S03–S10 use dynamic demand allocation with energy projection.",
        "The primary power metric down-weights the first three minutes of each session (weight 0.25).",
        "",
        "| Station | Family | Sessions | Power MAE (kW) | Module MAE | Relative module MAE | Bias (kW) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in confirm:
        rel = f"{row['relative_mae_pct']:.2f}%" if row["relative_mae_pct"] is not None else "—"
        lines.append(
            f"| {row['station_id']} | {row['family']} | {row['n_sessions']} | "
            f"{row['startup_weighted_point_mae_kw']:.3f} | {row['mae_modules']:.3f} | {rel} | {row['bias_kw']:+.3f} |"
        )
    lines += [
        "",
        "Quality-consistent sessions have at least five points, do not cross a split boundary, and keep order energy within 1 kWh of the measured-power integral. That filter affects scoring membership only.",
        "S06 energy projection may remain infeasible for a subset of sessions; those cases are retained in the diagnostic population.",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(pd.DataFrame(confirm)[["station_id", "n_sessions", "startup_weighted_point_mae_kw", "mae_modules", "relative_mae_pct"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
