"""Reproduce the station-equal 20-min module-demand scores from frozen windows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from station_allocation.module_metrics import score_module_demand

WINDOWS = ROOT / "data" / "module_20min" / "windows"
OUT = ROOT / "reports"
T_CRIT_DF9 = 2.2621571627409915  # Student t, two-sided 95%, df = 9
STATIONS = [f"S{i:02d}" for i in range(1, 11)]


def load_windows(aggregation: str = "mean") -> dict[str, pd.DataFrame]:
    frames = {}
    for sid in STATIONS:
        path = WINDOWS / f"{sid.lower()}_test_20min_{aggregation}.csv"
        frame = pd.read_csv(path, parse_dates=["window"])
        need = {"station_id", "window", "predicted_modules", "actual_modules"}
        if not need.issubset(frame.columns):
            raise ValueError(f"{path.name} missing {sorted(need - set(frame.columns))}")
        frames[sid] = frame
    return frames


def station_row(sid: str, frame: pd.DataFrame) -> dict:
    metrics = score_module_demand(frame)
    metrics["station_id"] = sid
    metrics["normalized_bias_pct"] = (
        metrics["bias_modules"] / metrics["actual_mean_modules"] * 100.0
        if metrics["actual_mean_modules"]
        else None
    )
    return metrics


def summarize(rows: list[dict]) -> dict:
    mae = np.array([row["mae_modules"] for row in rows], dtype=float)
    rel = np.array([row["relative_mae_pct"] for row in rows], dtype=float)
    nb = np.array([row["normalized_bias_pct"] for row in rows], dtype=float)
    se = float(mae.std(ddof=1) / np.sqrt(len(mae)))
    mean_mae = float(mae.mean())
    return {
        "n_stations": int(len(rows)),
        "mae_modules": mean_mae,
        "mae_modules_ci95": [mean_mae - T_CRIT_DF9 * se, mean_mae + T_CRIT_DF9 * se],
        "relative_mae_pct": float(rel.mean()),
        "normalized_bias_pct": float(nb.mean()),
    }


def write_report(rows: list[dict], summary: dict) -> str:
    lines = [
        "# Ten-station 20-min module demand",
        "",
        "Primary aggregation is mean over each 20-min cabinet window, then ceiling to integer modules and summing to the station. Scores are station-equal.",
        "",
        f"Station-level module-demand MAE = **{summary['mae_modules']:.2f}** modules "
        f"(95% CI {summary['mae_modules_ci95'][0]:.2f}–{summary['mae_modules_ci95'][1]:.2f}). "
        f"Relative MAE = {summary['relative_mae_pct']:.2f}%. Normalized bias = {summary['normalized_bias_pct']:.2f}%.",
        "",
        "| Station | Windows | MAE | RMSE | Bias | Actual mean | Relative MAE | Normalized bias |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['station_id']} | {row['windows']} | {row['mae_modules']:.3f} | {row['rmse_modules']:.3f} | "
            f"{row['bias_modules']:+.3f} | {row['actual_mean_modules']:.3f} | {row['relative_mae_pct']:.2f}% | "
            f"{row['normalized_bias_pct']:+.2f}% |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    frames = load_windows("mean")
    rows = [station_row(sid, frames[sid]) for sid in STATIONS]
    summary = summarize(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "module_20min_metrics.csv", index=False, encoding="utf-8-sig")
    (OUT / "module_20min_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUT / "module_20min.md").write_text(write_report(rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"MAE {summary['mae_modules']:.2f} (95% CI {summary['mae_modules_ci95'][0]:.2f}–{summary['mae_modules_ci95'][1]:.2f}); "
          f"relative {summary['relative_mae_pct']:.2f}%; bias {summary['normalized_bias_pct']:.2f}%", flush=True)


if __name__ == "__main__":
    main()
