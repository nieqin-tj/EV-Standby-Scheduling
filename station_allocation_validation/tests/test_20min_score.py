from scripts.score_20min import load_windows, station_row, summarize


def test_ten_station_mean_matches_paper_summary():
    frames = load_windows("mean")
    rows = [station_row(sid, frames[sid]) for sid in sorted(frames)]
    summary = summarize(rows)
    assert len(rows) == 10
    assert round(summary["mae_modules"], 2) == 0.62
    assert [round(v, 2) for v in summary["mae_modules_ci95"]] == [0.44, 0.80]
    assert round(summary["relative_mae_pct"], 2) == 5.44
    assert round(summary["normalized_bias_pct"], 2) == -2.74
