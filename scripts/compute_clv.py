#!/usr/bin/env python3
"""
Compute CLV (closing-line value) for the model's own flagged spread picks,
using the line-snapshot log that export_dashboard_data.py appends to on
every workflow run (config.CLV_SNAPSHOTS_PATH). See src/analysis/clv.py's
module docstring for the full pipeline explanation and the CLV sign
convention.

Safe to run with an empty or missing snapshot log (writes an empty-but-valid
docs/data/clv_results.json and exits 0) -- expected for every run before the
model has flagged its first real pick, same 'don't fail the whole workflow
over a diagnostic step having nothing to report yet' pattern as
run_backtest.py (see manual_run.yml's continue-on-error on both steps).

Usage:
  python scripts/compute_clv.py
"""
import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import cfbd_client as cfbd
from src.analysis import clv

OUT_PATH = "docs/data/clv_results.json"


def _empty_result(note: str) -> dict:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_games": 0,
        "avg_clv_points": None,
        "median_clv_points": None,
        "pct_positive_clv": None,
        "note": note,
        "games": [],
    }


if __name__ == "__main__":
    if not os.path.exists(config.CLV_SNAPSHOTS_PATH):
        result = _empty_result("No line snapshots captured yet — expected before the model has "
                                "flagged its first qualifying spread play (see src/analysis/clv.py).")
        os.makedirs("docs/data", exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[compute_clv] {config.CLV_SNAPSHOTS_PATH} not found — wrote empty {OUT_PATH}.")
        sys.exit(0)

    snapshots = pd.read_csv(config.CLV_SNAPSHOTS_PATH)
    if snapshots.empty:
        result = _empty_result("Line-snapshot log exists but is empty.")
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[compute_clv] {config.CLV_SNAPSHOTS_PATH} is empty — wrote empty {OUT_PATH}.")
        sys.exit(0)

    # Earliest captured snapshot per game -- the real price a follower would
    # have gotten by acting the moment the model first flagged this game as
    # a play (see append_line_snapshots' docstring for why every run's
    # snapshot is kept rather than deduped at write time).
    snapshots = snapshots.sort_values("captured_at")
    first_snapshots = snapshots.groupby("game_id", as_index=False).head(1).reset_index(drop=True)
    print(f"[compute_clv] {len(snapshots)} total snapshot row(s) on file, "
          f"{len(first_snapshots)} distinct flagged game(s) tracked so far.")

    seasons = sorted(int(s) for s in first_snapshots["season"].dropna().unique())
    if not seasons:
        result = _empty_result("Snapshot log has no usable 'season' values.")
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[compute_clv] no usable season values — wrote empty {OUT_PATH}.")
        sys.exit(0)

    closing_frames = []
    for year in seasons:
        try:
            raw = cfbd.get_historical_lines(year)
            df = cfbd.historical_lines_to_dataframe(raw)
            if not df.empty:
                closing_frames.append(df)
        except Exception as e:
            print(f"[compute_clv] [warn] could not fetch {year} closing lines: {e}")
    if not closing_frames:
        result = _empty_result(f"Could not fetch any closing-line data for season(s) {seasons} — "
                                f"nothing gradeable yet (normal before any tracked game has kicked off).")
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[compute_clv] no closing-line data available — wrote empty {OUT_PATH}.")
        sys.exit(0)

    closing = pd.concat(closing_frames, ignore_index=True)
    # Only games CFBD shows as actually completed (real final score on file)
    # count as having a real CLOSING line -- for an in-progress/upcoming
    # game, whatever CFBD returns from /lines is just a current/opening
    # snapshot, not the closing number, and grading against it would be
    # comparing the model's captured price to a number that can still move.
    closing = closing.dropna(subset=["homeScore", "awayScore"])
    print(f"[compute_clv] {len(closing)} completed game(s) with a real closing line on file "
          f"across season(s) {seasons}.")

    matched = clv.compute_clv_for_snapshots(first_snapshots, closing)
    if matched.empty:
        result = _empty_result("None of the tracked flagged games have a completed closing line yet — "
                                "normal before any of them have kicked off.")
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[compute_clv] 0 of {len(first_snapshots)} tracked game(s) gradeable yet — wrote empty {OUT_PATH}.")
        sys.exit(0)

    avg_clv = float(matched["clv_points"].mean())
    median_clv = float(matched["clv_points"].median())
    pct_positive = float((matched["clv_points"] > 0).mean())

    games_records = []
    for _, r in matched.sort_values("commence_time").iterrows():
        games_records.append({
            "game_id": int(r["game_id"]),
            "season": int(r["season"]) if pd.notna(r["season"]) else None,
            "week": int(r["week"]) if pd.notna(r["week"]) else None,
            "home_team": r.get("home_team"),
            "away_team": r.get("away_team"),
            "side": r.get("side"),
            "recommended_spread_home": round(float(r["spread_home_at_capture"]), 1),
            "closing_spread_home": round(float(r["closing_spread_home"]), 1),
            "clv_points": round(float(r["clv_points"]), 2),
            "captured_at": r.get("captured_at"),
            "commence_time": r.get("commence_time"),
        })

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_games": len(matched),
        "avg_clv_points": round(avg_clv, 3),
        "median_clv_points": round(median_clv, 3),
        "pct_positive_clv": round(pct_positive, 4),
        "note": None,
        "games": games_records,
    }
    os.makedirs("docs/data", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[compute_clv] {len(matched)} graded pick(s): avg CLV {avg_clv:+.2f} pts, "
          f"median {median_clv:+.2f} pts, {pct_positive * 100:.1f}% positive CLV.")
    print(f"[compute_clv] Wrote {OUT_PATH}.")
