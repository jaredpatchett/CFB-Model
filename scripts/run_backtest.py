#!/usr/bin/env python3
"""
Backtest the trained game model against REAL historical CFBD closing lines
and actual results — this is the real bar (beating the market), not just
whether the model's predicted margin is close to the actual final score.

Usage:
  python scripts/run_backtest.py
"""
import json
import os
import sys
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.models.game_model import GameMarginModel
from src.backtest import backtester


def load_historical_lines() -> pd.DataFrame:
    """Load every lines_{year}.csv found in data/raw/, already flattened to
    one row per game by cfbd_client.historical_lines_to_dataframe (see
    fetch_historical_data.py). Older cached CSVs fetched before that fix
    will be missing the market_spread_home/market_moneyline_* columns this
    script needs — caught explicitly below rather than failing on a
    confusing KeyError."""
    files = [f for f in os.listdir(config.DATA_RAW_DIR) if f.startswith("lines_") and f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f"{config.DATA_RAW_DIR}/{f}") for f in files]
    return pd.concat(frames, ignore_index=True)


def join_features_to_lines(features: pd.DataFrame, lines: pd.DataFrame) -> tuple:
    """Join on CFBD's own numeric game id when both sides have it. /games
    and /lines are both CFBD's own data sharing the same internal id
    scheme, so this is an exact join — no fuzzy team-name matching needed
    here (that's only required when matching CFBD to a DIFFERENT provider,
    like The Odds API, which is what export_dashboard_data.py's team
    matching handles for current/live lines).

    Falls back to a (season, week, homeTeam, awayTeam) join if 'id' is
    missing from either side, so this doesn't silently produce zero
    matches against an older cached features/lines CSV that predates this
    fix. Returns (joined_df, strategy_used) so the caller can report which
    path was actually taken.
    """
    if "id" in features.columns and "id" in lines.columns:
        merged = features.merge(lines, on="id", how="inner", suffixes=("", "_line"))
        return merged, "id"
    merged = features.merge(
        lines, on=["season", "week", "homeTeam", "awayTeam"], how="inner", suffixes=("", "_line")
    )
    return merged, "season+week+homeTeam+awayTeam (fallback — no shared 'id' column found)"


if __name__ == "__main__":
    features_path = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    model_path = f"{config.MODELS_DIR}/game_model.joblib"

    if not os.path.exists(features_path):
        raise SystemExit("Run scripts/build_features.py first.")
    if not os.path.exists(model_path):
        raise SystemExit("Run scripts/train_game_model.py first.")

    features = pd.read_csv(features_path)
    lines = load_historical_lines()
    if lines.empty:
        raise SystemExit("No historical lines found in data/raw/ — run scripts/fetch_historical_data.py first.")
    if "market_spread_home" not in lines.columns:
        raise SystemExit(
            "data/raw/lines_*.csv is missing market_spread_home — it was fetched before the "
            "historical_lines_to_dataframe fix. Re-run scripts/fetch_historical_data.py to refresh it."
        )

    print(f"Loaded {len(features)} historical feature rows and {len(lines)} historical lines with a real market number.")
    joined, strategy = join_features_to_lines(features, lines)
    print(f"  joined {len(joined)} of {len(features)} feature rows to a market line (join strategy: {strategy})")
    if joined.empty:
        raise SystemExit("Joined 0 games to a market line — nothing to backtest. Check that features and "
                          "lines cover the same season(s)/years.")

    model = GameMarginModel().load(model_path)
    scoreable = joined.dropna(subset=model.feature_columns).copy()
    print(f"  {len(scoreable)} of {len(joined)} joined games have complete model features "
          f"(early-season games with no rolling in-season history are excluded here, same as live use)")
    if scoreable.empty:
        raise SystemExit("0 games have complete features after the join — nothing to backtest.")

    scoreable["predicted_margin"] = model.predict_margin(scoreable)
    scoreable["predicted_home_win_prob"] = model.predict_home_win_prob(scoreable)

    spread_results = backtester.evaluate_spread(
        scoreable["predicted_margin"], scoreable["margin"], scoreable["market_spread_home"]
    )
    backtester.summarize(spread_results, "Spread (ATS vs. real CFBD closing line)")

    ml_results = backtester.evaluate_moneyline(
        scoreable["predicted_home_win_prob"], scoreable["home_win"]
    )
    backtester.summarize(ml_results, "Moneyline (calibration: model win-prob vs. actual outcome)")

    print("\nReminder: breakeven ATS win rate against standard -110 pricing is ~52.4%. "
          "Treat any result on a small early sample (well under ~200 graded games) as noisy, "
          "not a verdict — re-run this after more historical seasons/weeks are pulled.")

    # Persist results to the repo (rather than leaving them stranded in this
    # Action run's console log, which isn't fetchable outside the GitHub UI)
    # so build_dashboard.py can surface a real "Backtest Track Record" panel
    # and so results are diffable/trackable across runs as the model changes.
    seasons_covered = sorted(int(s) for s in scoreable["season"].dropna().unique()) if "season" in scoreable.columns else []
    backtest_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons_covered": seasons_covered,
        "join_strategy": strategy,
        "spread": {
            "n_games": spread_results["n_games"],
            "n_pushes": spread_results["n_pushes"],
            "ats_win_rate": (None if pd.isna(spread_results["ats_win_rate"]) else round(float(spread_results["ats_win_rate"]), 4)),
            "margin_mae": round(float(spread_results["margin_mae"]), 3),
            "breakeven_ats_rate": spread_results["breakeven_ats_rate"],
            "beat_market": (None if spread_results["beat_market"] is None else bool(spread_results["beat_market"])),
        },
        "moneyline": {
            "n_games": ml_results["n_games"],
            "accuracy": round(float(ml_results["accuracy"]), 4),
            "log_loss": round(float(ml_results["log_loss"]), 4),
        },
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/backtest_results.json", "w") as f:
        json.dump(backtest_output, f, indent=2)
    print(f"\nWrote docs/data/backtest_results.json ({spread_results['n_games']} graded spread games, "
          f"seasons {seasons_covered}).")
