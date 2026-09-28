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

    # ---- Walk-forward (out-of-sample) backtest -- rebuilt 9/2026 ----
    # The old version loaded the production model (trained on a random 80%
    # of ALL historical games) and graded it on 100% of those same games, so
    # roughly 4 of every 5 "backtest" games were ones the model had already
    # trained on, answers included. That in-sample overlap is what produced
    # the 61.25% ATS number, which live results never came close to.
    #
    # Now each season is graded by a model trained ONLY on earlier seasons
    # -- exactly the situation live betting is in. The first season on file
    # has nothing earlier to train on, so it's used for training only, never
    # graded. The production model (models/game_model.joblib) is untouched;
    # this script only trains temporary per-season models for grading.
    if "season" not in joined.columns:
        raise SystemExit("Joined data has no 'season' column -- can't run a walk-forward backtest.")
    # Only seasons with prior-year SP+ on file count -- the earliest season
    # has none (ratings are joined from the year before), so a model trained
    # on it alone would be missing its main inputs.
    has_sp = features["sp_rating_diff"].notna() if "sp_rating_diff" in features.columns else features["season"].notna()
    seasons = sorted(int(x) for x in features.loc[has_sp, "season"].dropna().unique())
    features = features[features["season"].isin(seasons)]
    print(f"\nWalk-forward backtest over seasons {seasons}: each season is predicted by a model "
          f"trained only on the seasons before it.")

    graded_parts = []
    for test_season in seasons[1:]:
        train_rows = features[features["season"] < test_season]
        fold_model = GameMarginModel()
        try:
            fold_model.fit(train_rows, verbose=False)
        except Exception as e:
            print(f"  {test_season}: skipped -- could not train on earlier seasons ({e})")
            continue
        test_rows = joined[joined["season"] == test_season].dropna(subset=fold_model.feature_columns).copy()
        if test_rows.empty:
            print(f"  {test_season}: skipped -- no games with complete features and a market line")
            continue
        test_rows["predicted_margin"] = fold_model.predict_margin(test_rows)
        test_rows["predicted_home_win_prob"] = fold_model.predict_home_win_prob(test_rows)
        fold = backtester.evaluate_spread(
            test_rows["predicted_margin"], test_rows["margin"], test_rows["market_spread_home"]
        )
        print(f"  {test_season}: trained on {len(train_rows)} earlier-season rows, graded "
              f"{fold['n_games']} games -> {fold['ats_win_rate']:.1%} ATS")
        graded_parts.append(test_rows)

    if not graded_parts:
        raise SystemExit("No season could be graded out-of-sample -- need at least two seasons of data.")
    scoreable = pd.concat(graded_parts, ignore_index=True)

    spread_results = backtester.evaluate_spread(
        scoreable["predicted_margin"], scoreable["margin"], scoreable["market_spread_home"]
    )
    backtester.summarize(spread_results, "Spread (ATS vs. real CFBD closing line, out-of-sample)")

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
        "method": "walk-forward: each season graded by a model trained only on earlier seasons",
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
