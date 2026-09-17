#!/usr/bin/env python3
"""
ONE-OFF, standalone diagnostic: "how would a simple total-points (Over/
Under) estimate have performed against real closing lines over the most
recently completed weeks of the CURRENT season?"

This is deliberately NOT wired into the production pipeline (manual_run.yml,
export_dashboard_data.py, build_dashboard.py) and changes nothing about the
live site -- it's a quick, throwaway analysis to answer one question, run
via its own one-off workflow (.github/workflows/backtest_totals_quick.yml),
not the scheduled/manual production run.

Why this exists at all: the trained GameMarginModel only ever predicts
MARGIN (home - away), never the TOTAL (home + away) -- see
src/models/game_model.py's own docstring. The market's totals number is
already fetched everywhere (The Odds API's total_over/total_under for live
games, CFBD's overUnder for historical ones) but nothing in this pipeline
has ever compared anything to it, simply because no total-points estimate
existed to compare it with. This script is the smallest reasonable one --
NOT a trained, persisted, production model -- built to answer the question
asked without committing to a whole new live market. (If this look
promising and a live Total market is wanted, that's a separate, bigger
build -- a real TotalPointsModel trained alongside game_model.py, wired
into export_dashboard_data.py and the Edge Board, same as Spread/Moneyline.)

Estimator: total_points ~ home_roll_ppg_for + home_roll_ppg_against +
away_roll_ppg_for + away_roll_ppg_against -- a plain linear regression fit
on every historical completed game in data/processed/team_game_features.csv
(the same rolling in-season scoring-average features the margin model
already computes -- see src/features/team_features.py's _rolling_team_form,
already lag-safe/leakage-free by construction). Graded against CFBD's real
historical closing totals line + real final score for the N most recently
completed weeks of the CURRENT season specifically (not a prior season --
"these past two weeks" per the question this was built to answer).

Usage:
  python scripts/backtest_totals_quick.py --season 2026 --weeks 2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.data import cfbd_client as cfbd
from src.backtest import backtester

TOTAL_FEATURES = ["home_roll_ppg_for", "home_roll_ppg_against", "away_roll_ppg_for", "away_roll_ppg_against"]


def fit_quick_total_model(features_df: pd.DataFrame) -> LinearRegression:
    df = features_df.dropna(subset=TOTAL_FEATURES + ["homePoints", "awayPoints"]).copy()
    if len(df) < 30:
        raise SystemExit(
            f"Only {len(df)} usable historical rows to fit a total-points estimate on (need "
            f"team_game_features.csv covering more seasons -- re-run fetch_historical_data.py / "
            f"build_features.py with a wider --years range)."
        )
    df["total_points"] = df["homePoints"] + df["awayPoints"]
    X = df[TOTAL_FEATURES].to_numpy()
    y = df["total_points"].to_numpy()
    reg = LinearRegression().fit(X, y)
    resid_std = float(np.std(y - reg.predict(X)))
    coefs = dict(zip(TOTAL_FEATURES, np.round(reg.coef_, 3)))
    print(f"[quick total model] fit on {len(df)} historical games | "
          f"train MAE: {mean_absolute_error(y, reg.predict(X)):.2f} pts | resid std: {resid_std:.2f} pts")
    print(f"  coefficients: {coefs} | intercept: {reg.intercept_:.2f}")
    return reg


def weeks_with_gradeable_data(raw_games: list, n_weeks: int) -> list:
    """The N most recent weeks that have at least one game with BOTH a real
    final score AND a real closing totals line -- determined from CFBD's
    own data rather than hardcoded, so this stays correct whenever it's
    actually run, not just as of when this script was written."""
    weeks_seen = set()
    for g in raw_games:
        if g.get("homeScore") is None or g.get("awayScore") is None:
            continue
        lines = g.get("lines") or []
        if any(l.get("overUnder") is not None for l in lines):
            weeks_seen.add(g.get("week"))
    weeks = sorted(w for w in weeks_seen if w is not None)
    if not weeks:
        raise SystemExit(
            "No completed games this season have both a final score and a real closing totals "
            "line yet -- too early in the season, or CFBD hasn't posted lines for it yet."
        )
    return weeks[-n_weeks:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--weeks", type=int, default=2, help="how many of the most recent completed weeks to grade")
    args = parser.parse_args()

    features_path = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if not os.path.exists(features_path):
        raise SystemExit(
            "Run scripts/fetch_historical_data.py and scripts/build_features.py first, with a "
            "--years range wide enough to fit on plus the current season for grading (e.g. "
            "'2022 2023 2024 2025 2026')."
        )
    features = pd.read_csv(features_path)

    reg = fit_quick_total_model(features)

    print(f"\nPulling {args.season} historical lines from CFBD...")
    raw = cfbd.get_historical_lines(args.season)
    weeks = weeks_with_gradeable_data(raw, args.weeks)
    print(f"Grading against {args.season} weeks: {weeks}")

    lines_df = cfbd.historical_lines_to_dataframe(raw)
    lines_df = lines_df[lines_df["week"].isin(weeks)]
    lines_df = lines_df.dropna(subset=["market_over_under", "homeScore", "awayScore"])
    print(f"  {len(lines_df)} games in those weeks have both a real closing total and a final score")

    this_season = features[features["season"] == args.season].copy()
    joined = lines_df.merge(
        this_season, on=["season", "week", "homeTeam", "awayTeam"], how="inner", suffixes=("", "_feat")
    )
    print(f"  {len(joined)} of those joined to real rolling-form features")
    scoreable = joined.dropna(subset=TOTAL_FEATURES).copy()
    print(f"  {len(scoreable)} of those have complete features to actually predict (early-week "
          f"games with no rolling in-season history yet are excluded, same as every other backtest "
          f"in this repo)")
    if scoreable.empty:
        raise SystemExit("Nothing scoreable -- too early in the season for rolling features to exist for these games yet.")

    scoreable["predicted_total"] = reg.predict(scoreable[TOTAL_FEATURES].to_numpy())
    scoreable["actual_total"] = scoreable["homeScore"] + scoreable["awayScore"]

    results = backtester.evaluate_totals(
        scoreable["predicted_total"], scoreable["actual_total"], scoreable["market_over_under"]
    )
    backtester.summarize(results, f"Totals (Over/Under), {args.season} weeks {weeks} vs real CFBD closing lines")

    print("\nPer-game detail:")
    for _, r in scoreable.sort_values(["week", "homeTeam"]).iterrows():
        lean = "OVER" if r["predicted_total"] > r["market_over_under"] else "UNDER"
        if r["actual_total"] > r["market_over_under"]:
            actual_side = "OVER"
        elif r["actual_total"] < r["market_over_under"]:
            actual_side = "UNDER"
        else:
            actual_side = "PUSH"
        outcome = "push" if actual_side == "PUSH" else ("hit" if lean == actual_side else "miss")
        print(f"  wk{int(r['week'])} {r['awayTeam']} @ {r['homeTeam']}: model {r['predicted_total']:.1f} vs "
              f"line {r['market_over_under']:.1f} -> leans {lean} | actual {r['actual_total']:.0f} "
              f"({actual_side}) -> {outcome}")

    print("\nReminder: this is a quick, not-trained-for-production estimator on a very small sample "
          "(a couple weeks of games) -- read this as a rough directional signal, not a validated "
          "edge. Breakeven O/U hit rate against standard -110 pricing is ~52.4%.")
