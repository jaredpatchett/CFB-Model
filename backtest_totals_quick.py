#!/usr/bin/env python3
"""
Quick, one-off totals (Over/Under) backtest -- NOT the trained game model
(that only predicts margin, see src/models/game_model.py and
scripts/run_backtest.py). This fits a simple team points-for/points-against
average from whatever's cached in data/raw/lines_*.csv, uses it to predict
a total for each of the most recently completed weeks of --season, and
grades those predictions against the real CFBD closing over/under and
actual final scores via backtester.evaluate_totals.

Deliberately does NOT train/persist a model, touch the live dashboard, or
commit anything -- see .github/workflows/backtest_totals_quick.yml. Prints
results to the run's log only.

Usage:
  python scripts/backtest_totals_quick.py --season 2026 --weeks 2
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.backtest import backtester


def load_all_lines() -> pd.DataFrame:
    """Every lines_{year}.csv currently cached in data/raw/ (however many
    years fetch_historical_data.py --years pulled this run), concatenated.
    Each row is already one-per-game with actual scores + a single
    provider's closing market_over_under (see
    cfbd_client.historical_lines_to_dataframe)."""
    files = sorted(glob.glob(f"{config.DATA_RAW_DIR}/lines_*.csv"))
    if not files:
        print(f"No lines_*.csv found in {config.DATA_RAW_DIR}/ -- run "
              f"fetch_historical_data.py first.")
        sys.exit(1)
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} games with a closing line from {len(files)} cached file(s).")
    return df


def build_team_scoring_table(fit_games: pd.DataFrame) -> tuple:
    """Simple points-for/points-against average per team from completed
    games only -- deliberately not opponent-adjusted or era-weighted, this
    is the 'quick' diagnostic, not the real model. Returns
    (team_avgs_df indexed by team, league_avg_total float) so callers can
    fall back to a league-average split for any team not seen in fit_games
    (e.g. an FCS opponent, or a team with no completed games yet)."""
    completed = fit_games.dropna(subset=["homeScore", "awayScore"])
    home_rows = completed.rename(columns={
        "homeTeam": "team", "awayTeam": "opponent",
        "homeScore": "points_for", "awayScore": "points_against",
    })[["team", "opponent", "points_for", "points_against"]]
    away_rows = completed.rename(columns={
        "awayTeam": "team", "homeTeam": "opponent",
        "awayScore": "points_for", "homeScore": "points_against",
    })[["team", "opponent", "points_for", "points_against"]]
    long = pd.concat([home_rows, away_rows], ignore_index=True)

    team_avgs = long.groupby("team")[["points_for", "points_against"]].mean()
    league_avg_total = (completed["homeScore"] + completed["awayScore"]).mean()
    print(f"Fit team scoring averages from {len(completed)} completed games "
          f"({len(team_avgs)} teams); league-average total = {league_avg_total:.1f}.")
    return team_avgs, league_avg_total


def predict_total(home_team: str, away_team: str, team_avgs: pd.DataFrame,
                   league_avg_total: float) -> float:
    """predicted_total = (home's avg points-for + away's avg points-against)/2
    + (away's avg points-for + home's avg points-against)/2 -- i.e. each
    side's predicted score is the average of its own scoring rate and its
    opponent's rate of allowing points. Falls back to half the league
    average total per side for a team with no fitting-set games."""
    half_league_avg = league_avg_total / 2

    if home_team in team_avgs.index:
        home_for = team_avgs.loc[home_team, "points_for"]
        home_against = team_avgs.loc[home_team, "points_against"]
    else:
        home_for = home_against = half_league_avg

    if away_team in team_avgs.index:
        away_for = team_avgs.loc[away_team, "points_for"]
        away_against = team_avgs.loc[away_team, "points_against"]
    else:
        away_for = away_against = half_league_avg

    predicted_home = (home_for + away_against) / 2
    predicted_away = (away_for + home_against) / 2
    return predicted_home + predicted_away


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True,
                         help="Season to grade recent weeks of, e.g. 2026")
    parser.add_argument("--weeks", type=int, default=2,
                         help="How many of the most recently completed weeks to grade")
    args = parser.parse_args()

    all_lines = load_all_lines()

    season_games = all_lines[all_lines["season"] == args.season]
    completed_season_games = season_games.dropna(subset=["homeScore", "awayScore"])
    if completed_season_games.empty:
        print(f"No completed {args.season} games with a closing line found -- nothing to grade.")
        sys.exit(0)

    recent_weeks = sorted(completed_season_games["week"].unique())[-args.weeks:]
    print(f"Grading {args.season} weeks {recent_weeks}...")

    test_games = completed_season_games[completed_season_games["week"].isin(recent_weeks)]
    fit_games = all_lines.drop(test_games.index)

    team_avgs, league_avg_total = build_team_scoring_table(fit_games)

    test_games = test_games.copy()
    test_games["predicted_total"] = test_games.apply(
        lambda g: predict_total(g["homeTeam"], g["awayTeam"], team_avgs, league_avg_total),
        axis=1,
    )
    test_games["actual_total"] = test_games["homeScore"] + test_games["awayScore"]

    print("\nGame-by-game:")
    for _, g in test_games.iterrows():
        pick = "OVER" if g["predicted_total"] > g["market_over_under"] else "UNDER"
        hit = "push" if g["actual_total"] == g["market_over_under"] else (
            "hit" if (pick == "OVER") == (g["actual_total"] > g["market_over_under"]) else "miss"
        )
        print(f"  wk{int(g['week']):>2}  {g['awayTeam']:>22} @ {g['homeTeam']:<22}  "
              f"model={g['predicted_total']:.1f}  market={g['market_over_under']:.1f}  "
              f"actual={g['actual_total']:.0f}  pick={pick:<5} [{hit}]")

    results = backtester.evaluate_totals(
        test_games["predicted_total"], test_games["actual_total"], test_games["market_over_under"]
    )
    backtester.summarize(results, f"Quick totals backtest -- {args.season} weeks {recent_weeks}")
