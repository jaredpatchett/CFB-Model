#!/usr/bin/env python3
"""
Backtest the trained game model against historical CFBD closing lines and
actual results.

Usage:
  python scripts/run_backtest.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.models.game_model import GameMarginModel
from src.backtest import backtester

if __name__ == "__main__":
    features_path = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    model_path = f"{config.MODELS_DIR}/game_model.joblib"
    lines_glob_years = [f for f in os.listdir(config.DATA_RAW_DIR) if f.startswith("lines_")]

    if not os.path.exists(features_path):
        raise SystemExit("Run scripts/build_features.py first.")
    if not os.path.exists(model_path):
        raise SystemExit("Run scripts/train_game_model.py first.")
    if not lines_glob_years:
        raise SystemExit("No historical lines found in data/raw/ — run scripts/fetch_historical_data.py first.")

    features = pd.read_csv(features_path)
    lines = pd.concat(
        [pd.read_csv(f"{config.DATA_RAW_DIR}/{f}") for f in lines_glob_years],
        ignore_index=True
    )

    model = GameMarginModel().load(model_path)
    predicted_margin = model.predict_margin(features.dropna(subset=model.feature_columns))
    valid = features.dropna(subset=model.feature_columns).copy()
    valid["predicted_margin"] = predicted_margin
    valid["predicted_home_win_prob"] = model.predict_home_win_prob(valid)

    # NOTE: lines_{year}.csv from CFBD needs matching to games by gameId + book
    # to get a market spread per game. This wiring is left explicit here
    # rather than hidden, since CFBD's /lines schema varies by sportsbook
    # coverage per game and deserves a manual check before trusting it.
    print(f"Scored {len(valid)} historical games.")
    print("To finish the backtest: merge `valid` with the lines_*.csv files on "
          "gameId, pick a consistent sportsbook column, then call "
          "backtester.evaluate_spread() and evaluate_moneyline().")

    results = backtester.evaluate_moneyline(
        valid["predicted_home_win_prob"], valid["home_win"]
    )
    backtester.summarize(results, "Moneyline (model self-consistency, no market comparison yet)")
