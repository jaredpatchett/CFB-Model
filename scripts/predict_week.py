#!/usr/bin/env python3
"""
Load trained models + current data/current lines and produce predictions
with edges vs. the market, saved to data/predictions/.

This does NOT place any bets or connect to a sportsbook account — it only
compares model output to lines you've already pulled, so you can decide
what (if anything) to act on yourself.

Usage:
  python scripts/predict_week.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.models.game_model import GameMarginModel

if __name__ == "__main__":
    game_lines_path = f"{config.DATA_CURRENT_DIR}/game_lines.csv"
    model_path = f"{config.MODELS_DIR}/game_model.joblib"

    if not os.path.exists(game_lines_path):
        raise SystemExit("Run scripts/fetch_current_lines.py first.")
    if not os.path.exists(model_path):
        raise SystemExit("Run scripts/train_game_model.py first.")

    lines = pd.read_csv(game_lines_path)
    model = GameMarginModel()
    model = model.load(model_path)

    print(f"Loaded {len(lines)} current games.")
    print("NOTE: to actually score these, you still need to build this week's")
    print("team features (roll_margin_diff, sp_rating_diff, etc.) for each")
    print("matchup — see build_features.py. This script is the wiring point;")
    print("plug in this week's feature rows and it will output predicted")
    print("margin / win prob / edge vs. the market lines saved above.")

    os.makedirs(config.DATA_PREDICTIONS_DIR, exist_ok=True)
    lines.to_csv(f"{config.DATA_PREDICTIONS_DIR}/game_lines_snapshot.csv", index=False)
