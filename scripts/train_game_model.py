#!/usr/bin/env python3
"""
Train the spread/moneyline (game margin) model on processed team features
and save it to models/game_model.joblib.

Usage:
  python scripts/train_game_model.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.models.game_model import GameMarginModel

if __name__ == "__main__":
    path = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run scripts/build_features.py first.")

    df = pd.read_csv(path)
    model = GameMarginModel()
    metrics = model.fit(df)
    model.save(f"{config.MODELS_DIR}/game_model.joblib")
    print(f"Saved model to {config.MODELS_DIR}/game_model.joblib")
    print(metrics)
