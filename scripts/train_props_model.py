#!/usr/bin/env python3
"""
Train one player-props model per tracked stat (rec_yds, rush_yds, pass_yds,
receptions, etc.) on processed player features. Saves each to
models/props/<stat>.joblib. Stats without enough usable rows are skipped
with a warning rather than crashing the whole run.

Usage:
  python scripts/train_props_model.py
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.models.props_model import PlayerStatModel
from src.features.player_features import STAT_MAP

if __name__ == "__main__":
    path = f"{config.DATA_PROCESSED_DIR}/player_game_features.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run scripts/build_features.py first.")

    df = pd.read_csv(path)
    stats_to_model = sorted(set(STAT_MAP.values()))

    for stat in stats_to_model:
        print(f"\nTraining model for {stat}...")
        try:
            model = PlayerStatModel(stat)
            metrics = model.fit(df)
            out_path = f"{config.MODELS_DIR}/props/{stat}.joblib"
            model.save(out_path)
            print(f"  saved to {out_path} | {metrics}")
        except ValueError as e:
            print(f"  [skip] {e}")
