#!/usr/bin/env python3
"""
Train the game totals model (11/2026) on every finished season in
data/processed/team_game_features.csv and save models/totals_model.joblib.

Quick (one gradient-boosted model, a few seconds), so the daily run trains it
every time from the saved feature table. See src/models/totals_model.py.

Usage (after build_features.py, or with a restored data/processed/):
  python scripts/train_totals_model.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

import config
from src.models.totals_model import TotalsModel

OUT = f"{config.MODELS_DIR}/totals_model.joblib"


def load_training_rows() -> pd.DataFrame:
    feats = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/team_game_features.csv", low_memory=False)
    if {"homePoints", "awayPoints"} <= set(feats.columns):
        feats["total"] = pd.to_numeric(feats["homePoints"], errors="coerce") + pd.to_numeric(feats["awayPoints"], errors="coerce")
        return feats
    frames = [pd.read_csv(f, usecols=lambda c: c in ("id", "homePoints", "awayPoints"))
              for f in glob.glob(f"{config.DATA_RAW_DIR}/games_*.csv")]
    g = pd.concat(frames, ignore_index=True).dropna()
    g["total"] = g["homePoints"] + g["awayPoints"]
    return feats.merge(g[["id", "total"]], on="id", how="left")


def main():
    rows = load_training_rows()
    m = TotalsModel().fit(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    m.save(OUT)
    print(f"[totals_model] trained on {m.n_train} games, {len(m.feature_columns)} inputs, "
          f"residual sd {m.residual_std:.2f} -> {OUT}")
    print(f"[totals_model] top inputs: {m.importances(8)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Never block the dashboard over the totals model: the export just
        # shows no totals this run.
        print(f"[totals_model] not trained ({type(e).__name__}: {e})")
