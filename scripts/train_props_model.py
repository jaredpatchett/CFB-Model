#!/usr/bin/env python3
"""
Train one player-props model per tracked stat (rec_yds, rush_yds, pass_yds,
receptions, etc.) on processed player features. Saves each to
models/props/<stat>.joblib. Stats without enough usable rows are skipped
with a warning rather than crashing the whole run.

Receptions and receiving yards (10/2026): when a PFF API key is available,
these two models also train on PFF routes run and targets -- the version
that had the lowest projection error on 12,000+ unseen 2025 player-games in
the props lab. They are trained exactly as tested there:
  - only on players who already have a catch that season,
  - hit chances from the latest season, projected by a model that never saw
    it (not from a random split).
A box-score-only version of each is always saved next to it as
<stat>_base.joblib; the live export falls back to it if PFF cannot be
reached, so props are still scored during an outage.

PFF data is pulled into a temp folder outside the repo and is never written
into data/ or docs/ -- the repo and its run artifacts are public.

Usage:
  python scripts/train_props_model.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.models.props_model import PlayerStatModel
from src.features.player_features import STAT_MAP, ROLLING_FEATURE_COLUMNS
from src.features import pff_features as pf


def load_pff_frame(df: pd.DataFrame):
    """The training frame with PFF features, or None (with the reason
    printed) if PFF is not set up or cannot be reached."""
    if not (os.environ.get("PFF_API_KEY") or "").strip():
        print("  [note] no PFF_API_KEY -- receiving models train on box-score inputs only")
        return None
    try:
        seasons = sorted(int(s) for s in df["season"].dropna().unique())
        frames = [pd.read_csv(f"{config.DATA_RAW_DIR}/games_{y}.csv") for y in seasons
                  if os.path.exists(f"{config.DATA_RAW_DIR}/games_{y}.csv")]
        if not frames:
            raise RuntimeError("no games_<year>.csv files found to line games up by date")
        frame, info = pf.training_frame(df, pd.concat(frames, ignore_index=True), seasons)
        print(f"  PFF routes and targets attached: {info}")
        return frame
    except Exception as e:
        print(f"  [warn] PFF data unavailable ({type(e).__name__}: {e}) -- "
              f"receiving models train on box-score inputs only this run")
        return None


def fit_with_pff(stat: str, frame: pd.DataFrame) -> tuple:
    """The props-lab 'pff' version. Returns (model, metrics)."""
    cols = list(ROLLING_FEATURE_COLUMNS) + list(pf.PFF_FEATURE_COLUMNS)
    rows = frame.dropna(subset=cols + [stat])
    rows = rows[(rows["games_played_prior"] >= 1) & (rows["roll_receptions"] > 0)]
    last = int(rows["season"].max())
    earlier, latest = rows[rows["season"] < last], rows[rows["season"] == last]
    if len(earlier) < 200 or len(latest) < 200:
        raise ValueError(f"not enough rows to train the PFF version of {stat} "
                         f"({len(earlier)} earlier, {len(latest)} in {last})")
    model = PlayerStatModel(stat)
    model.feature_columns = cols
    # Hit chances: how far off a model trained on earlier seasons was on the
    # latest season, which it never saw.
    probe = PlayerStatModel(stat).model.fit(earlier[cols], earlier[stat])
    preds = np.clip(probe.predict(latest[cols]), 0, None)
    resid = latest[stat].values.astype(float) - preds
    model.residual_std = float(np.std(resid))
    model._fit_residual_bins(preds, resid)
    # Projections: trained on every season.
    model.model.fit(rows[cols], rows[stat])
    mae = float(np.abs(resid).mean())
    print(f"[props_model:{stat}] PFF version | {last} error (model trained on earlier seasons): {mae:.2f} | "
          f"n={len(rows)} | inputs={len(cols)}")
    return model, {"holdout_mae": mae, "residual_std": model.residual_std, "n_train": len(rows)}


if __name__ == "__main__":
    path = f"{config.DATA_PROCESSED_DIR}/player_game_features.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found — run scripts/build_features.py first.")

    df = pd.read_csv(path)
    stats_to_model = sorted(set(STAT_MAP.values()))
    print("Checking for PFF routes/targets data...")
    pff_frame = load_pff_frame(df)

    for stat in stats_to_model:
        print(f"\nTraining model for {stat}...")
        try:
            model = PlayerStatModel(stat)
            metrics = model.fit(df)
            out_path = f"{config.MODELS_DIR}/props/{stat}.joblib"
            if stat in pf.PFF_STATS:
                # Box-score-only version, kept as the fallback for PFF outages.
                model.save(f"{config.MODELS_DIR}/props/{stat}_base.joblib")
                if pff_frame is not None:
                    try:
                        model, metrics = fit_with_pff(stat, pff_frame)
                    except Exception as e:
                        print(f"  [warn] PFF version of {stat} not trained ({e}) -- using the box-score version")
            model.save(out_path)
            print(f"  saved to {out_path} | {metrics}")
        except ValueError as e:
            print(f"  [skip] {e}")
