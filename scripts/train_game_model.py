#!/usr/bin/env python3
"""
Train the spread/moneyline (game margin) model on processed team features
and save it to models/game_model.joblib.

Usage:
  python scripts/train_game_model.py
"""
import json
import os
import sys
from datetime import datetime, timezone
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

    # Persisted (not just printed) so this is inspectable after the fact --
    # e.g. the next time a backtest number moves a lot after adding/changing
    # a feature, this answers "which feature actually drove that" from real
    # data instead of guessing. See GameMarginModel.fit()'s docstring for
    # why this matters: it's an impurity-based importance, a cheap first
    # look, not a rigorous ablation study on its own.
    diagnostics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns_used": model.feature_columns,
        "feature_importances": metrics.get("feature_importances", {}),
        "holdout_mae": metrics.get("holdout_mae"),
        "residual_std": metrics.get("residual_std"),
        "n_train": metrics.get("n_train"),
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/model_diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)
    print("Wrote docs/data/model_diagnostics.json")
