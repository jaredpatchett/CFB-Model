"""
Player props model.

One regression model per stat category (rec_yds, rush_yds, pass_yds,
receptions, etc.), trained on that stat's own rolling usage features.
Predicts the player's expected value for the stat, then compares it to a
PrizePicks line the same way the game model compares margin to a spread:
predicted mean vs line -> over/under lean, plus an implied probability using
the position's residual std under a normal approximation.

This is intentionally simple (no opponent-defense-adjusted matchup term yet —
see README "Next steps"). Treat early-season / low-sample-size predictions
(games_played_prior < 3) as low-confidence; the predict_and_compare() output
flags this.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
import joblib
import os

from src.features.player_features import ROLLING_FEATURE_COLUMNS


class PlayerStatModel:
    def __init__(self, stat_name: str):
        self.stat_name = stat_name
        self.model = GradientBoostingRegressor(
            n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42
        )
        self.residual_std = None
        self.feature_columns = ROLLING_FEATURE_COLUMNS

    def fit(self, player_features_df: pd.DataFrame, min_games_played: int = 1, verbose: bool = True):
        data = player_features_df[player_features_df["games_played_prior"] >= min_games_played]
        data = data.dropna(subset=self.feature_columns + [self.stat_name])
        X = data[self.feature_columns]
        y = data[self.stat_name]
        if len(data) < 20:
            raise ValueError(
                f"Only {len(data)} usable rows for stat '{self.stat_name}' after filtering "
                f"(min_games_played={min_games_played}). Pull more weeks of data before training."
            )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        mae = mean_absolute_error(y_test, self.model.predict(X_test))
        residuals = y_train - self.model.predict(X_train)
        self.residual_std = float(np.std(residuals))

        if verbose:
            print(f"[props_model:{self.stat_name}] holdout MAE: {mae:.2f} | "
                  f"residual std: {self.residual_std:.2f} | n={len(data)}")
        return {"holdout_mae": mae, "residual_std": self.residual_std, "n_train": len(X_train)}

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features_df[self.feature_columns])

    def predict_and_compare(self, features_row: pd.Series, prop_line: float) -> dict:
        pred = float(self.model.predict(features_row[self.feature_columns].to_frame().T)[0])
        confidence = "low" if features_row.get("games_played_prior", 0) < 3 else "normal"
        over_prob = None
        if self.residual_std and self.residual_std > 0:
            over_prob = float(1 - norm.cdf(prop_line, loc=pred, scale=self.residual_std))
        return {
            "stat": self.stat_name,
            "predicted_value": pred,
            "prop_line": prop_line,
            "edge": pred - prop_line,
            "lean": "over" if pred > prop_line else "under",
            "over_probability": over_prob,
            "confidence": confidence,
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "residual_std": self.residual_std,
                     "feature_columns": self.feature_columns, "stat_name": self.stat_name}, path)

    @classmethod
    def load(cls, path: str) -> "PlayerStatModel":
        payload = joblib.load(path)
        instance = cls(payload["stat_name"])
        instance.model = payload["model"]
        instance.residual_std = payload["residual_std"]
        instance.feature_columns = payload["feature_columns"]
        return instance
