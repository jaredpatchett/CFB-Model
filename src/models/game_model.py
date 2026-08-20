"""
Spread / moneyline model.

Approach: regress predicted point margin (home - away) from team features,
then derive:
  - a spread pick: compare predicted margin to the market spread
  - a moneyline win probability: assume margin residuals are approximately
    normal (a standard, if simplifying, assumption in point-spread modeling)
    and use the training residual std to convert predicted margin into
    P(home wins) via the normal CDF.

This is a solid baseline, not a finished sharp model. Realistic next steps
noted in the README: add turnover/PPA-based advanced stats, injury/roster
availability, weather, and travel/rest features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm
import joblib
import os

from src.features.team_features import FEATURE_COLUMNS, TARGET_MARGIN


class GameMarginModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
        )
        self.residual_std = None
        self.feature_columns = FEATURE_COLUMNS

    def fit(self, features_df: pd.DataFrame, verbose: bool = True):
        # Some FEATURE_COLUMNS (pace_diff, returning_production_diff) depend
        # on CFBD endpoints whose real-world coverage hasn't been verified
        # from this dev sandbox, and older cached team_game_features.csv
        # files predate these columns entirely. Two failure modes to guard
        # against, not just one: a column MISSING from features_df entirely
        # (dropna(subset=...) raises KeyError, a hard crash) and a column
        # present but ALL-NULL (dropna would silently drop every row, then
        # train_test_split on an empty frame crashes too). Both get the same
        # fix: drop that specific column from what THIS model actually uses,
        # loudly, rather than let either failure mode take down training
        # (and with it run_backtest.py and the in-season switchover, which
        # both depend on a trained model existing at all). self.feature_columns
        # is reassigned to the usable subset and persisted by save()/load(),
        # so predict_margin/score_with_trained_model automatically stay
        # consistent with whatever this specific model was actually trained on.
        usable, skipped = [], []
        for c in self.feature_columns:
            if c not in features_df.columns:
                skipped.append((c, "column not present in this data"))
            elif features_df[c].notna().sum() == 0:
                skipped.append((c, "100% null in this data"))
            else:
                usable.append(c)
        if skipped:
            if verbose:
                print(f"[game_model] dropping unusable feature column(s) for this training run: {skipped}")
            self.feature_columns = usable

        data = features_df.dropna(subset=self.feature_columns + [TARGET_MARGIN])
        if verbose and len(data) < len(features_df):
            # Which remaining feature(s) are still causing PARTIAL row drops
            # (as opposed to the all-or-nothing case handled above) — a
            # silently-shrunk training set is exactly the kind of thing this
            # project has tried hard not to let happen unnoticed.
            null_counts = {c: int(features_df[c].isna().sum()) for c in self.feature_columns + [TARGET_MARGIN]
                           if c in features_df.columns and features_df[c].isna().any()}
            print(f"[game_model] {len(features_df) - len(data)} of {len(features_df)} rows dropped "
                  f"(missing a required feature): {null_counts}")
        if data.empty:
            raise ValueError(
                "0 rows usable for training after dropping nulls — check that "
                "team_game_features.csv actually has real values, not just present columns."
            )
        X = data[self.feature_columns]
        y = data[TARGET_MARGIN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.model.fit(X_train, y_train)
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        residuals = y_train - self.model.predict(X_train)
        self.residual_std = float(np.std(residuals))

        if verbose:
            print(f"[game_model] holdout MAE: {mae:.2f} points | "
                  f"train residual std: {self.residual_std:.2f}")
        return {"holdout_mae": mae, "residual_std": self.residual_std, "n_train": len(X_train)}

    def predict_margin(self, features_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features_df[self.feature_columns])

    def predict_home_win_prob(self, features_df: pd.DataFrame) -> np.ndarray:
        margins = self.predict_margin(features_df)
        if self.residual_std is None or self.residual_std == 0:
            raise ValueError("Model must be fit before predicting win probability.")
        return norm.cdf(margins / self.residual_std)

    def compare_to_market_spread(self, predicted_margin: float, market_home_spread: float) -> dict:
        """market_home_spread uses standard sportsbook convention: negative
        means home team favored by that many points."""
        edge = predicted_margin - (-market_home_spread)
        return {
            "predicted_margin": predicted_margin,
            "market_home_spread": market_home_spread,
            "edge_points": edge,
            "lean": "home" if edge > 0 else "away",
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "residual_std": self.residual_std,
                     "feature_columns": self.feature_columns}, path)

    @classmethod
    def load(cls, path: str) -> "GameMarginModel":
        payload = joblib.load(path)
        instance = cls()
        instance.model = payload["model"]
        instance.residual_std = payload["residual_std"]
        instance.feature_columns = payload["feature_columns"]
        return instance
