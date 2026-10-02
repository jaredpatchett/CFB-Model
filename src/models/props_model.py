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
        self.residual_bin_edges = []
        self.residual_bin_stds = []
        self.residual_bin_samples = []
        self.feature_columns = ROLLING_FEATURE_COLUMNS

    def fit(self, player_features_df: pd.DataFrame, min_games_played: int = 1, verbose: bool = True):
        # Same robustness fix as GameMarginModel.fit() (src/models/game_model.py)
        # and for the same reason: opp_pass_def_success_rate/
        # opp_rush_def_success_rate depend on CFBD's /stats/season/advanced
        # coverage, unverified from this dev sandbox, and older cached
        # player_game_features.csv files predate these columns entirely.
        # Drop a column that's missing outright or 100% null from what THIS
        # model actually uses, rather than crash with a KeyError or train on
        # 0 rows -- self.feature_columns is reassigned so predict()/
        # predict_and_compare() automatically stay consistent.
        usable, skipped = [], []
        for c in self.feature_columns:
            if c not in player_features_df.columns:
                skipped.append((c, "column not present in this data"))
            elif player_features_df[c].notna().sum() == 0:
                skipped.append((c, "100% null in this data"))
            else:
                usable.append(c)
        if skipped:
            if verbose:
                print(f"[props_model:{self.stat_name}] dropping unusable feature column(s): {skipped}")
            self.feature_columns = usable

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
        test_preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, test_preds)
        # Holdout residuals, not training residuals -- same fix as
        # GameMarginModel (src/models/game_model.py), same reason. Measuring
        # spread on rows the GBM already fit made residual_std artificially
        # small, which pushed every over/under probability toward 0 or 1 and
        # produced the absurd +80-126% "EV" props seen in the live tracker
        # (9/2026). residual_std drives over_probability -> model_ev -> the
        # official-play bar, so this inflated how many props qualified too.
        residuals = y_test - test_preds
        self.residual_std = float(np.std(residuals))
        self._fit_residual_bins(test_preds, residuals)

        if verbose:
            bins_txt = ", ".join(f"<={hi:.0f}: sd {sd:.1f}" for hi, sd in
                                 zip(self.residual_bin_edges, self.residual_bin_stds)) \
                if self.residual_bin_edges else "n/a"
            print(f"[props_model:{self.stat_name}] holdout MAE: {mae:.2f} | "
                  f"pooled residual std: {self.residual_std:.2f} | by projection size: {bins_txt} | n={len(data)}")
        return {"holdout_mae": mae, "residual_std": self.residual_std, "n_train": len(X_train)}

    # Minimum holdout rows per projection bucket; buckets smaller than this
    # are merged into their neighbor rather than trusted on a tiny sample.
    MIN_ROWS_PER_BIN = 30
    N_BINS = 5

    def _fit_residual_bins(self, preds: np.ndarray, residuals: np.ndarray):
        """Uncertainty that scales with the size of the projection (added
        9/2026). A single pooled residual_std is dominated by backups whose
        stats are ~0 and easy to predict, so starters inherited a range far
        too narrow -- live examples: a QB projected 267 pass yards was
        treated as ~30 yds of spread (real-world is ~60-70), which put every
        official prop at ~23-25% EV. Here holdout predictions are split into
        quantile buckets and residual spread is measured within each, then
        forced non-decreasing (bigger projection -> at least as much spread),
        which also smooths out noisy buckets."""
        self.residual_bin_edges, self.residual_bin_stds, self.residual_bin_samples = [], [], []
        preds = np.asarray(preds, dtype=float)
        residuals = np.asarray(residuals, dtype=float)
        if len(preds) < self.MIN_ROWS_PER_BIN * 2:
            return
        qs = np.quantile(preds, np.linspace(0, 1, self.N_BINS + 1)[1:-1])
        uppers = list(np.unique(qs)) + [np.inf]
        edges, stds, lo, pending = [], [], -np.inf, None
        for hi in uppers:
            mask = (preds > lo) & (preds <= hi)
            if pending is not None:
                mask = mask | pending
            if mask.sum() < self.MIN_ROWS_PER_BIN and hi != np.inf:
                pending, lo = mask, hi
                continue
            if mask.sum() < self.MIN_ROWS_PER_BIN and edges:
                # trailing sliver: fold into the previous bucket
                edges[-1] = hi
                continue
            edges.append(hi)
            stds.append(float(np.std(residuals[mask])))
            pending, lo = None, hi
        stds = list(np.maximum.accumulate(stds))
        self.residual_bin_edges = [float(e) for e in edges]
        self.residual_bin_stds = [float(x) for x in stds]
        # Keep each bucket's actual holdout misses (up to 2,000) so over/under
        # chances come from the model's real track record instead of a bell
        # curve (10/2026). Prop stats are lumpy and lopsided -- lots of 0-catch
        # games, a few huge ones -- which a bell curve misrepresents; the live
        # props backtest showed the bell-curve hit chances didn't track reality.
        idx = np.searchsorted(np.array(self.residual_bin_edges), preds, side="left")
        rng = np.random.default_rng(0)
        samples = []
        for i in range(len(self.residual_bin_edges)):
            r = residuals[idx == i]
            if len(r) > 2000:
                r = rng.choice(r, 2000, replace=False)
            samples.append([float(x) for x in np.sort(r)])
        self.residual_bin_samples = samples

    def std_for_prediction(self, pred: float) -> float:
        """Residual spread to use for a projection of this size; falls back
        to the pooled residual_std for models trained before bins existed."""
        edges = getattr(self, "residual_bin_edges", None) or []
        stds = getattr(self, "residual_bin_stds", None) or []
        for hi, sd in zip(edges, stds):
            if pred <= hi:
                return sd
        return stds[-1] if stds else self.residual_std

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(features_df[self.feature_columns])

    # Stats that can't go below zero (everything except rushing yards, where
    # sacks and losses can make a game total negative).
    NONNEGATIVE = {"pass_yds", "pass_tds", "pass_att", "pass_comp", "pass_int", "rush_tds", "rush_att",
                   "rec_yds", "rec_tds", "receptions"}

    def over_probability(self, pred: float, prop_line: float):
        """Chance the real stat lands over the line: the model's projection
        plus its actual holdout misses for projections this size. Falls back
        to a bell curve for models saved before misses were stored."""
        edges = getattr(self, "residual_bin_edges", None) or []
        samples = getattr(self, "residual_bin_samples", None) or []
        if edges and samples and len(samples) == len(edges):
            i = next((k for k, hi in enumerate(edges) if pred <= hi), len(edges) - 1)
            r = np.asarray(samples[i], dtype=float)
            if len(r) >= 30:
                outcomes = pred + r
                if self.stat_name in self.NONNEGATIVE:
                    outcomes = np.clip(outcomes, 0, None)
                return float(np.mean(outcomes > prop_line))
        sd = self.std_for_prediction(pred)
        return float(1 - norm.cdf(prop_line, loc=pred, scale=sd)) if sd and sd > 0 else None

    def predict_and_compare(self, features_row: pd.Series, prop_line: float) -> dict:
        pred = float(self.model.predict(features_row[self.feature_columns].to_frame().T)[0])
        if self.stat_name in self.NONNEGATIVE:
            pred = max(pred, 0.0)
        confidence = "low" if features_row.get("games_played_prior", 0) < 3 else "normal"
        over_prob = self.over_probability(pred, prop_line)
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
                     "residual_bin_edges": getattr(self, "residual_bin_edges", []),
                     "residual_bin_stds": getattr(self, "residual_bin_stds", []),
                     "residual_bin_samples": getattr(self, "residual_bin_samples", []),
                     "feature_columns": self.feature_columns, "stat_name": self.stat_name}, path)

    @classmethod
    def load(cls, path: str) -> "PlayerStatModel":
        payload = joblib.load(path)
        instance = cls(payload["stat_name"])
        instance.model = payload["model"]
        instance.residual_std = payload["residual_std"]
        instance.residual_bin_edges = payload.get("residual_bin_edges", [])
        instance.residual_bin_stds = payload.get("residual_bin_stds", [])
        instance.residual_bin_samples = payload.get("residual_bin_samples", [])
        instance.feature_columns = payload["feature_columns"]
        return instance
