"""
Game totals model (added 11/2026).

Predicts total points (home + away) from per-team inputs -- each side's
scoring form, plays per drive (prior season), recent per-play efficiency on
offense and defense, CORE offense/defense (prior season), and the weather.
Every input is known before kickoff (same leakage rules as the spread
model's features; see src/features/team_features.py).

Backtested walk-forward in scripts/totals_lab.py (docs: new-features-
2026-11-17). Used live by src/analysis/totals_live.py.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

BASE_INPUTS = [
    "home_roll_ppg_for", "home_roll_ppg_against", "away_roll_ppg_for", "away_roll_ppg_against",
    "home_pace", "away_pace",
    "home_ewm_off_ppa", "home_ewm_def_ppa", "away_ewm_off_ppa", "away_ewm_def_ppa",
    "home_ewm_off_sr", "home_ewm_def_sr", "away_ewm_off_sr", "away_ewm_def_sr",
    "home_core_offense", "home_core_defense", "away_core_offense", "away_core_defense",
    "home_games_played_prior", "away_games_played_prior",
    "temperature", "wind_speed", "precipitation", "game_indoors",
    "home_field", "home_qb_out", "away_qb_out",
]
DERIVED = ["pts_formula", "pace_sum", "ppa_sum", "sr_sum"]


def add_inputs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    g = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    # Expected points: each offense's points for averaged with the other
    # defense's points allowed, summed over both teams.
    d["pts_formula"] = ((g("home_roll_ppg_for") + g("away_roll_ppg_against")) / 2
                        + (g("away_roll_ppg_for") + g("home_roll_ppg_against")) / 2)
    d["pace_sum"] = g("home_pace") + g("away_pace")
    # def PPA = points added per play ALLOWED, so higher = worse defense = more points.
    d["ppa_sum"] = g("home_ewm_off_ppa") + g("away_ewm_off_ppa") + g("home_ewm_def_ppa") + g("away_ewm_def_ppa")
    d["sr_sum"] = g("home_ewm_off_sr") + g("away_ewm_off_sr") + g("home_ewm_def_sr") + g("away_ewm_def_sr")
    if "game_indoors" in d.columns:
        d["game_indoors"] = d["game_indoors"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(
            pd.to_numeric(d["game_indoors"], errors="coerce"))
    return d


def usable(cols, df):
    return [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]


class TotalsModel:
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                               min_samples_leaf=20, random_state=42)
        self.feature_columns = []
        self.medians = {}
        self.residual_std = None
        self.n_train = 0

    def fit(self, df: pd.DataFrame, target: str = "total"):
        d = add_inputs(df)
        d = d[pd.to_numeric(d[target], errors="coerce").notna()]
        if "home_games_played_prior" in d.columns:
            d = d[d["home_games_played_prior"].fillna(0) + d["away_games_played_prior"].fillna(0) > 0]
        self.feature_columns = usable(BASE_INPUTS + DERIVED, d)
        X = d[self.feature_columns].apply(pd.to_numeric, errors="coerce")
        self.medians = {c: float(v) for c, v in X.median().items()}
        X = X.fillna(self.medians)
        y = d[target].astype(float)
        self.model.fit(X, y)
        self.residual_std = float(np.std(y - self.model.predict(X)))
        self.n_train = int(len(d))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = add_inputs(df)
        for c in self.feature_columns:
            if c not in d.columns:
                d[c] = np.nan
        X = d[self.feature_columns].apply(pd.to_numeric, errors="coerce").fillna(self.medians)
        return self.model.predict(X)

    def importances(self, top: int = 12) -> dict:
        pairs = sorted(zip(self.feature_columns, self.model.feature_importances_), key=lambda kv: -kv[1])
        return {k: round(float(v), 4) for k, v in pairs[:top]}

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TotalsModel":
        return joblib.load(path)
