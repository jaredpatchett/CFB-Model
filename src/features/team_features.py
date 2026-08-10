"""
Team-level feature engineering for the spread/moneyline model.

Design choice on leakage:
- SP+ ratings (from CFBD /ratings/sp) are a season-level signal. When used for
  LIVE weekly predictions this is safe (today's rating only reflects games
  already played). When used for BACKTESTING past seasons, pulling one
  end-of-season SP+ snapshot and applying it to Week 1 games is leakage —
  the rating "knows" about games that hadn't happened yet. The backtester
  (src/backtest/backtester.py) flags this; for rigorous backtesting, prefer
  the rolling in-season features below over SP+, or re-pull SP+ per week if
  your CFBD plan/endpoint supports it.
- Rolling scoring margin/PPG features are computed with an expanding window
  shifted by one game, so a team's Week 5 features only use Weeks 1-4. This
  part is leakage-safe regardless of season stage.
"""
import pandas as pd
import numpy as np


def _rolling_team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (team, season, week) with that team's average
    points scored/allowed and margin from ONLY prior games in the season."""
    long_rows = []
    for _, g in games.iterrows():
        long_rows.append({
            "season": g["season"], "week": g["week"], "team": g["homeTeam"],
            "points_for": g["homePoints"], "points_against": g["awayPoints"],
        })
        long_rows.append({
            "season": g["season"], "week": g["week"], "team": g["awayTeam"],
            "points_for": g["awayPoints"], "points_against": g["homePoints"],
        })
    long_df = pd.DataFrame(long_rows).dropna(subset=["points_for", "points_against"])
    long_df = long_df.sort_values(["team", "season", "week"])

    long_df["margin"] = long_df["points_for"] - long_df["points_against"]
    grp = long_df.groupby(["team", "season"])
    # shift(1) before the expanding mean so the current game is excluded
    # .transform() (not .apply()) is required here: apply() with a lambda that
    # returns a same-length Series can come back with a MultiIndex on some
    # pandas versions and fail to align back into long_df. transform() is
    # the correct API for "same shape in, same shape out, per group."
    long_df["roll_ppg_for"] = grp["points_for"].transform(lambda s: s.shift(1).expanding().mean())
    long_df["roll_ppg_against"] = grp["points_against"].transform(lambda s: s.shift(1).expanding().mean())
    long_df["roll_margin"] = grp["margin"].transform(lambda s: s.shift(1).expanding().mean())
    long_df["games_played_prior"] = grp.cumcount()
    return long_df[["team", "season", "week", "roll_ppg_for", "roll_ppg_against",
                     "roll_margin", "games_played_prior"]]


def build_game_team_features(games: pd.DataFrame, sp_ratings: pd.DataFrame = None) -> pd.DataFrame:
    """
    games: output of cfbd_client.get_games() for one or more seasons.
    sp_ratings: output of cfbd_client.get_sp_ratings() (optional, adds SP+ prior).

    Returns one row per completed game with home/away features and targets
    (margin, home_win) ready for modeling.
    """
    games = games.copy()
    games = games[games["completed"] == True] if "completed" in games.columns else games
    games = games.dropna(subset=["homePoints", "awayPoints"])

    form = _rolling_team_form(games)

    df = games.merge(
        form.rename(columns={c: f"home_{c}" for c in form.columns if c not in ("team", "season", "week")}),
        left_on=["homeTeam", "season", "week"], right_on=["team", "season", "week"], how="left"
    ).drop(columns=["team"])
    df = df.merge(
        form.rename(columns={c: f"away_{c}" for c in form.columns if c not in ("team", "season", "week")}),
        left_on=["awayTeam", "season", "week"], right_on=["team", "season", "week"], how="left"
    ).drop(columns=["team"])

    if sp_ratings is not None and not sp_ratings.empty:
        sp = sp_ratings.copy()
        sp_cols = {"team": "team", "rating": "sp_rating"}
        if "offense.rating" in sp.columns:
            sp_cols["offense.rating"] = "sp_off_rating"
        if "defense.rating" in sp.columns:
            sp_cols["defense.rating"] = "sp_def_rating"
        sp = sp[list(sp_cols.keys()) + ["year"]].rename(columns=sp_cols)

        df = df.merge(sp.add_prefix("home_"), left_on=["homeTeam", "season"],
                       right_on=["home_team", "home_year"], how="left")
        df = df.merge(sp.add_prefix("away_"), left_on=["awayTeam", "season"],
                       right_on=["away_team", "away_year"], how="left")
        df["sp_rating_diff"] = df.get("home_sp_rating") - df.get("away_sp_rating")

    df["home_field"] = np.where(df.get("neutralSite", False) == True, 0, 1)
    df["margin"] = df["homePoints"] - df["awayPoints"]
    df["home_win"] = (df["margin"] > 0).astype(int)
    df["roll_margin_diff"] = df.get("home_roll_margin") - df.get("away_roll_margin")
    df["roll_ppg_for_diff"] = df.get("home_roll_ppg_for") - df.get("away_roll_ppg_for")
    df["roll_ppg_against_diff"] = df.get("home_roll_ppg_against") - df.get("away_roll_ppg_against")

    return df


FEATURE_COLUMNS = [
    "home_field", "roll_margin_diff", "roll_ppg_for_diff", "roll_ppg_against_diff",
    "sp_rating_diff", "home_games_played_prior", "away_games_played_prior",
]
TARGET_MARGIN = "margin"
TARGET_WIN = "home_win"
