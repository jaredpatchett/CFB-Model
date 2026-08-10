"""
Player-level feature engineering for the props model.

Takes the long-format output of cfbd_client.get_player_game_stats (one row per
athlete/stat/game) and:
  1. Pivots it to one row per player-game with columns per stat (rec_yds,
     rush_yds, receptions, pass_yds, pass_tds, etc.)
  2. Attaches season/week from the games table
  3. Builds leakage-safe rolling usage features (shift(1) + expanding mean,
     same pattern as team_features) so a player's Week 5 features only use
     Weeks 1-4 of that season.

Small-sample caveat: early-season rolling averages are based on very few
games (sometimes zero, for a player's first game). Callers should treat
predictions for players with < 2-3 prior games as low-confidence.
"""
import pandas as pd

# Maps CFBD's (category, statType) pairs to a clean column name
STAT_MAP = {
    ("passing", "YDS"): "pass_yds",
    ("passing", "TD"): "pass_tds",
    ("passing", "ATT"): "pass_att",
    ("passing", "COMPLETIONS"): "pass_comp",
    ("passing", "INT"): "pass_int",
    ("rushing", "YDS"): "rush_yds",
    ("rushing", "TD"): "rush_tds",
    ("rushing", "CAR"): "rush_att",
    ("receiving", "YDS"): "rec_yds",
    ("receiving", "TD"): "rec_tds",
    ("receiving", "REC"): "receptions",
}


def pivot_player_game_stats(long_stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Long (athlete, stat) rows -> wide (one row per player per game)."""
    df = long_stats.copy()
    df["stat"] = pd.to_numeric(df["stat"], errors="coerce")
    df["stat_name"] = list(zip(df["category"].str.lower(), df["statType"]))
    df["stat_name"] = df["stat_name"].map(STAT_MAP)
    df = df.dropna(subset=["stat_name"])

    wide = df.pivot_table(
        index=["gameId", "athleteId", "player", "team"],
        columns="stat_name", values="stat", aggfunc="first"
    ).reset_index()

    games_small = games[["id", "season", "week"]].rename(columns={"id": "gameId"})
    wide = wide.merge(games_small, on="gameId", how="left")
    for col in STAT_MAP.values():
        if col not in wide.columns:
            wide[col] = 0.0
        wide[col] = wide[col].fillna(0.0)
    return wide.sort_values(["athleteId", "season", "week"])


def build_rolling_player_features(wide_stats: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling per-game averages for each tracked stat."""
    df = wide_stats.copy()
    grp = df.groupby(["athleteId", "season"])
    # transform(), not apply() — see team_features.py for why.
    for col in STAT_MAP.values():
        df[f"roll_{col}"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
    df["games_played_prior"] = grp.cumcount()
    return df


ROLLING_FEATURE_COLUMNS = [f"roll_{col}" for col in STAT_MAP.values()] + ["games_played_prior"]
