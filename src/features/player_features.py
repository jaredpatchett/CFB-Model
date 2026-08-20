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
import numpy as np
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

    games_cols = ["id", "season", "week"]
    if "homeTeam" in games.columns and "awayTeam" in games.columns:
        games_cols += ["homeTeam", "awayTeam"]
    games_small = games[games_cols].rename(columns={"id": "gameId"})
    wide = wide.merge(games_small, on="gameId", how="left")
    for col in STAT_MAP.values():
        if col not in wide.columns:
            wide[col] = 0.0
        wide[col] = wide[col].fillna(0.0)

    if "homeTeam" in wide.columns and "awayTeam" in wide.columns:
        # Opponent = whichever of the game's two teams ISN'T this player's own
        # team. Needed for opponent-defense-adjusted props (see
        # attach_opponent_defense below) -- a WR's expected receiving yards
        # should account for how tough the opposing pass defense actually is,
        # not just the WR's own volume.
        wide["opponent"] = np.where(wide["team"] == wide["homeTeam"], wide["awayTeam"], wide["homeTeam"])
        wide = wide.drop(columns=["homeTeam", "awayTeam"])

    return wide.sort_values(["athleteId", "season", "week"])


def attach_opponent_defense(wide_stats: pd.DataFrame, adv_stats: pd.DataFrame) -> pd.DataFrame:
    """Adds opp_pass_def_success_rate / opp_rush_def_success_rate to
    wide_stats (output of pivot_player_game_stats, which must include an
    'opponent' column), from adv_stats (cfbd_client.get_advanced_team_stats
    -- the SAME season-level advanced stats already fetched for the game
    model's pace feature, no new API call needed here).

    Source fields (verified against CFBD's own OpenAPI/client schema, not
    guessed): defense.passingPlays.successRate and
    defense.rushingPlays.successRate -- the rate at which the OPPONENT's
    defense allowed a "successful" play (CFBD's own down-and-distance-based
    definition) through the air vs. on the ground, respectively. Lower =
    tougher defense against that game plan.

    Same season-snapshot leakage caveat as SP+/pace elsewhere in this
    codebase: safe for live weekly use, a simplification for backtesting a
    past season against its own end-of-season snapshot (documented, not new).
    A player's game row with no matching opponent-defense row (opponent not
    in adv_stats, or 'opponent' column missing entirely) gets NaN, not a
    fabricated average."""
    if "opponent" not in wide_stats.columns:
        wide_stats = wide_stats.copy()
        wide_stats["opp_pass_def_success_rate"] = np.nan
        wide_stats["opp_rush_def_success_rate"] = np.nan
        return wide_stats

    def_cols_present = (adv_stats is not None and not adv_stats.empty
                         and "defense.passingPlays.successRate" in adv_stats.columns
                         and "defense.rushingPlays.successRate" in adv_stats.columns
                         and "team" in adv_stats.columns and "season" in adv_stats.columns)
    if not def_cols_present:
        wide_stats = wide_stats.copy()
        wide_stats["opp_pass_def_success_rate"] = np.nan
        wide_stats["opp_rush_def_success_rate"] = np.nan
        return wide_stats

    def_df = adv_stats[["team", "season", "defense.passingPlays.successRate",
                         "defense.rushingPlays.successRate"]].rename(columns={
        "team": "opponent", "season": "season",
        "defense.passingPlays.successRate": "opp_pass_def_success_rate",
        "defense.rushingPlays.successRate": "opp_rush_def_success_rate",
    })
    return wide_stats.merge(def_df, on=["opponent", "season"], how="left")


def build_rolling_player_features(wide_stats: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling per-game averages for each tracked stat."""
    df = wide_stats.copy()
    grp = df.groupby(["athleteId", "season"])
    # transform(), not apply() — see team_features.py for why.
    for col in STAT_MAP.values():
        df[f"roll_{col}"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
    df["games_played_prior"] = grp.cumcount()
    return df


# opp_pass_def_success_rate/opp_rush_def_success_rate are already per-row
# static values from attach_opponent_defense (not rolled — they describe the
# UPCOMING opponent, same "included as-is" treatment as games_played_prior).
# A row missing them (attach_opponent_defense not called, or no matching
# opponent-defense data) gets NaN, same as any other feature column —
# PlayerStatModel.fit() (src/models/props_model.py) already handles a
# feature column that's missing entirely or 100% null gracefully.
ROLLING_FEATURE_COLUMNS = ([f"roll_{col}" for col in STAT_MAP.values()]
                            + ["games_played_prior", "opp_pass_def_success_rate", "opp_rush_def_success_rate"])
