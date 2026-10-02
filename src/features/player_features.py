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


def _expand_c_att(df: pd.DataFrame) -> pd.DataFrame:
    """CFBD reports a passer's completions and attempts as ONE combined value
    (statType 'C/ATT', stat '20/31'). Nothing split it before 10/2026, so
    pd.to_numeric turned it into NaN and every pass_comp/pass_att was 0 --
    in training AND live scoring (the props backtest caught it: the model
    projected 0 attempts for every QB and 'won' every under). Split it into
    the two rows STAT_MAP already expects."""
    if "statType" not in df.columns or "category" not in df.columns:
        return df
    m = (df["category"].astype(str).str.lower() == "passing") & (df["statType"].astype(str).str.upper() == "C/ATT")
    if not m.any():
        return df
    parts = df.loc[m, "stat"].astype(str).str.split("/", n=1, expand=True)
    comp = df[m].copy(); comp["statType"] = "COMPLETIONS"; comp["stat"] = parts[0]
    att = df[m].copy(); att["statType"] = "ATT"; att["stat"] = parts[1] if parts.shape[1] > 1 else None
    return pd.concat([df[~m], comp, att], ignore_index=True)


def pivot_player_game_stats(long_stats: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Long (athlete, stat) rows -> wide (one row per player per game)."""
    df = _expand_c_att(long_stats.copy())
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


# ---- Volume / role / game-environment features (added 10/2026) ----------
# The original props model only saw each player's own per-game averages and
# a season-level opponent number. A live backtest on ~1,800 real posted
# props (Sep 2026) showed it had no edge and its hit chances didn't track
# reality. Props are driven by VOLUME (how many plays the team runs, how
# they split between run and pass, what share goes to this player) and by
# GAME SCRIPT (a big favorite runs the ball late; a shootout means more
# passing). These features give the model that information.
SHARE_COLS = ["carry_share", "rec_share", "rec_yds_share", "pass_share"]
TEAM_VOLUME_COLS = ["team_pass_att", "team_rush_att"]
LAST2_COLS = ["pass_yds", "pass_att", "rush_yds", "rush_att", "rec_yds", "receptions", "carry_share", "rec_share"]
OPP_ALLOWED_COLS = ["opp_pass_yds_allowed_prior", "opp_rush_yds_allowed_prior"]
GAME_ENV_COLS = ["team_spread", "team_implied_total"]


def _safe_div(a, b):
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def add_usage_features(wide: pd.DataFrame) -> pd.DataFrame:
    """Per-game team totals and this player's share of them, plus how many
    passing/rushing yards each player's OPPONENT had allowed per game in its
    games BEFORE this one (leakage-safe, computed from box scores -- replaces
    the season-end opponent success rates for training, which included games
    that hadn't happened yet)."""
    df = wide.copy()
    g = df.groupby(["gameId", "team"])
    for col in ("pass_att", "rush_att", "receptions", "rec_yds", "pass_yds", "rush_yds"):
        df[f"team_{col}"] = g[col].transform("sum")
    df["carry_share"] = _safe_div(df["rush_att"], df["team_rush_att"])
    df["rec_share"] = _safe_div(df["receptions"], df["team_receptions"])
    df["rec_yds_share"] = _safe_div(df["rec_yds"], df["team_rec_yds"])
    df["pass_share"] = _safe_div(df["pass_att"], df["team_pass_att"])

    if "opponent" in df.columns and "season" in df.columns and "week" in df.columns:
        tg = df.groupby(["gameId", "team"]).agg(
            season=("season", "first"), week=("week", "first"), opponent=("opponent", "first"),
            pass_yds=("team_pass_yds", "first"), rush_yds=("team_rush_yds", "first")).reset_index()
        # What each team ALLOWED in a game = its opponent's offensive totals in that game.
        allowed = tg.merge(tg[["gameId", "team", "pass_yds", "rush_yds"]].rename(
            columns={"team": "opponent", "pass_yds": "allowed_pass", "rush_yds": "allowed_rush"}),
            on=["gameId", "opponent"], how="left")
        allowed = allowed.sort_values(["team", "season", "week"])
        ag = allowed.groupby(["team", "season"])
        allowed["opp_pass_yds_allowed_prior"] = ag["allowed_pass"].transform(lambda s: s.shift(1).expanding().mean())
        allowed["opp_rush_yds_allowed_prior"] = ag["allowed_rush"].transform(lambda s: s.shift(1).expanding().mean())
        df = df.merge(allowed[["gameId", "team"] + OPP_ALLOWED_COLS].rename(columns={"team": "opponent"}),
                      on=["gameId", "opponent"], how="left")
    else:
        for c in OPP_ALLOWED_COLS:
            df[c] = np.nan
    return df


def attach_game_environment(wide: pd.DataFrame, lines: pd.DataFrame) -> pd.DataFrame:
    """team_spread (negative = this player's team favored) and
    team_implied_total (points the market expects this team to score) from
    each game's betting spread and total -- known before kickoff, and the
    single best summary of game script available."""
    df = wide.copy()
    need = {"id", "homeTeam", "awayTeam", "market_spread_home", "market_over_under"}
    if lines is None or lines.empty or not need.issubset(lines.columns):
        for c in GAME_ENV_COLS:
            df[c] = np.nan
        return df
    ln = lines[list(need)].drop_duplicates("id").rename(columns={"id": "gameId"})
    df = df.merge(ln, on="gameId", how="left")
    home = df["team"] == df["homeTeam"]
    sp = pd.to_numeric(df["market_spread_home"], errors="coerce")
    tot = pd.to_numeric(df["market_over_under"], errors="coerce")
    df["team_spread"] = np.where(home, sp, -sp)
    df["team_implied_total"] = tot / 2 - df["team_spread"] / 2
    return df.drop(columns=["homeTeam", "awayTeam", "market_spread_home", "market_over_under"])


def build_rolling_player_features(wide_stats: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-safe rolling per-game averages for each tracked stat,
    plus (when add_usage_features has run) usage shares, team volume, and
    last-2-game form so role changes show up quickly."""
    df = wide_stats.sort_values(["athleteId", "season", "week"]).copy()
    grp = df.groupby(["athleteId", "season"])
    # transform(), not apply() — see team_features.py for why.
    for col in list(STAT_MAP.values()) + [c for c in SHARE_COLS + TEAM_VOLUME_COLS if c in df.columns]:
        df[f"roll_{col}"] = grp[col].transform(lambda s: s.shift(1).expanding().mean())
    for col in [c for c in LAST2_COLS if c in df.columns]:
        df[f"last2_{col}"] = grp[col].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    df["games_played_prior"] = grp.cumcount()
    return df


def current_feature_snapshot(wide_with_usage: pd.DataFrame) -> dict:
    """athleteId -> the same feature values build_rolling_player_features
    would give that player's NEXT game: averages over all games so far,
    last-2-game form, games played. Keeps live scoring identical to training."""
    out = {}
    for aid, grp in wide_with_usage.sort_values(["season", "week"]).groupby("athleteId"):
        e = {"games_played_prior": int(len(grp))}
        for col in list(STAT_MAP.values()) + [c for c in SHARE_COLS + TEAM_VOLUME_COLS if c in grp.columns]:
            e[f"roll_{col}"] = float(grp[col].mean())
        for col in [c for c in LAST2_COLS if c in grp.columns]:
            e[f"last2_{col}"] = float(grp[col].tail(2).mean())
        out[aid] = e
    return out


def current_defense_allowed(wide_with_usage: pd.DataFrame) -> dict:
    """team -> passing/rushing yards allowed per game so far this season
    (live counterpart of the *_allowed_prior training features)."""
    if "opponent" not in wide_with_usage.columns:
        return {}
    tg = wide_with_usage.groupby(["gameId", "team"]).agg(
        opponent=("opponent", "first"), pass_yds=("team_pass_yds", "first"), rush_yds=("team_rush_yds", "first")).reset_index()
    allowed = tg.groupby("opponent").agg(opp_pass_yds_allowed_prior=("pass_yds", "mean"),
                                         opp_rush_yds_allowed_prior=("rush_yds", "mean"))
    return allowed.to_dict(orient="index")


# opp_pass_def_success_rate/opp_rush_def_success_rate are already per-row
# static values from attach_opponent_defense (not rolled — they describe the
# UPCOMING opponent, same "included as-is" treatment as games_played_prior).
# A row missing them (attach_opponent_defense not called, or no matching
# opponent-defense data) gets NaN, same as any other feature column —
# PlayerStatModel.fit() (src/models/props_model.py) already handles a
# feature column that's missing entirely or 100% null gracefully.
ROLLING_FEATURE_COLUMNS = ([f"roll_{col}" for col in STAT_MAP.values()]
                            + [f"roll_{c}" for c in SHARE_COLS + TEAM_VOLUME_COLS]
                            + [f"last2_{c}" for c in LAST2_COLS]
                            + ["games_played_prior"] + OPP_ALLOWED_COLS + GAME_ENV_COLS)
# opp_pass_def_success_rate/opp_rush_def_success_rate are no longer model
# inputs (10/2026): in training they came from season-END stats, i.e. games
# that hadn't happened yet. OPP_ALLOWED_COLS replace them, leakage-free.
