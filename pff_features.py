"""
PFF route and target features for the props model (added 10/2026).

PFF's receiving report has what the box score lacks: how many pass routes a
player ran and how many times he was thrown to. Catches are the result;
routes and targets are the opportunity. This module

  1. matches each PFF player-week row to the CFBD player-game row for the
     same player and game (the two sources share no player or game ids), and
  2. builds leakage-safe features from them -- every value for a game uses
     only that player's EARLIER games of the same season, exactly like the
     box-score features in player_features.py.

Nothing here writes PFF numbers to disk. The repo is public and PFF data is
licensed to the subscriber only.
"""
import re
import unicodedata

import numpy as np
import pandas as pd

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
MATCH_WINDOW = pd.Timedelta(hours=48)   # kickoff times in the two sources can differ a little
MISSING = -1.0                          # sentinel for "no PFF data for this player"

# Per-game values taken from PFF, then averaged over a player's earlier games.
GAME_COLS = ["pff_routes", "pff_targets", "pff_route_part", "pff_target_share", "pff_adot", "pff_slot_rate"]
ZERO_FILL_COLS = ["pff_routes", "pff_targets", "pff_route_part", "pff_target_share"]
LAST2_COLS = ["pff_routes", "pff_targets", "pff_route_part", "pff_target_share"]

PFF_FEATURE_COLUMNS = (
    [f"roll_{c}" for c in GAME_COLS]
    + [f"last2_{c}" for c in LAST2_COLS]
    + ["roll_pff_tprr", "trend_pff_route_part", "trend_pff_target_share", "pff_matched"]
)


def norm_name(name) -> str:
    """Lowercase, no accents or punctuation, no Jr./III-style suffix."""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).split()
    while len(tokens) > 1 and tokens[-1] in SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def prepare_pff_rows(pff_rows: pd.DataFrame, pff_games: pd.DataFrame) -> pd.DataFrame:
    """One clean row per PFF player-week: the per-game values the features
    are built from, plus the kickoff time of that team's game that week."""
    df = pff_rows.copy()
    df["n"] = df["player"].map(norm_name)
    df["pff_routes"] = _num(df, "routes").fillna(0.0)
    df["pff_targets"] = _num(df, "targets").fillna(0.0)
    df["pff_adot"] = _num(df, "avg_depth_of_target")
    df["pff_slot_rate"] = _num(df, "slot_rate")
    on_field = _num(df, "pass_plays").fillna(0.0)
    # Team pass plays that game ~ the most pass plays any one of its
    # receivers was on the field for; team targets = everyone's targets.
    grp = df.groupby(["season", "week", "franchise_id"])
    team_pass_plays = np.maximum(grp["pff_routes"].transform("max"), on_field.groupby(
        [df["season"], df["week"], df["franchise_id"]]).transform("max"))
    team_targets = grp["pff_targets"].transform("sum")
    df["pff_route_part"] = np.where(team_pass_plays > 0, df["pff_routes"] / team_pass_plays.replace(0, np.nan), 0.0)
    df["pff_target_share"] = np.where(team_targets > 0, df["pff_targets"] / team_targets.replace(0, np.nan), 0.0)

    df["kick"] = pd.NaT
    if pff_games is not None and not pff_games.empty and "start" in pff_games.columns:
        g = pff_games.copy()
        g["kick"] = pd.to_datetime(g["start"], utc=True, errors="coerce")
        long = pd.concat([
            g[["season", "week", "home_franchise_id", "kick"]].rename(columns={"home_franchise_id": "franchise_id"}),
            g[["season", "week", "away_franchise_id", "kick"]].rename(columns={"away_franchise_id": "franchise_id"}),
        ]).dropna(subset=["franchise_id"]).drop_duplicates(["season", "week", "franchise_id"])
        df = df.drop(columns=["kick"]).merge(long, on=["season", "week", "franchise_id"], how="left")
    df["kick"] = pd.to_datetime(df["kick"], utc=True, errors="coerce")
    keep = ["season", "week", "player_id", "franchise_id", "n", "kick"] + GAME_COLS
    return df[keep]


def _same_game(pairs: pd.DataFrame) -> pd.Series:
    """True where a PFF row and a CFBD row are the same game: kickoffs within
    48 hours when both are known, otherwise the same week number (PFF's
    Week 0 is CFBD's week 1)."""
    both = pairs["kick_pff"].notna() & pairs["kick_cfbd"].notna()
    by_time = both & ((pairs["kick_pff"] - pairs["kick_cfbd"]).abs() <= MATCH_WINDOW)
    pff_week = pairs["week_pff"].where(pairs["week_pff"] > 0, 1)
    by_week = ~both & (pff_week == pairs["week_cfbd"])
    return by_time | by_week


def match_pff_to_cfbd(wide: pd.DataFrame, games: pd.DataFrame, pff: pd.DataFrame):
    """Returns (matched, info). `matched` has one row per CFBD player-game
    that found its PFF row: gameId, athleteId and the pff_* game values.

    Step 1 learns which CFBD school each PFF franchise id is, from players
    whose names are unique in both sources that season. Step 2 matches on
    season + school + name + same game. Step 3 catches leftovers whose name
    is unique in both sources (school unknown or spelled differently)."""
    cf = wide[["season", "week", "gameId", "athleteId", "player", "team"]].copy()
    cf["n"] = cf["player"].map(norm_name)
    starts = games.drop_duplicates("id").set_index("id")["startDate"] if "startDate" in games.columns else pd.Series(dtype=str)
    cf["kick"] = pd.to_datetime(cf["gameId"].map(starts), utc=True, errors="coerce")
    cf = cf.rename(columns={"week": "week_cfbd", "kick": "kick_cfbd"})
    pf = pff.rename(columns={"week": "week_pff", "kick": "kick_pff"}).reset_index(drop=True)
    pf["pff_row"] = np.arange(len(pf))

    cf_unique = cf.groupby(["season", "n"])["athleteId"].transform("nunique") == 1
    pf_unique = pf.groupby(["season", "n"])["player_id"].transform("nunique") == 1

    # Step 1: franchise id -> school, by majority vote of unique-name players.
    votes = pf[pf_unique].merge(cf[cf_unique], on=["season", "n"])
    votes = votes[_same_game(votes)]
    tally = votes.groupby(["season", "franchise_id", "team"]).size().rename("votes").reset_index()
    tally["share"] = tally["votes"] / tally.groupby(["season", "franchise_id"])["votes"].transform("sum")
    best = tally.sort_values("votes", ascending=False).drop_duplicates(["season", "franchise_id"])
    best = best[(best["votes"] >= 3) & (best["share"] >= 0.6)]
    school = best.set_index(["season", "franchise_id"])["team"]
    pf["team"] = [school.get((s, f)) for s, f in zip(pf["season"], pf["franchise_id"])]

    # Step 2: season + school + name + same game.
    a = pf.dropna(subset=["team"]).merge(cf, on=["season", "team", "n"])
    a = a[_same_game(a)]
    # Step 3: unique-name fallback for PFF rows still unmatched.
    left = pf[pf_unique & ~pf["pff_row"].isin(a["pff_row"])].drop(columns=["team"])
    b = left.merge(cf[cf_unique], on=["season", "n"])
    b = b[_same_game(b)]

    m = pd.concat([a, b], ignore_index=True)
    if m.empty:
        return pd.DataFrame(columns=["gameId", "athleteId"] + GAME_COLS), {"matched_rows": 0}
    # One PFF row per CFBD player-game and vice versa (closest kickoff, then most routes).
    m["gap"] = (m["kick_pff"] - m["kick_cfbd"]).abs().dt.total_seconds().fillna(0)
    m = m.sort_values(["gap", "pff_routes"], ascending=[True, False])
    m = m.drop_duplicates("pff_row").drop_duplicates(["gameId", "athleteId"])
    info = {
        "pff_rows": int(len(pf)), "matched_rows": int(len(m)),
        "pct_pff_rows_matched": round(len(m) / max(len(pf), 1) * 100, 1),
        "franchises_mapped_to_school": int(len(best)),
        "pct_kickoff_known": round(float(pf["kick_pff"].notna().mean()) * 100, 1),
    }
    return m[["gameId", "athleteId"] + GAME_COLS], info


def add_pff_features(wide: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Adds PFF_FEATURE_COLUMNS to a player-game frame (must have gameId,
    athleteId, season, week). Leakage-safe: shift(1) before every average.

    A player PFF never matched gets the MISSING sentinel and pff_matched=0,
    so a model can still score him from box-score inputs alone. A matched
    player with a box-score line but no PFF row that week ran no routes."""
    df = wide.merge(matched, on=["gameId", "athleteId"], how="left")
    ever = df.groupby(["athleteId", "season"])["pff_routes"].transform(lambda s: s.notna().any())
    for c in ZERO_FILL_COLS:
        df[c] = df[c].where(~(ever & df[c].isna()), 0.0)
    df = df.sort_values(["athleteId", "season", "week"])
    grp = df.groupby(["athleteId", "season"])
    for c in GAME_COLS:
        df[f"roll_{c}"] = grp[c].transform(lambda s: s.shift(1).expanding().mean())
    for c in LAST2_COLS:
        df[f"last2_{c}"] = grp[c].transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    # Targets per route run, from season-to-date totals (steadier than a mean of per-game ratios).
    cum_t = grp["pff_targets"].transform(lambda s: s.shift(1).expanding().sum())
    cum_r = grp["pff_routes"].transform(lambda s: s.shift(1).expanding().sum())
    df["roll_pff_tprr"] = np.where(cum_r > 0, cum_t / cum_r.replace(0, np.nan), np.where(cum_r.notna(), 0.0, np.nan))
    df["trend_pff_route_part"] = df["last2_pff_route_part"] - df["roll_pff_route_part"]
    df["trend_pff_target_share"] = df["last2_pff_target_share"] - df["roll_pff_target_share"]
    df["pff_matched"] = df["roll_pff_routes"].notna().astype(float)
    for c in PFF_FEATURE_COLUMNS:
        if c.startswith("trend_"):
            df[c] = df[c].fillna(0.0)
        elif c != "pff_matched":
            df[c] = df[c].fillna(MISSING)
    return df.drop(columns=GAME_COLS)
