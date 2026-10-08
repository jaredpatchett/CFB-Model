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
import os
import re
import tempfile
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

# The stats whose live models use the PFF inputs (10/2026 props lab: lower
# projection error on 12,000+ unseen 2025 player-games for both).
PFF_STATS = ("receptions", "rec_yds")


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


def _last2_charted(s: pd.Series) -> pd.Series:
    """Average of the player's last two EARLIER games that PFF has charted.
    A game with no PFF value yet is skipped rather than shrinking the window
    to one game."""
    prev = s.shift(1)
    valid = prev.dropna()
    if valid.empty:
        return prev
    return valid.rolling(2, min_periods=1).mean().reindex(prev.index).ffill()


def add_pff_features(wide: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """Adds PFF_FEATURE_COLUMNS to a player-game frame (must have gameId,
    athleteId, season, week). Leakage-safe: shift(1) before every average.

    A player PFF never matched gets the MISSING sentinel and pff_matched=0,
    so a model can still score him from box-score inputs alone. A matched
    player with a box-score line but no PFF row in a game PFF charted ran
    no routes that game. `wide` must also carry `team`."""
    df = wide.merge(matched, on=["gameId", "athleteId"], how="left")
    has = df["pff_routes"].notna()
    ever = has.groupby([df["athleteId"], df["season"]]).transform("any")
    # "No PFF row" only means "ran no routes" if PFF has charted that game.
    # A game it has not charted yet (a Sunday run after a Saturday game)
    # is left out of the averages instead of being counted as zero.
    charted = has.groupby([df["gameId"], df["team"]]).transform("any")
    fill = ever & charted & ~has
    for c in ZERO_FILL_COLS:
        df.loc[fill, c] = 0.0
    df = df.sort_values(["athleteId", "season", "week"])
    grp = df.groupby(["athleteId", "season"])
    for c in GAME_COLS:
        df[f"roll_{c}"] = grp[c].transform(lambda s: s.shift(1).expanding().mean())
    for c in LAST2_COLS:
        df[f"last2_{c}"] = grp[c].transform(_last2_charted)
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


# ------------------------------------------------------------ live use ----
def missing_entry() -> dict:
    """Feature values for a player with no PFF match -- the same values the
    training rows of unmatched players carry."""
    out = {}
    for c in PFF_FEATURE_COLUMNS:
        out[c] = 0.0 if (c.startswith("trend_") or c == "pff_matched") else MISSING
    return out


def current_pff_snapshot(wide: pd.DataFrame, matched: pd.DataFrame) -> dict:
    """athleteId -> PFF feature values for that player's NEXT game, from all
    of his games so far this season. Built by adding one placeholder "next
    game" row per player and running the exact training code over it, so
    live and training values cannot drift apart."""
    base = wide[["gameId", "athleteId", "season", "week", "team"]].copy()
    nxt = base.sort_values(["season", "week"]).groupby("athleteId").tail(1).copy()
    nxt["week"] = nxt["week"] + 1000
    numeric = pd.api.types.is_numeric_dtype(base["gameId"])
    nxt["gameId"] = [-(i + 1) if numeric else f"next_{i}" for i in range(len(nxt))]
    placeholder = set(nxt["gameId"])
    feats = add_pff_features(pd.concat([base, nxt], ignore_index=True), matched)
    feats = feats[feats["gameId"].isin(placeholder)]
    return {aid: {c: float(v) for c, v in zip(PFF_FEATURE_COLUMNS, row)}
            for aid, row in zip(feats["athleteId"], feats[PFF_FEATURE_COLUMNS].values)}


def training_frame(feats: pd.DataFrame, games: pd.DataFrame, seasons: list, out_dir: str = None):
    """The training frame with PFF features added: (frame, match info).
    PFF data is pulled into a temp folder outside the repo; nothing derived
    from it is written to disk here."""
    from src.data import pff_client as pff
    out_dir = out_dir or os.environ.get("PFF_DATA_DIR") or os.path.join(tempfile.gettempdir(), "pff_train")
    recv, pgames = pff.fetch_receiving_seasons(list(seasons), out_dir)
    if recv.empty:
        raise RuntimeError("PFF returned no receiving rows")
    matched, info = match_pff_to_cfbd(feats, games, prepare_pff_rows(recv, pgames))
    info["pff_incomplete"] = pff.fetch_issues() or "no"
    return add_pff_features(feats, matched), info


def live_snapshot(player_stats_long: pd.DataFrame, games: pd.DataFrame, season: int):
    """(athleteId -> PFF features for the next game, match info) for the
    current season. Always pulls fresh -- a new temp folder each call."""
    from src.data import pff_client as pff
    from src.features.player_features import pivot_player_game_stats
    wide = pivot_player_game_stats(player_stats_long, games)
    recv, pgames = pff.fetch_receiving_seasons([season], tempfile.mkdtemp(prefix="pff_live_"), verbose=False)
    if recv.empty:
        raise RuntimeError(f"PFF returned no {season} receiving rows")
    matched, info = match_pff_to_cfbd(wide, games, prepare_pff_rows(recv, pgames))
    if matched.empty:
        raise RuntimeError("no PFF rows could be matched to CFBD players")
    charted = wide.merge(matched[["gameId"]].drop_duplicates().assign(c=1), on="gameId", how="left")
    info["cfbd_games"] = int(wide["gameId"].nunique())
    info["games_charted_by_pff"] = int(charted.loc[charted["c"].notna(), "gameId"].nunique())
    info["pff_incomplete"] = pff.fetch_issues()
    return current_pff_snapshot(wide, matched), info


def apply_live_pff(player_form: dict, prop_models: dict, player_stats_long: pd.DataFrame,
                   games: pd.DataFrame, season: int, models_dir: str, loader=None):
    """Give every player in `player_form` his PFF feature values so the
    receiving models can score him. Returns (prop_models, status text).

    If PFF cannot be reached (outage, expired key), the receiving models are
    swapped for their box-score-only fallbacks (saved next to them as
    <stat>_base.joblib by train_props_model.py) so props are still scored.
    Never raises."""
    needs = [s for s, m in prop_models.items() if any(c in PFF_FEATURE_COLUMNS for c in (m.feature_columns or []))]
    if not needs:
        return prop_models, "box score only (models were trained without PFF)"
    try:
        snap, info = live_snapshot(player_stats_long, games, season)
        for entry in player_form.values():
            entry.update(snap.get(entry.get("athlete_id")) or missing_entry())
        n = sum(1 for e in player_form.values() if e.get("pff_matched") == 1.0)
        status = (f"PFF routes and targets ({n} of {len(player_form)} players matched, "
                  f"{info.get('games_charted_by_pff')} of {info.get('cfbd_games')} games charted)")
        if info.get("pff_incomplete"):
            # PFF failed on some weeks even after retries: say so, so a short
            # games-charted count is never mistaken for a normal one.
            status += f" -- INCOMPLETE, PFF did not answer: {info['pff_incomplete']}"
        return prop_models, status
    except Exception as e:
        out = dict(prop_models)
        if loader is None:
            from src.models.props_model import PlayerStatModel
            loader = PlayerStatModel.load
        for s in needs:
            path = f"{models_dir}/{s}_base.joblib"
            try:
                out[s] = loader(path)
            except Exception:
                out.pop(s, None)
        return out, f"box score only -- PFF unavailable this run ({type(e).__name__}: {e})"
