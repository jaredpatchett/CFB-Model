"""
Matchup labels for receptions props (added 10/2026).

For a player's NEXT game: has the opposing defense been tough, average or
soft against his position group (wide receivers, tight ends, running backs)?

The number behind the label is the one props lab 3 tested
(scripts/props_lab3.py): the catches a defense has allowed to that position
group, against what those same offenses usually produce in their other
games. 1.00 is an average defense; with few games played the number is
pulled toward 1.00. Below TOUGH is a tough matchup, above SOFT a soft one --
the same cut-offs the lab's results were split on.

What the lab found, on receptions lines from three seasons:
  unders with a 10%+ edge against TOUGH defenses   260-164, up every season
  official overs against SOFT defenses             33-25, no better than
                                                   other overs (22-22 in the
                                                   two past seasons)
So a tough defense is real support for an under; a soft defense agreeing
with an over is context, not a proven edge.

Only the LABEL leaves this module (tough / average / soft and the position
group). The repo and the dashboard are public and PFF's numbers are licensed
to the subscriber, so no PFF value and no defense number is written out.
"""
import tempfile

import numpy as np
import pandas as pd

from src.features import pff_features as pf

SHRINK_GAMES = 3                    # same as props lab 3
TOUGH, SOFT = 0.93, 1.07            # same as props lab 3
GROUPS = ("wr", "te", "rb")
GROUP_NAMES = {"wr": "WR", "te": "TE", "rb": "RB"}
KEYS = ["season", "week", "franchise_id"]
METRICS = [f"rec_{g}" for g in GROUPS]


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def pos_group(position) -> pd.Series:
    p = position.astype(str).str.upper().str.strip()
    return pd.Series(np.select([p.str.startswith("WR"), p.str.startswith("TE"), p.isin(["HB", "RB", "FB", "TB"])],
                               list(GROUPS), default="other"), index=position.index)


def opponents(pgames) -> pd.DataFrame:
    """(season, week, franchise_id) -> the franchise it played that week."""
    cols = KEYS + ["opp"]
    if pgames is None or pgames.empty or "home_franchise_id" not in pgames.columns:
        return pd.DataFrame(columns=cols)
    g = pgames.dropna(subset=["home_franchise_id", "away_franchise_id"])
    long = pd.concat([
        g[["season", "week", "home_franchise_id", "away_franchise_id"]].set_axis(cols, axis=1),
        g[["season", "week", "away_franchise_id", "home_franchise_id"]].set_axis(cols, axis=1),
    ], ignore_index=True)
    return long.drop_duplicates(KEYS)


def offense_games(recv, pgames) -> pd.DataFrame:
    """One row per offense per game: catches by position group, and the
    defense they came against."""
    df = recv.copy()
    rec = _num(df, "receptions").fillna(0.0)
    g = pos_group(df["position"]) if "position" in df.columns else pd.Series("other", index=df.index)
    for grp in GROUPS:
        df[f"rec_{grp}"] = np.where(g == grp, rec, 0.0)
    og = df.groupby(KEYS)[METRICS].sum().reset_index()
    return og.merge(opponents(pgames), on=KEYS, how="left")


def defense_numbers_now(og) -> pd.DataFrame:
    """One row per defense: what it has allowed to each position group in
    every game so far this season, against what those offenses average in
    their OTHER games (the national average when an offense has played only
    once), pulled toward 1.00 by SHRINK_GAMES. Columns franchise_id, games,
    rec_wr, rec_te, rec_rb."""
    cols = ["franchise_id", "games"] + METRICS
    prior = og.dropna(subset=["opp"])
    if prior.empty:
        return pd.DataFrame(columns=cols)
    off = prior.groupby("franchise_id")[METRICS].agg(["sum", "count"])
    league = prior[METRICS].mean()
    exp = pd.DataFrame(index=prior.index)
    for m in METRICS:
        s, n = prior["franchise_id"].map(off[(m, "sum")]), prior["franchise_id"].map(off[(m, "count")])
        exp[m] = np.where(n > 1, (s - prior[m]) / (n - 1).replace(0, np.nan), league[m])
    allowed = prior.groupby("opp")[METRICS].sum()
    expected = exp.groupby(prior["opp"]).sum()
    games = prior.groupby("opp").size().reindex(allowed.index)
    out = pd.DataFrame({"franchise_id": allowed.index, "games": games.values})
    for m in METRICS:
        ratio = (allowed[m] / expected[m].replace(0, np.nan)).values
        out[m] = (out["games"] * ratio + SHRINK_GAMES) / (out["games"] + SHRINK_GAMES)
    return out[cols]


def link_players(wide, games, recv, pgames) -> pd.DataFrame:
    """Which PFF row is each CFBD player-game: gameId, athleteId, franchise_id
    and position group. Uses the live matching code as is, by sending a row
    number through it in place of one of its values."""
    recv = recv.reset_index(drop=True)
    prep = pf.prepare_pff_rows(recv, pgames).reset_index(drop=True)
    m, _ = pf.match_pff_to_cfbd(wide, games, prep.assign(pff_adot=np.arange(len(prep), dtype=float)))
    if m.empty:
        return pd.DataFrame(columns=["gameId", "athleteId", "franchise_id", "g"])
    rows = prep.iloc[m["pff_adot"].astype(int).values][KEYS + ["player_id"]].reset_index(drop=True)
    link = pd.concat([m[["gameId", "athleteId"]].reset_index(drop=True), rows], axis=1)
    raw = recv.assign(g=pos_group(recv["position"]) if "position" in recv.columns else "other")
    raw = raw.drop_duplicates(KEYS + ["player_id"])[KEYS + ["player_id", "g"]]
    return link.merge(raw, on=KEYS + ["player_id"], how="left")


def tier(value):
    """'tough' / 'average' / 'soft', or None when there is no number."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return "tough" if value < TOUGH else ("soft" if value > SOFT else "average")


def build_matchups(wide, games, recv, pgames):
    """(position group by athleteId, {school: {group: tier}}, info) from this
    season's CFBD player-game frame and PFF receiving rows."""
    link = link_players(wide, games, recv, pgames)
    known = link[link["g"].isin(GROUPS)]
    pos = known.groupby("athleteId")["g"].agg(lambda s: s.value_counts().index[0]).to_dict()
    # PFF franchise -> CFBD school, by what most of its matched players say.
    teams = link.merge(wide[["gameId", "athleteId", "team"]], on=["gameId", "athleteId"], how="left").dropna(subset=["team", "franchise_id"])
    votes = teams.groupby(["franchise_id", "team"]).size().rename("v").reset_index().sort_values("v", ascending=False)
    school = votes.drop_duplicates("franchise_id").set_index("franchise_id")["team"]
    dn = defense_numbers_now(offense_games(recv, pgames))
    dn["school"] = dn["franchise_id"].map(school)
    dn = dn.dropna(subset=["school"]).drop_duplicates("school")
    defense = {r["school"]: {g: tier(float(r[f"rec_{g}"])) for g in GROUPS} for _, r in dn.iterrows()}
    counts = {t: sum(1 for d in defense.values() for v in d.values() if v == t) for t in ("tough", "average", "soft")}
    info = {"players_with_a_position": len(pos), "defenses": len(defense), "labels": counts}
    return pos, defense, info


def live_matchups(player_stats_long, games, season, recv=None, pgames=None):
    """(position group by athleteId, {school: {group: tier}}, status text) for
    the current season. Pulls PFF itself unless the rows are handed in.
    Never raises: on any failure the lookups are empty and the status says
    why, and props simply carry no matchup label."""
    try:
        from src.features.player_features import pivot_player_game_stats
        if recv is None:
            from src.data import pff_client as pff
            recv, pgames = pff.fetch_receiving_seasons([season], tempfile.mkdtemp(prefix="pff_matchup_"), verbose=False)
        if recv is None or recv.empty:
            return {}, {}, f"no matchup labels -- PFF returned no {season} receiving rows"
        wide = pivot_player_game_stats(player_stats_long, games)
        pos, defense, info = build_matchups(wide, games, recv, pgames)
        if not defense:
            return {}, {}, "no matchup labels -- PFF's schedule did not line up with the box scores"
        c = info["labels"]
        return pos, defense, (f"matchup labels for {info['defenses']} defenses ({c['tough']} tough / {c['average']} average / "
                              f"{c['soft']} soft position matchups), {info['players_with_a_position']} players with a position")
    except Exception as e:
        return {}, {}, f"no matchup labels this run ({type(e).__name__}: {e})"


def label_props(props, player_form, pos, defense, name_key, market="Receptions"):
    """Adds matchup ('tough' / 'average' / 'soft') and matchup_vs ('WR' /
    'TE' / 'RB') to each prop in `market` whose player has a position and
    whose opponent has a number. Returns how many were labelled."""
    n = 0
    for p in props:
        if p.get("market_name") != market:
            continue
        entry = player_form.get(name_key(p.get("player_name"))) or {}
        g = pos.get(entry.get("athlete_id"))
        t = (defense.get(p.get("opponent")) or {}).get(g) if g else None
        if t:
            p["matchup"], p["matchup_vs"] = t, GROUP_NAMES[g]
            n += 1
    return n
