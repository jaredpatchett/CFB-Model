"""
Run-defense toughness (11/2026), for the rush yards unders rule.

For each defense, from that season's games BEFORE a given week only: rush
yards it allowed against what those same offenses usually gain in their
other games, shrunk toward average (3 games), then ranked. 100 = allows the
fewest. Tough = top third (67+), soft = bottom third (33 or less). Needs 3+
games; with fewer there is no grade. Same method as the Oct 30 matchup lab;
backtested on 2024-26 rush yds lines in scripts/rush_lab.py.
"""
import numpy as np
import pandas as pd

SHRINK_YDS, MIN_DEF_GAMES = 3, 3
TOUGH, SOFT = 67, 33


def run_defense(wide: pd.DataFrame) -> pd.DataFrame:
    """(season, week, opponent) -> run_def_tough for the defense `opponent`
    facing a game in that week. Includes week = last played week + 1, so the
    latest row per defense is its grade for its next game."""
    og = wide.groupby(["season", "week", "gameId", "team", "opponent"])["rush_yds"].sum().reset_index()
    og["rush_yds"] = pd.to_numeric(og["rush_yds"], errors="coerce").fillna(0.0)
    og = og.dropna(subset=["season", "week"])
    rows = []
    for (s, wk) in sorted(set(zip(og["season"], og["week"])) | {(s, w + 1) for s, w in zip(og["season"], og["week"])}):
        prior = og[(og["season"] == s) & (og["week"] < wk)]
        if prior.empty:
            continue
        off = prior.groupby("team")["rush_yds"].agg(["sum", "count"])
        league = prior["rush_yds"].mean()
        s_, n_ = prior["team"].map(off["sum"]), prior["team"].map(off["count"])
        exp = np.where(n_ > 1, (s_ - prior["rush_yds"]) / (n_ - 1).replace(0, np.nan), league)
        t = pd.DataFrame({"games": prior.groupby("opponent").size(),
                          "allowed": prior.groupby("opponent")["rush_yds"].sum(),
                          "expected": pd.Series(exp, index=prior.index).groupby(prior["opponent"]).sum()})
        t = t[t["games"] >= MIN_DEF_GAMES]
        if len(t) < 20:
            continue
        r = (t["allowed"] / t["expected"].replace(0, np.nan)).fillna(1.0)
        ratio = (t["games"] * r + SHRINK_YDS) / (t["games"] + SHRINK_YDS)
        tough = (-ratio).rank(pct=True) * 100
        rows.append(pd.DataFrame({"season": s, "week": wk, "opponent": tough.index, "run_def_tough": tough.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["season", "week", "opponent", "run_def_tough"])


def current_grades(wide: pd.DataFrame) -> dict:
    """defense -> toughness for its NEXT game (all of this season's games)."""
    rd = run_defense(wide)
    if rd.empty:
        return {}
    last = rd.sort_values(["season", "week"]).groupby("opponent").tail(1)
    return dict(zip(last["opponent"], last["run_def_tough"]))


def bucket(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "no grade"
    return "tough" if x >= TOUGH else ("soft" if x <= SOFT else "average")
