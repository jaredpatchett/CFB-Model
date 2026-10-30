"""
Defense vs position (added 10/30/2026, display only).

For every FBS defense this season, what it has given up to QBs, RBs, WRs
and TEs: rushing + receiving TDs, catches, receiving yards, rush yards --
per game, and against what those same offenses usually get in their other
games ("vs usual": 1.00 = average, 1.25 = 25% more than usual; pulled toward
1.00 with few games, 3 for yardage and 6 for TDs, same as scripts/matchup_lab.py).
Also each defense's game log, the most recent TDs it allowed and its next
opponent. Positions come from CFBD's roster; a player missing from it gets a
guess from his usage (passes thrown -> QB, more carries than catches -> RB,
otherwise WR). Passing TDs are not counted (they are the receiver's TD).

Feeds the dashboard's "TDs Allowed" tab only; nothing here touches the model
or the official plays.
"""
import numpy as np
import pandas as pd

GROUPS = ("qb", "rb", "wr", "te")
STATS = ("td", "rec", "recyds", "rushyds")
SHRINK = {"td": 6, "rec": 3, "recyds": 3, "rushyds": 3}


def _group(p):
    p = str(p).upper().strip()
    if p.startswith("QB"):
        return "qb"
    if p in ("RB", "HB", "FB", "TB", "APB"):
        return "rb"
    if p.startswith("WR"):
        return "wr"
    if p.startswith("TE"):
        return "te"
    return None


def _guess(r):
    if r["pass_att"] >= 5:
        return "qb"
    if r["rush_att"] >= max(3, r["receptions"]):
        return "rb"
    return "wr" if r["receptions"] > 0 else None


def _vs_usual(og, cols):
    """Per defense: allowed vs what those offenses get in their other games."""
    off = og.groupby("team")[cols].agg(["sum", "count"])
    league = og[cols].mean()
    exp = pd.DataFrame(index=og.index)
    for c in cols:
        s, n = og["team"].map(off[(c, "sum")]), og["team"].map(off[(c, "count")])
        exp[c] = np.where(n > 1, (s - og[c]) / (n - 1).replace(0, np.nan), league[c])
    allowed = og.groupby("opponent")[cols].sum()
    expected = exp.groupby(og["opponent"]).sum()
    games = og.groupby("opponent").size()
    out = pd.DataFrame(index=allowed.index)
    for c in cols:
        k = SHRINK[c.split("_")[0]]
        r = (allowed[c] / expected[c].replace(0, np.nan)).fillna(1.0)
        out[c] = (games * r + k) / (games + k)
    return out


def _next_opponents(games, pool):
    """school -> {'opp', 'start'} for its next game not yet played."""
    out = {}
    if games is None or games.empty or "homeTeam" not in games.columns:
        return out
    g = games.copy()
    if "completed" in g.columns:
        g = g[g["completed"] != True]  # noqa: E712
    g = g.sort_values("startDate") if "startDate" in g.columns else g
    for _, r in g.iterrows():
        for me, them in ((r["homeTeam"], r["awayTeam"]), (r["awayTeam"], r["homeTeam"])):
            if me not in out and (not pool or me in pool):
                out[me] = {"opp": them, "start": str(r.get("startDate") or "")}
    return out


def build(player_stats_long, games, season, pool=None):
    """(rows, note, position_of). position_of(team, normalized_name) -> 'QB'/'RB'/'WR'/'TE' or None.
    Never raises: on failure rows is empty and note says why."""
    none = lambda team, name: None  # noqa: E731
    try:
        from src.data import cfbd_client as cfbd
        from src.features.live_player_features import _normalize_name
        from src.features.player_features import pivot_player_game_stats
        if player_stats_long is None or player_stats_long.empty:
            return [], "no box scores yet this season", none
        w = pivot_player_game_stats(player_stats_long, games)
        for c in ("athleteId", "week"):
            w[c] = pd.to_numeric(w[c], errors="coerce")
        w = w.dropna(subset=["athleteId", "opponent"])
        pos = {}
        try:
            r = cfbd.get_roster(int(season))
            if not r.empty and "id" in r.columns:
                ids = pd.to_numeric(r["id"], errors="coerce")
                pos = dict(zip(ids, r.get("position", pd.Series(index=r.index)).map(_group)))
        except Exception as e:
            print(f"  [warn] roster pull failed ({e}); defense vs position uses usage-based positions")
        w["g"] = w["athleteId"].map(pos)
        w["g"] = w["g"].where(w["g"].notna(), w.apply(_guess, axis=1))
        w["td"] = w["rush_tds"] + w["rec_tds"]
        src = {"td": "td", "rec": "receptions", "recyds": "rec_yds", "rushyds": "rush_yds"}
        cols = []
        for g in GROUPS:
            for s, c in src.items():
                w[f"{s}_{g}"] = w[c].where(w["g"] == g, 0.0)
                cols.append(f"{s}_{g}")
        og = w.groupby(["gameId", "week", "team", "opponent"])[cols].sum().reset_index()
        adj = _vs_usual(og, cols)
        if pool:
            og = og[og["opponent"].isin(pool)]
        tot = og.groupby("opponent")[cols].sum()
        n = og.groupby("opponent").size()
        nxt = _next_opponents(games, pool)

        # Recent TDs allowed: who scored.
        sc = w[w["td"] > 0].sort_values("week", ascending=False)
        rows = []
        for team in tot.index:
            gp = int(n[team])
            row = {"team": team, "games": gp, "next": nxt.get(team)}
            for g in GROUPS:
                for s in STATS:
                    v = float(tot.loc[team, f"{s}_{g}"])
                    row[f"{s}_{g}"] = int(round(v))
                    row[f"{s}_{g}_pg"] = round(v / gp, 2)
                    row[f"{s}_{g}_adj"] = round(float(adj.loc[team, f"{s}_{g}"]), 2) if team in adj.index else None
            row["td_all"] = sum(row[f"td_{g}"] for g in GROUPS)
            row["td_all_pg"] = round(row["td_all"] / gp, 2)
            log = og[og["opponent"] == team].sort_values("week", ascending=False)
            row["log"] = [{"wk": int(r["week"]), "vs": r["team"], **{g: int(r[f"td_{g}"]) for g in GROUPS},
                           "recyds": int(sum(r[f"recyds_{g}"] for g in GROUPS)), "rushyds": int(sum(r[f"rushyds_{g}"] for g in GROUPS))}
                          for _, r in log.iterrows()]
            mine = sc[sc["opponent"] == team].head(12)
            row["scorers"] = [{"wk": int(r["week"]), "player": r["player"], "team": r["team"], "pos": (r["g"] or "?").upper(),
                               "rush": int(r["rush_tds"]), "rec": int(r["rec_tds"])} for _, r in mine.iterrows()]
            rows.append(row)
        rows.sort(key=lambda r: -r["td_all_pg"])

        # Position lookup for props: this season's players by team + name.
        w["nk"] = w["player"].map(_normalize_name)
        lk = w.dropna(subset=["g"]).groupby(["team", "nk"])["g"].agg(lambda s: s.value_counts().index[0]).to_dict()

        def position_of(team, name):
            g = lk.get((team, _normalize_name(name or "")))
            return g.upper() if g else None

        known = float(w.loc[w["td"] > 0, "athleteId"].map(pos).notna().mean()) if (w["td"] > 0).any() else 0.0
        return rows, f"{len(rows)} FBS defenses, games through week {int(og['week'].max())}; {known:.0%} of scorers' positions from rosters", position_of
    except Exception as e:
        return [], f"not built ({type(e).__name__}: {e})", none
