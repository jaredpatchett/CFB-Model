"""
TDs allowed by position (added 10/30/2026, display only).

For every FBS defense this season: rushing + receiving touchdowns it has
given up to QBs, RBs, WRs and TEs, per game and in total. Passing TDs are
not counted (a QB's passing TD is already the receiver's receiving TD).
Positions come from CFBD's roster; a player missing from it gets a guess
from his usage (passes thrown -> QB, more carries than catches -> RB,
otherwise WR).

Plain counts from box scores -- not adjusted for the offenses faced. Feeds
the dashboard's "TDs Allowed" tab only; nothing here touches the model or
the official plays.
"""
import pandas as pd

GROUPS = ("qb", "rb", "wr", "te")


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


def build(player_stats_long, games, season, pool=None):
    """(rows, note). Never raises: on failure rows is empty and note says why."""
    try:
        from src.data import cfbd_client as cfbd
        from src.features.player_features import pivot_player_game_stats
        if player_stats_long is None or player_stats_long.empty:
            return [], "no box scores yet this season"
        w = pivot_player_game_stats(player_stats_long, games)
        w["athleteId"] = pd.to_numeric(w["athleteId"], errors="coerce")
        w = w.dropna(subset=["athleteId", "opponent"])
        pos = {}
        try:
            r = cfbd.get_roster(int(season))
            if not r.empty and "id" in r.columns:
                ids = pd.to_numeric(r["id"], errors="coerce")
                pos = dict(zip(ids, r.get("position", pd.Series(index=r.index)).map(_group)))
        except Exception as e:
            print(f"  [warn] roster pull failed ({e}); TDs allowed uses usage-based positions")
        w["g"] = w["athleteId"].map(pos)
        w["g"] = w["g"].where(w["g"].notna(), w.apply(_guess, axis=1))
        w["td"] = w["rush_tds"] + w["rec_tds"]
        for g in GROUPS:
            w[g] = w["td"].where(w["g"] == g, 0.0)
        og = w.groupby(["gameId", "opponent"])[list(GROUPS)].sum().reset_index()
        if pool:
            og = og[og["opponent"].isin(pool)]
        tot = og.groupby("opponent")[list(GROUPS)].sum()
        n = og.groupby("opponent").size()
        rows = []
        for team in tot.index:
            gp = int(n[team])
            row = {"team": team, "games": gp}
            for g in GROUPS:
                row[g] = int(tot.loc[team, g])
                row[g + "_pg"] = round(tot.loc[team, g] / gp, 2)
            row["all"] = sum(row[g] for g in GROUPS)
            row["all_pg"] = round(row["all"] / gp, 2)
            rows.append(row)
        rows.sort(key=lambda r: -r["all_pg"])
        known = float(w.loc[w["td"] > 0, "athleteId"].map(pos).notna().mean()) if (w["td"] > 0).any() else 0.0
        return rows, f"{len(rows)} defenses, {int(og['gameId'].nunique())} games; {known:.0%} of scorers' positions from rosters"
    except Exception as e:
        return [], f"not built ({type(e).__name__}: {e})"
