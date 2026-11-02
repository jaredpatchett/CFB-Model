"""
Played check (11/2026): for every graded-tag prop the tracker left ungraded
because the player had no CFBD box-score line, ask PFF whether he took any
offensive snaps that week.

Why: books (DraftKings, FanDuel, BetMGM, Caesars) void a player prop only if
the player does not play a snap. A player who played but never touched the
ball is a real result (over loses, under wins), but CFBD box scores only list
players who recorded a stat, so the tracker can't tell him from a DNP and
voids both. That flatters overs and hurts unders.

Output: docs/data/played_check.csv -- one row per ungraded prop with
  played = yes      offensive snaps > 0 that week (bet had action; 0 stat)
           no       his team's game is charted but he has no offensive snaps
           pending  his team's game isn't charted by PFF yet -- re-run later
           unknown  name never found in PFF's 2026 offense data
No PFF numbers are written (public repo): only yes / no / unknown.
Special-teams-only snaps are not covered (PFF special summary is empty), so a
"no" could, rarely, be a special-teams-only player whose bet had action.

Usage: python scripts/played_check.py [--since 2026-10-07]
"""
import argparse
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import pff_client as pff  # noqa: E402

FLAGS = "data/clv/prop_flags.csv"
OUT = "docs/data/played_check.csv"
MATCH = pd.Timedelta(hours=36)


def norm(s):
    s = str(s or "").lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def pull(path, season, week):
    for attempt in range(3):
        try:
            payload = pff._get(path, {"league": pff.LEAGUE, "season": season, "week": week})
            return pd.DataFrame(pff._rows(payload))
        except pff.PFFError as e:
            if e.status in (401, 403):
                raise
            return pd.DataFrame()
        except Exception:
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def add_box_check(res, season):
    """team_box_players = how many of his teammates CFBD lists in that game's
    box score. A low count means CFBD's box score is missing/incomplete for
    the team (e.g. Iowa State vs BYU, Oct 9), so a 'yes' there is NOT a
    confirmed zero."""
    res["team_box_players"] = None
    if res.empty or not os.environ.get("CFBD_API_KEY"):
        return res
    try:
        from src.data import cfbd_client as cfbd
        from src.features.player_features import pivot_player_game_stats
        games = cfbd.get_games(season)
        games = games.assign(kick=pd.to_datetime(games["startDate"], utc=True, errors="coerce"))
        st = pd.to_datetime(res["start_time"], utc=True)
        near = games[games["kick"].apply(lambda k: pd.notna(k) and any(abs(k - x) <= MATCH for x in st.unique()))]
        weeks = sorted(int(w) for w in near["week"].dropna().unique())
        parts = [cfbd.get_player_game_stats(season, w) for w in weeks]
        wide = pivot_player_game_stats(pd.concat([p for p in parts if p is not None and not p.empty]), games)
        wide = wide.merge(games[["id", "kick"]].rename(columns={"id": "gameId"}), on="gameId", how="inner")
    except Exception as e:
        print(f"  [warn] CFBD box check skipped ({type(e).__name__}: {e})")
        return res
    from src.features.live_player_features import market_name_to_stat
    wide["n"] = wide["player"].map(norm)
    counts, now_stat = [], []
    for t, k, pl, mk in zip(res["team"], st, res["player"], res["market"]):
        rows = wide[(wide["team"] == t) & ((wide["kick"] - k).abs() <= MATCH)]
        counts.append(int(rows["player"].nunique()))
        me = rows[rows["n"] == norm(pl)]
        stat = market_name_to_stat(mk)
        now_stat.append(float(me.iloc[0][stat]) if not me.empty and stat in me.columns and pd.notna(me.iloc[0][stat]) else None)
    res["team_box_players"] = counts
    res["box_stat_now"] = now_stat   # CFBD stat if he's in the box score NOW (tracker's void may be stale)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-07")
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()

    tags = {"official", "capped", "under_play", "under_lean", "rec_yds_over", "rec_yds_play", "rush_under_play", "group"}

    f = pd.read_csv(FLAGS, keep_default_na=False)
    f["st"] = pd.to_datetime(f["start_time"], utc=True, errors="coerce")
    now = pd.Timestamp(datetime.now(timezone.utc))
    todo = f[(f["st"] >= pd.Timestamp(a.since, tz="UTC")) & (f["st"] <= now - pd.Timedelta(hours=6))
             & (f["result"].isin(["", "NA"])) & (f["line"] != "") & (f["price"] != "")]
    if tags:
        todo = todo[todo["tag"].isin(tags)]
    print(f"{len(todo)} ungraded props to check")

    # PFF weeks -> kickoff dates, and offense snaps per week.
    weeks, snaps = {}, []
    for w in pff.MODEL_WEEKS:
        try:
            g = pff.get_games(a.season, w)
        except Exception:
            g = pd.DataFrame()
        if g.empty:
            continue
        weeks[w] = pd.to_datetime(g["start"], utc=True, errors="coerce").dropna().tolist()
        o = pull("/v1/facet/offense/summary", a.season, w)
        if o.empty:
            continue
        name_col = "player" if "player" in o.columns else next(c for c in o.columns if "name" in c)
        snap_col = "snap_counts_total" if "snap_counts_total" in o.columns else "snap_counts_offense"
        o = pd.DataFrame({"n": o[name_col].map(norm), "team": o.get("team_name", o.get("team", "")),
                          "snaps": pd.to_numeric(o[snap_col], errors="coerce").fillna(0), "week": w})
        snaps.append(o)
        print(f"  PFF week {w}: {len(weeks[w])} games, {len(o)} offense rows")
    s = pd.concat(snaps, ignore_index=True) if snaps else pd.DataFrame(columns=["n", "team", "snaps", "week"])
    ever = set(s["n"])

    out = []
    for _, r in todo.iterrows():
        wk = next((w for w, ks in weeks.items() if any(abs(k - r["st"]) <= MATCH for k in ks)), None)
        n = norm(r["player"])
        rows = s[(s["n"] == n) & (s["week"] == wk)] if wk is not None else s.iloc[0:0]
        mine = s[s["n"] == n]
        pteam = mine["team"].mode().iloc[0] if not mine.empty else None
        charted = wk is not None and pteam is not None and ((s["week"] == wk) & (s["team"] == pteam)).any()
        if not rows.empty and rows["snaps"].max() > 0:
            played = "yes"
        elif n not in ever:
            played = "unknown"
        elif not charted:
            played = "pending"
        else:
            played = "no"
        out.append({"key": r["key"], "tag": r["tag"], "player": r["player"], "team": r["team"],
                    "opponent": r["opponent"], "market": r["market"], "side": r["side"],
                    "start_time": r["start_time"], "line": r["line"], "price": r["price"],
                    "pff_week": wk, "played": played})
    res = pd.DataFrame(out)
    res = add_box_check(res, a.season)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res.to_csv(OUT, index=False)
    if not res.empty:
        print(res.groupby(["tag", "played"]).size().to_string())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
