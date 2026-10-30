#!/usr/bin/env python3
"""
Grades a saved "This week's matchups" list (Defense vs Position tab) after the
games: data/clv/matchup_watch_<date>.csv -> docs/data/matchup_watch_<date>_results.csv
and _report.txt. For fun / research, not official plays. 1u per bet at the
saved price; anytime TD counts rushing, receiving and return TDs (what books
settle on). A player with no box-score line is listed as "no stats" and left
out of the record (books void most of these).

Usage (CFBD key; run from the workflow):
  python scripts/grade_matchup_watch.py --file data/clv/matchup_watch_2026-10-31.csv --season 2026
"""
import argparse
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import numpy as np
import pandas as pd

from src.data import cfbd_client as cfbd
from src.features.live_player_features import _normalize_name
from src.features.player_features import pivot_player_game_stats

STAT = {"Receptions": "receptions", "Reception Yards": "rec_yds", "Rush Yards": "rush_yds", "Rush Attempts": "rush_att",
        "Reception Touchdowns": "rec_tds", "Rush Touchdowns": "rush_tds", "Pass Touchdowns": "pass_tds", "Pass Yards": "pass_yds"}


def win_units(price):
    price = float(price)
    return price / 100.0 if price > 0 else 100.0 / -price


def box_scores(season, start, end):
    games = cfbd.get_games(season)
    games["start"] = pd.to_datetime(games["startDate"], utc=True, errors="coerce")
    g = games[(games["start"] >= start - pd.Timedelta(hours=6)) & (games["start"] <= end + pd.Timedelta(hours=6))]
    parts = []
    for wk in sorted(int(w) for w in g["week"].dropna().unique()):
        df = cfbd.get_player_game_stats(season, wk)
        if not df.empty:
            parts.append(df)
    long = pd.concat(parts, ignore_index=True)
    long = long[pd.to_numeric(long["gameId"], errors="coerce").isin(pd.to_numeric(g["id"], errors="coerce"))]
    wide = pivot_player_game_stats(long, games)
    r = long[long["category"].astype(str).str.lower().isin(["kickreturns", "puntreturns"]) & (long["statType"].astype(str).str.upper() == "TD")]
    ret = r.assign(v=pd.to_numeric(r["stat"], errors="coerce").fillna(0)).groupby(["gameId", "athleteId"])["v"].sum()
    wide["ret_tds"] = [ret.get((a, b), 0.0) for a, b in zip(wide["gameId"], wide["athleteId"])]
    wide["nk"] = wide["player"].map(_normalize_name)
    done = set(pd.to_numeric(g.loc[g.get("completed", pd.Series(True, index=g.index)) == True, "id"], errors="coerce"))  # noqa: E712
    return wide, g, done


def grade(row, wide):
    m = wide[(wide["team"] == row["team"]) & (wide["nk"] == _normalize_name(row["player"]))]
    if m.empty:
        return "no stats", np.nan, np.nan
    r = m.iloc[0]
    if row["bet"] == "Anytime TD":
        actual = r["rush_tds"] + r["rec_tds"] + r["ret_tds"]
        won = actual > 0
    else:
        actual = r[STAT[row["bet"]]]
        if actual == row["line"]:
            return "P", actual, 0.0
        won = (actual > row["line"]) == (row["side"] == "over")
    return ("W", actual, win_units(row["price"])) if won else ("L", actual, -1.0)


def rec(d):
    d = d[d["result"].isin(["W", "L", "P"])]
    if d.empty:
        return "-"
    w, l, p = (d["result"] == "W").sum(), (d["result"] == "L").sum(), (d["result"] == "P").sum()
    u = d["units"].sum()
    return f"{w}-{l}" + (f"-{p}" if p else "") + f", {u:+.2f}u ({u / len(d) * 100:+.1f}%)"


def main(path, season):
    df = pd.read_csv(path)
    ko = pd.to_datetime(df["kickoff"], utc=True)
    if pd.Timestamp.now(tz="UTC") < ko.max() + pd.Timedelta(hours=6):
        print(f"Too early: last kickoff {ko.max()} has not finished yet. Nothing graded.")
        return
    wide, games, done = box_scores(season, ko.min(), ko.max())
    out = df.apply(lambda r: pd.Series(grade(r, wide), index=["result", "actual", "units"]), axis=1)
    df = pd.concat([df, out], axis=1)
    base = os.path.basename(path).replace(".csv", "")
    df.to_csv(f"docs/data/{base}_results.csv", index=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        print(f"MATCHUP WATCH -- {base}  (opponent top 25 in TDs allowed to the player's position; 1u each at the saved price)")
        print(f"{len(df)} bets, {df['player'].nunique()} players; no box-score line: {(df['result'] == 'no stats').sum()}\n")
        a = df[df["bet"] == "Anytime TD"]
        print(f"ANYTIME TD (Yes): {rec(a)}")
        for pos, g in a.groupby("pos"):
            print(f"  {pos}: {rec(g)}")
        for lab, g in a.assign(b=pd.cut(a["opp_td_rank"], [0, 10, 25], labels=["opp top 10", "opp 11-25"])).groupby("b", observed=True):
            print(f"  {lab}: {rec(g)}")
        for lab, g in a.assign(b=pd.cut(a["price"], [-10000, -101, 150, 400, 100000], labels=["favorite (-101 or shorter)", "+100 to +150", "+151 to +400", "+401 or longer"])).groupby("b", observed=True):
            print(f"  {lab}: {rec(g)}")
        sc = a[a["result"] == "W"].sort_values("price", ascending=False)
        print("  scored: " + ", ".join(f"{r.player} ({r.pos}, {int(r.price):+d})" for r in sc.itertuples()))
        p = df[df["bet"] != "Anytime TD"]
        print(f"\nPROPS (model's lean): {rec(p)}")
        for m, g in p.groupby("bet"):
            print(f"  {m}: {rec(g)}")
        print(f"  model edge 10%+: {rec(p[p['model_ev'] >= 10])}")
        if "official" in p.columns:
            print(f"  official plays in the list: {rec(p[p['official'] == True])}")  # noqa: E712
    txt = buf.getvalue()
    print(txt)
    with open(f"docs/data/{base}_report.txt", "w") as f:
        f.write(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(a.file, a.season)
