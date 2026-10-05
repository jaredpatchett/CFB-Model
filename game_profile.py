"""
Game-profile labels for split testing (added 10/2026): is this a Power 4
game, a Group of 5 game, or a mix? Lower-profile games draw less betting
money and less oddsmaker attention, so their lines may be softer -- these
labels let the backtests check that directly.

Power 4 = SEC, Big Ten, Big 12, ACC, plus Notre Dame. (From 2024 on the
Pac-12 had only two members, so it counts as Group of 5 here.) Teams not in
CFBD's FBS list (FCS opponents) are labeled "FCS".
"""
import math

import pandas as pd

from src.data import cfbd_client as cfbd

POWER4 = {"SEC", "Big Ten", "Big 12", "ACC"}
POWER4_INDEPENDENTS = {"Notre Dame"}


def conference_map(seasons) -> dict:
    """(season, school) -> conference, from CFBD's FBS team list per season."""
    out = {}
    for s in sorted({int(x) for x in seasons if pd.notna(x)}):
        try:
            teams = cfbd.get_fbs_teams(s)
        except Exception as e:
            print(f"  [warn] could not load {s} conferences: {e}")
            continue
        for _, t in teams.iterrows():
            if t.get("school"):
                out[(s, t["school"])] = t.get("conference")
    return out


def tier(season, school, confs: dict) -> str:
    if school in POWER4_INDEPENDENTS:
        return "P4"
    conf = confs.get((int(season), school))
    if conf is None:
        return "FCS"
    return "P4" if conf in POWER4 else "G5"


def game_type(season, home, away, confs: dict) -> str:
    a, b = sorted([tier(season, home, confs), tier(season, away, confs)])
    return {("P4", "P4"): "Power 4 vs Power 4", ("G5", "P4"): "Power 4 vs Group of 5",
            ("G5", "G5"): "Group of 5 vs Group of 5"}.get((a, b), "involves FCS")


def summarize(df: pd.DataFrame, label: str) -> dict:
    """W-L-P, hit rate, units and ROI for a slice of graded bets (needs
    'result' in W/L/P and 'units'); 'season' or 'weekend' adds per-period ROI
    so you can see whether a slice was profitable consistently."""
    if df.empty:
        return {"group": label, "n": 0}
    w, l, p = int((df.result == "W").sum()), int((df.result == "L").sum()), int((df.result == "P").sum())
    n, units = len(df), float(df.units.sum())
    sd = df.units.std(ddof=1) if n > 1 else float("nan")
    z = (units / n) / (sd / math.sqrt(n)) if n > 1 and sd and sd > 0 else 0.0
    period = "season" if "season" in df.columns else ("weekend" if "weekend" in df.columns else None)
    by = {str(k): round(float(g.units.sum()) / len(g), 3) for k, g in df.groupby(period)} if period else {}
    return {"group": label, "n": n, "record": f"{w}-{l}-{p}", "hit_rate": round(w / (w + l), 3) if w + l else None,
            "units": round(units, 2), "roi": round(units / n, 3), "z": round(float(z), 2), "roi_by_period": by}


def print_rows(title: str, rows: list):
    print(f"\n{title}")
    print(f"  {'group':<40}{'n':>6}{'record':>12}{'hit%':>7}{'units':>9}{'ROI':>8}{'z':>6}   ROI by period")
    for r in rows:
        if not r.get("n"):
            print(f"  {r['group']:<40}{0:>6}")
            continue
        per = "  ".join(f"{k}:{v*100:+.1f}%" for k, v in sorted(r["roi_by_period"].items()))
        print(f"  {r['group']:<40}{r['n']:>6}{r['record']:>12}{(r['hit_rate'] or 0)*100:>7.1f}"
              f"{r['units']:>+9.2f}{r['roi']*100:>+7.1f}%{r['z']:>6.2f}   {per}")
