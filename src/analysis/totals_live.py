"""
Live game totals (11/2026, Rod): official overs/unders with a play grade and
the weather, tracked like the official player props.

OFFICIAL RULES (from the 2023-26 totals backtest vs opening totals,
scripts/totals_lab.py -- see docs: new-features-2026-11-17):
  UNDER  the model's total is 3+ points under the line, on a total of 50+
         (backtest 198-127, +16.2%; profitable every season. Unders on
         totals under 50 lost: 18-21.)
  OVER   the model's total is 7+ points over the line, with no 10+ mph wind
         forecast (backtest 92-53, +21.1%; 2023 was flat. Overs 3-7 points
         off lost money, and 7+ overs in 10+ mph wind went 13-17.) An over
         needs a forecast on file (or a dome) to be official, since the
         wind rule can't be checked without one.
  The edge is always the model's total minus the CURRENT line, so every
  official play is still a bet at the line shown.

PLAY GRADE (same scale as the props play grade; bet A's):
  under  edge 3-5 = 0, 5-7 = +1, 7+ = +2; +1 for a 10+ mph wind (outdoors)
  over   +1
  both   line move since first flagged: toward our side +1, against -1
  2+ = A (🔥 bet it), 0-1 = B, below 0 = C
  Backtest without the line move: A 74-40 (+23.7%), B 216-140 (+15.8%).
  The rules and grade cut-offs were picked on that same backtest, so expect
  the live record to come in below it.
"""
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

UNDER_MIN_EDGE = 3.0
UNDER_MIN_LINE = 50.0
OVER_MIN_EDGE = 7.0
OVER_MAX_WIND = 10.0
WIND_GRADE_MPH = 10.0
MIN_GAMES = 2
LOG_PATH = "docs/data/totals_log.json"
LAB_DETAIL = "docs/data/totals_lab_detail.csv"
KEEP_DAYS = 200


# ---------------------------------------------------------------- inputs

def prior_season_team_inputs(year: int, raw_dir: str = "data/raw") -> dict:
    """school -> {pace, core_offense, core_defense} from the PRIOR finished
    season's saved CFBD files -- the same numbers training joins to a
    season's games (team_features.py shifts pace and CORE by one year)."""
    from src.features.team_features import build_pace_returning_features, build_core_season_features
    out = {}
    try:
        adv = pd.read_csv(f"{raw_dir}/adv_stats_{year}.csv")
        pr = build_pace_returning_features(adv, None)
        for _, r in pr.iterrows():
            out.setdefault(r["team"], {})["pace"] = r.get("pace")
    except Exception as e:
        print(f"  [totals] no {year} pace on file ({e})")
    try:
        core = build_core_season_features(pd.read_csv(f"{raw_dir}/core_ratings_{year}.csv"))
        for _, r in core.iterrows():
            e = out.setdefault(r["team"], {})
            e["core_offense"], e["core_defense"] = r.get("core_offense"), r.get("core_defense")
    except Exception as e:
        print(f"  [totals] no {year} CORE on file ({e})")
    return out


def game_inputs(home, away, form, v2, prior, weather, neutral, qb_out):
    """One model-ready row for an upcoming game, or None if either team has
    fewer than MIN_GAMES games this season (same bar as the spread model)."""
    hf, af = (form or {}).get(home), (form or {}).get(away)
    if not hf or not af or hf["games_played_prior"] < MIN_GAMES or af["games_played_prior"] < MIN_GAMES:
        return None
    row = {"home_field": 0 if neutral else 1}
    for side, school, f in (("home", home, hf), ("away", away, af)):
        row[f"{side}_roll_ppg_for"] = f["roll_ppg_for"]
        row[f"{side}_roll_ppg_against"] = f["roll_ppg_against"]
        row[f"{side}_games_played_prior"] = f["games_played_prior"]
        v = (v2 or {}).get(school) or {}
        for c in ("off_ppa", "def_ppa", "off_sr", "def_sr"):
            row[f"{side}_ewm_{c}"] = v.get(f"ewm_{c}")
        p = (prior or {}).get(school) or {}
        row[f"{side}_pace"] = p.get("pace")
        row[f"{side}_core_offense"] = p.get("core_offense")
        row[f"{side}_core_defense"] = p.get("core_defense")
        row[f"{side}_qb_out"] = float((qb_out or {}).get(school, 0))
    w = weather or {}
    indoors = bool(w.get("game_indoors"))
    row["game_indoors"] = 1.0 if indoors else (0.0 if w else np.nan)
    row["temperature"] = w.get("temperature")
    row["wind_speed"] = 0.0 if indoors else w.get("wind_speed")
    row["precipitation"] = 0.0 if indoors else w.get("precipitation")
    return row


# ---------------------------------------------------------------- rules

def _wind(w):
    if not w:
        return None
    return 0.0 if w.get("game_indoors") else w.get("wind_speed")


def evaluate(pred, line, weather):
    """(side, edge, official, why_not) at this line."""
    edge = pred - line
    side = "over" if edge > 0 else "under"
    wind = _wind(weather)
    if side == "under":
        if -edge < UNDER_MIN_EDGE:
            return side, edge, False, f"under by {-edge:.1f} (needs {UNDER_MIN_EDGE:g}+)"
        if line < UNDER_MIN_LINE:
            return side, edge, False, f"total under {UNDER_MIN_LINE:g}"
        return side, edge, True, ""
    if edge < OVER_MIN_EDGE:
        return side, edge, False, f"over by {edge:.1f} (needs {OVER_MIN_EDGE:g}+)"
    if wind is None:
        return side, edge, False, "no forecast yet (overs need one)"
    if wind >= OVER_MAX_WIND:
        return side, edge, False, f"{wind:.0f} mph wind"
    return side, edge, True, ""


def grade(side, edge, weather, line_now, first_line):
    wind = _wind(weather) or 0.0
    if side == "under":
        a = -edge
        score = (2 if a >= 7 else 1 if a >= 5 else 0) + (1 if wind >= WIND_GRADE_MPH else 0)
    else:
        score = 1
    move = 0
    if first_line is not None and line_now is not None and float(line_now) != float(first_line):
        d = float(line_now) - float(first_line)
        move = 1 if ((side == "under") == (d < 0)) else -1
    score += move
    g = "A" if score >= 2 else ("B" if score >= 0 else "C")
    return g, move


# ---------------------------------------------------------------- log

def load_log(path=LOG_PATH):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {"plays": {}, "updated": None}


def save_log(log, path=LOG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=KEEP_DAYS)
    log["plays"] = {k: v for k, v in log["plays"].items()
                    if pd.to_datetime(v.get("kickoff"), utc=True, errors="coerce") is pd.NaT
                    or pd.to_datetime(v.get("kickoff"), utc=True) >= cutoff}
    with open(path, "w") as f:
        json.dump(log, f, indent=1, default=str)


def run(scored_games, log, now_iso):
    """scored_games: [{home, away, home_school, away_school, kickoff, line,
    pred, weather, book}] for games not yet kicked off. Adds this run to the
    log and returns the dashboard rows (official plays only)."""
    plays = log.setdefault("plays", {})
    live_keys = set()
    rows = []
    for g in scored_games:
        side, edge, official, why_not = evaluate(g["pred"], g["line"], g.get("weather"))
        key = f"{g['home']}|{g['away']}|{g['kickoff']}|{side}"
        p = plays.get(key)
        if not official and not p:
            continue
        if p is None:
            p = plays[key] = {"home": g["home"], "away": g["away"], "home_school": g.get("home_school"),
                              "away_school": g.get("away_school"), "kickoff": g["kickoff"], "side": side,
                              "first": {"at": now_iso, "line": g["line"], "pred": round(g["pred"], 1)}, "h": []}
        gr, move = grade(side, edge, g.get("weather"), g["line"], p["first"]["line"])
        w = g.get("weather") or {}
        p["h"].append([now_iso, g["line"], round(g["pred"], 1), round(edge, 1), gr, 1 if official else 0])
        p["weather"] = w or None
        p["book"] = g.get("book")
        live_keys.add(key)
        if official:
            rows.append({"key": key, "home": g["home"], "away": g["away"], "kickoff": g["kickoff"],
                         "side": side, "line": g["line"], "pred": round(g["pred"], 1), "edge": round(edge, 1),
                         "grade": gr, "move": move, "first_line": p["first"]["line"], "first_at": p["first"]["at"],
                         "weather": w or None, "book": g.get("book"), "new": len(p["h"]) == 1})
    log["updated"] = now_iso
    rows.sort(key=lambda r: ({"A": 0, "B": 1, "C": 2}[r["grade"]], -abs(r["edge"])))
    return rows


def _final(p):
    """The play's last logged state before kickoff (its bet line and grade)."""
    k = pd.to_datetime(p.get("kickoff"), utc=True, errors="coerce")
    hist = [h for h in p.get("h") or [] if k is pd.NaT or pd.to_datetime(h[0], utc=True) < k]
    return hist[-1] if hist else None


def grade_results(log, schedule_df, normalize):
    """Fill result W/L/P for official plays whose game is final (CFBD scores)."""
    if schedule_df is None or schedule_df.empty or "homePoints" not in schedule_df.columns:
        return 0
    s = schedule_df.dropna(subset=["homePoints", "awayPoints"]).copy()
    s["_start"] = pd.to_datetime(s.get("startDate"), utc=True, errors="coerce")
    idx = {}
    for _, r in s.iterrows():
        idx.setdefault(frozenset({normalize(str(r["homeTeam"])), normalize(str(r["awayTeam"]))}), []).append(r)
    n = 0
    for p in log.get("plays", {}).values():
        if p.get("result"):
            continue
        f = _final(p)
        if not f or not f[5]:
            continue
        cand = idx.get(frozenset({normalize(str(p.get("home_school") or p["home"])),
                                  normalize(str(p.get("away_school") or p["away"]))})) or []
        k = pd.to_datetime(p["kickoff"], utc=True)
        cand = [r for r in cand if pd.notna(r["_start"]) and abs((r["_start"] - k).total_seconds()) < 2 * 86400]
        if not cand:
            continue
        total = float(cand[0]["homePoints"]) + float(cand[0]["awayPoints"])
        line = float(f[1])
        diff = total - line
        p["total"] = total
        p["result"] = "P" if diff == 0 else ("W" if (diff > 0) == (p["side"] == "over") else "L")
        p["units"] = 0.0 if p["result"] == "P" else (round(100 / 110, 4) if p["result"] == "W" else -1.0)
        p["final_grade"], p["final_line"] = f[4], line
        n += 1
    return n


def record(log):
    """Official totals record, overall / by grade / by side, plus graded plays."""
    done = [p for p in log.get("plays", {}).values() if p.get("result")]
    def summ(lst):
        w = sum(1 for p in lst if p["result"] == "W"); l = sum(1 for p in lst if p["result"] == "L")
        u = sum(float(p.get("units") or 0) for p in lst)
        return {"w": w, "l": l, "p": len(lst) - w - l, "units": round(u, 2),
                "roi": round(u / len(lst) * 100, 1) if lst else None}
    out = {"all": summ(done), "grades": {g: summ([p for p in done if p.get("final_grade") == g]) for g in "ABC"},
           "sides": {s: summ([p for p in done if p["side"] == s]) for s in ("over", "under")}}
    out["recent"] = sorted(({"home": p["home"], "away": p["away"], "kickoff": p["kickoff"], "side": p["side"],
                             "line": p.get("final_line"), "grade": p.get("final_grade"), "total": p.get("total"),
                             "result": p["result"]} for p in done), key=lambda r: r["kickoff"], reverse=True)[:40]
    return out


def backtest_by_grade(path=LAB_DETAIL):
    """The official rules and grade (no line move) on the 2023-26 totals lab,
    vs opening totals. Shown on the dashboard next to each grade."""
    try:
        d = pd.read_csv(path)
    except Exception:
        return None
    d = d[(d.version == "model") & (d.vs == "open")].copy()
    if d.empty:
        return None
    d["ae"] = d.edge.abs()
    ind = d.game_indoors.fillna(0).astype(bool)
    wind = np.where(ind, 0.0, d.wind_speed.fillna(0))
    un = (d.side == "under") & (d.ae >= UNDER_MIN_EDGE) & (d.line >= UNDER_MIN_LINE)
    ov = (d.side == "over") & (d.ae >= OVER_MIN_EDGE) & (wind < OVER_MAX_WIND)
    d = d[un | ov].copy()
    w = np.where(d.game_indoors.fillna(0).astype(bool), 0.0, d.wind_speed.fillna(0))
    score = np.where(d.side == "under", np.select([d.ae >= 7, d.ae >= 5], [2, 1], 0) + (w >= WIND_GRADE_MPH), 1)
    d["g"] = np.where(score >= 2, "A", np.where(score >= 0, "B", "C"))
    def summ(x):
        if x.empty:
            return None
        wn, ls = int((x.result == "W").sum()), int((x.result == "L").sum())
        return {"w": wn, "l": ls, "units": round(float(x.units.sum()), 1), "roi": round(float(x.units.mean()) * 100, 1)}
    return {"all": summ(d), "A": summ(d[d.g == "A"]), "B": summ(d[d.g == "B"]),
            "under": summ(d[d.side == "under"]), "over": summ(d[d.side == "over"]),
            "seasons": sorted(int(s) for s in d.season.unique())}
