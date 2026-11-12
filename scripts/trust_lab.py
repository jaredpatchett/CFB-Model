#!/usr/bin/env python3
"""
Trust lab (11/2026): does the game breakdown's "trust check" sort the spread
plays into winners and losers?

The breakdown labels every Bet Card side by whether the model's own inputs
agree with its number (see build_dashboard.py, gbFootballSide):
  one_input            one input carries 60%+ (and 5+ pts) of the push toward the side
  situational          only home field / weather / rest / drive length push toward it,
                       and the on-field numbers (form, scoring, efficiency) don't
  ratings_vs_results   roster ratings (SP+/CORE) push toward it, on-field numbers push away
  solid / mixed / thin push toward the side vs away: 2x+ / 1-2x / less

Pushes are measured the same way as the live export (margin_drivers): each
group of inputs set to even and the game re-scored.

Walk-forward: each season graded by the live (upgraded) model trained only on
earlier seasons. Only games both teams had 2+ games for (where the live
dashboard uses the trained model and shows the read). Flagged = model side
1.9+ pts off the line, spreads under 20 (the live fade). Graded at -110 vs
the closing and the opening spread, for every flagged play and for the Bet
Card (top 10 a week by cover chance).

Outputs: docs/data/trust_lab.json, docs/data/trust_lab_detail.csv
Usage (after fetch_historical_data.py and build_features.py):
  python scripts/trust_lab.py --years 2022 2023 2024 2025 2026
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from scipy.stats import norm

import config
import run_backtest as rb
from src.analysis import game_profile as gp
from src.features.team_features_v2 import FEATURE_COLUMNS_V2
from src.models.game_model import GameMarginModel

OUT_JSON = "docs/data/trust_lab.json"
OUT_CSV = "docs/data/trust_lab_detail.csv"
MIN_EDGE = 1.9
BIG = 20

# Same groups and neutral values as export_dashboard_data.margin_drivers.
GROUPS = [
    ("form", ["ewm_adj_margin_diff"]),
    ("eff", ["ewm_net_ppa_diff", "ewm_off_ppa_diff", "ewm_def_ppa_diff", "ewm_off_sr_diff", "ewm_def_sr_diff"]),
    ("scoring", ["roll_margin_diff", "roll_ppg_for_diff", "roll_ppg_against_diff"]),
    ("ratings", ["sp_rating_diff", "core_overall_diff"]),
    ("returning", ["returning_production_diff"]),
    ("pace", ["pace_diff"]),
    ("home", ["home_field"]),
    ("rest", ["rest_diff"]),
    ("qb", ["home_qb_out", "away_qb_out"]),
    ("weather", ["temperature", "wind_speed", "precipitation"]),
]
NEUTRAL = {"temperature": 65.0, "wind_speed": 6.0, "precipitation": 0.0}
DIRECTION = {"form": "ewm_adj_margin_diff", "scoring": "roll_margin_diff", "eff": "ewm_net_ppa_diff",
             "ratings": "sp_rating_diff", "rest": "rest_diff"}
LABELS = ["one_input", "situational", "ratings_vs_results", "solid", "mixed", "thin"]


def pushes(model, frame):
    """{group: array of points toward HOME} for every row of frame."""
    cols = list(model.feature_columns)
    full = model.predict_margin(frame[cols])
    out = {}
    for key, feats in GROUPS:
        feats = [f for f in feats if f in cols]
        if not feats:
            continue
        alt = frame[cols].copy()
        for f in feats:
            alt[f] = NEUTRAL.get(f, 0.0)
        p = full - model.predict_margin(alt)
        if key == "weather" and "game_indoors" in frame.columns:
            p = np.where(frame["game_indoors"].fillna(0).astype(float) > 0, 0.0, p)
        out[key] = np.asarray(p, dtype=float)
    return out


def trust_label(row_pushes, row, sgn):
    """Same rules as the dashboard's trust check, from the picked side's view."""
    pro, con = [], []
    for key, arr_val in row_pushes.items():
        pts = sgn * arr_val
        if abs(pts) < 0.4:
            continue
        dcol = DIRECTION.get(key)
        if dcol and dcol in row and pd.notna(row[dcol]) and row[dcol] != 0 and ((sgn * row[dcol] > 0) != (pts > 0)):
            continue
        (pro if pts > 0 else con).append((key, pts))
    sp = sum(p for _, p in pro)
    sc = -sum(p for _, p in con)
    on = sum(p for k, p in pro + con if k in ("form", "scoring", "eff"))
    rat = sum(p for k, p in pro + con if k == "ratings")
    pro.sort(key=lambda x: -x[1])
    top = pro[0] if pro else None
    if top and top[1] >= 5 and top[1] >= 0.6 * sp:
        return "one_input", top[0]
    if pro and all(k in ("home", "weather", "rest", "pace") for k, _ in pro) and on <= 0:
        return "situational", None
    if rat >= 1 and on <= -1:
        return "ratings_vs_results", None
    if sp >= 2 * sc:
        return "solid", None
    if sp >= sc:
        return "mixed", None
    return "thin", None


def grade(d, line_col):
    d = d.dropna(subset=["m", line_col]).copy()
    d["edge"] = d["m"] + d[line_col]
    d = d[d["edge"] != 0]
    cover = d["margin"] + d[line_col]
    d["result"] = ["P" if c == 0 else ("W" if (c > 0) == (e > 0) else "L") for c, e in zip(cover, d["edge"])]
    d["units"] = d["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
    d["line_size"] = d[line_col].abs()
    d["cover_prob"] = norm.cdf(d["edge"].abs() / d["sd"])
    d["on_dog"] = np.where(d["edge"] > 0, d[line_col] > 0, d[line_col] < 0)
    return d


def rec(g):
    if g is None or len(g) == 0:
        return {"n": 0}
    w, l, p = int((g.result == "W").sum()), int((g.result == "L").sum()), int((g.result == "P").sum())
    u = float(g.units.sum())
    return {"n": int(len(g)), "record": f"{w}-{l}" + (f"-{p}" if p else ""), "units": round(u, 1), "roi": round(u / len(g) * 100, 1)}


def by_label(g):
    out = {lab: rec(g[g.trust == lab]) for lab in LABELS}
    out["weak (one_input + situational + ratings_vs_results + thin)"] = rec(g[g.trust.isin(["one_input", "situational", "ratings_vs_results", "thin"])])
    out["strong (solid + mixed)"] = rec(g[g.trust.isin(["solid", "mixed"])])
    for k, x in g[g.trust == "one_input"].groupby("trust_key"):
        out[f"one_input: {k}"] = rec(x)
    return out


def main(years):
    t0 = time.time()
    feats = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/team_game_features.csv")
    lines = rb.load_historical_lines()
    joined, strategy = rb.join_features_to_lines(feats, lines)
    has_sp = feats["sp_rating_diff"].notna()
    seasons = sorted(int(x) for x in feats.loc[has_sp, "season"].dropna().unique() if int(x) in years)
    feats = feats[feats["season"].isin(seasons)]
    print(f"Seasons: {seasons}; grading {seasons[1:]} walk-forward (join: {strategy})")
    parts = []
    for s in seasons[1:]:
        train = feats[(feats["season"] < s) & feats["margin"].notna()]
        test = joined[(joined["season"] == s) & joined["margin"].notna()].copy()
        gh = test.get("home_games_played_prior", pd.Series(0, index=test.index)).fillna(0)
        ga = test.get("away_games_played_prior", pd.Series(0, index=test.index)).fillna(0)
        test = test[(gh >= 2) & (ga >= 2)]
        m = GameMarginModel()
        m.feature_columns = list(FEATURE_COLUMNS_V2)
        m.fit(train, verbose=False)
        test = test[test[m.feature_columns].notna().all(axis=1)].copy()
        if test.empty:
            continue
        test["m"] = m.predict_margin(test[m.feature_columns])
        test["sd"] = m.residual_std
        P = pushes(m, test)
        for k, arr in P.items():
            test[f"push_{k}"] = arr
        parts.append(test)
        print(f"  {s}: {len(test)} games scored")
    scored = pd.concat(parts, ignore_index=True)
    results = {}
    details = []
    for line_name, col in (("closing", "market_spread_home"), ("opening", "market_spread_open_home")):
        if col not in scored.columns:
            continue
        d = grade(scored, col)
        d = d[(d.edge.abs() >= MIN_EDGE) & (d.line_size < BIG)].copy()
        labs = []
        for _, row in d.iterrows():
            sgn = 1 if row["edge"] > 0 else -1
            rp = {k: row[f"push_{k}"] for k, _ in GROUPS if f"push_{k}" in row}
            labs.append(trust_label(rp, row, sgn))
        d["trust"] = [x[0] for x in labs]
        d["trust_key"] = [x[1] for x in labs]
        d["rank"] = d.groupby(["season", "week"])["cover_prob"].rank(ascending=False, method="first")
        card = d[d["rank"] <= 10]
        results[line_name] = {
            "flagged": {"all": rec(d), **by_label(d)},
            "bet_card": {"all": rec(card), **by_label(card)},
            "bet_card_by_season": {str(int(s)): {"all": rec(x), "weak": rec(x[~x.trust.isin(["solid", "mixed"])]),
                                                 "strong": rec(x[x.trust.isin(["solid", "mixed"])])} for s, x in card.groupby("season")},
            "bet_card_mix": {k: int(v) for k, v in card.trust.value_counts().items()},
        }
        if line_name == "closing":
            keep = ["season", "week", "awayTeam", "homeTeam", col, "m", "margin", "edge", "cover_prob", "rank", "trust", "trust_key", "result", "units"]
            details.append(d[[c for c in keep if c in d.columns]])
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons_graded": sorted(int(s) for s in scored.season.unique()),
           "method": "walk-forward upgraded model; games with both teams 2+ games; flagged = 1.9+ off the line, spreads under 20; -110",
           "results": results}
    os.makedirs("docs/data", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if details:
        pd.concat(details).round(3).to_csv(OUT_CSV, index=False)
    for ln, r in results.items():
        print(f"\n== {ln} ==")
        for scope in ("flagged", "bet_card"):
            print(f"  {scope}:")
            for k, v in r[scope].items():
                print(f"    {k:60s} {v.get('record', '-'):>12s} {v.get('units', 0):+7.1f}u {v.get('roi', 0):+6.1f}%  n={v['n']}")
        print("  bet card by season:", json.dumps(r["bet_card_by_season"]))
    print(f"\nSaved {OUT_JSON} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    main(ap.parse_args().years)
