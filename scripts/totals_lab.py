#!/usr/bin/env python3
"""
Totals lab (added 11/2026): is there money in game totals (over/under)?

The spread model predicts margin only, and every game input is a home-minus-
away difference, which says nothing about how many points get scored. This
builds a totals model from the same leakage-safe inputs, kept per team
instead of differenced (each side's scoring form, plays per drive, per-play
efficiency on offense and defense, CORE offense/defense, weather), and grades
it walk-forward: each season is predicted by a model trained only on earlier
seasons, then bet against that game's real closing and opening totals.

Versions (same games for all):
  formula     no model: average of each offense's points for and the other
              defense's points against so far this season (baseline)
  model       gradient-boosted total points from the inputs above
  market      same model, trained to predict how far the final total lands
              from the OPENING total (learns where openers are off), then
              added back onto the opener

Also tests a weather rule with no model at all: unders in 15+ mph wind.

Outputs docs/data/totals_lab.json and docs/data/totals_lab_detail.csv.
Nothing live changes.

Usage (after fetch_historical_data.py and build_features.py):
  python scripts/totals_lab.py --years 2022 2023 2024 2025 2026
"""
import argparse
import glob
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
from sklearn.ensemble import GradientBoostingRegressor

import config
import run_backtest as rb
from src.analysis import game_profile as gp

OUT = "docs/data/totals_lab.json"
OUT_DETAIL = "docs/data/totals_lab_detail.csv"

BASE_INPUTS = [
    "home_roll_ppg_for", "home_roll_ppg_against", "away_roll_ppg_for", "away_roll_ppg_against",
    "home_pace", "away_pace",
    "home_ewm_off_ppa", "home_ewm_def_ppa", "away_ewm_off_ppa", "away_ewm_def_ppa",
    "home_ewm_off_sr", "home_ewm_def_sr", "away_ewm_off_sr", "away_ewm_def_sr",
    "home_core_offense", "home_core_defense", "away_core_offense", "away_core_defense",
    "home_games_played_prior", "away_games_played_prior",
    "temperature", "wind_speed", "precipitation", "game_indoors",
    "home_field", "home_qb_out", "away_qb_out",
]
DERIVED = ["pts_formula", "pace_sum", "ppa_sum", "sr_sum"]


def add_inputs(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    g = lambda c: pd.to_numeric(d[c], errors="coerce") if c in d.columns else pd.Series(np.nan, index=d.index)
    # Expected points: each offense's points for, averaged with the other
    # defense's points allowed, summed over both teams.
    d["pts_formula"] = ((g("home_roll_ppg_for") + g("away_roll_ppg_against")) / 2
                        + (g("away_roll_ppg_for") + g("home_roll_ppg_against")) / 2)
    d["pace_sum"] = g("home_pace") + g("away_pace")
    # def PPA = points added per play ALLOWED, so higher = worse defense = more points.
    d["ppa_sum"] = g("home_ewm_off_ppa") + g("away_ewm_off_ppa") + g("home_ewm_def_ppa") + g("away_ewm_def_ppa")
    d["sr_sum"] = g("home_ewm_off_sr") + g("away_ewm_off_sr") + g("home_ewm_def_sr") + g("away_ewm_def_sr")
    if "game_indoors" in d.columns:
        d["game_indoors"] = d["game_indoors"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(
            pd.to_numeric(d["game_indoors"], errors="coerce"))
    return d


def attach_totals(feats: pd.DataFrame) -> pd.DataFrame:
    """Final total points per game: from the features file when it carries
    the scores, else from games_{year}.csv by CFBD game id."""
    if {"homePoints", "awayPoints"} <= set(feats.columns):
        feats["total"] = pd.to_numeric(feats["homePoints"], errors="coerce") + pd.to_numeric(feats["awayPoints"], errors="coerce")
        return feats
    frames = [pd.read_csv(f, usecols=lambda c: c in ("id", "homePoints", "awayPoints"))
              for f in glob.glob(f"{config.DATA_RAW_DIR}/games_*.csv")]
    g = pd.concat(frames, ignore_index=True).dropna()
    g["total"] = g["homePoints"] + g["awayPoints"]
    return feats.merge(g[["id", "total"]], on="id", how="left")


def usable(cols, df):
    return [c for c in cols if c in df.columns and df[c].notna().sum() > 0]


def fit_gbm(X, y):
    m = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                                  min_samples_leaf=20, random_state=42)
    m.fit(X, y)
    return m


def grade(d: pd.DataFrame, pcol: str, line_col: str) -> pd.DataFrame:
    d = d.dropna(subset=[pcol, line_col, "total"]).copy()
    d["line"] = d[line_col].astype(float)
    d["edge"] = d[pcol] - d["line"]
    d = d[d["edge"] != 0]
    d["side"] = np.where(d["edge"] > 0, "over", "under")
    diff = d["total"] - d["line"]
    d["result"] = np.where(diff == 0, "P", np.where((diff > 0) == (d["side"] == "over"), "W", "L"))
    d["units"] = d["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
    return d


def card(d: pd.DataFrame, per_week: int = 5) -> pd.DataFrame:
    d = d.copy()
    d["rank"] = d.groupby(["season", "week"])["edge"].transform(lambda s: s.abs().rank(ascending=False, method="first"))
    return d[d["rank"] <= per_week]


def slices(c: pd.DataFrame, tag: str) -> dict:
    a = c.edge.abs()
    out = {
        "every_game": gp.summarize(c, f"{tag}: every game"),
        "edge_3_plus": gp.summarize(c[a >= 3], f"{tag}: 3+ pts off"),
        "edge_5_plus": gp.summarize(c[a >= 5], f"{tag}: 5+ pts off"),
        "edge_7_plus": gp.summarize(c[a >= 7], f"{tag}: 7+ pts off"),
        "top5_per_week": gp.summarize(card(c), f"{tag}: top 5 a week"),
        "overs_3_plus": gp.summarize(c[(a >= 3) & (c.side == "over")], f"{tag}: overs 3+"),
        "unders_3_plus": gp.summarize(c[(a >= 3) & (c.side == "under")], f"{tag}: unders 3+"),
    }
    return out


def main(years):
    t0 = time.time()
    fpath = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if not os.path.exists(fpath):
        raise SystemExit("Run scripts/build_features.py first.")
    feats = attach_totals(pd.read_csv(fpath, low_memory=False))
    feats = add_inputs(feats)
    lines = rb.load_historical_lines()
    if "market_over_under_open" not in lines.columns:
        lines["market_over_under_open"] = np.nan
        print("  [warn] no opening totals in the saved lines -- opener tests will be empty")
    lines = lines.drop(columns=[c for c in ("total",) if c in lines.columns])
    joined, strategy = rb.join_features_to_lines(feats, lines)
    joined = joined[pd.to_numeric(joined["market_over_under"], errors="coerce") > 0].copy()
    joined["market_over_under"] = joined["market_over_under"].astype(float)
    joined["market_over_under_open"] = pd.to_numeric(joined["market_over_under_open"], errors="coerce")
    joined.loc[joined["market_over_under_open"] <= 0, "market_over_under_open"] = np.nan

    seasons = sorted(int(s) for s in feats["season"].dropna().unique() if int(s) in years)
    inputs = usable(BASE_INPUTS + DERIVED, feats)
    print(f"  {len(inputs)} inputs: {inputs}")
    imps, parts = {}, []
    for test_season in seasons[1:]:
        train = feats[(feats["season"] < test_season) & feats["total"].notna() & feats["season"].isin(seasons)]
        train = train[train["home_games_played_prior"].fillna(0) + train["away_games_played_prior"].fillna(0) > 0] \
            if "home_games_played_prior" in train.columns else train
        test = joined[(joined["season"] == test_season) & joined["total"].notna()].copy()
        if train.empty or test.empty:
            continue
        cols = usable(inputs, train)
        Xtr = train[cols].astype(float).fillna(train[cols].astype(float).median())
        med = train[cols].astype(float).median()
        Xte = test[cols].astype(float).fillna(med)
        m = fit_gbm(Xtr, train["total"].astype(float))
        test["p_model"] = m.predict(Xte)
        test["p_formula"] = test["pts_formula"]
        imps[str(test_season)] = dict(sorted(((c, round(float(v), 4)) for c, v in zip(cols, m.feature_importances_)),
                                             key=lambda kv: -kv[1])[:12])
        # Market version: learn final total minus the opener, on earlier seasons' games.
        jtr = joined[(joined["season"] < test_season) & joined["total"].notna() & joined["market_over_under_open"].notna()]
        test["p_market"] = np.nan
        if len(jtr) > 300:
            cols_m = usable(cols, jtr) + ["market_over_under_open"]
            med_m = jtr[cols_m].astype(float).median()
            mm = fit_gbm(jtr[cols_m].astype(float).fillna(med_m),
                         (jtr["total"] - jtr["market_over_under_open"]).astype(float))
            ok = test["market_over_under_open"].notna()
            if ok.any():
                test.loc[ok, "p_market"] = test.loc[ok, "market_over_under_open"] + mm.predict(
                    test.loc[ok, cols_m].astype(float).fillna(med_m))
        parts.append(test)
        print(f"  {test_season}: {len(train)} training games, {len(test)} graded")
    scored = pd.concat(parts, ignore_index=True)
    # Early-season games have little or no this-season form; a separate
    # slice shows whether the edge only exists once teams have 3+ games.
    gp_min = scored[["home_games_played_prior", "away_games_played_prior"]].min(axis=1) \
        if "home_games_played_prior" in scored.columns else pd.Series(9, index=scored.index)
    scored["both_3plus_games"] = gp_min >= 3

    results, details = {}, []
    base = scored.dropna(subset=["p_model", "p_formula"])
    for name in ("formula", "model", "market"):
        pcol = f"p_{name}"
        b = base.dropna(subset=[pcol])
        if b.empty:
            continue
        r = {"n_games": int(len(b)), "avg_miss_pts": round(float((b[pcol] - b["total"]).abs().mean()), 2),
             "market_close_avg_miss": round(float((b["market_over_under"] - b["total"]).abs().mean()), 2)}
        if name != "market":   # the market version is built on the opener; grading it at the close is still fair
            pass
        c = grade(b, pcol, "market_over_under")
        r["vs_closing"] = slices(c, f"{name} vs close")
        r["vs_closing_3plus_games"] = slices(c[c.both_3plus_games], f"{name} vs close, wk 4+")
        o = grade(b.dropna(subset=["market_over_under_open"]), pcol, "market_over_under_open")
        if len(o):
            r["vs_opening"] = slices(o, f"{name} vs open")
        results[name] = r
        for lab, g in (("close", c), ("open", o)):
            if len(g):
                details.append(g.assign(version=name, vs=lab)[
                    ["version", "vs", "season", "week", "homeTeam", "awayTeam", "line", pcol, "edge", "side", "total",
                     "result", "units", "wind_speed", "temperature", "precipitation", "game_indoors",
                     "both_3plus_games"]].rename(columns={pcol: "pred"}))

    # Weather rule, no model: unders when the wind is 15+ mph (outdoors).
    weather = {}
    if "wind_speed" in scored.columns:
        w = scored.dropna(subset=["wind_speed"]).copy()
        w = w[w["game_indoors"].fillna(0).astype(float) == 0]
        for lab, line_col in (("close", "market_over_under"), ("open", "market_over_under_open")):
            u = w.dropna(subset=[line_col]).copy()
            u["line"] = u[line_col].astype(float)
            diff = u["total"] - u["line"]
            u["result"] = np.where(diff == 0, "P", np.where(diff < 0, "W", "L"))
            u["units"] = u["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
            ws = u["wind_speed"].astype(float)
            weather[f"unders_vs_{lab}"] = {
                "wind_under_10": gp.summarize(u[ws < 10], f"every under, wind <10 ({lab})"),
                "wind_10_15": gp.summarize(u[(ws >= 10) & (ws < 15)], f"every under, wind 10-15 ({lab})"),
                "wind_15_plus": gp.summarize(u[ws >= 15], f"every under, wind 15+ ({lab})"),
                "wind_20_plus": gp.summarize(u[ws >= 20], f"every under, wind 20+ ({lab})"),
                "rain": gp.summarize(u[u["precipitation"].fillna(0).astype(float) > 0], f"every under, rain/snow ({lab})"),
                "cold_under_35F": gp.summarize(u[u["temperature"].astype(float) < 35], f"every under, under 35F ({lab})"),
            }

    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons_graded": seasons[1:],
           "games": int(len(base)), "join_strategy": strategy, "results": results, "weather_rule": weather,
           "input_importance": imps,
           "method": "walk-forward by season; same games for every version; -110; edge = projected total minus the line"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if details:
        pd.concat(details, ignore_index=True).to_csv(OUT_DETAIL, index=False)

    print(f"\nTOTALS LAB -- {out['games']} games, seasons {out['seasons_graded']}")
    for n, r in results.items():
        print(f"\n== {n}: avg miss {r['avg_miss_pts']} pts (closing line misses by {r['market_close_avg_miss']})")
        rows = list(r["vs_closing"].values()) + list(r["vs_closing_3plus_games"].values()) + list((r.get("vs_opening") or {}).values())
        gp.print_rows(n, rows)
    for k, v in weather.items():
        gp.print_rows(f"weather rule {k}", list(v.values()))
    print(f"\nSaved {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    main(sorted(ap.parse_args().years))
