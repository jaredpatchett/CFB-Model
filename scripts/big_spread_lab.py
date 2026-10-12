#!/usr/bin/env python3
"""
Big-spread lab (added 10/2026): are 20+ point spreads the model flags
profitable, or is fading them (the live rule) right?

The live Bet Card skips every spread of 20+ points. That rule was set off the
first two weeks of 2026 (16-12 with them, 15-6 without). Tracked 20+ plays
have since gone 16-2, so this re-tests the rule on two full seasons the model
never trained on, plus 2026 so far when its data is on file.

Models, each trained walk-forward (only on seasons before the one graded):
  preseason  margin = slope * SP+ diff + home field (what the dashboard used
             in weeks 1-3, before teams had two games)
  current    the original trained model
  upgraded   the live model since 10/5 (per-play efficiency, QB
             availability, recent-game weighting)
  as_live    upgraded once both teams have 2+ games, preseason before that
             (how the dashboard actually priced games)

A "flagged" play is the model's side when it is 1.9+ points off the line
(the dashboard's edge bar). Graded at -110 against the closing spread and,
separately, the opening spread.

Outputs (model results and CFBD game data only, no PFF):
  docs/data/big_spread_lab.json
  docs/data/big_spread_detail.csv   one row per 20+ game, for further slicing

Usage (after fetch_historical_data.py and build_features.py):
  python scripts/big_spread_lab.py --years 2022 2023 2024 2025 2026
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
from src.models import fair_odds as fo
from src.models.game_model import GameMarginModel

OUT_JSON = "docs/data/big_spread_lab.json"
OUT_CSV = "docs/data/big_spread_detail.csv"
MIN_EDGE = 1.9
BIG = 20
MODELS = ["preseason", "current", "upgraded", "as_live"]


def grade(d, mcol, line_col):
    """Model side vs line_col; adds edge, side, result, units, cover_prob."""
    d = d.dropna(subset=[mcol, line_col]).copy()
    d["edge"] = d[mcol] + d[line_col]
    d = d[d["edge"] != 0]
    cover = d["margin"] + d[line_col]
    d["result"] = ["P" if c == 0 else ("W" if (c > 0) == (e > 0) else "L") for c, e in zip(cover, d["edge"])]
    d["units"] = d["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
    pick_home = d["edge"] > 0
    # Underdog = the side getting points. Home is the dog when the home line is positive.
    d["on_dog"] = np.where(pick_home, d[line_col] > 0, d[line_col] < 0)
    d["line_size"] = d[line_col].abs()
    return d


def slices(g, label):
    """Every split we care about for one graded frame of flagged plays."""
    s = lambda x, name: gp.summarize(x, name)
    e = g["edge"].abs()
    rows = {
        "all": s(g, f"{label}: all"),
        "on_dog": s(g[g.on_dog], f"{label}: taking the points"),
        "on_fav": s(g[~g.on_dog], f"{label}: laying the points"),
        "edge_1.9_5": s(g[e < 5], f"{label}: edge 1.9-5"),
        "edge_5_10": s(g[(e >= 5) & (e < 10)], f"{label}: edge 5-10"),
        "edge_10_14": s(g[(e >= 10) & (e < 14)], f"{label}: edge 10-14"),
        "edge_14_plus": s(g[e >= 14], f"{label}: edge 14+"),
        "weeks_1_3": s(g[g.week <= 3], f"{label}: weeks 1-3"),
        "weeks_4_plus": s(g[g.week >= 4], f"{label}: weeks 4+"),
    }
    if "game_type" in g.columns:
        for k, x in g.groupby("game_type"):
            rows[f"type: {k}"] = s(x, f"{label}: {k}")
    if len(g):
        rows["model_cover_prob_avg"] = round(float(g["cover_prob"].mean()), 3)
    return rows


def evaluate(base, name):
    mcol, sdcol = f"m_{name}", f"sd_{name}"
    out = {}
    for line_name, col in (("closing", "market_spread_home"), ("opening", "market_spread_open_home")):
        if col not in base.columns:
            continue
        d = grade(base, mcol, col)
        d["cover_prob"] = norm.cdf(d["edge"].abs() / d[sdcol])
        flagged = d[d["edge"].abs() >= MIN_EDGE]
        big = flagged[flagged["line_size"] >= BIG]
        under = flagged[flagged["line_size"] < BIG]
        res = {
            "20_plus_every_game": gp.summarize(d[d["line_size"] >= BIG], "20+: every game, model side"),
            "20_plus_flagged": slices(big, "20+ flagged"),
            "20_plus_by_line": {lab: gp.summarize(big[(big.line_size >= lo) & (big.line_size < hi)], f"20+ flagged, line {lab}")
                                for lab, lo, hi in (("20-27.5", 20, 28), ("28-34.5", 28, 35), ("35+", 35, 999))},
            "under_20_flagged": gp.summarize(under, "under 20 flagged (for comparison)"),
        }
        # Bet Card with and without the fade: top 10 per week by cover probability.
        rank_all = flagged.copy()
        rank_all["rank"] = rank_all.groupby(["season", "week"])["cover_prob"].rank(ascending=False, method="first")
        rank_u = under.copy()
        rank_u["rank"] = rank_u.groupby(["season", "week"])["cover_prob"].rank(ascending=False, method="first")
        card_no_fade = rank_all[rank_all["rank"] <= 10]
        res["bet_card_with_fade"] = gp.summarize(rank_u[rank_u["rank"] <= 10], "Bet Card, 20+ faded (live rule)")
        res["bet_card_no_fade"] = gp.summarize(card_no_fade, "Bet Card, 20+ allowed")
        res["bet_card_no_fade_20_plus_only"] = gp.summarize(card_no_fade[card_no_fade.line_size >= BIG],
                                                             "20+ plays that would make the card")
        res["bet_card_plus_all_20_flagged"] = gp.summarize(pd.concat([rank_u[rank_u["rank"] <= 10], big]),
                                                            "Bet Card + every flagged 20+ play")
        if line_name == "opening" and "market_spread_home" in big.columns:
            mv = (big["market_spread_open_home"] - big["market_spread_home"]) * np.sign(big["edge"])
            moved = mv[mv != 0]
            res["20_plus_line_moved_toward_model"] = round(float((moved > 0).mean()), 3) if len(moved) else None
            res["20_plus_lines_moved"] = int(len(moved))
        out[line_name] = res
    return out


def main(years):
    t0 = time.time()
    fpath = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if not os.path.exists(fpath):
        raise SystemExit("Run scripts/build_features.py first.")
    feats = pd.read_csv(fpath)
    lines = rb.load_historical_lines()
    if lines.empty:
        raise SystemExit("No historical lines in data/raw -- run scripts/fetch_historical_data.py first.")
    joined, strategy = rb.join_features_to_lines(feats, lines)
    has_sp = feats["sp_rating_diff"].notna()
    seasons = sorted(int(x) for x in feats.loc[has_sp, "season"].dropna().unique() if int(x) in years)
    feats = feats[feats["season"].isin(seasons)]
    print(f"Seasons with SP+ on file: {seasons}; grading {seasons[1:]} walk-forward (join: {strategy})")

    parts = []
    for test_season in seasons[1:]:
        train = feats[(feats["season"] < test_season) & feats["margin"].notna()]
        test = joined[(joined["season"] == test_season) & joined["margin"].notna()].copy()
        if test.empty:
            print(f"  {test_season}: no completed games with a line, skipped")
            continue
        # Preseason formula, fit on earlier seasons only.
        prior = fo.fit_preseason_prior(train)
        neutral = test["home_field"].eq(0) if "home_field" in test.columns else pd.Series(False, index=test.index)
        test["m_preseason"] = [fo.preseason_predicted_margin(x, prior, neutral_site=bool(n)) if pd.notna(x) else np.nan
                               for x, n in zip(test["sp_rating_diff"], neutral)]
        test["sd_preseason"] = prior["residual_std"]
        for name, cols in (("current", None), ("upgraded", list(FEATURE_COLUMNS_V2))):
            m = GameMarginModel()
            if cols:
                m.feature_columns = cols
            try:
                m.fit(train, verbose=False)
            except Exception as e:
                print(f"  {test_season} {name}: could not train ({e})")
                test[f"m_{name}"], test[f"sd_{name}"] = np.nan, np.nan
                continue
            ok = test[m.feature_columns].notna().all(axis=1)
            test[f"m_{name}"] = np.nan
            if ok.any():
                test.loc[ok, f"m_{name}"] = m.predict_margin(test[ok])
            test[f"sd_{name}"] = m.residual_std
        gp_h = test.get("home_games_played_prior", pd.Series(0, index=test.index)).fillna(0)
        gp_a = test.get("away_games_played_prior", pd.Series(0, index=test.index)).fillna(0)
        use_trained = test["m_upgraded"].notna() & (gp_h >= 2) & (gp_a >= 2)
        test["m_as_live"] = np.where(use_trained, test["m_upgraded"], test["m_preseason"])
        test["sd_as_live"] = np.where(use_trained, test["sd_upgraded"], test["sd_preseason"])
        parts.append(test)
        print(f"  {test_season}: {len(test)} games; scored preseason {int(test.m_preseason.notna().sum())}, "
              f"current {int(test.m_current.notna().sum())}, upgraded {int(test.m_upgraded.notna().sum())}")
    if not parts:
        raise SystemExit("No season could be graded.")
    scored = pd.concat(parts, ignore_index=True).dropna(subset=["market_spread_home"])
    try:
        confs = gp.conference_map(scored["season"].unique())
        scored["game_type"] = [gp.game_type(s, h, a, confs) for s, h, a in
                               zip(scored["season"], scored["homeTeam"], scored["awayTeam"])]
    except Exception as e:
        print(f"  [warn] game types not attached: {e}")

    results = {}
    for name in MODELS:
        if f"m_{name}" not in scored.columns or scored[f"m_{name}"].notna().sum() == 0:
            continue
        results[name] = evaluate(scored, name)
        results[name]["by_season_20_plus_flagged_closing"] = {
            str(int(s)): gp.summarize(g, f"{s}") for s, g in
            (lambda d: d[(d.edge.abs() >= MIN_EDGE) & (d.line_size >= BIG)])(grade(scored, f"m_{name}", "market_spread_home")).groupby("season")}

    big_rows = scored[(scored["market_spread_home"].abs() >= BIG) |
                      (scored.get("market_spread_open_home", pd.Series(np.nan, index=scored.index)).abs() >= BIG)]
    keep = [c for c in ["season", "week", "awayTeam", "homeTeam", "game_type", "market_spread_open_home", "market_spread_home",
                        "m_preseason", "m_current", "m_upgraded", "m_as_live", "margin", "home_games_played_prior",
                        "away_games_played_prior"] if c in big_rows.columns]
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    big_rows[keep].round(2).sort_values(["season", "week"]).to_csv(OUT_CSV, index=False)

    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons_graded": sorted(int(s) for s in scored.season.unique()),
           "games_graded": int(len(scored)), "games_20_plus": int((scored["market_spread_home"].abs() >= BIG).sum()),
           "min_edge": MIN_EDGE, "join_strategy": strategy, "results": results,
           "method": "walk-forward: each season graded by models trained only on earlier seasons; -110 pricing; "
                     "flagged = model side 1.9+ pts off the line"}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    report(out)
    print(f"\nSaved {OUT_JSON} and {OUT_CSV} ({len(big_rows)} rows) in {time.time() - t0:.0f}s")


def report(out):
    cell = lambda r: f"{r['record']:>11} {r['roi'] * 100:+6.1f}%" if r and r.get("n") else f"{'-':>19}"
    print(f"\nBIG-SPREAD LAB -- seasons {out['seasons_graded']}, {out['games_20_plus']} games with a 20+ closing spread")
    for name, r in out["results"].items():
        for ln in ("closing", "opening"):
            x = r.get(ln)
            if not x:
                continue
            print(f"\n  {name.upper()} vs {ln.upper()} lines")
            f = x["20_plus_flagged"]
            for k in ("all", "on_dog", "on_fav", "edge_1.9_5", "edge_5_10", "edge_10_14", "edge_14_plus", "weeks_1_3", "weeks_4_plus"):
                print(f"    {f[k]['group']:<44}{cell(f[k])}")
            for k in ("under_20_flagged", "bet_card_with_fade", "bet_card_no_fade", "bet_card_no_fade_20_plus_only",
                      "bet_card_plus_all_20_flagged"):
                print(f"    {x[k]['group']:<44}{cell(x[k])}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    main(sorted(ap.parse_args().years))
