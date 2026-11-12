#!/usr/bin/env python3
"""
Line-move lab (11/2026): when the line moves after the model flags a spread
play, does the direction tell us anything?

  toward  the market moved toward the model's side (worse number for us,
          the market agreeing): e.g. IU -7.5 -> -8.5 on an IU pick
  away    the market moved away from it (better number, the market
          disagreeing): e.g. SJSU +10.5 -> +12.5 on an SJSU pick
  none    no move

Rebuilt 2026 card (weeks 4-10, 68 plays): toward 16-4, away 17-15, none 4-8.
This re-tests that on 2024-26, walk-forward.

Setup, as you'd actually bet it: the play is flagged against the OPENING
spread (model side 1.9+ pts off the opener, spreads under 20), with the live
(upgraded) model trained only on earlier seasons, games where both teams had
2+ games. Then the open -> close move is measured from the picked side.
Graded at -110 three ways:
  at the open    you bet it when flagged
  at the close   you waited and bet the final number
  fade at close  you took the other side at the close
Also the Bet Card (top 10 a week by cover chance at the open), and the
"pass when it moves away" card.

Outputs: docs/data/move_lab.json, docs/data/move_lab_detail.csv
Usage: python scripts/move_lab.py --years 2022 2023 2024 2025 2026
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
from src.features.team_features_v2 import FEATURE_COLUMNS_V2
from src.models.game_model import GameMarginModel

OUT_JSON = "docs/data/move_lab.json"
OUT_CSV = "docs/data/move_lab_detail.csv"
MIN_EDGE = 1.9
BIG = 20
WIN = 100 / 110


def res(cover, pick_home):
    """W/L/P for a pick on the home side (pick_home) given home cover margin."""
    return np.where(cover == 0, "P", np.where((cover > 0) == pick_home, "W", "L"))


def rec(g, col):
    if g is None or len(g) == 0:
        return {"n": 0}
    r = g[col]
    w, l, p = int((r == "W").sum()), int((r == "L").sum()), int((r == "P").sum())
    u = w * WIN - l
    return {"n": int(len(g)), "record": f"{w}-{l}" + (f"-{p}" if p else ""), "units": round(u, 1), "roi": round(u / len(g) * 100, 1)}


def block(g):
    return {"at_open": rec(g, "r_open"), "at_close": rec(g, "r_close"), "fade_at_close": rec(g, "r_fade")}


def split(g):
    out = {"all": block(g)}
    for k in ("toward", "none", "away"):
        out[k] = block(g[g.move_dir == k])
    for lab, lo, hi in (("toward 0.5-1", 0.25, 1.25), ("toward 1.5-2.5", 1.25, 2.75), ("toward 3+", 2.75, 99),
                        ("away 0.5-1", -1.25, -0.25), ("away 1.5-2.5", -2.75, -1.25), ("away 3+", -99, -2.75)):
        out[lab] = block(g[(g.move > lo) & (g.move <= hi)] if lo > 0 else g[(g.move >= lo) & (g.move < hi)])
    return out


def main(years):
    t0 = time.time()
    feats = pd.read_csv(f"{config.DATA_PROCESSED_DIR}/team_game_features.csv")
    lines = rb.load_historical_lines()
    joined, strategy = rb.join_features_to_lines(feats, lines)
    if "market_spread_open_home" not in joined.columns:
        raise SystemExit("No opening spreads in the historical lines -- can't run this lab.")
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
        test["m"] = m.predict_margin(test[m.feature_columns])
        test["sd"] = m.residual_std
        parts.append(test)
        print(f"  {s}: {len(test)} games scored")
    d = pd.concat(parts, ignore_index=True).dropna(subset=["market_spread_open_home", "market_spread_home"])
    o, c = d["market_spread_open_home"].astype(float), d["market_spread_home"].astype(float)
    d["edge_open"] = d["m"] + o
    d = d[(d.edge_open.abs() >= MIN_EDGE) & (o.abs() < BIG)].copy()
    o, c = d["market_spread_open_home"].astype(float), d["market_spread_home"].astype(float)
    pick_home = (d["edge_open"] > 0).values
    sgn = np.where(pick_home, 1, -1)
    # Picked side's spread: home spread for a home pick, its negative for an away pick.
    side_open, side_close = sgn * o, sgn * c
    d["move"] = side_open - side_close            # + = number got worse = market moved toward our side
    d["move_dir"] = np.where(d["move"] > 0, "toward", np.where(d["move"] < 0, "away", "none"))
    d["r_open"] = res(d["margin"] + o, pick_home)
    d["r_close"] = res(d["margin"] + c, pick_home)
    d["r_fade"] = res(d["margin"] + c, ~pick_home)
    d["cover_prob"] = norm.cdf(d["edge_open"].abs() / d["sd"])
    d["rank"] = d.groupby(["season", "week"])["cover_prob"].rank(ascending=False, method="first")
    card = d[d["rank"] <= 10]
    results = {
        "flagged_at_open": split(d),
        "bet_card": split(card),
        "bet_card_by_season": {str(int(s)): {k: block(x[x.move_dir == k]) for k in ("toward", "none", "away")} | {"all": block(x)}
                               for s, x in card.groupby("season")},
        "strategies_bet_card": {
            "bet everything at the open": rec(card, "r_open"),
            "bet everything at the close": rec(card, "r_close"),
            "bet at the open, but pass when the line moved away": rec(card[card.move_dir != "away"], "r_open"),
            "wait, bet at the close only if it moved toward us": rec(card[card.move_dir == "toward"], "r_close"),
            "wait, bet at the close unless it moved away": rec(card[card.move_dir != "away"], "r_close"),
            "fade at the close when it moved away": rec(card[card.move_dir == "away"], "r_fade"),
        },
        "move_mix_bet_card": {k: int(v) for k, v in card.move_dir.value_counts().items()},
    }
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons_graded": sorted(int(s) for s in d.season.unique()),
           "method": "walk-forward upgraded model; both teams 2+ games; flagged vs the opener (1.9+, under 20); -110",
           "results": results}
    os.makedirs("docs/data", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    keep = ["season", "week", "awayTeam", "homeTeam", "market_spread_open_home", "market_spread_home", "m", "margin",
            "edge_open", "move", "move_dir", "rank", "r_open", "r_close", "r_fade"]
    d[[k for k in keep if k in d.columns]].round(3).to_csv(OUT_CSV, index=False)
    for scope in ("flagged_at_open", "bet_card"):
        print(f"\n== {scope} ==")
        for k, v in results[scope].items():
            print(f"  {k:18s} open {v['at_open'].get('record', '-'):>10s} {v['at_open'].get('roi', 0):+6.1f}% | "
                  f"close {v['at_close'].get('record', '-'):>10s} {v['at_close'].get('roi', 0):+6.1f}% | "
                  f"fade {v['fade_at_close'].get('record', '-'):>10s} {v['fade_at_close'].get('roi', 0):+6.1f}%  n={v['all' if False else 'at_open']['n']}")
    print("\nStrategies (Bet Card):")
    for k, v in results["strategies_bet_card"].items():
        print(f"  {k:55s} {v.get('record', '-'):>10s} {v.get('units', 0):+7.1f}u {v.get('roi', 0):+6.1f}%")
    print("\nBy season:", json.dumps(results["bet_card_by_season"]))
    print(f"\nSaved {OUT_JSON} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    main(ap.parse_args().years)
