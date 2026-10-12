#!/usr/bin/env python3
"""
Weather lab (added 10/2026): should the spread model use the weather forecast?

The live model's number for a game moved by up to 7 points from one day to
the next with no game played and no injury change. Its inputs include the
weather (temperature, wind, precipitation, indoors), which comes from a
forecast that updates daily, and a tree model's output can jump when a
forecast crosses one of its split points. This tests, walk-forward on the
same games, whether those inputs earn their place:

  upgraded      the live model
  no_temp       live model without temperature
  no_weather    live model without any weather input

For each it reports accuracy, results against closing and opening spreads
(every game and the Bet Card, spreads under 20), and how far the projected
margin moves when only the weather changes (temperature +/-10 F, rain
on/off) -- the size of the day-to-day swings a changing forecast can cause.

Outputs docs/data/weather_lab.json (model results only).

Usage (after fetch_historical_data.py and build_features.py):
  python scripts/weather_lab.py --years 2022 2023 2024 2025 2026
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

OUT = "docs/data/weather_lab.json"
WEATHER = ["temperature", "wind_speed", "precipitation", "game_indoors"]
VARIANTS = {
    "upgraded": list(FEATURE_COLUMNS_V2),
    "no_temp": [c for c in FEATURE_COLUMNS_V2 if c != "temperature"],
    "no_weather": [c for c in FEATURE_COLUMNS_V2 if c not in WEATHER],
}


def grade(d, mcol, line_col):
    d = d.dropna(subset=[mcol, line_col]).copy()
    d["edge"] = d[mcol] + d[line_col]
    d = d[d["edge"] != 0]
    cover = d["margin"] + d[line_col]
    d["result"] = ["P" if c == 0 else ("W" if (c > 0) == (e > 0) else "L") for c, e in zip(cover, d["edge"])]
    d["units"] = d["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
    return d


def card(d, sd):
    d = d.copy()
    d["cover_prob"] = norm.cdf(d["edge"].abs() / sd)
    d["rank"] = d.groupby(["season", "week"])["cover_prob"].rank(ascending=False, method="first")
    return d[d["rank"] <= 10]


def sensitivity(model, rows):
    """Average and largest change in projected margin when only the weather moves."""
    out = {}
    if rows.empty:
        return out
    base = model.predict_margin(rows)
    outdoor = ~rows["game_indoors"].fillna(False).astype(bool) if "game_indoors" in rows.columns else pd.Series(True, index=rows.index)
    if not outdoor.any():
        return out
    for label, col, fn in (("temp_plus_10F", "temperature", lambda s: s + 10), ("temp_minus_10F", "temperature", lambda s: s - 10),
                           ("rain_on", "precipitation", lambda s: s.clip(lower=0.1)), ("rain_off", "precipitation", lambda s: s * 0)):
        if col not in model.feature_columns:
            continue
        r = rows.copy()
        r.loc[outdoor, col] = fn(r.loc[outdoor, col])
        diff = np.abs(model.predict_margin(r) - base)[outdoor.to_numpy()]
        out[label] = {"avg_pts": round(float(diff.mean()), 2), "share_moved_2plus": round(float((diff >= 2).mean()), 3),
                      "max_pts": round(float(diff.max()), 1)}
    return out


def main(years):
    t0 = time.time()
    fpath = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if not os.path.exists(fpath):
        raise SystemExit("Run scripts/build_features.py first.")
    feats = pd.read_csv(fpath)
    lines = rb.load_historical_lines()
    joined, strategy = rb.join_features_to_lines(feats, lines)
    seasons = sorted(int(x) for x in feats.loc[feats["sp_rating_diff"].notna(), "season"].dropna().unique() if int(x) in years)
    feats = feats[feats["season"].isin(seasons)]
    parts, sens, imps = [], {}, {}
    for test_season in seasons[1:]:
        train = feats[(feats["season"] < test_season) & feats["margin"].notna()]
        test = joined[(joined["season"] == test_season) & joined["margin"].notna()].copy()
        for name, cols in VARIANTS.items():
            m = GameMarginModel()
            m.feature_columns = list(cols)
            try:
                met = m.fit(train, verbose=False)
            except Exception as e:
                print(f"  {test_season} {name}: could not train ({e})")
                continue
            ok = test[m.feature_columns].notna().all(axis=1)
            test[f"m_{name}"] = np.nan
            if ok.any():
                test.loc[ok, f"m_{name}"] = m.predict_margin(test[ok])
            test[f"sd_{name}"] = m.residual_std
            imps[f"{test_season}_{name}"] = {k: round(float(v), 4) for k, v in list(met["feature_importances"].items())
                                             if k in WEATHER}
            if name == "upgraded":
                sens[str(test_season)] = sensitivity(m, test[ok][m.feature_columns])
        parts.append(test)
        print(f"  {test_season}: graded {len(test)} games")
    scored = pd.concat(parts, ignore_index=True)
    names = [n for n in VARIANTS if f"m_{n}" in scored.columns]
    # Same games for every version: all three scored, spread under 20.
    base = scored.dropna(subset=[f"m_{n}" for n in names] + ["market_spread_home"])
    base = base[base["market_spread_home"].abs() < 20]
    results = {}
    for n in names:
        mcol, sd = f"m_{n}", float(base[f"sd_{n}"].mean())
        c = grade(base, mcol, "market_spread_home")
        r = {"margin_mae": round(float((base[mcol] - base["margin"]).abs().mean()), 3),
             "vs_closing": {"every_game": gp.summarize(c, "every game"), "bet_card": gp.summarize(card(c, sd), "bet card"),
                            "edge_3_plus": gp.summarize(c[c.edge.abs() >= 3], "3+ pts off")}}
        if "market_spread_open_home" in base.columns:
            o = grade(base, mcol, "market_spread_open_home")
            r["vs_opening"] = {"every_game": gp.summarize(o, "every game vs opener"), "bet_card": gp.summarize(card(o, sd), "bet card vs opener")}
        results[n] = r
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "seasons_graded": seasons[1:], "games": int(len(base)),
           "join_strategy": strategy, "results": results, "weather_sensitivity_live_model": sens,
           "weather_input_importance": imps,
           "method": "walk-forward; same games for every version; spreads under 20; -110"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    cell = lambda x: f"{x['record']} {x['roi'] * 100:+.1f}%" if x and x.get("n") else "-"
    print(f"\nWEATHER LAB -- {out['games']} games, seasons {out['seasons_graded']}")
    print(f"  {'':<26}" + "".join(f"{n:>22}" for n in names))
    print(f"  {'avg miss (pts)':<26}" + "".join(f"{results[n]['margin_mae']:>22.3f}" for n in names))
    for lab, a, b in (("every game vs close", "vs_closing", "every_game"), ("Bet Card vs close", "vs_closing", "bet_card"),
                      ("every game vs open", "vs_opening", "every_game"), ("Bet Card vs open", "vs_opening", "bet_card")):
        print(f"  {lab:<26}" + "".join(f"{cell((results[n].get(a) or {}).get(b)):>22}" for n in names))
    print("  weather sensitivity (live model):", json.dumps(sens))
    print(f"\nSaved {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    main(sorted(ap.parse_args().years))
