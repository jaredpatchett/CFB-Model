#!/usr/bin/env python3
"""
Props weather lab (added 11/2026): does weather change how the official
prop plays do?

Takes the current props models' past-season bets (docs/data/
injury_lab_detail.csv, version "current" -- the same graded rows behind the
official-rule records), joins each one to that game's actual weather from
CFBD (/games/weather, by game id), and splits the three official rules by
wind, rain/snow and cold:

  rec yds overs     15-30% edge, line 25.5+   (wind/rain should HURT these)
  rush yds unders   0%+ edge vs a tough run D (bad weather = more runs, could hurt)
  receptions unders 30%+ edge                 (bad weather = fewer catches, could help)

Each rule's every-edge version is shown too, since the official slices are
small once split by weather.

Outputs docs/data/props_weather_lab.json. Nothing live changes.

Usage (after fetch_historical_data.py, which saves weather_{year}.csv):
  python scripts/props_weather_lab.py
"""
import glob
import json
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import numpy as np
import pandas as pd

import config
from src.analysis import game_profile as gp

SRC = "docs/data/injury_lab_detail.csv"
OUT = "docs/data/props_weather_lab.json"

RULES = {
    "rec_yds_overs": {
        "stat": "rec_yds", "label": "Rec yds overs",
        "official": lambda d: (d.ev >= 15) & (d.ev < 30) & (d.line >= 25.5),
        "wide": lambda d: d.ev >= 0,
    },
    "rush_yds_unders": {
        "stat": "rush_yds", "label": "Rush yds unders",
        "official": lambda d: (d.ev >= 0) & (d.run_def == "tough"),
        "wide": lambda d: d.ev >= 0,
    },
    "receptions_unders": {
        "stat": "receptions", "label": "Receptions unders",
        "official": lambda d: d.ev >= 30,
        "wide": lambda d: d.ev >= 0,
    },
}


def load_weather() -> pd.DataFrame:
    frames = []
    for f in glob.glob(f"{config.DATA_RAW_DIR}/weather_*.csv"):
        w = pd.read_csv(f)
        w = w.rename(columns={"windSpeed": "wind_speed", "gameIndoors": "game_indoors"})
        keep = [c for c in ("id", "temperature", "wind_speed", "precipitation", "snowfall", "game_indoors",
                            "weatherCondition") if c in w.columns]
        frames.append(w[keep])
    if not frames:
        raise SystemExit("No weather_*.csv in data/raw -- run fetch_historical_data.py first.")
    w = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    indoor = w["game_indoors"].astype(str).str.lower() == "true"
    w["indoors"] = indoor
    for c in ("wind_speed", "precipitation", "snowfall", "temperature"):
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")
    w.loc[indoor, ["wind_speed", "precipitation"]] = 0.0
    if "snowfall" in w.columns:
        w.loc[indoor, "snowfall"] = 0.0
    return w


def weather_slices(d: pd.DataFrame, tag: str) -> dict:
    out = d[~d.indoors]
    ws = out.wind_speed
    wet = (out.precipitation.fillna(0) > 0) | (out.get("snowfall", pd.Series(0, index=out.index)).fillna(0) > 0)
    calm_dry = out[(ws < 15) & ~wet & (out.temperature >= 35)]
    return {
        "all": gp.summarize(d, f"{tag}: all"),
        "indoors": gp.summarize(d[d.indoors], f"{tag}: indoors"),
        "wind_under_10": gp.summarize(out[ws < 10], f"{tag}: wind <10"),
        "wind_10_15": gp.summarize(out[(ws >= 10) & (ws < 15)], f"{tag}: wind 10-15"),
        "wind_15_plus": gp.summarize(out[ws >= 15], f"{tag}: wind 15+"),
        "rain_snow": gp.summarize(out[wet], f"{tag}: rain/snow"),
        "cold_under_35F": gp.summarize(out[out.temperature < 35], f"{tag}: under 35F"),
        "any_bad_weather": gp.summarize(out[(ws >= 15) | wet | (out.temperature < 35)], f"{tag}: any bad weather"),
        "good_weather": gp.summarize(pd.concat([d[d.indoors], calm_dry]), f"{tag}: good weather/indoors"),
    }


def main():
    d = pd.read_csv(SRC, low_memory=False)
    d = d[d["version"] == "current"].copy()
    d["gameId"] = pd.to_numeric(d["gameId"], errors="coerce")
    w = load_weather()
    j = d.merge(w, left_on="gameId", right_on="id", how="left")
    matched = j["indoors"].notna()
    print(f"  {int(matched.sum())} of {len(j)} graded props matched to game weather")
    j = j[matched].copy()
    j["indoors"] = j["indoors"].astype(bool)
    results = {}
    for key, r in RULES.items():
        s = j[j.stat == r["stat"]]
        results[key] = {
            "official": weather_slices(s[r["official"](s)], f"{r['label']} (official)"),
            "every_edge": weather_slices(s[r["wide"](s)], f"{r['label']} (every 0%+ edge)"),
        }
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "source": SRC,
           "graded_props_matched": int(len(j)), "results": results,
           "notes": "actual game-day weather from CFBD; indoors = dome games; wind in mph; -price as bet"}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    for key, v in results.items():
        gp.print_rows(f"{key} -- official", list(v["official"].values()))
        gp.print_rows(f"{key} -- every 0%+ edge", list(v["every_edge"].values()))
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
