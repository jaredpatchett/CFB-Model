#!/usr/bin/env python3
"""
Rush lab (11/2026): is the rush yards model more profitable trained the way
it is now (box scores only) or with played-zero games added (a player PFF
shows running routes but with no CFBD box-score line -- the fix the
receiving models got on 11/2/2026)?

Both versions are the live PlayerStatModel("rush_yds"), trained exactly as
train_props_model.py trains it (random 80/20 split for the hit-chance
spread), only on seasons BEFORE the one being graded:
  2024 lines -> trained on 2022-23
  2025 lines -> trained on 2022-24
  2026 lines -> trained on 2022-25 (what the live model is trained on)

Lines: data/props_history/rush_yds_<season>.csv (pull_props_history.py
--market rush_yds), read 2 hours before kickoff, every book. Each player's
best-EV under across books is taken, like the live board, and the official
rule is applied: rush yards under, model edge 0%+, 2+ games played.

Grading: the player's real rushing yards. A player with no box-score line
counts as 0 if PFF has him playing (zero row), and the line is skipped
(void) if not. The same truth is used for both versions; only the model
differs. Lines are graded for a version only where it can price the player
(the box-only version has no row for a played-zero game -- counted and
reported).

Outputs (model results only; no PFF numbers):
  docs/data/rush_lab.json, docs/data/rush_lab_detail.csv, printed report.

Usage (CFBD and PFF keys, from the workflow):
  python scripts/rush_lab.py --years 2022 2023 2024 2025 --season 2026
"""
import argparse
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import backtest_props_live as bpl
import props_lab as lab
from src.data import pff_client as pff
from src.features import pff_features as pf
from src.features.live_player_features import _normalize_name, MIN_GAMES_FOR_PROP_MODEL
from src.features.player_features import (pivot_player_game_stats, add_usage_features, attach_game_environment,
                                          build_rolling_player_features)
from src.models import fair_odds as fo
from src.models.props_model import PlayerStatModel

STAT = "rush_yds"
HIST_DIR = "data/props_history"
OUT_JSON = "docs/data/rush_lab.json"
OUT_CSV = "docs/data/rush_lab_detail.csv"
BUCKETS = [(0, 10), (10, 20), (20, 30), (30, 999)]


def _ints(df):
    for c in ("athleteId", "gameId", "season", "week"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["athleteId", "gameId"])
    for c in ("athleteId", "gameId"):
        df[c] = df[c].astype("int64")
    return df


def frames(years, season):
    print("Loading CFBD box scores...")
    g_h, s_h, l_h = lab.load_history(years)
    g_n, s_n, l_n = lab.load_current(season)
    games = pd.concat([g_h, g_n], ignore_index=True)
    games = games.assign(id=pd.to_numeric(games["id"], errors="coerce")).dropna(subset=["id"])
    games["id"] = games["id"].astype("int64")
    lines = pd.concat([x for x in (l_h, l_n) if x is not None and not x.empty], ignore_index=True)
    if not lines.empty and "id" in lines.columns:
        lines = lines.assign(id=pd.to_numeric(lines["id"], errors="coerce")).dropna(subset=["id"])
        lines["id"] = lines["id"].astype("int64")
    wide = _ints(pd.concat([pivot_player_game_stats(s_h, g_h), pivot_player_game_stats(s_n, g_n)], ignore_index=True))
    print(f"  {len(wide)} player-games")

    print("PFF receiving data for the played-zero rows (kept outside the repo)...")
    out_dir = os.environ.get("PFF_DATA_DIR") or os.path.join(tempfile.gettempdir(), "pff_rush_lab")
    recv, pgames = pff.fetch_receiving_seasons(list(years) + [season], out_dir)
    zeros = pf.played_zero_rows(wide, games, pf.prepare_pff_rows(recv, pgames))
    print(f"  played-zero rows: {len(zeros)} ({len(zeros) / max(len(wide), 1) * 100:.1f}%)")

    starts = games.drop_duplicates("id").set_index("id")["startDate"]

    def build(w):
        f = build_rolling_player_features(attach_game_environment(add_usage_features(w), lines)).reset_index(drop=True)
        f["start"] = f["gameId"].map(starts)
        f[STAT] = pd.to_numeric(f[STAT], errors="coerce").fillna(0.0)
        return f

    fb = build(wide)
    fz = build(pd.concat([wide, zeros], ignore_index=True).sort_values(["athleteId", "season", "week"]))
    zkeys = set(zip(zeros["gameId"], zeros["athleteId"]))
    fz["zero_row"] = [(g, a) in zkeys for g, a in zip(fz["gameId"], fz["athleteId"])]
    fb["zero_row"] = False
    return fb, fz, wide, {"player_games": int(len(wide)), "played_zero_rows": int(len(zeros))}


from src.features.run_defense import run_defense, bucket as matchup_bucket   # shared with the live board (11/3/2026)


def train(frame, before):
    m = PlayerStatModel(STAT)
    m.fit(frame[frame["season"] < before], verbose=False)
    return m


def by_name(feats):
    out = {}
    for idx, start, player, team in zip(feats.index, feats["start"], feats["player"], feats["team"]):
        try:
            st = bpl._ts(start) if pd.notna(start) else None
        except Exception:
            st = None
        out.setdefault(_normalize_name(player), []).append({"team": team, "start": st, "idx": idx})
    return out


def under_ev(res, price):
    """Same formula as the live board (export_dashboard_data.py)."""
    if res is None or price is None or (isinstance(price, float) and np.isnan(price)):
        return None
    push = res.get("push_probability") or 0.0
    if push > 0 and res.get("under_probability") is not None:
        return (res["under_probability"] * (fo.american_to_decimal(float(price)) - 1) - res["over_probability"]) * 100
    if res.get("over_probability") is None:
        return None
    return fo.ev_percent(1 - res["over_probability"], float(price))


def grade(actual, line, price):
    if actual == line:
        return "P", 0.0
    if actual < line:
        return "W", fo.american_to_decimal(float(price)) - 1
    return "L", -1.0


def season_rows(name, model, feats, raw, season):
    cur = feats[feats["season"] == season]
    idx = by_name(cur)
    out, skipped = [], {"no_match": 0, "few_games": 0, "not_under": 0}
    for (_, player), g in raw.groupby(["event_id", "player"]):
        m = bpl.match_actual({"player": player, "start_time": g["commence_time"].iloc[0]}, idx)
        if m is None:
            skipped["no_match"] += 1
            continue
        row = feats.loc[m["idx"]]
        if int(row["games_played_prior"]) < MIN_GAMES_FOR_PROP_MODEL:
            skipped["few_games"] += 1
            continue
        x = row[model.feature_columns]
        if x.isna().any():
            skipped["no_match"] += 1
            continue
        best = None
        for r in g.itertuples():
            try:
                res = model.predict_and_compare(x, float(r.line))
            except Exception:
                continue
            if res.get("lean") != "under":
                continue
            ev = under_ev(res, r.under)
            if ev is not None and (best is None or ev > best["ev"]):
                best = {"ev": ev, "line": float(r.line), "price": float(r.under), "book": r.book, "pred": res["predicted_value"]}
        if best is None:
            skipped["not_under"] += 1
            continue
        actual = float(row[STAT])
        res_, units = grade(actual, best["line"], best["price"])
        out.append({"version": name, "season": season, "player": player, "team": row["team"], "event_id": g["event_id"].iloc[0],
                    "gameId": int(row["gameId"]), "week": row.get("week"), "opponent": row.get("opponent"),
                    "line": best["line"], "price": best["price"], "book": best["book"], "pred": round(best["pred"], 1),
                    "ev": round(best["ev"], 1), "actual": actual, "zero_row": bool(row.get("zero_row", False)),
                    "result": res_, "units": round(units, 4)})
    return out, skipped


def rec(df):
    if df.empty:
        return {"n": 0}
    w, l = int((df["result"] == "W").sum()), int((df["result"] == "L").sum())
    u = float(df["units"].sum())
    return {"n": int(len(df)), "record": f"{w}-{l}", "units": round(u, 1), "roi": round(u / len(df) * 100, 1)}


def main(years, season):
    t0 = time.time()
    fb, fz, wide, info = frames(years, season)
    rd = run_defense(wide)
    seasons = [s for s in (2024, 2025, season) if os.path.exists(f"{HIST_DIR}/rush_yds_{s}.csv")]
    if not seasons:
        raise SystemExit("No rush yds lines in data/props_history -- run the props history pull with market rush_yds.")
    rows, notes = [], {}
    for s in seasons:
        raw = pd.read_csv(f"{HIST_DIR}/rush_yds_{s}.csv").dropna(subset=["event_id", "player", "line", "commence_time"])
        notes[s] = {"players_with_a_line": int(raw.groupby(["event_id", "player"]).ngroups)}
        for name, feats in (("current", fb), ("played_zeros", fz)):
            model = train(feats, s)
            r, skipped = season_rows(name, model, feats, raw, s)
            rows += r
            notes[s][name] = skipped
            print(f"  {s} {name}: {len(r)} unders priced, skipped {skipped}")
    det = pd.DataFrame(rows)
    det["week"] = pd.to_numeric(det["week"], errors="coerce")
    det = det.merge(rd, on=["season", "week", "opponent"], how="left")
    det["run_def"] = det["run_def_tough"].map(matchup_bucket)
    det["run_def_tough"] = det["run_def_tough"].round(0)
    bets = det[det["ev"] >= 0]
    results = {}
    for v, g in bets.groupby("version"):
        results[v] = {"all": rec(g), "by_season": {int(s): rec(x) for s, x in g.groupby("season")},
                      "by_edge": {f"{lo}-{hi}%": rec(g[(g["ev"] >= lo) & (g["ev"] < hi)]) for lo, hi in BUCKETS},
                      "zero_row_bets": int(g["zero_row"].sum()),
                      # 11/3/2026: opposing run defense (tough = top third, earlier games only)
                      "by_run_defense": {b: rec(x) for b, x in g.groupby("run_def")},
                      "by_run_defense_season": {f"{b} {int(s)}": rec(x) for (b, s), x in g.groupby(["run_def", "season"])}}
    # Common lines: the same player-games priced by both versions.
    k = ["season", "event_id", "player"]
    both = set(map(tuple, bets[bets["version"] == "current"][k].values)) & set(map(tuple, bets[bets["version"] == "played_zeros"][k].values))
    results["common_lines_bet_by_both"] = {v: rec(g[[tuple(x) in both for x in g[k].values]]) for v, g in bets.groupby("version")}
    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "seasons": seasons, "info": info, "lines": notes, "results": results,
           "rule": "rush yards under, best-EV book, edge 0%+, 2+ games played; graded at that book's line and price"}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    det.to_csv(OUT_CSV, index=False)

    print("\nRush yards unders, edge 0%+ (official rule):")
    for v in ("current", "played_zeros"):
        r = results.get(v)
        if not r:
            continue
        a = r["all"]
        print(f"  {v:13s} all {a.get('record')} {a.get('units'):+}u {a.get('roi'):+}% (n={a['n']}, {r['zero_row_bets']} on played-zero games)")
        for s, x in r["by_season"].items():
            print(f"      {s}: {x.get('record')} {x.get('units'):+}u {x.get('roi'):+}% (n={x['n']})")
        for b, x in r["by_edge"].items():
            if x["n"]:
                print(f"      edge {b}: {x.get('record')} {x.get('units'):+}u {x.get('roi'):+}% (n={x['n']})")
        for b, x in sorted(r["by_run_defense_season"].items()):
            if x["n"]:
                print(f"      vs {b}: {x.get('record')} {x.get('units'):+}u {x.get('roi'):+}% (n={x['n']})")
        for b, x in r["by_run_defense"].items():
            print(f"      vs {b} run D, all seasons: {x.get('record')} {x.get('units'):+}u {x.get('roi'):+}% (n={x['n']})")
    print(f"  lines bet by both: {results['common_lines_bet_by_both']}")
    print(f"\nSaved {OUT_JSON} and {OUT_CSV} ({len(det)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(a.years, a.season)
