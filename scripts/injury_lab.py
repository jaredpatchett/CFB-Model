#!/usr/bin/env python3
"""
Injury lab (11/2026): does telling the props models who is OUT make the
official prop plays more profitable?

The live props models have no injury input. When a lead back or a top
receiver goes down, a player's bigger role shows up only in his last-2-game
inputs, while his season inputs still average in the games the teammate
played (FSU's Amari Thomas, 11/2026: 14% of carries on the season, 23% since
the lead back went out; the model projected 29 vs 49.5 a game since).

Versions (each trained only on seasons before the one graded):
  current   the live model's inputs
  injury    + vac_<share>, <share>_without, n_without_<share> for carries
            and catches (src/features/injury_features.py)

Official rules graded (best-edge book, real prices):
  rush yds    under, edge 0%+, 2+ games, vs a tough run defense (also shown:
              every rush under at 0%+). Lines 2024-26
              (data/props_history/rush_yds_<s>.csv). Live rush model = the
              box-score model (PlayerStatModel.fit), as in rush_lab.py.
  receptions  under, edge 30%+. Lines 2024-25 (props_history) + 2026
              (saved board). Live model = PFF version + played-zeros.
  rec yds     over, edge 15-30%, line 25.5+. 2026 lines only (no past-
              season receiving-yards lines pulled).

Also: projection error on the latest full season for every player-game, and
for the ones where a regular teammate was missing (where the inputs can
matter), so the lines aren't the only read.

In training, "missing" = no box-score row that game; live it would be the
injury report. Results only -- nothing here changes the live model.

Outputs: docs/data/injury_lab.json, docs/data/injury_lab_detail.csv
Usage (CFBD + PFF keys, from the workflow):
  PLAYED_ZEROS=1 python scripts/injury_lab.py --years 2022 2023 2024 2025 --season 2026
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import backtest_props_live as bpl
import props_lab as lab
import props_lab2 as lab2
import rush_lab
from src.features.injury_features import add_absence_features, INJURY_FEATURE_COLUMNS, ALL_INJURY_COLUMNS
from src.features.live_player_features import _normalize_name, MIN_GAMES_FOR_PROP_MODEL
from src.features.player_features import ROLLING_FEATURE_COLUMNS
from src.features.run_defense import run_defense, bucket as matchup_bucket
from src.models.props_model import PlayerStatModel

HIST_DIR = "data/props_history"
OUT_JSON = "docs/data/injury_lab.json"
OUT_CSV = "docs/data/injury_lab_detail.csv"


def rec(df):
    if df is None or len(df) == 0:
        return {"n": 0}
    w, l = int((df["result"] == "W").sum()), int((df["result"] == "L").sum())
    u = float(df["units"].sum())
    return {"n": int(len(df)), "record": f"{w}-{l}", "units": round(u, 1), "roi": round(u / len(df) * 100, 1)}


def mae_block(model_pred, actual, vac):
    err = np.abs(np.asarray(model_pred, float) - np.asarray(actual, float))
    vac = np.asarray(vac, float) > 0
    return {"all": {"n": int(len(err)), "mae": round(float(err.mean()), 3)},
            "teammate_missing": {"n": int(vac.sum()), "mae": round(float(err[vac].mean()), 3) if vac.any() else None},
            "no_one_missing": {"n": int((~vac).sum()), "mae": round(float(err[~vac].mean()), 3) if (~vac).any() else None}}


# ------------------------------------------------------------ rush yds ----
def rush_part(years, season):
    print("\n===== Rush yards =====")
    fb, _fz, wide, info = rush_lab.frames(years, season)
    t = time.time()
    fb = add_absence_features(fb)
    print(f"  absence inputs built in {time.time() - t:.0f}s; rows with a regular rusher missing: "
          f"{int((fb['vac_carry_share'] > 0).sum())} of {len(fb)}")
    rd = run_defense(wide)
    seasons = [s for s in (2024, 2025, season) if os.path.exists(f"{HIST_DIR}/rush_yds_{s}.csv")]
    cols = {"current": list(ROLLING_FEATURE_COLUMNS), "injury": list(ROLLING_FEATURE_COLUMNS) + INJURY_FEATURE_COLUMNS["carry_share"]}
    rows, err = [], {}
    for s in seasons:
        raw = pd.read_csv(f"{HIST_DIR}/rush_yds_{s}.csv").dropna(subset=["event_id", "player", "line", "commence_time"])
        for name, c in cols.items():
            m = PlayerStatModel("rush_yds")
            m.feature_columns = list(c)
            m.fit(fb[fb["season"] < s], verbose=False)
            r, skipped = rush_lab.season_rows(name, m, fb, raw, s)
            for x in r:
                x["stat"] = "rush_yds"
            rows += r
            print(f"  {s} {name}: {len(r)} unders priced, skipped {skipped}")
    # Projection error on the last full season (trained before it).
    last = max(years)
    ho = fb[(fb["season"] == last) & (fb["games_played_prior"] >= MIN_GAMES_FOR_PROP_MODEL) & (fb["roll_rush_att"] >= 3)]
    for name, c in cols.items():
        m = PlayerStatModel("rush_yds")
        m.feature_columns = list(c)
        m.fit(fb[fb["season"] < last], verbose=False)
        h = ho.dropna(subset=m.feature_columns)
        err[name] = mae_block(m.predict(h), h["rush_yds"], h["vac_carry_share"])
    det = pd.DataFrame(rows)
    if det.empty:
        return det, err
    det["week"] = pd.to_numeric(det["week"], errors="coerce")
    det = det.merge(rd, on=["season", "week", "opponent"], how="left")
    det["run_def"] = det["run_def_tough"].map(matchup_bucket)
    # Attach the absence inputs for the graded game.
    det = det.merge(fb[["gameId", "player", "vac_carry_share", "n_without_carry_share"]].drop_duplicates(["gameId", "player"]),
                    on=["gameId", "player"], how="left")
    return det, err


# ------------------------------------------------------ receiving stats ----
def ev_side(psm, pred, rows, actual, side):
    """Best-edge book for one side, at that book's line and price."""
    best = None
    for r in rows:
        price = r.get(side)
        if price is None or (isinstance(price, float) and np.isnan(price)):
            continue
        p_over = psm.over_probability(pred, float(r["line"]))
        if p_over is None:
            continue
        p = p_over if side == "over" else 1 - p_over
        ev = p * bpl.payout(float(price)) - (1 - p)
        if best is None or ev > best["ev"]:
            line = float(r["line"])
            res = "P" if actual == line else ("W" if (actual > line) == (side == "over") else "L")
            best = {"ev": ev * 100, "line": line, "price": float(price), "book": r.get("book"), "result": res,
                    "units": bpl.payout(float(price)) if res == "W" else (-1.0 if res == "L" else 0.0)}
    return best


def history_props(feats, stat, seasons):
    """Past-season lines from props_history, matched to box-score rows."""
    out = {}
    for s in seasons:
        path = f"{HIST_DIR}/{stat}_{s}.csv"
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path).dropna(subset=["event_id", "player", "line", "commence_time"])
        idx = rush_lab.by_name(feats[feats["season"] == s])
        props = []
        for (_, player), g in raw.groupby(["event_id", "player"]):
            m = bpl.match_actual({"player": player, "start_time": g["commence_time"].iloc[0]}, idx)
            if m is None:
                continue
            props.append({"idx": m["idx"], "stat": stat, "player": player,
                          "rows": [{"line": float(r.line), "over": r.over, "under": r.under, "book": r.book} for r in g.itertuples()]})
        out[s] = props
    return out


def receiving_part(years, season):
    print("\n===== Receptions / receiving yards (PFF version, played-zeros) =====")
    feats, info, _ = lab2.build_frame(years, season)
    t = time.time()
    feats = add_absence_features(feats)
    print(f"  absence inputs built in {time.time() - t:.0f}s; rows with a regular receiver missing: "
          f"{int((feats['vac_rec_share'] > 0).sum())} of {len(feats)}")
    live = list(lab2.BASE) + list(lab2.PFFC)
    cols = {"current": live, "injury": live + INJURY_FEATURE_COLUMNS["rec_share"]}
    ok = feats[sorted(set(live + ALL_INJURY_COLUMNS) - {"carry_share_without", "rec_share_without"})].notna().all(axis=1)
    pop = ok & (feats["games_played_prior"] >= 1) & (feats["roll_receptions"] > 0)
    usable = set(feats.index[pop])

    def model_for(stat, c, s):
        """Trained before s; hit chances from a model trained before s-1, graded on s-1 (forward, like the live probe)."""
        tr = feats[pop & (feats["season"] < s)].dropna(subset=c + [stat])
        prev = feats[pop & (feats["season"] < s - 1)].dropna(subset=c + [stat])
        ho = feats[pop & (feats["season"] == s - 1)].dropna(subset=c + [stat])
        probe = lab.fit_predict(prev, c, stat)
        gbm = lab.fit_predict(tr, c, stat)
        return lab.with_pools(gbm, c, stat, lab.predict(probe, ho, c), ho[stat]), gbm

    current = lab.collect_props(feats, season)
    sets = {"receptions": history_props(feats, "receptions", [s for s in (2024, 2025) if s in years or s < season]),
            "rec_yds": {}}
    sets["receptions"][season] = [p for p in current if p["stat"] == "receptions"]
    sets["rec_yds"][season] = [p for p in current if p["stat"] == "rec_yds"]
    rows, err = [], {}
    for stat, by_season in sets.items():
        side = "under" if stat == "receptions" else "over"
        for s, props in sorted(by_season.items()):
            props = [p for p in props if p["idx"] in usable]
            if not props:
                continue
            X = feats.loc[[p["idx"] for p in props]]
            for name, c in cols.items():
                psm, gbm = model_for(stat, c, s)
                preds = lab.predict(gbm, X, c)
                n = 0
                for p, pred, (_, row) in zip(props, preds, X.iterrows()):
                    if int(row["games_played_prior"]) < MIN_GAMES_FOR_PROP_MODEL:
                        continue
                    lean = "over" if pred > np.median([r["line"] for r in p["rows"]]) else "under"
                    if lean != side:
                        continue
                    b = ev_side(psm, float(pred), p["rows"], float(row[stat]), side)
                    if b is None:
                        continue
                    n += 1
                    rows.append(dict(b, version=name, stat=stat, season=s, player=p["player"], team=row["team"],
                                     gameId=int(row["gameId"]), pred=round(float(pred), 2), actual=float(row[stat]),
                                     vac_rec_share=row["vac_rec_share"], n_without_rec_share=row["n_without_rec_share"]))
                print(f"  {stat} {s} {name}: {n} {side}s priced")
    last = max(years)
    ho = feats[pop & (feats["season"] == last) & (feats["games_played_prior"] >= 2) & (feats["roll_receptions"] >= lab2.ROLE_MIN_CATCHES)]
    for stat in ("receptions", "rec_yds"):
        err[stat] = {}
        for name, c in cols.items():
            h = ho.dropna(subset=c + [stat])
            gbm = lab.fit_predict(feats[pop & (feats["season"] < last)].dropna(subset=c + [stat]), c, stat)
            err[stat][name] = mae_block(lab.predict(gbm, h, c), h[stat], h["vac_rec_share"])
    return pd.DataFrame(rows), err


# ----------------------------------------------------------------- main ----
def summarize(det, rule, label):
    out = {}
    for v, g in det.groupby("version"):
        b = g[rule(g)]
        out[v] = {"official": rec(b), "official_teammate_missing": rec(b[b["vac"] > 0]),
                  "official_no_one_missing": rec(b[~(b["vac"] > 0)]),
                  "by_season": {int(s): rec(x) for s, x in b.groupby("season")}}
    # The same lines bet by both versions, and the ones only one bets.
    k = ["season", "gameId", "player"]
    sets = {v: set(map(tuple, g[rule(g)][k].values)) for v, g in det.groupby("version")}
    if {"current", "injury"} <= set(sets):
        both = sets["current"] & sets["injury"]
        for v, g in det.groupby("version"):
            b = g[rule(g)]
            keys = [tuple(x) for x in b[k].values]
            out[v]["only_this_version"] = rec(b[[x not in both for x in keys]])
    print(f"\n{label}:")
    for v, r in out.items():
        o, m = r["official"], r["official_teammate_missing"]
        print(f"  {v:8s} {o.get('record')} {o.get('units', 0):+}u {o.get('roi', 0):+}% (n={o['n']}) | "
              f"teammate missing: {m.get('record')} {m.get('units', 0):+}u (n={m['n']}) | only this version: {r.get('only_this_version')}")
        for s, x in r["by_season"].items():
            print(f"      {s}: {x.get('record')} {x.get('units', 0):+}u {x.get('roi', 0):+}% (n={x['n']})")
    return out


def main(years, season):
    t0 = time.time()
    results, details = {}, []
    rush, rush_err = rush_part(years, season)
    if not rush.empty:
        rush["vac"] = rush["vac_carry_share"].fillna(0)
        rush["edge"] = rush["ev"]
        results["rush_yds"] = {
            "rule": "under, edge 0%+, 2+ games, vs a tough run defense",
            "official": summarize(rush, lambda g: (g["ev"] >= 0) & (g["run_def"] == "tough"), "Rush yds unders vs tough run D (official)"),
            "all_unders_0plus": summarize(rush, lambda g: g["ev"] >= 0, "Rush yds unders, any run D, edge 0%+"),
            "projection_error_last_season": rush_err}
        details.append(rush.assign(stat="rush_yds"))
    recv, recv_err = receiving_part(years, season)
    if not recv.empty:
        recv["vac"] = recv["vac_rec_share"].fillna(0)
        r = recv[recv["stat"] == "receptions"]
        y = recv[recv["stat"] == "rec_yds"]
        results["receptions"] = {"rule": "under, edge 30%+",
                                 "official": summarize(r, lambda g: g["ev"] >= 30, "Receptions unders 30%+ (official)") if len(r) else {},
                                 "projection_error_last_season": recv_err.get("receptions")}
        results["rec_yds"] = {"rule": "over, edge 15-30%, line 25.5+ (2026 lines only)",
                              "official": summarize(y, lambda g: (g["ev"] >= 15) & (g["ev"] < 30) & (g["line"] >= 25.5), "Rec yds overs 15-30%, 25.5+ (official)") if len(y) else {},
                              "projection_error_last_season": recv_err.get("rec_yds")}
        details.append(recv)
    print("\nProjection error, last full season (lower is better):")
    print(json.dumps({"rush_yds": rush_err, **recv_err}, indent=1, default=str))
    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "train_years": list(years), "season": season,
           "note": "In training, a teammate is 'missing' when he has no box-score row; live it would be the injury report.",
           "results": results}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    if details:
        pd.concat(details, ignore_index=True).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_JSON} and {OUT_CSV} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(a.years, a.season)
