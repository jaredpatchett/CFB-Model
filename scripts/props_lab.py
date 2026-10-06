#!/usr/bin/env python3
"""
Props lab (added 10/2026): does anything beat the current receiving models?

Trains the current receptions and receiving-yards models and several
candidate versions side by side, and grades every one of them the same
strictly forward-in-time way. Nothing here touches the live model, the
dashboard or the official-play rules.

Versions compared
  as_live     the live model exactly as deployed (all player rows, hit
              chances from a random 80/20 split). 2026 lines only -- it is
              the "before" and a check that this script reproduces the
              existing backtest.
  current     the same 30 inputs, but hit chances from a later season
              instead of a random split.
  receivers   same inputs, trained only on players who already have a catch
              that season (closer to the players who get prop lines).
  trimmed     receiving-specific inputs only.
  extras      + role trends, last-3 form, role stability, game total,
              spread size, opponent pass yards per attempt.
  pff         + PFF routes run, route participation, targets, target share,
              targets per route, depth of target, slot rate.
  pff_extras  everything.

How each version is graded
  2025  Train on 2022-2024, project every 2025 game. Projection error and
        bias for players with a real receiving role, and whether the hit
        chances are honest (using misses measured on 2024, at a test line
        set at each player's own average).
  2026  Train on 2022-2025, misses measured on 2025. Every posted receptions
        and receiving-yards line saved in the repo history is scored on BOTH
        sides at its real price, at the first line posted, best book.

Outputs
  docs/data/props_lab.json         summary tables for every version
  docs/data/props_lab_detail.csv   one row per prop per version, both sides

PFF data is pulled into a temp folder outside the repo and is never
committed: the repo is public and PFF's numbers are licensed to the
subscriber. The outputs contain model results only.

Usage (needs full git history, CFBD and PFF keys; run from the workflow):
  python scripts/props_lab.py --years 2022 2023 2024 2025 --season 2026
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import config
import backtest_props_live as bpl
import props_edge_finder as pef
from src.data import cfbd_client as cfbd
from src.data import pff_client as pff
from src.features import pff_features as pf
from src.features.live_player_features import market_name_to_stat, _normalize_name
from src.features.player_features import (
    ROLLING_FEATURE_COLUMNS, pivot_player_game_stats, add_usage_features,
    attach_game_environment, build_rolling_player_features,
)
from src.models.props_model import PlayerStatModel

STATS = ["receptions", "rec_yds"]
OUT_JSON = "docs/data/props_lab.json"
OUT_CSV = "docs/data/props_lab_detail.csv"
MIN_EDGE = 0.10                    # the official-play edge bar
ROLE_MIN_CATCHES = 1.5             # "real receiving role" for the 2025 test
PROB_BINS = [0, .3, .4, .5, .6, .7, 1.01]
PROB_LABELS = ["under 30%", "30-40%", "40-50%", "50-60%", "60-70%", "70%+"]

TRIMMED = ["roll_receptions", "roll_rec_yds", "roll_rec_tds", "roll_rec_share", "roll_rec_yds_share",
           "roll_team_pass_att", "roll_team_rush_att", "last2_receptions", "last2_rec_yds", "last2_rec_share",
           "games_played_prior", "opp_pass_yds_allowed_prior", "team_spread", "team_implied_total"]
EXTRAS = ["last3_receptions", "last3_rec_yds", "last3_rec_share",
          "trend_receptions", "trend_rec_yds", "trend_rec_share",
          "sd_receptions", "sd_rec_yds", "cv_receptions", "roll_yards_per_catch",
          "game_total", "abs_spread", "is_favorite",
          "opp_pass_ypa_allowed_prior", "opp_pass_att_faced_prior"]
BASE = list(ROLLING_FEATURE_COLUMNS)

# name -> (inputs, train on receivers only?, description)
VARIANTS = {
    "current":    (BASE, False, "Current 30 inputs, all players"),
    "receivers":  (BASE, True, "Current inputs, trained on players with a prior catch"),
    "trimmed":    (TRIMMED, True, "Receiving-specific inputs only"),
    "extras":     (BASE + EXTRAS, True, "Current + trends, stability, game total, opponent per-attempt"),
    "pff":        (BASE + pf.PFF_FEATURE_COLUMNS, True, "Current + PFF routes and targets"),
    "pff_extras": (BASE + EXTRAS + pf.PFF_FEATURE_COLUMNS, True, "Everything"),
}
AS_LIVE = "as_live"


# ----------------------------------------------------------------- data ----
def _read(pattern, years):
    frames = [pd.read_csv(pattern.format(year=y)) for y in years if os.path.exists(pattern.format(year=y))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_history(years):
    raw = config.DATA_RAW_DIR
    games = _read(raw + "/games_{year}.csv", years)
    stats = _read(raw + "/player_game_stats_{year}.csv", years)
    lines = _read(raw + "/lines_{year}.csv", years)
    if games.empty or stats.empty:
        raise SystemExit("Historical CFBD files missing -- run scripts/fetch_historical_data.py first.")
    return games, stats, lines


def load_current(season):
    """This season's completed games, box scores and betting lines from CFBD."""
    games = cfbd.get_games(season)
    done = games[games["completed"] == True] if "completed" in games.columns else games  # noqa: E712
    parts = []
    for wk in sorted(int(w) for w in done["week"].dropna().unique()):
        try:
            df = cfbd.get_player_game_stats(season, wk)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  [warn] {season} week {wk} player stats failed: {e}")
    stats = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    try:
        lines = cfbd.historical_lines_to_dataframe(cfbd.get_historical_lines(season))
    except Exception as e:
        print(f"  [warn] {season} betting lines unavailable ({e})")
        lines = pd.DataFrame()
    return games, stats, lines


def add_extra_features(df):
    """Box-score extras, all leakage-safe (shift(1) before every average)."""
    df = df.sort_values(["athleteId", "season", "week"]).copy()
    grp = df.groupby(["athleteId", "season"])
    for c in ("receptions", "rec_yds", "rec_share"):
        df[f"last3_{c}"] = grp[c].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        df[f"trend_{c}"] = df[f"last2_{c}"] - df[f"roll_{c}"]
    for c in ("receptions", "rec_yds"):
        df[f"sd_{c}"] = grp[c].transform(lambda s: s.shift(1).expanding().std()).fillna(0.0)
    df["cv_receptions"] = np.where(df["roll_receptions"] > 0, df["sd_receptions"] / df["roll_receptions"].replace(0, np.nan), 0.0)
    df["roll_yards_per_catch"] = np.where(df["roll_receptions"] > 0, df["roll_rec_yds"] / df["roll_receptions"].replace(0, np.nan), 0.0)
    df["game_total"] = 2 * df["team_implied_total"] + df["team_spread"]
    df["abs_spread"] = df["team_spread"].abs()
    df["is_favorite"] = (df["team_spread"] < 0).astype(float).where(df["team_spread"].notna())

    # Opponent pass defense per attempt, from the opponent's EARLIER games.
    tg = df.groupby(["gameId", "team"]).agg(season=("season", "first"), week=("week", "first"), opponent=("opponent", "first"),
                                             yds=("team_pass_yds", "first"), att=("team_pass_att", "first")).reset_index()
    faced = tg.rename(columns={"opponent": "defense"}).sort_values(["defense", "season", "week"])
    g2 = faced.groupby(["defense", "season"])
    cum_y = g2["yds"].transform(lambda s: s.shift(1).expanding().sum())
    cum_a = g2["att"].transform(lambda s: s.shift(1).expanding().sum())
    faced["opp_pass_ypa_allowed_prior"] = np.where(cum_a > 0, cum_y / cum_a.replace(0, np.nan), np.nan)
    faced["opp_pass_att_faced_prior"] = g2["att"].transform(lambda s: s.shift(1).expanding().mean())
    faced = faced.rename(columns={"defense": "opponent"})[["gameId", "opponent", "opp_pass_ypa_allowed_prior", "opp_pass_att_faced_prior"]]
    return df.merge(faced.drop_duplicates(["gameId", "opponent"]), on=["gameId", "opponent"], how="left")


def build_frame(years, season):
    print("Loading CFBD history and this season...")
    g_hist, s_hist, l_hist = load_history(years)
    g_now, s_now, l_now = load_current(season)
    if s_now.empty:
        raise SystemExit(f"No completed {season} box scores yet.")
    games = pd.concat([g_hist, g_now], ignore_index=True)
    lines = pd.concat([x for x in (l_hist, l_now) if not x.empty], ignore_index=True) if not (l_hist.empty and l_now.empty) else pd.DataFrame()
    wide = pd.concat([pivot_player_game_stats(s_hist, g_hist), pivot_player_game_stats(s_now, g_now)], ignore_index=True)
    # Saved history has numeric ids; the live CFBD pull returns them as text.
    # Make them one type, or sorting a mixed column fails.
    for c in ("athleteId", "gameId", "season", "week"):
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    wide = wide.dropna(subset=["athleteId", "gameId", "season", "week"])
    for c in ("athleteId", "gameId", "season", "week"):
        wide[c] = wide[c].astype("int64")
    games = games.assign(id=pd.to_numeric(games["id"], errors="coerce")).dropna(subset=["id"])
    games["id"] = games["id"].astype("int64")
    if not lines.empty and "id" in lines.columns:
        lines = lines.assign(id=pd.to_numeric(lines["id"], errors="coerce")).dropna(subset=["id"])
        lines["id"] = lines["id"].astype("int64")
    print(f"  {len(wide)} player-games, seasons {sorted(wide['season'].dropna().unique().astype(int))}")
    # Same three steps the live training data goes through.
    feats = build_rolling_player_features(attach_game_environment(add_usage_features(wide), lines))
    feats = add_extra_features(feats)

    print("Pulling PFF receiving data (kept outside the repo)...")
    out_dir = os.environ.get("PFF_DATA_DIR") or os.path.join(tempfile.gettempdir(), "pff_lab")
    recv, pgames = pff.fetch_receiving_seasons(list(years) + [season], out_dir)
    if recv.empty:
        raise SystemExit("PFF returned no receiving rows -- check the PFF key.")
    matched, info = pf.match_pff_to_cfbd(feats, games, pf.prepare_pff_rows(recv, pgames))
    feats = pf.add_pff_features(feats, matched)
    keys = set(zip(matched["gameId"], matched["athleteId"]))
    catchers = feats[feats["receptions"] > 0]
    hit = pd.Series([(g, a) in keys for g, a in zip(catchers["gameId"], catchers["athleteId"])], index=catchers.index)
    info["pct_catchers_matched_by_season"] = {int(k): round(float(v) * 100, 1) for k, v in hit.groupby(catchers["season"]).mean().items()}
    print(f"  PFF match: {info}")
    starts = games.drop_duplicates("id").set_index("id")["startDate"] if "startDate" in games.columns else pd.Series(dtype=str)
    feats["start"] = feats["gameId"].map(starts)
    return feats.reset_index(drop=True), info


# --------------------------------------------------------------- models ----
def new_gbm():
    # Same settings as the live PlayerStatModel.
    return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)


def fit_predict(train, cols, stat):
    m = new_gbm().fit(train[cols], train[stat])
    return m


def predict(m, rows, cols):
    return np.clip(m.predict(rows[cols]), 0, None)


def with_pools(gbm, cols, stat, preds, actuals):
    """A PlayerStatModel whose hit chances come from the given out-of-sample
    misses (same bucket-by-projection-size math the live model uses)."""
    psm = PlayerStatModel(stat)
    psm.model, psm.feature_columns = gbm, list(cols)
    resid = np.asarray(actuals, float) - np.asarray(preds, float)
    psm.residual_std = float(np.std(resid))
    psm._fit_residual_bins(np.asarray(preds, float), resid)
    return psm


def calibration(p, hit):
    p, hit = np.asarray(p, float), np.asarray(hit, float)
    if len(p) == 0:
        return {"n": 0}
    b = pd.cut(p, PROB_BINS, labels=PROB_LABELS, right=False)
    rows = [{"bucket": lab, "n": int((b == lab).sum()), "model": round(float(p[b == lab].mean()) * 100, 1),
             "actual": round(float(hit[b == lab].mean()) * 100, 1)} for lab in PROB_LABELS if (b == lab).sum() > 0]
    gap = sum(r["n"] * abs(r["model"] - r["actual"]) for r in rows) / max(sum(r["n"] for r in rows), 1)
    return {"n": int(len(p)), "brier": round(float(np.mean((p - hit) ** 2)), 4), "avg_gap_pts": round(gap, 1),
            "model_avg_over": round(float(p.mean()) * 100, 1), "actual_over": round(float(hit.mean()) * 100, 1), "buckets": rows}


def record(df, side):
    """Record / units / ROI for one side's bets in df (uses <side>_res, _units, _price)."""
    res, units, price = df[f"{side}_res"], df[f"{side}_units"].astype(float), df[f"{side}_price"].astype(float)
    n, w, l = len(df), int((res == "W").sum()), int((res == "L").sum())
    if n == 0:
        return {"n": 0}
    be = np.where(price < 0, -price / (-price + 100), 100 / (price.abs() + 100))
    out = {"n": n, "record": f"{w}-{l}", "hit": round(w / max(w + l, 1) * 100, 1), "needs": round(float(np.mean(be)) * 100, 1),
           "units": round(float(units.sum()), 1), "roi": round(float(units.sum()) / n * 100, 1)}
    if "weekend" in df.columns:
        out["roi_by_weekend"] = {k: round(float(g[f"{side}_units"].astype(float).sum()) / len(g) * 100, 1) for k, g in df.groupby("weekend")}
    return out


# ----------------------------------------------------------------- 2026 ----
def collect_props(feats, season):
    """Every saved receptions / receiving-yards prop this season, matched to
    the player's box score and pre-game feature row."""
    markets = pef.collect_markets(f"{season}-08-01")
    cur = feats[feats["season"] == season]
    by_name = {}
    for idx, row in cur.iterrows():
        try:
            st = bpl._ts(row["start"]) if pd.notna(row["start"]) else None
        except Exception:
            st = None
        by_name.setdefault(_normalize_name(row["player"]), []).append({"team": row["team"], "start": st, "row": row, "idx": idx})
    props = []
    for key, rec in markets.items():
        meta = rec["meta"]
        stat = market_name_to_stat(meta["market"])
        if stat not in STATS:
            continue
        m = bpl.match_actual(meta, by_name)
        if m is None:
            continue
        kick = pd.Timestamp(bpl._ts(meta["start_time"])) - pd.Timedelta(hours=5)
        sat = (kick + pd.Timedelta(days={0: -2, 1: -3, 2: 3, 3: 2, 4: 1, 5: 0, 6: -1}[kick.weekday()])).date()
        rows = [r for r in rec["first"] if r.get("line") is not None]
        if rows:
            props.append({"idx": m["idx"], "stat": stat, "player": meta["player"], "market": meta["market"],
                          "weekend": str(sat), "rows": rows})
    return props


def score_prop(psm, pred, actual, rows):
    """Both sides at the real prices: for each side, the book/line with the
    best edge. Also the side the projection points to (today's rule)."""
    out = {}
    for r in rows:
        p_over = psm.over_probability(pred, r["line"])
        if p_over is None:
            continue
        for side, price, p in (("over", r.get("over"), p_over), ("under", r.get("under"), 1 - p_over)):
            if price is None or (isinstance(price, float) and math.isnan(price)):
                continue
            ev = p * bpl.payout(price) - (1 - p)
            if side not in out or ev > out[side]["ev"]:
                res = "P" if actual == r["line"] else ("W" if (actual > r["line"]) == (side == "over") else "L")
                out[side] = {"ev": ev, "p": p, "price": float(price), "line": r["line"], "book": r["book"], "res": res,
                             "units": bpl.payout(price) if res == "W" else (-1.0 if res == "L" else 0.0)}
    return out


def evaluate_props(name, psm, cols, stat, feats, props, usable):
    rows = []
    sub = [p for p in props if p["stat"] == stat and p["idx"] in usable]
    if not sub:
        return rows
    X = feats.loc[[p["idx"] for p in sub]]
    preds = predict(psm.model, X, cols)
    for p, pred, (_, frow) in zip(sub, preds, X.iterrows()):
        actual = float(frow[stat])
        sides = score_prop(psm, float(pred), actual, p["rows"])
        if not sides:
            continue
        ref = p["rows"][0]
        rec = {"version": name, "stat": stat, "player": p["player"], "market": p["market"], "weekend": p["weekend"],
               "team": frow.get("team"), "opponent": frow.get("opponent"), "gp": int(frow["games_played_prior"]),
               "rec_share": frow.get("roll_rec_share"), "team_spread": frow.get("team_spread"),
               "pred": round(float(pred), 2), "actual": actual, "ref_line": ref["line"],
               "ref_p_over": psm.over_probability(float(pred), ref["line"]),
               "proj_side": "over" if pred > ref["line"] else "under"}
        for side in ("over", "under"):
            s = sides.get(side)
            for k in ("ev", "p", "price", "line", "book", "res", "units"):
                rec[f"{side}_{k}"] = s[k] if s else None
        rows.append(rec)
    return rows


def summarize_props(d):
    """Summary of one version's props for one stat."""
    if d.empty:
        return {"n": 0}
    out = {"n": int(len(d)), "mae": round(float((d.pred - d.actual).abs().mean()), 2),
           "bias": round(float((d.pred - d.actual).mean()), 2),
           "line_mae": round(float((d.ref_line - d.actual).abs().mean()), 2)}
    c = d[d.ref_p_over.notna() & (d.actual != d.ref_line)]
    out["calibration"] = calibration(c.ref_p_over, (c.actual > c.ref_line).astype(float))
    for side in ("over", "under"):
        s = d[d[f"{side}_ev"].notna()]
        out[f"{side}_all"] = record(s, side)
        out[f"{side}_edge10"] = record(s[s[f"{side}_ev"] >= MIN_EDGE], side)
        out[f"{side}_edge10_gp2"] = record(s[(s[f"{side}_ev"] >= MIN_EDGE) & (s.gp >= 2)], side)
    # Today's rule: bet the side the projection points to, at 10%+ edge.
    for side in ("over", "under"):
        s = d[(d.proj_side == side) & (d[f"{side}_ev"] >= MIN_EDGE)]
        out[f"proj_{side}_edge10"] = record(s, side)
    return out


# ----------------------------------------------------------------- main ----
def main(years, season):
    t0 = time.time()
    feats, match_info = build_frame(years, season)
    last, prev = max(years), max(years) - 1          # 2025, 2024
    all_cols = sorted({c for cols, _, _ in VARIANTS.values() for c in cols})
    missing = [c for c in all_cols if c not in feats.columns]
    if missing:
        raise SystemExit(f"Feature columns missing from the frame: {missing}")
    props = collect_props(feats, season)
    print(f"Props matched to box scores: {len(props)} ({sum(p['stat'] == 'receptions' for p in props)} receptions, "
          f"{sum(p['stat'] == 'rec_yds' for p in props)} receiving yards)")

    results = {s: {} for s in STATS}
    detail = []
    for stat in STATS:
        ok = feats[all_cols + [stat]].notna().all(axis=1) & (feats["games_played_prior"] >= 1)
        receiver = feats["roll_receptions"] > 0
        usable = set(feats.index[ok])                     # props every version can score
        test = feats[ok & (feats.season == last) & (feats.games_played_prior >= 2) & (feats.roll_receptions >= ROLE_MIN_CATCHES)]
        test_line = np.floor(test[f"roll_{stat}"]) + 0.5
        naive_mae = float((test[f"roll_{stat}"] - test[stat]).abs().mean()) if len(test) else None
        print(f"\n===== {stat}: {int(ok.sum())} usable player-games, {len(test)} in the {last} role-player test =====")

        for name, (cols, receivers_only, desc) in VARIANTS.items():
            t1 = time.time()
            pop = ok & (receiver if receivers_only else True)
            tr0, ho0 = feats[pop & (feats.season < prev)], feats[pop & (feats.season == prev)]
            trA, hoA = feats[pop & (feats.season < last)], feats[pop & (feats.season == last)]
            trB = feats[pop & (feats.season <= last)]
            if min(len(tr0), len(ho0), len(trA), len(hoA)) < 200:
                print(f"  {name}: not enough rows, skipped")
                continue
            # 2025 test: model from 2022-2024, misses measured on 2024 by a 2022-2023 model.
            g0 = fit_predict(tr0, cols, stat)
            gA = fit_predict(trA, cols, stat)
            psm_test = with_pools(gA, cols, stat, predict(g0, ho0, cols), ho0[stat])
            pt = predict(gA, test, cols)
            p_over = np.array([psm_test.over_probability(float(a), float(b)) for a, b in zip(pt, test_line)], dtype=float)
            hit = (test[stat].values > test_line.values).astype(float)
            r25 = {"n": int(len(test)), "mae": round(float(np.abs(pt - test[stat].values).mean()), 3),
                   "bias": round(float((pt - test[stat].values).mean()), 3), "naive_mae": round(naive_mae, 3),
                   "calibration": calibration(p_over, hit)}
            # 2026 lines: model from 2022-2025, misses measured on 2025 by the 2022-2024 model.
            gB = fit_predict(trB, cols, stat)
            psm_live = with_pools(gB, cols, stat, predict(gA, hoA, cols), hoA[stat])
            rows = evaluate_props(name, psm_live, cols, stat, feats, props, usable)
            detail.extend(rows)
            results[stat][name] = {"description": desc, "n_inputs": len(cols), "train_rows": int(len(trB)),
                                   "test_2025": r25, "props_2026": summarize_props(pd.DataFrame(rows))}
            c = r25["calibration"]
            print(f"  {name:<11} 2025: error {r25['mae']:.3f} bias {r25['bias']:+.3f} hit-chance gap {c.get('avg_gap_pts')} pts "
                  f"brier {c.get('brier')} | {len(rows)} props scored | {time.time() - t1:.0f}s")

        # The live model exactly as deployed, for the "before" line and as a
        # check against the existing props backtest.
        hist = feats[feats.season <= last]
        live = PlayerStatModel(stat)
        live.fit(hist, verbose=False)
        rows = evaluate_props(AS_LIVE, live, live.feature_columns, stat, feats, props, usable)
        detail.extend(rows)
        results[stat][AS_LIVE] = {"description": "Live model exactly as deployed", "n_inputs": len(live.feature_columns),
                                  "props_2026": summarize_props(pd.DataFrame(rows))}

    det = pd.DataFrame(detail)
    print_report(results)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "train_years": list(years), "season": season,
                   "min_edge": MIN_EDGE, "pff_match": match_info, "results": results}, f, indent=2, default=str)
    if not det.empty:
        det.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_JSON} and {OUT_CSV} ({len(det)} rows) in {time.time() - t0:.0f}s")


def print_report(results):
    for stat, res in results.items():
        print(f"\n################ {stat} ################")
        print(f"{'version':<11} | {'2025 error':>10} {'bias':>7} {'gap':>5} | {'2026 error':>10} {'bias':>7} {'gap':>5} | "
              f"{'overs 10%+ edge':>28} | {'unders 10%+ edge':>28}")
        for name, r in res.items():
            t, p = r.get("test_2025") or {}, r.get("props_2026") or {}
            def rec(x):
                return f"{x.get('record', '-'):>8} {x.get('roi', 0):+6.1f}% n={x.get('n', 0):<4}" if x and x.get("n") else f"{'-':>22}"
            print(f"{name:<11} | {t.get('mae', float('nan')):>10.3f} {t.get('bias', float('nan')):>+7.3f} "
                  f"{(t.get('calibration') or {}).get('avg_gap_pts', float('nan')):>5} | "
                  f"{p.get('mae', float('nan')):>10.2f} {p.get('bias', float('nan')):>+7.2f} "
                  f"{(p.get('calibration') or {}).get('avg_gap_pts', float('nan')):>5} | "
                  f"{rec(p.get('over_edge10')):>28} | {rec(p.get('under_edge10')):>28}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(sorted(a.years), a.season)
