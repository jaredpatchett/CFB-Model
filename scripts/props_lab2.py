#!/usr/bin/env python3
"""
Props lab 2 (added 10/2026): six follow-up tests on the receiving models.

Same rules as the first lab: everything is graded forward in time, and
nothing here touches the live model, the dashboard or the official rules.

Projection versions (receptions and receiving yards)
  pff                     the live model: current inputs + PFF routes/targets
  dropbacks               route participation measured against PFF's true
                          team dropbacks instead of the receiver-based estimate
  interactions            + expected targets (team pass volume x route
                          participation x targets per route) and role
                          opportunity (route participation x target share)
  dropbacks_interactions  both

Hit-chance versions (applied to the live model's projections)
  standard   past misses grouped by projection size (as live)
  tiers      also grouped by how steady the player's role has been
             (stable / average / volatile / short history)
  blend      the model's hit chance averaged 50/50 with the market's own
             (prices with the juice removed) -- this season's lines only

Also reported for every version: results by size of the model's edge
(including the official rule with 30%+ edges removed), and results for
players PFF matched vs did not match.

Math note: hit chances here handle whole-number lines properly. Simulated
outcomes are rounded to whole numbers on those lines, so a push is its own
outcome instead of being counted as a win for the under, and a push returns
the stake. On half-point lines this is identical to the live calculation.

2025  Train on 2022-2024, project 2025. Projection error and hit-chance
      honesty for players with a real receiving role.
2026  Train on 2022-2025, misses measured on 2025. Every saved receptions and
      receiving-yards line is scored on both sides at its real prices.

Outputs (model results only -- no PFF numbers; the repo is public)
  docs/data/props_lab2.json
  docs/data/props_lab2_detail.csv

Usage (full git history, CFBD and PFF keys; run from the workflow):
  python scripts/props_lab2.py --years 2022 2023 2024 2025 --season 2026
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

import backtest_props_live as bpl
import props_lab as lab
from src.data import pff_client as pff
from src.features import pff_features as pf
from src.features.player_features import (
    ROLLING_FEATURE_COLUMNS, pivot_player_game_stats, add_usage_features,
    attach_game_environment, build_rolling_player_features,
)
from src.models.props_model import PlayerStatModel

STATS = ["receptions", "rec_yds"]
OUT_JSON = "docs/data/props_lab2.json"
OUT_CSV = "docs/data/props_lab2_detail.csv"
MIN_EDGE, BIG_EDGE = 0.10, 0.30
ROLE_MIN_CATCHES = 1.5
MIN_TIER_ROWS = 300                 # a stability tier smaller than this falls back to the standard misses
BLEND_WEIGHT = 0.5
EDGE_BINS = [-9, 0, .05, .10, .15, .20, .30, 9]
EDGE_LABELS = ["below 0", "0-4.9%", "5-9.9%", "10-14.9%", "15-19.9%", "20-29.9%", "30%+"]
OVER_BINS = [0, .30, .40, .45, .50, .55, .60, .65, .70, 1.01]
OVER_LABELS = ["under 30%", "30-40%", "40-45%", "45-50%", "50-55%", "55-60%", "60-65%", "65-70%", "70%+"]
FAV_BINS = [.5, .55, .60, .65, .70, 1.01]
FAV_LABELS = ["50-55%", "55-60%", "60-65%", "65-70%", "70%+"]

BASE = list(ROLLING_FEATURE_COLUMNS)
PFFC = list(pf.PFF_FEATURE_COLUMNS)
ROUTE_PART_COLS = ["roll_pff_route_part", "last2_pff_route_part", "trend_pff_route_part"]
PFFC_DB = [c + "_db" if c in ROUTE_PART_COLS else c for c in PFFC]
INTER = ["x_role_opp", "x_exp_targets", "x_exp_targets_recent"]
INTER_DB = [c + "_db" for c in INTER]
VARIANTS = {
    "pff":                    (BASE + PFFC, "Live model: current inputs + PFF routes and targets"),
    "dropbacks":              (BASE + PFFC_DB, "Route participation from true team dropbacks"),
    "interactions":           (BASE + PFFC + INTER, "Live inputs + expected targets and role opportunity"),
    "dropbacks_interactions": (BASE + PFFC_DB + INTER_DB, "True dropbacks + interaction inputs"),
}
LIVE = "pff"
TIERS, BLEND = "pff+tiers", "pff+blend"


# ----------------------------------------------------------------- data ----
def fetch_passing(seasons, out_dir):
    """PFF passing report for every week (one row per quarterback-week) --
    the source of true team dropbacks. Returns an empty frame on failure."""
    os.makedirs(out_dir, exist_ok=True)
    parts = []
    for season in seasons:
        for week in pff.MODEL_WEEKS:
            try:
                df = pff.get_facet_summary("passing", season, week)
            except pff.PFFError as e:
                if e.status in (401, 403):
                    print(f"  [warn] PFF passing report not available: {e}")
                    return pd.DataFrame()
                continue
            if not df.empty:
                parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def team_dropbacks(passing):
    """(season, week, franchise_id) -> team dropbacks, plus the column used."""
    if passing is None or passing.empty or "franchise_id" not in passing.columns:
        return None, "PFF passing report returned no usable rows"
    col = next((c for c in ("dropbacks", "passing_snaps", "pass_snaps") if c in passing.columns), None)
    if col is None:
        return None, f"no dropbacks column in the PFF passing report (columns: {sorted(passing.columns)[:40]})"
    p = passing.assign(_d=pd.to_numeric(passing[col], errors="coerce").fillna(0.0))
    td = p.groupby(["season", "week", "franchise_id"])["_d"].sum().rename("team_dropbacks").reset_index()
    return td[td["team_dropbacks"] > 0], col


def per_game_volatility(feats, matched):
    """How much a player's targets and route participation have bounced
    around in his EARLIER games this season (standard deviation / mean)."""
    df = feats[["_rid", "gameId", "athleteId", "season", "week", "team"]].merge(matched, on=["gameId", "athleteId"], how="left")
    has = df["pff_routes"].notna()
    ever = has.groupby([df["athleteId"], df["season"]]).transform("any")
    charted = has.groupby([df["gameId"], df["team"]]).transform("any")
    fill = ever & charted & ~has
    for c in ("pff_targets", "pff_route_part"):
        df.loc[fill, c] = 0.0
    df = df.sort_values(["athleteId", "season", "week"])
    grp = df.groupby(["athleteId", "season"])
    for c in ("pff_targets", "pff_route_part"):
        sd = grp[c].transform(lambda s: s.shift(1).expanding().std())
        mean = grp[c].transform(lambda s: s.shift(1).expanding().mean())
        df[f"cv_{c}"] = np.where(mean > 0, sd / mean.replace(0, np.nan), np.nan)
    return df[["_rid", "cv_pff_targets", "cv_pff_route_part"]]


def add_interactions(feats, suffix=""):
    ok = feats["pff_matched"] == 1
    rp, l2 = feats["roll_pff_route_part" + suffix], feats["last2_pff_route_part" + suffix]
    vol, tprr, share = feats["roll_team_pass_att"], feats["roll_pff_tprr"], feats["roll_pff_target_share"]
    feats["x_role_opp" + suffix] = np.where(ok, rp * share, pf.MISSING)
    feats["x_exp_targets" + suffix] = np.where(ok, vol * rp * tprr, pf.MISSING)
    feats["x_exp_targets_recent" + suffix] = np.where(ok, vol * l2 * tprr, pf.MISSING)
    return feats


def build_frame(years, season):
    print("Loading CFBD history and this season...")
    g_hist, s_hist, l_hist = lab.load_history(years)
    g_now, s_now, l_now = lab.load_current(season)
    if s_now.empty:
        raise SystemExit(f"No completed {season} box scores yet.")
    games = pd.concat([g_hist, g_now], ignore_index=True)
    lines = pd.concat([x for x in (l_hist, l_now) if not x.empty], ignore_index=True) if not (l_hist.empty and l_now.empty) else pd.DataFrame()
    wide = pd.concat([pivot_player_game_stats(s_hist, g_hist), pivot_player_game_stats(s_now, g_now)], ignore_index=True)
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
    print(f"  {len(wide)} player-games")
    feats = build_rolling_player_features(attach_game_environment(add_usage_features(wide), lines))
    feats = lab.add_extra_features(feats).reset_index(drop=True)
    feats["_rid"] = np.arange(len(feats))

    print("Pulling PFF receiving and passing data (kept outside the repo)...")
    out_dir = os.environ.get("PFF_DATA_DIR") or os.path.join(tempfile.gettempdir(), "pff_lab2")
    seasons = list(years) + [season]
    recv, pgames = pff.fetch_receiving_seasons(seasons, out_dir)
    if recv.empty:
        raise SystemExit("PFF returned no receiving rows -- check the PFF key.")
    prep = pf.prepare_pff_rows(recv, pgames)
    matched, info = pf.match_pff_to_cfbd(feats, games, prep)
    feats = pf.add_pff_features(feats, matched)                     # the live (estimated-denominator) inputs
    feats = feats.merge(per_game_volatility(feats, matched), on="_rid", how="left")
    feats = add_interactions(feats)

    # True team dropbacks from the passing report.
    td, used = team_dropbacks(fetch_passing(seasons, out_dir))
    info["dropbacks_source"] = used
    if td is not None:
        p2 = prep.merge(td, on=["season", "week", "franchise_id"], how="left")
        true_part = (p2["pff_routes"] / p2["team_dropbacks"]).clip(0, 1.25)
        known = p2["team_dropbacks"].notna()
        ratio = (p2.loc[known & (p2["pff_route_part"] > 0), "pff_route_part"]
                 / true_part[known & (p2["pff_route_part"] > 0)].replace(0, np.nan)).dropna()
        info["pct_rows_with_true_dropbacks"] = round(float(known.mean()) * 100, 1)
        info["estimate_vs_true_median_ratio"] = round(float(ratio.median()), 3) if len(ratio) else None
        info["estimate_vs_true_p10_p90"] = [round(float(ratio.quantile(q)), 3) for q in (.1, .9)] if len(ratio) else None
        p2["pff_route_part"] = np.where(known, true_part, p2["pff_route_part"])
        m2, _ = pf.match_pff_to_cfbd(feats, games, p2.drop(columns=["team_dropbacks"]))
        slim = pf.add_pff_features(feats[["_rid", "gameId", "athleteId", "season", "week", "team"]], m2)
        slim = slim[["_rid"] + ROUTE_PART_COLS].rename(columns={c: c + "_db" for c in ROUTE_PART_COLS})
        feats = feats.merge(slim, on="_rid", how="left")
        feats = add_interactions(feats, "_db")
    print(f"  PFF: {info}")
    starts = games.drop_duplicates("id").set_index("id")["startDate"] if "startDate" in games.columns else pd.Series(dtype=str)
    feats["start"] = feats["gameId"].map(starts)
    # Role-stability score: average of the volatility measures available.
    vol = feats[["cv_receptions", "cv_pff_targets", "cv_pff_route_part"]].replace(0, np.nan).mean(axis=1, skipna=True)
    feats["volatility"] = vol.where(feats["games_played_prior"] >= 3)     # needs 3+ earlier games to mean anything
    return feats.reset_index(drop=True), info, td is not None


# -------------------------------------------------------- probabilities ----
def side_probs(psm, pred, line):
    """(P over, P under, P push) from the model's real past misses for
    projections this size. Whole-number lines: outcomes are rounded to whole
    numbers so a push is counted as a push."""
    edges, samples = psm.residual_bin_edges or [], psm.residual_bin_samples or []
    r = None
    if edges and samples and len(samples) == len(edges):
        i = next((k for k, hi in enumerate(edges) if pred <= hi), len(edges) - 1)
        r = np.asarray(samples[i], dtype=float)
    if r is None or len(r) < 30:
        p = psm.over_probability(pred, line)
        return None if p is None else (p, 1 - p, 0.0)
    o = pred + r
    if psm.stat_name in psm.NONNEGATIVE:
        o = np.clip(o, 0, None)
    if float(line).is_integer():
        o = np.rint(o)
    return float(np.mean(o > line)), float(np.mean(o < line)), float(np.mean(o == line))


class Pools:
    """Standard hit chances: one set of past misses, grouped by projection size."""
    def __init__(self, gbm, cols, stat, preds, actuals):
        self.psm = lab.with_pools(gbm, cols, stat, preds, actuals)

    def probs(self, pred, line, tier=None):
        return side_probs(self.psm, pred, line)


class TierPools(Pools):
    """Past misses also grouped by role stability. Cut-offs are the thirds of
    the volatility score among the rows the misses come from."""
    def __init__(self, gbm, cols, stat, preds, actuals, volatility):
        super().__init__(gbm, cols, stat, preds, actuals)
        preds, actuals, vol = np.asarray(preds, float), np.asarray(actuals, float), np.asarray(volatility, float)
        known = ~np.isnan(vol)
        self.cuts = [float(x) for x in np.quantile(vol[known], [1 / 3, 2 / 3])] if known.sum() >= 3 * MIN_TIER_ROWS else None
        self.by_tier, self.sizes = {}, {}
        if self.cuts:
            labels = np.array([self.tier(v) for v in vol])
            for t in ("stable", "average", "volatile", "short"):
                m = labels == t
                self.sizes[t] = int(m.sum())
                if m.sum() >= MIN_TIER_ROWS:
                    self.by_tier[t] = lab.with_pools(gbm, cols, stat, preds[m], actuals[m])

    def tier(self, v):
        if v is None or (isinstance(v, float) and math.isnan(v)) or not self.cuts:
            return "short"
        return "stable" if v <= self.cuts[0] else ("average" if v <= self.cuts[1] else "volatile")

    def probs(self, pred, line, tier=None):
        return side_probs(self.by_tier.get(tier, self.psm), pred, line)


def calibration(p_over, hit):
    """Brier score plus two views: by the model's over chance, and by the
    chance it gives whichever side it favors."""
    p, h = np.asarray(p_over, float), np.asarray(hit, float)
    if len(p) == 0:
        return {"n": 0}
    out = {"n": int(len(p)), "brier": round(float(np.mean((p - h) ** 2)), 4),
           "model_avg_over": round(float(p.mean()) * 100, 1), "actual_over": round(float(h.mean()) * 100, 1)}
    b = pd.cut(p, OVER_BINS, labels=OVER_LABELS, right=False)
    out["by_over_chance"] = [{"bucket": k, "n": int((b == k).sum()), "model": round(float(p[b == k].mean()) * 100, 1),
                              "actual": round(float(h[b == k].mean()) * 100, 1)} for k in OVER_LABELS if (b == k).sum()]
    q, fh = np.where(p >= .5, p, 1 - p), np.where(p >= .5, h, 1 - h)
    fb = pd.cut(q, FAV_BINS, labels=FAV_LABELS, right=False)
    rows = [{"bucket": k, "n": int((fb == k).sum()), "model": round(float(q[fb == k].mean()) * 100, 1),
             "actual": round(float(fh[fb == k].mean()) * 100, 1)} for k in FAV_LABELS if (fb == k).sum()]
    out["by_favored_side"] = rows
    out["avg_gap_pts"] = round(sum(r["n"] * abs(r["model"] - r["actual"]) for r in rows) / max(sum(r["n"] for r in rows), 1), 1)
    return out


# ----------------------------------------------------------------- 2026 ----
def decimal(price):
    return 1 + bpl.payout(price)


def market_over(over, under):
    """The market's own over chance: both prices with the juice removed."""
    io, iu = 1 / decimal(over), 1 / decimal(under)
    return io / (io + iu)


def grade(actual, line, side):
    if actual == line:
        return "P"
    return "W" if (actual > line) == (side == "over") else "L"


def score_rows(pools, pred, actual, rows, tier, blend):
    """Best-edge over and best-edge under across the books, at real prices.
    With blend=True the hit chance is averaged with the market's first."""
    best = {}
    for r in rows:
        pr = pools.probs(pred, r["line"], tier)
        if pr is None:
            continue
        po, pu, pp = pr
        have = {s: r.get(s) is not None and not (isinstance(r.get(s), float) and math.isnan(r.get(s))) for s in ("over", "under")}
        if blend:
            if not (have["over"] and have["under"]):
                continue
            mo = market_over(r["over"], r["under"])
            live = max(1 - pp, 1e-9)                                  # blend the non-push part
            po, pu = BLEND_WEIGHT * po + (1 - BLEND_WEIGHT) * mo * live, BLEND_WEIGHT * pu + (1 - BLEND_WEIGHT) * (1 - mo) * live
        for side, win, lose in (("over", po, pu), ("under", pu, po)):
            if not have[side]:
                continue
            price = float(r[side])
            ev = win * bpl.payout(price) - lose                       # a push returns the stake
            if side not in best or ev > best[side]["ev"]:
                res = grade(actual, r["line"], side)
                best[side] = {"ev": ev, "p": win, "price": price, "line": r["line"], "book": r["book"], "res": res,
                              "units": bpl.payout(price) if res == "W" else (-1.0 if res == "L" else 0.0)}
    return best


def evaluate_props(name, gbm, cols, pools, stat, feats, props, usable, blend=False):
    sub = [p for p in props if p["stat"] == stat and p["idx"] in usable]
    if not sub:
        return []
    X = feats.loc[[p["idx"] for p in sub]]
    preds = lab.predict(gbm, X, cols)
    tiered = isinstance(pools, TierPools)
    out = []
    for p, pred, (_, f) in zip(sub, preds, X.iterrows()):
        actual, pred = float(f[stat]), float(pred)
        tier = pools.tier(f["volatility"]) if tiered else None
        sides = score_rows(pools, pred, actual, p["rows"], tier, blend)
        if not sides:
            continue
        ref = p["rows"][0]
        rp = pools.probs(pred, ref["line"], tier)
        ref_over = rp[0] if rp else None
        both = all(ref.get(s) is not None and not (isinstance(ref.get(s), float) and math.isnan(ref.get(s))) for s in ("over", "under"))
        mkt = market_over(ref["over"], ref["under"]) if both else None
        if blend and ref_over is not None:
            ref_over = BLEND_WEIGHT * ref_over + (1 - BLEND_WEIGHT) * mkt if mkt is not None else None
        rec = {"version": name, "stat": stat, "player": p["player"], "market": p["market"], "weekend": p["weekend"],
               "team": f.get("team"), "opponent": f.get("opponent"), "gp": int(f["games_played_prior"]),
               "rec_share": f.get("roll_rec_share"), "team_spread": f.get("team_spread"),
               "pff_matched": int(f.get("pff_matched", 0) == 1), "stability": tier if tiered else None,
               "pred": round(pred, 2), "actual": actual, "ref_line": ref["line"], "ref_p_over": ref_over, "ref_market_over": mkt}
        for side in ("over", "under"):
            s = sides.get(side)
            for k in ("ev", "p", "price", "line", "book", "res", "units"):
                rec[f"{side}_{k}"] = s[k] if s else None
        out.append(rec)
    return out


def rec_line(df, side="over"):
    n = len(df)
    if n == 0:
        return {"n": 0}
    res, units, price, ev = df[f"{side}_res"], df[f"{side}_units"].astype(float), df[f"{side}_price"].astype(float), df[f"{side}_ev"].astype(float)
    w, l = int((res == "W").sum()), int((res == "L").sum())
    dec = np.where(price > 0, 1 + price / 100, 1 + 100 / price.abs())
    return {"n": n, "record": f"{w}-{l}", "hit": round(w / max(w + l, 1) * 100, 1), "needs": round(float(np.mean(1 / dec)) * 100, 1),
            "units": round(float(units.sum()), 1), "roi": round(float(units.sum()) / n * 100, 1),
            "avg_model_edge": round(float(ev.mean()) * 100, 1), "avg_decimal_odds": round(float(dec.mean()), 3),
            "roi_by_weekend": {k: round(float(g[f"{side}_units"].astype(float).sum()) / len(g) * 100, 1) for k, g in df.groupby("weekend")}}


def summarize(d):
    if d.empty:
        return {"n": 0}
    err = d.pred - d.actual
    out = {"n": int(len(d)), "mae": round(float(err.abs().mean()), 3), "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
           "bias": round(float(err.mean()), 3), "line_mae": round(float((d.ref_line - d.actual).abs().mean()), 3)}
    c = d[d.ref_p_over.notna() & (d.actual != d.ref_line)]
    out["calibration"] = calibration(c.ref_p_over, (c.actual > c.ref_line).astype(float))
    overs = d[d.over_ev.notna()]
    rule = overs[(overs.over_ev >= MIN_EDGE) & (overs.pred > overs.over_line)]      # today's official shape
    out["rule"] = rec_line(rule)
    out["rule_2plus_games"] = rec_line(rule[rule.gp >= 2])
    out["rule_under_30_edge"] = rec_line(rule[rule.over_ev < BIG_EDGE])
    out["rule_30_plus_edge"] = rec_line(rule[rule.over_ev >= BIG_EDGE])
    out["rule_pff_matched"] = rec_line(rule[rule.pff_matched == 1])
    out["rule_pff_unmatched"] = rec_line(rule[rule.pff_matched == 0])
    out["any_over_10"] = rec_line(overs[overs.over_ev >= MIN_EDGE])
    unders = d[d.under_ev.notna()]
    out["any_under_10"] = rec_line(unders[unders.under_ev >= MIN_EDGE], "under")
    for side, s in (("over", overs), ("under", unders)):
        b = pd.cut(s[f"{side}_ev"], EDGE_BINS, labels=EDGE_LABELS, right=False)
        out[f"{side}_edge_buckets"] = [dict(rec_line(s[b == k], side), bucket=k) for k in EDGE_LABELS if (b == k).sum()]
    out["props_pff_matched_pct"] = round(float(d.pff_matched.mean()) * 100, 1)
    if d.stability.notna().any():
        out["rule_by_stability"] = {k: rec_line(g) for k, g in rule.groupby("stability")}
    return out


# ----------------------------------------------------------------- main ----
def test_2025(gbm, cols, pools, test, stat, test_line, tiered):
    pt = lab.predict(gbm, test, cols)
    tiers = [pools.tier(v) for v in test["volatility"]] if tiered else [None] * len(test)
    pr = [pools.probs(float(a), float(b), t) for a, b, t in zip(pt, test_line, tiers)]
    p_over = np.array([x[0] if x else np.nan for x in pr], dtype=float)
    hit = (test[stat].values > test_line.values).astype(float)
    ok = ~np.isnan(p_over)
    err = pt - test[stat].values
    return {"n": int(len(test)), "mae": round(float(np.abs(err).mean()), 3), "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
            "bias": round(float(err.mean()), 3), "calibration": calibration(p_over[ok], hit[ok])}


def main(years, season):
    t0 = time.time()
    feats, info, have_db = build_frame(years, season)
    last, prev = max(years), max(years) - 1
    variants = {k: v for k, v in VARIANTS.items() if have_db or "dropbacks" not in k}
    all_cols = sorted({c for cols, _ in variants.values() for c in cols})
    missing = [c for c in all_cols if c not in feats.columns]
    if missing:
        raise SystemExit(f"Feature columns missing from the frame: {missing}")
    props = lab.collect_props(feats, season)
    print(f"Props matched to box scores: {len(props)}")

    results, detail = {s: {} for s in STATS}, []
    for stat in STATS:
        ok = feats[all_cols + [stat]].notna().all(axis=1) & (feats["games_played_prior"] >= 1)
        pop = ok & (feats["roll_receptions"] > 0)
        usable = set(feats.index[ok])
        test = feats[ok & (feats.season == last) & (feats.games_played_prior >= 2) & (feats.roll_receptions >= ROLE_MIN_CATCHES)]
        test_line = np.floor(test[f"roll_{stat}"]) + 0.5
        tr0, ho0 = feats[pop & (feats.season < prev)], feats[pop & (feats.season == prev)]
        trA, hoA = feats[pop & (feats.season < last)], feats[pop & (feats.season == last)]
        trB = feats[pop & (feats.season <= last)]
        print(f"\n===== {stat}: {len(trB)} training rows, {len(test)} in the {last} role-player test =====")
        if min(len(tr0), len(ho0), len(trA), len(hoA)) < 200:
            print("  not enough rows to run this stat, skipped")
            continue
        for name, (cols, desc) in variants.items():
            t1 = time.time()
            g0, gA, gB = (lab.fit_predict(x, cols, stat) for x in (tr0, trA, trB))
            p0, pA = lab.predict(g0, ho0, cols), lab.predict(gA, hoA, cols)
            std_test, std_live = Pools(gA, cols, stat, p0, ho0[stat]), Pools(gB, cols, stat, pA, hoA[stat])
            r25 = test_2025(gA, cols, std_test, test, stat, test_line, False)
            rows = evaluate_props(name, gB, cols, std_live, stat, feats, props, usable)
            detail.extend(rows)
            results[stat][name] = {"description": desc, "n_inputs": len(cols), "test_2025": r25, "props_2026": summarize(pd.DataFrame(rows))}
            c = r25["calibration"]
            print(f"  {name:<23} {last}: error {r25['mae']:.3f} rmse {r25['rmse']:.3f} brier {c.get('brier')} | {len(rows)} props | {time.time() - t1:.0f}s")
            if name != LIVE:
                continue
            # Hit-chance versions, on the live model's projections.
            tier_test = TierPools(gA, cols, stat, p0, ho0[stat], ho0["volatility"])
            tier_live = TierPools(gB, cols, stat, pA, hoA[stat], hoA["volatility"])
            rows = evaluate_props(TIERS, gB, cols, tier_live, stat, feats, props, usable)
            detail.extend(rows)
            results[stat][TIERS] = {"description": "Live projections, misses also grouped by role stability",
                                    "tier_cutoffs": tier_live.cuts, "tier_sizes": tier_live.sizes, "tiers_used": sorted(tier_live.by_tier),
                                    "test_2025": test_2025(gA, cols, tier_test, test, stat, test_line, True),
                                    "props_2026": summarize(pd.DataFrame(rows))}
            rows = evaluate_props(BLEND, gB, cols, std_live, stat, feats, props, usable, blend=True)
            detail.extend(rows)
            results[stat][BLEND] = {"description": "Live hit chances averaged 50/50 with the market's",
                                    "props_2026": summarize(pd.DataFrame(rows))}

    det = pd.DataFrame(detail)
    report(results)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "train_years": list(years), "season": season,
                   "pff": info, "results": results}, f, indent=2, default=str)
    if not det.empty:
        det.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_JSON} and {OUT_CSV} ({len(det)} rows) in {time.time() - t0:.0f}s")


def report(results):
    def r(x):
        return f"{x.get('record', '-'):>7} {x.get('roi', 0):+6.1f}% n={x.get('n', 0):<3}" if x and x.get("n") else f"{'-':>20}"
    for stat, res in results.items():
        print(f"\n################ {stat} ################")
        print(f"{'version':<23} | {'2025 err':>8} {'brier':>7} | {'2026 err':>8} {'brier':>7} {'gap':>5} | {'rule (overs 10%+)':>22} | {'under 30% edge':>22} | {'PFF unmatched':>22}")
        for name, x in res.items():
            t, p = x.get("test_2025") or {}, x.get("props_2026") or {}
            print(f"{name:<23} | {t.get('mae', float('nan')):>8.3f} {(t.get('calibration') or {}).get('brier', float('nan')):>7} | "
                  f"{p.get('mae', float('nan')):>8.3f} {(p.get('calibration') or {}).get('brier', float('nan')):>7} "
                  f"{(p.get('calibration') or {}).get('avg_gap_pts', float('nan')):>5} | {r(p.get('rule')):>22} | "
                  f"{r(p.get('rule_under_30_edge')):>22} | {r(p.get('rule_pff_unmatched')):>22}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(sorted(a.years), a.season)
