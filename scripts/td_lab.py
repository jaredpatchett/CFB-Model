#!/usr/bin/env python3
"""
Anytime touchdown lab (added 10/2026): could anytime TD props be bet?

Grades anytime TD lines that were really posted before kickoff (saved by
pull_props_history.py --market anytime_td) against several ways of pricing
a player's chance to score. Everything is forward in time: to grade season
S, every model is trained only on the seasons before S. Nothing here
touches the live model, the dashboard or the official-play rules.

Ways of pricing a player's chance to score
  live      the live props models as they work today: the rush TD and
            receiving TD models, each turned into a chance of 1+ with the
            model's real misses, combined as 1 - P(no rush TD) x P(no rec TD).
            This is what the dashboard's rush TD props do now.
  poisson   the same two projections added up and treated as a scoring rate:
            chance = 1 - e^(-(rush TD + rec TD projection)).
  direct    one model trained straight on "did he score", same 30 inputs.
  direct_cal  the direct model with its chances re-scaled on the season
            before (does it say 40% when players really score 40%?).

What a touchdown is
  Rushing, receiving, kick return and punt return touchdowns from the box
  score -- what sportsbooks settle anytime TD bets on. Passing TDs do not count.

How it is graded
  Yes side: the best price across books (and the median book, closer to
  having one account). No side where a book posted one. Edge = model chance
  x payout - chance of losing. By edge size, price range, player role and
  season, plus whether the chances themselves are honest next to the market.

Outputs
  docs/data/td_lab.json          summary tables
  docs/data/td_lab_detail.csv    one row per graded prop per version
  docs/data/td_lab_report.txt    the printed report

Usage (CFBD key; run from the workflow):
  python scripts/td_lab.py --years 2022 2023 2024 2025 --season 2026
"""
import argparse
import io
import json
import math
import os
import sys
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import backtest_props_live as bpl
import props_lab as lab
from src.features.live_player_features import _normalize_name
from src.features.player_features import (
    ROLLING_FEATURE_COLUMNS, pivot_player_game_stats, add_usage_features,
    attach_game_environment, build_rolling_player_features,
)

HIST_DIR = "data/props_history"
OUT_JSON = "docs/data/td_lab.json"
OUT_CSV = "docs/data/td_lab_detail.csv"
OUT_TXT = "docs/data/td_lab_report.txt"
COLS = list(ROLLING_FEATURE_COLUMNS)
VERSIONS = ["live", "poisson", "direct", "direct_cal"]
RETURN_CATS = {"kickreturns", "puntreturns"}
EDGE_BINS = [-9, 0, .05, .10, .20, .30, 99]
EDGE_LABELS = ["below 0", "0-5%", "5-10%", "10-20%", "20-30%", "30%+"]
PRICE_BINS = [-10000, -200, -110, 150, 300, 100000]
PRICE_LABELS = ["-200 or shorter", "-199 to -110", "-109 to +150", "+151 to +300", "+301 or longer"]
PROB_BINS = [0, .15, .25, .35, .45, .55, .65, 1.01]
PROB_LABELS = ["<15%", "15-25%", "25-35%", "35-45%", "45-55%", "55-65%", "65%+"]


# ----------------------------------------------------------------- data ----
def return_tds(stats):
    """Kick and punt return TDs per (gameId, athleteId) from the long box score."""
    if stats.empty:
        return pd.DataFrame(columns=["gameId", "athleteId", "ret_tds"])
    s = stats[stats["category"].astype(str).str.lower().isin(RETURN_CATS) & (stats["statType"].astype(str).str.upper() == "TD")].copy()
    s["ret_tds"] = pd.to_numeric(s["stat"], errors="coerce").fillna(0)
    for c in ("gameId", "athleteId"):
        s[c] = pd.to_numeric(s[c], errors="coerce")
    s = s.dropna(subset=["gameId", "athleteId"])
    return s.groupby(["gameId", "athleteId"], as_index=False)["ret_tds"].sum()


def build_frame(years, season):
    print("Loading CFBD history and this season...")
    g_hist, s_hist, l_hist = lab.load_history(years)
    g_now, s_now, l_now = lab.load_current(season)
    games = pd.concat([g_hist, g_now], ignore_index=True)
    lines = pd.concat([x for x in (l_hist, l_now) if not x.empty], ignore_index=True) if not (l_hist.empty and l_now.empty) else pd.DataFrame()
    parts = [pivot_player_game_stats(s_hist, g_hist)]
    if not s_now.empty:
        parts.append(pivot_player_game_stats(s_now, g_now))
    wide = pd.concat(parts, ignore_index=True)
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
    feats = build_rolling_player_features(attach_game_environment(add_usage_features(wide), lines))
    ret = return_tds(pd.concat([s_hist, s_now], ignore_index=True))
    ret[["gameId", "athleteId"]] = ret[["gameId", "athleteId"]].astype("int64")
    feats = feats.merge(ret, on=["gameId", "athleteId"], how="left")
    feats["ret_tds"] = feats["ret_tds"].fillna(0.0)
    feats["any_td"] = ((feats["rush_tds"] + feats["rec_tds"] + feats["ret_tds"]) > 0).astype(int)
    starts = games.drop_duplicates("id").set_index("id")["startDate"] if "startDate" in games.columns else pd.Series(dtype=str)
    feats["start"] = feats["gameId"].map(starts)
    feats = feats.reset_index(drop=True)
    print(f"  {len(feats)} player-games, seasons {sorted(feats['season'].unique().astype(int))}; "
          f"return TDs found: {int(feats['ret_tds'].sum())}")
    return feats


def load_lines(feats, seasons):
    """Saved anytime TD lines -> one prop per (game, player), matched to the
    player's box-score row."""
    props, notes = [], {}
    for s in seasons:
        path = f"{HIST_DIR}/anytime_td_{s}.csv"
        if not os.path.exists(path):
            notes[str(s)] = "no lines saved"
            continue
        try:
            raw = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            notes[str(s)] = "no lines saved"
            continue
        raw = raw.dropna(subset=["event_id", "player", "commence_time", "over"])
        cur = feats[feats["season"] == s]
        by_name = {}
        for idx, start, player, team in zip(cur.index, cur["start"], cur["player"], cur["team"]):
            try:
                st = bpl._ts(start) if pd.notna(start) else None
            except Exception:
                st = None
            by_name.setdefault(_normalize_name(player), []).append({"team": team, "start": st, "idx": idx})
        posted = matched = ghosts = 0
        for (eid, player), g in raw.groupby(["event_id", "player"]):
            posted += 1
            start_time = g["commence_time"].iloc[0]
            try:
                m = bpl.match_actual({"player": player, "start_time": start_time}, by_name)
            except Exception:
                m = None
            ghost = False
            if m is None:
                # No box-score row for this game. If he played for one of these
                # teams in an earlier game this season, he either sat (book
                # voids) or played without touching the ball (Yes loses) --
                # he did NOT score, since every TD shows up in the box score.
                # Priced from his most recent earlier game's inputs.
                m = prior_row(player, start_time, g, by_name)
                if m is None:
                    continue
                ghost = True
                ghosts += 1
            else:
                matched += 1
            kick = pd.Timestamp(bpl._ts(start_time)) - pd.Timedelta(hours=5)
            sat = (kick + pd.Timedelta(days={0: -2, 1: -3, 2: 3, 3: 2, 4: 1, 5: 0, 6: -1}[kick.weekday()])).date()
            books = [{"book": r.book, "yes": float(r.over), "no": (None if pd.isna(r.under) else float(r.under))} for r in g.itertuples()]
            books = [b for b in books if abs(b["yes"]) >= 100]
            if not books:
                continue
            props.append({"idx": m["idx"], "season": s, "player": player, "weekend": str(sat), "event_id": eid, "books": books, "ghost": ghost})
        notes[str(s)] = {"players_with_a_line": posted, "matched_to_a_box_score": matched,
                         "no_box_score_row_but_played_earlier": ghosts,
                         "games_with_lines": int(raw["event_id"].nunique()), "books": sorted(raw["book"].dropna().unique().tolist())}
    return props, notes


def prior_row(player, start_time, g, by_name, max_days=30):
    teams = [str(x) for x in (g["home_team"].iloc[0], g["away_team"].iloc[0])]
    kick = bpl._ts(start_time)
    cands = [c for c in by_name.get(_normalize_name(player), [])
             if c["start"] is not None and kick - pd.Timedelta(days=max_days) <= c["start"] < kick - pd.Timedelta(hours=36)
             and any(t == str(c["team"]) or t.startswith(str(c["team"]) + " ") for t in teams)]
    return max(cands, key=lambda c: c["start"]) if cands else None


def to_decimal(price):
    return 1 + bpl.payout(price)


def from_decimal(dec):
    return (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)


# --------------------------------------------------------------- models ----
def new_clf():
    return GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=42)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Season:
    """All four ways of pricing, trained only on seasons before S."""
    def __init__(self, feats, pop, s):
        tr, prev_tr, ho = feats[pop & (feats.season < s)], feats[pop & (feats.season < s - 1)], feats[pop & (feats.season == s - 1)]
        self.psm = {}
        for stat in ("rush_tds", "rec_tds"):
            g_now, g_prev = lab.fit_predict(tr, COLS, stat), lab.fit_predict(prev_tr, COLS, stat)
            self.psm[stat] = lab.with_pools(g_now, COLS, stat, lab.predict(g_prev, ho, COLS), ho[stat])
        self.clf = new_clf().fit(tr[COLS], tr["any_td"])
        prev_clf = new_clf().fit(prev_tr[COLS], prev_tr["any_td"])
        self.cal = LogisticRegression(C=1e6).fit(logit(prev_clf.predict_proba(ho[COLS])[:, 1]).reshape(-1, 1), ho["any_td"])
        self.n_train = int(len(tr))

    def price(self, X):
        rush = lab.predict(self.psm["rush_tds"].model, X, COLS)
        rec = lab.predict(self.psm["rec_tds"].model, X, COLS)
        p_r = np.array([self.psm["rush_tds"].over_probability(float(x), 0.5) for x in rush], float)
        p_c = np.array([self.psm["rec_tds"].over_probability(float(x), 0.5) for x in rec], float)
        direct = self.clf.predict_proba(X[COLS])[:, 1]
        return {"rush_proj": rush, "rec_proj": rec,
                "live": 1 - (1 - p_r) * (1 - p_c),
                "poisson": 1 - np.exp(-(rush + rec)),
                "direct": direct,
                "direct_cal": self.cal.predict_proba(logit(direct).reshape(-1, 1))[:, 1]}


# -------------------------------------------------------------- grading ----
def implied(price):
    return bpl.implied_prob(price)


def role(r):
    if r.get("roll_pass_att", 0) >= 10:
        return "QB"
    if r.get("roll_rush_att", 0) >= 8:
        return "lead rusher (8+ carries)"
    if r.get("roll_receptions", 0) >= 3:
        return "main receiver (3+ catches)"
    if r.get("roll_rush_att", 0) >= 3:
        return "backup rusher"
    return "minor role"


def grade_props(feats, props, pop, seasons):
    rows = []
    for s in seasons:
        sp = [p for p in props if p["season"] == s and pop.loc[p["idx"]]]
        if not sp:
            continue
        print(f"  grading {s}: training on seasons before {s} ({len(sp)} props)...")
        model = Season(feats, pop, s)
        X = feats.loc[[p["idx"] for p in sp]]
        priced = model.price(X)
        for i, (p, (_, fr)) in enumerate(zip(sp, X.iterrows())):
            yes = [b["yes"] for b in p["books"]]
            best_yes = max(yes, key=bpl.payout)
            med_yes = float(from_decimal(float(np.median([to_decimal(y) for y in yes]))))
            nv = [implied(b["yes"]) / (implied(b["yes"]) + implied(b["no"])) for b in p["books"] if b["no"] is not None]
            nos = [b["no"] for b in p["books"] if b["no"] is not None]
            best_no = max(nos, key=bpl.payout) if nos else None
            dk = next((b["yes"] for b in p["books"] if b["book"] == "draftkings"), None)
            ghost = bool(p.get("ghost"))
            scored = 0 if ghost else int(fr["any_td"])
            base = {"season": s, "weekend": p["weekend"], "player": p["player"], "team": fr.get("team"), "opponent": fr.get("opponent"),
                    "role": role(fr), "gp": int(fr["games_played_prior"]), "team_spread": fr.get("team_spread"),
                    "team_implied_total": fr.get("team_implied_total"), "n_books": len(yes),
                    "rush_proj": round(float(priced["rush_proj"][i]), 3), "rec_proj": round(float(priced["rec_proj"][i]), 3),
                    "best_yes": best_yes, "median_yes": med_yes, "dk_yes": dk, "best_no": best_no,
                    "mkt_implied": round(float(np.mean([implied(y) for y in yes])), 4),
                    "mkt_novig": round(float(np.mean(nv)), 4) if nv else None,
                    "ghost": ghost, "scored": scored}
            for v in VERSIONS:
                pr = float(priced[v][i])
                r = dict(base, version=v, p=round(pr, 4))
                for tag, price in (("best", best_yes), ("median", med_yes), ("dk", dk)):
                    if price is None:
                        r[f"yes_{tag}_ev"] = r[f"yes_{tag}_units"] = None
                        continue
                    r[f"yes_{tag}_ev"] = round(pr * bpl.payout(price) - (1 - pr), 4)
                    r[f"yes_{tag}_units"] = round(bpl.payout(price), 4) if scored else -1.0
                if best_no is not None:
                    r["no_ev"] = round((1 - pr) * bpl.payout(best_no) - pr, 4)
                    r["no_units"] = round(bpl.payout(best_no), 4) if not scored else -1.0
                else:
                    r["no_ev"] = r["no_units"] = None
                rows.append(r)
    return pd.DataFrame(rows)


def rec(d, ev, units, price_col=None):
    d = d[d[ev].notna()]
    n = len(d)
    if n == 0:
        return {"n": 0}
    w, u = int((d[units] > 0).sum()), float(d[units].sum())
    out = {"n": n, "record": f"{w}-{n - w}", "hit": round(w / n * 100, 1), "units": round(u, 1), "roi": round(u / n * 100, 1)}
    if price_col:
        out["needs"] = round(float(d[price_col].map(implied).mean()) * 100, 1)
    return out


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summarize(det):
    out = {}
    for v in VERSIONS:
        full = det[det.version == v]
        d = full[~full.ghost]
        if d.empty:
            continue
        res = {"n_props": int(len(d)), "n_no_box_score_row": int(full.ghost.sum())}
        res["yes_by_edge_best_no_stat_as_loss"] = {lb: rec(g, "yes_best_ev", "yes_best_units", "best_yes")
                                                   for lb, g in full.groupby(pd.cut(full.yes_best_ev, EDGE_BINS, labels=EDGE_LABELS), observed=True)}
        # Honesty: model vs market vs what happened.
        nv = d[d.mkt_novig.notna()]
        res["honesty"] = {
            "all": {"n": int(len(d)), "model": round(float(d.p.mean()) * 100, 1), "market_with_vig": round(float(d.mkt_implied.mean()) * 100, 1),
                    "actual": round(float(d.scored.mean()) * 100, 1), "logloss_model": round(logloss(d.p, d.scored), 4)},
            "rows_with_both_sides": {"n": int(len(nv)), "model": round(float(nv.p.mean()) * 100, 1) if len(nv) else None,
                                     "market_no_vig": round(float(nv.mkt_novig.mean()) * 100, 1) if len(nv) else None,
                                     "actual": round(float(nv.scored.mean()) * 100, 1) if len(nv) else None,
                                     "logloss_model": round(logloss(nv.p, nv.scored), 4) if len(nv) else None,
                                     "logloss_market": round(logloss(nv.mkt_novig, nv.scored), 4) if len(nv) else None},
        }
        b = pd.cut(d.p, PROB_BINS, labels=PROB_LABELS, right=False)
        res["by_model_chance"] = [{"bucket": lb, "n": int((b == lb).sum()), "model": round(float(d.p[b == lb].mean()) * 100, 1),
                                   "market": round(float(d.mkt_implied[b == lb].mean()) * 100, 1), "actual": round(float(d.scored[b == lb].mean()) * 100, 1)}
                                  for lb in PROB_LABELS if (b == lb).sum()]
        # Betting the Yes side.
        res["yes_by_edge_best"] = {lb: rec(g, "yes_best_ev", "yes_best_units", "best_yes")
                                   for lb, g in d.groupby(pd.cut(d.yes_best_ev, EDGE_BINS, labels=EDGE_LABELS), observed=True)}
        res["yes_by_edge_median"] = {lb: rec(g, "yes_median_ev", "yes_median_units", "median_yes")
                                     for lb, g in d.groupby(pd.cut(d.yes_median_ev, EDGE_BINS, labels=EDGE_LABELS), observed=True)}
        dk = d[d.dk_yes.notna()]
        res["yes_by_edge_draftkings"] = {lb: rec(g, "yes_dk_ev", "yes_dk_units", "dk_yes")
                                         for lb, g in dk.groupby(pd.cut(dk.yes_dk_ev, EDGE_BINS, labels=EDGE_LABELS), observed=True)}
        pos = d[d.yes_best_ev > 0]
        res["yes_edge0_by_season"] = {int(s): rec(g, "yes_best_ev", "yes_best_units", "best_yes") for s, g in pos.groupby("season")}
        res["yes_edge10_by_season"] = {int(s): rec(g, "yes_best_ev", "yes_best_units", "best_yes") for s, g in d[d.yes_best_ev >= .10].groupby("season")}
        res["yes_edge0_by_price"] = {lb: rec(g, "yes_best_ev", "yes_best_units", "best_yes")
                                     for lb, g in pos.groupby(pd.cut(pos.best_yes, PRICE_BINS, labels=PRICE_LABELS), observed=True)}
        res["yes_edge0_by_role"] = {k: rec(g, "yes_best_ev", "yes_best_units", "best_yes") for k, g in pos.groupby("role")}
        # Betting the No side, where posted.
        no = d[d.no_ev.notna()]
        res["no_by_edge"] = {lb: rec(g, "no_ev", "no_units", "best_no")
                             for lb, g in no.groupby(pd.cut(no.no_ev, EDGE_BINS, labels=EDGE_LABELS), observed=True)}
        out[v] = res
    # No model at all: every Yes at the best price, by price range and role.
    d = det[(det.version == VERSIONS[0]) & ~det.ghost]
    out["no_model_every_yes"] = {
        "all": rec(d, "yes_best_ev", "yes_best_units", "best_yes"),
        "by_price": {lb: rec(g, "yes_best_ev", "yes_best_units", "best_yes")
                     for lb, g in d.groupby(pd.cut(d.best_yes, PRICE_BINS, labels=PRICE_LABELS), observed=True)},
        "by_role": {k: rec(g, "yes_best_ev", "yes_best_units", "best_yes") for k, g in d.groupby("role")},
        "by_season": {int(s): rec(g, "yes_best_ev", "yes_best_units", "best_yes") for s, g in d.groupby("season")},
    }
    return out


# --------------------------------------------------------------- report ----
def fmt(r):
    if not r or not r.get("n"):
        return "-"
    return f"{r['record']:>9} {r['hit']:5.1f}% (needs {r.get('needs', float('nan')):4.1f}%) {r['units']:+7.1f}u {r['roi']:+6.1f}%"


def report(summary, notes):
    print("\n================ ANYTIME TD LAB ================")
    print(f"Lines: {json.dumps(notes)}")
    nm = summary.get("no_model_every_yes", {})
    print("\n-- No model: every Yes at the best price --")
    print(f"  all: {fmt(nm.get('all'))}")
    for k, r in (nm.get("by_season") or {}).items():
        print(f"  {k}: {fmt(r)}")
    for k, r in (nm.get("by_price") or {}).items():
        print(f"  {k:<18} {fmt(r)}")
    for k, r in (nm.get("by_role") or {}).items():
        print(f"  {k:<28} {fmt(r)}")
    for v in VERSIONS:
        res = summary.get(v)
        if not res:
            continue
        h = res["honesty"]
        print(f"\n######## {v} ########")
        print(f"  all {h['all']}")
        print(f"  both sides posted {h['rows_with_both_sides']}")
        print("  by model chance:")
        for b in res["by_model_chance"]:
            print(f"    {b['bucket']:<7} n={b['n']:<5} model {b['model']:5.1f}  market {b['market']:5.1f}  actual {b['actual']:5.1f}")
        for title, key in (("Yes, best price, by edge", "yes_by_edge_best"), ("Yes, median book, by edge", "yes_by_edge_median"),
                           ("Yes, DraftKings, by edge", "yes_by_edge_draftkings"), ("Yes 0%+ edge by season", "yes_edge0_by_season"),
                           ("Yes 10%+ edge by season", "yes_edge10_by_season"), ("Yes 0%+ edge by price", "yes_edge0_by_price"),
                           ("Yes 0%+ edge by role", "yes_edge0_by_role"), ("No side by edge", "no_by_edge"),
                           ("Yes, best price, by edge, players with no box-score line counted as losses", "yes_by_edge_best_no_stat_as_loss")):
            print(f"  {title}:")
            for k, r in (res.get(key) or {}).items():
                print(f"    {str(k):<28} {fmt(r)}")


# ----------------------------------------------------------------- main ----
def main(years, season):
    t0 = time.time()
    feats = build_frame(years, season)
    seasons = [s for s in range(min(years) + 2, season + 1)]
    props, notes = load_lines(feats, seasons)
    print(f"Lines matched to box scores: {len(props)} ({notes})")
    if not props:
        raise SystemExit("No anytime TD lines saved -- run pull_props_history.py --market anytime_td first.")
    pop = feats[COLS].notna().all(axis=1) & (feats["games_played_prior"] >= 1) & feats["rush_tds"].notna() & feats["rec_tds"].notna()
    usable = sum(bool(pop.loc[p["idx"]]) for p in props)
    notes["props_with_a_prior_game"] = usable
    det = grade_props(feats, props, pop, sorted({p["season"] for p in props}))
    summary = summarize(det)
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(summary, notes)
    text = buf.getvalue()
    print(text)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "train_years": list(years), "season": season,
                   "lines": notes, "results": summary}, f, indent=2, default=str)
    det.to_csv(OUT_CSV, index=False)
    with open(OUT_TXT, "w") as f:
        f.write(text)
    print(f"\nSaved {OUT_JSON}, {OUT_CSV} ({len(det)} rows), {OUT_TXT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    try:
        main(sorted(a.years), a.season)
    except BaseException as e:
        if isinstance(e, SystemExit) and not e.code:
            raise
        import traceback
        tb = traceback.format_exc()
        print(tb)
        # Shows the error on the run's summary page (readable without logs).
        print("::error title=td_lab failed::" + tb[-3000:].replace("%", "%25").replace("\r", "").replace("\n", "%0A"))
        sys.exit(1)
