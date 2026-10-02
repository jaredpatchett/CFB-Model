#!/usr/bin/env python3
"""
Props edge finder (added 10/2026).

The rebuilt props model roughly MATCHES the sportsbooks (54.5% hit vs ~56%
needed at the posted prices), so a projection edge alone won't beat the
juice. This script tests where an edge could still come from, using only
data already saved in the repo's docs/data/latest.json history:

  1. Line shopping   -- best line/price across all books vs. a single book.
  2. Bet timing      -- the FIRST line posted vs. the LAST line before
                        kickoff, and whether lines move toward the model's
                        side after they open (closing-line value).
  3. Edge threshold  -- only bet when the model's expected value clears
                        0%, 3%, 5% or 10%.
  4. Pockets         -- every measurable slice (stat, over/under, team
                        favored or not, game total, player role, sample
                        size, day of week, book).

Guarding against luck: testing many slices on two weekends WILL turn up some
that look profitable by chance. A slice is only listed as a CANDIDATE if it
made money in every graded weekend separately, has at least MIN_BETS bets,
and its overall result is at least ~1.5 standard errors above break-even.
Candidates still need confirming on future weekends before real money.

All projections come from the CURRENT (rebuilt) props models using only
pre-game information (same feature rows the models train on).

Usage (needs full git history; run from the props backtest workflow):
  python scripts/props_edge_finder.py --season 2026
"""
import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import pandas as pd

import config
import backtest_props_live as bpl
from src.features.live_player_features import market_name_to_stat, _normalize_name
from src.models.props_model import PlayerStatModel

SINGLE_BOOK_ORDER = ["draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "betrivers", "fanatics"]
EV_THRESHOLDS = [None, 0.0, 0.03, 0.05, 0.10]
MIN_BETS = 40


def _num(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def collect_markets(season_start):
    """(fixture, player, market) -> {'first': rows, 'last': rows, meta}, where
    rows are every book/line posted in the first and last pre-kickoff runs
    that carried this prop."""
    out = {}
    n = 0
    for when, d in bpl.snapshots(season_start):
        n += 1
        per = {}
        for p in d.get("props") or []:
            if not p.get("start_time") or p.get("line") is None:
                continue
            try:
                kick = bpl._ts(p["start_time"])
            except Exception:
                continue
            if kick <= when:
                continue
            key = (p.get("fixture_id"), _normalize_name(p["player_name"]), p["market_name"])
            row = {"line": _num(p["line"]), "over": _num(p.get("over_price")), "under": _num(p.get("under_price")),
                   "book": p.get("book_used") or "unknown", "at": when.isoformat()}
            if row["line"] is None:
                continue
            per.setdefault(key, {"rows": [], "meta": {"player": p["player_name"], "market": p["market_name"],
                                                      "start_time": p["start_time"], "team": p.get("team")}})["rows"].append(row)
        for key, v in per.items():
            rec = out.setdefault(key, {"meta": v["meta"], "first": v["rows"], "last": v["rows"]})
            rec["last"] = v["rows"]
            if v["meta"].get("team"):
                rec["meta"]["team"] = v["meta"]["team"]
    print(f"Scanned {n} saved runs -> {len(out)} distinct props with posted lines")
    return out


def choose(rows, model, pred, shop):
    """The bet you'd make from these posted rows: with shopping, the row and
    side with the best expected value across every book; without, the first
    available book in SINGLE_BOOK_ORDER."""
    if not rows:
        return None
    if not shop:
        by_book = {r["book"]: r for r in rows}
        pick = next((by_book[b] for b in SINGLE_BOOK_ORDER if b in by_book), rows[0])
        rows = [pick]
    best = None
    for r in rows:
        p_over = model.over_probability(pred, r["line"])
        if p_over is None:
            continue
        side = "over" if pred > r["line"] else "under"
        price = r["over"] if side == "over" else r["under"]
        if price is None:
            continue
        lp = p_over if side == "over" else 1 - p_over
        ev = lp * bpl.payout(price) - (1 - lp)
        if best is None or ev > best["ev"]:
            best = {"line": r["line"], "side": side, "price": price, "lp": lp, "ev": ev, "book": r["book"]}
    return best


def summarize(df, label):
    if df.empty:
        return {"group": label, "n": 0}
    w, l = int((df.result == "W").sum()), int((df.result == "L").sum())
    pu = int((df.result == "P").sum())
    n = len(df)
    units = float(df.units.sum())
    roi = units / n
    # z: how many standard errors the per-bet return sits above zero
    sd = df.units.std(ddof=1) if n > 1 else float("nan")
    z = roi / (sd / math.sqrt(n)) if n > 1 and sd and sd > 0 else 0.0
    out = {"group": label, "n": n, "record": f"{w}-{l}-{pu}", "hit_rate": round(w / (w + l), 3) if w + l else None,
           "needs": round(df.breakeven.mean(), 3), "units": round(units, 2), "roi": round(roi, 3), "z": round(z, 2)}
    by_wk = {}
    for wk, g in df.groupby("weekend"):
        by_wk[wk] = round(float(g.units.sum()) / len(g), 3)
    out["roi_by_weekend"] = by_wk
    return out


def fmt(r):
    if not r.get("n"):
        return f"  {str(r['group']):<34}{'0':>6}"
    wk = " ".join(f"{v*100:+5.1f}%" for _, v in sorted(r["roi_by_weekend"].items()))
    return (f"  {str(r['group']):<34}{r['n']:>6}{r['record']:>12}{(r['hit_rate'] or 0)*100:>7.1f}{r['needs']*100:>7.1f}"
            f"{r['units']:>+9.2f}{r['roi']*100:>+7.1f}%{r['z']:>6.2f}   {wk}")


def header(title):
    print(f"\n{title}")
    print(f"  {'group':<34}{'n':>6}{'record':>12}{'hit%':>7}{'needs':>7}{'units':>9}{'ROI':>8}{'z':>6}   ROI by weekend")


def main(season):
    markets = collect_markets(f"{season}-08-01")
    models = {}
    pdir = f"{config.MODELS_DIR}/props"
    for f in (os.listdir(pdir) if os.path.isdir(pdir) else []):
        if f.endswith(".joblib"):
            models[f[:-7]] = PlayerStatModel.load(f"{pdir}/{f}")
    wide, feats = bpl.load_actuals(season)
    by_name = {}
    for _, row in wide.iterrows():
        try:
            st = bpl._ts(row["start"]) if pd.notna(row["start"]) else None
        except Exception:
            st = None
        by_name.setdefault(row["norm"], []).append({"team": row["team"], "start": st, "row": row})

    bets = []
    for key, rec in markets.items():
        meta = rec["meta"]
        stat = market_name_to_stat(meta["market"])
        model = models.get(stat)
        if not stat or model is None or stat not in wide.columns:
            continue
        m = bpl.match_actual(meta, by_name)
        if m is None:
            continue
        fkey = (m["row"]["gameId"], m["row"]["athleteId"])
        if fkey not in feats.index:
            continue
        frow = feats.loc[fkey]
        if isinstance(frow, pd.DataFrame):
            frow = frow.iloc[0]
        cols = model.feature_columns
        if not cols or not frow[cols].notna().all() or frow.get("games_played_prior", 0) < 1:
            continue
        pred = float(model.model.predict(frow[cols].to_frame().T)[0])
        if stat in model.NONNEGATIVE:
            pred = max(pred, 0.0)
        actual = float(m["row"][stat])
        # Label by US Eastern day (a late Saturday-night kickoff is Sunday in
        # UTC), and group Thu-Mon games under that week's Saturday.
        kick = pd.Timestamp(bpl._ts(meta["start_time"])) - pd.Timedelta(hours=5)
        sat = (kick + pd.Timedelta(days={0: -2, 1: -3, 2: 3, 3: 2, 4: 1, 5: 0, 6: -1}[kick.weekday()])).date()
        base = {
            "player": meta["player"], "market": meta["market"], "stat": stat, "pred": pred, "actual": actual,
            "weekend": str(sat), "kick_day": "Thu/Fri" if kick.weekday() in (3, 4) else "Sat/Sun",
            "team_spread": frow.get("team_spread"), "implied": frow.get("team_implied_total"),
            "gp": int(frow.get("games_played_prior", 0)),
            "rec_share": frow.get("roll_rec_share"), "carry_share": frow.get("roll_carry_share"),
        }
        for variant, rows, shop in (("last line, one book", rec["last"], False), ("last line, best book", rec["last"], True),
                                    ("first line, one book", rec["first"], False), ("first line, best book", rec["first"], True)):
            pick = choose(rows, model, pred, shop)
            if not pick:
                continue
            g = bpl.grade(actual, pick["line"], pick["side"], pick["price"] if pick["side"] == "over" else None,
                          pick["price"] if pick["side"] == "under" else None)
            if not g:
                continue
            bets.append(dict(base, variant=variant, line=pick["line"], side=pick["side"], price=pick["price"],
                             book=pick["book"], lp=pick["lp"], ev=pick["ev"], result=g[0], units=g[2],
                             breakeven=bpl.implied_prob(pick["price"])))
        # Closing-line value: did the single-book line move toward the model's side?
        o, c = choose(rec["first"], model, pred, False), choose(rec["last"], model, pred, False)
        if o and c:
            move = (c["line"] - o["line"]) if o["side"] == "over" else (o["line"] - c["line"])
            bets.append(dict(base, variant="_clv", side=o["side"], line=o["line"], move=move, ev=o["ev"],
                             result="-", units=0.0, breakeven=0.0, price=o["price"], book=o["book"], lp=o["lp"]))

    df = pd.DataFrame(bets)
    if df.empty:
        raise SystemExit("Nothing to evaluate.")
    clv = df[df.variant == "_clv"]
    df = df[df.variant != "_clv"]
    print(f"Evaluated {df[df.variant == 'last line, one book'].shape[0]} props across weekends "
          f"{sorted(df.weekend.unique())}")
    results = {"generated_at": datetime.now(timezone.utc).isoformat(), "season": season}

    # 1-3. Variant x edge threshold
    header("1-3. LINE SHOPPING x BET TIMING x MINIMUM EDGE")
    grid = []
    for variant in ("last line, one book", "last line, best book", "first line, one book", "first line, best book"):
        v = df[df.variant == variant]
        for t in EV_THRESHOLDS:
            sub = v if t is None else v[v.ev >= t]
            r = summarize(sub, f"{variant} | " + ("every prop" if t is None else f"EV>={int(t*100)}%"))
            grid.append(r)
            print(fmt(r))
    results["grid"] = grid

    # Closing-line value
    if not clv.empty:
        moved = clv[clv.move != 0]
        toward = int((moved.move > 0).sum())
        print(f"\nCLOSING-LINE VALUE: of {len(moved)} props whose line moved between first post and kickoff, "
              f"it moved TOWARD the model's side {toward} times ({toward / max(len(moved), 1):.1%}). "
              f"Above 50% means the market tends to agree with the model after the fact.")
        for t in (0.0, 0.05, 0.10):
            mv = moved[moved.ev >= t]
            if len(mv):
                print(f"  model EV>={int(t*100)}% at open: line moved toward the model {(mv.move > 0).mean():.1%} of {len(mv)}")
        results["clv"] = {"moved": int(len(moved)), "toward_model": toward,
                          "share_toward": round(toward / max(len(moved), 1), 3)}

    # Pick the best-performing betting setup as the base for the pocket scan.
    scored = [g for g in grid if g.get("n", 0) >= 150]
    base_label = max(scored, key=lambda g: g["roi"])["group"] if scored else "last line, best book | EV>=3%"
    variant, rule = base_label.split(" | ")
    base = df[df.variant == variant]
    if rule.startswith("EV>="):
        base = base[base.ev >= int(rule[4:-1]) / 100]
    print(f"\n4. POCKETS -- scanned on the strongest setup above: {base_label}")

    def bucket(series, edges, labels):
        return pd.cut(pd.to_numeric(series, errors="coerce"), edges, labels=labels)
    base = base.assign(
        fav=bucket(base.team_spread, [-99, -7, -0.01, 7, 99], ["big favorite (7+)", "small favorite", "small underdog", "big underdog (7+)"]),
        total=bucket(base.implied, [0, 24, 31, 99], ["team total <24", "team total 24-31", "team total 31+"]),
        role=pd.Series([("featured" if (s or 0) >= 0.22 or (c or 0) >= 0.55 else
                         "secondary" if (s or 0) >= 0.12 or (c or 0) >= 0.30 else "depth")
                        for s, c in zip(base.rec_share, base.carry_share)], index=base.index),
        sample=pd.cut(base.gp, [0, 2, 3, 99], labels=["1-2 games", "3 games", "4+ games"]),
    )
    pockets = []
    for dim in ("market", "side", "fav", "total", "role", "sample", "kick_day", "book"):
        for val, g in base.groupby(dim, observed=True):
            pockets.append(dict(summarize(g, f"{dim}: {val}"), dim=dim))
    for (mk, sd), g in base.groupby(["market", "side"]):
        pockets.append(dict(summarize(g, f"market x side: {mk} {sd}"), dim="market x side"))
    weekends = sorted(df.weekend.unique())
    def is_candidate(p):
        return (p["n"] >= MIN_BETS and p["roi"] > 0 and p["z"] >= 1.5
                and len(p["roi_by_weekend"]) == len(weekends) and all(v > 0 for v in p["roi_by_weekend"].values()))
    for p in pockets:
        p["candidate"] = is_candidate(p)
    header("POCKETS (sorted by ROI; * = candidate: profitable every weekend, 40+ bets, z>=1.5)")
    for p in sorted(pockets, key=lambda p: -p["roi"]):
        print(("* " if p["candidate"] else "  ") + fmt(p)[2:])
    cands = [p for p in pockets if p["candidate"]]
    print(f"\n{len(cands)} candidate pocket(s). Tested {len(pockets)} slices -- at this many, a few false "
          f"positives are expected, so confirm any candidate on future weekends before betting it.")
    results["base_setup"] = base_label
    results["pockets"] = pockets
    results["candidates"] = cands

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/props_edge_finder.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    df.to_csv("docs/data/props_edge_finder_detail.csv", index=False)
    print("Wrote docs/data/props_edge_finder.json and docs/data/props_edge_finder_detail.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    main(ap.parse_args().season)
