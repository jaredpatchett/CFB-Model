#!/usr/bin/env python3
"""
Props backtest against REAL posted lines (added 10/2026).

No historical prop lines exist for past seasons, but every pipeline run this
season committed docs/data/latest.json -- including every posted prop line,
its price, and the model's projection, all captured BEFORE kickoff. This
script walks that git history, takes each prop's last pre-kickoff read, and
grades it against the player's actual box score from CFBD. The model could
not have seen any of these results, so this is a genuine out-of-sample test
of exactly what the dashboard showed.

Each prop is re-rated with TODAY'S confidence rules (same as the dashboard's
Player Props table) and today's projection-sized uncertainty, so a "High"
from three weeks ago means the same thing as a "High" today. The model's
projections themselves are the ones it actually made at the time.

Output: a summary in the run log, docs/data/props_backtest.json, and
docs/data/props_backtest_detail.csv (one row per graded prop).

Usage (needs full git history -- the workflow checks out with fetch-depth 0):
  python scripts/backtest_props_live.py --season 2026
"""
import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from scipy.stats import norm

import config
from src.data import cfbd_client as cfbd
from src.features.player_features import pivot_player_game_stats
from src.features.live_player_features import market_name_to_stat, _normalize_name
from src.models.props_model import PlayerStatModel

DATA_FILE = "docs/data/latest.json"
COUNT_MARKET_WORDS = ("touchdown", "interception")
BLOCKING_INJ = {"Out", "Out For Season", "IR", "Doubtful"}


def _ts(s):
    s = str(s).replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def implied_prob(price):
    price = float(price)
    return (-price) / ((-price) + 100) if price < 0 else 100 / (price + 100)


def payout(price):
    price = float(price)
    return price / 100 if price > 0 else 100 / (-price)


def snapshots(since):
    out = subprocess.run(["git", "log", "--format=%H %cI", f"--since={since}", "--", DATA_FILE],
                         capture_output=True, text=True, check=True).stdout
    rows = [l.split() for l in out.splitlines() if l.strip()]
    rows.reverse()  # oldest first, so later snapshots overwrite earlier reads
    for sha, when in rows:
        try:
            blob = subprocess.run(["git", "show", f"{sha}:{DATA_FILE}"], capture_output=True, text=True, check=True).stdout
            yield _ts(when), json.loads(blob)
        except Exception:
            continue


def collect_reads(season_start):
    """(fixture, player, market) -> the last pre-kickoff read of that prop,
    using the model's best line within each snapshot (as the dashboard does)."""
    reads, n_snaps = {}, 0
    for when, d in snapshots(season_start):
        n_snaps += 1
        per_snap = {}
        for p in d.get("props") or []:
            if p.get("model_predicted_value") is None or not p.get("model_lean") or not p.get("start_time"):
                continue
            try:
                kick = _ts(p["start_time"])
            except Exception:
                continue
            if kick <= when:
                continue  # read taken after kickoff -- not something you could have bet
            lean = p["model_lean"]
            price = p.get("over_price") if lean == "over" else p.get("under_price")
            if price is None or (isinstance(price, float) and math.isnan(price)):
                continue
            op = p.get("model_over_probability")
            lp_then = p.get("model_lean_probability")
            if lp_then is None and op is not None:
                lp_then = op if lean == "over" else 1 - op
            key = (p.get("fixture_id"), _normalize_name(p["player_name"]), p["market_name"])
            rec = {
                "fixture_id": p.get("fixture_id"), "player": p["player_name"], "market": p["market_name"],
                "line": float(p["line"]), "lean": lean, "price": float(price), "book": p.get("book_used"),
                "pred": float(p["model_predicted_value"]), "lp_then": lp_then,
                "official_then": bool(p.get("is_official_play")), "team": p.get("team"),
                "games_played": p.get("games_played"), "model_confidence": p.get("model_confidence"),
                "injury_status": p.get("injury_status"), "start_time": p["start_time"], "read_at": when.isoformat(),
            }
            cur = per_snap.get(key)
            if cur is None or rec["official_then"] and not cur["official_then"] or \
                    (not cur["official_then"] and (rec["lp_then"] or 0) > (cur["lp_then"] or 0)):
                per_snap[key] = rec
        reads.update(per_snap)  # later snapshot = closer to kickoff
    print(f"Scanned {n_snaps} saved runs -> {len(reads)} distinct pre-kickoff prop reads")
    return list(reads.values())


def load_actuals(season):
    schedule = cfbd.get_games(season)
    done = schedule[schedule.get("completed") == True] if "completed" in schedule.columns else schedule
    weeks = sorted(int(w) for w in done["week"].dropna().unique())
    parts = []
    for wk in weeks:
        try:
            df = cfbd.get_player_game_stats(season, wk)
            if not df.empty:
                parts.append(df)
        except Exception as e:
            print(f"  [warn] week {wk} player stats failed: {e}")
    if not parts:
        return pd.DataFrame(), schedule
    wide = pivot_player_game_stats(pd.concat(parts, ignore_index=True), schedule)
    starts = schedule.set_index("id")["startDate"] if "startDate" in schedule.columns else pd.Series(dtype=str)
    wide["start"] = wide["gameId"].map(starts)
    wide["norm"] = wide["player"].map(_normalize_name)
    print(f"Loaded box scores: {len(wide)} player-games across weeks {weeks[0]}-{weeks[-1]}")
    return wide, schedule


def match_actual(rec, by_name):
    cands = by_name.get(_normalize_name(rec["player"]), [])
    kick = _ts(rec["start_time"])
    near = [c for c in cands if c["start"] is not None and abs(c["start"] - kick) <= timedelta(hours=36)]
    if rec.get("team"):
        near = [c for c in near if c["team"] == rec["team"]] or near
    return near[0] if len(near) == 1 else None


def tier(r):
    lp, be, gp, inj = r["lp_now"], r["breakeven"], r["gp"], r.get("injury_status")
    if lp is None:
        return "NONE"
    gap = lp - be
    if inj in BLOCKING_INJ or gap <= 0:
        return "PASS"
    if lp >= 0.80 or (r["line"] > 0 and abs(r["pred"] - r["line"]) >= max(0.25 * r["line"], 12)):
        return "CHECK"
    if lp >= 0.60 and gap >= 0.05 and (gp is None or gp >= 3) and not inj:
        t = "HIGH"
    elif lp >= 0.55 and gap >= 0.02 and (gp is None or gp >= 2):
        t = "MEDIUM"
    else:
        t = "LOW"
    m = r["market"].lower()
    if t in ("HIGH", "MEDIUM") and (any(w in m for w in COUNT_MARKET_WORDS) or m == "receptions"):
        t = "LOW"
    return t


def summarize(df, by):
    out = []
    for key, g in df.groupby(by, dropna=False):
        w, l, pu = (g.result == "W").sum(), (g.result == "L").sum(), (g.result == "P").sum()
        dec = w + l
        out.append({
            "group": key if not isinstance(key, tuple) else " / ".join(map(str, key)),
            "n": int(len(g)), "record": f"{w}-{l}-{pu}",
            "hit_rate": round(w / dec, 3) if dec else None,
            "avg_model_prob": round(g.lp_now.mean(), 3),
            "avg_breakeven": round(g.breakeven.mean(), 3),
            "units": round(g.units.sum(), 2),
            "roi": round(g.units.sum() / len(g), 3) if len(g) else None,
        })
    return out


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'group':<28}{'n':>6}{'record':>14}{'hit%':>8}{'model%':>8}{'needs%':>8}{'units':>9}{'ROI':>8}")
    for r in rows:
        hr = f"{r['hit_rate']*100:.1f}" if r["hit_rate"] is not None else "-"
        print(f"  {str(r['group']):<28}{r['n']:>6}{r['record']:>14}{hr:>8}{r['avg_model_prob']*100:>8.1f}"
              f"{r['avg_breakeven']*100:>8.1f}{r['units']:>+9.2f}{(r['roi'] or 0)*100:>+7.1f}%")


def main(season):
    reads = collect_reads(f"{season}-08-01")
    if not reads:
        raise SystemExit("No pre-kickoff prop reads found in the repo history.")

    models = {}
    for f in os.listdir(f"{config.MODELS_DIR}/props") if os.path.isdir(f"{config.MODELS_DIR}/props") else []:
        if f.endswith(".joblib"):
            try:
                models[f[:-7]] = PlayerStatModel.load(f"{config.MODELS_DIR}/props/{f}")
            except Exception as e:
                print(f"  [warn] could not load {f}: {e}")
    print(f"Loaded {len(models)} trained props models for today's uncertainty math")

    wide, _ = load_actuals(season)
    if wide.empty:
        raise SystemExit("No box scores available to grade against.")
    by_name = {}
    for _, row in wide.iterrows():
        try:
            st = _ts(row["start"]) if pd.notna(row["start"]) else None
        except Exception:
            st = None
        by_name.setdefault(row["norm"], []).append({"team": row["team"], "start": st, "row": row})

    graded, unmatched, no_stat = [], 0, 0
    for r in reads:
        stat = market_name_to_stat(r["market"])
        if not stat or stat not in wide.columns:
            no_stat += 1
            continue
        m = match_actual(r, by_name)
        if m is None:
            unmatched += 1
            continue
        actual = float(m["row"][stat])
        if actual == r["line"]:
            res = "P"
        else:
            res = "W" if (actual > r["line"]) == (r["lean"] == "over") else "L"
        model = models.get(stat)
        sd = model.std_for_prediction(r["pred"]) if model and hasattr(model, "std_for_prediction") else (model.residual_std if model else None)
        if sd:
            p_over = float(1 - norm.cdf(r["line"], loc=r["pred"], scale=sd))
            lp_now = p_over if r["lean"] == "over" else 1 - p_over
        else:
            lp_now = r["lp_then"]
        gp = r["games_played"]
        if gp is None and r.get("model_confidence") == "low":
            gp = 2
        rec = dict(r, actual=actual, result=res, lp_now=lp_now, breakeven=implied_prob(r["price"]), gp=gp,
                   units=(payout(r["price"]) if res == "W" else (-1.0 if res == "L" else 0.0)))
        rec["tier"] = tier(rec)
        gap_pct = abs(r["pred"] - r["line"]) / r["line"] if r["line"] else None
        rec["edge_bucket"] = ("<10% off line" if gap_pct is None or gap_pct < 0.10 else
                              "10-20% off line" if gap_pct < 0.20 else
                              "20-35% off line" if gap_pct < 0.35 else "35%+ off line")
        rec["prob_bucket"] = ("<55%" if lp_now is None or lp_now < 0.55 else "55-60%" if lp_now < 0.60 else
                              "60-65%" if lp_now < 0.65 else "65-70%" if lp_now < 0.70 else
                              "70-80%" if lp_now < 0.80 else "80%+")
        graded.append(rec)

    df = pd.DataFrame(graded)
    print(f"Graded {len(df)} props ({unmatched} couldn't be matched to a box score -- usually the player "
          f"didn't play or a name mismatch; {no_stat} were markets with no box-score stat)")
    if df.empty:
        raise SystemExit("Nothing graded.")

    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "CHECK": 3, "PASS": 4, "NONE": 5}
    tiers = sorted(summarize(df, "tier"), key=lambda r: order.get(r["group"], 9))
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "season": season, "n_graded": int(len(df)),
        "overall": summarize(df.assign(all="All props"), "all"),
        "by_tier": tiers,
        "by_side": summarize(df, "lean"),
        "by_tier_and_side": summarize(df[df.tier.isin(["HIGH", "MEDIUM", "LOW", "CHECK"])], ["tier", "lean"]),
        "by_market": sorted(summarize(df, "market"), key=lambda r: -r["n"]),
        "by_distance_from_line": summarize(df, "edge_bucket"),
        "by_model_probability": summarize(df, "prob_bucket"),
        "official_at_the_time": summarize(df.assign(o=df.official_then.map({True: "Official", False: "Not official"})), "o"),
    }
    print_table("OVERALL", results["overall"])
    print_table("BY CONFIDENCE TIER (today's rules)", results["by_tier"])
    print_table("OVERS vs UNDERS", results["by_side"])
    print_table("TIER x SIDE", results["by_tier_and_side"])
    print_table("BY MARKET", results["by_market"])
    print_table("BY HOW FAR THE MODEL WAS FROM THE LINE", results["by_distance_from_line"])
    print_table("CALIBRATION: model's hit chance vs actual hit rate", results["by_model_probability"])
    print_table("OFFICIAL PLAYS (as flagged at the time)", results["official_at_the_time"])

    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/props_backtest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    df.drop(columns=["lp_then"], errors="ignore").to_csv("docs/data/props_backtest_detail.csv", index=False)
    print("\nWrote docs/data/props_backtest.json and docs/data/props_backtest_detail.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    main(ap.parse_args().season)
