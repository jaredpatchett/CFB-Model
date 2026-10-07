#!/usr/bin/env python3
"""
Props line-movement tracker (added 10/2026).

Wins and losses take months to prove an edge. Line movement shows it much
sooner: if the price keeps moving TOWARD the side the model flagged after
it flagged it, the market is agreeing with the model. This script follows
every flagged prop from the price it was flagged at to the last price seen
before kickoff.

What gets tracked (always the OVER, at the book it was flagged at)
  official      receptions overs flagged as official plays
  capped        receptions overs moved to Watch by the 30% edge cap
  rec_yds_over  receiving-yards overs at a 10%+ edge (the candidate second
                market; best book per player and game)
  baseline      every receptions and receiving-yards over the first time it
                is seen -- the comparison group. Flagged plays only mean
                something if they move our way more often than these do.

How it works
  Each live run looks at the props just exported (docs/data/latest.json):
    1. anything newly flagged is logged with its line, price and time;
    2. every logged prop whose game has not started gets its latest line and
       price at the same book updated.
  The log lives in data/clv/prop_flags.csv and is committed with the
  dashboard. The summary goes to docs/data/props_clv.json.

Reading the numbers
  moved our way   the price got worse for someone betting the over later
                  (or the line went up) -- we had the better number
  price move      change in the price's break-even chance, in points
  value at close  what our price was worth against the closing price with
                  the book's cut removed; above zero means we beat the close

Usage:
  python scripts/compute_props_clv.py                 # live run: one update
  python scripts/compute_props_clv.py --backfill-since 2026-10-06T21:00:00
                                                      # rebuild from git history
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import datetime, timezone

import pandas as pd

STATE = "data/clv/prop_flags.csv"
OUT = "docs/data/props_clv.json"
LATEST = "docs/data/latest.json"
MIN_EDGE = 10.0
BLOCKING = {"Out", "Out for Season", "IR", "Doubtful"}
BASELINE_MARKETS = {"Receptions", "Reception Yards"}
TAGS = ["official", "capped", "rec_yds_over", "baseline"]
COLS = ["key", "tag", "player", "team", "opponent", "market", "start_time", "flagged_at", "book", "line", "price",
        "other_price", "model_prob", "model_ev", "last_seen_at", "last_line", "last_price", "last_other_price", "looks"]


def _ts(s):
    d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def _norm(name):
    a = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", a.lower()).strip()


def _num(x):
    try:
        x = float(x)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def implied(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def decimal(price):
    return 1 + price / 100 if price > 0 else 1 + 100 / -price


def prop_key(p):
    return f"{p.get('fixture_id') or str(p.get('start_time'))[:10]}|{_norm(p.get('player_name'))}|{p.get('market_name')}"


def load_state():
    if not os.path.exists(STATE):
        return {}
    df = pd.read_csv(STATE)
    return {(r["key"], r["tag"]): {c: (None if pd.isna(r.get(c)) else r.get(c)) for c in COLS} for _, r in df.iterrows()}


def save_state(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    rows = sorted(state.values(), key=lambda r: (str(r["start_time"]), r["key"], r["tag"]))
    pd.DataFrame(rows, columns=COLS).to_csv(STATE, index=False)


def _record(p, tag, when):
    return {"key": prop_key(p), "tag": tag, "player": p.get("player_name"), "team": p.get("team"), "opponent": p.get("opponent"),
            "market": p.get("market_name"), "start_time": p.get("start_time"), "flagged_at": when.isoformat(),
            "book": p.get("book_used") or "unknown", "line": _num(p.get("line")), "price": _num(p.get("over_price")),
            "other_price": _num(p.get("under_price")), "model_prob": _num(p.get("model_over_probability")),
            "model_ev": _num(p.get("model_ev")) if p.get("model_lean") == "over" else None,
            "last_seen_at": when.isoformat(), "last_line": _num(p.get("line")), "last_price": _num(p.get("over_price")),
            "last_other_price": _num(p.get("under_price")), "looks": 0}


def update_state(state, props, when):
    """One snapshot of posted props: log new flags, refresh open ones."""
    live = []
    for p in props or []:
        try:
            if p.get("start_time") and p.get("line") is not None and _num(p.get("over_price")) is not None and _ts(p["start_time"]) > when:
                live.append(p)
        except ValueError:
            continue

    # 1) Refresh every open record at the book it was flagged at.
    by_book = {}
    for p in live:
        by_book[(prop_key(p), p.get("book_used") or "unknown")] = p
    for (key, tag), r in state.items():
        try:
            if _ts(r["start_time"]) <= when or when <= _ts(r["last_seen_at"] or r["flagged_at"]):
                continue                     # game started, or this snapshot was already counted
        except (ValueError, TypeError):
            continue
        p = by_book.get((key, r["book"]))
        if p is None:
            continue
        r.update({"last_seen_at": when.isoformat(), "last_line": _num(p.get("line")), "last_price": _num(p.get("over_price")),
                  "last_other_price": _num(p.get("under_price")), "looks": int(r.get("looks") or 0) + 1})

    # 2) Log anything newly flagged.
    def add(p, tag):
        k = (prop_key(p), tag)
        if k not in state:
            state[k] = _record(p, tag, when)

    best_yds = {}
    for p in live:
        if p.get("is_official_play"):
            add(p, "official")
        if p.get("edge_capped"):
            add(p, "capped")
        if (p.get("market_name") == "Reception Yards" and p.get("model_lean") == "over"
                and (_num(p.get("model_ev")) or -99) >= MIN_EDGE and p.get("injury_status") not in BLOCKING):
            k = prop_key(p)
            if k not in best_yds or p["model_ev"] > best_yds[k]["model_ev"]:
                best_yds[k] = p
    for p in best_yds.values():
        add(p, "rec_yds_over")
    for p in live:                                   # comparison group: first sighting of every receiving over
        if p.get("market_name") in BASELINE_MARKETS:
            add(p, "baseline")
    return state


def measure(r):
    """How one tracked prop moved between its flag and its last look."""
    out = {"later_look": bool(r.get("looks")) and r.get("last_price") is not None}
    if not out["later_look"]:
        return out
    if r["last_line"] != r["line"]:
        out["direction"] = "our way" if r["last_line"] > r["line"] else "against"
        out["line_moved"] = True
        return out
    move = implied(r["last_price"]) - implied(r["price"])
    out["price_move_pts"] = round(move * 100, 2)
    out["direction"] = "our way" if move > 1e-9 else ("against" if move < -1e-9 else "unchanged")
    if r.get("last_other_price") is not None:
        io, iu = implied(r["last_price"]), implied(r["last_other_price"])
        out["value_at_close_pct"] = round((decimal(r["price"]) * io / (io + iu) - 1) * 100, 1)
    return out


def summarize(state, now):
    summary, plays = {}, []
    for tag in TAGS:
        for label, markets in ((tag, None),) if tag != "baseline" else (("baseline_receptions", {"Receptions"}), ("baseline_rec_yds", {"Reception Yards"})):
            rows = [r for (k, t), r in state.items() if t == tag and (markets is None or r["market"] in markets)]
            done = [r for r in rows if _ts(r["start_time"]) <= now]
            m = [measure(r) for r in done]
            seen = [x for x in m if x["later_look"]]
            n = len(seen)
            moves = [x["price_move_pts"] for x in seen if "price_move_pts" in x]
            vals = [x["value_at_close_pct"] for x in seen if "value_at_close_pct" in x]
            cnt = lambda d: sum(1 for x in seen if x.get("direction") == d)
            summary[label] = {
                "tracked": len(rows), "games_started": len(done), "with_a_later_look": n,
                "moved_our_way": cnt("our way"), "moved_against": cnt("against"), "unchanged": cnt("unchanged"),
                "pct_our_way": round(cnt("our way") / n * 100, 1) if n else None,
                "avg_price_move_pts": round(sum(moves) / len(moves), 2) if moves else None,
                "avg_value_at_close_pct": round(sum(vals) / len(vals), 1) if vals else None,
            }
    for (k, t), r in sorted(state.items(), key=lambda kv: str(kv[1]["start_time"])):
        if t == "baseline":
            continue
        plays.append({"tag": t, "player": r["player"], "team": r["team"], "opponent": r["opponent"], "market": r["market"],
                      "start_time": r["start_time"], "started": _ts(r["start_time"]) <= now, "book": r["book"],
                      "flagged_at": r["flagged_at"], "line": r["line"], "price": r["price"], "model_ev": r["model_ev"],
                      "last_seen_at": r["last_seen_at"], "last_line": r["last_line"], "last_price": r["last_price"], **measure(r)})
    return {"generated_at": now.isoformat(), "summary": summary, "plays": plays,
            "note": "Over side only, at the book each prop was flagged at. 'games_started' rows are final; the rest are still moving."}


def main(backfill_since=None):
    if backfill_since:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import backtest_props_live as bpl
        state, n, last = {}, 0, None
        for when, d in bpl.snapshots(backfill_since):
            update_state(state, d.get("props") or [], when)
            n, last = n + 1, when
        print(f"Rebuilt from {n} saved snapshot(s) since {backfill_since}")
        now = last or datetime.now(timezone.utc)
    else:
        state = load_state()
        with open(LATEST) as f:
            d = json.load(f)
        now = _ts(d.get("generated_at")) if d.get("generated_at") else datetime.now(timezone.utc)
        update_state(state, d.get("props") or [], now)
    save_state(state)
    out = summarize(state, now)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    for tag, s in out["summary"].items():
        print(f"  {tag:<20} tracked {s['tracked']:<4} started {s['games_started']:<4} later look {s['with_a_later_look']:<4} "
              f"our way {s['moved_our_way']} / against {s['moved_against']} / unchanged {s['unchanged']} "
              f"| avg move {s['avg_price_move_pts']} pts | value at close {s['avg_value_at_close_pct']}%")
    print(f"Saved {STATE} ({len(state)} rows) and {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-since", default=None, help="rebuild the log from git history since this time (needs full history)")
    a = ap.parse_args()
    main(a.backfill_since)
