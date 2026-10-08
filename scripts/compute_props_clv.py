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
  group         (10/2026) every prop where the model shows an edge of 0% or
                more, on the side it leans to, filed under its backtest group
                (market + side + edge size, e.g. "Reception Yards under
                20-29.9%"). Each is marked with whether that group was UP in
                the backtest. These are also GRADED after the game from the
                box score, so every group builds a live record.

How it works
  Each live run looks at the props just exported (docs/data/latest.json):
    1. anything newly flagged is logged with its line, price and time;
    2. every logged prop whose game has not started gets its latest line and
       price at the same book updated.
    3. every logged prop whose game is over is graded from the CFBD box
       score: win, loss or push, and units at the price it was logged at.
       A player with no box-score line is left ungraded (he may not have
       played, and books void those).
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
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
from src.analysis import prop_edge_history as peh

STATE = "data/clv/prop_flags.csv"
OUT = "docs/data/props_clv.json"
LATEST = "docs/data/latest.json"
MIN_EDGE = 10.0
BLOCKING = {"Out", "Out for Season", "IR", "Doubtful"}
BASELINE_MARKETS = {"Receptions", "Reception Yards"}
TAGS = ["official", "capped", "rec_yds_over", "baseline"]
GRADED_TAGS = ("official", "capped", "rec_yds_over", "group")
GRADE_AFTER_HOURS = 5              # don't look for a box score until the game has had time to finish
MATCH_HOURS = 36                   # a logged kickoff and CFBD's kickoff for the same game
COLS = ["key", "tag", "player", "team", "opponent", "market", "start_time", "flagged_at", "book", "line", "price",
        "other_price", "model_prob", "model_ev", "last_seen_at", "last_line", "last_price", "last_other_price", "looks",
        "side", "group", "group_up", "actual", "result", "units"]


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


def _side(r):
    return "under" if str(r.get("side") or "over") == "under" else "over"


def _record(p, tag, when, side="over", group=None, group_up=None):
    """One logged prop. `price` is always the price of OUR side."""
    other = "under" if side == "over" else "over"
    over_prob = _num(p.get("model_over_probability"))
    prob = over_prob if side == "over" else (_num(p.get("model_under_probability")) or (1 - over_prob if over_prob is not None else None))
    return {"key": prop_key(p), "tag": tag, "player": p.get("player_name"), "team": p.get("team"), "opponent": p.get("opponent"),
            "market": p.get("market_name"), "start_time": p.get("start_time"), "flagged_at": when.isoformat(),
            "book": p.get("book_used") or "unknown", "line": _num(p.get("line")), "price": _num(p.get(f"{side}_price")),
            "other_price": _num(p.get(f"{other}_price")), "model_prob": prob,
            "model_ev": _num(p.get("model_ev")) if p.get("model_lean") == side else None,
            "last_seen_at": when.isoformat(), "last_line": _num(p.get("line")), "last_price": _num(p.get(f"{side}_price")),
            "last_other_price": _num(p.get(f"{other}_price")), "looks": 0,
            "side": side, "group": group, "group_up": group_up, "actual": None, "result": None, "units": None}


def update_state(state, props, when, history=None):
    """One snapshot of posted props: log new flags, refresh open ones."""
    live_all = []
    for p in props or []:
        try:
            if p.get("start_time") and p.get("line") is not None and _ts(p["start_time"]) > when:
                live_all.append(p)
        except ValueError:
            continue
    live = [p for p in live_all if _num(p.get("over_price")) is not None]      # over-side tags need an over price

    # 1) Refresh every open record at the book it was flagged at.
    by_book = {}
    for p in live_all:
        by_book[(prop_key(p), p.get("book_used") or "unknown")] = p
    for (key, tag), r in state.items():
        try:
            if _ts(r["start_time"]) <= when or when <= _ts(r["last_seen_at"] or r["flagged_at"]):
                continue                     # game started, or this snapshot was already counted
        except (ValueError, TypeError):
            continue
        p = by_book.get((key, r["book"]))
        side = _side(r)
        if p is None or _num(p.get(f"{side}_price")) is None:
            continue
        other = "under" if side == "over" else "over"
        r.update({"last_seen_at": when.isoformat(), "last_line": _num(p.get("line")), "last_price": _num(p.get(f"{side}_price")),
                  "last_other_price": _num(p.get(f"{other}_price")), "looks": int(r.get("looks") or 0) + 1})

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

    # Every play with an edge of 0%+ on the side the model leans to, filed
    # under its backtest group. Best edge per player, game and market.
    if history:
        best = {}
        for p in live_all:
            side, ev = p.get("model_lean"), _num(p.get("model_ev"))
            if side not in ("over", "under") or ev is None or ev < 0 or _num(p.get(f"{side}_price")) is None:
                continue
            if p.get("injury_status") in BLOCKING:
                continue
            k = prop_key(p)
            if k not in best or ev > best[k]["model_ev"]:
                best[k] = p
        for k, p in best.items():
            if (k, "group") in state:
                continue
            side = p["model_lean"]
            row = peh.find_group(history, p.get("market_name"), side, _num(p.get("model_ev")), capped=bool(p.get("edge_capped")))
            if row is None:
                continue
            state[(k, "group")] = _record(p, "group", when, side=side, group=peh.group_label(p.get("market_name"), side, row),
                                          group_up=bool(row["up"]))
    return state


def payout(price):
    return price / 100 if price > 0 else 100 / -price


def _season_of(start):
    d = _ts(start)
    return d.year if d.month >= 7 else d.year - 1


def _record_units(rows):
    """Record / units / ROI for graded rows (pushes count as bets, as elsewhere)."""
    done = [r for r in rows if r.get("result") in ("W", "L", "P")]
    w, l = sum(r["result"] == "W" for r in done), sum(r["result"] == "L" for r in done)
    units = sum(float(r.get("units") or 0) for r in done)
    return {"graded": len(done), "record": f"{w}-{l}", "units": round(units, 2),
            "roi": round(units / len(done) * 100, 1) if done else None}


def grade_state(state, now):
    """Grade finished props from CFBD box scores: W / L / P and units at the
    logged price. 'NA' = the game is final but the player has no box-score
    line (he may not have played; books void those). Never raises; returns
    how many were graded this run."""
    todo = []
    for (k, t), r in state.items():
        try:
            if (t in GRADED_TAGS and not r.get("result") and r.get("line") is not None and r.get("price") is not None
                    and _ts(r["start_time"]) + timedelta(hours=GRADE_AFTER_HOURS) <= now):
                todo.append(r)
        except (ValueError, TypeError):
            continue
    if not todo:
        return 0
    try:
        from src.data import cfbd_client as cfbd
        from src.features.player_features import pivot_player_game_stats
        from src.features.live_player_features import market_name_to_stat
    except Exception as e:
        print(f"  [note] grading skipped ({e})")
        return 0
    window = pd.Timedelta(hours=MATCH_HOURS)
    graded = 0
    for season in sorted({_season_of(r["start_time"]) for r in todo}):
        recs = [r for r in todo if _season_of(r["start_time"]) == season]
        try:
            games = cfbd.get_games(season)
            games = games.assign(kick=pd.to_datetime(games["startDate"], utc=True, errors="coerce"))
            done = games[games["completed"] == True] if "completed" in games.columns else games       # noqa: E712
            kicks = sorted({pd.Timestamp(_ts(r["start_time"])) for r in recs})
            near = done[done["kick"].apply(lambda k: pd.notna(k) and any(abs(k - x) <= window for x in kicks))]
            weeks = sorted(int(w) for w in near["week"].dropna().unique())
            if not weeks:
                continue
            parts = [cfbd.get_player_game_stats(season, wk) for wk in weeks]
            parts = [x for x in parts if x is not None and not x.empty]
            if not parts:
                continue
            wide = pivot_player_game_stats(pd.concat(parts, ignore_index=True), games)
            wide = wide.merge(done[["id", "kick"]].rename(columns={"id": "gameId"}), on="gameId", how="inner")
            wide["n"] = wide["player"].map(_norm)
        except Exception as e:
            print(f"  [warn] could not load {season} box scores for grading ({type(e).__name__}: {e})")
            continue
        for r in recs:
            kick = pd.Timestamp(_ts(r["start_time"]))
            stat = market_name_to_stat(r["market"])
            if stat is None or stat not in wide.columns:
                r["result"] = "NA"
                continue
            team = r.get("team")
            final = near[(near["kick"] - kick).abs() <= window]
            if team:
                final = final[(final["homeTeam"] == team) | (final["awayTeam"] == team)]
            rows = wide[(wide["n"] == _norm(r["player"])) & ((wide["kick"] - kick).abs() <= window)]
            if team and (rows["team"] == team).any():
                rows = rows[rows["team"] == team]
            if rows.empty:
                # Game is final but he has no box-score line -> can't tell a
                # zero from a DNP. Mark NA once the game is clearly over.
                if not final.empty or now - _ts(r["start_time"]) > timedelta(days=10):
                    r["result"] = "NA"
                continue
            actual = float(rows.iloc[0][stat])
            line, price, side = float(r["line"]), float(r["price"]), _side(r)
            res = "P" if actual == line else ("W" if (actual > line) == (side == "over") else "L")
            r.update({"actual": actual, "result": res, "units": round(payout(price) if res == "W" else (-1.0 if res == "L" else 0.0), 4)})
            graded += 1
    return graded


def measure(r):
    """How one tracked prop moved between its flag and its last look."""
    out = {"later_look": bool(r.get("looks")) and r.get("last_price") is not None}
    if not out["later_look"]:
        return out
    if r["last_line"] != r["line"]:
        up = r["last_line"] > r["line"]                 # a higher line helps an over we already hold, hurts an under
        out["direction"] = "our way" if up == (_side(r) == "over") else "against"
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
        if t == "baseline" or (t == "group" and str(r.get("group_up")).lower() != "true"):
            continue                                   # the play list keeps flagged plays and plays in groups that are up
        plays.append({"tag": t, "player": r["player"], "team": r["team"], "opponent": r["opponent"], "market": r["market"],
                      "side": _side(r), "group": r.get("group"),
                      "start_time": r["start_time"], "started": _ts(r["start_time"]) <= now, "book": r["book"],
                      "flagged_at": r["flagged_at"], "line": r["line"], "price": r["price"], "model_ev": r["model_ev"],
                      "last_seen_at": r["last_seen_at"], "last_line": r["last_line"], "last_price": r["last_price"],
                      "actual": r.get("actual"), "result": r.get("result"), "units": r.get("units"), **measure(r)})
    # Live results (graded from box scores).
    results = {t: _record_units([r for (k, tag), r in state.items() if tag == t]) for t in ("official", "capped", "rec_yds_over")}
    groups = {}
    for (k, t), r in state.items():
        if t == "group" and r.get("group"):
            groups.setdefault(r["group"], []).append(r)
    is_up = lambda v: str(v).lower() == "true"
    group_rows = []
    for name, rows in groups.items():
        row = {"group": name, "up_in_backtest": is_up(rows[0].get("group_up")), "logged": len(rows)}
        row.update(_record_units(rows))
        group_rows.append(row)
    group_rows.sort(key=lambda x: (not x["up_in_backtest"], -(x["units"] or 0)))
    up_rows = [r for (k, t), r in state.items() if t == "group" and is_up(r.get("group_up"))]
    other_rows = [r for (k, t), r in state.items() if t == "group" and not is_up(r.get("group_up"))]
    return {"generated_at": now.isoformat(), "summary": summary, "results": results,
            "groups_up_in_backtest": dict(_record_units(up_rows), logged=len(up_rows)),
            "groups_not_up_in_backtest": dict(_record_units(other_rows), logged=len(other_rows)),
            "groups": group_rows, "plays": plays,
            "note": "Each prop is followed at the book it was flagged at. Results are graded from box scores at the logged price; "
                    "NA means the player had no box-score line. 'group' plays are every 0%+ edge the model showed, filed by backtest group."}


def main(backfill_since=None):
    history = peh.build_history()
    if backfill_since:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import backtest_props_live as bpl
        state, n, last = {}, 0, None
        for when, d in bpl.snapshots(backfill_since):
            update_state(state, d.get("props") or [], when, history)
            n, last = n + 1, when
        print(f"Rebuilt from {n} saved snapshot(s) since {backfill_since}")
        now = last or datetime.now(timezone.utc)
    else:
        state = load_state()
        with open(LATEST) as f:
            d = json.load(f)
        now = _ts(d.get("generated_at")) if d.get("generated_at") else datetime.now(timezone.utc)
        update_state(state, d.get("props") or [], now, history)
    n_graded = grade_state(state, now)
    save_state(state)
    out = summarize(state, now)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    for tag, s in out["summary"].items():
        print(f"  {tag:<20} tracked {s['tracked']:<4} started {s['games_started']:<4} later look {s['with_a_later_look']:<4} "
              f"our way {s['moved_our_way']} / against {s['moved_against']} / unchanged {s['unchanged']} "
              f"| avg move {s['avg_price_move_pts']} pts | value at close {s['avg_value_at_close_pct']}%")
    print(f"  graded this run: {n_graded}")
    for tag, r in out["results"].items():
        print(f"  {tag:<20} graded {r['graded']:<4} {r['record']:<7} units {r['units']:+.2f}")
    u, o = out["groups_up_in_backtest"], out["groups_not_up_in_backtest"]
    print(f"  groups UP in backtest:     logged {u['logged']:<4} graded {u['graded']:<4} {u['record']:<7} units {u['units']:+.2f}")
    print(f"  groups not up in backtest: logged {o['logged']:<4} graded {o['graded']:<4} {o['record']:<7} units {o['units']:+.2f}")
    print(f"Saved {STATE} ({len(state)} rows) and {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill-since", default=None, help="rebuild the log from git history since this time (needs full history)")
    a = ap.parse_args()
    main(a.backfill_since)
