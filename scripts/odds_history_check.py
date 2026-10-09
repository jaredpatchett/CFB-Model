#!/usr/bin/env python3
"""
Odds history check (added 10/2026). Run once, from the Actions tab.

Answers the one question that decides whether the receptions props model can
be backtested for PROFIT on past seasons, not just for accuracy:

  Does this Odds API key return HISTORICAL player-prop lines for college
  football, and for which seasons?

Background: The Odds API sells historical odds on paid plans only. Player
props ("additional markets") exist in its history from 2023-05-03 onward.
The pipeline has only ever used the live endpoints, and
scripts/backtest_props_live.py notes that no past-season prop lines exist --
which is true of what is saved in this repo, but says nothing about what the
key could pull. This script finds out.

What it does, for one October Saturday in each of 2023, 2024 and 2025:
  1. asks for the games on the board that morning         (1 credit)
  2. asks for that morning's receptions lines on up to two
     of those games, biggest programs first               (10 credits each,
     and nothing if the game had no props)
So a full run costs about 33 credits and at most 63. If the plan has no
historical access, the first call is refused and the run costs nothing.

Writes docs/data/odds_history_check.json: counts and sportsbook names only.
No prices or lines are written.

Usage:
  python scripts/odds_history_check.py
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests

import config

OUT = "docs/data/odds_history_check.json"
MARKET = "player_receptions"
REGION = "us"
# A mid-season Saturday per season, read at 14:00 UTC (10am Eastern): after
# books have posted props, before the noon kickoffs.
SAMPLES = [(2023, "2023-10-07T14:00:00Z"), (2024, "2024-10-05T14:00:00Z"), (2025, "2025-10-04T14:00:00Z")]
MAX_GAMES_PER_SAMPLE = 2
# Books post college props for the biggest games first, so those are the ones
# to try: a sample that only looked at a small-conference game could come back
# empty and wrongly read as "no history".
# Full names as The Odds API writes them, so "Texas" can't match Texas State
# or "Michigan" Western Michigan.
BIG_PROGRAMS = {
    "Alabama Crimson Tide", "Georgia Bulldogs", "Ohio State Buckeyes", "Michigan Wolverines", "Texas Longhorns",
    "Oregon Ducks", "Notre Dame Fighting Irish", "Penn State Nittany Lions", "LSU Tigers", "Tennessee Volunteers",
    "Oklahoma Sooners", "Florida State Seminoles", "USC Trojans", "Clemson Tigers", "Ole Miss Rebels",
    "Florida Gators", "Auburn Tigers", "Miami Hurricanes", "Washington Huskies", "Wisconsin Badgers",
    "Texas A&M Aggies", "Iowa Hawkeyes", "Nebraska Cornhuskers", "Missouri Tigers",
}


def call(path: str, params: dict) -> dict:
    """One GET. Never raises: returns {ok, status, body, quota} so a refusal
    can be reported in plain words instead of as a stack trace."""
    params = dict(params, apiKey=config.ODDS_API_KEY)
    try:
        resp = requests.get(f"{config.ODDS_API_BASE_URL}{path}", params=params, timeout=30)
    except requests.RequestException as e:
        return {"ok": False, "status": None, "body": {"message": f"network error: {type(e).__name__}"}, "quota": {}}
    try:
        body = resp.json()
    except ValueError:
        body = {"message": (resp.text or "")[:200]}
    quota = {k: resp.headers.get(f"x-requests-{k}") for k in ("remaining", "used", "last")}
    return {"ok": resp.status_code == 200, "status": resp.status_code, "body": body, "quota": quota}


def refusal(res: dict) -> str:
    body = res.get("body") if isinstance(res.get("body"), dict) else {}
    return str(body.get("message") or body.get("error_code") or "no message")[:240]


def big_first(events: list) -> list:
    def score(ev):
        return -sum(1 for side in ("home_team", "away_team") if ev.get(side) in BIG_PROGRAMS)
    return sorted(events, key=score)


def receptions_books(event_odds: dict) -> dict:
    """{sportsbook key: number of players with a receptions line}."""
    out = {}
    for bk in (event_odds or {}).get("bookmakers") or []:
        for m in bk.get("markets") or []:
            if m.get("key") == MARKET:
                players = {o.get("description") for o in m.get("outcomes") or [] if o.get("description")}
                if players:
                    out[bk.get("key")] = len(players)
    return out


def check_sample(season: int, when: str) -> dict:
    start = datetime.fromisoformat(when.replace("Z", "+00:00"))
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    out = {"season": season, "asked_for": when, "credits_spent": 0}
    ev = call(f"/historical/sports/{config.ODDS_API_NCAAF_KEY}/events",
              {"date": when, "commenceTimeFrom": start.strftime(fmt),
               "commenceTimeTo": (start + timedelta(hours=20)).strftime(fmt)})
    out["quota"] = ev["quota"]
    if not ev["ok"]:
        out.update(events_ok=False, status=ev["status"], message=refusal(ev))
        return out
    out["credits_spent"] += int(ev["quota"].get("last") or 0)
    events = (ev["body"] or {}).get("data") or []
    out.update(events_ok=True, snapshot=(ev["body"] or {}).get("timestamp"), games_on_board=len(events), games_tried=[])
    for game in big_first(events)[:MAX_GAMES_PER_SAMPLE]:
        od = call(f"/historical/sports/{config.ODDS_API_NCAAF_KEY}/events/{game.get('id')}/odds",
                  {"date": when, "regions": REGION, "markets": MARKET, "oddsFormat": "american"})
        out["quota"] = od["quota"] if od["quota"].get("remaining") is not None else out["quota"]
        tried = {"game": f"{game.get('away_team')} at {game.get('home_team')}"}
        if not od["ok"]:
            tried.update(ok=False, status=od["status"], message=refusal(od))
        else:
            out["credits_spent"] += int(od["quota"].get("last") or 0)
            books = receptions_books((od["body"] or {}).get("data") or {})
            tried.update(ok=True, books_with_receptions_lines=books, players_priced=max(books.values()) if books else 0)
        out["games_tried"].append(tried)
        if tried.get("players_priced"):
            break                      # found props for this season; no need to spend on a second game
    out["props_found"] = any(t.get("players_priced") for t in out["games_tried"])
    return out


def main():
    config.require_keys("ODDS_API_KEY")
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "market": MARKET, "region": REGION, "samples": []}
    for season, when in SAMPLES:
        s = check_sample(season, when)
        result["samples"].append(s)
        if not s.get("events_ok") and s.get("status") in (401, 403, 422):
            break                      # the plan itself is refused; asking again for other seasons would say the same
    samples = result["samples"]
    access = any(s.get("events_ok") for s in samples)
    seasons_with_props = [s["season"] for s in samples if s.get("props_found")]
    result["historical_access"] = access
    result["seasons_with_receptions_props"] = seasons_with_props
    result["credits_spent"] = sum(s.get("credits_spent", 0) for s in samples)
    result["credits_remaining"] = next((s["quota"].get("remaining") for s in reversed(samples) if (s.get("quota") or {}).get("remaining")), None)
    boards = [s["games_on_board"] for s in samples if s.get("games_on_board")]
    if boards:
        # One reading of one market in one region costs 10 credits a game.
        result["credits_per_saturday_one_reading"] = 10 * round(sum(boards) / len(boards))
    tried = [t for s in samples for t in s.get("games_tried", [])]
    refused = [t for t in tried if not t.get("ok")]
    if not access:
        first = samples[0] if samples else {}
        if first.get("status") is None:
            # Never got an answer at all -- says nothing about the plan.
            result["verdict"] = f"Could not reach The Odds API ({first.get('message')}). Nothing learned; run it again."
        else:
            result["verdict"] = (f"No historical access on this key (status {first.get('status')}: {first.get('message')}). "
                                 f"The Odds API only serves historical odds on paid plans.")
    elif not tried:
        result["verdict"] = ("The key can read historical odds, but no games were on the board for the dates asked, "
                             "so prop lines were never tried. Nothing learned about props.")
    elif not seasons_with_props and len(refused) == len(tried):
        result["verdict"] = (f"The key can read historical game listings, but every request for prop lines was refused "
                             f"(status {refused[0].get('status')}: {refused[0].get('message')}). "
                             f"Historical player props are not available on this plan.")
    elif not seasons_with_props:
        result["verdict"] = ("The key can read historical odds, but no receptions props came back for the games tried. "
                             "Past-season prop lines cannot be relied on from this source.")
    else:
        result["verdict"] = (f"Historical receptions props are available for {', '.join(str(x) for x in seasons_with_props)}. "
                             f"A profit backtest on past seasons is possible.")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Odds history check ===")
    for s in samples:
        if not s.get("events_ok"):
            print(f"{s['season']}: refused (status {s.get('status')}): {s.get('message')}")
            continue
        print(f"{s['season']}: {s['games_on_board']} games on the board at {s.get('snapshot')}")
        for t in s["games_tried"]:
            if not t.get("ok"):
                print(f"   {t['game']}: refused (status {t.get('status')}): {t.get('message')}")
            elif t["players_priced"]:
                print(f"   {t['game']}: receptions lines for {t['players_priced']} players at {len(t['books_with_receptions_lines'])} book(s): "
                      f"{', '.join(sorted(t['books_with_receptions_lines']))}")
            else:
                print(f"   {t['game']}: no receptions lines")
    print(f"\nVERDICT: {result['verdict']}")
    print(f"Credits spent: {result['credits_spent']}   remaining: {result['credits_remaining']}")
    if result.get("credits_per_saturday_one_reading"):
        print(f"Rough cost to pull one reading of receptions lines for every game on a Saturday: "
              f"{result['credits_per_saturday_one_reading']} credits")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
