#!/usr/bin/env python3
"""
Phone alerts (11/2026): after each model run, push what changed to Rod's
phone through ntfy (https://ntfy.sh -- free app, no account).

Sends ONE notification per run, and only when something changed:

  Props    a new 🔥 A-grade official prop (new play, or it moved up to A)
           an A-grade prop that is no longer A (dropped to B/C, or pulled
           from the official list) before kickoff
  Spreads  NEW plays on the Bet Card, and DROPPED plays (with why), from
           card_watch.py's log of this run

Reads docs/data/prop_grade_log.json (written by build_dashboard.py) and
docs/data/card_log.json (written by card_watch.py), so it runs after both.

The ntfy topic is the secret NTFY_TOPIC. Anyone who knows a topic name can
read it, and this repo is public, so it is never written in the code. With
no topic set this prints the message and sends nothing. Never fails the run.

Usage:
  python scripts/push_alerts.py            # send (if NTFY_TOPIC is set)
  python scripts/push_alerts.py --dry-run  # print only
  python scripts/push_alerts.py --test     # send a test notification
"""
import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

GRADE_LOG = "docs/data/prop_grade_log.json"
CARD_LOG = "docs/data/card_log.json"
MAX_LINES = 14


def _dt(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _price(p):
    try:
        p = float(p)
        return f"+{p:g}" if p > 0 else f"{p:g}"
    except (TypeError, ValueError):
        return ""


def _side(side):
    return "O" if side == "over" else "U"


def _mkt(m):
    return {"Reception Yards": "rec yds", "Rush Yards": "rush yds", "Receptions": "rec"}.get(m, str(m).lower())


def prop_changes(now):
    """(new A plays, plays no longer A) for the latest run in the grade log."""
    if not os.path.exists(GRADE_LOG):
        return [], []
    log = json.load(open(GRADE_LOG))
    run_at = log.get("updated")
    new_a, lost_a = [], []
    for p in (log.get("plays") or {}).values():
        h = p.get("h") or []
        if not h:
            continue
        start = _dt(p.get("start"))
        if start is not None and start <= now:
            continue
        last = h[-1]
        this_run = last[0] == run_at
        prev = h[-2] if this_run and len(h) > 1 else (None if this_run else last)
        label = (f"{p.get('player')} {_side(p.get('side'))}{last[2]:g} {_mkt(p.get('market'))}"
                 if isinstance(last[2], (int, float)) else f"{p.get('player')} {_mkt(p.get('market'))}")
        opp = p.get("opponent") or ""
        if this_run and last[1] == "A" and (prev is None or prev[1] != "A"):
            new_a.append(f"🔥 {label} ({_price(last[3])}) vs {opp}".replace(" ()", ""))
        elif prev is not None and prev[1] == "A" and (not this_run or last[1] != "A"):
            why = f"now {last[1]}" if this_run else "off the official list"
            lost_a.append(f"❌ {label} vs {opp}: {why}")
    return new_a, lost_a


def spread_changes(now):
    """(new, dropped) on the spread Bet Card, from card_watch's latest run.
    Skipped if that run is older than the grade log's (card watch failed)."""
    if not os.path.exists(CARD_LOG):
        return [], []
    runs = (json.load(open(CARD_LOG)).get("runs") or [])
    if not runs:
        return [], []
    r = runs[-1]
    at = _dt(r.get("at"))
    if at is None or (now - at).total_seconds() > 3 * 3600:
        return [], []
    by_key = {p["key"]: p for p in r.get("plays") or []}
    new = [f"✅ {by_key[k]['label']} ({by_key[k]['away']} at {by_key[k]['home']})" for k in r.get("new") or [] if k in by_key]
    dropped = [f"⛔ {p['label']} ({p['away']} at {p['home']}): {p.get('why', '')}" for p in r.get("dropped") or []]
    return new, dropped


def build_message(now):
    new_a, lost_a = prop_changes(now)
    new_s, drop_s = spread_changes(now)
    lines = []
    if new_s or drop_s:
        lines.append("SPREADS")
        lines += new_s + drop_s
    if new_a or lost_a:
        lines.append("PROPS")
        lines += new_a + lost_a
    if not lines:
        return None, None
    if len(lines) > MAX_LINES:
        extra = len(lines) - MAX_LINES + 1
        lines = lines[:MAX_LINES - 1] + [f"...and {extra} more on the dashboard"]
    bits = []
    if new_s:
        bits.append(f"{len(new_s)} new spread{'s' if len(new_s) > 1 else ''}")
    if new_a:
        bits.append(f"{len(new_a)} new 🔥 A prop{'s' if len(new_a) > 1 else ''}")
    if drop_s or lost_a:
        bits.append(f"{len(drop_s) + len(lost_a)} dropped")
    return "CFB: " + ", ".join(bits), "\n".join(lines)


def send(topic, title, body, click=None, tags="football"):
    req = urllib.request.Request(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), method="POST")
    # ntfy reads headers as latin-1; emoji go in the body, the title is sent RFC 2047-encoded.
    import base64
    req.add_header("Title", "=?UTF-8?B?" + base64.b64encode(title.encode("utf-8")).decode() + "?=")
    req.add_header("Tags", tags)
    if click:
        req.add_header("Click", click)
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    topic = (os.environ.get("NTFY_TOPIC") or "").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "jaredpatchett/CFB-Model")
    owner, name = repo.split("/", 1)
    click = f"https://{owner.lower()}.github.io/{name}/dashboard.html"
    now = datetime.now(timezone.utc)
    if a.test:
        title, body = "CFB alerts are on ✅", "You'll get a ping after each model run when a play joins or drops off the Bet Card, or a prop turns 🔥 A."
    else:
        title, body = build_message(now)
        if not title:
            print("Alerts: nothing changed this run -- no notification.")
            return
    print(f"Alert:\n  {title}\n  " + body.replace("\n", "\n  "))
    if a.dry_run:
        return
    if not topic:
        print("[note] NTFY_TOPIC secret not set -- not sent.")
        return
    print(f"Sent (HTTP {send(topic, title, body, click)})")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[note] alerts skipped ({type(e).__name__}: {e})")
    sys.exit(0)
