#!/usr/bin/env python3
"""
Card watch (11/2026): what changed on the spread Bet Card since the last run.

Runs after build_dashboard.py. Computes the spread Bet Card exactly as the
dashboard does (it runs the dashboard's own ModelMath code from
docs/dashboard.html in node, on the same MODEL_DATA), then compares it with
the card logged by the previous run:

  NEW      on the card now, not last run
  DROPPED  on the card last run, not now (game not kicked off yet), with why:
           the edge at the current line fell under the threshold, or it was
           ranked out of the top 10
  KEPT     still on the card -- still a bet at the current line

Every play on the card is, by construction, still a bet at the line shown:
the card is rebuilt from the current line every run.

Writes docs/data/card_log.json (last 30 runs; the dashboard reads the
previous one to tag NEW plays and list dropped ones) and prints a summary.
Never fails the workflow: on any error it prints a note and exits 0.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

HTML = "docs/dashboard.html"
LOG = "docs/data/card_log.json"
KEEP_RUNS = 30

NODE_JS = r"""
const fs = require('fs');
const src = fs.readFileSync(process.argv[2], 'utf8');
const D = eval('(' + fs.readFileSync(process.argv[3], 'utf8') + ')');   // the page embeds NaN, as JS
global.window = {};
eval(src);
const M = window.ModelMath;
const abbr = {};
(D.teams || []).forEach(t => { if (t && t.name) abbr[t.name] = t.abbr || t.name; });
const abbrOf = n => abbr[n] || n;
const opts = { market: 'Spread', minEdge: D.meta.minEdge, abbrOf: abbrOf };
const card = M.buildBetCard(D.games, opts);
const all = D.games.map(g => M.priceGame(g, opts));
const out = {
  card: card.map(c => ({
    key: c.game.home + '|' + c.game.away + '|' + c.game.kickoffIso,
    label: c.playLabel, side: abbrOf(c.side), home: abbrOf(c.game.home), away: abbrOf(c.game.away),
    kickoff: c.game.kickoffIso, marketSpread: c.game.marketSpread, modelSpread: c.game.modelSpread,
    prob: Math.round(c.sideProb * 1000) / 1000 })),
  status: Object.fromEntries(all.map(p => [p.game.home + '|' + p.game.away + '|' + p.game.kickoffIso,
    { qualifies: !!p.qualifies, label: p.playLabel, marketSpread: p.game.marketSpread,
      prob: p.sideProb == null ? null : Math.round(p.sideProb * 1000) / 1000 }]))
};
process.stdout.write(JSON.stringify(out));
"""


def compute_card():
    html = open(HTML, encoding="utf-8").read()
    a = html.index("(function (root) {")
    b = html.index("})(window);", a) + len("})(window);")
    i = html.index("window.MODEL_DATA = ") + len("window.MODEL_DATA = ")
    data, _ = json.JSONDecoder().raw_decode(html[i:])
    with tempfile.TemporaryDirectory() as td:
        src, dat, js = (os.path.join(td, n) for n in ("mm.js", "data.json", "run.js"))
        open(src, "w").write(html[a:b])
        json.dump(data, open(dat, "w"))
        open(js, "w").write(NODE_JS)
        out = subprocess.run(["node", js, src, dat], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(out.stderr[-500:])
    return json.loads(out.stdout), data


def main():
    now = datetime.now(timezone.utc)
    res, data = compute_card()
    log = {"runs": []}
    if os.path.exists(LOG):
        try:
            log = json.load(open(LOG))
        except Exception:
            log = {"runs": []}
    prev = log["runs"][-1] if log.get("runs") else None
    cur_keys = {p["key"] for p in res["card"]}
    prev_plays = {p["key"]: p for p in (prev or {}).get("plays", [])}
    new = [p for p in res["card"] if prev is not None and p["key"] not in prev_plays]
    dropped = []
    for k, p in prev_plays.items():
        if k in cur_keys:
            continue
        try:
            future = datetime.fromisoformat(p["kickoff"].replace("Z", "+00:00")) > now
        except Exception:
            future = False
        if not future:
            continue
        st = res["status"].get(k)
        why = ("off the board (no line now)" if st is None else
               "still a play, ranked out of the top 10" if st["qualifies"] else
               "the edge at the current line is under the threshold")
        dropped.append({**p, "why": why, "nowSpread": (st or {}).get("marketSpread")})
    run = {"at": now.isoformat(), "plays": res["card"], "new": [p["key"] for p in new], "dropped": dropped}
    log["runs"] = (log.get("runs") or [])[-(KEEP_RUNS - 1):] + [run]
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    json.dump(log, open(LOG, "w"), indent=1)
    print(f"Spread Bet Card: {len(res['card'])} plays"
          + (f" (vs last run {prev['at'][:16]})" if prev else " (first logged run)"))
    for p in res["card"]:
        print(f"  {'NEW  ' if p in new else 'KEPT '} {p['label']:<14} {p['away']} at {p['home']}  {p['prob']:.1%}")
    for p in dropped:
        print(f"  DROPPED {p['label']:<14} {p['away']} at {p['home']}: {p['why']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[note] card watch skipped ({type(e).__name__}: {e})")
    sys.exit(0)
