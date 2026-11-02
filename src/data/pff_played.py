"""
Did a player take an offensive snap? (11/2026)

Books void a player prop only when the player takes no snaps. CFBD box
scores list only players who recorded a stat, so "no box-score line" can mean
"sat" (void) or "played, never touched the ball" (actual 0). PFF's weekly
offense summary has every player who took an offensive snap, which settles it.

played_status(season, items) -> list of "yes" / "no" / "pending"
  items: dicts with player, kick (UTC Timestamp), teammates (names of his
  teammates in that game's CFBD box score).
  yes      he has offensive snaps that week               -> grade as 0
  no       his team's game is charted, he has no snaps     -> void
  pending  PFF has no charted data for that game yet (or PFF is unreachable)

Only yes / no / pending leave this module: raw PFF numbers are never
returned or written (licensed data, public repo). Special-teams-only snaps
are not covered (PFF's special summary is empty), so a rare special-teams-
only player is treated as a void.
"""
import re
import time

import pandas as pd

from src.data import pff_client as pff

MATCH = pd.Timedelta(hours=36)


def norm(s):
    s = str(s or "").lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", s)
    return re.sub(r"[^a-z]", "", s)


def _offense(season, week):
    for attempt in range(3):
        try:
            payload = pff._get("/v1/facet/offense/summary", {"league": pff.LEAGUE, "season": season, "week": week})
            o = pd.DataFrame(pff._rows(payload))
            if o.empty:
                return o
            name_col = "player" if "player" in o.columns else next(c for c in o.columns if "name" in c)
            snap_col = "snap_counts_total" if "snap_counts_total" in o.columns else "snap_counts_offense"
            snaps = pd.to_numeric(o[snap_col], errors="coerce").fillna(0)
            return pd.DataFrame({"n": o[name_col].map(norm), "played": snaps > 0})
        except pff.PFFError as e:
            if e.status in (401, 403):
                raise
            return pd.DataFrame()
        except Exception:
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def played_status(season, items):
    if not items:
        return []
    try:
        kicks = [i["kick"] for i in items]
        weeks = {}
        for w in pff.MODEL_WEEKS:
            try:
                g = pff.get_games(season, w)
            except Exception:
                continue
            if g.empty:
                continue
            ks = pd.to_datetime(g["start"], utc=True, errors="coerce").dropna()
            if any((ks - k).abs().min() <= MATCH for k in kicks):
                weeks[w] = list(ks)
        data = {w: _offense(season, w) for w in weeks}
    except Exception as e:
        print(f"  [note] PFF played check unavailable ({type(e).__name__}); no-box-score props stay pending")
        return ["pending"] * len(items)

    out = []
    for i in items:
        wk = next((w for w, ks in weeks.items() if any(abs(k - i["kick"]) <= MATCH for k in ks)), None)
        d = data.get(wk)
        if d is None or d.empty:
            out.append("pending")
            continue
        me = d[d["n"] == norm(i["player"])]
        if not me.empty and me["played"].any():
            out.append("yes")
        elif d["n"].isin({norm(t) for t in i.get("teammates") or []}).any():
            out.append("no")
        else:
            out.append("pending")
    return out
