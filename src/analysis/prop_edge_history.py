"""
How props did in the backtest, by market, side and size of the model's edge
(added 10/2026).

One place builds this table so the dashboard and the tracker always agree on
which groups are "up":
  - the dashboard shows it next to every prop (Edge / track record column),
  - the tracker (scripts/compute_props_clv.py) logs and grades every play.

Sources
  docs/data/props_lab2.json              receptions and receiving yards, both
                                         sides, from the live PFF models
  docs/data/props_edge_finder_detail.csv every other market, the side the
                                         model leaned to (those models have
                                         not changed since that backtest)

A group counts as "up" when the model showed an edge of 0% or more, the
group won at least MIN_GROUP_UNITS units with a positive ROI, and it has at
least MIN_GROUP_BETS bets behind it. Smaller groups are shown as a small
sample rather than as up or down.
"""
import json
import os
from datetime import datetime

LAB_PATH = "docs/data/props_lab2.json"
FINDER_PATH = "docs/data/props_edge_finder_detail.csv"
FINDER_VARIANT = "first line, best book"
MIN_GROUP_BETS = 15
MIN_GROUP_UNITS = 2.0          # "up" means up by a real amount, not +0.3 units
LAB_MARKETS = {"Receptions": "receptions", "Reception Yards": "rec_yds"}
# (label, low edge %, high edge %)
BUCKETS = [("below 0%", -999, 0), ("0-4.9%", 0, 5), ("5-9.9%", 5, 10), ("10-14.9%", 10, 15),
           ("15-19.9%", 15, 20), ("20-29.9%", 20, 30), ("30%+", 30, 999)]
_LAB_LABELS = {"below 0": "below 0%"}


def _row(label, wins, losses, units, n):
    lo, hi = next((b[1], b[2]) for b in BUCKETS if b[0] == label)
    roi = units / n * 100 if n else 0.0
    return {"label": label, "lo": lo, "hi": hi, "record": f"{int(wins)}-{int(losses)}", "units": round(float(units), 1),
            "roi": round(float(roi), 1), "n": int(n),
            "up": bool(lo >= 0 and n >= MIN_GROUP_BETS and units >= MIN_GROUP_UNITS and roi > 0),
            "small": bool(n < MIN_GROUP_BETS)}


def build_history(lab_path: str = LAB_PATH, finder_path: str = FINDER_PATH):
    """{'markets': {market: {'over': [rows], 'under': [rows]}}, ...} or None
    if neither source file is there. Never raises."""
    markets, weekends = {}, set()
    try:
        if os.path.exists(finder_path):
            import pandas as pd
            df = pd.read_csv(finder_path)
            df = df[df["variant"] == FINDER_VARIANT] if "variant" in df.columns else df
            weekends.update(str(w) for w in df["weekend"].dropna().unique())
            edge = df["ev"] * 100
            for label, lo, hi in BUCKETS:
                part = df[(edge >= lo) & (edge < hi)]
                for (market, side), g in part.groupby(["market", "side"]):
                    markets.setdefault(market, {}).setdefault(side, []).append(
                        _row(label, (g["result"] == "W").sum(), (g["result"] == "L").sum(), g["units"].sum(), len(g)))
    except Exception as e:
        print(f"  [note] could not read {finder_path} ({e})")
    try:
        if os.path.exists(lab_path):
            with open(lab_path) as f:
                lab = json.load(f)
            for market, stat in LAB_MARKETS.items():
                res = lab["results"][stat]["pff"]["props_2026"]
                sides = {}
                for side in ("over", "under"):
                    rows = []
                    for b in res.get(f"{side}_edge_buckets") or []:
                        label = _LAB_LABELS.get(b.get("bucket"), b.get("bucket"))
                        if not b.get("n") or label not in {x[0] for x in BUCKETS}:
                            continue
                        wins, losses = (int(x) for x in str(b["record"]).split("-"))
                        rows.append(_row(label, wins, losses, b["units"], b["n"]))
                        weekends.update((b.get("roi_by_weekend") or {}).keys())
                    if rows:
                        sides[side] = rows
                if sides:
                    markets[market] = sides                      # the live PFF models replace the older numbers
    except Exception as e:
        print(f"  [note] could not read {lab_path} ({e})")
    if not markets:
        return None
    order = [b[0] for b in BUCKETS]
    for sides in markets.values():
        for rows in sides.values():
            rows.sort(key=lambda r: order.index(r["label"]))
    span = ""
    try:
        days = sorted(datetime.strptime(w, "%Y-%m-%d") for w in weekends)
        span = f"{days[0]:%b} {days[0].day} to {days[-1]:%b} {days[-1].day}"
    except Exception:
        pass
    return {"markets": markets, "nWeekends": len(weekends), "span": span, "minBets": MIN_GROUP_BETS, "minUnits": MIN_GROUP_UNITS}


def find_group(history, market, side, edge_pct, capped=False):
    """The backtest row for a play, or None. A play capped at first flag is
    judged against the 30%+ group whatever its edge is now."""
    if not history or edge_pct is None or side not in ("over", "under"):
        return None
    rows = ((history.get("markets") or {}).get(market) or {}).get(side) or []
    for r in rows:
        if (r["lo"] >= 30) if capped else (r["lo"] <= edge_pct < r["hi"]):
            return r
    return None


def group_label(market, side, row):
    return f"{market} {side} {row['label']}"
