#!/usr/bin/env python3
"""
PFF data check (added 10/2026). Run once after adding the PFF Pro API key.

Answers the three questions that decide whether routes run can go into the
props model:
  1. Does the key work, and is the account entitled to data?
  2. Is there routes-run data per player PER WEEK for college football?
  3. Does it go back to 2022 (the first season the props models train on)?
Plus one integration question: how many CFBD pass-catchers can be matched to
a PFF row by name (the two sources share no player id)?

Writes docs/data/pff_check.json. That file holds ONLY metadata -- column
names, row counts, coverage percentages. No PFF stat values and no account
details are written, because this repo is public.

Usage:
  python scripts/pff_check.py
"""
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd

from src.data import pff_client as pff
from src.features.live_player_features import _normalize_name

OUT = "docs/data/pff_check.json"
FIRST_TRAINING_SEASON = 2022
HISTORY_SAMPLES = [(2022, 5), (2023, 5), (2024, 5), (2025, 5)]
NAME_COLS = ("player", "player_name", "name")
TEAM_COLS = ("team_name", "team", "franchise_id")


def describe(df: pd.DataFrame) -> dict:
    """Metadata only: what columns came back and how complete the routes are."""
    if df is None or df.empty:
        return {"rows": 0}
    cols = [c for c in df.columns if c not in ("season", "week")]
    route_cols = [c for c in cols if "route" in c.lower() or c.lower() == "yprr"]
    target_cols = [c for c in cols if "target" in c.lower()]
    out = {"rows": int(len(df)), "n_columns": len(cols), "columns": sorted(cols),
           "route_columns": route_cols, "target_columns": target_cols}
    team_col = next((c for c in TEAM_COLS if c in df.columns), None)
    if team_col:
        out["teams"] = int(df[team_col].nunique())
    # "routes" is the count we want; fall back to the first route-like column.
    count_col = "routes" if "routes" in df.columns else next((c for c in route_cols if "rate" not in c and "grade" not in c), None)
    if count_col:
        vals = pd.to_numeric(df[count_col], errors="coerce")
        out["routes_count_column"] = count_col
        out["pct_rows_with_routes"] = round(float((vals > 0).mean()) * 100, 1)
    return out


def latest_week_with_data(season: int, weeks: list):
    """Most recent regular-season week that already has charted rows."""
    for wk in sorted((w for w in weeks if 0 <= w <= 16), reverse=True):
        try:
            df = pff.get_receiving_summary(season, wk)
        except pff.PFFError as e:
            print(f"  week {wk}: {e}")
            continue
        if not df.empty:
            return wk, df
    return None, pd.DataFrame()


def name_match(season: int, week: int, pff_by_week: dict) -> dict:
    """Share of CFBD players with a catch that week whose name appears in the
    PFF rows. PFF and CFBD number weeks slightly differently (PFF has a Week
    0), so the neighbouring PFF weeks are tried too and the best is kept."""
    from src.data import cfbd_client as cfbd
    from src.features.player_features import pivot_player_game_stats

    wide = pivot_player_game_stats(cfbd.get_player_game_stats(season, week), cfbd.get_games(season))
    catchers = wide[wide["receptions"] > 0]
    names = set(catchers["player"].map(_normalize_name))
    if not names:
        return {"cfbd_players_with_catch": 0}
    best = {"cfbd_season": season, "cfbd_week": week, "cfbd_players_with_catch": len(names)}
    for wk in (week, week - 1, week + 1):
        df = pff_by_week.get((season, wk))
        if df is None:
            try:
                df = pff.get_receiving_summary(season, wk)
            except pff.PFFError:
                continue
        name_col = next((c for c in NAME_COLS if c in df.columns), None)
        if df.empty or not name_col:
            continue
        pff_names = set(df[name_col].map(_normalize_name))
        pct = round(len(names & pff_names) / len(names) * 100, 1)
        if pct > best.get("pct_matched_by_name", -1):
            best.update({"pff_week": wk, "pct_matched_by_name": pct, "pff_name_column": name_col})
    return best


def main():
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "ok": False}
    try:
        # 1) Key and entitlement. Email / account id are deliberately not saved.
        who = pff.whoami()
        result["account"] = {k: who.get(k) for k in ("tier", "entitled", "entitlement_reason", "credential")}
        print(f"Key accepted: tier={who.get('tier')} entitled={who.get('entitled')} credential={who.get('credential')}")
        if not who.get("entitled"):
            result["problem"] = f"Key is valid but the account is not entitled to data ({who.get('entitlement_reason')})."
            return result

        # 2) Which college seasons and weeks exist.
        cal = pff.ncaa_calendar()
        result["ncaa_seasons"] = cal["seasons"]
        result["history_back_to_2022"] = bool(cal["seasons"]) and min(cal["seasons"]) <= FIRST_TRAINING_SEASON
        print(f"College seasons available: {cal['seasons']}")

        # 3) This season: latest charted week, and what columns come back.
        by_week = {}
        season_now = max(cal["seasons"]) if cal["seasons"] else datetime.now(timezone.utc).year
        wk, df = latest_week_with_data(season_now, cal["weeks"] or list(range(0, 17)))
        result["current"] = dict(describe(df), season=season_now, latest_week_with_data=wk)
        if wk is not None:
            by_week[(season_now, wk)] = df
        print(f"{season_now}: latest week with data = {wk}, rows = {len(df)}, "
              f"route columns = {result['current'].get('route_columns')}")

        # 4) Past seasons: is week-level routes data really there?
        result["history"] = []
        for season, week in HISTORY_SAMPLES:
            try:
                hdf = pff.get_receiving_summary(season, week)
                by_week[(season, week)] = hdf
                d = describe(hdf)
                d.pop("columns", None)
                result["history"].append(dict(d, season=season, week=week))
                print(f"{season} week {week}: rows = {d.get('rows')}, with routes = {d.get('pct_rows_with_routes')}%")
            except pff.PFFError as e:
                result["history"].append({"season": season, "week": week, "error": str(e)})
                print(f"{season} week {week}: {e}")

        # 5) Can PFF rows be joined to the CFBD box scores by player name?
        try:
            result["name_match"] = name_match(2025, 5, by_week)
            print(f"Name match vs CFBD: {result['name_match']}")
        except Exception as e:  # CFBD key missing, etc. -- not a PFF problem
            result["name_match"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"Name match skipped: {e}")

        cur = result["current"]
        hist_ok = [h for h in result["history"] if h.get("rows", 0) > 0 and h.get("pct_rows_with_routes", 0) > 0]
        result["routes_per_week_available"] = bool(cur.get("routes_count_column")) and cur.get("rows", 0) > 0
        result["routes_history_available"] = len(hist_ok) == len(HISTORY_SAMPLES)
        result["ok"] = result["routes_per_week_available"] and result["routes_history_available"]
        # One request per week returns every player, so a full backfill is small.
        result["backfill_requests_estimate"] = (2025 - FIRST_TRAINING_SEASON + 1) * 21
    except pff.PFFError as e:
        result["problem"] = str(e)
        result["error_code"], result["error_reason"] = e.code, e.reason
        print(f"PFF API problem: {e}")
    except EnvironmentError as e:
        result["problem"] = str(e)
        print(str(e))
    return result


if __name__ == "__main__":
    res = main()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nSaved {OUT}")
    print("RESULT:", "routes data is usable" if res.get("ok") else f"NOT ready -- {res.get('problem') or 'see details above'}")
    sys.exit(0 if res.get("ok") or "problem" not in res else 1)
