#!/usr/bin/env python3
"""
PFF reports check (added 10/2026). Run once before using PFF data for the
spread model.

The first PFF check only looked at the receiving report. This one walks
PFF's own published list of endpoints (its OpenAPI spec) and, for college
football, tries every report a key can pull without naming a single player:
the league-wide leaderboards (passing, rushing, receiving, blocking,
defense, special teams, passing under pressure, ...) and the team tables.

For each one it records
  - whether the call worked,
  - how many rows came back for a recent week and for the same week of 2022
    (the first season the models train on),
  - the column names.

Writes docs/data/pff_reports_check.json. Metadata only -- endpoint names,
row counts and column names. No PFF stat values and no account details are
saved, because this repo is public.

Usage:
  python scripts/pff_reports_check.py
"""
import json
import os
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests

from src.data import pff_client as pff

OUT = "docs/data/pff_reports_check.json"
SPEC_URLS = ["https://api.pff.com/openapi.json", "https://developer.pff.com/openapi.json"]
LEAGUE = "ncaa"
HISTORY_SEASON = 2022
MAX_ENUM = 30                 # most values tried for a required "which report" style parameter
MAX_CALLS = 400               # hard stop; the account allows 100 requests a minute
# Parameters this check knows how to fill in. Anything else that is REQUIRED
# (a player id, a game id) means the endpoint needs a specific target and is
# listed as skipped rather than guessed at.
SEASON_NAMES = {"season", "seasons", "year"}
WEEK_NAMES = {"week", "weeks"}
LEAGUE_NAMES = {"league"}
TEAM_NAMES = {"team", "slug", "team_slug", "teamslug", "franchise", "franchise_id", "franchiseid", "team_id", "teamid"}


def load_spec():
    for url in SPEC_URLS:
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.json(), url
        except Exception:
            continue
    raise RuntimeError("could not download PFF's OpenAPI spec")


def resolve(spec, obj):
    """Follow a {'$ref': '#/components/...'} pointer (one level is enough here)."""
    seen = 0
    while isinstance(obj, dict) and "$ref" in obj and seen < 5:
        node = spec
        for part in obj["$ref"].lstrip("#/").split("/"):
            node = node.get(part, {}) if isinstance(node, dict) else {}
        obj, seen = node, seen + 1
    return obj if isinstance(obj, dict) else {}


def operations(spec):
    """Every GET endpoint: path, name, and its parameters with enums resolved."""
    out = []
    for path, item in (spec.get("paths") or {}).items():
        item = resolve(spec, item)
        op = item.get("get")
        if not isinstance(op, dict):
            continue
        params = []
        for raw in (item.get("parameters") or []) + (op.get("parameters") or []):
            p = resolve(spec, raw)
            schema = resolve(spec, p.get("schema") or {})
            enum = schema.get("enum") or resolve(spec, schema.get("items") or {}).get("enum")
            params.append({"name": p.get("name"), "in": p.get("in"), "required": bool(p.get("required")) or p.get("in") == "path",
                           "enum": list(enum) if enum else None})
        out.append({"path": path, "name": op.get("operationId") or path, "tags": op.get("tags") or [],
                    "summary": (op.get("summary") or "")[:120], "params": [p for p in params if p["name"]]})
    return out


def plan(op, season, week, team_value):
    """How to call an endpoint for college football: (list of (path, query,
    label) calls) or (None, reason it is skipped)."""
    path, query, expand = op["path"], {}, None
    for p in op["params"]:
        key = re.sub(r"[^a-z_]", "", p["name"].lower())
        if key in LEAGUE_NAMES:
            value = LEAGUE
        elif key in SEASON_NAMES:
            value = season
        elif key in WEEK_NAMES:
            value = week
        elif key in TEAM_NAMES and p["required"]:
            if team_value is None:
                return None, "needs a team and no team list was available"
            value = team_value.get(key) or team_value.get("slug") or team_value.get("franchise_id")
            if value is None:
                return None, "needs a team id this check could not find"
        elif p["required"] and p["enum"]:
            if p["enum"] and LEAGUE in p["enum"] and len(p["enum"]) <= 6:
                value = LEAGUE
            elif expand is None:
                expand = p
                continue
            else:
                return None, "needs more than one choice parameter"
        elif p["required"]:
            return None, f"needs a specific {p['name']}"
        else:
            continue                                  # optional and not one we fill
        if p["in"] == "path":
            path = path.replace("{" + p["name"] + "}", str(value))
        else:
            query[p["name"]] = value
    if "{" in path and expand is None:
        return None, "has a path parameter this check does not fill"
    if expand is None:
        return [(path, query, None)], None
    calls = []
    for v in expand["enum"][:MAX_ENUM]:
        pth = path.replace("{" + expand["name"] + "}", str(v)) if expand["in"] == "path" else path
        q = dict(query) if expand["in"] == "path" else dict(query, **{expand["name"]: v})
        if "{" in pth:
            return None, "has a path parameter this check does not fill"
        calls.append((pth, q, str(v)))
    return calls, None


def shape(payload):
    """Row count and column names of a response, whatever its layout."""
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):          # PFF's team tables
        cols = [c.get("key") for c in (payload.get("columns") or []) if isinstance(c, dict) and c.get("key")]
        rows = payload["rows"]
        if not cols and rows and isinstance(rows[0], dict):
            cols = sorted(rows[0].keys())
        return len(rows), cols
    rows = payload if isinstance(payload, list) else next((v for v in (payload or {}).values() if isinstance(v, list)), None)
    if rows is None:
        return None, sorted(payload.keys()) if isinstance(payload, dict) else []
    cols = sorted({k for r in rows[:50] if isinstance(r, dict) for k in r.keys()})
    return len(rows), cols


def call(path, query, counter):
    if counter["n"] >= MAX_CALLS:
        return {"error": "call limit for this check reached"}
    counter["n"] += 1
    try:
        n, cols = shape(pff._get(path, query))
        return {"rows": n, "n_columns": len(cols), "columns": cols}
    except pff.PFFError as e:
        return {"error": f"{e.status} {e.code or ''} {e.reason or ''}".strip()}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:160]}


def find_team(ops, season, counter):
    """One college team's identifiers, for endpoints that need a team."""
    for op in ops:
        if "director" in op["name"].lower() or op["path"].rstrip("/").endswith("/teams"):
            calls, _ = plan(op, season, None, None)
            for path, query, _ in calls or []:
                if counter["n"] >= MAX_CALLS:
                    return None
                counter["n"] += 1
                try:
                    payload = pff._get(path, query)
                except Exception:
                    continue
                rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else \
                    next((v for v in (payload or {}).values() if isinstance(v, list)), [])
                for r in rows:
                    if not isinstance(r, dict):
                        continue
                    slug = r.get("slug")
                    fid = r.get("franchiseId") or r.get("franchise_id") or r.get("id")
                    if slug or fid:
                        t = {"slug": slug, "franchise_id": fid}
                        for k in TEAM_NAMES:
                            t[k] = slug if "slug" in k or k == "team" else fid
                        if slug is None:
                            for k in TEAM_NAMES:
                                t[k] = fid
                        return t
    return None


def main():
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "league": LEAGUE, "ok": False}
    counter = {"n": 0}
    try:
        who = pff.whoami()
        if not who.get("entitled"):
            result["problem"] = f"key is valid but not entitled to data ({who.get('entitlement_reason')})"
            return result
        spec, url = load_spec()
        ops = operations(spec)
        result["spec"] = {"source": url, "version": (spec.get("info") or {}).get("version"), "get_endpoints": len(ops)}
        cal = pff.ncaa_calendar()
        season = max(cal["seasons"]) if cal["seasons"] else datetime.now(timezone.utc).year
        # Latest regular-season week with charted data, found from the receiving report.
        week = None
        for wk in sorted((w for w in cal["weeks"] if 0 <= w <= 16), reverse=True):
            counter["n"] += 1
            try:
                if not pff.get_receiving_summary(season, wk).empty:
                    week = wk
                    break
            except pff.PFFError:
                continue
        result["season"], result["week"] = season, week
        print(f"{len(ops)} endpoints in PFF's spec | checking {LEAGUE} {season} week {week}")
        team = find_team(ops, season, counter)
        result["team_lookup_worked"] = team is not None

        reports, skipped = [], []
        for op in sorted(ops, key=lambda o: o["path"]):
            if "/auth/" in op["path"]:
                continue
            calls, why = plan(op, season, week, team)
            if calls is None:
                skipped.append({"endpoint": op["name"], "path": op["path"], "why": why})
                continue
            for path, query, variant in calls:
                rec = {"endpoint": op["name"], "path": op["path"], "variant": variant, "tags": op["tags"], "summary": op["summary"],
                       "needs_team": any(re.sub(r"[^a-z_]", "", p["name"].lower()) in TEAM_NAMES and p["required"] for p in op["params"])}
                rec.update(call(path, query, counter))
                # Same report for 2022, for league-wide weekly reports only (history check).
                takes_week = any(re.sub(r"[^a-z_]", "", p["name"].lower()) in WEEK_NAMES for p in op["params"])
                if "error" not in rec and rec.get("rows") and takes_week and not rec["needs_team"]:
                    hcalls, _ = plan(op, HISTORY_SEASON, week, team)
                    h = next((c for c in hcalls or [] if c[2] == variant), None)
                    if h:
                        hist = call(h[0], h[1], counter)
                        rec["rows_2022"] = hist.get("rows") if "error" not in hist else hist["error"]
                reports.append(rec)
                label = f"{op['name']}{' [' + variant + ']' if variant else ''}"
                print(f"  {label:<48} " + (f"ERROR {rec['error']}" if "error" in rec else
                                           f"rows {rec.get('rows')}  columns {rec.get('n_columns')}  2022 rows {rec.get('rows_2022', '-')}"))
        result.update({"reports": reports, "skipped": skipped, "calls_made": counter["n"],
                       "working": sum(1 for r in reports if "error" not in r and r.get("rows")),
                       "empty": sum(1 for r in reports if "error" not in r and not r.get("rows")),
                       "failed": sum(1 for r in reports if "error" in r)})
        result["ok"] = result["working"] > 0
    except pff.PFFError as e:
        result["problem"] = str(e)
    except Exception as e:
        result["problem"] = f"{type(e).__name__}: {e}"
    return result


if __name__ == "__main__":
    res = main()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2, default=str)
    print(f"\nSaved {OUT}")
    if res.get("ok"):
        print(f"RESULT: {res['working']} report(s) returned data, {res['empty']} empty, {res['failed']} failed, "
              f"{len(res['skipped'])} skipped (need a specific player or game)")
    else:
        print(f"RESULT: check did not complete -- {res.get('problem')}")
    sys.exit(0 if res.get("ok") else 1)
