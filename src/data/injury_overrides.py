"""
Manual injury / starter-availability override file.

Why this exists: no free, comprehensive, real-time CFB injury API exists.
CFBD doesn't have one; neither The Odds API nor OddsPapi (PrizePicks) expose
injury data either (checked during the Aug 2026 audit). A starting QB being
out is easily a 10+ point swing that this model would otherwise never see
and would silently ignore.

Rather than fabricate injury data or pretend the gap doesn't exist, this
reads a hand-maintained, version-controlled CSV (config/injury_overrides.csv)
so a real person can flag a known absence/return before a slate locks. Any
override that's actually applied shows up ON the dashboard (a flag chip on
the affected game + a labeled line in the score decomposition + a note in
the model explanation), rather than being baked into a number with no
visible trace of why it changed.

File format (config/injury_overrides.csv):
    team,margin_points,note
    Ohio State,-7,Starting QB out (ankle) per team announcement 8/25
    Clemson,3,Starting CB back from suspension

- team: must match the school name this pipeline already uses elsewhere
  (CFBD's "school" field, e.g. "Ohio State", not "Ohio State Buckeyes" or
  a nickname) -- see get_team_override's matching, which is exact-match
  only, same reasoning as export_dashboard_data.py's match_team (silent
  substring matching has produced wrong-team false positives in this
  pipeline before).
- margin_points: a signed number of points added directly to THAT team's
  own expected scoring margin. Negative = weaker without the player,
  positive = stronger (e.g. a suspension ending, a star returning from
  injury). This is a manual judgment call, not derived from any data
  source -- keep it conservative and re-check it before kickoff.
- note: required. Shown on the dashboard so it's visibly clear the model's
  number for that game isn't purely math-derived.

Multiple rows for the same team stack (net summed, notes concatenated). A
team with no row is unaffected (0 points), which is the default/normal case
for every team, every week. An empty or missing file means "no overrides
active" -- this whole mechanism is opt-in and does nothing unless someone
edits the CSV.
"""
import os
import re
import unicodedata

import pandas as pd

DEFAULT_PATH = "config/injury_overrides.csv"


def _normalize(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).strip()


def load_injury_overrides(path: str = DEFAULT_PATH) -> dict:
    """normalized team name -> {'points': float, 'notes': [str], 'school': str}.

    Missing file, empty file, or a file with only a header row all return {}
    (no overrides active) rather than raising -- this is an opt-in mechanism
    that should never block a pipeline run."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    if df.empty or "team" not in df.columns or "margin_points" not in df.columns:
        return {}

    overrides = {}
    for _, row in df.iterrows():
        team = row.get("team")
        points = row.get("margin_points")
        note = row.get("note")
        if not team or pd.isna(team) or pd.isna(points):
            continue
        norm = _normalize(str(team))
        entry = overrides.setdefault(norm, {"points": 0.0, "notes": [], "school": str(team)})
        entry["points"] += float(points)
        if note is not None and not pd.isna(note) and str(note).strip():
            entry["notes"].append(str(note).strip())
    return overrides


def get_team_override(school_name: str, overrides: dict):
    """Returns {'points': float, 'notes': [str]} for this team, or None if it
    has no active override. EXACT match only (after normalization) -- same
    reasoning as export_dashboard_data.py's match_team: silently matching a
    typo'd or unrelated school name to the wrong team's override would apply
    a fabricated point swing to a game with no actual basis for it."""
    if not school_name or not overrides:
        return None
    return overrides.get(_normalize(school_name))
