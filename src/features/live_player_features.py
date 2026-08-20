"""
Live, in-season PLAYER-level features — the player-props counterpart to
live_features.py's team-level live scoring. Lets export_dashboard_data.py
score a posted PrizePicks prop line against the trained PlayerStatModel
(src/models/props_model.py) once real in-season player data exists.

Same shape as the game-model switchover (live_features.py, task before
this one): before kickoff there's zero real current-season player data, so
build_current_player_form() returns {} and score_prop() returns None for
everything — the Props tab correctly shows just the posted line with no
model overlay, not a fabricated one. Once real weeks are played, this
activates automatically, no manual step.

MATCHING NOTE (read before trusting this in production): PrizePicks/
OddsPapi's player_name strings have no shared ID with CFBD's athleteId —
unlike team names, which this pipeline resolves via a hand-verified
TEAM_ALIASES table (see export_dashboard_data.py), there is no equivalent
verified alias table for player names here, because there's no way to test
real name strings from either provider against each other from this dev
sandbox. Matching is EXACT (after normalization) and REQUIRES the matched
name to be unique among this season's rostered players with real stat rows
— an ambiguous or unmatched name is skipped, never guessed at. Spot-check
the first real week of live output against actual PrizePicks lines before
trusting the model overlay; a systematic name-format mismatch (e.g.
suffixes, nicknames) would silently show as "just the line, no overlay"
for real matchups, not as a visible error — which is the safe failure mode,
but still worth verifying isn't happening for EVERY player.
"""
import re
import unicodedata

import pandas as pd

from src.features.player_features import STAT_MAP, pivot_player_game_stats

# Lower bar than the team model's 3-game threshold (live_features.py) —
# player usage stabilizes faster than a team's full-game outcome variance
# does, and props markets tend to open earlier in a player's own season.
# Still a real, chosen threshold, not zero.
MIN_GAMES_FOR_PROP_MODEL = 2

# PrizePicks/OddsPapi market-name keywords -> this pipeline's stat column
# names. Keyword-based (not exact string match) because the precise market
# name strings haven't been verified against a live fetch from this
# sandbox — kept as a small, explicit, auditable rule set rather than
# broad/fuzzy matching, and unrecognized market names are simply not
# scored, never guessed at. Spot-check against
# prizepicks_client.get_prop_market_catalog()'s real output on the first
# live run and tighten/correct these rules if any real market name doesn't
# map the way it's expected to.
def market_name_to_stat(market_name: str):
    m = (market_name or "").lower()
    if "pass" in m:
        if "yard" in m:
            return "pass_yds"
        if "td" in m or "touchdown" in m:
            return "pass_tds"
        if "completion" in m:
            return "pass_comp"
        if "attempt" in m:
            return "pass_att"
    if "interception" in m:
        return "pass_int"
    if "rush" in m:
        if "yard" in m:
            return "rush_yds"
        if "td" in m or "touchdown" in m:
            return "rush_tds"
        if "attempt" in m or "carr" in m:
            return "rush_att"
    if "rece" in m:  # matches both "Receiving" and "Receptions" (NOT "recei",
                      # which matches "Receiving" but misses "Receptions" --
                      # caught by a test before this shipped)
        if "yard" in m:
            return "rec_yds"
        if "td" in m or "touchdown" in m:
            return "rec_tds"
        return "receptions"  # bare "Receptions" market
    return None


def _normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).strip()


def build_current_player_form(player_stats_long_current: pd.DataFrame, games_current: pd.DataFrame) -> dict:
    """normalized player name -> {'team', 'games_played_prior', roll_<stat>
    for every STAT_MAP stat}, i.e. that player's real numbers as of right
    now — built the same way player_features.py builds training rows, just
    collapsed to one snapshot per player instead of one row per game.

    Ambiguous names (two+ distinct athleteIds normalizing to the same
    string) are DROPPED, not arbitrarily assigned to one of them — safer to
    show no model overlay for a name collision than to attach the wrong
    player's numbers to a real prop line.

    Returns {} (not an error) if there's no real current-season player data
    yet — the normal case before kickoff."""
    if player_stats_long_current is None or player_stats_long_current.empty or games_current is None or games_current.empty:
        return {}
    wide = pivot_player_game_stats(player_stats_long_current, games_current)
    if wide.empty or "athleteId" not in wide.columns:
        return {}

    by_name = {}
    ambiguous = set()
    for athlete_id, grp in wide.groupby("athleteId"):
        grp = grp.sort_values(["season", "week"])
        name_key = _normalize_name(grp["player"].iloc[-1])
        if name_key in by_name and by_name[name_key]["athlete_id"] != athlete_id:
            ambiguous.add(name_key)
            continue
        entry = {
            "athlete_id": athlete_id,
            "team": grp["team"].iloc[-1],
            "games_played_prior": int(len(grp)),
        }
        for col in STAT_MAP.values():
            entry[f"roll_{col}"] = float(grp[col].mean())
        by_name[name_key] = entry

    for name_key in ambiguous:
        by_name.pop(name_key, None)
    return by_name


def find_upcoming_opponent(team: str, schedule_df: pd.DataFrame):
    """The opponent in this team's next not-yet-completed game, per CFBD's
    own already-fetched schedule (no new API call). None if not found
    (team-name mismatch against the schedule, or no upcoming game in the
    fetched range) — callers should skip scoring, not guess an opponent."""
    if schedule_df is None or schedule_df.empty or not team:
        return None
    df = schedule_df
    mask = (df.get("homeTeam") == team) | (df.get("awayTeam") == team)
    upcoming = df[mask] if mask is not False else df.iloc[0:0]
    if "completed" in upcoming.columns:
        upcoming = upcoming[upcoming["completed"] != True]
    if upcoming.empty:
        return None
    upcoming = upcoming.sort_values("week")
    row = upcoming.iloc[0]
    return row["awayTeam"] if row["homeTeam"] == team else row["homeTeam"]


def score_prop(player_name: str, market_name: str, prop_line, player_form: dict,
               schedule_df: pd.DataFrame, opp_defense_lookup: dict, models: dict):
    """Returns {'model_predicted_value', 'model_edge', 'model_lean',
    'model_confidence'} for this prop, or None if it shouldn't be scored
    yet — unrecognized market, no trained model for that stat, player not
    matched (or matched ambiguously), not enough games this season, no
    upcoming-opponent found, or any required feature still missing. None
    always means 'show the posted line with no model overlay', never an
    error."""
    if prop_line is None or (isinstance(prop_line, float) and pd.isna(prop_line)):
        return None
    stat = market_name_to_stat(market_name)
    if not stat:
        return None
    model = models.get(stat)
    if model is None:
        return None

    entry = player_form.get(_normalize_name(player_name))
    if not entry or entry["games_played_prior"] < MIN_GAMES_FOR_PROP_MODEL:
        return None

    opponent = find_upcoming_opponent(entry["team"], schedule_df)
    if not opponent:
        return None
    opp_def = opp_defense_lookup.get(opponent) or {}

    row = {}
    for c in model.feature_columns:
        if c in ("opp_pass_def_success_rate", "opp_rush_def_success_rate"):
            val = opp_def.get(c)
        else:
            val = entry.get(c)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None  # a required feature is unavailable -- don't guess
        row[c] = val

    result = model.predict_and_compare(pd.Series(row), float(prop_line))
    return {
        "model_predicted_value": round(result["predicted_value"], 1),
        "model_edge": round(result["edge"], 1),
        "model_lean": result["lean"],
        "model_confidence": result["confidence"],
    }
