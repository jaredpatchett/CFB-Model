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

# Fantasy projections use a separate, lower bar than props (1 game instead
# of 2) -- per explicit user request, to get projections showing right
# away off Week 1 data instead of waiting for a second game. A single
# game's "rolling" average is just that game's real stat line (still real
# data, not fabricated), so this is a real tradeoff, not a hack: less
# sample to smooth out a fluke performance, in exchange for not waiting an
# extra week. Props keeps its own MIN_GAMES_FOR_PROP_MODEL untouched --
# this only affects project_player_fantasy()'s caller in
# export_dashboard_data.py.
MIN_GAMES_FOR_FANTASY = 1

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
            # Real display name (e.g. "Bo Nix"), distinct from the
            # lowercased/punctuation-stripped name_key this dict is keyed
            # by. score_prop() never needed this -- a scored prop's
            # displayed name always comes straight from the real posted
            # PrizePicks line, untouched -- but fantasy projections have
            # no posted line to source a name from, so they read this
            # field directly (see export_dashboard_data.py).
            "display_name": grp["player"].iloc[-1],
            "team": grp["team"].iloc[-1],
            "games_played_prior": int(len(grp)),
        }
        for col in STAT_MAP.values():
            entry[f"roll_{col}"] = float(grp[col].mean())
        by_name[name_key] = entry

    for name_key in ambiguous:
        by_name.pop(name_key, None)
    return by_name


# Standard PPR fantasy scoring weights, applied to each stat's MODEL-
# PREDICTED value (not a posted line -- there's no sportsbook fantasy
# line to compare against, this is a pure projection). Deliberately
# excludes pass_att/pass_comp/rush_att -- those are usage inputs the
# per-stat models consume as FEATURES, not events PPR scoring itself
# awards points for.
PPR_SCORING = {
    "pass_yds": 0.04,   # 1 pt / 25 yards
    "pass_tds": 4.0,
    "pass_int": -2.0,
    "rush_yds": 0.1,    # 1 pt / 10 yards
    "rush_tds": 6.0,
    "rec_yds": 0.1,     # 1 pt / 10 yards
    "rec_tds": 6.0,
    "receptions": 1.0,  # full PPR
}


def infer_position(entry: dict) -> str:
    """Soft, display-only position guess from real usage -- this pipeline
    has no actual position field (CFBD's player-game stats are keyed by
    stat category, not roster position), so this is a heuristic, not a
    verified label. QB if real pass-attempt volume is meaningful; RB if
    carries outweigh catches; otherwise WR/TE (can't distinguish the two
    from usage alone). Never affects the point projection itself, which
    sums whatever the player's own stat-category models actually predict
    for them regardless of this label."""
    pass_att = entry.get("roll_pass_att") or 0
    rush_att = entry.get("roll_rush_att") or 0
    receptions = entry.get("roll_receptions") or 0
    if pass_att >= 5:
        return "QB"
    if rush_att >= receptions:
        return "RB"
    return "WR/TE"


def project_player_fantasy(entry: dict, schedule_df: pd.DataFrame,
                            opp_defense_lookup: dict, models: dict):
    """Returns {'opponent', 'position', 'projected_points', 'stat_breakdown'}
    for this player's NEXT upcoming game, or None if it shouldn't be
    projected yet -- no upcoming opponent found, or a required feature
    (same feature set every stat model shares, see ROLLING_FEATURE_COLUMNS)
    is unavailable. Same 'never guess' failure mode as score_prop(): a
    missing input means no projection at all, not a fabricated one.

    Deliberately projects EVERY stat category this player has a trained
    model for (not just ones matching their inferred position) and sums
    them all into one PPR total -- a true pass-catching back or a
    rushing QB SHOULD get credit across categories, and a player with
    near-zero real volume in an irrelevant category (e.g. a WR's rushing
    attempts) will simply get a near-zero predicted value there, not a
    fabricated one, since the model is trained on that player's own real
    usage pattern."""
    team = entry.get("team")
    opponent = find_upcoming_opponent(team, schedule_df)
    if not opponent:
        return None
    opp_def = opp_defense_lookup.get(opponent) or {}

    stat_predictions = {}
    for stat, model in models.items():
        row = {}
        ok = True
        for c in model.feature_columns:
            if c in ("opp_pass_def_success_rate", "opp_rush_def_success_rate"):
                val = opp_def.get(c)
            else:
                val = entry.get(c)
            if val is None or (isinstance(val, float) and pd.isna(val)):
                ok = False
                break
            row[c] = val
        if not ok:
            continue
        pred = float(model.predict(pd.DataFrame([row]))[0])
        # A regressor can output a small negative number for a low-usage
        # player/stat (e.g. -1.3 predicted INTs) -- not a meaningful
        # real-world outcome, and NEGATIVE * PPR_SCORING's own negative
        # weight (pass_int) would perversely ADD points instead of
        # subtracting them if left unclipped.
        stat_predictions[stat] = max(pred, 0.0)

    if not stat_predictions:
        return None

    projected_points = sum(
        stat_predictions.get(stat, 0.0) * weight
        for stat, weight in PPR_SCORING.items()
    )
    return {
        "opponent": opponent,
        "position": infer_position(entry),
        "projected_points": round(projected_points, 1),
        "stat_breakdown": {k: round(v, 1) for k, v in stat_predictions.items()},
    }


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
