"""
Live, in-season team-form features — the counterpart to
src/features/team_features.py's build_game_team_features(), which is built
for TRAINING (one row per completed historical game, using each team's form
as of just before that specific game).

For live use we don't want a row per past game — we want ONE snapshot per
team, "their form right now," to score an upcoming game that hasn't been
played yet. Same underlying math (mean of a team's own completed games this
season), just without the per-past-week shifting that training needs to
avoid leaking a game's own result into its own features — a future game
that hasn't happened yet can't leak into anything by construction.

This is also what lets export_dashboard_data.py auto-switch a game from the
preseason SP+ prior (src/models/fair_odds.py) to the trained GameMarginModel
once both teams have enough real in-season games under their belt — see
MIN_GAMES_FOR_TRAINED_MODEL below and fair_odds.py's module docstring for
why that switch matters (feeding the trained model rolling features that
are still blanked to 0 this early would be extrapolating outside anything
it learned from).
"""
import re
import unicodedata

import pandas as pd

from src.features.team_features import build_pace_returning_features


def _normalize_school(name: str) -> str:
    """Same normalization export_dashboard_data.py's _normalize_team_name
    uses (strip accents, lowercase, drop punctuation) -- duplicated here
    rather than imported to avoid a circular import (export_dashboard_data.py
    already imports FROM this module). Used only to match a game's
    home/away team name against an sp_lookup dict built the same way, for
    the opponent-strength adjustment in build_current_season_form below."""
    if not name:
        return ""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).strip()

# Matches the reasoning in fair_odds.py's module docstring: below this many
# real games, a team's rolling form is too thin to trust over the SP+ prior.
# Both teams in a matchup need to clear this before the trained model takes
# over for that specific game — mixed-experience matchups (e.g. a team's
# Week 1 opponent who already has 3+ games from a Week 0 opener) still fall
# back to the preseason prior until BOTH sides qualify.
#
# Lowered 3 -> 2 on 9/19/2026 (confirmed with the user): the rolling
# features use an expanding window (shift(1).expanding().mean() in
# team_features._rolling_team_form), valid starting at 1 prior game, and
# home/away_games_played_prior are themselves trained-model FEATURE_COLUMNS
# -- so the model has genuinely seen games_played_prior=1/2 situations in
# training (only games_played_prior=0 is truly out-of-distribution, since
# shift(1) on a team's first game of a season is NaN and gets dropped by
# GameMarginModel.fit()'s dropna). run_backtest.py's own inclusion
# criterion is "features non-null," not "3+ games either" -- so the
# validated 61.25% ATS backtest already included 1-2-game situations, this
# just makes live scoring match what was actually backtested instead of
# being more conservative than the validation itself.
MIN_GAMES_FOR_TRAINED_MODEL = 2


def build_current_season_form(games_df: pd.DataFrame, sp_lookup: dict = None) -> dict:
    """CFBD school name -> {'roll_ppg_for', 'roll_ppg_against', 'roll_margin',
    'games_played_prior'}, computed from ONLY this team's own COMPLETED games
    in games_df (future/unplayed games are correctly excluded since they
    have no scores). A team that hasn't played yet this season simply has no
    entry — callers should treat that as "not enough in-season data," not
    guess at zero.

    sp_lookup: optional {normalized school name -> SP+ rating} (see
    export_dashboard_data.py's build_sp_lookup). When given, each game's
    contribution to roll_margin is OPPONENT-ADJUSTED -- the opponent's own
    SP+ rating (already scaled as points-above-average) gets added to the
    raw margin before averaging, so a 50-point win over a heavily
    overmatched opponent doesn't inflate a team's "current form" by
    anywhere near as much as it does unadjusted, and a close game against
    a strong opponent gets real credit instead of looking mediocre.
    Without this, roll_margin was just a flat average with zero
    opponent-quality awareness -- confirmed as a real driver of live
    misses 9/2026 (Texas A&M: 50-0 over an overmatched Missouri State +
    48-20 over Arizona State produced a flat +39ppg "current form" that
    had nothing to do with how A&M would actually fare against a real
    opponent; Kentucky then won outright as a 16.5-point underdog).
    roll_ppg_for/roll_ppg_against stay raw/unadjusted on purpose -- they're
    directly-observed scoring stats, not a proxy for team strength, so
    there's nothing to adjust there. Any single game whose opponent isn't
    in sp_lookup (unmatched or a non-FBS opponent with no rating) falls
    back to that one game's raw, unadjusted margin rather than dropping
    the game or guessing a rating -- same "never fabricate" policy as
    everywhere else in this file."""
    if games_df is None or games_df.empty:
        return {}
    df = games_df.copy()
    if "completed" in df.columns:
        df = df[df["completed"] == True]
    if "homePoints" not in df.columns or "awayPoints" not in df.columns:
        return {}
    df = df.dropna(subset=["homePoints", "awayPoints"])
    if df.empty:
        return {}

    sp_lookup = sp_lookup or {}

    long_rows = []
    for _, g in df.iterrows():
        home, away = g["homeTeam"], g["awayTeam"]
        home_margin = g["homePoints"] - g["awayPoints"]
        away_rating = sp_lookup.get(_normalize_school(away))
        home_rating = sp_lookup.get(_normalize_school(home))
        long_rows.append({
            "team": home, "points_for": g["homePoints"], "points_against": g["awayPoints"],
            "adj_margin": (home_margin + away_rating) if away_rating is not None else home_margin,
        })
        long_rows.append({
            "team": away, "points_for": g["awayPoints"], "points_against": g["homePoints"],
            "adj_margin": (-home_margin + home_rating) if home_rating is not None else -home_margin,
        })
    long_df = pd.DataFrame(long_rows)

    form = {}
    for team, grp in long_df.groupby("team"):
        form[team] = {
            "roll_ppg_for": float(grp["points_for"].mean()),
            "roll_ppg_against": float(grp["points_against"].mean()),
            "roll_margin": float(grp["adj_margin"].mean()),
            "games_played_prior": int(len(grp)),
        }
    return form


def build_current_core_ratings(core_df: pd.DataFrame) -> dict:
    """CFBD school name -> {'core_overall'}, using each team's MOST RECENT
    through_week on file in `core_df`. CALLER is responsible for passing in
    the PRIOR completed season's core_df here, not the current in-progress
    season's -- same treatment as SP+/pace (see export_dashboard_data.py,
    which builds this from `year`, not `season_year`) and for the same
    reason: team_features.py's training-time fix (see that module's
    docstring) switched CORE to the prior-season snapshot after a real
    feature-importance diagnostic showed core_overall_diff at 80%
    importance, an implausible dominance suggesting CFBD's same-season
    CORE numbers may encode more than they should. This function doesn't
    know or care which season's data it's handed -- it just takes the
    latest through_week in whatever df it's given -- so keeping training
    and live scoring consistent is entirely on the caller passing the same
    kind of season. An earlier version of this docstring called live use of
    the CURRENT season 'genuinely safe' on the theory that CORE's per-week
    values were strictly causal; that theory is what the training-time diagnostic
    disproved, so it no longer applies here either. No entry for a team
    means CORE has no rating on file for that prior season (e.g. a team
    that wasn't in a CFBD-tracked conference yet), not an error."""
    if core_df is None or core_df.empty or "team" not in core_df.columns:
        return {}
    df = core_df.copy()
    if "throughWeek" in df.columns and "through_week" not in df.columns:
        df = df.rename(columns={"throughWeek": "through_week"})
    if "overall" not in df.columns or "through_week" not in df.columns:
        return {}
    latest = df.sort_values("through_week").groupby("team").tail(1)
    out = {}
    for _, row in latest.iterrows():
        out[row["team"]] = {"core_overall": row.get("overall")}
    return out


def build_current_pace_returning(adv_stats_df: pd.DataFrame, returning_df: pd.DataFrame) -> dict:
    """CFBD school name -> {'pace', 'returning_production'} for the CURRENT
    season — reuses team_features.build_pace_returning_features (same
    math/caveats documented there), just reshaped into a flat team -> dict
    lookup since live scoring only ever looks at one season at a time (that
    function's output is keyed by (team, year) since it's shared with the
    multi-season historical/training path)."""
    df = build_pace_returning_features(adv_stats_df, returning_df)
    out = {}
    for _, row in df.iterrows():
        out[row["team"]] = {"pace": row.get("pace"), "returning_production": row.get("returning_production")}
    return out


def score_with_trained_model(home_school: str, away_school: str, home_rating, away_rating,
                              neutral_site: bool, current_season_form: dict, model,
                              pace_returning: dict = None, core_ratings: dict = None,
                              weather: dict = None, diagnostics: dict = None) -> float:
    """Returns the trained GameMarginModel's predicted home margin for this
    matchup, or None if it shouldn't be trusted yet — either team missing
    from current_season_form (hasn't played this season), either team under
    MIN_GAMES_FOR_TRAINED_MODEL, no SP+ rating for either team, missing
    pace/returning-production/CORE/weather data the model needs (see caveat
    below), or no model loaded at all. None means "fall back to the
    preseason prior," not an error.

    diagnostics: optional dict the CALLER owns and passes in blank ({}) --
    when given, every early-return path below increments a counter in it
    (e.g. diagnostics['missing_feature:temperature'] += 1) instead of just
    silently returning None, so a caller running this across a whole slate
    can print which specific reason is actually blocking games at the end,
    rather than only knowing the aggregate "N games didn't switch over."
    Added 9/19/2026 after the live dashboard showed 0 of 90 games on the
    trained model despite 198 teams clearing the games-played threshold --
    with 4 independent feed-coverage requirements (pace, returning
    production, CORE, weather) all gating the SAME switchover with zero
    partial credit, "which one" needed an actual answer, not a guess.
    Purely additive: default None means zero behavior change for any
    existing caller.

    core_ratings: output of build_current_core_ratings() -- team -> {'core_overall'}.
    weather: this SPECIFIC upcoming game's own weather dict (temperature/
    wind_speed/precipitation/game_indoors), looked up by the CALLER (weather
    is per-game, not per-team, so it can't be threaded through the same
    team-keyed dict shape as pace_returning/core_ratings). None if no
    forecast is available yet (CFBD's forecast window doesn't reach a game
    this far out) -- same "don't fabricate" handling as everything else.

    IMPORTANT COUPLING: since pace_diff/returning_production_diff/
    core_overall_diff/temperature/wind_speed/precipitation/game_indoors are
    now part of GameMarginModel.feature_columns (see team_features.py), the
    trained model literally cannot score a game without ALL of them
    present — there's no "impute a default and hope" option here without
    risking a biased, unvalidated prediction. So if CFBD's /player/returning
    coverage turns out sparse for some team, or no weather forecast has
    posted yet for a game, this now ALSO blocks the in-season switchover
    for that game, not just the corresponding display badge — a real
    trade-off, not an oversight. Building the feature row generically off
    whatever model.feature_columns actually asks for (rather than
    hardcoding a fixed dict) so this stays correct if FEATURE_COLUMNS
    changes again later."""
    def _bump(key):
        if diagnostics is not None:
            diagnostics[key] = diagnostics.get(key, 0) + 1

    def _detail(entry):
        # Sample of the actual matchup + team-level games-played counts
        # behind the two most common block reasons, added right after the
        # first diagnostics pass (9/19/2026) showed 69 under_min_games_
        # threshold + 21 team_not_in_current_season_form but gave no way to
        # tell "that's genuinely how this week's slate looks" (bye weeks,
        # FCS/small-conference opponents on a different schedule) apart
        # from "there's a real name-mismatch bug" without eyeballing the
        # actual teams involved.
        if diagnostics is not None:
            diagnostics.setdefault('_team_detail', []).append(entry)

    if model is None or not home_school or not away_school:
        _bump('no_model_or_missing_team_name')
        return None
    home_form = current_season_form.get(home_school)
    away_form = current_season_form.get(away_school)
    if not home_form or not away_form:
        _bump('team_not_in_current_season_form')
        _detail(f"{home_school} (games={home_form['games_played_prior'] if home_form else 'NOT FOUND'}) "
                f"vs {away_school} (games={away_form['games_played_prior'] if away_form else 'NOT FOUND'})")
        return None
    if (home_form["games_played_prior"] < MIN_GAMES_FOR_TRAINED_MODEL
            or away_form["games_played_prior"] < MIN_GAMES_FOR_TRAINED_MODEL):
        _bump('under_min_games_threshold')
        _detail(f"{home_school} (games={home_form['games_played_prior']}) "
                f"vs {away_school} (games={away_form['games_played_prior']})")
        return None
    if home_rating is None or away_rating is None:
        _bump('missing_sp_rating')
        return None

    pace_returning = pace_returning or {}
    home_pr = pace_returning.get(home_school) or {}
    away_pr = pace_returning.get(away_school) or {}
    home_pace, away_pace = home_pr.get("pace"), away_pr.get("pace")
    home_rp, away_rp = home_pr.get("returning_production"), away_pr.get("returning_production")

    core_ratings = core_ratings or {}
    home_core = (core_ratings.get(home_school) or {}).get("core_overall")
    away_core = (core_ratings.get(away_school) or {}).get("core_overall")

    weather = weather or {}

    available = {
        "home_field": 0 if neutral_site else 1,
        "roll_margin_diff": home_form["roll_margin"] - away_form["roll_margin"],
        "roll_ppg_for_diff": home_form["roll_ppg_for"] - away_form["roll_ppg_for"],
        "roll_ppg_against_diff": home_form["roll_ppg_against"] - away_form["roll_ppg_against"],
        "sp_rating_diff": home_rating - away_rating,
        "home_games_played_prior": home_form["games_played_prior"],
        "away_games_played_prior": away_form["games_played_prior"],
        "pace_diff": (home_pace - away_pace) if (pd.notna(home_pace) and pd.notna(away_pace)) else None,
        "returning_production_diff": (home_rp - away_rp) if (pd.notna(home_rp) and pd.notna(away_rp)) else None,
        "core_overall_diff": (home_core - away_core) if (pd.notna(home_core) and pd.notna(away_core)) else None,
        "temperature": weather.get("temperature"),
        "wind_speed": weather.get("wind_speed"),
        "precipitation": weather.get("precipitation"),
        "game_indoors": weather.get("game_indoors"),
    }
    missing = [c for c in model.feature_columns if c not in available or available[c] is None]
    if missing:
        for m in missing:
            _bump('missing_feature:' + m)
        return None

    row = pd.DataFrame([{c: available[c] for c in model.feature_columns}])
    return float(model.predict_margin(row)[0])
