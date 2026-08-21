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
import pandas as pd

from src.features.team_features import build_pace_returning_features

# Matches the reasoning in fair_odds.py's module docstring: below this many
# real games, a team's rolling form is too thin to trust over the SP+ prior.
# Both teams in a matchup need to clear this before the trained model takes
# over for that specific game — mixed-experience matchups (e.g. a team's
# Week 1 opponent who already has 3+ games from a Week 0 opener) still fall
# back to the preseason prior until BOTH sides qualify.
MIN_GAMES_FOR_TRAINED_MODEL = 3


def build_current_season_form(games_df: pd.DataFrame) -> dict:
    """CFBD school name -> {'roll_ppg_for', 'roll_ppg_against', 'roll_margin',
    'games_played_prior'}, computed from ONLY this team's own COMPLETED games
    in games_df (future/unplayed games are correctly excluded since they
    have no scores). A team that hasn't played yet this season simply has no
    entry — callers should treat that as "not enough in-season data," not
    guess at zero."""
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

    long_rows = []
    for _, g in df.iterrows():
        long_rows.append({"team": g["homeTeam"], "points_for": g["homePoints"], "points_against": g["awayPoints"]})
        long_rows.append({"team": g["awayTeam"], "points_for": g["awayPoints"], "points_against": g["homePoints"]})
    long_df = pd.DataFrame(long_rows)
    long_df["margin"] = long_df["points_for"] - long_df["points_against"]

    form = {}
    for team, grp in long_df.groupby("team"):
        form[team] = {
            "roll_ppg_for": float(grp["points_for"].mean()),
            "roll_ppg_against": float(grp["points_against"].mean()),
            "roll_margin": float(grp["margin"].mean()),
            "games_played_prior": int(len(grp)),
        }
    return form


def build_current_core_ratings(core_df: pd.DataFrame) -> dict:
    """CFBD school name -> {'core_overall'} for the CURRENT season, using
    each team's MOST RECENT through_week on file -- i.e. their real,
    opponent-adjusted rating as of right now. Unlike SP+/pace (which this
    pipeline deliberately shifts to the PRIOR season for training to avoid
    leakage -- see team_features.py's module docstring), CORE ratings are
    genuinely safe to use live at their current value: 'as of right now' is
    exactly what a live prediction is allowed to know. No entry for a team
    means CORE hasn't published a rating for them yet this season (normal
    before their first game), not an error."""
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
                              weather: dict = None) -> float:
    """Returns the trained GameMarginModel's predicted home margin for this
    matchup, or None if it shouldn't be trusted yet — either team missing
    from current_season_form (hasn't played this season), either team under
    MIN_GAMES_FOR_TRAINED_MODEL, no SP+ rating for either team, missing
    pace/returning-production/CORE/weather data the model needs (see caveat
    below), or no model loaded at all. None means "fall back to the
    preseason prior," not an error.

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
    if model is None or not home_school or not away_school:
        return None
    home_form = current_season_form.get(home_school)
    away_form = current_season_form.get(away_school)
    if not home_form or not away_form:
        return None
    if (home_form["games_played_prior"] < MIN_GAMES_FOR_TRAINED_MODEL
            or away_form["games_played_prior"] < MIN_GAMES_FOR_TRAINED_MODEL):
        return None
    if home_rating is None or away_rating is None:
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
        return None

    row = pd.DataFrame([{c: available[c] for c in model.feature_columns}])
    return float(model.predict_margin(row)[0])
