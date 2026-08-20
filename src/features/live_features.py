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


def score_with_trained_model(home_school: str, away_school: str, home_rating, away_rating,
                              neutral_site: bool, current_season_form: dict, model) -> float:
    """Returns the trained GameMarginModel's predicted home margin for this
    matchup, or None if it shouldn't be trusted yet — either team missing
    from current_season_form (hasn't played this season), either team under
    MIN_GAMES_FOR_TRAINED_MODEL, no SP+ rating for either team (still one of
    the model's own feature columns), or no model loaded at all. None means
    "fall back to the preseason prior," not an error."""
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

    row = pd.DataFrame([{
        "home_field": 0 if neutral_site else 1,
        "roll_margin_diff": home_form["roll_margin"] - away_form["roll_margin"],
        "roll_ppg_for_diff": home_form["roll_ppg_for"] - away_form["roll_ppg_for"],
        "roll_ppg_against_diff": home_form["roll_ppg_against"] - away_form["roll_ppg_against"],
        "sp_rating_diff": home_rating - away_rating,
        "home_games_played_prior": home_form["games_played_prior"],
        "away_games_played_prior": away_form["games_played_prior"],
    }])
    return float(model.predict_margin(row)[0])
