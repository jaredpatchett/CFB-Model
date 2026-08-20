"""
Team-level feature engineering for the spread/moneyline model.

Design choice on leakage:
- SP+ ratings (from CFBD /ratings/sp) are a season-level signal. When used for
  LIVE weekly predictions this is safe (today's rating only reflects games
  already played). When used for BACKTESTING past seasons, pulling one
  end-of-season SP+ snapshot and applying it to Week 1 games is leakage —
  the rating "knows" about games that hadn't happened yet. The backtester
  (src/backtest/backtester.py) flags this; for rigorous backtesting, prefer
  the rolling in-season features below over SP+, or re-pull SP+ per week if
  your CFBD plan/endpoint supports it.
- Rolling scoring margin/PPG features are computed with an expanding window
  shifted by one game, so a team's Week 5 features only use Weeks 1-4. This
  part is leakage-safe regardless of season stage.
"""
import pandas as pd
import numpy as np


def _rolling_team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Build one row per (team, season, week) with that team's average
    points scored/allowed and margin from ONLY prior games in the season."""
    long_rows = []
    for _, g in games.iterrows():
        long_rows.append({
            "season": g["season"], "week": g["week"], "team": g["homeTeam"],
            "points_for": g["homePoints"], "points_against": g["awayPoints"],
        })
        long_rows.append({
            "season": g["season"], "week": g["week"], "team": g["awayTeam"],
            "points_for": g["awayPoints"], "points_against": g["homePoints"],
        })
    long_df = pd.DataFrame(long_rows).dropna(subset=["points_for", "points_against"])
    long_df = long_df.sort_values(["team", "season", "week"])

    long_df["margin"] = long_df["points_for"] - long_df["points_against"]
    grp = long_df.groupby(["team", "season"])
    # shift(1) before the expanding mean so the current game is excluded
    # .transform() (not .apply()) is required here: apply() with a lambda that
    # returns a same-length Series can come back with a MultiIndex on some
    # pandas versions and fail to align back into long_df. transform() is
    # the correct API for "same shape in, same shape out, per group."
    long_df["roll_ppg_for"] = grp["points_for"].transform(lambda s: s.shift(1).expanding().mean())
    long_df["roll_ppg_against"] = grp["points_against"].transform(lambda s: s.shift(1).expanding().mean())
    long_df["roll_margin"] = grp["margin"].transform(lambda s: s.shift(1).expanding().mean())
    long_df["games_played_prior"] = grp.cumcount()
    return long_df[["team", "season", "week", "roll_ppg_for", "roll_ppg_against",
                     "roll_margin", "games_played_prior"]]


def build_pace_returning_features(adv_stats: pd.DataFrame = None, returning: pd.DataFrame = None) -> pd.DataFrame:
    """One row per (team, year): 'pace' and 'returning_production', built
    from cfbd_client.get_advanced_team_stats(year) and
    cfbd_client.get_returning_production(year) respectively.

    pace = offense.plays / offense.drives (plays per drive) from CFBD's
    season-level advanced stats. This is a real, self-contained tempo-
    ADJACENT proxy — CFBD has no direct seconds-per-play/possession-time
    field at this granularity (verified against their own OpenAPI schema,
    not guessed), so this is NOT the same as the "adjusted pace" stat
    published on sites like billconnelly.football. It's labeled as
    "plays/drive" everywhere it's surfaced, not as unqualified "pace", to
    avoid overclaiming precision it doesn't have.

    returning_production = CFBD's own percentPPA (0-1 float): share of last
    season's total PPA production that's back on the roster this year.

    Same leakage caveat as SP+ (see this module's docstring): both are
    SEASON-level snapshots. Safe for live weekly use (today's numbers only
    reflect games/rosters already known); using a season's own snapshot to
    "predict" that same season's Week 1 IS leakage for backtesting — same
    simplification already accepted for SP+ here, not a new problem."""
    frames = []
    if (adv_stats is not None and not adv_stats.empty
            and "offense.plays" in adv_stats.columns and "offense.drives" in adv_stats.columns
            and "team" in adv_stats.columns and "season" in adv_stats.columns):
        pace_df = adv_stats[["team", "season", "offense.plays", "offense.drives"]].copy()
        pace_df = pace_df.rename(columns={"season": "year"})
        drives = pace_df["offense.drives"].replace(0, np.nan)
        pace_df["pace"] = pace_df["offense.plays"] / drives
        frames.append(pace_df[["team", "year", "pace"]])
    if (returning is not None and not returning.empty
            and "percentPPA" in returning.columns and "team" in returning.columns and "season" in returning.columns):
        ret_df = returning[["team", "season", "percentPPA"]].copy()
        ret_df = ret_df.rename(columns={"season": "year", "percentPPA": "returning_production"})
        frames.append(ret_df[["team", "year", "returning_production"]])

    if not frames:
        return pd.DataFrame(columns=["team", "year", "pace", "returning_production"])
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["team", "year"], how="outer")
    return out


def build_game_team_features(games: pd.DataFrame, sp_ratings: pd.DataFrame = None,
                              pace_returning: pd.DataFrame = None) -> pd.DataFrame:
    """
    games: output of cfbd_client.get_games() for one or more seasons.
    sp_ratings: output of cfbd_client.get_sp_ratings() (optional, adds SP+ prior).
    pace_returning: output of build_pace_returning_features() (optional,
      adds pace_diff/returning_production_diff — see that function's
      docstring for what these actually measure and their caveats).

    Returns one row per completed game with home/away features and targets
    (margin, home_win) ready for modeling.
    """
    games = games.copy()
    games = games[games["completed"] == True] if "completed" in games.columns else games
    games = games.dropna(subset=["homePoints", "awayPoints"])

    form = _rolling_team_form(games)

    df = games.merge(
        form.rename(columns={c: f"home_{c}" for c in form.columns if c not in ("team", "season", "week")}),
        left_on=["homeTeam", "season", "week"], right_on=["team", "season", "week"], how="left"
    ).drop(columns=["team"])
    df = df.merge(
        form.rename(columns={c: f"away_{c}" for c in form.columns if c not in ("team", "season", "week")}),
        left_on=["awayTeam", "season", "week"], right_on=["team", "season", "week"], how="left"
    ).drop(columns=["team"])

    if sp_ratings is not None and not sp_ratings.empty:
        sp = sp_ratings.copy()
        sp_cols = {"team": "team", "rating": "sp_rating"}
        if "offense.rating" in sp.columns:
            sp_cols["offense.rating"] = "sp_off_rating"
        if "defense.rating" in sp.columns:
            sp_cols["defense.rating"] = "sp_def_rating"
        sp = sp[list(sp_cols.keys()) + ["year"]].rename(columns=sp_cols)

        df = df.merge(sp.add_prefix("home_"), left_on=["homeTeam", "season"],
                       right_on=["home_team", "home_year"], how="left")
        df = df.merge(sp.add_prefix("away_"), left_on=["awayTeam", "season"],
                       right_on=["away_team", "away_year"], how="left")
        df["sp_rating_diff"] = df.get("home_sp_rating") - df.get("away_sp_rating")

    if pace_returning is not None and not pace_returning.empty:
        pr = pace_returning.copy()
        df = df.merge(pr.add_prefix("home_"), left_on=["homeTeam", "season"],
                       right_on=["home_team", "home_year"], how="left")
        df = df.merge(pr.add_prefix("away_"), left_on=["awayTeam", "season"],
                       right_on=["away_team", "away_year"], how="left")
        df["pace_diff"] = df.get("home_pace") - df.get("away_pace")
        df["returning_production_diff"] = df.get("home_returning_production") - df.get("away_returning_production")
    else:
        # Always create these columns (as all-NaN) even with no pace_returning
        # data, so team_game_features.csv has a stable, predictable schema
        # regardless of whether that data was available this run — anything
        # reading the CSV later can rely on the column existing. GameMarginModel.fit()
        # (src/models/game_model.py) separately handles an all-null or even a
        # fully-missing column by dropping it from that specific model's
        # feature set rather than crashing, so this is defense in depth, not
        # the only thing standing between a data gap and a crash.
        df["pace_diff"] = np.nan
        df["returning_production_diff"] = np.nan

    df["home_field"] = np.where(df.get("neutralSite", False) == True, 0, 1)
    df["margin"] = df["homePoints"] - df["awayPoints"]
    df["home_win"] = (df["margin"] > 0).astype(int)
    df["roll_margin_diff"] = df.get("home_roll_margin") - df.get("away_roll_margin")
    df["roll_ppg_for_diff"] = df.get("home_roll_ppg_for") - df.get("away_roll_ppg_for")
    df["roll_ppg_against_diff"] = df.get("home_roll_ppg_against") - df.get("away_roll_ppg_against")

    return df


# pace_diff/returning_production_diff are here (used by the trained
# GameMarginModel) precisely because build_features.py always builds
# pace_returning and passes it in — see that script. If CFBD coverage for
# /player/returning turns out sparse for some team/year, GameMarginModel.fit
# drops any row missing ANY feature column (see game_model.py), which could
# shrink the training set more than expected — its fit() now prints a
# per-column null-count breakdown specifically so that's visible in the
# training log rather than silently degrading the model.
FEATURE_COLUMNS = [
    "home_field", "roll_margin_diff", "roll_ppg_for_diff", "roll_ppg_against_diff",
    "sp_rating_diff", "home_games_played_prior", "away_games_played_prior",
    "pace_diff", "returning_production_diff",
]
TARGET_MARGIN = "margin"
TARGET_WIN = "home_win"
