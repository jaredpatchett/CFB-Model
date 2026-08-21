"""
Team-level feature engineering for the spread/moneyline model.

Design choice on leakage:
- SP+ ratings (CFBD /ratings/sp) and the pace component of
  build_pace_returning_features (CFBD /stats/season/advanced) are season-
  level signals. CFBD's /ratings/sp has no week granularity at all (checked
  against their own API docs: get_sp takes only year/team, no week/
  start_week/end_week) -- it is always that season's FINAL rating. Using
  season S's own end-of-season SP+ to featurize season S's games -- Week 1
  included -- is leakage: the "prediction" already encodes how that season
  turned out. This isn't a theoretical concern -- a real backtest run on
  four real historical seasons came back at 71% ATS against the market's
  actual closing lines, an implausible number (a legitimate, durable edge
  of even 3-5 points over the 52.4% breakeven would be a big deal; 71% means
  something is leaking, not that this is a great model).

  Fix: build_game_team_features shifts SP+ and pace by one year before
  joining, so season S's games are matched to season (S-1)'s rating/pace --
  the actual last-known numbers a bettor (or this model, in live use) would
  have going into that season. This also happens to fix a pre-existing
  train/serve mismatch: export_dashboard_data.py's LIVE scoring already used
  last-completed-season SP+ (via the --year CLI arg) rather than the
  in-progress season's, since CFBD has nothing better to offer mid-season
  either -- training was the one place still using the leaked same-season
  number. The first training season on file has no prior year to match
  against and is correctly dropped (null features -> excluded by
  GameMarginModel.fit's dropna), not a bug.

  returning_production is NOT part of this leak -- it's CFBD's own
  percentPPA, computed from last season's departing production vs. this
  year's roster, so it's genuinely known before Week 1 of the season it's
  labeled with; shifting it doesn't change its meaning, just keeps its
  join consistent with pace/SP+.

  Known follow-up gap, not fixed here: once the in-season trained model
  activates for a real matchup (live_features.py, after both teams have
  MIN_GAMES_FOR_TRAINED_MODEL real games), its pace_diff is still built from
  the CURRENT season's own in-season-so-far advanced stats (correct, not
  leakage, live use only ever sees games already played) -- but that's a
  different definition of "pace" than what training now uses (prior
  season's final pace). Pace is a much smaller-magnitude feature than SP+
  here; left as a known inconsistency rather than risk a rushed second
  change to the live path.
- Rolling scoring margin/PPG features are computed with an expanding window
  shifted by one game, so a team's Week 5 features only use Weeks 1-4. This
  part is leakage-safe regardless of season stage and was NOT the source of
  the inflated backtest number.
- CORE ratings (CFBD's opponent-adjusted metrics) got the SAME prior-season-
  shift fix as SP+/pace above, after a real feature-importance diagnostic
  caught core_overall_diff at 80% importance (vs. SP+'s 4.3%) despite an
  earlier merge_asof-based within-season as-of-week join that was verified
  correct in isolation. See build_core_season_features/attach_core_ratings
  below for the full story -- short version: CFBD's per-week CORE snapshots
  likely encode some full-season information internally in a way that can't
  be verified or ruled out from their public docs, so this now uses only a
  fully-completed prior season's rating, which is leakage-proof regardless
  of CORE's internal methodology.
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


def build_core_season_features(core_df: pd.DataFrame) -> pd.DataFrame:
    """Tidy (team, year) -> core_overall/core_offense/core_defense, using
    each team's LATEST through_week snapshot in a season as that season's
    final rating. One row per team per year -- same shape as SP+/pace, and
    (as of this version) joined the SAME leakage-conservative way: see
    attach_core_ratings below.

    REVISED FROM AN EARLIER, MORE OPTIMISTIC VERSION OF THIS FUNCTION: CORE
    genuinely publishes a real per-week time series (throughWeek per row,
    confirmed against CFBD's own API docs), which looked like it should
    support a truly leakage-safe WITHIN-season as-of-week join -- each game
    matched only to a STRICTLY PRIOR week's snapshot via merge_asof,
    verified correct with a standalone test before shipping. It was wired
    in on that basis.

    A real backtest run then showed core_overall_diff commanding 80% of the
    trained model's total feature importance (vs. 4.3% for sp_rating_diff)
    -- see docs/data/model_diagnostics.json from that run. That's not a
    plausible "this feature just happens to be great" result; it's the same
    shape of red flag the SP+ leak produced (71% ATS) before that got
    fixed. The likely mechanism: CFBD's opponent-adjustment for CORE may
    use each opponent's FULL-SEASON strength to adjust a rating, even at an
    early through_week -- which the per-week join can't detect or prevent,
    since the leak would be baked into the snapshot itself, not in how it
    was joined. This isn't confirmed (CFBD doesn't publish enough
    methodology detail to verify it from here), and it doesn't need to be:
    switching to a fully-completed PRIOR season's final rating removes any
    possible leakage regardless of the reason, the same way it already did
    for SP+ and pace -- an entire prior season is fully resolved by
    definition, no matter how CORE computed anything during it.

    CFBD's raw JSON uses camelCase ('throughWeek'), not the snake_case the
    official Python client docs show (that's the generated client's own
    naming, not the wire format)."""
    cols = ["team", "year", "core_overall", "core_offense", "core_defense"]
    if core_df is None or core_df.empty:
        return pd.DataFrame(columns=cols)
    df = core_df.rename(columns={"year": "season"}) if "year" in core_df.columns else core_df.copy()
    if "throughWeek" in df.columns and "through_week" not in df.columns:
        df = df.rename(columns={"throughWeek": "through_week"})
    rename = {"overall": "core_overall", "offense": "core_offense", "defense": "core_defense"}
    have = [c for c in ("team", "season", "through_week", "overall", "offense", "defense") if c in df.columns]
    if not {"team", "season", "through_week"}.issubset(have):
        return pd.DataFrame(columns=cols)
    df = df[have].rename(columns=rename)
    latest = df.sort_values("through_week").groupby(["team", "season"]).tail(1)
    return latest.rename(columns={"season": "year"})[cols]


def attach_core_ratings(games: pd.DataFrame, core_df: pd.DataFrame) -> pd.DataFrame:
    """Adds home_core_*/away_core_*/core_*_diff columns to `games` by
    joining each game's season to the PRIOR season's final CORE rating --
    same treatment, same reasoning, and same join shape as SP+ (see this
    module's docstring and build_core_season_features above for why this
    replaced an earlier within-season as-of-week join). The first season on
    file correctly has no prior year to match (NaN, excluded downstream by
    the model's own null-handling), same as SP+."""
    out_cols = ["home_core_overall", "away_core_overall", "core_overall_diff",
                "home_core_offense", "away_core_offense", "core_offense_diff",
                "home_core_defense", "away_core_defense", "core_defense_diff"]
    core_season = build_core_season_features(core_df)
    out = games.copy()
    if core_season.empty:
        for c in out_cols:
            out[c] = np.nan
        return out

    shifted = core_season.copy()
    shifted["year"] = shifted["year"] + 1
    for side in ("home", "away"):
        out = out.merge(shifted.add_prefix(f"{side}_"), left_on=[f"{side}Team", "season"],
                         right_on=[f"{side}_team", f"{side}_year"], how="left")

    out["core_overall_diff"] = out["home_core_overall"] - out["away_core_overall"]
    out["core_offense_diff"] = out["home_core_offense"] - out["away_core_offense"]
    out["core_defense_diff"] = out["home_core_defense"] - out["away_core_defense"]
    return out


def attach_weather(games: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Adds temperature/wind_speed/precipitation/game_indoors to `games` by
    CFBD's own game id (games and weather are both CFBD's own data, so this
    is an exact join -- same reasoning as run_backtest.py's lines join, no
    team-name matching needed here). No leakage risk: a game's own weather
    (actual or forecast) was always knowable at or shortly before that
    specific kickoff, unlike a season-level aggregate.

    Indoor games get their weather fields zeroed to a neutral value rather
    than left null or fed a stray outdoor reading some providers attach to
    dome games -- wind/precip/cold have no real effect under a roof, and a
    GBM given a nonsense/null value for a dome game would otherwise have to
    learn that association on very few dome-game rows."""
    out_cols = ["temperature", "wind_speed", "precipitation", "game_indoors"]
    out = games.copy()
    if (weather_df is None or weather_df.empty or "id" not in games.columns
            or "id" not in weather_df.columns):
        for c in out_cols:
            out[c] = np.nan
        return out

    # Same raw-JSON-is-camelCase situation as CORE ratings (see
    # build_core_asof) -- windSpeed/gameIndoors, not wind_speed/game_indoors.
    w = weather_df.rename(columns={"windSpeed": "wind_speed", "gameIndoors": "game_indoors"})
    w = w[["id"] + [c for c in
        ("temperature", "wind_speed", "precipitation", "game_indoors") if c in w.columns]].copy()
    out = out.merge(w, on="id", how="left")
    for c in out_cols:
        if c not in out.columns:
            out[c] = np.nan
    indoor = out["game_indoors"] == True
    out.loc[indoor, "wind_speed"] = 0.0
    out.loc[indoor, "precipitation"] = 0.0
    out["game_indoors"] = out["game_indoors"].map({True: 1, False: 0}).astype("float")
    return out


def build_game_team_features(games: pd.DataFrame, sp_ratings: pd.DataFrame = None,
                              pace_returning: pd.DataFrame = None, core_ratings: pd.DataFrame = None,
                              weather: pd.DataFrame = None) -> pd.DataFrame:
    """
    games: output of cfbd_client.get_games() for one or more seasons.
    sp_ratings: output of cfbd_client.get_sp_ratings() (optional, adds SP+ prior).
    pace_returning: output of build_pace_returning_features() (optional,
      adds pace_diff/returning_production_diff — see that function's
      docstring for what these actually measure and their caveats).
    core_ratings: output of cfbd_client.get_core_ratings() (optional, adds
      core_overall_diff/core_offense_diff/core_defense_diff — joined to the
      PRIOR season's final rating, same treatment as SP+/pace; see
      attach_core_ratings and this module's docstring for why).
    weather: output of cfbd_client.get_weather() (optional, adds
      temperature/wind_speed/precipitation/game_indoors — see attach_weather).

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
        # Leakage fix -- see module docstring. Season S's games join to
        # season (S-1)'s SP+, not season S's own (only) rating.
        sp["year"] = sp["year"] + 1

        df = df.merge(sp.add_prefix("home_"), left_on=["homeTeam", "season"],
                       right_on=["home_team", "home_year"], how="left")
        df = df.merge(sp.add_prefix("away_"), left_on=["awayTeam", "season"],
                       right_on=["away_team", "away_year"], how="left")
        df["sp_rating_diff"] = df.get("home_sp_rating") - df.get("away_sp_rating")

    if pace_returning is not None and not pace_returning.empty:
        pr = pace_returning.copy()
        # Same leakage fix, same reasoning, applied to pace (see module
        # docstring). returning_production isn't actually leaky on its own
        # (preseason-known), but shares this join so it stays aligned with
        # pace under one consistent "prior season's numbers" join.
        pr["year"] = pr["year"] + 1
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

    df = attach_core_ratings(df, core_ratings)
    df = attach_weather(df, weather)

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
# core_overall_diff: opponent-adjusted CORE rating, joined to the PRIOR
# season's final rating (see attach_core_ratings and the module docstring
# for why) -- CFBD Tier 1+ ('Opponent Adjusted Metrics'). temperature/
# wind_speed/precipitation/game_indoors: per-game
# weather (see attach_weather) -- CFBD Tier 1+ ('Weather Data'). Both added
# once the project moved off the CFBD free tier; GameMarginModel.fit()
# drops any of these that turn out unavailable/all-null rather than crash,
# same as every other optional feature here.
FEATURE_COLUMNS = [
    "home_field", "roll_margin_diff", "roll_ppg_for_diff", "roll_ppg_against_diff",
    "sp_rating_diff", "home_games_played_prior", "away_games_played_prior",
    "pace_diff", "returning_production_diff",
    "core_overall_diff", "temperature", "wind_speed", "precipitation", "game_indoors",
]
TARGET_MARGIN = "margin"
TARGET_WIN = "home_win"
