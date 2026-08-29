"""
CLV (closing-line value) tracking for the model's own spread picks.

What CLV is and why it matters here: the price you get when you place a
bet vs. the market's FINAL closing price right before kickoff. If you
consistently get a better number than the closing line (positive CLV),
that's the standard forward-looking signal that a betting approach has
real edge -- it doesn't depend on whether any single bet actually won,
which matters a lot for a model that's only a few weeks old and hasn't
graded enough real games yet for a win/loss record to mean much on its
own (see docs/data/backtest_results.json's small-sample caveats, which
apply here too, just less severely -- CLV needs a smaller sample to be
informative than ATS win rate does).

Scope, deliberately narrow for v1: SPREAD picks only, not moneyline or
props. Spread CLV is the standard, most commonly tracked form of CLV in
sports betting, and it's the cleanest to compute (a single point-based
number, no vig-removal or probability math needed) -- see
qualifying_spread_play's docstring for the exact formula, which is a
straight port of build_dashboard.py's client-side "does this game qualify
as a flagged play" logic, not a new threshold invented here.

Pipeline (see scripts/export_dashboard_data.py and scripts/compute_clv.py
for where these get called):
  1. Every workflow run, export_dashboard_data.py calls
     append_line_snapshots() right after building games_out. For every game
     that currently qualifies as a flagged spread play, it appends ONE row
     to config.CLV_SNAPSHOTS_PATH recording the market line at THIS moment
     -- the price a follower would actually get if they bet right when the
     dashboard showed the pick. This file is committed to the repo (unlike
     data/raw, data/processed, data/current -- see config.py's comment on
     CLV_SNAPSHOTS_PATH) so it accumulates across every run all season,
     not just the current one.
  2. scripts/compute_clv.py (run separately, later in the workflow) reads
     that accumulated log, takes each game's EARLIEST snapshot (the first
     time it qualified -- the real "you'd have bet it right here" price),
     fetches the real closing line from CFBD for any of those games that
     have since been played (same free /lines endpoint already trusted for
     backtesting -- see cfbd_client.get_historical_lines), and computes CLV
     for each. Games that haven't kicked off yet are simply not in CFBD's
     completed-game data and get skipped for now, not faked.

REAL BUG, FOUND AND FIXED ON THE SEASON OPENER ITSELF: append_line_snapshots
originally matched games_out entries to CFBD's schedule using
game['home_team']/game['away_team'] directly. Those fields are The Odds
API's own team-name style (e.g. "TCU Horned Frogs"), but CFBD's /games
schedule (what build_game_id_lookup is built from) only ever uses the bare
school name (e.g. "TCU") -- so the match NEVER succeeded, on any game,
ever. This wasn't caught by this module's own unit tests because those
used simple, single-word team names on both sides (e.g. "Ohio State"),
which happened to match either way and never exercised the real
mascot-suffix mismatch. It surfaced for real on 2026-08-29 (season opener):
4 real games qualified for a snapshot per qualifying_spread_play, and 0
were ever captured. Fixed by having export_dashboard_data.py additionally
store home_school/away_school (CFBD's own resolved name, already computed
there via match_team() for other purposes like neutral-site/weather
matching) on each game_out dict, and having append_line_snapshots key off
THOSE instead of home_team/away_team. Falls back to home_team/away_team if
home_school/away_school are missing (e.g. a stale caller), which preserves
the ORIGINAL bug rather than crashing -- intentional, so a caller that
hasn't been updated fails the same obvious way instead of a new one.
"""
import numpy as np
import pandas as pd
import os
import unicodedata
import re
from datetime import datetime, timezone


# Must match build_dashboard.py's MIN_EDGE_POINTS (1.9). Kept as a separate
# constant (not imported cross-module, since build_dashboard.py's copy is
# embedded inside a big JS string, not a plain Python value this module
# could import) -- if one changes, the other needs to change too, or the
# CLV log will start/stop tracking games that don't match what the
# dashboard actually shows as a flagged play. Called out explicitly here
# and in build_dashboard.py's MIN_EDGE_POINTS comment so this coupling
# isn't silently missed later.
SPREAD_EDGE_THRESHOLD_POINTS = 1.9


def _normalize_team_name(name: str) -> str:
    """Same normalization as export_dashboard_data.py's helper of the same
    name -- duplicated rather than imported to keep this module usable
    standalone (e.g. from compute_clv.py, which doesn't otherwise need
    export_dashboard_data.py's other Odds-API-specific matching logic)."""
    if not name:
        return ""
    ascii_name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", ascii_name.lower()).strip()


def build_game_id_lookup(schedule_df: pd.DataFrame) -> dict:
    """frozenset({normalized_home_school, normalized_away_school}) ->
    {'id': CFBD's own numeric game id, 'season': int, 'week': int}, from
    cfbd_client.get_games(). Same unordered-pair keying as
    export_dashboard_data.py's build_neutral_site_lookup (see that
    function's docstring) and for the same reason -- this pipeline's
    home/away labeling comes from The Odds API, which doesn't necessarily
    agree with CFBD's, so match on the matchup, not on which side is home.

    Bundles season/week alongside id because export_dashboard_data.py's
    live game_out dicts (built from The Odds API's current-lines response)
    don't carry CFBD's season/week fields at all -- this schedule lookup is
    the only place this pipeline has them available for a live/upcoming
    game, so it has to supply them here rather than assuming append_line_snapshots'
    caller already has them.

    This id is what lets compute_clv.py later join a captured snapshot
    back to CFBD's OWN closing-line data by an exact numeric id -- the same
    reliable join scripts/run_backtest.py already uses for historical
    backtesting (see join_features_to_lines there), not a new, unverified
    join method."""
    lookup = {}
    if schedule_df is None or schedule_df.empty:
        return lookup
    for _, row in schedule_df.iterrows():
        home, away, gid = row.get("homeTeam"), row.get("awayTeam"), row.get("id")
        if not home or not away or pd.isna(gid):
            continue
        key = frozenset({_normalize_team_name(home), _normalize_team_name(away)})
        lookup[key] = {
            "id": int(gid),
            "season": int(row["season"]) if pd.notna(row.get("season")) else None,
            "week": int(row["week"]) if pd.notna(row.get("week")) else None,
        }
    return lookup


def match_game_id(home_school: str, away_school: str, lookup: dict):
    """Returns {'id', 'season', 'week'} for this matchup if found in the
    current season's schedule, else None (a game the schedule lookup
    couldn't resolve just doesn't get a CLV snapshot -- same 'don't
    fabricate' handling as everywhere else in this pipeline)."""
    if not home_school or not away_school:
        return None
    key = frozenset({_normalize_team_name(home_school), _normalize_team_name(away_school)})
    return lookup.get(key)


def qualifying_spread_play(game_out: dict):
    """Returns {'side': 'home'|'away', 'edge_points': float} if this game
    currently qualifies as a flagged SPREAD play on the dashboard, else
    None. This is a direct port of build_dashboard.py's client-side
    priceGame()/qualifies logic for market === 'Spread':

      modelSpread = -model_predicted_margin   (spread-sign convention:
                                                 negative = home favored)
      edge        = spread_home - modelSpread
                  = spread_home + model_predicted_margin
      qualifies   = abs(edge) >= SPREAD_EDGE_THRESHOLD_POINTS
      side        = 'home' if edge > 0 else 'away'

    No normal-CDF/vig math needed here (unlike the dashboard's moneyline
    market or its 'confidence' display) -- the spread qualification
    threshold is pure points, which is exactly why CLV scope was kept to
    spread-only for v1 (see this module's docstring)."""
    if not game_out.get("has_model_line"):
        return None
    spread_home = game_out.get("spread_home")
    predicted_margin = game_out.get("model_predicted_margin")
    if spread_home is None or predicted_margin is None or pd.isna(spread_home) or pd.isna(predicted_margin):
        return None
    edge = float(spread_home) + float(predicted_margin)
    if abs(edge) < SPREAD_EDGE_THRESHOLD_POINTS:
        return None
    return {"side": "home" if edge > 0 else "away", "edge_points": edge}


SNAPSHOT_COLUMNS = [
    "game_id", "season", "week", "home_team", "away_team", "commence_time",
    "captured_at", "side", "spread_home_at_capture", "model_predicted_margin",
    "edge_points_at_capture",
]


def append_line_snapshots(games_out: list, schedule_df: pd.DataFrame, path: str) -> int:
    """For every game in games_out that currently qualifies as a flagged
    spread play (see qualifying_spread_play), appends ONE row to the CSV at
    `path` recording today's captured line -- creating the file with a
    header if it doesn't exist yet. Returns the number of rows appended.

    Deliberately APPENDS rather than upserts/dedupes: running this multiple
    times in a week for the same game is expected (the workflow gets
    re-run manually, not on a fixed schedule -- see manual_run.yml's own
    docstring for why) and each snapshot is real data about what the line
    looked like at that moment. compute_clv.py is responsible for picking
    the EARLIEST snapshot per game_id as 'the' price for CLV purposes; nothing
    is lost by keeping the rest around, and it's the only source this
    pipeline has for real intra-week line-movement history if that's ever
    worth surfacing later."""
    if not games_out:
        return 0
    game_id_lookup = build_game_id_lookup(schedule_df)
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for g in games_out:
        play = qualifying_spread_play(g)
        if not play:
            continue
        # IMPORTANT: match on home_school/away_school (CFBD's own resolved
        # school name, e.g. "TCU"), NOT home_team/away_team (The Odds API's
        # own style, e.g. "TCU Horned Frogs") -- build_game_id_lookup below
        # is built from CFBD's /games schedule, which only ever uses the
        # bare school name. Matching against home_team/away_team directly
        # silently matched 0 games, ever, on real data (caught on the
        # season opener itself: 4 real games qualified for a snapshot and 0
        # were captured). See export_dashboard_data.py's game_out dict for
        # where home_school/away_school come from. Falls back to
        # home_team/away_team only if home_school/away_school are missing
        # (e.g. an older games_out dict from before this fix), so this
        # degrades to the previous, still-broken behavior rather than
        # crashing -- better to keep investigating than to silently swallow
        # every game from a caller that hasn't been updated yet.
        home_key = g.get("home_school") or g.get("home_team")
        away_key = g.get("away_school") or g.get("away_team")
        match = match_game_id(home_key, away_key, game_id_lookup)
        if match is None:
            continue
        rows.append({
            "game_id": match["id"],
            "season": match["season"],
            "week": match["week"],
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "commence_time": g.get("commence_time"),
            "captured_at": now,
            "side": play["side"],
            "spread_home_at_capture": g.get("spread_home"),
            "model_predicted_margin": g.get("model_predicted_margin"),
            "edge_points_at_capture": play["edge_points"],
        })
    if not rows:
        return 0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    new_df = pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)
    if os.path.exists(path):
        new_df.to_csv(path, mode="a", header=False, index=False)
    else:
        new_df.to_csv(path, mode="w", header=True, index=False)
    return len(rows)


def compute_clv_for_snapshots(first_snapshots: pd.DataFrame, closing_lines: pd.DataFrame) -> pd.DataFrame:
    """first_snapshots: one row per game_id (the EARLIEST captured snapshot
    -- caller's responsibility to have already reduced to this, see
    scripts/compute_clv.py), with 'game_id' and 'side'/'spread_home_at_capture'
    columns from SNAPSHOT_COLUMNS above.
    closing_lines: output of cfbd_client.historical_lines_to_dataframe(),
    with an 'id' column matching CFBD's own game id and a
    'market_spread_home' column -- the same closing-line source and column
    names scripts/run_backtest.py already trusts for backtesting (see that
    script's join_features_to_lines).

    Inner-joins on game_id == id, so a game with no closing line on file
    yet (hasn't been played, or CFBD hasn't archived it) is correctly
    dropped rather than given a fabricated CLV.

    CLV sign convention: positive means the captured price was BETTER than
    the closing price for whichever side was recommended.
      - side == 'home': CLV = spread_home_at_capture - closing_spread_home
        (a smaller/more-negative home spread at capture than at close means
        you got fewer points to give up than someone betting at closing --
        good for a home bet)
      - side == 'away': CLV = closing_spread_home - spread_home_at_capture
        (the mirror image, since away's effective spread is -spread_home)
    """
    if first_snapshots.empty or closing_lines.empty:
        return pd.DataFrame()
    closing = closing_lines.rename(columns={"id": "game_id"})
    merged = first_snapshots.merge(
        closing[["game_id", "homeScore", "awayScore", "market_spread_home", "startDate"]],
        on="game_id", how="inner",
    )
    if merged.empty:
        return merged
    merged["closing_spread_home"] = merged["market_spread_home"]
    home_side = merged["side"] == "home"
    merged["clv_points"] = np.where(
        home_side,
        merged["spread_home_at_capture"] - merged["closing_spread_home"],
        merged["closing_spread_home"] - merged["spread_home_at_capture"],
    )
    return merged
