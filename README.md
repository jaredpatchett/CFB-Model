# CFB Model

A college football model for spreads, moneylines, and player props — built on
free/API data (CollegeFootballData.com for stats, The Odds API for
moneyline/spread/totals lines, PrizePicks player props via OddsPapi).

This is a working baseline, not a finished sharp model. Read the "Honest
limitations" section before trusting any output with real money.

## What's here

```
config.py                      # reads API keys from .env or GitHub secrets
src/data/
  cfbd_client.py                # team/player stats, schedules, historical lines
  odds_api_client.py            # current moneyline/spread/totals (The Odds API)
  prizepicks_client.py          # current player props (OddsPapi -> PrizePicks)
src/features/
  team_features.py              # team-level features for the game model
  player_features.py             # player usage features for the props model
src/models/
  game_model.py                  # predicts margin -> spread/moneyline
  props_model.py                  # predicts stat value -> over/under lean
src/backtest/
  backtester.py                   # ATS win rate, log loss, hit rate vs. market
src/analysis/
  clv.py                           # CLV tracking for the model's own flagged spread picks
scripts/                           # CLI entry points, run in this order:
  1. fetch_historical_data.py     # pull past seasons for training/backtesting
  2. build_features.py            # raw data -> model-ready features
  3. train_game_model.py          # trains + saves the spread/ML model
  4. train_props_model.py         # trains + saves one model per prop stat
  5. fetch_current_lines.py       # pulls today's live lines + props
  6. predict_week.py              # scores current lines against the model
  7. run_backtest.py               # evaluates the model against history
  8. compute_clv.py                # grades flagged picks against real closing lines (CLV)
```

## Setup

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in your three keys (CFBD, Odds API,
   OddsPapi) for local runs. In GitHub Actions these are read from repo
   secrets instead — nothing to configure there.
3. Run the scripts in the order listed above.

Example, training on three seasons:

```bash
python scripts/fetch_historical_data.py --years 2022 2023 2024
python scripts/build_features.py --years 2022 2023 2024
python scripts/train_game_model.py
python scripts/train_props_model.py
python scripts/fetch_current_lines.py
python scripts/run_backtest.py
```

## Running from GitHub Actions

`.github/workflows/manual_run.yml` runs the full pipeline (fetch -> features
-> train -> fetch current lines) using your repo secrets. It's manual-trigger
only (Actions tab -> "Manual CFB model run" -> Run workflow) so it doesn't
burn API quota automatically — this repo has no scheduled/cron run set up.

## How the models work

**Spreads/moneylines**: gradient-boosted regression predicts point margin
(home minus away) from team form (leakage-safe rolling scoring margin),
SP+ rating differential, pace (plays/drive, from CFBD's advanced season
stats — see `src/features/team_features.py`), returning production
(CFBD's own percentPPA — share of last season's production back this year),
opponent-adjusted CORE rating differential (CFBD Tier 1+ — joined to the
PRIOR season's final rating, same treatment as SP+; see below), and
per-game weather (temperature, wind speed,
precipitation, indoor flag — CFBD Tier 1+). Margin is converted to a
moneyline win probability by assuming roughly normal residuals and using
the model's own training residual spread. The predicted margin is compared
to the market's spread to get an edge in points. Pace and returning
production also show on the Power Ratings table (hover a team, or look for
the "RP xx%" badge) whenever CFBD has coverage for that team.

**SP+ and CORE both use the same prior-season-shift treatment.** SP+
(`get_sp_ratings`) is one number per team per SEASON with no week
granularity at all — CFBD's own docs confirm there's no way to ask for
"SP+ as of week N." Using season S's own final SP+ to featurize season S's
Week 1 games is leakage (the rating already knows how the season turned
out), so training joins each season to the PRIOR season's SP+ instead —
see `src/features/team_features.py`'s module docstring for how a real
backtest at 71% ATS caught this. CORE ratings (`get_core_ratings`, CFBD's
"Opponent Adjusted Metrics") were originally treated differently, on the
theory that CFBD's real per-week time series (`throughWeek` per row) let
`attach_core_ratings` join each game to the most recent PRIOR week's rating
within that SAME season via `merge_asof`, verified correct in isolation —
and that theory turned out to be wrong. A real feature-importance
diagnostic (`docs/data/model_diagnostics.json`) caught `core_overall_diff`
at 80% importance vs. SP+'s 4.3%, the same shape of red flag the SP+ leak
produced. CFBD doesn't publish enough methodology detail to confirm why (a
plausible mechanism: CORE's opponent-adjustment may use full-season
opponent strength internally even at an early `through_week`), and it
doesn't need to be confirmed — CORE now gets the exact same fix as SP+:
`attach_core_ratings` joins each game to the PRIOR season's fully-completed
final CORE rating, provably leakage-free regardless of CORE's internal
methodology. Live scoring (`build_current_core_ratings` in
`src/features/live_features.py`, called from `export_dashboard_data.py`)
was switched to match — it's now fed the PRIOR completed season's CORE
data, not the in-progress season's, both to stay train/serve-consistent
and because the same unverifiable leak concern could contaminate an
in-progress season's snapshot too. Weather is per-game (CFBD's own game id
or, live, matched by team-name pair the same way this pipeline matches
every other cross-provider game), so there's no leakage question there at
all — a game's own weather was always knowable at or shortly before that
kickoff. All three (pace/returning, CORE, weather) are Tier 1+ CFBD
features; on a free-tier key they're skipped and the model trains/scores
on whatever the remaining features are, not faked.

**Player props**: one model per stat (receiving yards, rushing yards, passing
yards, receptions, etc.), predicting expected value from that player's own
leakage-safe rolling usage AND the upcoming opponent's real pass/rush
defense efficiency (CFBD's advanced season stats — see
`src/features/player_features.py`'s `attach_opponent_defense`). Compared
against the PrizePicks line for an over/under lean, live on the dashboard
whenever a real player/opponent match succeeds (see the switchover note
above).

**Backtesting**: historical CFBD closing lines (free, unlike The Odds API's
paid historical tier) are used to check whether the model would have beaten
the market, not just whether it predicted the final score accurately. Two
different things — a model can have great point-prediction accuracy and
still lose against the spread if the market is already pricing that
accurately. `run_backtest.py` joins the trained model's predictions to real
CFBD closing lines by CFBD's own game id (both `/games` and `/lines` are
CFBD's own data, so this join is exact — no fuzzy team-name matching needed,
unlike matching CFBD to a different provider like The Odds API) and prints
real ATS win rate, margin MAE, and moneyline log loss/accuracy against the
52.4% break-even threshold. This only scores games where the trained model
has complete rolling in-season features, so very early-season games are
excluded from the backtest the same way they'd be low-confidence live.

**CLV tracking**: a forward-looking companion to backtesting, for once the
model starts making REAL live picks instead of only historical ones — see
`src/analysis/clv.py`. Every workflow run, `export_dashboard_data.py`
snapshots the current market line for any game that currently qualifies as
a flagged spread play on the dashboard (same 1.9-point edge threshold as
the dashboard's own filter — see that module's `SPREAD_EDGE_THRESHOLD_POINTS`),
appending it to `data/clv/line_snapshots.csv`, which is committed to the
repo so it accumulates across the whole season, not just one run.
`compute_clv.py` then takes each game's EARLIEST snapshot (the real price a
follower would have gotten by acting the moment the pick first appeared)
and, for any of those games that have since been played, fetches the real
closing line from CFBD (the same free, already-trusted source `run_backtest.py`
uses) and computes CLV — how much better or worse the captured price was
than the market's final number. Positive CLV is the standard forward-looking
signal that a betting approach has real edge, and it's informative on a much
smaller sample than ATS win rate is, which matters early in a season before
enough real bets have graded to trust win/loss alone. Scope is spread picks
only for now (see that module's docstring for why); results land in
`docs/data/clv_results.json` and the dashboard's "CLV Track Record" panel.

## Honest limitations

- **Small sample, high variance.** CFB teams play ~12-13 games/year with
  heavy roster turnover between seasons. Early-season predictions for any
  team or player lean hard on limited current-season data plus a prior
  (SP+ or last-season stats) — treat Week 1-3 output as lower-confidence
  than Week 8+.
- **No automated injury/availability data** — no free, comprehensive,
  real-time CFB injury API exists (checked directly; CFBD/Odds
  API/OddsPapi don't have one). What exists instead: a hand-maintained,
  version-controlled override file, `config/injury_overrides.csv`
  (see `src/data/injury_overrides.py` for the format). Add a row for a
  team with a known key absence/return and a signed point adjustment, and
  it flows into the model's predicted margin, shows as a labeled
  decomposition line, and shows as a flag chip on the dashboard — nothing
  is silently baked in. Empty by default; someone has to actually notice
  the news and edit the file before kickoff for this to do anything.
- **Weather is now modeled** (temperature/wind/precipitation/indoor flag,
  CFBD Tier 1+) but only for the trained in-season `GameMarginModel` path —
  the preseason `fair_odds.py` estimator is still a pure SP+-diff-plus-
  home-field formula and doesn't take weather into account, since it has no
  in-season game-level features at all by design (see that file's
  docstring). A cold/windy Week 1 game still prices off SP+ alone until
  the trained model activates for that matchup.
- **SP+ and CORE both had leakage risk in backtesting, not live use — and
  both already caught and fixed, twice confirmed by real re-runs.** See the
  docstring in `src/features/team_features.py`: using a season-end SP+
  snapshot to "predict" Week 1 of that same season is leakage, and a real
  backtest run came back at an implausible 71% ATS before training was
  switched to the PRIOR season's SP+/pace. Re-ran after that fix: 60.2%
  ATS. CORE was believed safe at the time (see "How the models work" above
  for the full story) via a per-week as-of join, until a feature-importance
  diagnostic caught it at 80% importance after wiring it in — a real
  backtest run had jumped right back up to 69.6% ATS, the same shape of red
  flag as the SP+ leak. CORE was switched to the same prior-season-shift
  fix as SP+, and a real re-run confirms it worked the same way: importance
  dropped from 80% to 6.4% (now below sp_rating_diff's 22.3% and
  roll_margin_diff's 34.7%, a normal-looking spread across features instead
  of one feature dominating), and ATS win rate dropped from 69.6% to
  **61.25% over 2,013 real graded games (2023-2025)** — landing right next
  to the 60.2% SP+-only number, which is exactly what you'd expect once
  both leaks are gone and what's left is closer to the model's real signal.
  Still worth real skepticism (a sustained edge over 60% would be
  unusually large for a baseline model like this against closing lines),
  but this number now has two independent leak-removal events pointing at
  roughly the same place rather than one suspicious outlier.
- **Efficient markets.** Major/marquee matchups are heavily bet and
  sharp-adjusted; this kind of baseline model is more likely to find real
  edges in thinner markets — mid-tier games for spreads, and non-star-player
  props specifically — than in a ranked-vs-ranked Saturday night game.
- **The trained game model now reaches the live dashboard, per game, once
  it has enough to work with.** Every game starts priced by the preseason
  `src/models/fair_odds.py` SP+ estimator. `export_dashboard_data.py` also
  loads `models/game_model.joblib` (regenerated each run — still gitignored,
  read back within the same job rather than committed) and, for any specific
  matchup where BOTH teams have played `MIN_GAMES_FOR_TRAINED_MODEL` (3) or
  more real games this season, scores that game with the trained
  `GameMarginModel` instead (see `src/features/live_features.py`). This
  switches over automatically and independently per game — no manual step,
  and a real "In-season model" chip on the dashboard shows which games have
  switched. **This "current season" gating is computed from real wall-clock
  time** (`current_cfb_season_year()` in `export_dashboard_data.py`), not
  from the `--year` CLI flag — `--year` is set from the workflow's training
  `years` input and can legitimately be a fully-completed past season (e.g.
  used for team metadata/logos and the SP+ preseason prior), so using it to
  gate "has this team played games yet this season" would make every team
  falsely look fully in-season. Caught from a real run where the actual
  season opener showed the trained model active before a single 2026 game
  had been played, because `--year` was 2025.

  The **props model now reaches the dashboard too** (previously the standing
  gap here) — `export_dashboard_data.py` matches each posted
  PrizePicks line to a real player + opponent via
  `src/features/live_player_features.py`, and where that match succeeds
  (player has real in-season stats, a trained model exists for that stat,
  and the upcoming opponent's defensive numbers are available) shows a real
  model prediction/edge line under the posted price. **Read the matching
  caveat in that file's module docstring before trusting it**: PrizePicks
  player names have no shared ID with CFBD's athlete data, so matching is
  exact-normalized-string-only and drops any ambiguous name rather than
  guessing — spot-check the first live week's output against real
  PrizePicks lines once games start, since a silent name-format mismatch
  would just look like "no overlay" for real players, not an error.
- **`predict_week.py` still has an explicit wiring gap** (`run_backtest.py`'s
  is now fixed — see above): it needs this week's live team-feature rows
  built the same way `build_features.py` builds them for historical
  training, then joined to the current lines pulled by
  `fetch_current_lines.py`, by team name (The Odds API and CFBD use
  different naming conventions, so this needs the same kind of matching
  `export_dashboard_data.py` already does for the dashboard, not a
  shared ID). Left as a clearly-marked next step rather than silently
  guessed at.
- **CLV tracking is brand new and unvalidated against real data** (see "How
  the models work" above) — it's been tested against synthetic snapshots
  and a mocked closing-line fetch, confirming the join/sign-convention math
  is correct, but it hasn't graded a single REAL pick yet, since the season
  hasn't started. The `docs/data/clv_results.json` note field and the
  dashboard panel are both designed to say nothing (rather than show
  placeholder zeros) until at least one tracked game has actually been
  played — treat the first few weeks' numbers as "does this look
  sane" checks, not a verdict, the same "under N graded samples, don't
  read into it" caution as the ATS backtest above, just with a much
  smaller N needed before CLV starts being informative.

## Next steps worth prioritizing

1. Once real games are played, spot-check the props player-name matching
   (`src/features/live_player_features.py`) against actual PrizePicks
   output — it's untested against real name-format differences between
   PrizePicks and CFBD, by necessity (this dev environment can't reach
   either API directly). Add a hand-verified alias table if real mismatches
   turn up, the same way `TEAM_ALIASES` was added for team names.
2. Weather/CORE ratings are wired into the trained in-season model but not
   the preseason estimator (`fair_odds.py`) or the props model — worth
   revisiting once there's a full season of props data to see whether
   opponent-adjusted defense (already used for props via
   `attach_opponent_defense`) should also switch to CORE's opponent
   adjustment instead of raw success-rate stats.
3. Wire `predict_week.py` the same way `run_backtest.py` was just wired, so
   it scores this week's live lines against the trained model, not just
   historical ones.
4. Once real picks start grading in `docs/data/clv_results.json`, watch
   whether CLV and ATS backtest performance point the same direction —
   if they diverge (e.g. good ATS record but negative CLV, or vice versa),
   that itself is a useful signal worth digging into rather than trusting
   either number alone.

## Disclaimer

Built for research/decision-support, not as a guarantee of profit. No model
reliably beats a well-priced sportsbook line on every bet. Bet only what
you're comfortable losing, and treat any single week's results (good or bad)
as too small a sample to update your confidence on much.
