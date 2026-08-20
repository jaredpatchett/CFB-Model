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
scripts/                           # CLI entry points, run in this order:
  1. fetch_historical_data.py     # pull past seasons for training/backtesting
  2. build_features.py            # raw data -> model-ready features
  3. train_game_model.py          # trains + saves the spread/ML model
  4. train_props_model.py         # trains + saves one model per prop stat
  5. fetch_current_lines.py       # pulls today's live lines + props
  6. predict_week.py              # scores current lines against the model
  7. run_backtest.py               # evaluates the model against history
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
(home minus away) from team form (leakage-safe rolling scoring margin) and,
where available, SP+ rating differential. Margin is converted to a moneyline
win probability by assuming roughly normal residuals and using the model's
own training residual spread. The predicted margin is compared to the
market's spread to get an edge in points.

**Player props**: one model per stat (receiving yards, rushing yards, passing
yards, receptions, etc.), predicting expected value from that player's own
leakage-safe rolling usage. Compared against the PrizePicks line for an
over/under lean. No opponent-defense adjustment yet (see Next steps).

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
- **No weather.** Wind and rain meaningfully move total and passing-prop
  lines; not modeled here.
- **SP+ leakage risk in backtesting**, not live use. See the docstring in
  `src/features/team_features.py` — using a season-end SP+ snapshot to
  "predict" Week 1 of that same season is leakage. For live weekly
  predictions this isn't an issue (today's rating only reflects games
  already played).
- **Efficient markets.** Major/marquee matchups are heavily bet and
  sharp-adjusted; this kind of baseline model is more likely to find real
  edges in thinner markets — mid-tier games for spreads, and non-star-player
  props specifically — than in a ranked-vs-ranked Saturday night game.
- **The trained game/props models never actually reach the live dashboard.**
  `train_game_model.py` and `train_props_model.py` run and print a holdout
  MAE on every pipeline run, but `models/*.joblib` is gitignored, so those
  trained models are never committed or read back — the live dashboard runs
  entirely on the separate, simpler `src/models/fair_odds.py` SP+ linear
  estimator instead (built as a stand-in for the preseason window, before
  any in-season rolling features exist for the trained model to use). Worth
  a deliberate decision once in-season data exists: wire the trained model
  into the dashboard then, or stop training it every run.
- **`predict_week.py` still has an explicit wiring gap** (`run_backtest.py`'s
  is now fixed — see above): it needs this week's live team-feature rows
  built the same way `build_features.py` builds them for historical
  training, then joined to the current lines pulled by
  `fetch_current_lines.py`, by team name (The Odds API and CFBD use
  different naming conventions, so this needs the same kind of matching
  `export_dashboard_data.py` already does for the dashboard, not a
  shared ID). Left as a clearly-marked next step rather than silently
  guessed at.

## Next steps worth prioritizing

1. Wire `predict_week.py` the same way `run_backtest.py` was just wired, so
   it scores this week's live lines against the trained model, not just
   historical ones.
2. Add opponent-adjusted matchup features for props (e.g., a WR's yards
   prediction should account for the opposing pass defense's efficiency, not
   just the WR's own volume).
3. Track closing-line value (CLV) over time, not just win/loss — CLV is a
   better early signal of whether a model has real edge than a small sample
   of bet outcomes.

## Disclaimer

Built for research/decision-support, not as a guarantee of profit. No model
reliably beats a well-priced sportsbook line on every bet. Bet only what
you're comfortable losing, and treat any single week's results (good or bad)
as too small a sample to update your confidence on much.
