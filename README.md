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
accurately.

## Honest limitations

- **Small sample, high variance.** CFB teams play ~12-13 games/year with
  heavy roster turnover between seasons. Early-season predictions for any
  team or player lean hard on limited current-season data plus a prior
  (SP+ or last-season stats) — treat Week 1-3 output as lower-confidence
  than Week 8+.
- **No injury/availability data.** A starting QB being out isn't reflected
  anywhere in this pipeline yet. This is probably the single biggest gap.
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
- **`predict_week.py` and `run_backtest.py` have an explicit wiring gap**:
  matching current/historical games to market lines by game ID, and building
  this week's live feature rows, are left as clearly-marked next steps rather
  than silently guessed at. Check the inline comments before assuming full
  automation.

## Next steps worth prioritizing

1. Wire up the game-ID matching between predictions and market lines (both
   current and historical) so `run_backtest.py` reports real ATS/CLV numbers.
2. Add opponent-adjusted matchup features for props (e.g., a WR's yards
   prediction should account for the opposing pass defense's efficiency, not
   just the WR's own volume).
3. Add injury/starter-status as a feature or manual override.
4. Track closing-line value (CLV) over time, not just win/loss — CLV is a
   better early signal of whether a model has real edge than a small sample
   of bet outcomes.

## Disclaimer

Built for research/decision-support, not as a guarantee of profit. No model
reliably beats a well-priced sportsbook line on every bet. Bet only what
you're comfortable losing, and treat any single week's results (good or bad)
as too small a sample to update your confidence on much.
