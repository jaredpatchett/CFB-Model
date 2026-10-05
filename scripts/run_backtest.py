#!/usr/bin/env python3
"""
Backtest the trained game model against REAL historical CFBD closing lines
and actual results — this is the real bar (beating the market), not just
whether the model's predicted margin is close to the actual final score.

Usage:
  python scripts/run_backtest.py
"""
import json
import os
import sys
from datetime import datetime, timezone
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from src.models.game_model import GameMarginModel
from src.backtest import backtester


def run_splits(scoreable: pd.DataFrame) -> dict:
    """Where (if anywhere) the model beats the closing spread, out of sample
    (added 10/2026): by game type (Power 4 / Group of 5), spread size, model
    edge size, and a historical re-creation of the Bet Card -- each week's top
    10 plays by model cover probability, with spreads of 20+ faded."""
    from scipy.stats import norm
    from src.analysis import game_profile as gp

    df = scoreable.copy()
    df["edge"] = df["predicted_margin"] + df["market_spread_home"]   # model margin minus market's expected margin
    df = df[df["edge"] != 0]
    pick_home = df["edge"] > 0
    cover = df["margin"] + df["market_spread_home"]
    df["result"] = ["P" if c == 0 else ("W" if (c > 0) == ph else "L") for c, ph in zip(cover, pick_home)]
    df["units"] = df["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
    df["cover_prob"] = norm.cdf(df["edge"].abs() / df["fold_sd"])
    confs = gp.conference_map(df["season"].unique())
    df["game_type"] = [gp.game_type(s, h, a, confs) for s, h, a in zip(df["season"], df["homeTeam"], df["awayTeam"])]
    big = df["market_spread_home"].abs() >= 20
    play = df[~big].copy()
    play["rank"] = play.groupby(["season", "week"])["cover_prob"].rank(ascending=False, method="first")
    play["bet_card"] = play["rank"] <= 10

    out = {"note": "every graded game, standard -110 pricing; 'bet card' = each week's top 10 by model cover "
                   "probability with spreads of 20+ faded"}
    out["overall"] = [gp.summarize(df, "every game"), gp.summarize(play, "spreads under 20"),
                      gp.summarize(play[play.bet_card], "bet card (top 10/week)"), gp.summarize(play[~play.bet_card], "not on bet card")]
    out["game_type"] = [gp.summarize(g, k) for k, g in play.groupby("game_type")]
    out["game_type_bet_card"] = [gp.summarize(g, f"{k} | bet card") for k, g in play[play.bet_card].groupby("game_type")]
    out["spread_size"] = [gp.summarize(g, str(k)) for k, g in df.groupby(pd.cut(df["market_spread_home"].abs(), [-0.1, 3, 7, 14, 20, 99],
                          labels=["spread 0-3", "spread 3.5-7", "spread 7.5-14", "spread 14.5-20", "spread 20+"]), observed=True)]
    out["edge_size"] = [gp.summarize(g, str(k)) for k, g in play.groupby(pd.cut(play["edge"].abs(), [0, 3, 7, 14, 99],
                        labels=["model off market 0-3 pts", "3-7 pts", "7-14 pts", "14+ pts"]), observed=True)]
    # Season phase (added 10/2026): the live Bet Card has only run on the
    # trained model from mid-September on, while the full backtest also
    # includes the thin-data early weeks. This checks whether the card does
    # better once teams have several games of current-season data.
    play["phase"] = pd.cut(play["week"], [0, 3, 8, 13, 99], labels=["weeks 1-3", "weeks 4-8", "weeks 9-13", "weeks 14+"])
    out["by_phase"] = ([gp.summarize(g, f"bet card | {k}") for k, g in play[play.bet_card].groupby("phase", observed=True)]
                       + [gp.summarize(play[play.bet_card & (play.week >= 4)], "bet card | weeks 4+ combined")]
                       + [gp.summarize(g, f"not on card | {k}") for k, g in play[~play.bet_card].groupby("phase", observed=True)])
    gp.print_rows("WHERE DOES THE MODEL BEAT THE SPREAD? (out of sample)", out["overall"])
    gp.print_rows("Bet Card vs. the rest by point in the season", out["by_phase"])
    gp.print_rows("By game type (spreads under 20)", out["game_type"])
    gp.print_rows("Bet Card plays by game type", out["game_type_bet_card"])
    gp.print_rows("By spread size (all games)", out["spread_size"])
    gp.print_rows("By how far the model is from the market (spreads under 20)", out["edge_size"])
    return out


def run_moneyline_tests(scoreable: pd.DataFrame) -> dict:
    """Where (if anywhere) the model has a MONEYLINE edge, graded at real
    closing moneyline prices, out of sample (added 10/2026):
      1. the model's picks as the dashboard makes them, by price range and
         model confidence (incl. the near-even target: +100..+150, model 65%+);
      2. calibration -- when the model says X%, how often does that team win?
      3. corrected + market-blended probabilities: a correction and blend
         weight learned on 2024 only, applied to 2025 (so 2025 stays honest);
      4. spread/moneyline mismatches: a book's moneyline that disagrees with
         its own spread, independent of the model.
    """
    import numpy as np
    from scipy.stats import norm
    from sklearn.isotonic import IsotonicRegression
    from src.analysis import game_profile as gp

    def implied(a):
        a = np.asarray(a, dtype=float)
        return np.where(a < 0, -a / (-a + 100), 100 / (a + 100))

    def payout(a):
        a = np.asarray(a, dtype=float)
        return np.where(a > 0, a / 100, 100 / -a)

    need = ["market_moneyline_home", "market_moneyline_away", "predicted_home_win_prob", "home_win"]
    df = scoreable.dropna(subset=[c for c in need if c in scoreable.columns]).copy()
    if not set(need).issubset(df.columns) or df.empty:
        print("  [warn] no historical moneyline prices -- skipping moneyline tests")
        return {}
    df = df[(df.market_moneyline_home.abs() >= 100) & (df.market_moneyline_away.abs() >= 100)]
    ih, ia = implied(df.market_moneyline_home), implied(df.market_moneyline_away)
    df["mkt_home"] = ih / (ih + ia)                          # market win chance, vig removed
    df["p_model"] = df["predicted_home_win_prob"].clip(0.01, 0.99)

    def bets(frame, pcol, min_edge_pp=4.94):
        """The dashboard's moneyline rule on probability column pcol: bet the
        side the model rates above the market by 4.94+ pts (the 1.9 'edge'
        threshold rescaled), price within +/-450, positive EV at that price."""
        f = frame.copy()
        home_side = f[pcol] > f["mkt_home"]
        f["side_prob"] = np.where(home_side, f[pcol], 1 - f[pcol])
        f["price"] = np.where(home_side, f.market_moneyline_home, f.market_moneyline_away)
        f["won"] = np.where(home_side, f.home_win == 1, f.home_win == 0)
        f["ev"] = f.side_prob * payout(f.price) - (1 - f.side_prob)
        edge = (f[pcol] - f["mkt_home"]).abs() * 100
        f = f[(edge >= min_edge_pp) & (f.price.abs() <= 450) & (f.ev > 0)].copy()
        f["result"] = np.where(f.won, "W", "L")
        f["units"] = np.where(f.won, payout(f.price), -1.0)
        return f

    out = {}
    raw = bets(df, "p_model")
    price_b = pd.cut(raw.price, [-1000, -250, -150, -100, 150, 300, 1000],
                     labels=["favorite -250 or more", "favorite -150 to -249", "favorite -101 to -149",
                             "near even +100 to +150", "underdog +151 to +300", "underdog +301 or more"])
    conf_b = pd.cut(raw.side_prob, [0, 0.55, 0.65, 0.75, 1], labels=["model <55%", "model 55-65%", "model 65-75%", "model 75%+"])
    target = raw[(raw.price >= 100) & (raw.price <= 150) & (raw.side_prob >= 0.65)]
    out["model_picks"] = [gp.summarize(raw, "every model moneyline pick"), gp.summarize(target, "near-even target (+100..+150, model 65%+)")]
    out["by_price"] = [gp.summarize(g, str(k)) for k, g in raw.groupby(price_b, observed=True)]
    out["by_confidence"] = [gp.summarize(g, str(k)) for k, g in raw.groupby(conf_b, observed=True)]
    gp.print_rows("MONEYLINES: the model's picks at real closing prices (out of sample)", out["model_picks"])
    gp.print_rows("Moneylines by price", out["by_price"])
    gp.print_rows("Moneylines by model confidence", out["by_confidence"])

    # 2. Calibration (favorite's perspective so every bucket has games)
    fav_p = np.maximum(df.p_model, 1 - df.p_model)
    fav_won = np.where(df.p_model >= 0.5, df.home_win == 1, df.home_win == 0)
    mkt_fav = np.where(df.p_model >= 0.5, df.mkt_home, 1 - df.mkt_home)
    cal = pd.DataFrame({"p": fav_p, "won": fav_won, "mkt": mkt_fav})
    cal["bucket"] = pd.cut(cal.p, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], labels=["50-60%", "60-70%", "70-80%", "80-90%", "90%+"])
    calib = []
    print("\nCALIBRATION: when the model says a team wins X%, how often did it?")
    print(f"  {'model says':<12}{'games':>7}{'model avg':>11}{'actually won':>14}{'market said':>13}")
    for k, g in cal.groupby("bucket", observed=True):
        row = {"bucket": str(k), "n": int(len(g)), "model_avg": round(float(g.p.mean()), 3),
               "actual": round(float(g.won.mean()), 3), "market_avg": round(float(g.mkt.mean()), 3)}
        calib.append(row)
        print(f"  {row['bucket']:<12}{row['n']:>7}{row['model_avg']*100:>10.1f}%{row['actual']*100:>13.1f}%{row['market_avg']*100:>12.1f}%")
    out["calibration"] = calib

    # 3. Corrected + blended probabilities: learn on 2024, apply to 2025
    seasons = sorted(df.season.unique())
    if len(seasons) >= 2:
        train, test = df[df.season < seasons[-1]].copy(), df[df.season == seasons[-1]].copy()
        iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(train.p_model, train.home_win)
        train["p_cal"], test["p_cal"] = iso.predict(train.p_model), iso.predict(test.p_model)
        best_w, best_ll = 0.0, 9e9
        for w in np.linspace(0, 1, 21):
            pb = np.clip(w * train.p_cal + (1 - w) * train.mkt_home, 0.01, 0.99)
            ll = -np.mean(train.home_win * np.log(pb) + (1 - train.home_win) * np.log(1 - pb))
            if ll < best_ll:
                best_w, best_ll = w, ll
        test["p_blend"] = np.clip(best_w * test.p_cal + (1 - best_w) * test.mkt_home, 0.01, 0.99)
        raw_t = raw[raw.season == seasons[-1]]
        cal_b, blend_b = bets(test, "p_cal"), bets(test, "p_blend", min_edge_pp=2.0)
        out["blend_weight_on_model"] = round(float(best_w), 2)
        out["corrected"] = [gp.summarize(raw_t, f"{seasons[-1]}: model as-is"),
                            gp.summarize(cal_b, f"{seasons[-1]}: corrected model"),
                            gp.summarize(blend_b, f"{seasons[-1]}: corrected + blended ({best_w:.0%} model)")]
        gp.print_rows(f"CORRECTED / BLENDED probabilities (learned on {seasons[:-1]}, graded on {seasons[-1]})", out["corrected"])
        if not blend_b.empty:
            bp = pd.cut(blend_b.price, [-1000, -150, -100, 150, 1000], labels=["favorite -150+", "favorite -101..-149", "near even +100..+150", "underdog +151+"])
            out["blended_by_price"] = [gp.summarize(g, f"blended | {k}") for k, g in blend_b.groupby(bp, observed=True)]
            gp.print_rows("Blended picks by price", out["blended_by_price"])

    # 4. Moneyline vs the book's own spread (model-independent)
    if "market_spread_home" in df.columns:
        sp = df.dropna(subset=["market_spread_home"]).copy()
        sp["resid"] = sp["margin"] + sp["market_spread_home"]
        sigma = {}
        for s_ in sp.season.unique():
            other = sp[sp.season != s_]
            sigma[s_] = float(other.resid.std()) if len(other) > 100 else float(sp.resid.std())
        sp["spread_home_p"] = [norm.cdf(-m / sigma[s_]) for m, s_ in zip(sp.market_spread_home, sp.season)]
        rows = []
        for gap in (3, 5, 8):
            home_cheap = sp.spread_home_p - sp.mkt_home >= gap / 100
            away_cheap = sp.mkt_home - sp.spread_home_p >= gap / 100
            f = pd.concat([sp[home_cheap].assign(price=lambda d: d.market_moneyline_home, won=lambda d: d.home_win == 1),
                           sp[away_cheap].assign(price=lambda d: d.market_moneyline_away, won=lambda d: d.home_win == 0)])
            f = f[f.price.abs() <= 450]
            f["result"] = np.where(f.won, "W", "L")
            f["units"] = np.where(f.won, payout(f.price), -1.0)
            rows.append(gp.summarize(f, f"moneyline cheaper than its spread by {gap}+ pts"))
        out["spread_vs_moneyline"] = rows
        gp.print_rows("SPREAD vs MONEYLINE mismatches (no model involved)", rows)
    return out


def load_historical_lines() -> pd.DataFrame:
    """Load every lines_{year}.csv found in data/raw/, already flattened to
    one row per game by cfbd_client.historical_lines_to_dataframe (see
    fetch_historical_data.py). Older cached CSVs fetched before that fix
    will be missing the market_spread_home/market_moneyline_* columns this
    script needs — caught explicitly below rather than failing on a
    confusing KeyError."""
    files = [f for f in os.listdir(config.DATA_RAW_DIR) if f.startswith("lines_") and f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    frames = [pd.read_csv(f"{config.DATA_RAW_DIR}/{f}") for f in files]
    return pd.concat(frames, ignore_index=True)


def join_features_to_lines(features: pd.DataFrame, lines: pd.DataFrame) -> tuple:
    """Join on CFBD's own numeric game id when both sides have it. /games
    and /lines are both CFBD's own data sharing the same internal id
    scheme, so this is an exact join — no fuzzy team-name matching needed
    here (that's only required when matching CFBD to a DIFFERENT provider,
    like The Odds API, which is what export_dashboard_data.py's team
    matching handles for current/live lines).

    Falls back to a (season, week, homeTeam, awayTeam) join if 'id' is
    missing from either side, so this doesn't silently produce zero
    matches against an older cached features/lines CSV that predates this
    fix. Returns (joined_df, strategy_used) so the caller can report which
    path was actually taken.
    """
    if "id" in features.columns and "id" in lines.columns:
        merged = features.merge(lines, on="id", how="inner", suffixes=("", "_line"))
        return merged, "id"
    merged = features.merge(
        lines, on=["season", "week", "homeTeam", "awayTeam"], how="inner", suffixes=("", "_line")
    )
    return merged, "season+week+homeTeam+awayTeam (fallback — no shared 'id' column found)"


if __name__ == "__main__":
    features_path = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    model_path = f"{config.MODELS_DIR}/game_model.joblib"

    if not os.path.exists(features_path):
        raise SystemExit("Run scripts/build_features.py first.")

    features = pd.read_csv(features_path)
    lines = load_historical_lines()
    if lines.empty:
        raise SystemExit("No historical lines found in data/raw/ — run scripts/fetch_historical_data.py first.")
    if "market_spread_home" not in lines.columns:
        raise SystemExit(
            "data/raw/lines_*.csv is missing market_spread_home — it was fetched before the "
            "historical_lines_to_dataframe fix. Re-run scripts/fetch_historical_data.py to refresh it."
        )

    print(f"Loaded {len(features)} historical feature rows and {len(lines)} historical lines with a real market number.")
    joined, strategy = join_features_to_lines(features, lines)
    print(f"  joined {len(joined)} of {len(features)} feature rows to a market line (join strategy: {strategy})")
    if joined.empty:
        raise SystemExit("Joined 0 games to a market line — nothing to backtest. Check that features and "
                          "lines cover the same season(s)/years.")

    # ---- Walk-forward (out-of-sample) backtest -- rebuilt 9/2026 ----
    # The old version loaded the production model (trained on a random 80%
    # of ALL historical games) and graded it on 100% of those same games, so
    # roughly 4 of every 5 "backtest" games were ones the model had already
    # trained on, answers included. That in-sample overlap is what produced
    # the 61.25% ATS number, which live results never came close to.
    #
    # Now each season is graded by a model trained ONLY on earlier seasons
    # -- exactly the situation live betting is in. The first season on file
    # has nothing earlier to train on, so it's used for training only, never
    # graded. The production model (models/game_model.joblib) is untouched;
    # this script only trains temporary per-season models for grading.
    if "season" not in joined.columns:
        raise SystemExit("Joined data has no 'season' column -- can't run a walk-forward backtest.")
    # Only seasons with prior-year SP+ on file count -- the earliest season
    # has none (ratings are joined from the year before), so a model trained
    # on it alone would be missing its main inputs.
    has_sp = features["sp_rating_diff"].notna() if "sp_rating_diff" in features.columns else features["season"].notna()
    seasons = sorted(int(x) for x in features.loc[has_sp, "season"].dropna().unique())
    features = features[features["season"].isin(seasons)]
    print(f"\nWalk-forward backtest over seasons {seasons}: each season is predicted by a model "
          f"trained only on the seasons before it.")

    graded_parts = []
    for test_season in seasons[1:]:
        train_rows = features[features["season"] < test_season]
        fold_model = GameMarginModel()
        try:
            fold_model.fit(train_rows, verbose=False)
        except Exception as e:
            print(f"  {test_season}: skipped -- could not train on earlier seasons ({e})")
            continue
        test_rows = joined[joined["season"] == test_season].dropna(subset=fold_model.feature_columns).copy()
        if test_rows.empty:
            print(f"  {test_season}: skipped -- no games with complete features and a market line")
            continue
        test_rows["predicted_margin"] = fold_model.predict_margin(test_rows)
        test_rows["predicted_home_win_prob"] = fold_model.predict_home_win_prob(test_rows)
        test_rows["fold_sd"] = fold_model.residual_std
        fold = backtester.evaluate_spread(
            test_rows["predicted_margin"], test_rows["margin"], test_rows["market_spread_home"]
        )
        print(f"  {test_season}: trained on {len(train_rows)} earlier-season rows, graded "
              f"{fold['n_games']} games -> {fold['ats_win_rate']:.1%} ATS")
        graded_parts.append(test_rows)

    if not graded_parts:
        raise SystemExit("No season could be graded out-of-sample -- need at least two seasons of data.")
    scoreable = pd.concat(graded_parts, ignore_index=True)

    spread_results = backtester.evaluate_spread(
        scoreable["predicted_margin"], scoreable["margin"], scoreable["market_spread_home"]
    )
    backtester.summarize(spread_results, "Spread (ATS vs. real CFBD closing line, out-of-sample)")

    ml_results = backtester.evaluate_moneyline(
        scoreable["predicted_home_win_prob"], scoreable["home_win"]
    )
    backtester.summarize(ml_results, "Moneyline (calibration: model win-prob vs. actual outcome)")

    print("\nReminder: breakeven ATS win rate against standard -110 pricing is ~52.4%. "
          "Treat any result on a small early sample (well under ~200 graded games) as noisy, "
          "not a verdict — re-run this after more historical seasons/weeks are pulled.")

    splits = run_splits(scoreable)
    try:
        splits["moneyline"] = run_moneyline_tests(scoreable)
    except Exception as e:
        print(f"  [warn] moneyline tests failed: {e}")

    # Persist results to the repo (rather than leaving them stranded in this
    # Action run's console log, which isn't fetchable outside the GitHub UI)
    # so build_dashboard.py can surface a real "Backtest Track Record" panel
    # and so results are diffable/trackable across runs as the model changes.
    seasons_covered = sorted(int(s) for s in scoreable["season"].dropna().unique()) if "season" in scoreable.columns else []
    backtest_output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons_covered": seasons_covered,
        "join_strategy": strategy,
        "method": "walk-forward: each season graded by a model trained only on earlier seasons",
        "spread": {
            "n_games": spread_results["n_games"],
            "n_pushes": spread_results["n_pushes"],
            "ats_win_rate": (None if pd.isna(spread_results["ats_win_rate"]) else round(float(spread_results["ats_win_rate"]), 4)),
            "margin_mae": round(float(spread_results["margin_mae"]), 3),
            "breakeven_ats_rate": spread_results["breakeven_ats_rate"],
            "beat_market": (None if spread_results["beat_market"] is None else bool(spread_results["beat_market"])),
        },
        "moneyline": {
            "n_games": ml_results["n_games"],
            "accuracy": round(float(ml_results["accuracy"]), 4),
            "log_loss": round(float(ml_results["log_loss"]), 4),
        },
        "splits": splits,
    }
    os.makedirs("docs/data", exist_ok=True)
    with open("docs/data/backtest_results.json", "w") as f:
        json.dump(backtest_output, f, indent=2)
    print(f"\nWrote docs/data/backtest_results.json ({spread_results['n_games']} graded spread games, "
          f"seasons {seasons_covered}).")
