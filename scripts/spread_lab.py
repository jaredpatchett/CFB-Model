#!/usr/bin/env python3
"""
Spread lab (added 10/2026): does PFF data make the spread model better?

Backtests the live spread model against versions that add PFF team ratings,
on exactly the same games. Nothing here touches the live model, the
dashboard or the Bet Card.

Versions
  current    the live spread model (per-play efficiency, QB availability,
             recent-game weighting)
  pff_core   current + four PFF ratings: offense grade, defense grade,
             quarterback passing grade, quarterback EPA per dropback
  pff_full   current + every PFF rating below + three matchup inputs

PFF team ratings (built per team, per game, from player grades weighted by
snaps): offense, pass blocking, run blocking, receiving, rushing, quarterback
grade, quarterback EPA per dropback, turnover-worthy play rate, pressure
allowed per dropback, defense, pass rush, coverage, run defense, tackling,
pressure generated, missed-tackle rate.
  Each rating going into a game uses only that team's EARLIER games this
  season, pulled toward where the team finished last season until enough
  games are played. Matchups: pass protection vs the opponent's pass rush,
  run blocking vs their run defense, receivers vs their coverage.

How it is graded (same method as scripts/run_backtest.py)
  Walk-forward: each season is predicted by models trained only on earlier
  seasons. All versions are graded on the same games:
    - how far the projected margin was from the final margin,
    - win-probability accuracy (log loss),
    - against the CLOSING spread, every game and each version's own Bet Card
      (top 10 per week, spreads under 20),
    - against the OPENING spread, and how often the line then moved toward
      the version's side.

Outputs docs/data/spread_lab.json -- model results only. PFF data is pulled
into a temp folder outside the repo and never committed (the repo is public).

Usage (after fetch_historical_data.py and build_features.py; needs PFF key):
  python scripts/spread_lab.py --years 2022 2023 2024 2025
"""
import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
from scipy.stats import norm

import config
import run_backtest as rb
from src.analysis import game_profile as gp
from src.data import pff_client as pff
from src.features import pff_features as pf
from src.features.player_features import pivot_player_game_stats
from src.features.team_features_v2 import FEATURE_COLUMNS_V2
from src.models.game_model import GameMarginModel

OUT = "docs/data/spread_lab.json"
PRIOR_GAMES = 2            # last season's rating counts as this many games
MATCH_WINDOW = pd.Timedelta(hours=48)
REPORTS = {                # name -> API path
    "offense": "/v1/facet/offense/summary",
    "defense": "/v1/facet/defense/summary",
    "passing": "/v1/facet/passing/summary",
    "blocking": "/v1/facet/offense/blocking",
}
METRICS = ["off_grade", "pass_block", "run_block", "route", "run_grade", "qb_grade", "qb_epa", "qb_twp",
           "pressure_allowed", "def_grade", "pass_rush", "coverage", "run_def", "tackle", "pressure_made",
           "missed_tackle"]
MATCHUPS = ["pff_pass_pro_matchup", "pff_run_matchup", "pff_pass_game_matchup"]
PFF_CORE = ["pff_off_grade_diff", "pff_def_grade_diff", "pff_qb_grade_diff", "pff_qb_epa_diff"]
PFF_FULL = [f"pff_{m}_diff" for m in METRICS] + MATCHUPS
VARIANTS = {
    "current": (list(FEATURE_COLUMNS_V2), "Live spread model"),
    "pff_core": (list(FEATURE_COLUMNS_V2) + PFF_CORE, "Live + 4 PFF ratings (offense, defense, QB grade, QB EPA)"),
    "pff_full": (list(FEATURE_COLUMNS_V2) + PFF_FULL, "Live + every PFF rating and matchup"),
}


# ------------------------------------------------------------- PFF pull ----
def _pull(path, season, week):
    for attempt in range(3):
        try:
            payload = pff._get(path, {"league": pff.LEAGUE, "season": season, "week": week})
            rows = payload if isinstance(payload, list) else next((v for v in (payload or {}).values() if isinstance(v, list)), [])
            df = pd.DataFrame(rows)
            if not df.empty:
                df["season"], df["week"] = season, week
            return df
        except pff.PFFError as e:
            if e.status in (401, 403):
                raise
            return pd.DataFrame()
        except Exception:
            time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def fetch_pff(seasons, out_dir):
    """{report: all weeks} plus game kickoffs. Cached per season in out_dir,
    which is outside the repo."""
    os.makedirs(out_dir, exist_ok=True)
    data = {k: [] for k in list(REPORTS) + ["games"]}
    for season in seasons:
        for name, path in REPORTS.items():
            fp = f"{out_dir}/pff_{name}_{season}.csv"
            if os.path.exists(fp):
                df = pd.read_csv(fp)
            else:
                parts = [d for d in (_pull(path, season, w) for w in pff.MODEL_WEEKS) if not d.empty]
                df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
                if not df.empty:
                    df.to_csv(fp, index=False)
            print(f"  PFF {season} {name}: {len(df)} rows")
            data[name].append(df)
        gp_path = f"{out_dir}/pff_games_{season}.csv"
        if os.path.exists(gp_path):
            g = pd.read_csv(gp_path)
        else:
            parts = []
            for w in pff.MODEL_WEEKS:
                try:
                    d = pff.get_games(season, w)
                    if not d.empty:
                        parts.append(d)
                except Exception:
                    continue
            g = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
            if not g.empty:
                g.to_csv(gp_path, index=False)
        data["games"].append(g)
    return {k: (pd.concat([d for d in v if not d.empty], ignore_index=True) if any(not d.empty for d in v) else pd.DataFrame())
            for k, v in data.items()}


# ------------------------------------------------- team ratings per game ----
def _n(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _weighted(df, grade, snaps, keys):
    """Snap-weighted average of a grade per team-week."""
    g, s = _n(df, grade), _n(df, snaps).fillna(0)
    ok = g.notna() & (s > 0)
    t = pd.DataFrame({"k0": df[keys[0]], "k1": df[keys[1]], "k2": df[keys[2]], "gs": (g * s).where(ok), "s": s.where(ok)})
    a = t.groupby(["k0", "k1", "k2"])[["gs", "s"]].sum(min_count=1)
    a.index.names = keys
    return (a["gs"] / a["s"]).rename(None)


def team_game_ratings(d):
    """One row per PFF team-week with the METRICS columns."""
    keys = ["season", "week", "franchise_id"]
    off, deff, pas, blk = d["offense"], d["defense"], d["passing"], d["blocking"]
    out = {}
    out["off_grade"] = _weighted(off, "grades_offense", "snap_counts_total", keys)
    out["pass_block"] = _weighted(off, "grades_pass_block", "snap_counts_pass_block", keys)
    out["run_block"] = _weighted(off, "grades_run_block", "snap_counts_run_block", keys)
    out["route"] = _weighted(off, "grades_pass_route", "snap_counts_pass_route", keys)
    out["run_grade"] = _weighted(off, "grades_run", "snap_counts_run", keys)
    out["qb_grade"] = _weighted(pas, "grades_pass", "dropbacks", keys)
    out["def_grade"] = _weighted(deff, "grades_defense", "snap_counts_defense", keys)
    out["pass_rush"] = _weighted(deff, "grades_pass_rush_defense", "snap_counts_pass_rush", keys)
    out["coverage"] = _weighted(deff, "grades_coverage_defense", "snap_counts_coverage", keys)
    out["run_def"] = _weighted(deff, "grades_run_defense", "snap_counts_run_defense", keys)
    out["tackle"] = _weighted(deff, "grades_tackle", "snap_counts_defense", keys)
    t = pd.DataFrame(out)

    def sums(df, cols):
        x = pd.DataFrame({c: _n(df, c) for c in cols})
        for k in keys:
            x[k] = df[k].values
        return x.groupby(keys)[cols].sum(min_count=1)

    p = sums(pas, ["dropbacks", "epa", "turnover_worthy_plays"])
    b = sums(blk, ["pressures_allowed"])
    dd = sums(deff, ["total_pressures", "missed_tackles", "tackles", "assists"])
    cov = pd.DataFrame({"c": _n(deff, "snap_counts_coverage").values, **{k: deff[k].values for k in keys}}).groupby(keys)["c"].max()
    t = t.join(p, how="outer").join(b, how="outer").join(dd, how="outer").join(cov.rename("pass_plays_faced"), how="outer")
    db = t["dropbacks"].where(t["dropbacks"] > 0)
    t["qb_epa"] = t["epa"] / db
    t["qb_twp"] = t["turnover_worthy_plays"] / db
    t["pressure_allowed"] = t["pressures_allowed"] / db
    t["pressure_made"] = t["total_pressures"] / t["pass_plays_faced"].where(t["pass_plays_faced"] > 0)
    att = t["missed_tackles"] + t["tackles"] + t["assists"]
    t["missed_tackle"] = t["missed_tackles"] / att.where(att > 0)
    t = t[METRICS].reset_index()
    t.columns = keys + METRICS
    return t


def franchise_to_school(offense, player_stats_long, games):
    """PFF franchise id -> CFBD school name, per season, by majority vote of
    players whose names are unique in both sources that season."""
    wide = pivot_player_game_stats(player_stats_long, games)
    cf = wide[["season", "athleteId", "player", "team"]].drop_duplicates()
    cf["n"] = cf["player"].map(pf.norm_name)
    cf = cf[cf.groupby(["season", "n"])["athleteId"].transform("nunique") == 1].drop_duplicates(["season", "n", "team"])
    cf = cf[cf.groupby(["season", "n"])["team"].transform("nunique") == 1]
    po = offense[["season", "player", "player_id", "franchise_id"]].drop_duplicates()
    po["n"] = po["player"].map(pf.norm_name)
    po = po[po.groupby(["season", "n"])["player_id"].transform("nunique") == 1].drop_duplicates(["season", "n", "franchise_id"])
    po = po[po.groupby(["season", "n"])["franchise_id"].transform("nunique") == 1]
    v = po.merge(cf[["season", "n", "team"]], on=["season", "n"])
    tally = v.groupby(["season", "franchise_id", "team"]).size().rename("votes").reset_index()
    tally["share"] = tally["votes"] / tally.groupby(["season", "franchise_id"])["votes"].transform("sum")
    best = tally.sort_values("votes", ascending=False).drop_duplicates(["season", "franchise_id"])
    best = best[(best["votes"] >= 5) & (best["share"] >= 0.6)]
    return best[["season", "franchise_id", "team"]]


def attach_to_cfbd_games(ratings, fmap, pff_games, games):
    """PFF team-week ratings -> CFBD (game id, school) rows, lined up by
    kickoff time (the two sources number weeks differently)."""
    r = ratings.merge(fmap, on=["season", "franchise_id"], how="inner")
    g = pff_games.copy()
    g["kick_pff"] = pd.to_datetime(g["start"], utc=True, errors="coerce")
    long = pd.concat([g[["season", "week", "home_franchise_id", "kick_pff"]].rename(columns={"home_franchise_id": "franchise_id"}),
                      g[["season", "week", "away_franchise_id", "kick_pff"]].rename(columns={"away_franchise_id": "franchise_id"})])
    long = long.dropna(subset=["franchise_id", "kick_pff"]).drop_duplicates(["season", "week", "franchise_id"])
    r = r.merge(long, on=["season", "week", "franchise_id"], how="inner")
    cg = games.dropna(subset=["id"]).drop_duplicates("id")
    tg = pd.concat([cg[["id", "season", "homeTeam", "startDate"]].rename(columns={"homeTeam": "team"}),
                    cg[["id", "season", "awayTeam", "startDate"]].rename(columns={"awayTeam": "team"})])
    tg["kick"] = pd.to_datetime(tg["startDate"], utc=True, errors="coerce")
    m = r.merge(tg[["id", "season", "team", "kick"]], on=["season", "team"])
    m = m[(m["kick_pff"] - m["kick"]).abs() <= MATCH_WINDOW]
    m["gap"] = (m["kick_pff"] - m["kick"]).abs()
    m = m.sort_values("gap").drop_duplicates(["id", "team"])
    return tg, m[["id", "team"] + METRICS]


def pregame_ratings(team_games, rated):
    """Each team's rating going INTO each game: its earlier games this
    season, pulled toward last season's average (worth PRIOR_GAMES games)."""
    df = team_games.merge(rated, on=["id", "team"], how="left").sort_values(["team", "season", "kick"])
    season_avg = df.groupby(["team", "season"])[METRICS].mean().reset_index()
    season_avg["season"] = season_avg["season"] + 1                      # last season's average, keyed to this season
    df = df.merge(season_avg, on=["team", "season"], how="left", suffixes=("", "_prior"))
    grp = df.groupby(["team", "season"])
    for m in METRICS:
        s = grp[m].transform(lambda x: x.shift(1).expanding().sum())
        n = grp[m].transform(lambda x: x.shift(1).expanding().count())
        prior = df[f"{m}_prior"]
        with_prior = (s.fillna(0) + PRIOR_GAMES * prior) / (n.fillna(0) + PRIOR_GAMES)
        df[f"pre_{m}"] = np.where(prior.notna(), with_prior, np.where(n > 0, s / n.where(n > 0), np.nan))
    return df[["id", "team"] + [f"pre_{m}" for m in METRICS]]


def add_pff_game_features(features, pre):
    h = pre.rename(columns={"team": "homeTeam", **{f"pre_{m}": f"h_{m}" for m in METRICS}})
    a = pre.rename(columns={"team": "awayTeam", **{f"pre_{m}": f"a_{m}" for m in METRICS}})
    df = features.merge(h, on=["id", "homeTeam"], how="left").merge(a, on=["id", "awayTeam"], how="left")
    for m in METRICS:
        df[f"pff_{m}_diff"] = df[f"h_{m}"] - df[f"a_{m}"]
    df["pff_pass_pro_matchup"] = (df["h_pass_block"] - df["a_pass_rush"]) - (df["a_pass_block"] - df["h_pass_rush"])
    df["pff_run_matchup"] = (df["h_run_block"] - df["a_run_def"]) - (df["a_run_block"] - df["h_run_def"])
    df["pff_pass_game_matchup"] = (df["h_route"] - df["a_coverage"]) - (df["a_route"] - df["h_coverage"])
    return df.drop(columns=[c for c in df.columns if c.startswith("h_") and c[2:] in METRICS or c.startswith("a_") and c[2:] in METRICS])


# -------------------------------------------------------------- grading ----
def grade(d, mcol, line_col):
    d = d.copy()
    d["edge"] = d[mcol] + d[line_col]
    d = d[d["edge"] != 0]
    cover = d["margin"] + d[line_col]
    d["result"] = ["P" if c == 0 else ("W" if (c > 0) == (e > 0) else "L") for c, e in zip(cover, d["edge"])]
    d["units"] = d["result"].map({"W": 100 / 110, "L": -1.0, "P": 0.0})
    return d


def card(d, sdcol):
    d = d.copy()
    d["cover_prob"] = norm.cdf(d["edge"].abs() / d[sdcol])
    d["rank"] = d.groupby(["season", "week"])["cover_prob"].rank(ascending=False, method="first")
    return d[d["rank"] <= 10]


def evaluate(base, name):
    mcol, sdcol, pcol = f"m_{name}", f"sd_{name}", f"p_{name}"
    out = {"margin_mae": round(float((base[mcol] - base["margin"]).abs().mean()), 3),
           "margin_rmse": round(float(np.sqrt(((base[mcol] - base["margin"]) ** 2).mean())), 3)}
    p = base[pcol].clip(0.01, 0.99)
    out["win_prob_log_loss"] = round(float(-np.mean(base["home_win"] * np.log(p) + (1 - base["home_win"]) * np.log(1 - p))), 4)
    out["mae_by_season"] = {str(int(s)): round(float((g[mcol] - g["margin"]).abs().mean()), 3) for s, g in base.groupby("season")}
    close = grade(base, mcol, "market_spread_home")
    out["vs_closing"] = {"every_game": gp.summarize(close, "every game"), "bet_card": gp.summarize(card(close, sdcol), "bet card"),
                         "bet_card_weeks_4_plus": gp.summarize(card(close, sdcol).query("week >= 4"), "bet card weeks 4+"),
                         "edge_3_plus": gp.summarize(close[close.edge.abs() >= 3], "model 3+ pts off the line"),
                         "edge_7_plus": gp.summarize(close[close.edge.abs() >= 7], "model 7+ pts off the line")}
    if "market_spread_open_home" in base.columns:
        ob = base.dropna(subset=["market_spread_open_home"])
        ob = ob[ob["market_spread_open_home"].abs() < 20]
        op = grade(ob, mcol, "market_spread_open_home")
        op["move_toward"] = (op["market_spread_open_home"] - op["market_spread_home"]) * np.sign(op["edge"])
        moved = op[op["move_toward"] != 0]
        oc = card(op, sdcol)
        ocm = oc[oc["move_toward"] != 0]
        out["vs_opening"] = {"every_game": gp.summarize(op, "every game vs opener"), "bet_card": gp.summarize(oc, "bet card vs opener"),
                             "lines_moved": int(len(moved)),
                             "share_moved_toward_model": round(float((moved["move_toward"] > 0).mean()), 3) if len(moved) else None,
                             "avg_move_toward_model_pts": round(float(op["move_toward"].mean()), 3),
                             "card_share_moved_toward_model": round(float((ocm["move_toward"] > 0).mean()), 3) if len(ocm) else None,
                             "card_avg_move_pts": round(float(oc["move_toward"].mean()), 3) if len(oc) else None}
    return out


def main(years):
    t0 = time.time()
    fpath = f"{config.DATA_PROCESSED_DIR}/team_game_features.csv"
    if not os.path.exists(fpath):
        raise SystemExit("Run scripts/build_features.py first.")
    features = pd.read_csv(fpath)
    games = rb_load("games", years)
    stats = rb_load("player_game_stats", years)
    lines = rb.load_historical_lines()
    if games.empty or stats.empty or lines.empty:
        raise SystemExit("Historical CFBD files missing -- run scripts/fetch_historical_data.py first.")
    print(f"{len(features)} games with spread-model inputs; pulling PFF team data (kept outside the repo)...")
    d = fetch_pff(years, os.environ.get("PFF_DATA_DIR") or os.path.join(tempfile.gettempdir(), "pff_spread_lab"))
    if any(d[k].empty for k in REPORTS):
        raise SystemExit(f"PFF returned no rows for: {[k for k in REPORTS if d[k].empty]}")
    ratings = team_game_ratings(d)
    fmap = franchise_to_school(d["offense"], stats, games)
    team_games, rated = attach_to_cfbd_games(ratings, fmap, d["games"], games)
    pre = pregame_ratings(team_games, rated)
    feats = add_pff_game_features(features, pre)
    info = {"pff_team_weeks": int(len(ratings)), "franchises_mapped": int(len(fmap)),
            "cfbd_team_games_with_pff": int(len(rated)), "cfbd_team_games": int(len(team_games)),
            "pct_games_with_pff_inputs": {str(int(s)): round(float(g[PFF_CORE].notna().all(axis=1).mean()) * 100, 1)
                                          for s, g in feats.groupby("season")}}
    print(f"  PFF coverage: {info}")

    joined, strategy = rb.join_features_to_lines(feats, lines)
    has_sp = feats["sp_rating_diff"].notna() if "sp_rating_diff" in feats.columns else feats["season"].notna()
    seasons = sorted(int(x) for x in feats.loc[has_sp, "season"].dropna().unique())
    feats = feats[feats["season"].isin(seasons)]
    parts, folds = [], []
    for test_season in seasons[1:]:
        train = feats[feats["season"] < test_season]
        test = joined[joined["season"] == test_season].copy()
        row = {"season": test_season}
        for name, (cols, _) in VARIANTS.items():
            m = GameMarginModel()
            m.feature_columns = list(cols)
            try:
                met = m.fit(train, verbose=False)
            except Exception as e:
                print(f"  {test_season} {name}: could not train ({e})")
                continue
            ok = test[m.feature_columns].notna().all(axis=1)
            test[f"m_{name}"], test[f"p_{name}"] = np.nan, np.nan
            if ok.any():
                test.loc[ok, f"m_{name}"] = m.predict_margin(test[ok])
                test.loc[ok, f"p_{name}"] = m.predict_home_win_prob(test[ok])
            test[f"sd_{name}"] = m.residual_std
            row[name] = {"train_games": int(met["n_train"]), "inputs": len(m.feature_columns), "games_scored": int(ok.sum()),
                         "top_inputs": list(met["feature_importances"].items())[:8]}
        folds.append(row)
        parts.append(test)
        print(f"  {test_season}: " + ", ".join(f"{k} scored {v['games_scored']}" for k, v in row.items() if k != "season"))
    if not parts:
        raise SystemExit("No season could be graded -- need at least two seasons.")
    scored = pd.concat(parts, ignore_index=True)
    names = [n for n in VARIANTS if f"m_{n}" in scored.columns]
    base = scored.dropna(subset=[f"m_{n}" for n in names] + ["market_spread_home", "margin"])
    base = base[base["market_spread_home"].abs() < 20]
    results = {n: dict(evaluate(base, n), description=VARIANTS[n][1]) for n in names}

    # Where the PFF versions and the live model take opposite sides.
    disagreements = {}
    for n in names:
        if n == "current":
            continue
        e0, e1 = base["m_current"] + base["market_spread_home"], base[f"m_{n}"] + base["market_spread_home"]
        dis = base[(np.sign(e0) != np.sign(e1)) & (e0 != 0) & (e1 != 0)]
        disagreements[n] = {"games": int(len(dis)), "current": gp.summarize(grade(dis, "m_current", "market_spread_home"), "current"),
                            n: gp.summarize(grade(dis, f"m_{n}", "market_spread_home"), n)}
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "train_years": list(years), "seasons_graded": seasons[1:],
           "games_graded": int(len(base)), "join_strategy": strategy, "pff": info, "folds": folds,
           "results": results, "opposite_sides": disagreements,
           "method": "walk-forward: each season graded by models trained only on earlier seasons; same games for every version; spreads under 20"}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2, default=str)
    report(out)
    print(f"\nSaved {OUT} in {time.time() - t0:.0f}s")


def rb_load(name, years):
    frames = [pd.read_csv(f"{config.DATA_RAW_DIR}/{name}_{y}.csv") for y in years if os.path.exists(f"{config.DATA_RAW_DIR}/{name}_{y}.csv")]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def report(out):
    r = out["results"]
    names = list(r)
    cell = lambda x: f"{x['record']} {x['roi'] * 100:+.1f}%" if x and x.get("n") else "-"
    print(f"\nSPREAD LAB -- same {out['games_graded']} games, seasons {out['seasons_graded']}, out of sample")
    print(f"  {'':<34}" + "".join(f"{n:>24}" for n in names))
    print(f"  {'avg miss vs final margin (pts)':<34}" + "".join(f"{r[n]['margin_mae']:>24.3f}" for n in names))
    print(f"  {'win-probability log loss':<34}" + "".join(f"{r[n]['win_prob_log_loss']:>24.4f}" for n in names))
    for label, a, b in (("every game vs closing", "vs_closing", "every_game"), ("Bet Card vs closing", "vs_closing", "bet_card"),
                        ("Bet Card weeks 4+ vs closing", "vs_closing", "bet_card_weeks_4_plus"),
                        ("every game vs opening", "vs_opening", "every_game"), ("Bet Card vs opening", "vs_opening", "bet_card")):
        print(f"  {label:<34}" + "".join(f"{cell((r[n].get(a) or {}).get(b)):>24}" for n in names))
    print(f"  {'line moved toward model':<34}" + "".join(f"{str((r[n].get('vs_opening') or {}).get('share_moved_toward_model')):>24}" for n in names))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    main(sorted(ap.parse_args().years))
