#!/usr/bin/env python3
"""
Sack lab (11/16/2026, Rod): does the rush yards model get better at QB rush
yds unders if it knows about sacks? In college stats every sack counts
against the QB's rushing yards, but the model has no sack inputs.

Two versions of the live PlayerStatModel("rush_yds"), trained exactly like
rush_lab.py's "current" version (box scores only, seasons BEFORE the one
graded):
  current  -- today's inputs
  sacks    -- today's inputs + two sack inputs, both leakage-safe (only
              games before this one, same season):
                opp_def_sacks_prior     sacks per game this player's
                                        OPPONENT's defense has made
                team_sacks_taken_prior  sacks per game this player's own
                                        offense has given up (the QB/line's
                                        sack-proneness)
              A team with no prior games gets that season's league average.

Sacks come from CFBD box scores (category "defensive", statType "SACKS").

Lines, grading and the official rule are rush_lab.py's: past-season rush yds
lines (data/props_history/rush_yds_<season>.csv), best-EV under across books,
edge 0%+, 2+ games played, graded at that book's line and price. QBs are
players averaging 5+ pass attempts a game before this one.

Also reported: within the CURRENT model's QB unders, how they did split by
the opponent's pass rush (sacks per game, terciles) -- i.e. whether sacks
work as a filter even if the model doesn't need them as an input.

Outputs: docs/data/sack_lab.json, docs/data/sack_lab_detail.csv, printed
report. Does NOT touch the live model or the dashboard.

Usage:
  python scripts/sack_lab.py --years 2022 2023 2024 2025 --season 2026
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import props_lab as lab
import rush_lab as rl
from src.features.player_features import (pivot_player_game_stats, add_usage_features, attach_game_environment,
                                          build_rolling_player_features, ROLLING_FEATURE_COLUMNS)
from src.features.run_defense import run_defense, bucket as matchup_bucket
from src.models.props_model import PlayerStatModel

STAT = "rush_yds"
SACK_COLS = ["opp_def_sacks_prior", "team_sacks_taken_prior"]
OUT_JSON = "docs/data/sack_lab.json"
OUT_CSV = "docs/data/sack_lab_detail.csv"
QB_MIN_PASS_ATT = 5


def team_sacks(stats, games):
    """(gameId, team) -> sacks made by that team's defense in that game."""
    s = stats.copy()
    m = (s["category"].astype(str).str.lower() == "defensive") & (s["statType"].astype(str).str.upper() == "SACKS")
    s = s[m].copy()
    s["stat"] = pd.to_numeric(s["stat"], errors="coerce").fillna(0.0)
    s["gameId"] = pd.to_numeric(s["gameId"], errors="coerce")
    s = s.dropna(subset=["gameId"])
    s["gameId"] = s["gameId"].astype("int64")
    return s.groupby(["gameId", "team"])["stat"].sum().rename("sacks_made").reset_index()


def add_sack_features(wide, sacks):
    tg = wide.groupby(["gameId", "team"]).agg(season=("season", "first"), week=("week", "first"),
                                              opponent=("opponent", "first")).reset_index()
    tg = tg.merge(sacks, on=["gameId", "team"], how="left")
    tg["sacks_made"] = tg["sacks_made"].fillna(0.0)
    tg = tg.merge(sacks.rename(columns={"team": "opponent", "sacks_made": "sacks_taken"}), on=["gameId", "opponent"], how="left")
    tg["sacks_taken"] = tg["sacks_taken"].fillna(0.0)
    tg = tg.sort_values(["team", "season", "week"])
    g = tg.groupby(["team", "season"])
    tg["def_sacks_prior"] = g["sacks_made"].transform(lambda s: s.shift(1).expanding().mean())
    tg["team_sacks_taken_prior"] = g["sacks_taken"].transform(lambda s: s.shift(1).expanding().mean())
    league = tg.groupby("season")["sacks_made"].mean()
    print("  sacks per team-game by season: " + ", ".join(f"{int(k)} {v:.2f}" for k, v in league.items()))
    df = wide.merge(tg[["gameId", "team", "team_sacks_taken_prior"]], on=["gameId", "team"], how="left")
    df = df.merge(tg[["gameId", "team", "def_sacks_prior"]].rename(columns={"team": "opponent", "def_sacks_prior": "opp_def_sacks_prior"}),
                  on=["gameId", "opponent"], how="left")
    fill = df["season"].map(league)
    for c in SACK_COLS:
        df[c] = df[c].fillna(fill)
    return df, {int(k): round(float(v), 2) for k, v in league.items()}


def frames(years, season):
    print("Loading CFBD box scores...")
    g_h, s_h, l_h = lab.load_history(years)
    g_n, s_n, l_n = lab.load_current(season)
    games = pd.concat([g_h, g_n], ignore_index=True)
    games = games.assign(id=pd.to_numeric(games["id"], errors="coerce")).dropna(subset=["id"])
    games["id"] = games["id"].astype("int64")
    lines = pd.concat([x for x in (l_h, l_n) if x is not None and not x.empty], ignore_index=True)
    if not lines.empty and "id" in lines.columns:
        lines = lines.assign(id=pd.to_numeric(lines["id"], errors="coerce")).dropna(subset=["id"])
        lines["id"] = lines["id"].astype("int64")
    wide = rl._ints(pd.concat([pivot_player_game_stats(s_h, g_h), pivot_player_game_stats(s_n, g_n)], ignore_index=True))
    sacks = team_sacks(pd.concat([s_h, s_n], ignore_index=True), games)
    print(f"  {len(wide)} player-games, {len(sacks)} team-games with sacks")
    if sacks.empty or sacks["sacks_made"].sum() == 0:
        raise SystemExit("No defensive SACKS rows found in the CFBD box scores.")
    starts = games.drop_duplicates("id").set_index("id")["startDate"]
    usage = add_usage_features(wide)
    usage, league = add_sack_features(usage, sacks)
    f = build_rolling_player_features(attach_game_environment(usage, lines)).reset_index(drop=True)
    f["start"] = f["gameId"].map(starts)
    f[STAT] = pd.to_numeric(f[STAT], errors="coerce").fillna(0.0)
    f["zero_row"] = False
    return f, wide, league


def train(frame, before, cols):
    m = PlayerStatModel(STAT)
    m.feature_columns = list(cols)
    m.fit(frame[frame["season"] < before], verbose=False)
    return m


def terciles(s):
    q = s.quantile([1 / 3, 2 / 3]).values
    return pd.cut(s, [-np.inf, q[0], q[1], np.inf], labels=["light rush", "average rush", "heavy rush"]), [round(float(x), 2) for x in q]


def main(years, season):
    t0 = time.time()
    feats, wide, league = frames(years, season)
    rd = run_defense(wide)
    seasons = [s for s in (2024, 2025, season) if os.path.exists(f"{rl.HIST_DIR}/rush_yds_{s}.csv")]
    versions = {"current": ROLLING_FEATURE_COLUMNS, "sacks": ROLLING_FEATURE_COLUMNS + SACK_COLS}
    rows, notes, importance = [], {}, {}
    for s in seasons:
        raw = pd.read_csv(f"{rl.HIST_DIR}/rush_yds_{s}.csv").dropna(subset=["event_id", "player", "line", "commence_time"])
        notes[s] = {}
        for name, cols in versions.items():
            model = train(feats, s, cols)
            if name == "sacks":
                imp = dict(zip(model.feature_columns, model.model.feature_importances_))
                importance[s] = {c: round(float(imp.get(c, 0)), 4) for c in SACK_COLS}
                importance[s]["rank_of_" + SACK_COLS[0]] = int(sorted(imp.values(), reverse=True).index(imp[SACK_COLS[0]]) + 1)
                importance[s]["rank_of_" + SACK_COLS[1]] = int(sorted(imp.values(), reverse=True).index(imp[SACK_COLS[1]]) + 1)
                importance[s]["n_inputs"] = len(imp)
            r, skipped = rl.season_rows(name, model, feats, raw, s)
            rows += r
            notes[s][name] = skipped
            print(f"  {s} {name}: {len(r)} unders priced, skipped {skipped}")
    det = pd.DataFrame(rows)
    det["week"] = pd.to_numeric(det["week"], errors="coerce")
    det = det.merge(rd, on=["season", "week", "opponent"], how="left")
    det["run_def"] = det["run_def_tough"].map(matchup_bucket)
    det["run_def_tough"] = det["run_def_tough"].round(0)
    extra = feats[["gameId", "player", "roll_pass_att"] + SACK_COLS].drop_duplicates(["gameId", "player"])
    det = det.merge(extra, on=["gameId", "player"], how="left")
    det["pos"] = np.where(det["roll_pass_att"].fillna(0) >= QB_MIN_PASS_ATT, "QB", "non-QB")
    det["opp_rush"], cuts = terciles(det["opp_def_sacks_prior"])
    for c in SACK_COLS + ["roll_pass_att"]:
        det[c] = det[c].round(2)

    bets = det[det["ev"] >= 0]
    rec = rl.rec
    k = ["season", "event_id", "player"]

    def block(g):
        out = {"all": rec(g), "by_season": {int(s): rec(x) for s, x in g.groupby("season")}}
        for p, x in g.groupby("pos"):
            out[p] = rec(x)
            out[p + " vs tough run D"] = rec(x[x["run_def"] == "tough"])
            out[p + " vs other run D"] = rec(x[x["run_def"] != "tough"])
            out[p + " vs tough run D by season"] = {int(s): rec(y) for s, y in x[x["run_def"] == "tough"].groupby("season")}
        return out

    results = {v: block(g) for v, g in bets.groupby("version")}
    both = set(map(tuple, bets[bets["version"] == "current"][k].values)) & set(map(tuple, bets[bets["version"] == "sacks"][k].values))
    results["common_lines_bet_by_both"] = {v: block(g[[tuple(x) in both for x in g[k].values]]) for v, g in bets.groupby("version")}
    # Same QB player-games, did the sack version flip any current QB plays?
    cq = bets[(bets["version"] == "current") & (bets["pos"] == "QB")]
    sq = bets[(bets["version"] == "sacks") & (bets["pos"] == "QB")]
    ck, sk = set(map(tuple, cq[k].values)), set(map(tuple, sq[k].values))
    results["qb_only_current"] = rec(cq[[tuple(x) not in sk for x in cq[k].values]])
    results["qb_only_sacks"] = rec(sq[[tuple(x) not in ck for x in sq[k].values]])
    # Pass rush as a filter on today's model.
    filt = {}
    for label, x in cq.groupby("opp_rush", observed=True):
        filt[str(label)] = rec(x)
        filt[str(label) + " + tough run D"] = rec(x[x["run_def"] == "tough"])
    results["current_qb_by_opp_pass_rush"] = filt

    out = {"generated_at": pd.Timestamp.utcnow().isoformat(), "seasons": seasons, "sacks_per_team_game": league,
           "opp_pass_rush_tercile_cuts_sacks_per_game": cuts, "lines": notes, "sack_input_importance": importance,
           "results": results, "qb_rule": f"roll_pass_att >= {QB_MIN_PASS_ATT}",
           "rule": "rush yards under, best-EV book, edge 0%+, 2+ games played; graded at that book's line and price"}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    det.to_csv(OUT_CSV, index=False)
    print(json.dumps(out, indent=1, default=str))
    print(f"\nSaved {OUT_JSON} and {OUT_CSV} ({len(det)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(a.years, a.season)
