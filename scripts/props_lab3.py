#!/usr/bin/env python3
"""
Props lab 3 (added 10/2026): does the MATCHUP improve the receptions model?

The live receptions model knows a lot about the player (catches, routes,
targets, slot rate) and almost nothing about who is covering him: its only
defensive inputs are the opponent's total pass and rush yards allowed. This
lab adds the matchup, measures whether it helps, and prices the result
against the real posted lines. Same rules as the first two labs: everything
is graded forward in time, and nothing here touches the live model, the
dashboard or the official rules.

What is measured about each defense (always from its EARLIER games only)
  Catches and targets it has allowed to wide receivers, to tight ends and to
  running backs, and to players lined up in the slot versus out wide. Each
  is stated against what those same offenses usually produce in their other
  games, so 1.00 is an average defense, 1.15 has allowed 15% more than its
  opponents normally get, and a defense is not flattered by a soft schedule.
  With few games played the number is pulled toward 1.00.
  Plus PFF coverage grades for its corners, safeties and linebackers, as
  points above or below the national average for that position.

What is measured about each player (already in the live model)
  Routes run, targets, target share, slot rate. Added here: which position
  group he is, and how his snaps split between slot, wide and inline.

Versions compared (receptions)
  pff            the live model
  matchup_core   live inputs + the defense's number for THIS player's
                 position, for his slot/wide mix, and the coverage grade of
                 the players most likely to cover him
  matchup_full   + every defensive number above, and two interaction inputs
  formula        no retraining: the live projection multiplied by
                   position^a x alignment^b x coverage adjustment
                 where a, b and the coverage weight are three numbers fitted
                 on an earlier season. The plain-formula version.

How each is graded
  2025  Train on 2022-2024, project 2025. Projection error, hit-chance
        honesty, and the live model's miss in tough / average / soft
        matchups (is there anything for the matchup to fix?).
  2026  Train on 2022-2025. Every saved receptions line scored on both
        sides at its real prices: the official rule, by edge size, and by
        matchup.

Outputs (model results only -- no PFF numbers; the repo is public)
  docs/data/props_lab3.json
  docs/data/props_lab3_detail.csv

Usage (full git history, CFBD and PFF keys; run from the workflow):
  python scripts/props_lab3.py --years 2022 2023 2024 2025 --season 2026
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
from scipy.optimize import minimize

import props_lab as lab
import props_lab2 as lab2
from src.data import pff_client as pff
from src.features import pff_features as pf

STAT = "receptions"
OUT_JSON = "docs/data/props_lab3.json"
OUT_CSV = "docs/data/props_lab3_detail.csv"
SHRINK_GAMES = 3                    # a defense's number is pulled toward 1.00 as if it had 3 average games behind it
MIN_COVERAGE_SNAPS = 40             # coverage snaps a position group needs before its grade is used
TOUGH, SOFT = 0.93, 1.07            # matchup buckets on the position number
KEYS = ["season", "week", "franchise_id"]
SPLITS = ["all", "wr", "te", "rb", "slot", "wide"]
OG_METRICS = [f"{m}_{s}" for m in ("rec", "tgt") for s in SPLITS]
COV_GROUPS = ["cb", "s", "lb"]

BASE, PFFC = lab2.BASE, lab2.PFFC
DEF_COLS = ["d_games"] + [f"d_{m}" for m in OG_METRICS] + [f"d_cov_{g}" for g in COV_GROUPS]
CORE_COLS = ["d_games", "m_pos_rec", "m_pos_tgt", "m_align_rec", "m_cov"]
FULL_COLS = DEF_COLS + ["m_pos_rec", "m_pos_tgt", "m_align_rec", "m_cov", "x_match_rec", "x_match_tgt"]
F_COLS = ["f_pos", "f_align", "f_cov"]                  # the formula's three numbers (1, 1, 0 = no adjustment)
VARIANTS = {
    "pff":          (BASE + PFFC, "Live model: current inputs + PFF routes and targets"),
    "matchup_core": (BASE + PFFC + CORE_COLS, "Live inputs + the defense vs this player's position, alignment and coverage"),
    "matchup_full": (BASE + PFFC + FULL_COLS, "Live inputs + every defensive number and two interaction inputs"),
}
LIVE, FORMULA = "pff", "formula"


# ---------------------------------------------------- the defense side ----
def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def pos_group(position) -> pd.Series:
    p = position.astype(str).str.upper().str.strip()
    return pd.Series(np.select([p.str.startswith("WR"), p.str.startswith("TE"), p.isin(["HB", "RB", "FB", "TB"])],
                               ["wr", "te", "rb"], default="other"), index=position.index)


def alignment_shares(df):
    """Share of a player's lined-up snaps in the slot, out wide and inline.
    From snap counts; from PFF's rates when the counts are not there."""
    slot, wide, inline = _num(df, "slot_snaps"), _num(df, "wide_snaps"), _num(df, "inline_snaps")
    if slot.notna().sum() == 0 and wide.notna().sum() == 0:
        slot, wide, inline = _num(df, "slot_rate"), _num(df, "wide_rate"), _num(df, "inline_rate")
    slot, wide, inline = slot.fillna(0.0).clip(lower=0), wide.fillna(0.0).clip(lower=0), inline.fillna(0.0).clip(lower=0)
    total = slot + wide + inline
    return slot, wide, inline, total


def opponents(pgames) -> pd.DataFrame:
    """(season, week, franchise_id) -> the franchise it played that week."""
    cols = KEYS + ["opp"]
    if pgames is None or pgames.empty or "home_franchise_id" not in pgames.columns:
        return pd.DataFrame(columns=cols)
    g = pgames.dropna(subset=["home_franchise_id", "away_franchise_id"])
    long = pd.concat([
        g[["season", "week", "home_franchise_id", "away_franchise_id"]].set_axis(cols, axis=1),
        g[["season", "week", "away_franchise_id", "home_franchise_id"]].set_axis(cols, axis=1),
    ], ignore_index=True)
    return long.drop_duplicates(KEYS)


def offense_games(recv, pgames) -> pd.DataFrame:
    """One row per offense per game: catches and targets by position group
    and by alignment, and the defense they came against."""
    df = recv.copy()
    df["rec"], df["tgt"] = _num(df, "receptions").fillna(0.0), _num(df, "targets").fillna(0.0)
    df["g"] = pos_group(df["position"]) if "position" in df.columns else "other"
    slot, wide, _, total = alignment_shares(df)
    lined_up = df["g"].isin(["wr", "te"]) & (total > 0)
    df["s_slot"] = np.where(lined_up, slot / total.replace(0, np.nan), 0.0)
    df["s_wide"] = np.where(lined_up, wide / total.replace(0, np.nan), 0.0)
    for m in ("rec", "tgt"):
        df[f"{m}_all"] = np.where(df["g"] != "other", df[m], 0.0)
        for g in ("wr", "te", "rb"):
            df[f"{m}_{g}"] = np.where(df["g"] == g, df[m], 0.0)
        df[f"{m}_slot"], df[f"{m}_wide"] = df[m] * df["s_slot"], df[m] * df["s_wide"]
    og = df.groupby(KEYS)[OG_METRICS].sum().reset_index()
    og = og.merge(opponents(pgames), on=KEYS, how="left")
    return og


def defense_numbers(og) -> pd.DataFrame:
    """For every defense going into every week: what it has allowed so far,
    against what those offenses usually produce in their other games.
    Columns d_games and d_<metric> (1.00 = average). Uses earlier weeks only,
    for the defense and for the offenses it is compared against."""
    out = []
    og = og.dropna(subset=["opp"])
    for season, sg in og.groupby("season"):
        weeks = sorted(sg["week"].unique())
        for w in weeks[1:] + [weeks[-1] + 1]:           # the last one is "the next game", for live use
            prior = sg[sg["week"] < w]
            if prior.empty:
                continue
            off = prior.groupby("franchise_id")[OG_METRICS].agg(["sum", "count"])
            league = prior[OG_METRICS].mean()
            exp = pd.DataFrame(index=prior.index)
            for m in OG_METRICS:
                s, n = prior["franchise_id"].map(off[(m, "sum")]), prior["franchise_id"].map(off[(m, "count")])
                # what this offense averages in its OTHER games so far; the
                # national average when this is the only game it has played
                exp[m] = np.where(n > 1, (s - prior[m]) / (n - 1).replace(0, np.nan), league[m])
            d = prior.groupby("opp")[OG_METRICS].sum()
            e = exp.groupby(prior["opp"]).sum()
            n = prior.groupby("opp").size()
            res = pd.DataFrame({"season": season, "week": w, "franchise_id": d.index, "d_games": n.reindex(d.index).values})
            for m in OG_METRICS:
                ratio = (d[m] / e[m].replace(0, np.nan)).values
                res[f"d_{m}"] = (res["d_games"] * ratio + SHRINK_GAMES) / (res["d_games"] + SHRINK_GAMES)
            out.append(res)
    cols = KEYS + ["d_games"] + [f"d_{m}" for m in OG_METRICS]
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


def fetch_coverage(seasons) -> pd.DataFrame:
    """PFF's coverage report for every week: one row per defender-week. An
    empty frame if the report cannot be read -- the lab then runs without
    coverage grades and says so."""
    parts, fails = [], 0
    for season in seasons:
        for week in pff.MODEL_WEEKS:
            try:
                rows = pff._rows(pff._get("/v1/facet/defense/coverage", {"league": pff.LEAGUE, "season": season, "week": week}))
            except pff.PFFError as e:
                if e.status in (401, 403, 404):
                    print(f"  [warn] PFF coverage report not available: {e}")
                    return pd.DataFrame()
                fails += 1
                if fails >= 8:
                    print(f"  [warn] PFF coverage report failing repeatedly ({e}); continuing without the rest")
                    break
                continue
            if rows:
                parts.append(pd.DataFrame(rows).assign(season=season, week=week))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def cov_group(position) -> pd.Series:
    p = position.astype(str).str.upper().str.strip()
    return pd.Series(np.select([p.str.contains("CB"), p.isin(["S", "FS", "SS", "DB"]), p.str.contains("LB")],
                               ["cb", "s", "lb"], default="other"), index=position.index)


def coverage_numbers(cov) -> pd.DataFrame:
    """For every defense going into every week: snap-weighted PFF coverage
    grade of its corners, safeties and linebackers in earlier games, as
    points above (+) or below (-) the national average for that position."""
    cols = KEYS + [f"d_cov_{g}" for g in COV_GROUPS]
    need = {"franchise_id", "position", "grades_coverage_defense", "snap_counts_coverage"}
    if cov is None or cov.empty or not need.issubset(cov.columns):
        return pd.DataFrame(columns=cols)
    df = cov.copy()
    df["g"] = cov_group(df["position"])
    df["snaps"] = _num(df, "snap_counts_coverage").fillna(0.0)
    df["grade"] = _num(df, "grades_coverage_defense")
    df = df[(df["g"] != "other") & (df["snaps"] > 0) & df["grade"].notna()]
    df["gs"] = df["grade"] * df["snaps"]
    out = []
    for season, sg in df.groupby("season"):
        weeks = sorted(sg["week"].unique())
        for w in weeks[1:] + [weeks[-1] + 1]:
            prior = sg[sg["week"] < w]
            if prior.empty:
                continue
            t = prior.groupby(["franchise_id", "g"])[["gs", "snaps"]].sum()
            lg = prior.groupby("g")[["gs", "snaps"]].sum()
            lg_avg = lg["gs"] / lg["snaps"]
            t["rel"] = np.where(t["snaps"] >= MIN_COVERAGE_SNAPS, (t["gs"] / t["snaps"]).values - np.asarray(t.index.get_level_values("g").map(lg_avg), dtype=float), np.nan)
            wide = t["rel"].unstack("g").reindex(columns=COV_GROUPS)
            wide.columns = [f"d_cov_{g}" for g in COV_GROUPS]
            out.append(wide.reset_index().assign(season=season, week=w))
    return pd.concat(out, ignore_index=True)[cols] if out else pd.DataFrame(columns=cols)


# ----------------------------------------------------- the player side ----
def link_players(feats, games, recv, pgames):
    """Which PFF row is each CFBD player-game? Returns one row per matched
    player-game: gameId, athleteId, plus PFF's season, week, franchise,
    position group and alignment snaps. Uses the live matching code as is,
    by sending a row number through it in place of one of its values."""
    recv = recv.reset_index(drop=True)
    prep = pf.prepare_pff_rows(recv, pgames).reset_index(drop=True)
    carry = prep.assign(pff_adot=np.arange(len(prep), dtype=float))
    m, _ = pf.match_pff_to_cfbd(feats, games, carry)
    if m.empty:
        return pd.DataFrame(columns=["gameId", "athleteId"] + KEYS + ["g", "slot", "wide", "inline"])
    rows = prep.iloc[m["pff_adot"].astype(int).values][KEYS + ["player_id"]].reset_index(drop=True)
    link = pd.concat([m[["gameId", "athleteId"]].reset_index(drop=True), rows], axis=1)
    raw = recv.copy()
    raw["g"] = pos_group(raw["position"]) if "position" in raw.columns else "other"
    raw["slot"], raw["wide"], raw["inline"], _ = alignment_shares(raw)
    raw = raw.drop_duplicates(KEYS + ["player_id"])[KEYS + ["player_id", "g", "slot", "wide", "inline"]]
    return link.merge(raw, on=KEYS + ["player_id"], how="left")


def add_matchup_features(feats, link, dnum, cnum, pgames):
    """Adds DEF_COLS, the m_* / x_* model inputs and the f_* formula numbers
    to the player-game frame. Returns (frame, info)."""
    df = feats.copy()
    # Each CFBD team-game -> PFF (season, week, franchise): what most of its matched players say.
    tg = df[["_rid", "gameId", "athleteId", "team", "opponent", "season", "week"]].rename(columns={"week": "week_cfbd"})
    tg = tg.merge(link.drop(columns=["season", "player_id"], errors="ignore"), on=["gameId", "athleteId"], how="left")
    votes = tg.dropna(subset=["franchise_id"]).groupby(["gameId", "team", "week", "franchise_id"]).size().rename("v").reset_index()
    team_map = votes.sort_values("v", ascending=False).drop_duplicates(["gameId", "team"])[["gameId", "team", "week", "franchise_id"]]
    tg = tg.drop(columns=["week", "franchise_id"]).merge(
        team_map.rename(columns={"week": "pff_week", "franchise_id": "own_fr"}), on=["gameId", "team"], how="left")
    # The defense: the other team in the same game, or PFF's own schedule.
    tg = tg.merge(team_map.rename(columns={"team": "opponent", "franchise_id": "opp_fr"})[["gameId", "opponent", "opp_fr"]],
                  on=["gameId", "opponent"], how="left")
    sched = opponents(pgames).rename(columns={"week": "pff_week", "franchise_id": "own_fr", "opp": "opp_sched"})
    tg = tg.merge(sched, on=["season", "pff_week", "own_fr"], how="left")
    tg["opp_fr"] = tg["opp_fr"].fillna(tg["opp_sched"])

    d = dnum.rename(columns={"week": "pff_week", "franchise_id": "opp_fr"})
    c = cnum.rename(columns={"week": "pff_week", "franchise_id": "opp_fr"})
    tg = tg.merge(d, on=["season", "pff_week", "opp_fr"], how="left").merge(c, on=["season", "pff_week", "opp_fr"], how="left")
    for col in list(d.columns[3:]) + list(c.columns[3:]):      # an empty table merges in as text; these are numbers
        tg[col] = pd.to_numeric(tg[col], errors="coerce").astype(float)

    # The player: position group (his usual one that season) and his slot /
    # wide / inline split in EARLIER games.
    known = tg[tg["g"].isin(["wr", "te", "rb"])]
    usual = known.groupby(["athleteId", "season"])["g"].agg(lambda s: s.value_counts().index[0]).rename("pos").reset_index()
    tg = tg.merge(usual, on=["athleteId", "season"], how="left")
    tg = tg.sort_values(["athleteId", "season", "week_cfbd"])
    grp = tg.groupby(["athleteId", "season"])
    cum = {k: grp[k].transform(lambda s: s.fillna(0.0).shift(1).expanding().sum()).fillna(0.0) for k in ("slot", "wide", "inline")}
    lined = cum["slot"] + cum["wide"] + cum["inline"]
    sh = {k: np.where(lined > 0, cum[k] / lined.replace(0, np.nan), np.nan) for k in cum}

    pos, has_def = tg["pos"], tg["d_games"].notna()
    pick = lambda prefix: np.select([pos == g for g in ("wr", "te", "rb")], [tg[f"{prefix}_{g}"] for g in ("wr", "te", "rb")], default=np.nan)
    m_pos_rec, m_pos_tgt = pick("d_rec"), pick("d_tgt")
    # Alignment: his own slot / wide / inline mix against what the defense
    # allows in each spot (inline uses its tight-end number). Running backs,
    # and anyone with no alignment history yet, use the position number.
    mix = sh["slot"] * tg["d_rec_slot"] + sh["wide"] * tg["d_rec_wide"] + sh["inline"] * tg["d_rec_te"]
    m_align = np.where(pos.isin(["wr", "te"]) & (lined > 0) & pd.notna(mix), mix, m_pos_rec)
    # Coverage: corners for receivers, safeties and linebackers for tight
    # ends, linebackers for backs.
    te_cov = tg[["d_cov_s", "d_cov_lb"]].mean(axis=1, skipna=True)
    m_cov = np.select([pos == "wr", pos == "te", pos == "rb"], [tg["d_cov_cb"], te_cov, tg["d_cov_lb"]], default=np.nan)

    tg["m_pos_rec"], tg["m_pos_tgt"], tg["m_align_rec"], tg["m_cov_raw"] = m_pos_rec, m_pos_tgt, m_align, m_cov
    tg = tg.sort_values("_rid").reset_index(drop=True)
    df = df.sort_values("_rid").reset_index(drop=True)
    assert (tg["_rid"].values == df["_rid"].values).all()

    for col in ["d_games"] + [f"d_{m}" for m in OG_METRICS]:
        df[col] = tg[col].fillna(pf.MISSING).values
    for g in COV_GROUPS:                                 # a grade can be negative, so "unknown" is the average (0)
        df[f"d_cov_{g}"] = tg[f"d_cov_{g}"].fillna(0.0).values
    for col in ("m_pos_rec", "m_pos_tgt", "m_align_rec"):
        df[col] = tg[col].fillna(pf.MISSING).values
    df["m_cov"] = tg["m_cov_raw"].fillna(0.0).values
    ok = tg["m_pos_rec"].notna().values
    df["x_match_rec"] = np.where(ok, df["roll_receptions"] * tg["m_pos_rec"].values, pf.MISSING)
    tgt_ok = ok & (df["pff_matched"] == 1).values
    df["x_match_tgt"] = np.where(tgt_ok, df["roll_pff_targets"] * tg["m_pos_tgt"].values, pf.MISSING)
    df["f_pos"] = tg["m_pos_rec"].fillna(1.0).clip(0.4, 2.5).values
    df["f_align"] = tg["m_align_rec"].fillna(1.0).clip(0.4, 2.5).values
    df["f_cov"] = tg["m_cov_raw"].fillna(0.0).clip(-30, 30).values
    df["has_matchup"] = ok.astype(int)

    info = {
        "team_games_linked_to_pff_pct": round(float(tg["own_fr"].notna().mean()) * 100, 1),
        "player_games_with_a_defense_number_pct": round(float(has_def.mean()) * 100, 1),
        "player_games_with_a_position_matchup_pct": round(float(ok.mean()) * 100, 1),
        "player_games_with_a_coverage_grade_pct": round(float(tg["m_cov_raw"].notna().mean()) * 100, 1),
        "position_groups": {k: int(v) for k, v in tg.drop_duplicates(["athleteId", "season"])["pos"].value_counts().items()},
    }
    return df, info


# ------------------------------------------------------------- formula ----
def multiplier(X, k):
    a, b, c = k
    return np.power(X["f_pos"].values, a) * np.power(X["f_align"].values, b) * np.exp(-c * X["f_cov"].values / 10.0)


class FormulaModel:
    """The live projection times the matchup multiplier. Looks like a fitted
    model to the lab code (it only needs .predict)."""
    def __init__(self, base, base_cols, k):
        self.base, self.base_cols, self.k = base, list(base_cols), tuple(float(x) for x in k)

    def predict(self, X):
        return np.clip(self.base.predict(X[self.base_cols]), 0, None) * multiplier(X, self.k)


def fit_formula(pred, actual, X):
    """The three strengths that make projection x multiplier closest to what
    happened (least squares). 0 means that term adds nothing."""
    pred, actual = np.asarray(pred, float), np.asarray(actual, float)
    loss = lambda k: float(np.mean((pred * multiplier(X, k) - actual) ** 2))
    res = minimize(loss, x0=[0.3, 0.2, 0.05], method="L-BFGS-B", bounds=[(0, 1.5), (0, 1.5), (0, 0.5)])
    k = tuple(round(float(x), 3) for x in res.x)
    return k, {"position": k[0], "alignment": k[1], "coverage_per_10_grade_points": k[2],
               "squared_error_before": round(loss((0, 0, 0)), 4), "squared_error_after": round(loss(k), 4)}


# --------------------------------------------------------- diagnostics ----
def bucket(f_pos, has):
    return np.where(~np.asarray(has, bool), "unknown", np.where(f_pos < TOUGH, "tough", np.where(f_pos > SOFT, "soft", "average")))


def by_matchup(pred, actual, f_pos, has):
    """Projection miss in tough / average / soft matchups."""
    b, err = bucket(f_pos, has), np.asarray(pred, float) - np.asarray(actual, float)
    return {k: {"n": int((b == k).sum()), "bias": round(float(err[b == k].mean()), 3), "mae": round(float(np.abs(err[b == k]).mean()), 3)}
            for k in ("tough", "average", "soft", "unknown") if (b == k).sum()}


def signal(pred, actual, X):
    """Is there anything for the matchup to fix? How the live model's miss
    (actual minus projection) moves with each matchup number, among players
    who have one."""
    m = X["has_matchup"].values == 1
    if m.sum() < 200:
        return {"n": int(m.sum())}
    miss = (np.asarray(actual, float) - np.asarray(pred, float))[m]
    out = {"n": int(m.sum())}
    for name, v in (("position", np.log(X["f_pos"].values[m])), ("alignment", np.log(X["f_align"].values[m])), ("coverage_grade", X["f_cov"].values[m])):
        if np.std(v) == 0:
            continue
        slope, corr = float(np.polyfit(v, miss, 1)[0]), float(np.corrcoef(v, miss)[0, 1])
        # stated per 10% for the two ratios, per 10 grade points for coverage
        out[name] = {"correlation": round(corr, 4), "catches_per_step": round(float(slope * (np.log(1.1) if name != "coverage_grade" else 10.0)), 3)}
    return out


def props_by_matchup(rows, idx_of, feats):
    """The official-shape overs (10%+ edge, projection above the line), split
    by matchup."""
    d = pd.DataFrame(rows)
    if d.empty:
        return {}
    idx = [idx_of.get((r.player, r.market, r.weekend)) for r in d.itertuples()]
    ok = [i is not None for i in idx]
    d = d[ok].copy()
    sel = feats.loc[[i for i in idx if i is not None]]
    d["matchup"] = bucket(sel["f_pos"].values, sel["has_matchup"].values == 1)
    overs = d[d.over_ev.notna()]
    rule = overs[(overs.over_ev >= lab2.MIN_EDGE) & (overs.pred > overs.over_line)]
    unders = d[d.under_ev.notna()]
    under_rule = unders[unders.under_ev >= lab2.MIN_EDGE]
    return {"rule_overs": {k: lab2.rec_line(g) for k, g in rule.groupby("matchup")},
            "all_overs": {k: lab2.rec_line(g) for k, g in overs.groupby("matchup")},
            "unders_10_plus": {k: lab2.rec_line(g, "under") for k, g in under_rule.groupby("matchup")}}


# ----------------------------------------------------------------- main ----
def build(years, season):
    feats, info, _ = lab2.build_frame(years, season)
    print("Building the matchup numbers...")
    out_dir = os.environ.get("PFF_DATA_DIR") or os.path.join(tempfile.gettempdir(), "pff_lab2")
    seasons = list(years) + [season]
    recv, pgames = pff.fetch_receiving_seasons(seasons, out_dir, verbose=False)      # already on disk from build_frame
    games = pd.concat([lab.load_history(years)[0], lab.load_current(season)[0]], ignore_index=True)
    games = games.assign(id=pd.to_numeric(games["id"], errors="coerce")).dropna(subset=["id"])
    games["id"] = games["id"].astype("int64")
    og = offense_games(recv, pgames)
    dnum = defense_numbers(og)
    cov = fetch_coverage(seasons)
    cnum = coverage_numbers(cov)
    link = link_players(feats, games, recv, pgames)
    feats, minfo = add_matchup_features(feats, link, dnum, cnum, pgames)
    minfo.update({
        "offense_games": int(len(og)), "offense_games_with_opponent_pct": round(float(og["opp"].notna().mean()) * 100, 1) if len(og) else 0.0,
        "defense_weeks": int(len(dnum)), "coverage_rows": int(len(cov)), "coverage_defense_weeks": int(len(cnum)),
        "coverage_grades": "used" if len(cnum) else "NOT AVAILABLE -- run without coverage grades",
    })
    print(f"  matchup: {minfo}")
    info["matchup"] = minfo
    if not minfo["player_games_with_a_position_matchup_pct"]:
        raise SystemExit("No matchup numbers could be built (PFF's schedule or receiving rows did not line up with the "
                         "box scores -- see the line above). Nothing was graded.")
    return feats, info


def main(years, season):
    t0 = time.time()
    feats, info = build(years, season)
    last, prev = max(years), max(years) - 1
    all_cols = sorted({c for cols, _ in VARIANTS.values() for c in cols} | set(F_COLS))
    missing = [c for c in all_cols if c not in feats.columns]
    if missing:
        raise SystemExit(f"Feature columns missing from the frame: {missing}")
    props = [p for p in lab.collect_props(feats, season) if p["stat"] == STAT]
    idx_of = {(p["player"], p["market"], p["weekend"]): p["idx"] for p in props}
    print(f"Receptions props matched to box scores: {len(props)}")

    stat = STAT
    ok = feats[all_cols + [stat]].notna().all(axis=1) & (feats["games_played_prior"] >= 1)
    pop = ok & (feats["roll_receptions"] > 0)
    usable = set(feats.index[ok])
    test = feats[ok & (feats.season == last) & (feats.games_played_prior >= 2) & (feats.roll_receptions >= lab2.ROLE_MIN_CATCHES)]
    test_line = np.floor(test[f"roll_{stat}"]) + 0.5
    tr0, ho0 = feats[pop & (feats.season < prev)], feats[pop & (feats.season == prev)]
    trA, hoA = feats[pop & (feats.season < last)], feats[pop & (feats.season == last)]
    trB = feats[pop & (feats.season <= last)]
    print(f"\n===== {stat}: {len(trB)} training rows, {len(test)} in the {last} role-player test =====")
    if min(len(tr0), len(ho0), len(trA), len(hoA)) < 200:
        raise SystemExit("Not enough rows to run.")

    results, detail, live = {}, [], {}

    def grade(name, desc, cols, mA, mB, p0, pA, extra=None):
        std_test, std_live = lab2.Pools(mA, cols, stat, p0, ho0[stat]), lab2.Pools(mB, cols, stat, pA, hoA[stat])
        r25 = lab2.test_2025(mA, cols, std_test, test, stat, test_line, False)
        pt = lab.predict(mA, test, cols)
        r25["by_matchup"] = by_matchup(pt, test[stat].values, test["f_pos"].values, test["has_matchup"].values == 1)
        rows = lab2.evaluate_props(name, mB, cols, std_live, stat, feats, props, usable)
        detail.extend(rows)
        p26 = lab2.summarize(pd.DataFrame(rows)) if rows else {"n": 0}
        p26["by_matchup"] = props_by_matchup(rows, idx_of, feats)
        results[name] = dict({"description": desc, "n_inputs": len(cols), "test_2025": r25, "props_2026": p26}, **(extra or {}))
        c = r25["calibration"]
        print(f"  {name:<13} {last}: error {r25['mae']:.3f} rmse {r25['rmse']:.3f} brier {c.get('brier')} | {len(rows)} props")
        return pt

    for name, (cols, desc) in VARIANTS.items():
        g0, gA, gB = (lab.fit_predict(x, cols, stat) for x in (tr0, trA, trB))
        p0, pA = lab.predict(g0, ho0, cols), lab.predict(gA, hoA, cols)
        pt = grade(name, desc, cols, gA, gB, p0, pA)
        if name == LIVE:
            live = {"g0": g0, "gA": gA, "gB": gB, "p0": p0, "pA": pA, "cols": cols}
            results[name]["matchup_signal_2025"] = signal(pt, test[stat].values, test)

    # The plain formula: live projection x matchup multiplier. Strengths are
    # fitted on the season BEFORE the one being graded, both times.
    fcols = live["cols"] + F_COLS
    k0, fit0 = fit_formula(live["p0"], ho0[stat].values, ho0)           # fitted on prev, graded on last
    kA, fitA = fit_formula(live["pA"], hoA[stat].values, hoA)           # fitted on last, graded on this season
    mA, mB = FormulaModel(live["gA"], live["cols"], k0), FormulaModel(live["gB"], live["cols"], kA)
    p0f = FormulaModel(live["g0"], live["cols"], k0).predict(ho0)
    pAf = FormulaModel(live["gA"], live["cols"], kA).predict(hoA)
    grade(FORMULA, "Live projection x position^a x alignment^b x coverage adjustment", fcols, mA, mB, p0f, pAf,
          {"strengths_for_2025_test": fit0, "strengths_for_2026_lines": fitA})
    print(f"  formula strengths (position, alignment, coverage): {last} test {k0}, {season} lines {kA}")

    report(results, last, season)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "train_years": list(years), "season": season,
                   "stat": stat, "buckets": {"tough_below": TOUGH, "soft_above": SOFT}, "pff": info, "results": results}, f, indent=2, default=str)
    det = pd.DataFrame(detail)
    if not det.empty:
        det.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_JSON} and {OUT_CSV} ({len(det)} rows) in {time.time() - t0:.0f}s")


def report(results, last, season):
    def r(x):
        return f"{x.get('record', '-'):>7} {x.get('roi', 0):+6.1f}% n={x.get('n', 0):<3}" if x and x.get("n") else f"{'-':>20}"
    print(f"\n################ receptions ################")
    print(f"{'version':<13} | {str(last) + ' err':>8} {'brier':>7} | {str(season) + ' err':>8} {'brier':>7} | {'rule (overs 10%+)':>22} | {'under 30% edge':>22}")
    for name, x in results.items():
        t, p = x.get("test_2025") or {}, x.get("props_2026") or {}
        print(f"{name:<13} | {t.get('mae', float('nan')):>8.3f} {(t.get('calibration') or {}).get('brier', float('nan')):>7} | "
              f"{p.get('mae', float('nan')):>8.3f} {(p.get('calibration') or {}).get('brier', float('nan')):>7} | "
              f"{r(p.get('rule')):>22} | {r(p.get('rule_under_30_edge')):>22}")
    live = results.get(LIVE) or {}
    print(f"\nLive model's miss by matchup, {last} (bias = projection minus actual):")
    for k, v in ((live.get("test_2025") or {}).get("by_matchup") or {}).items():
        print(f"  {k:<8} n={v['n']:<6} bias {v['bias']:+.3f}  error {v['mae']:.3f}")
    print(f"Matchup signal in the live model's misses: {live.get('matchup_signal_2025')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(sorted(a.years), a.season)
