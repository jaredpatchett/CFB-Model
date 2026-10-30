#!/usr/bin/env python3
"""
Matchup lab (added 10/30/2026): do matchup grades help the props that are bet?

Two questions:
  1. Which defenses give up the most touchdowns (and catches / yards) to each
     position -- QB, RB, WR, TE?
  2. Backtest: does a matchup grade built from that help the official props
     (or anytime TDs) once the model's edge is already known?

The grade (box scores only -- no PFF numbers, so it is safe for the public repo)
  For each defense, from its EARLIER games of the same season only:
    catches and receiving yards allowed to WRs / TEs / RBs
    rush yards allowed
    rushing + receiving TDs allowed to QBs / RBs / WRs / TEs
  each stated against what those same offenses produce in their OTHER prior
  games (1.00 = average, 1.20 = allows 20% more than usual), pulled toward
  1.00 with few games (3 games for catches/yards, 6 for TDs -- TDs are rarer).
  Ranked among FBS defenses that week into a toughness percentile (100 =
  toughest). Positions come from CFBD rosters.

  For a pick: support = toughness if it is an under, 100 - toughness if an
  over. Buckets fixed BEFORE looking at results:
    with     support >= 67   (under vs a tough D, over vs a soft D)
    neutral  33 < support < 67
    against  support <= 33
  Only defenses with 3+ prior games get a grade.

What is graded (all at the prices really posted, using the edge the model gave
at the time -- nothing is re-fitted)
  receptions      props_lab3_detail.csv (live "pff" version; 2024, 2025, Sept 2026)
  rec yds / rush  props_edge_finder_detail.csv ("first line, best book"; Sept 2026)
  live            data/clv/prop_flags.csv (graded props since Oct 7, 2026)
  anytime TD      td_lab_detail.csv (direct_cal model, DraftKings Yes price)

Pass rule decided in advance: a grade "helps" a group only if "with" beats
"against" on ROI in every season / sample that has 20+ bets in both buckets.

Outputs
  docs/data/matchup_lab.json
  docs/data/matchup_lab_report.txt
  docs/data/matchup_lab_detail.csv          one row per graded prop
  docs/data/td_allowed_by_position_2026.csv  this season's table (box scores)

Usage (CFBD key; run from the workflow):
  python scripts/matchup_lab.py --years 2022 2023 2024 2025 --season 2026
"""
import argparse
import io
import json
import os
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import props_lab as lab
from src.data import cfbd_client as cfbd
from src.features.live_player_features import _normalize_name
from src.features.player_features import pivot_player_game_stats

OUT_JSON = "docs/data/matchup_lab.json"
OUT_TXT = "docs/data/matchup_lab_report.txt"
OUT_CSV = "docs/data/matchup_lab_detail.csv"
OUT_TD = "docs/data/td_allowed_by_position_2026.csv"
REC_GROUPS = ("wr", "te", "rb")
TD_GROUPS = ("qb", "rb", "wr", "te")
SHRINK_YDS, SHRINK_TD = 3, 6
MIN_GAMES = 3
WITH, AGAINST = 67, 33
MIN_N = 20


# ------------------------------------------------------------- positions ----
def pos_group(p):
    p = str(p).upper().strip()
    if p.startswith("QB"):
        return "qb"
    if p in ("RB", "HB", "FB", "TB", "APB"):
        return "rb"
    if p.startswith("WR"):
        return "wr"
    if p.startswith("TE"):
        return "te"
    return None


def rosters(seasons):
    parts = []
    for s in seasons:
        try:
            r = cfbd.get_roster(int(s))
        except Exception as e:
            print(f"  [warn] {s} roster failed: {e}")
            continue
        if r.empty or "id" not in r.columns:
            continue
        parts.append(pd.DataFrame({"season": int(s), "athleteId": pd.to_numeric(r["id"], errors="coerce"),
                                   "pos": r.get("position", pd.Series(index=r.index)).map(pos_group)}))
    if not parts:
        return pd.DataFrame(columns=["season", "athleteId", "pos"])
    out = pd.concat(parts).dropna(subset=["athleteId"])
    out["athleteId"] = out["athleteId"].astype("int64")
    return out.drop_duplicates(["season", "athleteId"])


def usage_guess(r):
    if r["pass_att"] >= 5:
        return "qb"
    if r["rush_att"] >= max(3, r["receptions"]):
        return "rb"
    return "wr" if r["receptions"] > 0 else None


# ------------------------------------------------------------------ data ----
def load(years, season):
    print("Loading CFBD box scores...")
    g_h, s_h, _ = lab.load_history(years)
    g_n, s_n, _ = lab.load_current(season)
    games = pd.concat([g_h, g_n], ignore_index=True)
    games["id"] = pd.to_numeric(games["id"], errors="coerce")
    games = games.dropna(subset=["id"]).drop_duplicates("id")
    games["id"] = games["id"].astype("int64")
    parts = [pivot_player_game_stats(s_h, g_h)]
    if not s_n.empty:
        parts.append(pivot_player_game_stats(s_n, g_n))
    wide = pd.concat(parts, ignore_index=True)
    for c in ("athleteId", "gameId", "season", "week"):
        wide[c] = pd.to_numeric(wide[c], errors="coerce")
    wide = wide.dropna(subset=["athleteId", "gameId", "season", "week"])
    for c in ("athleteId", "gameId", "season", "week"):
        wide[c] = wide[c].astype("int64")

    seasons = sorted(wide["season"].unique())
    ros = rosters(seasons)
    wide = wide.merge(ros, on=["season", "athleteId"], how="left")
    from_roster = wide["pos"].notna().mean()
    # Same player, another season's roster.
    any_pos = ros.dropna(subset=["pos"]).sort_values("season").drop_duplicates("athleteId", keep="last").set_index("athleteId")["pos"]
    wide["pos"] = wide["pos"].fillna(wide["athleteId"].map(any_pos))
    guess = wide.apply(usage_guess, axis=1)
    wide["pos"] = wide["pos"].fillna(guess)
    info = {"player_games": int(len(wide)), "position_from_roster_pct": round(from_roster * 100, 1),
            "position_any_pct": round(wide["pos"].notna().mean() * 100, 1)}
    print(f"  {info}")

    # FBS pool.
    fbs = set()
    for side in ("home", "away"):
        c = f"{side}Classification"
        if c in games.columns:
            fbs |= set(games.loc[games[c].astype(str).str.lower() == "fbs", f"{side}Team"])
    info["fbs_teams"] = len(fbs)
    starts = games.set_index("id")["startDate"] if "startDate" in games.columns else pd.Series(dtype=str)
    return wide, games, fbs, starts, info


def offense_games(wide):
    """One row per offense per game: what it produced, by position."""
    w = wide.copy()
    w["td"] = w["rush_tds"] + w["rec_tds"]
    cols = {}
    for g in REC_GROUPS:
        m = w["pos"] == g
        w[f"c_{g}"] = np.where(m, w["receptions"], 0.0)
        w[f"y_{g}"] = np.where(m, w["rec_yds"], 0.0)
    for g in TD_GROUPS:
        w[f"td_{g}"] = np.where(w["pos"] == g, w["td"], 0.0)
    metrics = [f"c_{g}" for g in REC_GROUPS] + [f"y_{g}" for g in REC_GROUPS] + ["rush"] + [f"td_{g}" for g in TD_GROUPS]
    w["rush"] = w["rush_yds"]
    og = w.groupby(["season", "week", "gameId", "team", "opponent"])[metrics].sum().reset_index()
    return og, metrics


def ratios(prior, metrics):
    """Per defense: allowed vs what those offenses usually get, shrunk."""
    off = prior.groupby("team")[metrics].agg(["sum", "count"])
    league = prior[metrics].mean()
    exp = pd.DataFrame(index=prior.index)
    for m in metrics:
        s, n = prior["team"].map(off[(m, "sum")]), prior["team"].map(off[(m, "count")])
        exp[m] = np.where(n > 1, (s - prior[m]) / (n - 1).replace(0, np.nan), league[m])
    allowed = prior.groupby("opponent")[metrics].sum()
    expected = exp.groupby(prior["opponent"]).sum()
    games = prior.groupby("opponent").size()
    out = pd.DataFrame({"games": games})
    for m in metrics:
        k = SHRINK_TD if m.startswith("td_") else SHRINK_YDS
        r = allowed[m] / expected[m].replace(0, np.nan)
        out[m] = (out["games"] * r.fillna(1.0) + k) / (out["games"] + k)
    return out


def pct_tough(table, cols, weights=None):
    """Toughness percentile (100 = allows the least) from ratio columns."""
    weights = weights or [1.0] * len(cols)
    num = pd.Series(0.0, index=table.index)
    for c, w in zip(cols, weights):
        sd = table[c].std()
        z = (table[c] - table[c].mean()) / sd if sd and not np.isnan(sd) else 0.0
        num = num - z * w
    return num.rank(pct=True) * 100


def weekly_scores(og, metrics, fbs):
    """(season, week, defense) -> games, ratios, toughness percentiles,
    using only that season's games before that week."""
    rows = []
    for (s, wk) in sorted(set(zip(og["season"], og["week"])) | {(s, w + 1) for s, w in zip(og["season"], og["week"])}):
        prior = og[(og["season"] == s) & (og["week"] < wk)]
        if prior.empty:
            continue
        t = ratios(prior, metrics)
        if fbs:
            t = t[t.index.isin(fbs)]
        t = t[t["games"] >= MIN_GAMES]
        if len(t) < 20:
            continue
        sc = pd.DataFrame(index=t.index)
        for g in REC_GROUPS:
            sc[f"rec_{g}"] = pct_tough(t, [f"c_{g}", f"y_{g}"])
            sc[f"recyds_{g}"] = pct_tough(t, [f"y_{g}"])
        sc["rush"] = pct_tough(t, ["rush"])
        for g in TD_GROUPS:
            sc[f"td_{g}"] = pct_tough(t, [f"td_{g}"])
        sc = sc.join(t.add_suffix("_ratio").rename(columns={"games_ratio": "games"}))
        sc["season"], sc["week"] = s, wk
        rows.append(sc.rename_axis("defense").reset_index())
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------- joining ----
def game_index(og, starts):
    gi = og[["season", "week", "gameId", "team", "opponent"]].drop_duplicates(["gameId", "team"]).copy()
    gi["date"] = pd.to_datetime(gi["gameId"].map(starts), errors="coerce", utc=True).dt.tz_convert("America/Phoenix").dt.normalize().dt.tz_localize(None)
    return gi


def attach(props, gi, wide, scores):
    """props need season, date, team, opponent, player -> adds week, pos,
    defense scores."""
    p = props.copy()
    p["date"] = pd.to_datetime(p["date"], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    m = p.reset_index().merge(gi, on=["season", "team", "opponent"], how="inner", suffixes=("", "_g"))
    m["gap"] = (m["date"] - m["date_g"]).dt.days.abs()
    m = m[m["gap"] <= 4].sort_values("gap").drop_duplicates("index")
    # Player position: name within team-season.
    w = wide[["season", "team", "player", "pos"]].dropna(subset=["pos"]).copy()
    w["nk"] = w["player"].map(_normalize_name)
    pos = w.groupby(["season", "team", "nk"])["pos"].agg(lambda s: s.value_counts().index[0]).rename("pos_p").reset_index()
    m["nk"] = m["player"].map(_normalize_name)
    m = m.merge(pos, on=["season", "team", "nk"], how="left")
    m = m.merge(scores, left_on=["season", "week", "opponent"], right_on=["season", "week", "defense"], how="left")
    return m.set_index("index").sort_index()


def support(df, key_fn):
    tough = df.apply(key_fn, axis=1)
    s = np.where(df["side"] == "under", tough, 100 - tough)
    df = df.assign(tough=tough, support=s)
    df["bucket"] = np.select([df["support"] >= WITH, df["support"] <= AGAINST], ["with", "against"], "neutral")
    df.loc[df["support"].isna(), "bucket"] = "no grade"
    return df


def rec_key(r):
    g = r.get("pos_p")
    return r.get(f"rec_{g}") if g in REC_GROUPS else np.nan


def recyds_key(r):
    g = r.get("pos_p")
    return r.get(f"recyds_{g}") if g in REC_GROUPS else np.nan


def rush_key(r):
    return r.get("rush")


def td_key(r):
    g = r.get("pos_p")
    return r.get(f"td_{g}") if g in TD_GROUPS else np.nan


# --------------------------------------------------------------- grading ----
def rec(d):
    d = d[d["units"].notna()]
    n = len(d)
    if not n:
        return {"n": 0}
    w, l = int((d["units"] > 0).sum()), int((d["units"] < 0).sum())
    u = float(d["units"].sum())
    return {"n": n, "record": f"{w}-{l}" + (f"-{n - w - l}" if n - w - l else ""), "units": round(u, 1), "roi": round(u / n * 100, 1)}


def by_bucket(d, split="sample"):
    out = {}
    for s, g in d.groupby(split):
        out[str(s)] = {b: rec(g[g["bucket"] == b]) for b in ("with", "neutral", "against", "no grade")}
    out["all"] = {b: rec(d[d["bucket"] == b]) for b in ("with", "neutral", "against", "no grade")}
    return out


def verdict(tab):
    tests = []
    for s, t in tab.items():
        if s == "all":
            continue
        a, b = t["with"], t["against"]
        if a.get("n", 0) >= MIN_N and b.get("n", 0) >= MIN_N:
            tests.append((s, a["roi"] - b["roi"]))
    if not tests:
        return "not enough bets to test"
    ok = all(x > 0 for _, x in tests)
    txt = ", ".join(f"{s} {x:+.1f}" for s, x in tests)
    return ("PASSES" if ok else "fails") + f" (with minus against ROI: {txt})"


# ----------------------------------------------------------------- inputs ----
def receptions_rows():
    d = pd.read_csv("docs/data/props_lab3_detail.csv")
    d = d[d["version"] == "pff"]
    rows = []
    for side in ("over", "under"):
        x = d[["player", "team", "opponent", "weekend", "pred", "actual", "lines_season", f"{side}_ev", f"{side}_units", f"{side}_line"]].copy()
        x.columns = ["player", "team", "opponent", "date", "pred", "actual", "season", "ev", "units", "line"]
        x["side"] = side
        rows.append(x)
    x = pd.concat(rows, ignore_index=True)
    x["sample"] = x["season"].astype(int).astype(str)
    x["season"] = x["season"].astype(int)
    return x


def edge_finder_rows(stat):
    d = pd.read_csv("docs/data/props_edge_finder_detail.csv")
    d = d[(d["stat"] == stat) & (d["variant"] == "first line, best book")]
    x = d[["player", "team", "opponent", "weekend", "pred", "actual", "side", "ev", "units", "line"]].rename(columns={"weekend": "date"})
    x["season"], x["sample"] = 2026, "Sept 2026"
    return x


def live_rows():
    d = pd.read_csv("data/clv/prop_flags.csv")
    d = d[d["result"].isin(["W", "L", "P"]) & d["team"].notna() & d["opponent"].notna()]
    d = d.assign(ev=pd.to_numeric(d["model_ev"], errors="coerce") / 100.0, date=d["start_time"], season=2026, sample="live Oct 2026")
    return d[["key", "tag", "player", "team", "opponent", "market", "date", "side", "ev", "units", "actual", "line", "season", "sample"]]


def td_rows():
    d = pd.read_csv("docs/data/td_lab_detail.csv")
    d = d[(d["version"] == "direct_cal") & d["dk_yes"].notna() & (d["ghost"] != True)]  # noqa: E712
    x = d[["season", "weekend", "player", "team", "opponent", "role", "p", "dk_yes", "yes_dk_ev", "yes_dk_units", "scored"]].rename(
        columns={"weekend": "date", "yes_dk_ev": "ev", "yes_dk_units": "units"})
    x["side"], x["sample"] = "over", x["season"].astype(str)
    x["mkt"] = np.where(x["dk_yes"] > 0, 100 / (x["dk_yes"] + 100), -x["dk_yes"] / (-x["dk_yes"] + 100))
    return x


# ------------------------------------------------------------------- main ----
def main(years, season):
    wide, games, fbs, starts, info = load(years, season)
    og, metrics = offense_games(wide)
    print("Building week-by-week defense numbers...")
    scores = weekly_scores(og, metrics, fbs)
    print(f"  {len(scores)} defense-weeks graded")
    gi = game_index(og, starts)

    # This season's table.
    now = og[og["season"] == season]
    if not now.empty:
        t = ratios(now, metrics)
        if fbs:
            t = t[t.index.isin(fbs)]
        per = now.groupby("opponent")[[f"td_{g}" for g in TD_GROUPS] + [f"c_{g}" for g in REC_GROUPS] + [f"y_{g}" for g in REC_GROUPS] + ["rush"]].mean()
        tab = per.add_suffix("_per_game").join(t.add_suffix("_vs_usual")).loc[t.index]
        tab = tab.rename(columns={"games_vs_usual": "games"}).round(2).rename_axis("defense").reset_index()
        tab.to_csv(OUT_TD, index=False)

    results, details = {"info": info}, []

    def run(name, df, key_fn, mask=None, note=""):
        a = attach(df, gi, wide, scores)
        a = support(a, key_fn)
        if mask is not None:
            a = a[mask(a)]
        a = a[a["units"].notna()]
        tab = by_bucket(a)
        res = {"note": note, "matched_pct": round(a["support"].notna().mean() * 100, 1) if len(a) else 0,
               "by_bucket": tab, "verdict": verdict(tab)}
        if "pred" in a.columns and a["pred"].notna().any():
            a2 = a[a["support"].notna()]
            dt = np.select([a2["tough"] >= WITH, a2["tough"] <= AGAINST], ["tough D", "soft D"], "average D")
            res["model_miss_by_defense"] = {k: {"n": int(len(g)), "actual_minus_proj": round(float((g["actual"] - g["pred"]).mean()), 2)}
                                            for k, g in a2.assign(dt=dt).drop_duplicates(["player", "date"]).groupby("dt")}
        results[name] = res
        keep = [c for c in ("sample", "player", "team", "opponent", "date", "side", "line", "ev", "units", "pos_p", "tough", "support", "bucket") if c in a.columns]
        details.append(a[keep].assign(group=name))
        print(f"  {name}: {len(a)} bets, {res['verdict']}")
        return a

    print("Grading...")
    r = receptions_rows()
    run("receptions unders 30%+ (official)", r, rec_key, lambda a: (a["side"] == "under") & (a["ev"] >= 0.30))
    run("receptions overs 10-30% (watch)", r, rec_key, lambda a: (a["side"] == "over") & a["ev"].between(0.10, 0.30, inclusive="left"))
    run("receptions, every 0%+ edge", r, rec_key, lambda a: a["ev"] >= 0)
    ry = edge_finder_rows("rec_yds")
    run("rec yds overs 15%+ (official, Sept 2026)", ry, recyds_key, lambda a: (a["side"] == "over") & (a["ev"] >= 0.15))
    run("rec yds, every 0%+ edge (Sept 2026)", ry, recyds_key, lambda a: a["ev"] >= 0)
    ru = edge_finder_rows("rush_yds")
    run("rush yds unders 0%+ (official, Sept 2026)", ru, rush_key, lambda a: (a["side"] == "under") & (a["ev"] >= 0))
    run("rush yds, every 0%+ edge (Sept 2026)", ru, rush_key, lambda a: a["ev"] >= 0)

    lv = live_rows()
    lv_rec = lv[lv["market"] == "Receptions"].drop_duplicates(["key", "side"])
    lv_ry = lv[lv["market"] == "Reception Yards"].drop_duplicates(["key", "side"])
    lv_ru = lv[lv["market"] == "Rush Yards"].drop_duplicates(["key", "side"])
    run("LIVE receptions unders 30%+", lv_rec, rec_key, lambda a: (a["side"] == "under") & (a["ev"] >= 0.30))
    run("LIVE rec yds overs 15%+", lv_ry, recyds_key, lambda a: (a["side"] == "over") & (a["ev"] >= 0.15))
    run("LIVE rush yds unders 0%+", lv_ru, rush_key, lambda a: (a["side"] == "under") & (a["ev"] >= 0))
    run("LIVE every graded rec/rec yds/rush yds prop", pd.concat([lv_rec.assign(k="rec"), lv_ry.assign(k="ry"), lv_ru.assign(k="ru")]),
        lambda r: rec_key(r) if r["k"] == "rec" else (recyds_key(r) if r["k"] == "ry" else rush_key(r)), lambda a: a["ev"] >= 0)

    # Anytime TD: does the TD-allowed grade move the actual scoring rate
    # beyond what the model and DraftKings already say?
    td = td_rows()
    a = support(attach(td, gi, wide, scores), td_key)
    a = a[a["support"].notna()]
    cal = {}
    for b, g in a.groupby("bucket"):
        cal[b] = {"n": int(len(g)), "actual": round(float(g["scored"].mean()) * 100, 1),
                  "model": round(float(g["p"].mean()) * 100, 1), "dk_implied": round(float(g["mkt"].mean()) * 100, 1)}
    by_pos = {}
    for (pg, b), g in a.groupby(["pos_p", "bucket"]):
        by_pos.setdefault(pg, {})[b] = {"n": int(len(g)), "actual": round(float(g["scored"].mean()) * 100, 1),
                                         "model": round(float(g["p"].mean()) * 100, 1)}
    results["anytime TD"] = {"calibration_by_bucket": cal, "calibration_by_position": by_pos,
                             "yes_10plus_dk": by_bucket(a[a["ev"] >= 0.10]), "verdict_10plus": verdict(by_bucket(a[a["ev"] >= 0.10])),
                             "yes_all_dk": by_bucket(a), "verdict_all": verdict(by_bucket(a))}
    details.append(a[["sample", "player", "team", "opponent", "date", "side", "ev", "units", "pos_p", "tough", "support", "bucket"]].assign(group="anytime TD"))
    print(f"  anytime TD: {len(a)} graded, 10%+ {results['anytime TD']['verdict_10plus']}")

    os.makedirs("docs/data", exist_ok=True)
    pd.concat(details, ignore_index=True).to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as f:
        json.dump({"generated_at": datetime.now(timezone.utc).isoformat(), "buckets": {"with": WITH, "against": AGAINST, "min_games": MIN_GAMES},
                   "results": results}, f, indent=2, default=str)
    buf = io.StringIO()
    with redirect_stdout(buf):
        report(results, OUT_TD if not now.empty else None)
    txt = buf.getvalue()
    print(txt)
    with open(OUT_TXT, "w") as f:
        f.write(txt)


def report(results, td_path):
    print("=" * 72)
    print("MATCHUP LAB -- does the matchup grade help?")
    print(f"with = grade supports the pick (>= {WITH}), against = grade opposes it (<= {AGAINST})")
    print("=" * 72)
    for name, r in results.items():
        if name in ("info", "anytime TD"):
            continue
        print(f"\n{name}   [{r['verdict']}]  graded {r['matched_pct']}%")
        for s, t in r["by_bucket"].items():
            print("  " + f"{s:<14}" + "  ".join(f"{b}: {v.get('record', '-')} {v.get('roi', 0):+.1f}%" for b, v in t.items() if v.get("n")))
        if r.get("model_miss_by_defense"):
            print("  model miss (actual - projection): " + ", ".join(f"{k} {v['actual_minus_proj']:+.2f} (n={v['n']})" for k, v in r["model_miss_by_defense"].items()))
    t = results.get("anytime TD")
    if t:
        print("\nANYTIME TD (DraftKings Yes, direct_cal model)")
        for b, v in t["calibration_by_bucket"].items():
            print(f"  {b:<8} n={v['n']:<6} scored {v['actual']}%  model {v['model']}%  DK implied {v['dk_implied']}%")
        for pg, d in t["calibration_by_position"].items():
            print(f"  {pg.upper():<3} " + "  ".join(f"{b}: {v['actual']}% vs model {v['model']}% (n={v['n']})" for b, v in d.items()))
        print(f"  Yes 10%+ edge: {t['verdict_10plus']}")
        for s, x in t["yes_10plus_dk"].items():
            print("  " + f"{s:<8}" + "  ".join(f"{b}: {v.get('record', '-')} {v.get('roi', 0):+.1f}%" for b, v in x.items() if v.get("n")))
    if td_path and os.path.exists(td_path):
        tab = pd.read_csv(td_path)
        print("\nMOST TDs ALLOWED PER GAME BY POSITION, 2026 (FBS defenses, box scores)")
        for g in TD_GROUPS:
            top = tab.sort_values(f"td_{g}_per_game", ascending=False).head(10)
            print(f"  {g.upper()}: " + ", ".join(f"{r['defense']} {r[f'td_{g}_per_game']:.2f}" for _, r in top.iterrows()))
    print(f"\ninfo: {results.get('info')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--season", type=int, default=2026)
    a = ap.parse_args()
    main(a.years, a.season)
