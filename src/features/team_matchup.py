"""
Team-vs-team matchup ranks (11/2026), for the Matchup Projector.

For every FBS team this season: where its offense and its defense rank
nationally in each category, so a game can be read as "this offense vs that
defense" category by category. Rank 1 = best for that unit (most productive
offense, stingiest defense).

Sources
  CFBD advanced season stats  EPA/play, success rate, explosiveness, points
                              per scoring opportunity, rushing / passing EPA,
                              line yards, stuff rate, havoc, passing downs
  CFBD box scores             rush and pass yards per game, for and against
  CFBD schedule               points per game, for and against
  CFBD season stats           sacks, turnovers, third downs (when present)
  PFF (season, snap-weighted) offense / passing / pass block / run block /
                              receiving / rushing grades against defense /
                              coverage / pass rush / run defense / tackling,
                              plus pressure rate allowed vs pressure rate made

What gets published: the rank (and "of N") for everything; the value for
CFBD numbers only. PFF values stay out unless config.PUBLISH_PFF_NUMBERS is
True (PFF's data is licensed to the subscriber and the repo is public).

build(...) -> ({school: {"off": {key: {rank, of, value?}}, "def": {...}}}, cats, status)
"""
import numpy as np
import pandas as pd

# key, offense label, defense label, offense higher-is-better, defense higher-is-better, source, value format
CATS = [
    ("ppg",      "Points per game",          "Points allowed per game",   True,  False, "cfbd", "{:.1f}"),
    ("epa",      "EPA per play",             "EPA per play allowed",      True,  False, "cfbd", "{:+.3f}"),
    ("sr",       "Success rate",             "Success rate allowed",      True,  False, "cfbd", "{:.1%}"),
    ("expl",     "Explosiveness",            "Explosiveness allowed",     True,  False, "cfbd", "{:.2f}"),
    ("ppo",      "Points per scoring opp.",  "Points per opp. allowed",   True,  False, "cfbd", "{:.2f}"),
    ("rush_ypg", "Rush yards per game",      "Rush yards allowed per game", True, False, "cfbd", "{:.0f}"),
    ("rush_epa", "Rush EPA per play",        "Rush EPA allowed",          True,  False, "cfbd", "{:+.3f}"),
    ("line",     "Line yards",               "Line yards allowed",        True,  False, "cfbd", "{:.2f}"),
    ("stuff",    "Stuff rate (runs stopped)", "Stuff rate",               False, True,  "cfbd", "{:.1%}"),
    ("pass_ypg", "Pass yards per game",      "Pass yards allowed per game", True, False, "cfbd", "{:.0f}"),
    ("pass_epa", "Pass EPA per play",        "Pass EPA allowed",          True,  False, "cfbd", "{:+.3f}"),
    ("pd_sr",    "Passing-downs success",    "Passing-downs success allowed", True, False, "cfbd", "{:.1%}"),
    ("havoc",    "Havoc allowed",            "Havoc rate",                False, True,  "cfbd", "{:.1%}"),
    ("sacks",    "Sacks allowed per game",   "Sacks per game",            False, True,  "cfbd", "{:.1f}"),
    ("third",    "Third-down rate",          "Third-down rate allowed",   True,  False, "cfbd", "{:.1%}"),
    ("to",       "Turnovers per game",       "Takeaways per game",        False, True,  "cfbd", "{:.1f}"),
    ("pff_all",  "PFF offense grade",        "PFF defense grade",         True,  True,  "pff",  None),
    ("pff_pass", "PFF passing grade",        "PFF coverage grade",        True,  True,  "pff",  None),
    ("pff_recv", "PFF receiving grade",      "PFF coverage grade",        True,  True,  "pff",  None),
    ("pff_pblk", "PFF pass blocking",        "PFF pass rush",             True,  True,  "pff",  None),
    ("pff_pres", "Pressure rate allowed",    "Pressure rate",             False, True,  "pff",  None),
    ("pff_rblk", "PFF run blocking",         "PFF run defense",           True,  True,  "pff",  None),
    ("pff_rush", "PFF rushing grade",        "PFF tackling grade",        True,  True,  "pff",  None),
]
GROUPS = [("Scoring & efficiency", ["ppg", "epa", "sr", "expl", "ppo"]),
          ("Run game", ["rush_ypg", "rush_epa", "line", "stuff", "pff_rblk", "pff_rush"]),
          ("Pass game", ["pass_ypg", "pass_epa", "pd_sr", "pff_pass", "pff_recv", "pff_pblk", "pff_pres", "sacks", "havoc"]),
          ("Situational & overall", ["third", "to", "pff_all"])]


def _n(df, c):
    return pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series(np.nan, index=df.index)


def _cfbd_advanced(adv):
    """school -> {(unit, key): value} from /stats/season/advanced."""
    out = {}
    if adv is None or adv.empty or "team" not in adv.columns:
        return out
    m = {"epa": "ppa", "sr": "successRate", "expl": "explosiveness", "ppo": "pointsPerOpportunity",
         "rush_epa": "rushingPlays.ppa", "line": "lineYards", "stuff": "stuffRate", "pass_epa": "passingPlays.ppa",
         "pd_sr": "passingDowns.successRate", "havoc": "havoc.total"}
    for unit in ("offense", "defense"):
        for key, col in m.items():
            v = _n(adv, f"{unit}.{col}")
            for t, x in zip(adv["team"], v):
                if pd.notna(x):
                    out.setdefault(t, {})[(unit[:3], key)] = float(x)
    return out


def _box(wide, schedule):
    """Rush / pass yards per game for and against (box scores), points per
    game for and against (schedule)."""
    out = {}
    if wide is not None and not wide.empty and {"gameId", "team", "opponent"} <= set(wide.columns):
        w = wide.copy()
        for c in ("rush_yds", "pass_yds"):
            w[c] = _n(w, c).fillna(0.0)
        tg = w.groupby(["gameId", "team", "opponent"])[["rush_yds", "pass_yds"]].sum().reset_index()
        off = tg.groupby("team")[["rush_yds", "pass_yds"]].mean()
        dfn = tg.groupby("opponent")[["rush_yds", "pass_yds"]].mean()
        for t, r in off.iterrows():
            out.setdefault(t, {})[("off", "rush_ypg")] = float(r["rush_yds"])
            out[t][("off", "pass_ypg")] = float(r["pass_yds"])
        for t, r in dfn.iterrows():
            out.setdefault(t, {})[("def", "rush_ypg")] = float(r["rush_yds"])
            out[t][("def", "pass_ypg")] = float(r["pass_yds"])
    if schedule is not None and not schedule.empty:
        g = schedule.copy()
        hp, ap = _n(g, "homePoints"), _n(g, "awayPoints")
        done = hp.notna() & ap.notna()
        if "completed" in g.columns:
            done &= g["completed"] == True        # noqa: E712
        g = g[done]
        if not g.empty:
            long = pd.concat([pd.DataFrame({"team": g["homeTeam"], "pf": _n(g, "homePoints"), "pa": _n(g, "awayPoints")}),
                              pd.DataFrame({"team": g["awayTeam"], "pf": _n(g, "awayPoints"), "pa": _n(g, "homePoints")})])
            pts = long.groupby("team")[["pf", "pa"]].mean()
            for t, r in pts.iterrows():
                out.setdefault(t, {})[("off", "ppg")] = float(r["pf"])
                out[t][("def", "ppg")] = float(r["pa"])
    return out


def _season_stats(st):
    """Sacks, turnovers and third downs from /stats/season, when present.
    CFBD lists a team's own stats and its opponents' as '<name>Opponent'."""
    out, names = {}, set()
    if st is None or st.empty or not {"team", "statName", "statValue"} <= set(st.columns):
        return out, names
    names = set(st["statName"].dropna())
    piv = st.assign(v=pd.to_numeric(st["statValue"], errors="coerce")).pivot_table(index="team", columns="statName", values="v", aggfunc="first")
    gcol = piv["games"] if "games" in piv.columns else None
    per = lambda c: (piv[c] / gcol) if (c in piv.columns and gcol is not None) else None
    pairs = {
        ("def", "sacks"): per("sacks"), ("off", "sacks"): per("sacksOpponent"),
        ("off", "to"): per("turnovers"), ("def", "to"): per("turnoversOpponent"),
    }
    if {"thirdDownConversions", "thirdDowns"} <= set(piv.columns):
        pairs[("off", "third")] = piv["thirdDownConversions"] / piv["thirdDowns"].replace(0, np.nan)
    if {"thirdDownConversionsOpponent", "thirdDownsOpponent"} <= set(piv.columns):
        pairs[("def", "third")] = piv["thirdDownConversionsOpponent"] / piv["thirdDownsOpponent"].replace(0, np.nan)
    for k, ser in pairs.items():
        if ser is None:
            continue
        for t, v in ser.items():
            if pd.notna(v):
                out.setdefault(t, {})[k] = float(v)
    return out, names


def _weighted(df, grade, snaps):
    if df is None or df.empty or "franchise_id" not in df.columns:
        return pd.Series(dtype=float)
    g, s = _n(df, grade), _n(df, snaps).fillna(0.0)
    ok = g.notna() & (s > 0)
    t = pd.DataFrame({"f": df["franchise_id"], "gs": (g * s).where(ok), "s": s.where(ok)})
    a = t.groupby("f")[["gs", "s"]].sum(min_count=1)
    return a["gs"] / a["s"]


def _pff(off, pas, blk, deff, school_of):
    """school -> {(unit, key): value}, season snap-weighted grades."""
    out = {}
    series = {
        ("off", "pff_all"): _weighted(off, "grades_offense", "snap_counts_total"),
        ("off", "pff_pblk"): _weighted(off, "grades_pass_block", "snap_counts_pass_block"),
        ("off", "pff_rblk"): _weighted(off, "grades_run_block", "snap_counts_run_block"),
        ("off", "pff_recv"): _weighted(off, "grades_pass_route", "snap_counts_pass_route"),
        ("off", "pff_rush"): _weighted(off, "grades_run", "snap_counts_run"),
        ("off", "pff_pass"): _weighted(pas, "grades_pass", "dropbacks"),
        ("def", "pff_all"): _weighted(deff, "grades_defense", "snap_counts_defense"),
        ("def", "pff_pblk"): _weighted(deff, "grades_pass_rush_defense", "snap_counts_pass_rush"),
        ("def", "pff_rblk"): _weighted(deff, "grades_run_defense", "snap_counts_run_defense"),
        ("def", "pff_pass"): _weighted(deff, "grades_coverage_defense", "snap_counts_coverage"),
        ("def", "pff_rush"): _weighted(deff, "grades_tackle", "snap_counts_defense"),
    }
    series[("def", "pff_recv")] = series[("def", "pff_pass")]
    # Pressure rates: pressures allowed per dropback, pressures made per pass play faced.
    try:
        if blk is not None and not blk.empty and pas is not None and not pas.empty:
            pa = pd.DataFrame({"f": blk["franchise_id"], "p": _n(blk, "pressures_allowed")}).groupby("f")["p"].sum(min_count=1)
            db = pd.DataFrame({"f": pas["franchise_id"], "d": _n(pas, "dropbacks")}).groupby("f")["d"].sum(min_count=1)
            series[("off", "pff_pres")] = pa / db.where(db > 0)
        if deff is not None and not deff.empty and {"season", "week"} <= set(deff.columns):
            pm = pd.DataFrame({"f": deff["franchise_id"], "p": _n(deff, "total_pressures")}).groupby("f")["p"].sum(min_count=1)
            faced = pd.DataFrame({"f": deff["franchise_id"], "w": deff["week"], "c": _n(deff, "snap_counts_coverage")}) \
                .groupby(["f", "w"])["c"].max().groupby("f").sum(min_count=1)
            series[("def", "pff_pres")] = pm / faced.where(faced > 0)
    except Exception:
        pass
    for k, ser in series.items():
        for f, v in ser.items():
            school = school_of.get(f)
            if school is not None and pd.notna(v):
                out.setdefault(school, {})[k] = float(v)
    return out


def rank_profiles(values, pool=None, publish_pff=False):
    """values: school -> {(unit, key): value}. Ranks among `pool` schools."""
    table = pd.DataFrame.from_dict({s: {f"{u}|{k}": v for (u, k), v in d.items()} for s, d in values.items()}, orient="index")
    if pool:
        table = table[table.index.isin(pool)]
    prof = {s: {"off": {}, "def": {}} for s in table.index}
    for key, _ol, _dl, off_hi, def_hi, src, fmt in CATS:
        for unit, hi in (("off", off_hi), ("def", def_hi)):
            col = f"{unit}|{key}"
            if col not in table.columns:
                continue
            ser = table[col].dropna()
            if len(ser) < 20:
                continue
            rk = ser.rank(ascending=not hi, method="min").astype(int)
            for s, r in rk.items():
                e = {"rank": int(r), "of": int(len(ser))}
                if src == "cfbd" or publish_pff:
                    e["value"] = round(float(ser[s]), 4)
                prof[s][unit][key] = e
    return prof


def build(season, adv=None, wide=None, schedule=None, season_stats=None, pff_frames=None, school_of=None, pool=None, publish_pff=False):
    """(profiles, cats, status). pff_frames: dict offense/passing/blocking/defense."""
    values = {}

    def merge(d):
        for s, kv in d.items():
            values.setdefault(s, {}).update(kv)
    merge(_cfbd_advanced(adv))
    merge(_box(wide, schedule))
    ss, names = _season_stats(season_stats)
    merge(ss)
    n_pff = 0
    if pff_frames and school_of:
        p = _pff(pff_frames.get("offense"), pff_frames.get("passing"), pff_frames.get("blocking"), pff_frames.get("defense"), school_of)
        n_pff = len(p)
        merge(p)
    prof = rank_profiles(values, pool=pool, publish_pff=publish_pff)
    cats = [{"key": k, "off": ol, "def": dl, "src": src, "fmt": fmt} for k, ol, dl, _a, _b, src, fmt in CATS]
    groups = [{"name": n, "keys": ks} for n, ks in GROUPS]
    status = (f"team matchup ranks for {len(prof)} teams ({n_pff} with PFF grades)"
              + ("" if {"sacksOpponent", "turnoversOpponent"} <= names or not names else
                 f"; season stats without opponent sacks/turnovers ({', '.join(sorted(n for n in names if 'Opponent' in n))[:120]})"))
    return prof, {"cats": cats, "groups": groups}, status


def fetch_pff_frames(season):
    """This season's PFF offense / passing / blocking / defense rows, week by
    week (kept in memory only). A failed week is skipped; a bad key raises."""
    from src.data import pff_client as pff
    reports = {"offense": "/v1/facet/offense/summary", "passing": "/v1/facet/passing/summary",
               "blocking": "/v1/facet/offense/blocking", "defense": "/v1/facet/defense/summary"}
    out = {}
    for name, path in reports.items():
        parts = []
        for wk in pff.MODEL_WEEKS:
            try:
                payload = pff._get(path, {"league": pff.LEAGUE, "season": season, "week": wk})
                df = pd.DataFrame(pff._rows(payload))
            except pff.PFFError as e:
                if e.status in (401, 403):
                    raise
                continue
            if not df.empty:
                df["season"], df["week"] = season, wk
                parts.append(df)
        out[name] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out
