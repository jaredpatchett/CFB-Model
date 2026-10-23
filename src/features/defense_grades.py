"""
Matchup grades on every player prop (added 10/2026).

For each prop, how does the opposing defense rate against what this pick
needs? The defense gets a toughness score (0-100, 100 = toughest in the
country) for the kind of play the prop is about, then the pick gets a grade
from it: an under (or an interceptions over) is helped by a tough defense,
an over by a soft one.

    receptions / receiving yards / receiving TDs   vs this player's position
        catches allowed to WR / TE / RB against what those offenses usually
        get (PFF receiving rows), receiving yards allowed the same way, and
        the PFF coverage grade
    rush yards / rush TDs
        rush yards allowed against those offenses' usual (box scores), PFF
        run-defense grade, PFF tackling grade
    pass yards / TDs / completions / interceptions
        pass yards allowed against those offenses' usual (box scores), PFF
        coverage grade, PFF pass-rush grade

Attempts props get no grade: a defense's quality doesn't clearly push the
number of carries or throws either way.

Each input is turned into a z-score across every defense with a number,
the inputs are averaged by weight (a missing one is left out), and the
average is ranked into a 0-100 percentile. "Allowed against usual" numbers
are pulled toward average with few games played (same as props lab 3).

What gets published: the grade, the score, the defense's national rank and
the box-score "allowed" numbers always. The PFF grades and the PFF-derived
receiving numbers only when config.PUBLISH_PFF_NUMBERS is True -- PFF's data
is licensed to the subscriber, so that switch stays off while the repo is
public.

This is context, not a tested edge. The one matchup split props lab 3 found
worth anything is receptions unders with a 10%+ edge against tough
defenses. The props models already use the opponent's yards allowed per
game, so a grade will sometimes repeat what the edge already says.
"""
import numpy as np
import pandas as pd

from src.features import matchup as mu

SHRINK_GAMES = mu.SHRINK_GAMES

PFF_DEF = {                         # name -> (grade column, snaps column)
    "coverage": ("grades_coverage_defense", "snap_counts_coverage"),
    "run_def": ("grades_run_defense", "snap_counts_run_defense"),
    "pass_rush": ("grades_pass_rush_defense", "snap_counts_pass_rush"),
    "tackle": ("grades_tackle", "snap_counts_defense"),
}
PFF_LABELS = {"coverage": "Coverage", "run_def": "Run defense", "pass_rush": "Pass rush", "tackle": "Tackling"}

FAMILY = {
    "Receptions": "rec", "Reception Yards": "rec", "Reception Touchdowns": "rec",
    "Rush Yards": "rush", "Rush Touchdowns": "rush",
    "Pass Yards": "pass", "Pass Touchdowns": "pass", "Pass Completions": "pass", "Pass Interceptions": "pass",
}
INVERTED = {"Pass Interceptions"}   # a tough defense helps the OVER here

# (input, weight, higher number = tougher defense)
COMPONENTS = {
    "rec": [("catches_{g}", 0.35, False), ("recyds_{g}", 0.35, False), ("coverage", 0.30, True)],
    "rush": [("rush_yds", 0.50, False), ("run_def", 0.35, True), ("tackle", 0.15, True)],
    "pass": [("pass_yds", 0.50, False), ("coverage", 0.30, True), ("pass_rush", 0.20, True)],
}
PFF_INPUTS = {"coverage", "run_def", "pass_rush", "tackle"} | {f"{k}_{g}" for k in ("catches", "recyds") for g in mu.GROUPS}

VS_TEXT = {"rec_wr": "WRs", "rec_te": "TEs", "rec_rb": "RBs", "rush": "the run", "pass": "the pass"}
ALLOWED_TEXT = {"catches": "catches", "recyds": "receiving yds", "rush_yds": "rush yds", "pass_yds": "pass yds"}


def letter(score):
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return None
    for cut, g in ((90, "A+"), (80, "A"), (70, "B+"), (60, "B"), (40, "C"), (25, "D")):
        if score >= cut:
            return g
    return "F"


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def allowed_ratios(og, metrics, team_col, opp_col):
    """One row per defense (index = opp): games, and for each metric what it
    allowed against what those offenses average in their OTHER games,
    pulled toward 1.00 by SHRINK_GAMES. Same method as matchup.py."""
    g = og.dropna(subset=[opp_col])
    if g.empty:
        return pd.DataFrame(columns=["games"] + metrics)
    off = g.groupby(team_col)[metrics].agg(["sum", "count"])
    league = g[metrics].mean()
    exp = pd.DataFrame(index=g.index)
    for m in metrics:
        s, n = g[team_col].map(off[(m, "sum")]), g[team_col].map(off[(m, "count")])
        exp[m] = np.where(n > 1, (s - g[m]) / (n - 1).replace(0, np.nan), league[m])
    allowed = g.groupby(opp_col)[metrics].sum()
    expected = exp.groupby(g[opp_col]).sum()
    games = g.groupby(opp_col).size().reindex(allowed.index)
    out = pd.DataFrame({"games": games}, index=allowed.index)
    for m in metrics:
        ratio = allowed[m] / expected[m].replace(0, np.nan)
        out[m] = (out["games"] * ratio + SHRINK_GAMES) / (out["games"] + SHRINK_GAMES)
    return out


def box_score_allowed(wide):
    """school -> rush_yds / pass_yds ratio allowed, from CFBD box scores."""
    need = {"gameId", "team", "opponent"}
    if wide is None or wide.empty or not need <= set(wide.columns):
        return pd.DataFrame(columns=["games", "rush_yds", "pass_yds"])
    w = wide.copy()
    w["rush_yds"], w["pass_yds"] = _num(w, "rush_yds").fillna(0.0), _num(w, "pass_yds").fillna(0.0)
    og = w.groupby(["gameId", "team", "opponent"])[["rush_yds", "pass_yds"]].sum().reset_index()
    return allowed_ratios(og, ["rush_yds", "pass_yds"], "team", "opponent")


def pff_receiving_allowed(recv, pgames):
    """franchise_id -> catches / receiving yards allowed to each position
    group against usual, from PFF receiving rows."""
    if recv is None or recv.empty:
        return pd.DataFrame()
    df = recv.copy()
    rec, yds = _num(df, "receptions").fillna(0.0), _num(df, "yards").fillna(0.0)
    g = mu.pos_group(df["position"]) if "position" in df.columns else pd.Series("other", index=df.index)
    metrics = []
    for grp in mu.GROUPS:
        df[f"catches_{grp}"] = np.where(g == grp, rec, 0.0)
        df[f"recyds_{grp}"] = np.where(g == grp, yds, 0.0)
        metrics += [f"catches_{grp}", f"recyds_{grp}"]
    og = df.groupby(mu.KEYS)[metrics].sum().reset_index().merge(mu.opponents(pgames), on=mu.KEYS, how="left")
    return allowed_ratios(og, metrics, "franchise_id", "opp")


def pff_defense_grades(deff):
    """franchise_id -> season snap-weighted PFF coverage / run defense /
    pass rush / tackling grades."""
    if deff is None or deff.empty or "franchise_id" not in deff.columns:
        return pd.DataFrame()
    out = {}
    for name, (gcol, scol) in PFF_DEF.items():
        gr, sn = _num(deff, gcol), _num(deff, scol).fillna(0.0)
        ok = gr.notna() & (sn > 0)
        t = pd.DataFrame({"f": deff["franchise_id"], "gs": (gr * sn).where(ok), "s": sn.where(ok)})
        a = t.groupby("f")[["gs", "s"]].sum(min_count=1)
        out[name] = a["gs"] / a["s"]
    return pd.DataFrame(out)


def build_profiles(wide=None, recv=None, pgames=None, deff=None, school_of=None):
    """{school: {"inputs": {name: value}, "scores": {key: {pct, rank, of}}}}
    where key is rec_wr / rec_te / rec_rb / rush / pass."""
    school_of = school_of or {}
    inputs = {}

    def put(df, by_franchise):
        if df is None or df.empty:
            return
        for idx, row in df.iterrows():
            school = school_of.get(idx) if by_franchise else idx
            if school is None or (isinstance(school, float) and np.isnan(school)):
                continue
            d = inputs.setdefault(school, {})
            for c in df.columns:
                if c == "games":
                    continue
                v = row[c]
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    d.setdefault(c, float(v))

    put(box_score_allowed(wide), False)
    put(pff_receiving_allowed(recv, pgames), True)
    put(pff_defense_grades(deff), True)
    if not inputs:
        return {}
    table = pd.DataFrame.from_dict(inputs, orient="index")
    keys = {"rush": COMPONENTS["rush"], "pass": COMPONENTS["pass"]}
    for grp in mu.GROUPS:
        keys[f"rec_{grp}"] = [(n.format(g=grp), w, hi) for n, w, hi in COMPONENTS["rec"]]
    scores = {}
    for key, comps in keys.items():
        num = pd.Series(0.0, index=table.index)
        wsum = pd.Series(0.0, index=table.index)
        for name, w, higher_tough in comps:
            if name not in table.columns:
                continue
            col = table[name]
            sd = col.std()
            if col.notna().sum() < 10 or not sd or np.isnan(sd):
                continue
            z = (col - col.mean()) / sd * (1 if higher_tough else -1)
            num = num + (z * w).fillna(0.0)
            wsum = wsum + col.notna() * w
        comp = (num / wsum.replace(0, np.nan)).dropna()
        if comp.empty:
            continue
        pct = comp.rank(pct=True) * 100
        rank = comp.rank(ascending=False, method="min").astype(int)
        for school in comp.index:
            scores.setdefault(school, {})[key] = {"pct": float(pct[school]), "rank": int(rank[school]), "of": int(len(comp))}
    return {s: {"inputs": inputs.get(s, {}), "scores": scores.get(s, {})} for s in set(inputs) | set(scores)}


def _pct_txt(ratio):
    """1.14 -> +14 (allows 14% more than usual)."""
    return int(round((ratio - 1.0) * 100))


def label_props(props, player_form, pos, profiles, name_key, publish_pff=False):
    """Adds to each gradable prop:
        mu_grade  'A+'..'F'   the matchup for THIS pick
        mu_score  0-100       same, as a number (higher = better for the pick)
        mu_rank / mu_of       the defense's national rank vs this kind of play (1 = toughest)
        mu_vs                 'WRs' / 'TEs' / 'RBs' / 'the run' / 'the pass'
        mu_def                'tough' / 'average' / 'soft'
        mu_allowed            {label: +/- % vs usual} (receiving ones only with publish_pff)
        mu_pff                {label: PFF grade} (only with publish_pff)
    Returns how many props were graded."""
    n = 0
    for p in props:
        market = p.get("market_name")
        fam = FAMILY.get(market)
        lean = p.get("model_lean")
        if not fam or lean not in ("over", "under"):
            continue
        prof = profiles.get(p.get("opponent"))
        if not prof:
            continue
        if fam == "rec":
            entry = player_form.get(name_key(p.get("player_name"))) or {}
            g = pos.get(entry.get("athlete_id"))
            if not g:
                continue
            key = f"rec_{g}"
            comps = [(c.format(g=g), w, hi) for c, w, hi in COMPONENTS["rec"]]
        else:
            g, key, comps = None, fam, COMPONENTS[fam]
        sc = prof["scores"].get(key)
        if not sc:
            continue
        tough_helps = (lean == "under") != (market in INVERTED)
        support = sc["pct"] if tough_helps else 100.0 - sc["pct"]
        p["mu_grade"] = letter(support)
        p["mu_score"] = int(round(support))
        p["mu_rank"], p["mu_of"] = sc["rank"], sc["of"]
        p["mu_vs"] = VS_TEXT[key]
        p["mu_def"] = "tough" if sc["pct"] >= 67 else ("soft" if sc["pct"] <= 33 else "average")
        inp = prof["inputs"]
        allowed, pffg = {}, {}
        for name, _, _ in comps:
            v = inp.get(name)
            if v is None:
                continue
            if name in PFF_DEF:
                if publish_pff:
                    pffg[PFF_LABELS[name]] = round(v, 1)
            elif name in PFF_INPUTS:
                if publish_pff:
                    allowed[ALLOWED_TEXT[name.split("_")[0]] + " to " + VS_TEXT[key]] = _pct_txt(v)
            else:
                allowed[ALLOWED_TEXT[name]] = _pct_txt(v)
        if allowed:
            p["mu_allowed"] = allowed
        if pffg:
            p["mu_pff"] = pffg
        n += 1
    return n


def fetch_pff_defense(season, weeks=None):
    """This season's PFF defense rows, one week at a time. A failed week is
    skipped (the grades just use fewer games); a bad key raises."""
    from src.data import pff_client as pff
    parts = []
    for wk in (weeks or pff.MODEL_WEEKS):
        df, err = pff._attempt(lambda s, w: pff.get_facet_summary("defense", s, w), season, wk)
        if err is None and df is not None and not df.empty:
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def live_profiles(season):
    """(profiles, status text). Uses the receiving rows, box scores and
    franchise -> school map the matchup labels just built (matchup.LAST),
    plus this season's PFF defense grades. Never raises."""
    try:
        last = mu.LAST
        deff, note = pd.DataFrame(), ""
        try:
            deff = fetch_pff_defense(season)
        except Exception as e:
            note = f" (no PFF defense grades: {type(e).__name__})"
        profiles = build_profiles(last.get("wide"), last.get("recv"), last.get("pgames"), deff, last.get("school"))
        if not profiles:
            return {}, "no matchup grades this run -- no defense numbers"
        n = sum(1 for v in profiles.values() if v["scores"])
        return profiles, f"matchup grades for {n} defenses{note}"
    except Exception as e:
        return {}, f"no matchup grades this run ({type(e).__name__}: {e})"
