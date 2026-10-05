"""
Challenger ("v2") game features, added 10/2026.

Two seasons of honest backtesting showed the current game model has no
reliable spread edge, and its biggest disagreements with the market -- the
Bet Card -- lost most (likely the model missing information the market has).
These features add the kinds of information that most often explain those
misses. Every value uses only games played BEFORE the game being predicted.

  ewm_adj_margin_diff  scoring margin adjusted for opponent strength (prior-
                       season SP+), recent games weighted more (half-life of
                       3 games). Live scoring already opponent-adjusts margin;
                       training never did -- this makes them consistent.
  ewm_off_ppa_diff     offensive expected points added per play (CFBD PPA)
  ewm_def_ppa_diff     defensive PPA allowed per play
  ewm_net_ppa_diff     offense minus defense PPA -- per-play team strength,
                       far less noisy than final scores
  ewm_off_sr_diff,     success rate on offense / allowed on defense
  ewm_def_sr_diff
  rest_diff            days since each team's previous game (capped at 21)
  home_qb_out,         1 if the team's leading passer so far this season
  away_qb_out          didn't play. In training this comes from the box score
                       (he had no pass attempts); live, it comes from the
                       injury report (starting QB listed Out/Doubtful).

All diffs are home minus away. Used side by side with the current features
in scripts/run_backtest.py; nothing live changes unless this model wins.
"""
import numpy as np
import pandas as pd

from src.features.team_features import FEATURE_COLUMNS

HALF_LIFE_GAMES = 3
V2_EXTRA = ["ewm_adj_margin_diff", "ewm_net_ppa_diff", "ewm_off_ppa_diff", "ewm_def_ppa_diff",
            "ewm_off_sr_diff", "ewm_def_sr_diff", "rest_diff", "home_qb_out", "away_qb_out"]
FEATURE_COLUMNS_V2 = list(FEATURE_COLUMNS) + V2_EXTRA


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    if "completed" in g.columns:
        g = g[g["completed"] == True]
    g = g.dropna(subset=["homePoints", "awayPoints"])
    start = pd.to_datetime(g.get("startDate"), utc=True, errors="coerce")
    home = pd.DataFrame({"gameId": g["id"], "season": g["season"], "week": g["week"], "start": start,
                         "team": g["homeTeam"], "opponent": g["awayTeam"],
                         "margin": g["homePoints"] - g["awayPoints"]})
    away = pd.DataFrame({"gameId": g["id"], "season": g["season"], "week": g["week"], "start": start,
                         "team": g["awayTeam"], "opponent": g["homeTeam"],
                         "margin": g["awayPoints"] - g["homePoints"]})
    return pd.concat([home, away], ignore_index=True)


def _prior_sp(sp: pd.DataFrame) -> dict:
    """(season, team) -> that team's SP+ rating from the season BEFORE."""
    out = {}
    if sp is None or sp.empty or "team" not in sp.columns:
        return out
    year_col = "year" if "year" in sp.columns else ("season" if "season" in sp.columns else None)
    if year_col is None:
        return out
    for _, r in sp.dropna(subset=["rating"]).iterrows():
        out[(int(r[year_col]) + 1, r["team"])] = float(r["rating"])
    return out


def _qb_out(player_wide: pd.DataFrame, tg: pd.DataFrame) -> pd.Series:
    """Per team-game: 1 if the team's season-to-date leader in pass attempts
    (10+ attempts before this game) didn't throw a pass in this game."""
    out = pd.Series(np.nan, index=tg.index)
    if player_wide is None or player_wide.empty or "pass_att" not in player_wide.columns:
        return out
    pw = player_wide[["gameId", "team", "athleteId", "pass_att"]].copy()
    pw["pass_att"] = pd.to_numeric(pw["pass_att"], errors="coerce").fillna(0)
    passed = pw[pw.pass_att > 0].groupby(["gameId", "team"])["athleteId"].apply(set).to_dict()
    att = pw.groupby(["gameId", "team", "athleteId"])["pass_att"].sum()
    order = tg.sort_values(["team", "season", "start", "week"])
    for (team, season), grp in order.groupby(["team", "season"]):
        totals = {}
        for idx, row in grp.iterrows():
            leader = max(totals.items(), key=lambda kv: kv[1]) if totals else None
            if leader and leader[1] >= 10:
                out.at[idx] = 0.0 if leader[0] in passed.get((row.gameId, team), set()) else 1.0
            else:
                out.at[idx] = 0.0
            if (row.gameId, team) in passed:
                for aid in passed[(row.gameId, team)]:
                    totals[aid] = totals.get(aid, 0) + float(att.get((row.gameId, team, aid), 0))
    return out


def build_v2_features(team_features: pd.DataFrame, games: pd.DataFrame, game_adv: pd.DataFrame,
                      sp: pd.DataFrame, player_wide: pd.DataFrame = None) -> pd.DataFrame:
    """Returns team_features with the V2_EXTRA columns added (NaN where a
    team has no prior games yet this season, matching the current features)."""
    tg = _team_games(games)
    prior = _prior_sp(sp)
    opp_sp = [prior.get((int(s), o)) for s, o in zip(tg["season"], tg["opponent"])]
    tg["adj_margin"] = tg["margin"] + pd.Series(opp_sp, index=tg.index).fillna(0.0)

    if game_adv is not None and not game_adv.empty:
        cols = {"offense.ppa": "off_ppa", "offense.successRate": "off_sr",
                "defense.ppa": "def_ppa", "defense.successRate": "def_sr"}
        have = [c for c in cols if c in game_adv.columns]
        adv = game_adv[["gameId", "team"] + have].rename(columns=cols)
        tg = tg.merge(adv.drop_duplicates(["gameId", "team"]), on=["gameId", "team"], how="left")
    for c in ("off_ppa", "off_sr", "def_ppa", "def_sr"):
        if c not in tg.columns:
            tg[c] = np.nan

    tg = tg.sort_values(["team", "season", "start", "week"]).reset_index(drop=True)
    grp = tg.groupby(["team", "season"])
    for c in ("adj_margin", "off_ppa", "def_ppa", "off_sr", "def_sr"):
        tg[f"ewm_{c}"] = grp[c].transform(lambda s: s.shift(1).ewm(halflife=HALF_LIFE_GAMES, min_periods=1).mean())
    tg["ewm_net_ppa"] = tg["ewm_off_ppa"] - tg["ewm_def_ppa"]
    rest = grp["start"].transform(lambda s: s.diff().dt.days)
    tg["rest_days"] = rest.fillna(14).clip(upper=21)
    tg["qb_out"] = _qb_out(player_wide, tg)

    keep = ["gameId", "team", "ewm_adj_margin", "ewm_off_ppa", "ewm_def_ppa", "ewm_net_ppa",
            "ewm_off_sr", "ewm_def_sr", "rest_days", "qb_out"]
    side = tg[keep]
    df = team_features.merge(side.add_prefix("home_").rename(columns={"home_gameId": "id", "home_team": "homeTeam"}),
                             on=["id", "homeTeam"], how="left")
    df = df.merge(side.add_prefix("away_").rename(columns={"away_gameId": "id", "away_team": "awayTeam"}),
                  on=["id", "awayTeam"], how="left")
    for c in ("adj_margin", "off_ppa", "def_ppa", "net_ppa", "off_sr", "def_sr"):
        df[f"ewm_{c}_diff"] = df[f"home_ewm_{c}"] - df[f"away_ewm_{c}"]
    df["rest_diff"] = df["home_rest_days"] - df["away_rest_days"]
    df["home_qb_out"], df["away_qb_out"] = df["home_qb_out"], df["away_qb_out"]
    return df
