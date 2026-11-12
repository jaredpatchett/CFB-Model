"""
Injury / absence inputs for the player props models (11/2026, lab only).

The props models have no injury input. A player whose lead back or top
receiver went down only shows it through his last-2-game inputs, while his
season inputs still average in the games the teammate played. These inputs
tell the model directly how much work is freed up and what this player did
the last time that work was free.

For each player-game and each share (carries, catches):

  vac_<share>        share of the team's work that belonged to REGULARS who
                     are missing from this game. A regular: 2+ games played
                     this season before this one, averaging >= the threshold
                     share in them. Missing: no box-score row for this game
                     (a played-zero row from PFF counts as playing).
  <share>_without    this player's average share in his earlier games this
                     season where at least one of those same regulars was
                     also missing; his season share when there are none.
  n_without_<share>  how many earlier games that is.

In training, "missing" comes from the box score of the game itself. Live it
would come from the injury report (Out / Doubtful) -- close but not the
same (some absences never make the report), noted in the lab results.
"""
import numpy as np
import pandas as pd

SHARES = {"carry_share": 0.15, "rec_share": 0.12}
INJURY_FEATURE_COLUMNS = {
    c: [f"vac_{c}", f"{c}_without", f"n_without_{c}"] for c in SHARES
}
ALL_INJURY_COLUMNS = [x for cols in INJURY_FEATURE_COLUMNS.values() for x in cols]


def add_absence_features(feats: pd.DataFrame) -> pd.DataFrame:
    """Adds ALL_INJURY_COLUMNS to a player-game frame that has team, season,
    week, gameId, athleteId and the per-game share columns. Uses only games
    before each row (plus who is missing from the row's own game)."""
    df = feats.copy()
    for c in ALL_INJURY_COLUMNS:
        df[c] = np.nan
    need = {"team", "season", "week", "gameId", "athleteId"}
    if not need.issubset(df.columns):
        return df
    out = {c: np.full(len(df), np.nan) for c in ALL_INJURY_COLUMNS}
    pos = pd.Series(np.arange(len(df)), index=df.index)
    for (_, _), tg in df.groupby(["team", "season"], sort=False):
        order = tg[["week", "gameId"]].drop_duplicates().sort_values(["week", "gameId"])
        games = list(order["gameId"])
        played = {g: set(tg.loc[tg["gameId"] == g, "athleteId"]) for g in games}
        for share, thr in SHARES.items():
            if share not in tg.columns:
                continue
            val = {(a, g): float(v) for a, g, v in zip(tg["athleteId"], tg["gameId"], pd.to_numeric(tg[share], errors="coerce").fillna(0.0))}
            hist = {}          # athleteId -> list of shares in games so far
            absent_at = {}     # gameId -> {athleteId: their prior mean share}
            for g in games:
                regs = {a: float(np.mean(v)) for a, v in hist.items() if len(v) >= 2 and np.mean(v) >= thr}
                absent_at[g] = {a: s for a, s in regs.items() if a not in played[g]}
                for a in played[g]:
                    hist.setdefault(a, []).append(val[(a, g)])
            for g_i, g in enumerate(games):
                gone = absent_at[g]
                vac = sum(gone.values())
                rows = tg.index[tg["gameId"] == g]
                for idx in rows:
                    a = tg.at[idx, "athleteId"]
                    prior = [h for h in games[:g_i] if a in played[h]]
                    season_mean = float(np.mean([val[(a, h)] for h in prior])) if prior else np.nan
                    if gone:
                        w = [val[(a, h)] for h in prior if set(absent_at[h]) & set(gone)]
                    else:
                        w = []
                    p = pos[idx]
                    out[f"vac_{share}"][p] = vac
                    out[f"{share}_without"][p] = float(np.mean(w)) if w else season_mean
                    out[f"n_without_{share}"][p] = float(len(w))
    for c in ALL_INJURY_COLUMNS:
        df[c] = out[c]
    return df
