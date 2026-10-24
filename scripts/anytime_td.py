"""
Anytime touchdown props (added 10/2026): live prices from every book, the
model's chance each player scores, and the fair odds that chance implies.

Kept in scripts/ (not src/) on purpose: the daily run retrains every model
whenever anything under src/ changes, and nothing here touches training.

How the model's chance is worked out
  The live rush TD and receiving TD models each project a player's
  touchdowns for his next game. Added together they are a scoring rate, and
  the chance of at least one touchdown is 1 - e^(-rate). In the anytime TD
  backtest (docs: anytime-td-backtest-2026-10-24; scripts/td_lab.py, the
  "poisson" version), this said 27.7% across 20,486 lines and players
  scored 27.3%, with every probability range within ~4 points.

What the backtest said about betting them
  Books price Yes well above the real scoring rate (34.5% implied vs 27.3%
  actual at the best price), so most lines lose. QBs and lead running backs
  lost at every edge size. Main receivers at a 20%+ edge were about
  break-even at DraftKings and +12% at the best price across books. The
  section is display only: nothing here is an official play or tracked.

Saved / shown
  data/current/anytime_td.csv     one row per game, book, player (Yes price)
  latest.json "anytime_td"        one row per player and game, all books
"""
import math
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import requests

import config

MARKET = "player_anytime_td"
CSV_PATH = f"{config.DATA_CURRENT_DIR}/anytime_td.csv"
CSV_COLS = ["fixture_id", "start_time", "home_team", "away_team", "book", "player_name", "price"]
BOOK_LABELS = {
    "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM", "williamhill_us": "Caesars",
    "espnbet": "ESPN BET", "betrivers": "BetRivers", "fanatics": "Fanatics", "bovada": "Bovada",
    "betonlineag": "BetOnline", "mybookieag": "MyBookie", "ballybet": "Bally Bet", "betparx": "betPARX",
    "fliff": "Fliff", "hardrockbet": "Hard Rock",
}
# Lines from these books are shown but never used for "best price": they are
# offshore books, not available in most legal US states (incl. Arizona).
OFFSHORE = {"bovada", "betonlineag", "mybookieag"}
VALUE_MIN_EDGE = 20.0          # "price beats the model by 20%+" tag
CAUTION_ROLES = {"QB", "Lead RB"}
# Minor-role players and backup backs are tagged "longshot", never "value":
# about a third of their lines in the backtest had no box-score row (sat, or
# played without touching the ball), and their apparent edge disappeared
# once those were counted.
LONGSHOT_ROLES = {"Minor role", "Backup RB"}


# ---------------------------------------------------------------- fetch ----
def fetch(events, regions="us,us2", max_days_out=8.0):
    """Yes prices from every book for every upcoming game. `events` is the
    raw list from odds_api_client.get_ncaaf_odds(). Never raises; returns a
    DataFrame (possibly empty)."""
    if not config.ODDS_API_KEY:
        print("  [anytime TD] no Odds API key -- skipped")
        return pd.DataFrame(columns=CSV_COLS)
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_days_out)
    rows, n_checked, n_with = [], 0, 0
    for g in events or []:
        try:
            kick = datetime.fromisoformat(str(g.get("commence_time")).replace("Z", "+00:00"))
        except ValueError:
            continue
        if kick < now or kick > cutoff or not g.get("id"):
            continue
        n_checked += 1
        try:
            resp = requests.get(
                f"{config.ODDS_API_BASE_URL}/sports/{config.ODDS_API_NCAAF_KEY}/events/{g['id']}/odds",
                params={"apiKey": config.ODDS_API_KEY, "regions": regions, "markets": MARKET, "oddsFormat": "american"},
                timeout=30)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [anytime TD] fetch failed for {g.get('away_team')} @ {g.get('home_team')}: {e}")
            continue
        got = False
        for bk in data.get("bookmakers") or []:
            for m in bk.get("markets") or []:
                if m.get("key") != MARKET:
                    continue
                for o in m.get("outcomes") or []:
                    side = str(o.get("name") or "").lower()
                    if side not in ("yes", "over") or not o.get("description") or o.get("price") is None:
                        continue
                    rows.append({"fixture_id": g["id"], "start_time": g.get("commence_time"),
                                 "home_team": g.get("home_team"), "away_team": g.get("away_team"),
                                 "book": bk.get("key"), "player_name": o["description"], "price": float(o["price"])})
                    got = True
        n_with += int(got)
    print(f"  [anytime TD] checked {n_checked} upcoming game(s), {n_with} had anytime TD lines ({len(rows)} book prices)")
    return pd.DataFrame(rows, columns=CSV_COLS)


# ---------------------------------------------------------------- odds -----
def implied(price):
    price = float(price)
    return -price / (-price + 100) if price < 0 else 100 / (price + 100)


def payout(price):
    price = float(price)
    return price / 100 if price > 0 else 100 / -price


def fair_american(p):
    """The American odds at which a bet with chance p breaks even."""
    if p is None or p <= 0 or p >= 1:
        return None
    return round(-100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


def edge_pct(p, price):
    return round((p * payout(price) - (1 - p)) * 100, 1)


def role(entry):
    pass_att = entry.get("roll_pass_att") or 0
    rush_att = entry.get("roll_rush_att") or 0
    catches = entry.get("roll_receptions") or 0
    if pass_att >= 10:
        return "QB"
    if rush_att >= 8:
        return "Lead RB"
    if catches >= 3:
        return "Main receiver"
    if rush_att >= 3:
        return "Backup RB"
    return "Minor role"


# --------------------------------------------------------------- export ----
def build(player_form, schedule_df, opp_defense_lookup, prop_models, score_prop, normalize_name,
          env_lookup=None, opp_allowed_lookup=None, injury_lookup=None, raw=None, path=CSV_PATH):
    """One row per player and game with an anytime TD line: model chance,
    fair odds, every book's price and the edge at the best legal price and
    at DraftKings. Players the model can't price are kept, without a chance,
    so every posted line is still visible. Never raises."""
    try:
        if not os.path.exists(path):
            return [], "no anytime TD lines fetched this run"
        try:
            lines = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return [], "no anytime TD lines posted"
        if lines.empty:
            return [], "no anytime TD lines posted"
        out, n_priced = [], 0
        have_models = prop_models and "rush_tds" in prop_models and "rec_tds" in prop_models and player_form
        for (fid, player), g in lines.groupby(["fixture_id", "player_name"], sort=False):
            books = {}
            for r in g.itertuples():
                if r.book and r.price == r.price and abs(float(r.price)) >= 100:
                    books[r.book] = float(r.price)       # one price per book
            if not books:
                continue
            row = {"fixture_id": fid, "start_time": g["start_time"].iloc[0], "home_team": g["home_team"].iloc[0],
                   "away_team": g["away_team"].iloc[0], "player_name": player,
                   "books": [{"book": b, "label": BOOK_LABELS.get(b, b), "price": p, "offshore": b in OFFSHORE}
                             for b, p in sorted(books.items(), key=lambda kv: -payout(kv[1]))]}
            legal = {b: p for b, p in books.items() if b not in OFFSHORE}
            pool = legal or books
            best_book = max(pool, key=lambda b: payout(pool[b]))
            row.update(best_book=best_book, best_book_label=BOOK_LABELS.get(best_book, best_book),
                       best_price=pool[best_book], dk_price=books.get("draftkings"),
                       market_implied=round(implied(pool[best_book]), 4))
            entry = player_form.get(normalize_name(player)) if player_form else None
            if entry:
                row["team"], row["role"] = entry.get("team"), role(entry)
            if have_models and entry:
                def _proj(market):
                    # score_prop rounds to 0.1, too coarse for a touchdown
                    # rate, so read the unrounded projection when available.
                    if raw is not None:
                        raw.pop("value", None)
                    res = score_prop(player, market, 0.5, player_form, schedule_df, opp_defense_lookup,
                                     prop_models, env_lookup=env_lookup, opp_allowed_lookup=opp_allowed_lookup)
                    if not res:
                        return None, None
                    v = raw.get("value") if raw is not None and raw.get("value") is not None else res["model_predicted_value"]
                    return res, max(float(v), 0.0)
                rush, rush_v = _proj("Rush Touchdowns")
                rec, rec_v = _proj("Reception Touchdowns")
                if rush and rec:
                    p = 1 - math.exp(-(rush_v + rec_v))
                    if 0 < p < 1:
                        n_priced += 1
                        row.update(opponent=rush.get("opponent"), games_played=rush.get("games_played"),
                                   rush_td_proj=round(rush_v, 3), rec_td_proj=round(rec_v, 3),
                                   model_prob=round(p, 4), fair_odds=fair_american(p),
                                   edge_best=edge_pct(p, row["best_price"]),
                                   edge_dk=edge_pct(p, row["dk_price"]) if row["dk_price"] is not None else None)
            if injury_lookup and row.get("team"):
                inj = injury_lookup(row["team"], player)
                if inj:
                    row["injury_status"] = inj.get("status")
            if row.get("model_prob") is not None:
                if row.get("role") in CAUTION_ROLES:
                    row["tag"] = "caution"
                elif row.get("role") in LONGSHOT_ROLES:
                    row["tag"] = "longshot"
                elif row["edge_best"] >= VALUE_MIN_EDGE and row.get("injury_status") not in ("Out", "Out For Season", "IR", "Doubtful"):
                    row["tag"] = "value"
            out.append(row)
        out.sort(key=lambda r: (r.get("edge_best") is None, -(r.get("edge_best") or 0)))
        note = f"{len(out)} player(s) with anytime TD lines, {n_priced} priced by the model"
        return out, note
    except Exception as e:
        print(f"  [warn] anytime TD section skipped ({type(e).__name__}: {e})")
        return [], f"skipped ({type(e).__name__})"
