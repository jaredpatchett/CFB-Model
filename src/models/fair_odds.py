"""
Fair-odds / EV math, plus a PRESEASON margin estimator.

Why this exists separately from game_model.py:
The 2026 season hasn't started yet (opens Aug 29), so every 2026 game has
zero in-season rolling data — home_games_played_prior and
away_games_played_prior are 0 for literally every team. The trained
GameMarginModel (src/models/game_model.py) was fit on rows where those
rolling features are almost always non-null and meaningfully non-zero
whenever games_played_prior > 0 — feeding it an out-of-distribution input
(rolling features blanked to 0/NaN while games_played_prior is also 0)
would be extrapolating outside anything it learned from, which is a good
way to get a confident-looking but meaningless number.

Instead, for the preseason window, this uses just SP+ rating differential
(CFBD's advanced team-strength rating, which is deliberately calibrated so
the rating gap between two teams approximates the expected point margin)
plus a home-field edge — both an empirically-fit home-field constant and the
prediction's residual std are derived from the real 2021-2025 historical
data in team_game_features.csv, not assumed. Once actual 2026 games are
played and rolling in-season features exist, the full GameMarginModel is the
better tool and this preseason estimator should stop being used for that
team/game.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_preseason_prior(team_game_features: pd.DataFrame) -> dict:
    """Empirically fit predicted_margin = slope * sp_rating_diff + intercept
    from real historical games (home_field == 1 only, since The Odds API
    doesn't expose a neutral-site flag for the current slate — a known
    simplification).

    IMPORTANT: this fits the slope rather than assuming it's 1.0. SP+ is
    *designed* to approximate point margin at roughly a 1:1 scale, but
    "roughly" isn't good enough when the output feeds real EV math — an
    early version of this function assumed slope=1 and residual_std came out
    unrealistically tight, which silently inflated EV%. A cheap way to catch
    this class of bug going forward: if a large fraction of games come back
    "BET" (very high EV), or the fitted slope isn't close to 1.0, something
    is off — recheck before trusting the numbers.
    """
    df = team_game_features.dropna(subset=["sp_rating_diff", "margin"])
    df = df[df["home_field"] == 1]
    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} historical games with SP+ data available to fit "
            f"the preseason prior — need more historical seasons pulled."
        )
    x = df["sp_rating_diff"].to_numpy(dtype=float)
    y = df["margin"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual_std = float(np.std(y - (slope * x + intercept)))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_std": residual_std,
        "n_games": len(df),
    }


def preseason_predicted_margin(sp_rating_diff: float, prior: dict, neutral_site: bool = False) -> float:
    """neutral_site=True zeroes out the fitted home-field constant, since
    that constant specifically represents home-field advantage (it was fit
    ONLY from real home games in fit_preseason_prior — neutral-site games
    are excluded from the fit itself). Applying it to a neutral-site game
    would misattribute a home-field edge to a team that doesn't have one.
    Defaults to False (i.e. assume a normal home game) because the current
    live game feed (The Odds API) doesn't expose a neutral-site flag itself
    — callers that have it from elsewhere (CFBD's /games endpoint does)
    should pass it through explicitly."""
    intercept = 0.0 if neutral_site else prior["intercept"]
    return prior["slope"] * sp_rating_diff + intercept


def margin_to_win_prob(margin: float, residual_std: float) -> float:
    if not residual_std or residual_std <= 0:
        raise ValueError("residual_std must be positive.")
    return float(norm.cdf(margin / residual_std))


def prob_to_american(prob: float) -> float:
    """Convert a win probability into the American odds a perfectly fair
    (no-vig) book would post for it."""
    prob = min(max(prob, 1e-6), 1 - 1e-6)
    if prob >= 0.5:
        return -100 * prob / (1 - prob)
    return 100 * (1 - prob) / prob


def american_to_implied_prob(odds: float) -> float:
    """Raw implied probability from an American odds price — includes the
    book's vig, NOT yet a 'fair' probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def devig_two_way(prob_a: float, prob_b: float) -> tuple:
    """Remove the vig from a two-sided market by normalizing so the two
    implied probabilities sum to 1, splitting the overround proportionally."""
    total = prob_a + prob_b
    if total <= 0:
        return None, None
    return prob_a / total, prob_b / total


def american_to_decimal(odds: float) -> float:
    return 1 + odds / 100 if odds > 0 else 1 + 100 / abs(odds)


def ev_percent(model_prob: float, book_odds: float) -> float:
    """Expected value, as a percentage of stake, of betting a side at the
    book's actual price if the model's probability is the true probability.
    Positive = the model thinks this is a +EV bet at the current price."""
    decimal_odds = american_to_decimal(book_odds)
    return (model_prob * decimal_odds - 1) * 100
