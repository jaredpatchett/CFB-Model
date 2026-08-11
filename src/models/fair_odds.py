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
    """Empirically derive the home-field edge and residual std of
    (actual margin - sp_rating_diff) from real historical games, restricted
    to non-neutral-site games (home_field == 1) since that's what we'll
    assume for the current slate (The Odds API doesn't expose a neutral-site
    flag, so this is a known simplification)."""
    df = team_game_features.dropna(subset=["sp_rating_diff", "margin"])
    df = df[df["home_field"] == 1]
    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} historical games with SP+ data available to fit "
            f"the preseason prior — need more historical seasons pulled."
        )
    residual = df["margin"] - df["sp_rating_diff"]
    home_field_adv = float(residual.mean())
    residual_std = float((residual - home_field_adv).std())
    return {
        "home_field_adv": home_field_adv,
        "residual_std": residual_std,
        "n_games": len(df),
    }


def preseason_predicted_margin(sp_rating_diff: float, prior: dict) -> float:
    return sp_rating_diff + prior["home_field_adv"]


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
