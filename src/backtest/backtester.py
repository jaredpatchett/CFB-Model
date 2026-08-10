"""
Backtesting harness: evaluate model predictions against what actually happened
and against the market's own closing lines. Beating your own predicted MAE is
easy; beating the closing line is the real bar, and this is designed around
that distinction.

Metrics reported:
  Spread (ATS - against the spread):
    - ATS win rate (predicted lean vs. actual result, using CFBD historical
      closing spread)
    - Mean absolute error of predicted margin vs actual margin
  Moneyline:
    - Accuracy (favorite/underdog pick vs actual winner)
    - Log loss (calibration of predicted win probabilities)
  Props:
    - Over/under hit rate vs actual result
    - MAE of predicted stat value vs actual stat value

A win rate of ~52.4%+ ATS over a meaningful sample (100+ games) is the
break-even threshold against standard -110 pricing — treat anything below
that as "not yet beating the market," not as a bug.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, mean_absolute_error


def evaluate_spread(predicted_margin: pd.Series, actual_margin: pd.Series,
                     market_home_spread: pd.Series) -> dict:
    """market_home_spread: negative = home favored (standard book convention)."""
    df = pd.DataFrame({
        "predicted_margin": predicted_margin,
        "actual_margin": actual_margin,
        "market_spread": market_home_spread,
    }).dropna()

    df["model_lean_home"] = df["predicted_margin"] > -df["market_spread"]
    df["home_covered"] = (df["actual_margin"] + df["market_spread"]) > 0
    df["push"] = (df["actual_margin"] + df["market_spread"]) == 0
    graded = df[~df["push"]]
    df["ats_correct"] = graded["model_lean_home"] == graded["home_covered"]

    ats_win_rate = df["ats_correct"].mean() if len(graded) else float("nan")
    mae = mean_absolute_error(df["actual_margin"], df["predicted_margin"])

    return {
        "n_games": len(df),
        "n_pushes": int(df["push"].sum()),
        "ats_win_rate": ats_win_rate,
        "margin_mae": mae,
        "breakeven_ats_rate": 0.524,
        "beat_market": (ats_win_rate > 0.524) if not np.isnan(ats_win_rate) else None,
    }


def evaluate_moneyline(predicted_home_win_prob: pd.Series, actual_home_win: pd.Series) -> dict:
    df = pd.DataFrame({
        "pred_prob": predicted_home_win_prob, "actual": actual_home_win
    }).dropna()
    preds_binary = (df["pred_prob"] > 0.5).astype(int)
    return {
        "n_games": len(df),
        "accuracy": accuracy_score(df["actual"], preds_binary),
        "log_loss": log_loss(df["actual"], df["pred_prob"], labels=[0, 1]),
    }


def evaluate_props(predicted_value: pd.Series, actual_value: pd.Series,
                    prop_line: pd.Series) -> dict:
    df = pd.DataFrame({
        "predicted": predicted_value, "actual": actual_value, "line": prop_line
    }).dropna()
    df["model_lean_over"] = df["predicted"] > df["line"]
    df["actual_over"] = df["actual"] > df["line"]
    df["push"] = df["actual"] == df["line"]
    graded = df[~df["push"]]
    hit_rate = (graded["model_lean_over"] == graded["actual_over"]).mean() if len(graded) else float("nan")

    return {
        "n_props": len(df),
        "n_pushes": int(df["push"].sum()),
        "hit_rate": hit_rate,
        "mae": mean_absolute_error(df["actual"], df["predicted"]),
        "breakeven_hit_rate": 0.524,
        "beat_market": (hit_rate > 0.524) if not np.isnan(hit_rate) else None,
    }


def summarize(results: dict, label: str):
    print(f"\n=== {label} ===")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
