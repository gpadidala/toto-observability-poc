"""Forecast evaluation metrics used in the POC backtest."""

from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def smape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Symmetric MAPE in percent (robust to near-zero values)."""
    denom = np.abs(actual) + np.abs(pred)
    denom = np.where(denom == 0, 1.0, denom)
    return float(np.mean(2.0 * np.abs(actual - pred) / denom) * 100.0)


def interval_coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals inside the [lower, upper] band (target ~0.8 for q10–q90)."""
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside))


def evaluate(actual: np.ndarray, median: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> dict:
    return {
        "mae": round(mae(actual, median), 4),
        "rmse": round(rmse(actual, median), 4),
        "smape_pct": round(smape(actual, median), 4),
        "coverage_80pct_interval": round(interval_coverage(actual, q10, q90), 4),
    }
