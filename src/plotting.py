"""Plot actual-vs-forecast with the 80% prediction interval."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt
import numpy as np


def plot_forecast(
    name: str,
    unit: str,
    context: np.ndarray,
    actual: np.ndarray,
    median: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
    out_path: Path,
    context_tail: int = 168,
) -> None:
    ctx = context[-context_tail:]
    horizon = len(actual)
    x_ctx = np.arange(len(ctx))
    x_fc = np.arange(len(ctx), len(ctx) + horizon)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(x_ctx, ctx, color="black", lw=1.0, label="History (context)")
    ax.plot(x_fc, actual, color="tab:green", lw=1.6, label="Actual (held-out)")
    ax.plot(x_fc, median, color="tab:blue", lw=1.8, label="Toto median forecast")
    ax.fill_between(
        x_fc, q10, q90, color="tab:blue", alpha=0.20, label="80% interval (q10–q90)"
    )
    ax.axvline(len(ctx) - 1, color="grey", ls="--", lw=0.8)
    ax.set_title(f"{name}  —  Toto 2.0 forecast vs actual")
    ax.set_xlabel("Hours")
    ax.set_ylabel(unit)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
