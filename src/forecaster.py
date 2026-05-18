"""Thin wrapper around the Toto 2.0 foundation model.

Toto's `forecast()` takes a dict with `target`, `target_mask`, `series_ids`
and returns a quantile tensor of shape (Q, batch, n_var, horizon) where Q=9
for quantile levels [0.1, 0.2, ..., 0.9].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MEDIAN_IDX = 4   # 0.5
Q10_IDX = 0      # 0.1
Q90_IDX = 8      # 0.9


@dataclass
class ForecastResult:
    median: np.ndarray   # (n_var, horizon)
    q10: np.ndarray      # (n_var, horizon)
    q90: np.ndarray      # (n_var, horizon)
    quantiles: np.ndarray  # (Q, n_var, horizon)


class TotoForecaster:
    def __init__(self, size: str = "22m", device: str | None = None):
        # Imported lazily so the module can be inspected without torch+toto.
        from toto2 import Toto2Model

        self.size = size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = f"Datadog/Toto-2.0-{size}"
        self.model = Toto2Model.from_pretrained(checkpoint, map_location=self.device)
        self.model = self.model.to(self.device).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def forecast(self, context: np.ndarray, horizon: int) -> ForecastResult:
        """Forecast `horizon` steps from `context` of shape (n_var, ctx_len)."""
        n_var = context.shape[0]
        target = torch.tensor(context, dtype=torch.float32, device=self.device)
        target = target.unsqueeze(0)  # (1, n_var, ctx_len)
        target_mask = torch.ones_like(target, dtype=torch.bool)
        series_ids = torch.zeros(1, n_var, dtype=torch.long, device=self.device)

        quantiles = self.model.forecast(
            {"target": target, "target_mask": target_mask, "series_ids": series_ids},
            horizon=horizon,
        )  # (Q, 1, n_var, horizon)

        q = quantiles[:, 0].cpu().numpy()  # (Q, n_var, horizon)
        return ForecastResult(
            median=q[MEDIAN_IDX],
            q10=q[Q10_IDX],
            q90=q[Q90_IDX],
            quantiles=q,
        )
