"""Generate synthetic observability metrics with realistic structure.

Each metric mimics a signal you would actually scrape from Prometheus / a
Grafana LGTM stack: a baseline level, a daily (24h) seasonality, a slow
trend, gaussian noise, and occasional incident-like spikes.

The series are returned at 1-sample-per-hour resolution so a 24-step cycle
is one day and a 96-step horizon is four days.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    baseline: float
    daily_amplitude: float
    trend_per_day: float
    noise_std: float
    spike_prob: float        # per-step probability of an incident spike
    spike_magnitude: float   # spike size relative to baseline
    floor: float = 0.0       # values are clipped at this lower bound


# Three correlated-looking observability signals.
DEFAULT_METRICS: list[MetricSpec] = [
    MetricSpec(
        name="cpu_utilization_pct",
        unit="%",
        baseline=45.0,
        daily_amplitude=18.0,
        trend_per_day=0.4,
        noise_std=2.5,
        spike_prob=0.012,
        spike_magnitude=0.6,
        floor=0.0,
    ),
    MetricSpec(
        name="request_latency_p95_ms",
        unit="ms",
        baseline=120.0,
        daily_amplitude=40.0,
        trend_per_day=1.2,
        noise_std=8.0,
        spike_prob=0.015,
        spike_magnitude=1.4,
        floor=1.0,
    ),
    MetricSpec(
        name="request_rate_rps",
        unit="req/s",
        baseline=850.0,
        daily_amplitude=380.0,
        trend_per_day=3.0,
        noise_std=22.0,
        spike_prob=0.008,
        spike_magnitude=0.5,
        floor=0.0,
    ),
]

STEPS_PER_DAY = 24


def generate_metric(spec: MetricSpec, length: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single metric series of `length` hourly samples."""
    t = np.arange(length, dtype=np.float64)

    # Daily seasonality: peak in the working part of the day.
    season = spec.daily_amplitude * np.sin(2 * np.pi * (t % STEPS_PER_DAY) / STEPS_PER_DAY)
    # A weaker weekly ripple so the model has multi-scale structure to learn.
    weekly = 0.3 * spec.daily_amplitude * np.sin(2 * np.pi * t / (STEPS_PER_DAY * 7))

    trend = spec.trend_per_day * (t / STEPS_PER_DAY)
    noise = rng.normal(0.0, spec.noise_std, size=length)

    series = spec.baseline + season + weekly + trend + noise

    # Inject incident-like spikes (a burst that decays over a few steps).
    spikes = rng.random(length) < spec.spike_prob
    for idx in np.flatnonzero(spikes):
        decay = np.exp(-np.arange(6) / 2.0)
        bump = spec.baseline * spec.spike_magnitude * decay
        end = min(idx + len(bump), length)
        series[idx:end] += bump[: end - idx]

    return np.clip(series, spec.floor, None)


def generate_dataset(
    metrics: list[MetricSpec] | None = None,
    length: int = 608,
    seed: int = 42,
) -> tuple[np.ndarray, list[MetricSpec]]:
    """Return an (n_var, length) array of synthetic observability metrics.

    Default length 608 = 512-step context + 96-step (4-day) forecast horizon.
    """
    metrics = metrics or DEFAULT_METRICS
    rng = np.random.default_rng(seed)
    data = np.stack([generate_metric(m, length, rng) for m in metrics], axis=0)
    return data.astype(np.float32), metrics
