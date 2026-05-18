"""End-to-end Toto 2.0 observability-forecasting POC.

Pipeline:
  1. Generate synthetic observability metrics (CPU, p95 latency, request rate).
  2. Split each series into a context window and a held-out forecast horizon.
  3. Forecast the horizon multivariately with the Toto 2.0 foundation model.
  4. Score the forecast (MAE / RMSE / sMAPE / 80% interval coverage).
  5. Save per-metric plots and a JSON report under outputs/.

Run:  python run_poc.py --size 22m --horizon 96
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.metrics import evaluate
from src.plotting import plot_forecast
from src.synthetic import generate_dataset

OUTPUT_DIR = Path(__file__).parent / "outputs"


def main() -> None:
    parser = argparse.ArgumentParser(description="Toto 2.0 observability POC")
    parser.add_argument("--size", default="22m",
                        choices=["4m", "22m", "313m", "1B", "2.5B"],
                        help="Toto 2.0 model size (default: 22m)")
    parser.add_argument("--context", type=int, default=512,
                        help="Context window length in hourly steps")
    parser.add_argument("--horizon", type=int, default=96,
                        help="Forecast horizon in hourly steps (96 = 4 days)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    total_len = args.context + args.horizon

    print(f"[1/5] Generating synthetic observability metrics "
          f"({total_len} hourly steps)...")
    data, specs = generate_dataset(length=total_len, seed=args.seed)
    context = data[:, : args.context]            # (n_var, context)
    actual = data[:, args.context :]             # (n_var, horizon)
    print(f"      metrics: {[s.name for s in specs]}")

    print(f"[2/5] Loading Toto-2.0-{args.size} (first run downloads the "
          f"checkpoint from HuggingFace)...")
    from src.forecaster import TotoForecaster  # lazy: needs torch + toto2

    t0 = time.time()
    fc = TotoForecaster(size=args.size)
    print(f"      loaded {fc.n_params:,} params on {fc.device} "
          f"in {time.time() - t0:.1f}s")

    print(f"[3/5] Forecasting {args.horizon} steps ahead "
          f"for {context.shape[0]} metrics...")
    t0 = time.time()
    result = fc.forecast(context, horizon=args.horizon)
    infer_s = time.time() - t0
    print(f"      inference took {infer_s:.2f}s")

    print("[4/5] Scoring forecast against held-out actuals...")
    report: dict = {
        "model": f"Datadog/Toto-2.0-{args.size}",
        "device": fc.device,
        "n_params": fc.n_params,
        "context_steps": args.context,
        "horizon_steps": args.horizon,
        "inference_seconds": round(infer_s, 3),
        "metrics": {},
    }
    for i, spec in enumerate(specs):
        scores = evaluate(actual[i], result.median[i], result.q10[i], result.q90[i])
        report["metrics"][spec.name] = {"unit": spec.unit, **scores}
        print(f"      {spec.name:>26} | MAE={scores['mae']:>9.3f} "
              f"sMAPE={scores['smape_pct']:>6.2f}% "
              f"cov80={scores['coverage_80pct_interval']:.2f}")

    print("[5/5] Writing plots + report to outputs/ ...")
    for i, spec in enumerate(specs):
        plot_forecast(
            name=spec.name,
            unit=spec.unit,
            context=context[i],
            actual=actual[i],
            median=result.median[i],
            q10=result.q10[i],
            q90=result.q90[i],
            out_path=OUTPUT_DIR / f"forecast_{spec.name}.png",
        )

    agg = {
        "mean_smape_pct": round(
            float(np.mean([m["smape_pct"] for m in report["metrics"].values()])), 4
        ),
        "mean_coverage_80pct": round(
            float(np.mean([m["coverage_80pct_interval"]
                           for m in report["metrics"].values()])), 4
        ),
    }
    report["aggregate"] = agg
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    print("\nDone. Aggregate:")
    print(f"  mean sMAPE         : {agg['mean_smape_pct']:.2f}%")
    print(f"  mean 80% coverage  : {agg['mean_coverage_80pct']:.2f}  (ideal ~0.80)")
    print(f"  artifacts          : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
