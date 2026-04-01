"""Benchmark: Weight Averaging Techniques.

Compares Exponential Moving Average (EMA) and Stochastic Weight Averaging (SWA):
  - wa_baseline: no weight averaging
  - ema_0.80: EMA with 0.80 decay
  - ema_0.90: EMA with 0.90 decay
  - ema_0.95: EMA with 0.95 decay
  - swa_35pct: SWA starting at 35% of training
  - swa_50pct: SWA starting at 50% of training
"""

from __future__ import annotations

from dataclasses import replace

import torch

from bench_utils import BenchmarkResult, TinyGPT, VRAMTracker, run_micro_train
from config import BenchmarkConfig, TinyModelConfig
from logger import log


def run(
    device: torch.device,
    model_cfg: TinyModelConfig,
    bench_cfg: BenchmarkConfig,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    # Baseline with no weight averaging
    base = replace(
        bench_cfg,
        use_ema=False,
        use_swa=False,
    )

    # Variant configurations
    variants: list[tuple[str, BenchmarkConfig]] = [
        ("wa_baseline", base),
        # EMA variants
        ("ema_0.80", replace(base, use_ema=True, ema_decay=0.80)),
        ("ema_0.90", replace(base, use_ema=True, ema_decay=0.90)),
        ("ema_0.95", replace(base, use_ema=True, ema_decay=0.95)),
        # SWA variants
        ("swa_35pct", replace(base, use_swa=True, swa_start_frac=0.35, swa_every=5)),
        ("swa_50pct", replace(base, use_swa=True, swa_start_frac=0.50, swa_every=5)),
    ]

    for name, cfg in variants:
        log(f"\n── Weight Averaging: {name} ──")
        torch.manual_seed(cfg.seed)
        with VRAMTracker(device) as vt:
            model = TinyGPT(model_cfg, cfg).to(device)
            result = run_micro_train(model, model_cfg, cfg, device, label=name)
        if not result.cached:
            result.peak_vram_mb = vt.peak_mb
        results.append(result)
        del model
        torch.cuda.empty_cache()

    return results
