"""Benchmark: Training Dynamics & Schedules.

Compares scheduling techniques across the "Holy Trinity" of training dynamics:
  - td_baseline: Flat LR, constant batch size, constant momentum 0.95

  # Learning Rate
  - lr_warmup: Linear warmup then flat
  - lr_warmup_wd30: Warmup + cosine warmdown in last 30% of steps
  - lr_warmup_wd50: Warmup + cosine warmdown in last 50% of steps

  # Momentum
  - mom_ramp_85_95: Ramp 0.85 → 0.95 over 30 steps
  - mom_ramp_92_99: Ramp 0.92 → 0.99 over 30 steps (v3 default)
  - mom_flat_0.99: Constant high momentum 0.99

  # Batch Size
  - bw_33pct_50s: Ramp batch from 33% to 100% over 50 steps
  - bw_50pct_50s: Ramp batch from 50% to 100% over 50 steps
  - bw_33pct_100s: Ramp batch from 33% to 100% over 100 steps
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

    # Baseline with no warmups and flat dynamics
    base = replace(
        bench_cfg,
        use_lr_warmup=False,
        use_lr_warmdown=False,
        use_momentum_warmup=False,
        muon_momentum=0.95,
        use_batch_warmup=False,
    )

    # Variant configurations
    variants: list[tuple[str, BenchmarkConfig]] = [
        ("td_baseline", base),
        # LR Schedules (Assuming global lr_warmup_steps is 20)
        ("lr_warmup", replace(base, use_lr_warmup=True, use_lr_warmdown=False)),
        (
            "lr_warmup_wd30",
            replace(
                base, use_lr_warmup=True, use_lr_warmdown=True, lr_warmdown_frac=0.3
            ),
        ),
        (
            "lr_warmup_wd50",
            replace(
                base, use_lr_warmup=True, use_lr_warmdown=True, lr_warmdown_frac=0.5
            ),
        ),
        # Momentum Warmup
        (
            "mom_ramp_85_95",
            replace(
                base,
                use_momentum_warmup=True,
                momentum_warmup_start=0.85,
                momentum_warmup_steps=30,
            ),
        ),
        (
            "mom_ramp_92_99",
            replace(
                base,
                muon_momentum=0.99,
                use_momentum_warmup=True,
                momentum_warmup_start=0.92,
                momentum_warmup_steps=30,
            ),
        ),
        ("mom_flat_0.99", replace(base, muon_momentum=0.99)),
        # Batch Warmup
        (
            "bw_33pct_50s",
            replace(
                base,
                use_batch_warmup=True,
                batch_warmup_steps=50,
                batch_warmup_start_frac=0.333,
            ),
        ),
        (
            "bw_50pct_50s",
            replace(
                base,
                use_batch_warmup=True,
                batch_warmup_steps=50,
                batch_warmup_start_frac=0.5,
            ),
        ),
        (
            "bw_33pct_100s",
            replace(
                base,
                use_batch_warmup=True,
                batch_warmup_steps=100,
                batch_warmup_start_frac=0.333,
            ),
        ),
    ]

    for name, cfg in variants:
        log(f"\n── Training Dynamics: {name} ──")
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
