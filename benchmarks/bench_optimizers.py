"""Benchmark: Optimizers & Update Rules.

Compares competing gradient-processing strategies:
  - muon_baseline: Standard Muon (Newton-Schulz orthogonalization)
  - muon_huber: Muon + Huber weight decay
  - muon_vs: Muon + Variance Normalization
  - muon_huber_vs: Both Muon extensions
  - mano_base: Oblique manifold projection (no Newton-Schulz)
  - mano_nesterov: Mano + Nesterov momentum
  - magma_tau1: Muon + Magma alignment damping (tau=1.0)
  - magma_tau2: Muon + Magma alignment damping (tau=2.0)
  - magma_tau4: Muon + Magma alignment damping (tau=4.0)
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

    # Define a strict baseline with all competing techniques disabled
    base = replace(
        bench_cfg,
        use_huber_decay=False,
        use_muon_vs=False,
        use_mano=False,
        use_magma=False,
    )

    variants: list[tuple[str, BenchmarkConfig]] = [
        # Baseline Muon
        ("muon_baseline", base),
        # Muon Extensions
        ("muon_huber", replace(base, use_huber_decay=True, huber_delta=0.5)),
        ("muon_vs", replace(base, use_muon_vs=True, muon_vs_beta2=0.999)),
        (
            "muon_huber_vs",
            replace(
                base,
                use_huber_decay=True,
                huber_delta=0.5,
                use_muon_vs=True,
                muon_vs_beta2=0.999,
            ),
        ),
        # Mano (Oblique Manifold)
        (
            "mano_base",
            replace(base, use_mano=True, mano_rescale=0.2, mano_nesterov=False),
        ),
        (
            "mano_nesterov",
            replace(base, use_mano=True, mano_rescale=0.2, mano_nesterov=True),
        ),
        # Magma (Alignment Damping)
        (
            "magma_tau1",
            replace(base, use_magma=True, magma_tau=1.0, magma_ema_decay=0.9),
        ),
        (
            "magma_tau2",
            replace(base, use_magma=True, magma_tau=2.0, magma_ema_decay=0.9),
        ),
        (
            "magma_tau4",
            replace(base, use_magma=True, magma_tau=4.0, magma_ema_decay=0.9),
        ),
    ]

    for name, cfg in variants:
        log(f"\n── Optimizers: {name} ──")
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
