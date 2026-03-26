"""Benchmark: Architecture & Routing.

Compares structural and topological changes to the transformer blocks:
  - arch_baseline: Standard GPT block (no extras, relu_sq MLP)
  - Activations: leaky_relu_sq, gelu_sq, swiglu
  - Residuals: per-dim scale, residual mix, or both
  - Routing: U-Net skip connections, SmearGate
  - Experimental: XSA (Cross-Sample Attention), APB (Prune Binarization)
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

    # 1. Define the pure baseline (turn everything OFF)
    base = replace(
        bench_cfg,
        mlp_activation="relu_sq",
        use_residual_mix=False,
        use_per_dim_scale=False,
        use_unet=False,
        use_smear_gate=False,
        use_xsa=False,
        xsa_last_n=0,
        use_apb=False,
    )

    # 2. Build the master list of variants
    variants: list[tuple[str, BenchmarkConfig]] = [
        ("arch_baseline", base),
        # Activations
        ("act_leaky_relu_sq", replace(base, mlp_activation="leaky_relu_sq")),
        ("act_relu_sq", replace(base, mlp_activation="relu_sq")),
        ("act_gelu_sq", replace(base, mlp_activation="gelu_sq")),
        ("act_swiglu", replace(base, mlp_activation="swiglu")),
        # Residual Stream
        ("resid_per_dim_scale", replace(base, use_per_dim_scale=True)),
        ("resid_mix", replace(base, use_residual_mix=True)),
        (
            "resid_mix_scale",
            replace(base, use_residual_mix=True, use_per_dim_scale=True),
        ),
        # Advanced Routing
        ("unet_on", replace(base, use_unet=True)),
        ("smear_gate", replace(base, use_smear_gate=True)),
        # XSA / APB
        ("xsa_all", replace(base, use_xsa=True, xsa_last_n=model_cfg.num_layers)),
        ("apb_on", replace(base, use_apb=True, apb_prune_frac=0.05)),
        (
            "xsa_apb_combined",
            replace(
                base,
                use_xsa=True,
                xsa_last_n=model_cfg.num_layers,
                use_apb=True,
                apb_prune_frac=0.05,
            ),
        ),
    ]

    for name, cfg in variants:
        log(f"\n── Architecture: {name} ──")
        torch.manual_seed(cfg.seed)
        model = TinyGPT(model_cfg, cfg).to(device)

        if cfg.use_apb:
            model.setup_apb(prune_frac=cfg.apb_prune_frac)

        with VRAMTracker(device) as vt:
            result = run_micro_train(model, model_cfg, cfg, device, label=name)

        result.peak_vram_mb = vt.peak_mb
        results.append(result)
        del model
        torch.cuda.empty_cache()

    return results
