"""Benchmark: Embedding & Input Enrichment.

Compares early token enrichment strategies:
  - embed_baseline: standard token embeddings only
  - bigram_on: BigramHash local context embedding
  - ve_last_layer: Value Embedding injected into the last layer's V pathway
  - ve_last_two: Value Embedding injected into the last two layers
  - embed_combined: Both BigramHash and VE (last two layers) combined
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

    # Baseline with all embedding enrichment disabled
    base = replace(
        bench_cfg,
        use_bigram_hash=False,
        use_value_embed=False,
    )

    # Compute VE layer indices dynamically so they target the last layers
    # regardless of model depth (e.g. 2L, 3L, etc.).
    n = model_cfg.num_layers
    last_layer = str(n - 1)
    last_two = ",".join(str(i) for i in range(max(0, n - 2), n))

    # Variant configurations
    variants: list[tuple[str, BenchmarkConfig]] = [
        ("embed_baseline", base),
        # BigramHash
        (
            "bigram_on",
            replace(base, use_bigram_hash=True, bigram_vocab_size=256, bigram_dim=64),
        ),
        # Value Embeddings
        (
            "ve_last_layer",
            replace(base, use_value_embed=True, ve_dim=64, ve_layers=last_layer),
        ),
        (
            "ve_last_two",
            replace(base, use_value_embed=True, ve_dim=64, ve_layers=last_two),
        ),
        # Synergy Test (The power of a consolidated suite!)
        (
            "embed_combined",
            replace(
                base,
                use_bigram_hash=True,
                bigram_vocab_size=256,
                bigram_dim=64,
                use_value_embed=True,
                ve_dim=64,
                ve_layers=last_two,
            ),
        ),
    ]

    for name, cfg in variants:
        log(f"\n── Embeddings: {name} ──")
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
