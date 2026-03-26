"""Tiny model configurations for local benchmarking on consumer GPUs.

All configs target <500MB VRAM.
Synthetic random tokens — no FineWeb download required.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TinyModelConfig:
    """Model architecture config — deliberately minimal for technique testing.

    These are NOT training configs. They're just big enough to exercise each
    technique's code path and measure relative overhead/benefit. VRAM target
    is <500MB so Windows can keep running the desktop, browser, etc.
    """

    name: str = "micro_2L_64d"
    num_layers: int = 2
    model_dim: int = 64
    num_heads: int = 2
    num_kv_heads: int = 1
    vocab_size: int = 512
    mlp_mult: float = 2.0
    seq_len: int = 64
    rope_base: float = 10000.0
    rope_dims: int = 8
    logit_softcap: float = 30.0
    qk_gain_init: float = 1.5
    tie_embeddings: bool = True
    ln_scale: bool = True


@dataclass
class BenchmarkConfig:
    """Benchmark run settings — ultra-light for quick A/B technique tests."""

    # Budget: just enough steps to see loss move and compare techniques.
    train_steps: int = 200
    warmup_steps: int = 5
    batch_size: int = 64  # tiny — we're testing techniques, not training
    grad_accum_steps: int = 1
    # Log every step by default so benchmark discrepancies cannot hide
    # behind sparse sampling.
    log_every: int = 20

    # Optimizer defaults.
    matrix_lr: float = 0.02
    scalar_lr: float = 0.02
    embed_lr: float = 0.6
    weight_decay: float = 0.04
    muon_momentum: float = 0.95
    muon_backend_steps: int = 3  # fewer NS steps for speed
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip_norm: float = 0.3

    # Technique toggles — benchmarks override these per-test
    use_xsa: bool = False
    xsa_last_n: int = 0
    use_apb: bool = False
    apb_prune_frac: float = 0.05
    use_ema: bool = False
    ema_decay: float = 0.997
    use_huber_decay: bool = False
    huber_delta: float = 0.5
    use_muon_vs: bool = False
    muon_vs_beta2: float = 0.999
    use_batch_warmup: bool = False
    batch_warmup_steps: int = 20
    batch_warmup_start_frac: float = 0.333
    use_gptq: bool = False
    gptq_calib_steps: int = 4
    gptq_blocksize: int = 64
    gptq_damp: float = 0.01

    # LR schedule.
    use_lr_warmup: bool = True
    lr_warmup_steps: int = 20
    use_lr_warmdown: bool = True
    lr_warmdown_frac: float = 1.0  # Fraction of steps to decay LR

    # Momentum warmup.
    use_momentum_warmup: bool = False
    momentum_warmup_start: float = 0.85
    momentum_warmup_steps: int = 30

    # MLP activation variant.
    mlp_activation: str = "relu_sq"  # "relu_sq", "leaky_relu_sq", "gelu_sq", "swiglu"

    # Residual mixing.
    use_residual_mix: bool = False
    use_per_dim_scale: bool = False

    # SmearGate.
    use_smear_gate: bool = False

    # Mano optimizer (Oblique manifold).
    use_mano: bool = False
    mano_rescale: float = 0.2
    mano_nesterov: bool = False

    # Magma alignment damping.
    use_magma: bool = False
    magma_tau: float = 2.0
    magma_ema_decay: float = 0.9

    # U-net skip connections.
    use_unet: bool = False

    # BigramHash embedding.
    use_bigram_hash: bool = False
    bigram_vocab_size: int = 1024
    bigram_dim: int = 32

    # Value Embedding (VE) — injected into attention V pathway.
    use_value_embed: bool = False
    ve_dim: int = 32
    ve_layers: str = ""  # comma-separated layer indices, e.g. "0,1"

    # QAT/STE (quantization-aware training via straight-through estimator).
    use_qat: bool = False
    qat_clip_attn: int = 31  # int6 for attention
    qat_clip_mlp: int = 15  # int5 for MLP

    # SWA (Stochastic Weight Averaging).
    use_swa: bool = False
    swa_start_frac: float = 0.35
    swa_every: int = 5

    # Seed for reproducibility across A/B comparisons.
    seed: int = 42


# ── Pre-built configs ──────────────────────────────────────────────────────────
# All configs target <500MB VRAM so Windows desktop + browser stay responsive.

MICRO_2L_64D = TinyModelConfig(
    name="micro_2L_64d",
    num_layers=2,
    model_dim=64,
    num_heads=2,
    num_kv_heads=1,
    seq_len=64,
)

MICRO_3L_96D = TinyModelConfig(
    name="micro_3L_96d",
    num_layers=3,
    model_dim=96,
    num_heads=2,
    num_kv_heads=1,
    seq_len=64,
)

SMALL_3L_128D = TinyModelConfig(
    name="small_3L_128d",
    num_layers=3,
    model_dim=128,
    num_heads=4,
    num_kv_heads=2,
    seq_len=128,
    mlp_mult=2.0,
)

ALL_MODEL_CONFIGS: dict[str, TinyModelConfig] = {
    "micro_2L_64d": MICRO_2L_64D,
    "micro_3L_96d": MICRO_3L_96D,
    "small_3L_128d": SMALL_3L_128D,
}

DEFAULT_MODEL = "micro_2L_64d"
DEFAULT_BENCH = BenchmarkConfig()
