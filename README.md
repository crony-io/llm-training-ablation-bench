# LLM training ablation benchmark

A/B experimentation framework for LLM training techniques on micro-scale models. Rapidly validate optimizers, architectures, training dynamics, embeddings, weight averaging, and quantization on consumer GPUs using synthetic data.

--

I created this project inspired by OpenAI's Parameter Golf Challenge to test different approaches and combinations, in order to easily identify what works and what doesn't. The code was written using a 3060 GPU with 12 GB of VRAM on Windows 11, but I believe it can be better adapted to other environments.


## Installation

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/crony-io/llm-training-ablation-bench.git
cd llm-training-ablation-bench

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
# source .venv/bin/activate

# Install PyTorch with CUDA support (pick your CUDA version)
# For CUDA 13.0:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu130

# For CUDA 12.8:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.6:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.4:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.1:
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8 (older GPUs):
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```
**Note**: Install PyTorch with CUDA support first (as shown above) before running `pip install -r requirements.txt`. The default PyTorch package from PyPI is CPU-only.

# Verify GPU is detected
```
python -c 'import torch; print("CUDA:", torch.cuda.is_available(), "GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")'
```

## Usage

### Run Benchmarks

```bash
# All benchmarks with default model
python -m runner

# Specific benchmark
python -m runner --bench optimizers

# Multiple benchmarks
python -m runner --bench optimizers,architecture,training_dynamics

# Use larger model
python -m runner --model micro_3L_96d

# Override training steps
python -m runner --steps 100

# Custom run ID (prepended to timestamp)
python -m runner --id experiment1

```

### Batch Script Generation (--commands)

The `--commands` flag generates ready-to-run commands for each benchmark individually. This is useful for:
- Running benchmarks across multiple GPUs/machines
- Creating shell scripts for overnight batch runs
- Running benchmarks in separate processes to isolate memory usage

```bash
# Generate commands with custom ID
python -m runner --commands --id experiment1

# Output:
# python -m runner --bench optimizers --id experiment1
# python -m runner --bench training_dynamics --id experiment1
# python -m runner --bench weight_averaging --id experiment1
# ... etc

# Save to file for batch execution
python -m runner --commands --id experiment1 > run_all.sh
# Then run: bash run_all.sh
```

You can combine with other options:
```bash
# Generate commands for specific benchmarks only
python -m runner --bench optimizers,architecture --commands --id test

# Generate with custom model and steps
python -m runner --model micro_3L_96d --steps 500 --commands --id large_model
```

### Plot Results

Charts are **auto-generated** at the end of each benchmark run if matplotlib is installed:

```bash
python -m plotter results/bench_20260101_120000.jsonl
```

If matplotlib is not installed, the benchmark still runs successfully - it simply skips chart generation with a notification. To enable plots: `pip install matplotlib`



## Project Structure

```
llm-training-ablation-bench/
├── benchmarks/              # Benchmark modules
│   ├── bench_optimizers.py      # Optimizer comparisons
│   ├── bench_architecture.py    # Architecture variations
│   ├── bench_training_dynamics.py  # LR/momentum/batch schedules
│   ├── bench_embeddings.py      # Embedding strategies
│   ├── bench_weight_averaging.py # EMA/SWA techniques
│   └── bench_quantization.py    # QAT and PTQ methods
├── config.py               # Model configs and BenchmarkConfig
├── bench_utils.py          # TinyGPT model, optimizers, training loop
├── runner.py               # Main benchmark orchestrator
├── logger.py               # Dual console/file logging
├── plotter.py              # Auto-chart generation
└── requirements.txt        # Dependencies (torch, matplotlib)
```

## Model Configurations

All configs target <500MB VRAM for desktop use:

| Config | Layers | Dim | Heads | Seq Len | Params |
|--------|--------|-----|-------|---------|--------|
| `micro_2L_64d` | 2 | 64 | 2 | 64 | ~90K |
| `micro_3L_96d` | 3 | 96 | 2 | 64 | ~243K |
| `small_3L_128d` | 3 | 128 | 4 | 128 | ~586K |

## Benchmarks

### Optimizers

Compares gradient-processing strategies via `benchmarks/bench_optimizers.py`:

| Variant | Description |
|---------|-------------|
| `muon_baseline` | Standard Muon (Newton-Schulz orthogonalization) |
| `muon_huber` | Muon + Huber weight decay |
| `muon_vs` | Muon + Variance Scaling normalization |
| `muon_huber_vs` | Both Muon extensions combined |
| `mano_base` | Oblique manifold projection (no Newton-Schulz) |
| `mano_nesterov` | Mano + Nesterov momentum |
| `magma_tau1` | Muon + Magma alignment damping (tau=1.0) |
| `magma_tau2` | Muon + Magma alignment damping (tau=2.0) |
| `magma_tau4` | Muon + Magma alignment damping (tau=4.0) |

### Architecture

Tests structural transformer changes via `benchmarks/bench_architecture.py`:

| Category | Variants |
|----------|----------|
| **Activations** | `relu_sq`, `leaky_relu_sq`, `gelu_sq`, `swiglu` |
| **Residuals** | Per-dim scale, residual mix, or both |
| **Routing** | U-Net skip connections, SmearGate |
| **Experimental** | XSA (Cross-Sample Attention), APB (Prune Binarization) |

### Training Dynamics

Compares scheduling techniques via `benchmarks/bench_training_dynamics.py`:

| Category | Variants |
|----------|----------|
| **LR Schedules** | Warmup, warmup+warmdown (30%/50%) |
| **Momentum** | Ramp 0.85→0.95, ramp 0.92→0.99, flat 0.99 |
| **Batch Size** | Warmup from 33% or 50% to 100% |

### Embeddings

Tests input enrichment via `benchmarks/bench_embeddings.py`:

| Variant | Description |
|---------|-------------|
| `embed_baseline` | Standard token embeddings only |
| `bigram_on` | BigramHash local context embedding |
| `ve_last_layer` | Value Embedding in last layer (`ve_layers="1"` for 2L model, `"2"` for 3L) |
| `ve_last_two` | Value Embedding in last two layers (`ve_layers="0,1"` for 2L) |
| `embed_combined` | BigramHash + VE in last two layers combined |

**Note**: `ve_layers` is a comma-separated string of 0-based layer indices. For a 2-layer model, layer indices are 0,1. For 3-layer: 0,1,2.

### Weight Averaging

Compares EMA/SWA via `benchmarks/bench_weight_averaging.py`:

| Variant | Description |
|---------|-------------|
| `wa_baseline` | No weight averaging |
| `ema_0.80` | EMA with 0.80 decay |
| `ema_0.90` | EMA with 0.90 decay |
| `ema_0.95` | EMA with 0.95 decay |
| `swa_35pct` | SWA starting at 35% of training |
| `swa_50pct` | SWA starting at 50% of training |

### Quantization

Compares QAT vs PTQ via `benchmarks/bench_quantization.py`:

| Variant | Description |
|---------|-------------|
| `qat_baseline` | FP16/BF16 training (no quantization) |
| `qat_int6` | QAT with int6 on all linear layers |
| `qat_int5_mlp` | QAT with int6 attention + int5 MLPs |
| `ptq_per_row_int6` | Post-training per-row int6 |
| `ptq_gptq_int6` | Post-training GPTQ int6 |

## How It Works

### Synthetic Data

The framework uses a predictable sine-wave pattern mapped to vocabulary indices - no dataset downloads required. This gives the model an actual pattern to learn, so better techniques achieve measurably lower loss.

### Model Architecture (TinyGPT)

Located in `bench_utils.py`:

```python
class TinyGPT(nn.Module):
    # Key features:
    # - GQA (Grouped Query Attention)
    # - RoPE positional encoding
    # - RMSNorm layer normalization
    # - QK-Norm with learned temperature
    # - Multiple activation options (SwiGLU, ReLU², GELU²)
    # - Tied or untied embeddings
    # - Logit softcapping
    # - XSA, APB, U-Net skips, SmearGate support
```

### Training Loop

Located in `bench_utils.py:run_micro_train()`:

1. **Parameter Groups**:
   - Matrix params (2D+): Muon/Mano/Magma optimizer
   - Scalar params (1D): AdamW
   - Embedding params: AdamW with higher LR

2. **Schedules**:
   - LR warmup/warmdown
   - Momentum warmup
   - Batch size warmup

3. **Techniques**:
   - Gradient clipping (selective - only non-matrix params)
   - EMA shadow model evaluation
   - SWA (Stochastic Weight Averaging)
   - QAT (Quantization-Aware Training via STE)

4. **Caching**: Identical configs reuse cached results

### Optimizers

**Muon** (`bench_utils.py:Muon`):
- Newton-Schulz 5-step orthogonalization
- Optional Huber weight decay
- Optional Variance Scaling (VS)

**Mano** (`bench_utils.py:Mano`):
- Oblique manifold projection
- No Newton-Schulz (faster)
- Optional Nesterov momentum

**Magma** (`bench_utils.py:MagmaMuon`):
- Muon base + alignment damping
- EMA of momentum-gradient cosine similarity
- Dampens updates when momentum and gradient misalign

## Configuration System

### TinyModelConfig (`config.py`)

```python
@dataclass
class TinyModelConfig:
    name: str              # e.g., "micro_2L_64d"
    num_layers: int        # Transformer layers
    model_dim: int         # Hidden dimension
    num_heads: int         # Attention heads
    num_kv_heads: int      # GQA KV heads
    vocab_size: int        # Token vocabulary
    mlp_mult: float        # MLP hidden multiplier
    seq_len: int           # Sequence length
    rope_base: float       # RoPE base frequency
    tie_embeddings: bool   # Tie input/output embeddings
    ln_scale: bool         # Learnable LayerNorm scale
```

### BenchmarkConfig (`config.py`)

```python
@dataclass
class BenchmarkConfig:
    train_steps: int       # Training steps (default: 200)
    batch_size: int        # Batch size (default: 64)
    matrix_lr: float        # Matrix param LR (default: 0.02)
    scalar_lr: float        # Scalar param LR (default: 0.02)
    embed_lr: float        # Embedding LR (default: 0.6)
    weight_decay: float    # Weight decay (default: 0.04)
    muon_momentum: float   # Momentum (default: 0.95)

    # Technique toggles (all default to False):
    use_huber_decay: bool
    use_muon_vs: bool
    use_mano: bool
    use_magma: bool
    use_ema: bool
    use_swa: bool
    use_qat: bool
    use_bigram_hash: bool
    use_value_embed: bool
    use_xsa: bool
    use_apb: bool
    use_unet: bool
    use_smear_gate: bool
    use_residual_mix: bool
    use_per_dim_scale: bool
    # ... and more
```

## Adding New Benchmarks

Create `benchmarks/bench_yours.py`:

```python
"""Benchmark: Your Technique Category."""

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

    # 1. Define strict baseline (all toggles OFF)
    base = replace(bench_cfg, use_your_technique=False)

    # 2. Define variants
    variants: list[tuple[str, BenchmarkConfig]] = [
        ("baseline", base),
        ("variant1", replace(base, use_your_technique=True, param=value)),
    ]

    # 3. Run each variant
    for name, cfg in variants:
        log(f"\n── YourCategory: {name} ──")
        torch.manual_seed(cfg.seed)
        model = TinyGPT(model_cfg, cfg).to(device)

        with VRAMTracker(device) as vt:
            result = run_micro_train(model, model_cfg, cfg, device, label=name)
        result.peak_vram_mb = vt.peak_mb
        results.append(result)
        del model
        torch.cuda.empty_cache()

    return results
```

If your technique needs new config toggles, add to `BenchmarkConfig` in `config.py`:

```python
@dataclass
class BenchmarkConfig:
    # ... existing fields
    use_your_technique: bool = False
    your_param: float = 0.5
```

Test with:
```bash
python -m runner --bench yours --steps 30
```

## Results & Visualization

### Output Files

Each run generates:
- `results/bench_{timestamp}.jsonl` — Structured results data
- `results/bench_{timestamp}.log` — Full console output
- `results/bench_{timestamp}_plots/*.png` — Loss curve charts

### JSONL Format

```jsonl
{"_type": "header", "timestamp": "...", "gpu": "...", "model": "...", "benchmarks": [...]}
{"_type": "result", "benchmark": "optimizers", "variant": "muon_baseline", "final_loss": 2.45, ...}
{"_type": "result", "benchmark": "optimizers", "variant": "muon_huber", "final_loss": 2.38, ...}
{"_type": "footer", "total_time_s": 123.4, "timestamp": "..."}
```

### Metrics Tracked

- `final_loss` — Last training step loss
- `best_loss` — Best loss achieved
- `avg_step_ms` — Average time per training step
- `peak_vram_mb` — Peak GPU memory usage
- `total_params` — Model parameter count
- `loss_curve` — Full loss history

## Troubleshooting

### CUDA Out of Memory

Use a smaller model:
```bash
python -m runner --model micro_2L_64d
```

### PyTorch Not Finding CUDA

Reinstall PyTorch with correct CUDA version:
```bash
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu128
```

Verify with:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ with CUDA
- (Optional) matplotlib for charts

## License

MIT

---

**Note**: This is a research tool for rapid technique validation. Results indicate relative performance and trends, not absolute training quality. For production training, scale up model size and use real datasets.
