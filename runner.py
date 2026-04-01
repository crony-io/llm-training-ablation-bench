"""Local Benchmark Runner.

Results are streamed to a JSONL file as each variant completes — no RAM
accumulation. The file is created at benchmark start so you can watch it grow.

Usage:
    python -m runner                           # Run all benchmarks
    python -m runner --bench optimizer         # Single benchmark
    python -m runner --bench optimizer,ema     # Multiple benchmarks
    python -m runner --model micro_3L_96d      # Different model size
    python -m runner --steps 30                # Override training steps
    python -m runner --list                    # List available options

All benchmarks use synthetic random tokens — no FineWeb download required.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from bench_utils import BenchmarkResult
from config import ALL_MODEL_CONFIGS, DEFAULT_MODEL, BenchmarkConfig, TinyModelConfig
from cuda_memory import configure_cuda_memory, log_cuda_memory_config
from logger import close_log, init_log, log


# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARK_DIR = Path(__file__).parent / "benchmarks"

AVAILABLE_BENCHMARKS = {
    "optimizers": "benchmarks.bench_optimizers",
    "training_dynamics": "benchmarks.bench_training_dynamics",
    "weight_averaging": "benchmarks.bench_weight_averaging",
    "architecture": "benchmarks.bench_architecture",
    "embeddings": "benchmarks.bench_embeddings",
    "quantization": "benchmarks.bench_quantization",
}


# ── Param Estimation ──────────────────────────────────────────────────────────


def estimate_param_count(
    model_cfg: TinyModelConfig,
    mlp_activation: str = "relu_sq",
) -> int:
    """Estimate total trainable parameters for a model + benchmark config pair.

    Covers the base model architecture (embedding, attention, MLP, norms,
    per-head gain). Optional technique modules (SmearGate, U-Net skips,
    BigramHash, ValueEmbedding, etc.) are excluded since they depend on
    BenchmarkConfig toggles — the actual count from model.param_count()
    is used in BenchmarkResult for precise reporting.

    Args:
        model_cfg: Model architecture configuration.
        mlp_activation: MLP activation name from BenchmarkConfig (determines
            whether SwiGLU's 3-matrix MLP is used).

    Returns:
        Estimated total parameter count.
    """
    head_dim = model_cfg.model_dim // model_cfg.num_heads
    mlp_hidden = int(model_cfg.model_dim * model_cfg.mlp_mult)
    embed_params = model_cfg.vocab_size * model_cfg.model_dim
    head_params = (
        0 if model_cfg.tie_embeddings else (model_cfg.vocab_size * model_cfg.model_dim)
    )
    # Attention: Q + K (GQA) + V (GQA) + output projection
    attn_params = model_cfg.model_dim * (
        model_cfg.num_heads * head_dim
        + model_cfg.num_kv_heads * head_dim
        + model_cfg.num_kv_heads * head_dim
        + model_cfg.num_heads * head_dim
    )
    # Per-head learned temperature (q_gain): num_heads per layer
    attn_params += model_cfg.num_heads
    # SwiGLU has 3 matrices (c_fc, c_gate, c_proj), others have 2
    mlp_matrices = 3 if mlp_activation == "swiglu" else 2
    mlp_params = mlp_matrices * mlp_hidden * model_cfg.model_dim
    # RMSNorm scales: 2 per block (attn_norm + mlp_norm) + 1 final_norm
    norm_params = model_cfg.model_dim if model_cfg.ln_scale else 0
    per_block_norm = 2 * model_cfg.model_dim if model_cfg.ln_scale else 0
    return (
        embed_params
        + head_params
        + model_cfg.num_layers * (attn_params + mlp_params + per_block_norm)
        + norm_params
    )


# ── Discovery ──────────────────────────────────────────────────────────────────


def discover_benchmarks() -> dict[str, str]:
    """Return dict of benchmark_name -> module_path for all bench_*.py files."""
    found = dict(AVAILABLE_BENCHMARKS)
    for p in BENCHMARK_DIR.glob("bench_*.py"):
        name = p.stem.removeprefix("bench_")
        mod_path = f"benchmarks.{p.stem}"
        if name not in found:
            found[name] = mod_path
    return found


# ── Disk I/O ───────────────────────────────────────────────────────────────────


def _init_results_file(
    results_path: Path,
    gpu_name: str,
    model_cfg_name: str,
    bench_names: list[str],
    bench_cfg: BenchmarkConfig,
) -> None:
    """Create the JSONL results file with a header line."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "_type": "header",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu_name,
        "model": model_cfg_name,
        "benchmarks": bench_names,
        "train_steps": bench_cfg.train_steps,
        "batch_size": bench_cfg.batch_size,
        "seed": bench_cfg.seed,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")


def _append_result(
    results_path: Path, bench_name: str, result: BenchmarkResult
) -> None:
    """Append one result to the JSONL file. Flushes immediately."""
    entry = {"_type": "result", "benchmark": bench_name, **result.to_dict()}
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _append_footer(results_path: Path, total_elapsed_s: float) -> None:
    """Append a footer line marking the run as complete."""
    footer = {
        "_type": "footer",
        "total_time_s": round(total_elapsed_s, 1),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(footer) + "\n")


def _load_results(results_path: Path) -> dict[str, list[dict]]:
    """Read results JSONL, grouped by benchmark name.

    Silently skips malformed lines to stay robust against partial/corrupt files.
    """
    grouped: dict[str, list[dict]] = {}
    with open(results_path, encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("_type") == "result":
                bench = entry["benchmark"]
                grouped.setdefault(bench, []).append(entry)
    return grouped


# ── Report ─────────────────────────────────────────────────────────────────────


def _print_table_from_dicts(rows: list[dict], title: str) -> None:
    """Log a comparison table from serialized result dicts."""
    if not rows:
        log("  (no results)")
        return

    log(f"\n{'=' * 90}")
    log(f"  {title}")
    log(f"{'=' * 90}")
    log(
        f"  {'Variant':<30s} {'Loss':>8s} {'Best':>8s} {'ms/step':>8s} {'VRAM MB':>8s} {'Params':>10s}"
    )
    log(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 10}")

    is_training = any(r.get("train_steps", 0) > 0 for r in rows)
    sorted_rows = sorted(rows, key=lambda r: r["final_loss"]) if is_training else rows
    best_final_loss = min(r["final_loss"] for r in rows)

    for r in sorted_rows:
        marker = " *" if r["final_loss"] == best_final_loss and is_training else "  "
        p = r.get("total_params", 0)
        params_str = f"{p:>10,}" if p > 0 else f"{'n/a':>10s}"
        log(
            f"{marker}{r['variant']:<30s} {r['final_loss']:>8.4f} {r['best_loss']:>8.4f} "
            f"{r['avg_step_ms']:>8.1f} {r['peak_vram_mb']:>8.0f} {params_str}"
        )

    if is_training:
        log("\n  * = lowest final loss (most stable)")
    log()


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local benchmark suite",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default="all",
        help="Comma-separated benchmarks, or 'all' (default: all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model config (default: {DEFAULT_MODEL}). Options: {', '.join(ALL_MODEL_CONFIGS.keys())}",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Override training steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Custom run identifier (prepended to timestamp)",
    )
    parser.add_argument(
        "--commands",
        action="store_true",
        help="Print copy-pasteable commands to run each benchmark separately with --id",
    )
    parser.add_argument(
        "--vram-fraction",
        type=float,
        default=0.7,
        help="Max fraction of GPU memory PyTorch may reserve (0=unlimited, default 0.7)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available benchmarks and models"
    )
    args = parser.parse_args()

    if args.list:
        available = discover_benchmarks()
        print("\nAvailable benchmarks:")
        for name in sorted(available):
            print(f"  - {name}")
        print("\nAvailable model configs:")
        for name, cfg in ALL_MODEL_CONFIGS.items():
            est_params = estimate_param_count(cfg)
            est_mb = est_params * 4 / (1024 * 1024)
            print(
                f"  - {name}: {cfg.num_layers}L {cfg.model_dim}d {cfg.num_heads}h seq={cfg.seq_len} ~{est_params:,} params ~{est_mb:.1f}MB"
            )
        return

    if args.commands:
        available = discover_benchmarks()
        bench_list = (
            list(available.keys())
            if args.bench == "all"
            else [b.strip() for b in args.bench.split(",")]
        )
        print("\n# Copy-paste commands to run each benchmark separately:")
        print(f"# Model: {args.model}, Steps: {args.steps or 200}, Seed: {args.seed}\n")
        for name in bench_list:
            if name in available:
                cmd = f"python -m runner --bench {name} --id {name}"
                if args.model != DEFAULT_MODEL:
                    cmd += f" --model {args.model}"
                if args.steps is not None:
                    cmd += f" --steps {args.steps}"
                if args.seed != 42:
                    cmd += f" --seed {args.seed}"
                print(cmd)
        print()
        return

    # Device setup.
    if not torch.cuda.is_available():
        print("ERROR: CUDA required.")
        sys.exit(1)

    device = torch.device("cuda", 0)

    # Configure CUDA allocator BEFORE any tensor allocation.
    cuda_settings = configure_cuda_memory(device, vram_fraction=args.vram_fraction)

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)

    # Model config.
    model_cfg = ALL_MODEL_CONFIGS.get(args.model)
    if model_cfg is None:
        print(
            f"ERROR: Unknown model '{args.model}'. Options: {', '.join(ALL_MODEL_CONFIGS.keys())}"
        )
        sys.exit(1)

    # Bench config.
    bench_cfg = BenchmarkConfig(seed=args.seed)
    if args.steps is not None:
        bench_cfg.train_steps = args.steps

    # Discover and filter benchmarks.
    available = discover_benchmarks()
    if args.bench == "all":
        to_run = list(available.keys())
    else:
        to_run = [b.strip() for b in args.bench.split(",")]
        for b in to_run:
            if b not in available:
                print(
                    f"ERROR: Unknown benchmark '{b}'. Available: {', '.join(available.keys())}"
                )
                sys.exit(1)

    # Create results files immediately — user can see them right away.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.id}_{timestamp}" if args.id else timestamp
    results_path = RESULTS_DIR / f"bench_{run_name}.jsonl"
    log_path = init_log(results_path.with_suffix(".log"))
    _init_results_file(results_path, gpu_name, model_cfg.name, to_run, bench_cfg)

    # Now everything goes to both console and log file.
    est_params = estimate_param_count(model_cfg, mlp_activation=bench_cfg.mlp_activation)

    log(f"\nGPU: {gpu_name} ({gpu_mem_mb:.0f} MB)")
    if gpu_mem_mb < 2048:
        log("WARNING: <2GB VRAM. Use --model micro_2L_64d for smallest footprint.")
    log(
        f"Model: {model_cfg.name} ({model_cfg.num_layers}L {model_cfg.model_dim}d {model_cfg.num_heads}h seq={model_cfg.seq_len}) ~{est_params:,} params"
    )
    log(
        f"Steps: {bench_cfg.train_steps}  Seed: {bench_cfg.seed}  Batch: {bench_cfg.batch_size}"
    )
    log_cuda_memory_config(cuda_settings)
    log(f"Benchmarks: {', '.join(to_run)}")
    log(f"Results → {results_path}")
    log(f"Log     → {log_path}")
    log()

    # Run benchmarks — stream each result to disk as it completes.
    total_t0 = time.perf_counter()

    try:
        for bench_name in to_run:
            mod_path = available[bench_name]
            log(f"\n{'#' * 70}")
            log(f"# Benchmark: {bench_name}")
            log(f"{'#' * 70}")

            mod = importlib.import_module(mod_path)
            results = mod.run(device, model_cfg, bench_cfg)

            # Stream to disk immediately, then discard from RAM.
            result_dicts = []
            for r in results:
                d = r.to_dict()
                _append_result(results_path, bench_name, r)
                result_dicts.append(d)

            # Log table from dicts (no BenchmarkResult objects kept).
            _print_table_from_dicts(result_dicts, title=f"Benchmark: {bench_name}")
            del results, result_dicts
            torch.cuda.empty_cache()

        total_elapsed = time.perf_counter() - total_t0
        _append_footer(results_path, total_elapsed)

        # Final summary — read everything back from disk.
        log(f"\n{'#' * 70}")
        log(f"# SUMMARY ({total_elapsed:.1f}s total)")
        log(f"{'#' * 70}")

        grouped = _load_results(results_path)
        for bench_name, rows in grouped.items():
            _print_table_from_dicts(rows, title=bench_name)

        log(f"Results saved: {results_path}")
        log(f"Log saved:     {log_path}")
        log(f"Total time: {total_elapsed:.1f}s")

        from plotter import plot_jsonl

        plot_jsonl(results_path)
    finally:
        close_log()


if __name__ == "__main__":
    main()
