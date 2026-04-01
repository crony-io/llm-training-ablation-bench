"""Benchmark: Quantization (QAT vs PTQ).

Compares Quantization-Aware Training (during training) against
Post-Training Quantization (applied after training).

QAT Variants:
  - qat_baseline: FP16/BF16 training (no quantization)
  - qat_int6: QAT with int6 on all linear layers
  - qat_int5_mlp: QAT with int6 attention and int5 MLPs

PTQ Variants (applied to a fully trained baseline):
  - ptq_per_row_int6: Post-training per-row int6 clipping
  - ptq_gptq_int6: Post-training Full GPTQ int6
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import torch

from bench_utils import (
    BenchmarkResult,
    CastedLinear,
    SyntheticTokenLoader,
    TinyGPT,
    gptq_quantize,
    measure_quant_error,
    quantize_intN_per_row,
    run_micro_train,
    VRAMTracker,
)
from config import BenchmarkConfig, TinyModelConfig
from logger import log


def _collect_synthetic_hessian(
    model: TinyGPT,
    model_cfg: TinyModelConfig,
    device: torch.device,
    num_batches: int = 4,
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Run forward passes and collect H = X^T X for each CastedLinear."""
    hessians: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = {}
    hooks = []

    for name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            layer_name = name

            def make_hook(ln: str):
                def hook_fn(_mod, args, _out):
                    x = args[0].detach().float()
                    x = x.reshape(-1, x.size(-1))
                    H = x.T @ x
                    if ln not in hessians:
                        hessians[ln] = torch.zeros_like(H)
                        counts[ln] = 0
                    hessians[ln].add_(H)
                    counts[ln] += x.size(0)

                return hook_fn

            hooks.append(module.register_forward_hook(make_hook(layer_name)))

    loader = SyntheticTokenLoader(model_cfg.vocab_size, device, seed=123)
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
    ):
        for _ in range(num_batches):
            x, _ = loader.next_batch(batch_size, model_cfg.seq_len)
            model(x)

    for h in hooks:
        h.remove()

    for name in hessians:
        if counts[name] > 0:
            hessians[name] /= counts[name]

    return hessians


def run(
    device: torch.device,
    model_cfg: TinyModelConfig,
    bench_cfg: BenchmarkConfig,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    # Quantization-Aware Training (QAT)
    base_qat = replace(bench_cfg, use_qat=False)

    qat_variants: list[tuple[str, BenchmarkConfig]] = [
        ("qat_baseline", base_qat),
        (
            "qat_int6",
            replace(base_qat, use_qat=True, qat_clip_attn=31, qat_clip_mlp=31),
        ),
        (
            "qat_int5_mlp",
            replace(base_qat, use_qat=True, qat_clip_attn=31, qat_clip_mlp=15),
        ),
    ]

    for name, cfg in qat_variants:
        log(f"\n── Quantization (QAT): {name} ──")
        torch.manual_seed(cfg.seed)
        with VRAMTracker(device) as vt:
            model = TinyGPT(model_cfg, cfg).to(device)
            result = run_micro_train(model, model_cfg, cfg, device, label=name)
        if not result.cached:
            result.peak_vram_mb = vt.peak_mb
        results.append(result)
        del model
        torch.cuda.empty_cache()

    # Post-Training Quantization (PTQ)
    log("\n── Quantization (PTQ): Training Baseline Model ──")

    # Use the same seed as QAT for fair comparison; skip_cache ensures we
    # actually train (we need the weights for quantization, not just metrics).
    ptq_cfg = replace(base_qat)
    torch.manual_seed(ptq_cfg.seed)
    ptq_model = TinyGPT(model_cfg, ptq_cfg).to(device)

    # Train the model to full completion
    run_micro_train(ptq_model, model_cfg, ptq_cfg, device, label="ptq_pretrain", skip_cache=True)

    log("  Collecting Hessian data...")
    hessians = _collect_synthetic_hessian(
        ptq_model, model_cfg, device, num_batches=2, batch_size=2
    )

    q_pr_state = {}
    q_gptq_state = {}
    time_pr_ms = 0.0
    time_gptq_ms = 0.0
    mse_pr_total = 0.0
    mse_gptq_total = 0.0
    layer_count = 0

    for name, module in ptq_model.named_modules():
        if not isinstance(module, CastedLinear) or module.weight.ndim != 2:
            continue
        w = module.weight.detach().cpu().float()
        H = hessians.get(name)
        layer_count += 1

        # Per-row quantization
        t0 = time.perf_counter()
        q_pr, s_pr = quantize_intN_per_row(w, clip_range=31)
        time_pr_ms += 1000.0 * (time.perf_counter() - t0)
        mse_pr_total += measure_quant_error(w, q_pr, s_pr)
        q_pr_state[f"{name}.weight"] = (
            (q_pr.float() * (s_pr.float() if s_pr.ndim == 0 else s_pr.float()[:, None]))
            .to(module.weight.dtype)
            .to(device)
        )

        # GPTQ quantization
        if H is not None and H.shape[0] == w.shape[1]:
            t0 = time.perf_counter()
            q_gptq, s_gptq = gptq_quantize(
                w, H.cpu(), clip_range=31, blocksize=128, damp_frac=0.01
            )
            time_gptq_ms += 1000.0 * (time.perf_counter() - t0)
            mse_gptq_total += measure_quant_error(w, q_gptq, s_gptq)
            q_gptq_state[f"{name}.weight"] = (
                (
                    q_gptq.float()
                    * (s_gptq.float() if s_gptq.ndim == 0 else s_gptq.float()[:, None])
                )
                .to(module.weight.dtype)
                .to(device)
            )

    avg_mse_pr = mse_pr_total / max(layer_count, 1)
    avg_mse_gptq = (
        mse_gptq_total / max(layer_count, 1) if q_gptq_state else float("nan")
    )

    # Evaluate loss on a batch
    loader = SyntheticTokenLoader(model_cfg.vocab_size, device, seed=999)
    x, y = loader.next_batch(bench_cfg.batch_size * 4, model_cfg.seq_len)

    # Save original trained weights so each PTQ method is evaluated independently
    orig_trained_state = {k: v.clone() for k, v in ptq_model.state_dict().items()}

    # Evaluate per-row quantized model
    ptq_model.load_state_dict(q_pr_state, strict=False)
    with VRAMTracker(device) as vt_pr:
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
        ):
            loss_pr = ptq_model(x, y).item()
    peak_vram_pr = vt_pr.peak_mb

    # Restore original trained weights before GPTQ evaluation
    ptq_model.load_state_dict(orig_trained_state)

    # Evaluate GPTQ quantized model
    peak_vram_gptq = 0.0
    if q_gptq_state:
        ptq_model.load_state_dict(q_gptq_state, strict=False)
        with VRAMTracker(device) as vt_gptq:
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
            ):
                loss_gptq = ptq_model(x, y).item()
        peak_vram_gptq = vt_gptq.peak_mb
    else:
        loss_gptq = float("nan")

    mse_change = (
        ((avg_mse_gptq - avg_mse_pr) / avg_mse_pr * 100)
        if not math.isnan(avg_mse_gptq)
        else 0.0
    )
    ce_change = (
        ((loss_gptq - loss_pr) / loss_pr * 100)
        if not math.isnan(loss_gptq)
        else 0.0
    )

    log(f"\n  Quantization comparison ({layer_count} layers):")
    log(
        f"    Per-row int6:  avg_MSE={avg_mse_pr:.6e}  CE_loss={loss_pr:.4f}  time={time_pr_ms:.1f}ms"
    )
    log(
        f"    GPTQ int6:     avg_MSE={avg_mse_gptq:.6e}  CE_loss={loss_gptq:.4f}  time={time_gptq_ms:.1f}ms"
    )
    log(f"    GPTQ vs per-row weight MSE: {mse_change:+.3f}%")
    log(f"    GPTQ vs per-row CE Loss:    {ce_change:+.3f}%")

    del ptq_model
    torch.cuda.empty_cache()

    # Append PTQ results to the master list
    results.extend(
        [
            BenchmarkResult(
                name="quantization",
                variant="ptq_per_row_int6",
                model_config=model_cfg.name,
                train_steps=bench_cfg.train_steps,
                final_loss=loss_pr,
                best_loss=loss_pr,
                avg_step_ms=time_pr_ms / max(layer_count, 1),
                peak_vram_mb=peak_vram_pr,
                total_params=0,
                loss_curve=[],
                extra={
                    "method": "per_row",
                    "layers": layer_count,
                    "avg_mse": avg_mse_pr,
                },
            ),
            BenchmarkResult(
                name="quantization",
                variant="ptq_gptq_int6",
                model_config=model_cfg.name,
                train_steps=bench_cfg.train_steps,
                final_loss=loss_gptq,
                best_loss=loss_gptq,
                avg_step_ms=time_gptq_ms / max(layer_count, 1),
                peak_vram_mb=peak_vram_gptq,
                total_params=0,
                loss_curve=[],
                extra={
                    "method": "gptq",
                    "layers": layer_count,
                    "avg_mse": avg_mse_gptq,
                    "mse_change_pct": mse_change,
                    "ce_change_pct": ce_change,
                },
            ),
        ]
    )

    return results
