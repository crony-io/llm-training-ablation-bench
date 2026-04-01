"""CUDA Memory Allocator Configuration.

Controls PyTorch's CUDA caching allocator to reduce VRAM overhead on consumer
GPUs. By default, PyTorch pre-allocates large memory pools and rarely releases
them, causing Windows Task Manager to report near-100% GPU memory even when
actual model usage is tiny (e.g. 42-162 MB for this benchmark suite).

This module provides two strategies:

1. **Fraction limit**: Caps the maximum GPU memory PyTorch may reserve to a
   fraction of total VRAM, leaving headroom for the OS/desktop/browser.

2. **Allocator tuning**: Configures ``PYTORCH_CUDA_ALLOC_CONF`` to use
   smaller allocation blocks and more aggressive garbage collection, trading
   a tiny amount of allocation speed for significantly lower peak reserved
   memory.

Usage (automatic):
    The runner calls ``configure_cuda_memory(device)`` before any benchmark.
    No user action needed.

Usage (manual override):
    ``python -m runner --vram-fraction 0.5``   # use at most 50% of GPU memory
    ``python -m runner --vram-fraction 0``      # disable limit (PyTorch default)
"""

from __future__ import annotations

import os

import torch

from logger import log


# Sensible defaults for a 12 GB consumer GPU running Windows with desktop.
# These leave ~4 GB free for the OS, driver, and other applications.
DEFAULT_VRAM_FRACTION = 0.7
DEFAULT_ALLOCATOR_CONF = (
    "expandable_segments:True,"
    "garbage_collection_threshold:0.6,"
    "max_split_size_mb:128"
)


def configure_cuda_memory(
    device: torch.device,
    vram_fraction: float = DEFAULT_VRAM_FRACTION,
) -> dict[str, str | float]:
    """Configure PyTorch's CUDA allocator for lower peak VRAM reservation.

    Must be called **before** any CUDA tensor allocation to take full effect.

    Args:
        device: Target CUDA device.
        vram_fraction: Maximum fraction of total GPU memory that PyTorch may
            reserve. 0.0 disables the limit (PyTorch default behaviour).
            Values in (0, 1] are valid.

    Returns:
        Dict with the applied settings for logging purposes.
    """
    settings: dict[str, str | float] = {}

    # Configure allocator via environment variable (must precede the first CUDA call)
    prev_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if not prev_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_ALLOCATOR_CONF
        settings["PYTORCH_CUDA_ALLOC_CONF"] = DEFAULT_ALLOCATOR_CONF
    else:
        settings["PYTORCH_CUDA_ALLOC_CONF"] = prev_conf + " (user-set, kept)"

    # Apply per-process memory fraction cap
    if vram_fraction > 0:
        vram_fraction = max(0.1, min(vram_fraction, 1.0))
        torch.cuda.set_per_process_memory_fraction(vram_fraction, device)
        settings["vram_fraction"] = vram_fraction

        total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 * 1024)
        cap_mb = total_mb * vram_fraction
        settings["cap_mb"] = round(cap_mb, 0)
    else:
        settings["vram_fraction"] = 0.0
        settings["cap_mb"] = "unlimited"

    return settings


def log_cuda_memory_config(settings: dict[str, str | float]) -> None:
    """Pretty-print the applied CUDA memory settings to the benchmark log."""
    frac = settings.get("vram_fraction", 0)
    cap = settings.get("cap_mb", "unlimited")
    alloc = settings.get("PYTORCH_CUDA_ALLOC_CONF", "default")

    if frac and frac > 0:
        log(f"CUDA Memory: cap={cap} MB ({frac:.0%} of total) | allocator={alloc}")
    else:
        log(f"CUDA Memory: unlimited | allocator={alloc}")
