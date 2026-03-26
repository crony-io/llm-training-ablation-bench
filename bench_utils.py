"""Core utilities for the local benchmark suite.

Provides a self-contained tiny GPT model with technique toggles, synthetic data
generation, VRAM tracking, and a reusable micro-training loop. No FineWeb
download required — all data is synthetic random tokens.

Targets <500MB VRAM so Windows desktop + browser stay responsive.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import BenchmarkConfig, TinyModelConfig
from logger import log

import copy

# Global cache to store benchmark results for identical configurations
_BENCHMARK_CACHE = {}
# ── Results ────────────────────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    """Collected metrics from a single benchmark run."""

    name: str
    variant: str
    model_config: str
    train_steps: int
    final_loss: float
    best_loss: float
    avg_step_ms: float
    peak_vram_mb: float
    total_params: int
    loss_curve: list[float]
    extra: dict | None = None

    def summary_line(self) -> str:
        return (
            f"{self.variant:<30s} loss={self.final_loss:.5f} best={self.best_loss:.5f} "
            f"step={self.avg_step_ms:.5f}ms vram={self.peak_vram_mb:.0f}MB "
            f"params={self.total_params:,}"
        )

    def to_dict(self) -> dict:
        """Serializable dict for JSONL storage."""
        return {
            "name": self.name,
            "variant": self.variant,
            "model_config": self.model_config,
            "train_steps": self.train_steps,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "avg_step_ms": round(self.avg_step_ms, 2),
            "peak_vram_mb": round(self.peak_vram_mb, 1),
            "total_params": self.total_params,
            "loss_curve": [round(v, 4) for v in self.loss_curve],
            "extra": self.extra,
        }


# ── Synthetic Data ─────────────────────────────────────────────────────────────
class SyntheticTokenLoader:
    """Yields predictable pattern batches. No disk I/O — pure GPU tensor generation."""

    def __init__(self, vocab_size: int, device: torch.device, seed: int = 42):
        self.vocab_size = vocab_size
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

        # Generate a predictable, repeating sine-wave pattern mapped to the vocab.
        # This gives the model an actual pattern to learn, so better architectures will achieve lower losses!
        t = torch.arange(20000, device=device).float()
        self.pattern = ((torch.sin(t * 0.1) + 1.0) / 2.0 * (vocab_size - 1)).long()

    def next_batch(self, batch_size: int, seq_len: int) -> tuple[Tensor, Tensor]:
        # Pick random start indices within the pattern
        starts = torch.randint(
            0,
            len(self.pattern) - seq_len - 1,
            (batch_size,),
            generator=self.gen,
            device=self.device,
        )
        tokens = torch.stack([self.pattern[s : s + seq_len + 1] for s in starts])
        return tokens[:, :-1], tokens[:, 1:]


# ── VRAM Tracking ──────────────────────────────────────────────────────────────


class VRAMTracker:
    """Context manager that measures peak VRAM usage during a block."""

    def __init__(self, device: torch.device):
        self.device = device
        self.peak_mb: float = 0.0

    def __enter__(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize(self.device)
        # Using reserved memory gives a more complete picture of actual GPU usage than allocated memory
        self.peak_mb = torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)


# ── Tiny Model Components ──────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, scale: bool = True):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x: Tensor) -> Tensor:
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        out = (x_f * rms).to(x.dtype)
        if self.scale is not None:
            out = out * self.scale.to(x.dtype)
        return out


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.base = base
        # Precompute frequencies in float32 for numerical stability [4]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Initialize the cache
        self._update_cos_sin_cache(max_seq_len)

    def _update_cos_sin_cache(self, seq_len: int):
        # Generate position indices t: (seq_len)
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim/2)

        # Shape (1, 1, seq_len, dim/2) for broadcasting with (batch, heads, T, dim/2)
        self.register_buffer(
            "cos_cached", freqs.cos().view(1, 1, seq_len, -1), persistent=False
        )
        self.register_buffer(
            "sin_cached", freqs.sin().view(1, 1, seq_len, -1), persistent=False
        )

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        # x shape: (batch, num_heads, T, head_dim)
        T = x.size(2)

        # Dynamically expand the cache if the required sequence length exceeds capacity
        if offset + T > self.cos_cached.size(2):
            self._update_cos_sin_cache(offset + T)

        # --- FIX: Select the relevant slice from the cache ---
        cos = self.cos_cached[:, :, offset : offset + T, :]
        sin = self.sin_cached[:, :, offset : offset + T, :]

        # Standard "Chunked" RoPE (Llama style): Pairs dimensions (i, i + dim/2)
        x1, x2 = x.chunk(2, dim=-1)

        # rotation: (x1 + ix2) * (cos + isin) = (x1*cos - x2*sin) + i(x1*sin + x2*cos)
        rotated_x = torch.cat(
            [
                (x1.float() * cos - x2.float() * sin),
                (x1.float() * sin + x2.float() * cos),
            ],
            dim=-1,
        )

        return rotated_x.type_as(x)


class CastedLinear(nn.Linear):
    """Linear layer with optional APB (Automatic Prune Binarization) or QAT during training."""

    _apb_clip: int = 0
    _apb_prune_frac: float = 0.0
    _qat_clip: int = 0

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype)
        if self.training and self._apb_clip > 0 and w.ndim == 2:
            w32 = w.float()
            clip = self._apb_clip
            row_max = w32.abs().amax(dim=1).clamp_min(1e-12)
            scale = row_max / clip
            if self._apb_prune_frac > 0:
                alpha = (row_max * self._apb_prune_frac)[:, None]
                w32 = w32.masked_fill(w32.abs() < alpha, 0.0)
            bound = (1.0 * clip * scale)[:, None]
            w32 = w32.clamp(-bound, bound)
            w_q = (
                torch.round(w32 / scale[:, None]).clamp(-clip, clip) * scale[:, None]
            ).to(x.dtype)
            # STE: full precision weight relative to the quantized variant
            w = w.to(x.dtype) + (w_q - w.to(x.dtype)).detach()
        elif self.training and self._qat_clip > 0 and w.ndim == 2:
            w32 = w.float()
            clip = self._qat_clip
            row_max = w32.abs().amax(dim=1).clamp_min(1e-12)
            scale = row_max / clip
            w_q = (
                torch.round(w32 / scale[:, None]).clamp(-clip, clip) * scale[:, None]
            ).to(x.dtype)
            w = w + (w_q - w).detach()
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        self.bigram_vocab_size = bigram_vocab_size
        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = (
            CastedLinear(bigram_dim, model_dim, bias=False)
            if bigram_dim != model_dim
            else None
        )
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def bigram_hash(self, tokens: Tensor) -> Tensor:
        """
        Correct implementation of bigram hashing for parameter-efficient embeddings.
        Combines prime-based randomization with bitwise mixing.
        """
        # Ensure we are working with integers for bitwise operations
        t = tokens.to(torch.int32)

        # Set the modulo.
        # Using (size - 1) trick to reserve an index,
        # but ensure size > 1 to avoid a ZeroDivisionError.
        mod = max(1, self.bigram_vocab_size - 1)

        out = torch.empty_like(t)

        # Randomize the 0th position
        # This prevents 'vector collapse' where every sequence starts with the same embedding.
        # It treats the first token as its own unique 'start' bigram.
        out[..., 0] = (36313 * t[..., 0]) % mod

        # Compute the Bigram Hash for the rest of the sequence
        # We use functional XOR to ensure the mixing happens BEFORE the modulo.
        curr_term = 36313 * t[..., 1:]
        prev_term = 27191 * t[..., :-1]

        # XOR effectively mixes the bit-patterns of the two words
        out[..., 1:] = torch.bitwise_xor(curr_term, prev_term) % mod

        return out.long()

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, ve_dim)
        nn.init.zeros_(self.embed.weight)
        self.proj = (
            CastedLinear(ve_dim, model_dim, bias=False) if ve_dim != model_dim else None
        )
        if self.proj is not None:
            nn.init.zeros_(self.proj.weight)
        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))

    def forward(self, token_ids: Tensor) -> Tensor:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = self.proj(h)
        return h * self.scale.to(dtype=h.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyModelConfig, layer_idx: int, use_xsa: bool = False):
        super().__init__()
        head_dim = cfg.model_dim // cfg.num_heads
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = head_dim
        self.use_xsa = use_xsa

        self.c_q = CastedLinear(cfg.model_dim, cfg.num_heads * head_dim, bias=False)
        self.c_k = CastedLinear(cfg.model_dim, cfg.num_kv_heads * head_dim, bias=False)
        self.c_v = CastedLinear(cfg.model_dim, cfg.num_kv_heads * head_dim, bias=False)
        self.c_proj = CastedLinear(cfg.num_heads * head_dim, cfg.model_dim, bias=False)

        self.rotary = Rotary(
            cfg.rope_dims, max_seq_len=cfg.seq_len * 2, base=cfg.rope_base
        )
        self.q_gain = nn.Parameter(torch.full((cfg.num_heads,), cfg.qk_gain_init))
        if use_xsa:
            self.xsa_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: Tensor, v_embed: Tensor | None = None) -> Tensor:
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x)
        if v_embed is not None:
            v = v + v_embed
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Partial RoPE: apply rotation to first rope_dims of head_dim.
        rd = self.rotary.cos_cached.size(-1) * 2
        q_rot, q_pass = q[..., :rd], q[..., rd:]
        k_rot, k_pass = k[..., :rd], k[..., rd:]
        q = torch.cat([self.rotary(q_rot), q_pass], dim=-1)
        k = torch.cat([self.rotary(k_rot), k_pass], dim=-1)

        # QK-Norm: L2-normalize Q,K + learned per-head temperature.
        q = F.normalize(q, dim=-1) * self.q_gain[None, :, None, None].to(q.dtype)
        k = F.normalize(k, dim=-1)

        # GQA: expand KV heads.
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # XSA: exclusive self-attention (mask diagonal) requires manual tril mask to prevent PyTorch SDPA crash.
        # Pass scale=1.0 universally to prevent double-scaling of the normalized query-key dot products!
        if self.use_xsa:
            causal_mask = torch.tril(
                torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1
            )
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal_mask, scale=1.0
            )
        else:
            # QK-Norm produces unit vectors whose dot product is in [-1, 1].
            # SDPA natively divides by sqrt(head_dim), which squashes logits to ~±0.125
            # with default q_gain≈1 — making softmax nearly uniform until q_gain compensates.
            # Passing scale=1.0 disables that division; q_gain controls the temperature.
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1.0)

        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: TinyModelConfig, activation: str = "leaky_relu_sq"):
        super().__init__()
        hidden = int(cfg.model_dim * cfg.mlp_mult)
        self.activation = activation
        if activation == "swiglu":
            self.c_fc = CastedLinear(cfg.model_dim, hidden, bias=False)
            self.c_gate = CastedLinear(cfg.model_dim, hidden, bias=False)
            self.c_proj = CastedLinear(hidden, cfg.model_dim, bias=False)
        else:
            self.c_fc = CastedLinear(cfg.model_dim, hidden, bias=False)
            self.c_proj = CastedLinear(hidden, cfg.model_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "relu_sq":
            return self.c_proj(F.relu(self.c_fc(x)).square())
        elif self.activation == "leaky_relu_sq":
            return self.c_proj(F.leaky_relu(self.c_fc(x), 0.5).square())
        elif self.activation == "gelu_sq":
            return self.c_proj(F.gelu(self.c_fc(x)).square())
        elif self.activation == "swiglu":
            return self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x))
        return self.c_proj(F.leaky_relu(self.c_fc(x), 0.5).square())


class SmearGate(nn.Module):
    """Blend each token's embedding with the previous token's embedding."""

    def __init__(self, dim: int):
        super().__init__()
        # Use a real projection matrix to calculate the gate
        self.gate_proj = CastedLinear(dim, dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate_proj(x))
        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        return (1 - g) * x + g * x_prev


class Block(nn.Module):
    def __init__(
        self,
        cfg: TinyModelConfig,
        layer_idx: int,
        use_xsa: bool = False,
        activation: str = "leaky_relu_sq",
        use_residual_mix: bool = False,
        use_per_dim_scale: bool = False,
    ):
        super().__init__()
        self.use_residual_mix = use_residual_mix
        self.use_per_dim_scale = use_per_dim_scale
        self.attn_norm = RMSNorm(cfg.model_dim, scale=cfg.ln_scale)
        self.mlp_norm = RMSNorm(cfg.model_dim, scale=cfg.ln_scale)
        self.attn = CausalSelfAttention(cfg, layer_idx, use_xsa=use_xsa)
        self.mlp = MLP(cfg, activation=activation)
        # Layer-wise LN attenuation: 1/√(layer+1).
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if cfg.ln_scale else 1.0

        if use_residual_mix:
            self.resid_mix = nn.Parameter(
                torch.stack(
                    (torch.ones(cfg.model_dim), torch.zeros(cfg.model_dim))
                ).float()
            )
        if use_per_dim_scale:
            self.attn_scale = nn.Parameter(
                torch.ones(cfg.model_dim, dtype=torch.float32)
            )
            self.mlp_scale = nn.Parameter(
                torch.ones(cfg.model_dim, dtype=torch.float32)
            )
        if hasattr(cfg, "use_apb") and cfg.use_apb:
            self.apb_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self, x: Tensor, x0: Tensor | None = None, v_embed: Tensor | None = None
    ) -> Tensor:
        if self.use_residual_mix and x0 is not None:
            mix = self.resid_mix.to(dtype=x.dtype)
            x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0

        s = self.ln_scale_factor
        attn_out = self.attn(self.attn_norm(x) * s, v_embed=v_embed)
        if self.use_per_dim_scale:
            attn_out = self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + attn_out

        mlp_out = self.mlp(self.mlp_norm(x) * s)
        if self.use_per_dim_scale:
            mlp_out = self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out
        x = x + mlp_out
        return x


class TinyGPT(nn.Module):
    """Minimal GPT for benchmarking. Supports technique toggles via config."""

    def __init__(self, model_cfg: TinyModelConfig, bench_cfg: BenchmarkConfig):
        super().__init__()
        self.cfg = model_cfg
        self.bench_cfg = bench_cfg

        self.embed = nn.Embedding(model_cfg.vocab_size, model_cfg.model_dim)
        nn.init.normal_(self.embed.weight, std=0.005)

        if bench_cfg.use_bigram_hash:
            self.bigram = BigramHashEmbedding(
                bench_cfg.bigram_vocab_size, bench_cfg.bigram_dim, model_cfg.model_dim
            )
        else:
            self.bigram = None

        self.smear_gate = (
            SmearGate(model_cfg.model_dim) if bench_cfg.use_smear_gate else None
        )

        ve_layers = (
            set(int(x) for x in bench_cfg.ve_layers.split(",") if x.strip())
            if bench_cfg.use_value_embed and bench_cfg.ve_layers
            else set()
        )
        self.ve_shared = None
        self.ve_layer_set = ve_layers
        self.ve_layer_projs = nn.ModuleList()
        if ve_layers:
            kv_dim = model_cfg.num_kv_heads * (
                model_cfg.model_dim // model_cfg.num_heads
            )
            self.ve_shared = ValueEmbedding(
                model_cfg.vocab_size, bench_cfg.ve_dim, kv_dim
            )
            for _ in sorted(ve_layers):
                self.ve_layer_projs.append(CastedLinear(kv_dim, kv_dim, bias=False))

        self.use_unet = bench_cfg.use_unet
        if self.use_unet:
            self.num_encoder_layers = model_cfg.num_layers // 2
            self.num_decoder_layers = model_cfg.num_layers - self.num_encoder_layers
            self.num_skip_weights = min(
                self.num_encoder_layers, self.num_decoder_layers
            )

            # Use projection matrices AND norms for skip connection stability
            self.skip_projections = nn.ModuleList(
                [
                    CastedLinear(model_cfg.model_dim, model_cfg.model_dim, bias=False)
                    for _ in range(self.num_skip_weights)
                ]
            )
            self.skip_norms = nn.ModuleList(
                [RMSNorm(model_cfg.model_dim) for _ in range(self.num_skip_weights)]
            )

        xsa_n = bench_cfg.xsa_last_n if bench_cfg.use_xsa else 0
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_cfg,
                    i,
                    use_xsa=(i >= model_cfg.num_layers - xsa_n) if xsa_n > 0 else False,
                    activation=bench_cfg.mlp_activation,
                    use_residual_mix=bench_cfg.use_residual_mix,
                    use_per_dim_scale=bench_cfg.use_per_dim_scale,
                )
                for i in range(model_cfg.num_layers)
            ]
        )

        self.final_norm = RMSNorm(model_cfg.model_dim, scale=model_cfg.ln_scale)

        if model_cfg.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = CastedLinear(
                model_cfg.model_dim, model_cfg.vocab_size, bias=False
            )

    def forward(self, input_ids: Tensor, target_ids: Tensor | None = None) -> Tensor:
        x = self.embed(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        if self.smear_gate is not None:
            x = self.smear_gate(x)
        x0 = x

        ve_cache = None
        if self.ve_shared is not None:
            ve_cache = self.ve_shared(input_ids)

        if self.use_unet:
            skips = []
            for i in range(self.num_encoder_layers):
                ve = (
                    self.ve_layer_projs[sorted(self.ve_layer_set).index(i)](ve_cache)
                    if ve_cache is not None and i in self.ve_layer_set
                    else None
                )
                x = self.blocks[i](x, x0=x0, v_embed=ve)
                if (
                    i < self.num_encoder_layers - 1
                ):  # Fix: stop pops one layer early to form bottleneck
                    skips.append(x)
            for i in range(self.num_decoder_layers):
                bi = self.num_encoder_layers + i
                if skips:
                    skip_val = skips.pop()
                    # Pass the skip connection through projection AND norm to stabilize Decoder merge
                    x = x + self.skip_norms[i](self.skip_projections[i](skip_val))
                # Decoder block bi=num_encoder_layers+i must use the absolute index bi.
                ve = (
                    self.ve_layer_projs[sorted(self.ve_layer_set).index(bi)](ve_cache)
                    if ve_cache is not None and bi in self.ve_layer_set
                    else None
                )
                x = self.blocks[bi](x, x0=x0, v_embed=ve)
        else:
            for i, block in enumerate(self.blocks):
                ve = (
                    self.ve_layer_projs[sorted(self.ve_layer_set).index(i)](ve_cache)
                    if ve_cache is not None and i in self.ve_layer_set
                    else None
                )
                x = block(x, x0=x0, v_embed=ve)

        x = self.final_norm(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed.weight)

        if (model_cfg := self.cfg) and getattr(
            model_cfg, "logit_softcap", 0
        ) > 0:  # Fix: avoid NaN crash div-by-zero
            logits = model_cfg.logit_softcap * torch.tanh(
                logits / model_cfg.logit_softcap
            )

        if target_ids is not None:
            return F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1)
            )
        return logits

    def setup_apb(self, prune_frac: float = 0.05) -> None:
        for name, module in self.named_modules():
            if isinstance(module, CastedLinear):
                if ".mlp." in name:
                    module._apb_clip = 15
                elif ".attn." in name:
                    module._apb_clip = 31
                module._apb_prune_frac = prune_frac

    def setup_qat(self) -> None:
        cfg = self.bench_cfg
        if not cfg.use_qat:
            return
        for name, module in self.named_modules():
            if isinstance(module, CastedLinear):
                if ".mlp." in name:
                    module._qat_clip = cfg.qat_clip_mlp
                elif ".attn." in name or name.endswith(".proj"):
                    module._qat_clip = cfg.qat_clip_attn

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Muon Optimizer (simplified, no distributed) ───────────────────────────────


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    """Muon optimizer — simplified single-GPU version for benchmarking."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        huber_delta: float = 0.0,
        vs_beta2: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                huber_delta=huber_delta,
                vs_beta2=vs_beta2,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            huber_delta = group.get("huber_delta", 0.0)
            vs_beta2 = group.get("vs_beta2", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)

                # Muon-VS: Variance Scaling should be careful not to destroy spectral info.
                if vs_beta2 > 0:
                    if "vs_v" not in state:
                        state["vs_v"] = torch.zeros_like(g)
                    state["vs_v"].mul_(vs_beta2).addcmul_(g, g, value=1 - vs_beta2)
                    g = g / (state["vs_v"].sqrt() + 1e-8)

                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5

                # Weight decay: Huber or standard L2.
                if wd > 0:
                    if huber_delta > 0:
                        abs_p = p.data.abs()
                        decay_grad = torch.where(
                            abs_p <= huber_delta,
                            p.data,
                            huber_delta * p.data.sign(),
                        )
                        p.data.add_(decay_grad, alpha=-lr * wd)
                    else:
                        p.data.mul_(1 - lr * wd)

                p.data.add_(g.to(p.dtype), alpha=-lr)


# ── Mano Optimizer (Oblique manifold, no Newton-Schulz) ────────────────────────


def _oblique_project_and_normalize(
    momentum: Tensor,
    theta: Tensor,
    dim: int,
    eps: float = 1e-8,
) -> Tensor:
    """Project momentum onto tangent space of Oblique manifold at theta, then normalize."""
    theta_norm = theta.norm(dim=dim, keepdim=True).clamp_min(eps)
    theta_hat = theta / theta_norm
    dot = (momentum * theta_hat).sum(dim=dim, keepdim=True)
    tangent = momentum - theta_hat * dot
    tangent_norm = tangent.norm(dim=dim, keepdim=True).clamp_min(eps)
    return tangent / tangent_norm


class Mano(torch.optim.Optimizer):
    """Mano optimizer — Oblique manifold projection (no Newton-Schulz)."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        rescale: float = 0.2,
        nesterov: bool = False,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                rescale=rescale,
                nesterov=nesterov,
                weight_decay=weight_decay,
            ),
        )
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        norm_dim = self._step_count % 2

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            rescale = group["rescale"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                if g.ndim == 2:
                    v = _oblique_project_and_normalize(g, p.data, dim=norm_dim)
                    n_k = p.shape[norm_dim]
                    v = v * (rescale * math.sqrt(n_k))
                else:
                    v = g / (g.norm() + 1e-8)

                if wd > 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(v.to(p.dtype), alpha=-lr)

        self._step_count += 1


# ── Magma Alignment Damping Wrapper ────────────────────────────────────────────


class MagmaMuon(torch.optim.Optimizer):
    """Muon + Magma: momentum-gradient alignment damping after Newton-Schulz."""

    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int = 5,
        nesterov: bool = True,
        weight_decay: float = 0.0,
        huber_delta: float = 0.0,
        vs_beta2: float = 0.0,
        magma_tau: float = 2.0,
        magma_ema_decay: float = 0.9,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
                huber_delta=huber_delta,
                vs_beta2=vs_beta2,
                magma_tau=magma_tau,
                magma_ema_decay=magma_ema_decay,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            wd = group.get("weight_decay", 0.0)
            huber_delta = group.get("huber_delta", 0.0)
            vs_beta2 = group.get("vs_beta2", 0.0)
            magma_tau = group.get("magma_tau", 2.0)
            magma_ema_decay = group.get("magma_ema_decay", 0.9)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                    state["magma_score"] = torch.ones(1, device=p.device)
                buf = state["momentum_buffer"]

                # Row-wise alignment calculation prevents Magma from being diluted by high dimensions
                if g.ndim == 2:
                    cos_sim = (buf * g).sum(dim=1) / (
                        buf.norm(dim=1) * g.norm(dim=1) + 1e-8
                    )
                    raw_score = torch.sigmoid(cos_sim / magma_tau).mean()
                else:
                    cos_sim = (buf * g).sum() / (buf.norm() * g.norm() + 1e-8)
                    raw_score = torch.sigmoid(cos_sim / magma_tau)

                state["magma_score"].mul_(magma_ema_decay).add_(
                    raw_score, alpha=(1.0 - magma_ema_decay)
                )

                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)

                if vs_beta2 > 0:
                    if "vs_v" not in state:
                        state["vs_v"] = torch.zeros_like(g)
                    state["vs_v"].mul_(vs_beta2).addcmul_(g, g, value=1 - vs_beta2)
                    g = g / (state["vs_v"].sqrt() + 1e-8)

                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5

                # Apply Magma damping.
                g = g * state["magma_score"]

                if wd > 0:
                    if huber_delta > 0:
                        abs_p = p.data.abs()
                        decay_grad = torch.where(
                            abs_p <= huber_delta,
                            p.data,
                            huber_delta * p.data.sign(),
                        )
                        p.data.add_(decay_grad, alpha=-lr * wd)
                    else:
                        p.data.mul_(1 - lr * wd)

                p.data.add_(g.to(p.dtype), alpha=-lr)


# ── Quantization Helpers ───────────────────────────────────────────────────────


def quantize_intN_per_row(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:
    """Per-row symmetric quantization with percentile clip search."""
    t32 = t.float()
    if t32.ndim < 2:
        amax = t32.abs().max().item()
        scale = torch.tensor(
            amax / clip_range if amax > 0 else 1.0, dtype=torch.float16
        )
        q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(
            torch.int8
        )
        return q, scale

    best_s, best_err = None, float("inf")
    for pct in [0.9995, 0.9999, 1.0]:
        row_clip = (
            torch.quantile(t32.abs(), pct, dim=1)
            if pct < 1.0
            else t32.abs().amax(dim=1)
        )
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q_test = torch.clamp(
            torch.round(t32 / s.float()[:, None]), -clip_range, clip_range
        )
        err = (t32 - q_test * s.float()[:, None]).pow(2).mean().item()
        if err < best_err:
            best_s, best_err = s, err
    q = torch.clamp(
        torch.round(t32 / best_s.float()[:, None]), -clip_range, clip_range
    ).to(torch.int8)
    return q, best_s


def gptq_quantize(
    W: Tensor,
    H: Tensor,
    clip_range: int = 31,
    blocksize: int = 128,
    damp_frac: float = 0.01,
) -> tuple[Tensor, Tensor]:
    """Full GPTQ: Hessian-calibrated quantization with column-wise error compensation."""
    _rows, cols = W.shape
    W = W.float().clone()
    H = H.float().clone()

    diag_mean = H.diag().mean()
    H.diagonal().add_(damp_frac * diag_mean)

    try:
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
    except torch.linalg.LinAlgError:
        H.diagonal().add_(0.1 * diag_mean)
        try:
            L = torch.linalg.cholesky(H)
            Hinv = torch.cholesky_inverse(L)
        except torch.linalg.LinAlgError:
            return quantize_intN_per_row(W, clip_range)

    best_s, best_err = None, float("inf")
    for pct in [0.9995, 0.9999, 1.0]:
        row_clip = (
            torch.quantile(W.abs(), pct, dim=1) if pct < 1.0 else W.abs().amax(dim=1)
        )
        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)
        q_test = torch.clamp(
            torch.round(W / s.float()[:, None]), -clip_range, clip_range
        )
        err = (W - q_test * s.float()[:, None]).pow(2).mean().item()
        if err < best_err:
            best_s, best_err = s, err
    scale = best_s.float()
    Q = torch.zeros_like(W)

    for col_start in range(0, cols, blocksize):
        col_end = min(col_start + blocksize, cols)
        block_W = W[:, col_start:col_end].clone()
        block_Hinv = Hinv[col_start:col_end, col_start:col_end]
        block_Err = torch.zeros_like(block_W)

        for j in range(col_end - col_start):
            col_j = col_start + j
            w_col = block_W[:, j]
            d = block_Hinv[j, j].clamp_min(1e-12)
            q_col = torch.clamp(torch.round(w_col / scale[:]), -clip_range, clip_range)
            Q[:, col_j] = q_col

            # It MUST be (Quantized - True) so subtracting it later cancels the error out!
            quantized_val = q_col * scale[:]
            err_col = (quantized_val - w_col) / d

            block_Err[:, j] = err_col
            if j + 1 < col_end - col_start:
                block_W[:, j + 1 :] -= (
                    err_col[:, None]
                    * block_Hinv[j, j + 1 : col_end - col_start][None, :]
                )

        if col_end < cols:
            W[:, col_end:] -= block_Err @ Hinv[col_start:col_end, col_end:]

    return Q.to(torch.int8), best_s


def measure_quant_error(weight: Tensor, q: Tensor, scale: Tensor) -> float:
    """Compute reconstruction MSE for a quantized weight."""
    w32 = weight.float()
    if scale.ndim == 0:
        recon = q.float() * scale.float()
    else:
        recon = q.float() * scale.float()[:, None]
    return (w32 - recon).pow(2).mean().item()


# ── LR Schedule ────────────────────────────────────────────────────────────────


def compute_lr_multiplier(step: int, bench_cfg: BenchmarkConfig) -> float:
    """Compute LR multiplier for warmup + warmdown cosine schedule."""
    total = bench_cfg.train_steps

    # Warmup phase: strictly interpolate from 0.0 to 1.0
    if bench_cfg.use_lr_warmup and step < bench_cfg.lr_warmup_steps:
        return step / max(1, bench_cfg.lr_warmup_steps)

    # Warmdown phase: cosine decay
    if bench_cfg.use_lr_warmdown:
        warmdown_start = int(total * (1.0 - bench_cfg.lr_warmdown_frac))
        if step >= warmdown_start:
            progress = (step - warmdown_start) / max(total - warmdown_start - 1, 1)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return 1.0


def compute_momentum(step: int, bench_cfg: BenchmarkConfig) -> float:
    """Compute momentum with optional warmup ramp."""
    if bench_cfg.use_momentum_warmup and step < bench_cfg.momentum_warmup_steps:
        frac = step / bench_cfg.momentum_warmup_steps
        return bench_cfg.momentum_warmup_start + frac * (
            bench_cfg.muon_momentum - bench_cfg.momentum_warmup_start
        )
    return bench_cfg.muon_momentum


# ── Training Loop ──────────────────────────────────────────────────────────────


def run_micro_train(
    model: TinyGPT,
    model_cfg: TinyModelConfig,
    bench_cfg: BenchmarkConfig,
    device: torch.device,
    label: str = "baseline",
) -> BenchmarkResult:
    """Run a tiny training loop and collect metrics. Returns BenchmarkResult."""

    # --- CACHE CHECK ---
    global _BENCHMARK_CACHE
    cfg_state = str(model_cfg.__dict__) + str(bench_cfg.__dict__)
    cfg_hash = hash(cfg_state)

    if cfg_hash in _BENCHMARK_CACHE:
        log(f"  [{label}] Using cached baseline result! (Skipping redundant compute)")
        # Deepcopy to avoid mutating the cached object's label
        cached_res = copy.deepcopy(_BENCHMARK_CACHE[cfg_hash])
        cached_res.name = label
        cached_res.variant = label
        return cached_res
    # ----------------------

    torch.manual_seed(bench_cfg.seed)
    model.setup_qat()
    loader = SyntheticTokenLoader(model_cfg.vocab_size, device, seed=bench_cfg.seed)

    # Split params into matrix (spectral optimizer) and scalar (AdamW) groups.
    matrix_params = []
    scalar_params = []
    embed_params = []
    embed_param_tensors: list[Tensor] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name and "lm_head" not in name:
            matrix_params.append(p)
        elif "embed" in name or "lm_head" in name:
            embed_param_tensors.append(p)
            embed_params.append(
                {"params": [p], "lr": bench_cfg.embed_lr, "base_lr": bench_cfg.embed_lr}
            )
        else:
            scalar_params.append(p)

    adam_kw = dict(
        betas=(bench_cfg.beta1, bench_cfg.beta2),
        weight_decay=bench_cfg.weight_decay,
        fused=True,
    )
    optimizers = []

    if embed_params:
        optimizers.append(torch.optim.AdamW(embed_params, **adam_kw))

    if matrix_params:
        if bench_cfg.use_mano:
            matrix_opt = Mano(
                matrix_params,
                lr=bench_cfg.matrix_lr,
                momentum=bench_cfg.muon_momentum,
                rescale=bench_cfg.mano_rescale,
                nesterov=bench_cfg.mano_nesterov,
                weight_decay=bench_cfg.weight_decay,
            )
        elif bench_cfg.use_magma:
            matrix_opt = MagmaMuon(
                matrix_params,
                lr=bench_cfg.matrix_lr,
                momentum=bench_cfg.muon_momentum,
                backend_steps=bench_cfg.muon_backend_steps,
                weight_decay=bench_cfg.weight_decay,
                huber_delta=bench_cfg.huber_delta if bench_cfg.use_huber_decay else 0.0,
                vs_beta2=bench_cfg.muon_vs_beta2 if bench_cfg.use_muon_vs else 0.0,
                magma_tau=bench_cfg.magma_tau,
                magma_ema_decay=bench_cfg.magma_ema_decay,
            )
        else:
            matrix_opt = Muon(
                matrix_params,
                lr=bench_cfg.matrix_lr,
                momentum=bench_cfg.muon_momentum,
                backend_steps=bench_cfg.muon_backend_steps,
                weight_decay=bench_cfg.weight_decay,
                huber_delta=bench_cfg.huber_delta if bench_cfg.use_huber_decay else 0.0,
                vs_beta2=bench_cfg.muon_vs_beta2 if bench_cfg.use_muon_vs else 0.0,
            )
        optimizers.append(matrix_opt)

    if scalar_params:
        optimizers.append(
            torch.optim.AdamW(
                [{"params": scalar_params, "lr": bench_cfg.scalar_lr}],
                **adam_kw,
            )
        )

    # Validation loader for independent anchor.
    val_loader = SyntheticTokenLoader(
        model_cfg.vocab_size, device, seed=bench_cfg.seed + 999
    )

    # EMA state (optional).
    ema_state = None
    if bench_cfg.use_ema:
        ema_state = {
            n: t.detach().float().clone() for n, t in model.state_dict().items()
        }

    sma_state = None
    sma_count = 0
    swa_start_step_global = int(bench_cfg.train_steps * bench_cfg.swa_start_frac)
    if bench_cfg.use_swa:
        sma_state = {
            n: torch.zeros_like(t, dtype=torch.float32)
            for n, t in model.state_dict().items()
        }

    loss_curve: list[float] = []
    best_loss = float("inf")

    model.train()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()

    for step in range(bench_cfg.train_steps):
        # LR schedule.
        lr_mult = compute_lr_multiplier(step, bench_cfg)
        current_lr = 0.0
        for opt in optimizers:
            for pg in opt.param_groups:
                base_lr = pg.get("base_lr", pg["lr"])
                if "base_lr" not in pg:
                    pg["base_lr"] = pg["lr"]
                pg["lr"] = base_lr * lr_mult
                if pg["lr"] > current_lr:
                    current_lr = pg["lr"]

        # Momentum warmup.
        if bench_cfg.use_momentum_warmup:
            mom = compute_momentum(step, bench_cfg)
            for opt in optimizers:
                for pg in opt.param_groups:
                    if "momentum" in pg:
                        pg["momentum"] = mom

        # Batch size warmup.
        if bench_cfg.use_batch_warmup and step < bench_cfg.batch_warmup_steps:
            bw_frac = step / bench_cfg.batch_warmup_steps
            bs = max(
                1,
                int(
                    bench_cfg.batch_size
                    * (
                        bench_cfg.batch_warmup_start_frac
                        + bw_frac * (1.0 - bench_cfg.batch_warmup_start_frac)
                    )
                ),
            )
        else:
            bs = bench_cfg.batch_size

        x, y = loader.next_batch(bs, model_cfg.seq_len)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = model(x, y)

        loss.backward()

        def _group_grad_norm(params: list[Tensor]) -> float:
            sq_sum = 0.0
            any_seen = False
            for p in params:
                g = p.grad
                if g is None:
                    continue
                any_seen = True
                sq_sum += g.detach().float().pow(2).sum().item()
            if not any_seen:
                return 0.0
            return math.sqrt(sq_sum)

        grad_norm_matrix = _group_grad_norm(matrix_params)
        grad_norm_scalar = _group_grad_norm(scalar_params)
        grad_norm_embed = _group_grad_norm(embed_param_tensors)

        # Calculate true overall norm before clip for accurate logging
        total_grad_norm = math.sqrt(
            grad_norm_matrix**2 + grad_norm_scalar**2 + grad_norm_embed**2
        )
        grad_norm = total_grad_norm

        # Fix: Placebo clipping! Matrix optimizer normalization erases the global scalar clip scale.
        # We selectively apply the clip only to non-matrix parameters that are vulnerable to exploding gradients
        if bench_cfg.grad_clip_norm > 0:
            params_to_clip = scalar_params + embed_param_tensors
            if params_to_clip:
                torch.nn.utils.clip_grad_norm_(params_to_clip, bench_cfg.grad_clip_norm)
        else:
            params_to_clip = scalar_params + embed_param_tensors
            if params_to_clip:
                torch.nn.utils.clip_grad_norm_(params_to_clip, float("inf"))

        is_log_step = bench_cfg.log_every > 0 and (step + 1) % bench_cfg.log_every == 0
        pre_matrix = [p.detach().clone() for p in matrix_params] if is_log_step else []
        pre_scalar = [p.detach().clone() for p in scalar_params] if is_log_step else []
        pre_embed = (
            [p.detach().clone() for p in embed_param_tensors] if is_log_step else []
        )
        params_all = matrix_params + scalar_params + embed_param_tensors
        pre_all = [p.detach().clone() for p in params_all] if is_log_step else []

        for opt in optimizers:
            opt.step()

        if is_log_step:

            def _upd_ratio(params: list[Tensor], pre_clones: list[Tensor]) -> float:
                if not params:
                    return 0.0
                delta_sq = sum(
                    (p - pre_p).pow(2).sum().item()
                    for p, pre_p in zip(params, pre_clones)
                )
                w_sq = sum(p.pow(2).sum().item() for p in params)
                return (
                    math.sqrt(delta_sq) / (math.sqrt(w_sq) + 1e-8) if w_sq > 0 else 0.0
                )

            upd_ratio_matrix = _upd_ratio(matrix_params, pre_matrix)
            upd_ratio_scalar = _upd_ratio(scalar_params, pre_scalar)
            upd_ratio_embed = _upd_ratio(embed_param_tensors, pre_embed)
            upd_ratio_all = _upd_ratio(params_all, pre_all)

            # Keep legacy name but make it more informative: overall update ratio.
            upd_ratio = upd_ratio_all

        # EMA update.
        if ema_state is not None:
            with torch.no_grad():
                for name, t in model.state_dict().items():
                    ema_state[name].mul_(bench_cfg.ema_decay).add_(
                        t.detach().float(), alpha=1.0 - bench_cfg.ema_decay
                    )

        # SWA update.
        if sma_state is not None:
            swa_start_step = int(bench_cfg.train_steps * bench_cfg.swa_start_frac)
            if (
                step >= swa_start_step
                and (step - swa_start_step) % bench_cfg.swa_every == 0
            ):
                with torch.no_grad():
                    for name, t in model.state_dict().items():
                        sma_state[name].add_(t.detach().float())
                sma_count += 1

        loss_val = loss.item()

        # Track both active loss and variant loss for the logs.
        shadow_loss = None
        shadow_str = ""

        # Use an independent batch to correctly evaluate shadow models
        if (ema_state is not None) or (sma_state is not None and sma_count > 0):
            shadow_x, shadow_y = val_loader.next_batch(bs, model_cfg.seq_len)

        if ema_state is not None:
            with torch.no_grad():
                orig_state = {
                    n: t.detach().clone() for n, t in model.state_dict().items()
                }
                # Evaluate in native mixed bfloat16 alignment instead of float32
                model.load_state_dict({n: t.bfloat16() for n, t in ema_state.items()})
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    shadow_loss = model(shadow_x, shadow_y).item()
                # Restore without re-binding Parameters (keeps optimizer refs valid).
                model.load_state_dict(orig_state)
                shadow_str = f" (Shadow: {shadow_loss:.5f})"
        elif sma_state is not None and sma_count > 0:
            with torch.no_grad():
                orig_state = {
                    n: t.detach().clone() for n, t in model.state_dict().items()
                }
                avg_state = {
                    n: (t / sma_count).bfloat16() for n, t in sma_state.items()
                }
                model.load_state_dict(avg_state)
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=True
                ):
                    shadow_loss = model(shadow_x, shadow_y).item()
                model.load_state_dict(orig_state)
                shadow_str = f" (Shadow: {shadow_loss:.5f})"

        # We keep the variant loss in the curve if it exists so benchmarks track the tested technique.
        record_loss = shadow_loss if shadow_loss is not None else loss_val
        loss_curve.append(record_loss)
        if record_loss < best_loss:
            best_loss = record_loss

        if is_log_step:
            val_x, val_y = val_loader.next_batch(8, model_cfg.seq_len)
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True),
            ):
                val_loss = model(val_x, val_y).item()

            log(
                f"  [{label}] {step + 1}/{bench_cfg.train_steps} | Loss: {loss_val:.5f}{shadow_str} | Val: {val_loss:.5f} | LR: {current_lr:.5f} (lr_mult={lr_mult:.6f}) | bs={bs} | Grad: {grad_norm:.5f} | "
                f"grad_matrix={grad_norm_matrix:.5f} grad_scalar={grad_norm_scalar:.5f} grad_embed={grad_norm_embed:.5f} | "
                f"Upd/W_total: {upd_ratio:.8f} | Upd/W_matrix: {upd_ratio_matrix:.8f} | Upd/W_scalar: {upd_ratio_scalar:.8f} | Upd/W_embed: {upd_ratio_embed:.8f} | swa_cnt={sma_count} | swa_start={swa_start_step_global}"
            )

    final_loss_val = loss_curve[-1] if loss_curve else float("nan")

    torch.cuda.synchronize(device)
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)

    # Use max memory reserved for peak_vram as it includes contexts/kernels, unlike allocated
    res = BenchmarkResult(
        name=label,
        variant=label,
        model_config=model_cfg.name,
        train_steps=bench_cfg.train_steps,
        final_loss=final_loss_val,
        best_loss=best_loss,
        avg_step_ms=elapsed_ms / max(bench_cfg.train_steps, 1),
        peak_vram_mb=torch.cuda.max_memory_reserved(device) / (1024 * 1024),
        total_params=model.param_count(),
        loss_curve=loss_curve,
    )

    # --- CACHE SAVE ---
    _BENCHMARK_CACHE[cfg_hash] = copy.deepcopy(res)
    # ---------------------

    return res
