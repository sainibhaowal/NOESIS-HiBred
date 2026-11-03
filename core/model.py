import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal

import torch
import torch.distributed as dist
import torch.nn.functional as F
from engram import EngramLayout, NgramHashState
from image_processor import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
from kernel import (
    act_quant,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    sparse_attn,
)
from torch import nn
from vision import Aligner, ViT
from osc_block import OSCLatentAttractor

# Logical model-parallel layout. These are deliberately *not* derived from the process group:
# FSDP shards the model across all ranks while the model itself stays logically unsharded.
# Call `configure_model_parallel(mp_size, mp_rank)` before constructing a model.
world_size = 1
rank = 0
default_dtype = torch.float8_e4m3fn  # storage dtype for Linear weights, from ModelArgs.dtype
fp8_block_size = 32  # one fp8 scale per 32x32 weight block / 32 activations
fp4_block_size = 32  # one fp4 scale per 32 elements along K
scale_fmt = "ue8m0"
scale_dtype = torch.float8_e8m0fnu

MP_LAYOUT_CONFIGURED = False


def configure_model_parallel(mp_size: int = 1, mp_rank: int = 0) -> None:
    """Set the logical model-parallel size/rank used by all sharded modules.

    The data-parallel process group is independent of this. A production FSDP run uses
    ``configure_model_parallel(1, 0)``: every parameter is logically complete and FSDP owns
    the physical sharding. Tensor parallelism remains available for legacy profiles, but the
    model must be built after this call.
    """
    global world_size, rank, MP_LAYOUT_CONFIGURED
    if mp_size <= 0:
        raise ValueError("mp_size must be positive")
    if not 0 <= mp_rank < mp_size:
        raise ValueError(f"mp_rank must be in [0, {mp_size}), received {mp_rank}")
    world_size = int(mp_size)
    rank = int(mp_rank)
    MP_LAYOUT_CONFIGURED = True


@dataclass
class AttentionState:
    """Per-forward values attention layers hand down the stack.

    This replaces the previous process-global singleton, which made forwards non-reentrant and
    leaked state between batches, chunks, and models. A new state object is created for every
    chunk; source layers write into it and later layers read from it in stack order.
    """

    compress_kv: torch.Tensor | None = None
    index_k: torch.Tensor | None = None
    topk_idxs: torch.Tensor | None = None
    candidates: torch.Tensor | None = None

    def reset(self) -> None:
        self.compress_kv = None
        self.index_k = None
        self.topk_idxs = None
        self.candidates = None

    def detach(self) -> None:
        """Cut the autograd graph at a truncated-BPTT boundary."""
        for name in ("compress_kv", "index_k", "topk_idxs", "candidates"):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.detach())


@dataclass
class ChunkState:
    """Cross-chunk state for long-context training and chunked prefill."""

    attention: AttentionState = field(default_factory=AttentionState)
    retro: torch.Tensor | None = None
    mtp_window: torch.Tensor | None = None
    mtp_start_pos: int = 0

    def detach(self) -> None:
        """Cut the autograd graph at a truncated-BPTT boundary."""
        self.attention.detach()
        if isinstance(self.retro, torch.Tensor):
            self.retro = self.retro.detach()
        if isinstance(self.mtp_window, torch.Tensor):
            self.mtp_window = self.mtp_window.detach()


@contextmanager
def set_dtype(dtype):
    """Temporarily override torch's default dtype, restoring it even if the body raises."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


@dataclass
class ModelArgs:
    """Field names are exactly the config JSON keys. The defaults are a small model that
    `python model.py` can run, not the released shapes -- though the scale-independent
    values (norm_eps, score_func, hc_*, engram_*) do match it."""

    # runtime limits rather than model shape: they size the KV caches
    max_batch_size: int = 4
    max_seq_len: int = 4096
    temperature: float = 1
    dtype: Literal["bf16", "fp8"] = "fp8"
    expert_dtype: Literal["fp4"] | None = "fp4"
    vocab_size: int = 129280
    dim: int = 1024
    moe_inter_dim: int = 1024
    n_layers: int = 5
    n_mtp_layers: int = 1  # extra draft layers appended after the backbone, indices n_layers..
    n_heads: int = 16
    # moe
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    n_activated_experts: int = 2
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    route_scale: float = 1.0
    swiglu_limit: float = 0.0
    # attention: latent q/kv projections, plus a LoRA-factorised output projection over o_groups
    q_lora_rank: int = 256
    head_dim: int = 128
    rope_head_dim: int = 32
    norm_eps: float = 1e-20
    o_groups: int = 8
    o_lora_rank: int = 256
    # sparse attention: every layer attends over a sliding window, and may add compressed KV on top
    window_size: int = 128
    # one entry per layer, MTP layers included: 0 = sliding window only, r = KV compressed r-to-1
    compress_ratios: tuple[int, ...] = (0, 2, 2, 1, 1, 0)
    # layers sharing a ratio also share one compressed KV and one indexer, produced by the first
    kv_source_layers: tuple[int, ...] = (1, 3)
    index_source_layers: tuple[int, ...] = (1, 3)
    # rope, with YaRN extrapolation when original_seq_len > 0. Compressed KV rotates at its own
    # theta because one latent stands for compress_ratio tokens, so its positions are further apart.
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 0
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    # the indexer: a small extra attention that scores compressed positions, so each query can keep
    # just `index_topk` of them for sparse MLA attention in NOESIS-Σ.
    index_n_heads: int = 16
    index_head_dim: int = 64
    index_topk: int = 64
    # Score every N-th query and share its top-k within the group. 1 preserves exact per-query
    # retrieval; >1 trades a little retrieval resolution for much lower long-context index cost.
    index_query_stride: int = 1
    index_query_chunk: int = 1024
    # candidate pre-filtering: candidate_source_layer < 0 turns it off and the other two are unused
    candidate_source_layer: int = -1
    candidate_topk_blocks: int = 0
    candidate_block_size: int = 0
    # hyper-connections: the residual stream is carried as hc_mult parallel copies
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6
    # engram: n-gram hash lookups added into the residual stream at a few layers
    engram_layer_ids: tuple[int, ...] = ()
    engram_num_embeddings: tuple[int, ...] = ()  # unpadded table rows; each rank allocates ceil(rows / world_size)
    engram_max_ngram_size: int = 1
    engram_vocab_size: int = 0  # bucket size each (n-gram size, head) starts searching primes from
    engram_n_heads: int = 0
    engram_head_dim: int = 0
    engram_pad_id: int = 2  # token that fills n-gram slots with no history; matches training
    # size of the compressed tokenizer vocab; every hash multiplier is derived from it
    engram_compressed_vocab_size: int = 0
    # vision (VL); vision_n_layers == 0 disables the vision path
    vision_n_layers: int = 0
    vision_dim: int = 1024
    vision_n_heads: int = 16
    vision_inter_dim: int = 2816
    vision_patch_size: int = 14
    vision_rope_theta: float = 10000.0
    vision_downsample_ratio: int = 3
    vision_max_n_token: int = 1024
    vision_min_pixels: int = 544 * 544
    vision_max_wh_ratio: int | None = None
    # raw id of image placeholder token; every position of an image span carries this id in input_ids
    image_token_id: int = 129264
    # dspark draft head. Only the forward pass is implemented here -- nothing calls forward_spec,
    # so these are read but the speculative-decoding loop itself is out of scope for this repo.
    dspark_block_size: int = 0
    dspark_noise_token_id: int = 0
    dspark_target_layer_ids: tuple[int, ...] = ()
    dspark_markov_rank: int = 256
    dspark_n_routed_experts: int = 0
    dspark_n_activated_experts: int = 0
    # NOESIS-Σ Operator-Split Continuous (OSC) cognitive engine parameters
    osc_enabled: bool = True
    osc_rank: int = 32
    osc_icnn_hidden: int = 128
    osc_steps: int = 4
    osc_dt: float = 0.05
    osc_c_min: float = 0.01
    osc_max_norm: float = 16.0
    osc_retrograde: bool = True

    def __post_init__(self):
        self.compress_ratios = tuple(self.compress_ratios)
        self.kv_source_layers = tuple(self.kv_source_layers)
        self.index_source_layers = tuple(self.index_source_layers)
        self.engram_layer_ids = tuple(self.engram_layer_ids)
        self.engram_num_embeddings = tuple(self.engram_num_embeddings)
        self.dspark_target_layer_ids = tuple(self.dspark_target_layer_ids)
        # Small test and research shapes often override ``n_layers`` without
        # specifying attention-layout fields.  The dataclass defaults describe
        # the five-layer reference shape only, so replace that implicit layout
        # with an explicitly local-attention-only layout for other shapes.
        reference_ratios = (0, 2, 2, 1, 1, 0)
        reference_sources = (1, 3)
        if (
            len(self.compress_ratios) != self.n_layers + self.n_mtp_layers
            and self.compress_ratios == reference_ratios
            and self.kv_source_layers == reference_sources
            and self.index_source_layers == reference_sources
        ):
            self.compress_ratios = (0,) * (self.n_layers + self.n_mtp_layers)
            self.kv_source_layers = ()
            self.index_source_layers = ()

    def validate_for_model(self) -> None:
        """Reject feature layouts that would otherwise fail inside a forward pass.

        This deliberately validates relationships between optional components at
        construction time.  It keeps a text-only profile from silently creating
        a partially configured Engram, compressed-KV, or MTP path.
        """
        total_layers = self.n_layers + self.n_mtp_layers
        if self.n_layers <= 0 or self.n_mtp_layers < 0:
            raise ValueError("n_layers must be positive and n_mtp_layers cannot be negative")
        if self.index_query_stride <= 0 or self.index_query_chunk <= 0:
            raise ValueError("index_query_stride and index_query_chunk must be positive")
        if len(self.compress_ratios) != total_layers:
            raise ValueError(
                "compress_ratios must contain one entry for every backbone and MTP layer "
                f"({total_layers} entries required, received {len(self.compress_ratios)})"
            )
        if any(ratio < 0 for ratio in self.compress_ratios):
            raise ValueError("compress_ratios cannot contain negative values")

        active_compression = {i for i, ratio in enumerate(self.compress_ratios) if ratio > 0}
        if active_compression:
            if not self.kv_source_layers or not self.index_source_layers:
                raise ValueError("active compressed-KV layers require KV and index source layers")
            for source in (*self.kv_source_layers, *self.index_source_layers):
                if source not in active_compression:
                    raise ValueError("compressed-KV source layers must have a positive compression ratio")
            for layer_id in active_compression:
                ratio = self.compress_ratios[layer_id]
                if layer_id >= self.n_layers:
                    raise ValueError("DSpark MTP layers cannot use compressed-KV")
                if not any(
                    source <= layer_id and self.compress_ratios[source] == ratio
                    for source in self.kv_source_layers
                ):
                    raise ValueError(
                        f"compressed-KV layer {layer_id} has no earlier KV source with ratio {ratio}"
                    )
                if not any(
                    source <= layer_id and self.compress_ratios[source] == ratio
                    for source in self.index_source_layers
                ):
                    raise ValueError(
                        f"compressed-KV layer {layer_id} has no earlier index source with ratio {ratio}"
                    )
        elif self.kv_source_layers or self.index_source_layers:
            raise ValueError("compressed-KV sources were configured while every compression ratio is zero")

        if self.engram_layer_ids:
            if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
                raise ValueError("engram_num_embeddings must have one table size for each engram layer")
            if len(set(self.engram_layer_ids)) != len(self.engram_layer_ids):
                raise ValueError("engram_layer_ids cannot contain duplicates")
            if any(layer_id < 0 or layer_id >= self.n_layers for layer_id in self.engram_layer_ids):
                raise ValueError("Engram layers must refer to backbone layers")
            if min(self.engram_num_embeddings) <= 0:
                raise ValueError("Engram table sizes must be positive")
            if min(self.engram_max_ngram_size, self.engram_vocab_size, self.engram_n_heads, self.engram_head_dim) <= 0:
                raise ValueError("active Engram requires positive n-gram, vocabulary, head, and head-dimension settings")
            if self.engram_compressed_vocab_size <= 0:
                raise ValueError("active Engram requires engram_compressed_vocab_size")
        elif self.engram_num_embeddings:
            raise ValueError("engram_num_embeddings was configured without engram_layer_ids")

        if self.dspark_block_size:
            if self.n_mtp_layers <= 0:
                raise ValueError("active DSpark MTP requires at least one MTP layer")
            if not self.dspark_target_layer_ids:
                raise ValueError("active DSpark MTP requires dspark_target_layer_ids")
            if any(layer_id < 0 or layer_id >= self.n_layers for layer_id in self.dspark_target_layer_ids):
                raise ValueError("DSpark target layers must refer to backbone layers")
            if self.dspark_noise_token_id < 0 or self.dspark_noise_token_id >= self.vocab_size:
                raise ValueError("dspark_noise_token_id must be a valid tokenizer id")
        elif self.dspark_target_layer_ids:
            raise ValueError("dspark_target_layer_ids was configured while DSpark MTP is disabled")

    @property
    def vision_enabled(self) -> bool:
        return self.vision_n_layers > 0

    def get_moe_config(self, layer_id: int) -> tuple[int, int]:
        """Return the routed/activated expert counts for a given layer."""
        if layer_id < self.n_layers:
            return self.n_routed_experts, self.n_activated_experts
        return (
            self.dspark_n_routed_experts or self.n_routed_experts,
            self.dspark_n_activated_experts or self.n_activated_experts,
        )


class ParallelEmbedding(nn.Module):
    """Embedding sharded along the vocab dimension. Each rank holds vocab_size // world_size rows.
    Out-of-range indices are zero-masked before all_reduce to combine partial embeddings."""

    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, (
            f"Vocabulary size must be divisible by world size (world_size={world_size})"
        )
        self.part_vocab_size = vocab_size // world_size
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if world_size > 1:
            # ids off this rank read row 0 then get zeroed, so the all_reduce sums one real row
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Pick a GEMM from the weight dtype. Quantized weights need a quantized activation, and both
    fp4 and fp8 weights take an fp8 one -- for fp4 the kernel handles the mixed precision."""
    assert bias is None

    if weight.dtype == torch.float4_e2m1fn_x2:
        x, s = act_quant(x, fp8_block_size, scale_fmt, scale_dtype)
        return fp4_gemm(
            x,
            s,
            weight,
            weight.scale,
            scale_dtype,
            act_block_size=fp8_block_size,
        )
    elif weight.dtype == torch.float8_e4m3fn:
        x, s = act_quant(x, fp8_block_size, scale_fmt, scale_dtype)
        return fp8_gemm(
            x,
            s,
            weight,
            weight.scale,
            scale_dtype,
            block_size=fp8_block_size,
        )
    else:
        return F.linear(x.to(weight.dtype), weight)


class Linear(nn.Module):
    """bf16, fp8 or fp4 weights. Quantized ones get a `scale`, also attached to `.weight` so that
    `linear()` can reach it."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        dtype = dtype or default_dtype
        if dtype == torch.float4_e2m1fn_x2:
            # two values per byte: [out, in] logically, [out, in//2] stored
            # Packed fp4/fp8 weights are inference-only storage: their dtypes have no autograd
            # kernels for gradient accumulation. Production training uses bf16 Linears.
            self.weight = nn.Parameter(
                torch.zeros(out_features, in_features // 2, dtype=torch.int8).view(torch.float4_e2m1fn_x2),
                requires_grad=False,
            )
            # e8m0 has no autograd kernel, so block scales are calibrated constants, not
            # trainable parameters. Production training uses bf16 weights and has no scales.
            self.weight.scale = self.scale = nn.Parameter(
                torch.ones(out_features, in_features // fp4_block_size, dtype=torch.float8_e8m0fnu),
                requires_grad=False,
            )
        elif dtype == torch.float8_e4m3fn:
            self.weight = nn.Parameter(
                (torch.randn(out_features, in_features) * 0.02).to(dtype), requires_grad=False
            )
            self.weight.scale = self.scale = nn.Parameter(
                torch.ones(
                    (out_features + fp8_block_size - 1) // fp8_block_size,
                    (in_features + fp8_block_size - 1) // fp8_block_size,
                    dtype=torch.float8_e8m0fnu,
                ),
                requires_grad=False,
            )
        else:
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype))
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """Splits the output dim across ranks; each rank's slice of the output is already complete."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert out_features % world_size == 0, (
            f"Output features must be divisible by world size (world_size={world_size})"
        )
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class RowParallelLinear(Linear):
    """Splits the reduction dim, so each rank holds a partial sum: hence the fp32 all_reduce, with
    the bias added only after it."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype=None):
        assert in_features % world_size == 0, (
            f"Input features must be divisible by world size (world_size={world_size})"
        )
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, None)
        if world_size > 1:
            y = y.float()
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y.type_as(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


class ParallelEngramEmbedding(nn.Module):
    """The n-gram hash table, sharded over its rows.

    The table is stored in bf16. The previous implementation kept it in fp8 and cast the *entire*
    table to bf16 on every lookup, which both re-materialised the table per call and made the fp8
    leaf parameter untrainable. At 1.5B scale the whole table is only ~150 MB, so bf16 storage is
    the right trade: gather rows first, then cast.
    """

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dim = dim
        self.part_num_embeddings = (num_embeddings + world_size - 1) // world_size
        self.vocab_start_idx = rank * self.part_num_embeddings
        self.vocab_end_idx = self.vocab_start_idx + self.part_num_embeddings
        self.weight = nn.Parameter(torch.randn(self.part_num_embeddings, dim, dtype=torch.bfloat16) * 0.02)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        mask = (indices < self.vocab_start_idx) | (indices >= self.vocab_end_idx)
        local_indices = (indices - self.vocab_start_idx).masked_fill(mask, 0)
        values = F.embedding(local_indices, self.weight)
        values = values.masked_fill(mask.unsqueeze(-1), 0)
        if world_size > 1:
            dist.all_reduce(values)
        return values

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Older checkpoints stored an fp8 table plus per-block scales. Keep loading them, but the
        # in-memory representation is now bf16.
        state_dict.pop(prefix + "scale", None)
        weight_key = prefix + "weight"
        if weight_key in state_dict:
            state_dict[weight_key] = state_dict[weight_key].to(torch.bfloat16)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by how well it matches that stream.

    The hash ids fetch `n_hash_cols` rows; `wkv` turns them into one key per hc copy plus a shared
    value. The gate is a normalized dot product of stream against key.
    """

    def __init__(self, args: ModelArgs, layer_id: int, layout: EngramLayout):
        super().__init__()
        self.layer_id = layer_id
        self.layer_hash_index = layout.layer_ids.index(layer_id)
        self.dim = args.dim
        self.hc_mult = args.hc_mult
        self.clamp_value = 1e-6

        self.embed = ParallelEngramEmbedding(layout.num_embeddings[self.layer_hash_index], layout.head_dim)
        n_hash_cols = (layout.max_ngram_size - 1) * layout.n_heads
        self.wkv = Linear(n_hash_cols * layout.head_dim, args.dim * (args.hc_mult + 1))
        self.eps = args.norm_eps
        self.q_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))
        self.k_weight = nn.Parameter(torch.ones(args.hc_mult, args.dim))

    def forward(self, x: torch.Tensor, hash_ids: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, hc_mult, dim]; hash_ids: [B, L, n_hash_cols]; token_mask: [B, L], False shuts
        the gate so those positions pass through untouched."""
        kv = self.wkv(self.embed(hash_ids).flatten(-2))
        key, value = kv.split([self.hc_mult * self.dim, self.dim], dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.dim))
        weight = self.q_weight.float() * self.k_weight.float()  # only ever used as a product
        h, eps = x.float(), self.eps
        # normalized per (token, hc copy) over `dim`, NOT jointly over the copies
        rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
        dot = (h * weight * key).sum(-1) * rstd * self.dim**-0.5
        # signed sqrt before the sigmoid, matching the training kernel
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(self.clamp_value).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """Rotary frequencies as complex exponentials, one row per position.

    With original_seq_len > 0 this applies YaRN: dimensions whose wavelength already fits inside the
    training context keep their frequency, those far beyond it are divided by `factor`, and the
    `beta_fast`..`beta_slow` band in between is faded across with a linear ramp.
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        # the dim whose wavelength completes `rotations` turns over the training context
        def corrected_dim(rotations):
            return dim * math.log(original_seq_len / (rotations * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


# One RoPE table per (configuration, device), shared by every layer that uses it. Registering a
# copy as a per-layer buffer made `.to(device)` materialise one table per layer (~1.2 GB at 256K).
_FREQS_DEVICE_CACHE: dict[tuple, torch.Tensor] = {}


def get_shared_freqs(
    dim: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: int,
    beta_slow: int,
    device: torch.device,
) -> torch.Tensor:
    key = (
        dim,
        seqlen,
        original_seq_len,
        float(base),
        float(factor),
        int(beta_fast),
        int(beta_slow),
        device.type,
        device.index,
    )
    cached = _FREQS_DEVICE_CACHE.get(key)
    if cached is None:
        # Build the shared table outside inference mode: an inference tensor cached here would
        # later be unusable in a training forward (it cannot be saved for backward).
        with torch.inference_mode(False):
            cached = precompute_freqs_cis(
                dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow
            ).to(device)
        _FREQS_DEVICE_CACHE[key] = cached
    return cached


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate `x`, taking adjacent element pairs as complex numbers. Accepts [b, s, d] and
    [b, s, h, d]; `inverse` conjugates the rotation, which is how the attention output gets the
    query's rotation removed again so the cache can stay in one shared rotated form."""
    y = x
    freqs_cis = freqs_cis.to(x.device)
    xc = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if xc.ndim == 3:
        freqs_cis = freqs_cis.view(1, xc.size(1), xc.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, xc.size(1), 1, xc.size(-1))
    rotated = torch.view_as_real(xc * freqs_cis).flatten(-2).to(y.dtype)
    if not torch.is_grad_enabled():
        y.copy_(rotated)
        return y
    return rotated


def rotate_tail(x: torch.Tensor, rd: int, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Autograd-safe rotary embedding on the last `rd` dimensions of `x`."""
    if not torch.is_grad_enabled():
        apply_rotary_emb(x[..., -rd:], freqs_cis, inverse)
        return x
    rotated_tail = apply_rotary_emb(x[..., -rd:], freqs_cis, inverse)
    return torch.cat([x[..., :-rd], rotated_tail], dim=-1)


def get_window_topk_idxs(
    window_size: int,
    bsz: int,
    seqlen: int,
    start_pos: int,
    history_len: int | None = None,
):
    """Local indices into ``[window history | current chunk]`` for every query in the chunk.

    ``-1`` marks a slot that does not exist yet. The window is packed rather than returned as a
    full ring, so it works identically for single-token decode and multi-thousand-token chunks.
    """
    if history_len is None:
        history_len = min(start_pos, window_size)
    total = history_len + seqlen
    width = min(window_size, total)
    first_abs = start_pos - history_len
    end = torch.arange(start_pos, start_pos + seqlen, device="cpu").unsqueeze(1)
    begin = (end - window_size + 1).clamp(min=0)
    absolute = begin + torch.arange(width, device="cpu")
    local = absolute - first_abs
    local = torch.where(absolute > end, torch.full_like(local, -1), local)
    return local.int().unsqueeze(0).expand(bsz, -1, -1).contiguous()


class Compressor(nn.Module):
    """Pools `compress_ratio` consecutive tokens into one KV latent with a learned softmax gate.

    Returns the latent before RoPE, or None while a group is still filling up -- so during decode it
    only yields every `compress_ratio` steps, holding the partial group in `kv_state`/`score_state`.
    Pre-RoPE is deliberate: the indexer needs the unrotated form, so Attention rotates afterwards.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        compress_ratio = args.compress_ratios[layer_id]
        head_dim = args.head_dim
        self.compress_ratio = compress_ratio
        self.head_dim = head_dim
        self.norm = RMSNorm(head_dim, args.norm_eps)
        # ratio 1 is a plain projection, so it stays in the checkpoint's bf16; the softmax pooling
        # above ratio 1 runs in fp32, so those weights are promoted to fp32 to match
        self.wkv = Linear(args.dim, head_dim, dtype=torch.float32 if compress_ratio > 1 else torch.bfloat16)
        if compress_ratio > 1:
            self.wgate = Linear(args.dim, head_dim, dtype=torch.float32)
            # tail of an incomplete group, carried across decode steps
            self.kv_state: torch.Tensor
            self.score_state: torch.Tensor
            state_shape = (args.max_batch_size, compress_ratio, head_dim)
            self.register_buffer("kv_state", torch.zeros(state_shape, dtype=torch.float32), persistent=False)
            self.register_buffer(
                "score_state", torch.full(state_shape, -torch.inf, dtype=torch.float32), persistent=False
            )

    def forward(self, x: torch.Tensor, start_pos: int) -> tuple[torch.Tensor | None, int]:
        """Pool completed compression groups in this chunk.

        Returns ``(latents, first_group_index)``. ``latents`` is ``None`` when no group completes.
        Groups are aligned to absolute positions, so a group may span a chunk boundary; the partial
        tail waits in ``kv_state``/``score_state`` exactly as in the old decode-only path.
        """
        bsz, seqlen, _ = x.size()
        ratio, dtype = self.compress_ratio, x.dtype
        if ratio == 1:  # one token per group: nothing to pool, so no gate and no fp32
            return self.norm(self.wkv(x)), start_pos

        x = x.float()
        kv, score = self.wkv(x), self.wgate(x)
        if start_pos == 0:
            self.kv_state[:bsz].zero_()
            self.score_state[:bsz].fill_(-torch.inf)

        pooled: list[torch.Tensor] = []
        first_group: int | None = None
        idx = 0

        # Finish a group left partial by the previous chunk. The buffered part is detached (it is
        # state carried across chunks), while the current part keeps its gradient.
        slot = start_pos % ratio
        if slot:
            take = min(ratio - slot, seqlen)
            if slot + take == ratio:
                group_kv = torch.cat([self.kv_state[:bsz, :slot], kv[:, :take]], dim=1)
                group_score = torch.cat([self.score_state[:bsz, :slot], score[:, :take]], dim=1)
                pooled.append((group_kv * group_score.softmax(dim=1)).sum(dim=1, keepdim=True))
                first_group = (start_pos + take - ratio) // ratio
                idx = take
            else:
                self.kv_state[:bsz, slot : slot + take] = kv[:, :take].detach()
                self.score_state[:bsz, slot : slot + take] = score[:, :take].detach()
                idx = take

        # Complete groups fully inside this chunk.
        n_full = (seqlen - idx) // ratio
        if n_full:
            if first_group is None:
                first_group = (start_pos + idx) // ratio
            group_kv = kv[:, idx : idx + n_full * ratio].unflatten(1, (n_full, ratio))
            group_score = score[:, idx : idx + n_full * ratio].unflatten(1, (n_full, ratio))
            pooled.append((group_kv * group_score.softmax(dim=2)).sum(dim=2))
            idx += n_full * ratio

        # Trailing partial group waits for the next chunk.
        remainder = seqlen - idx
        if remainder:
            self.kv_state[:bsz, :remainder] = kv[:, idx:].detach()
            self.score_state[:bsz, :remainder] = score[:, idx:].detach()

        if not pooled:
            return None, (start_pos + seqlen) // ratio
        latent = torch.cat(pooled, dim=1)
        return self.norm(latent.to(dtype)), int(first_group)


class Indexer(torch.nn.Module):
    """Keeps the `index_topk` best compressed positions per query.

    A small side attention: fp4 query heads against one shared key per compressed position, scores
    rectified then combined by `weights_proj`. With a candidate source this is the second of two
    levels; `select_candidate_blocks` is the first.
    """

    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        # the index keys are derived from the compressor's latent, so only a layer that compresses
        # its own KV can produce them; every other indexer reads them from that layer's cache
        self.owns_k = layer_id in args.kv_source_layers
        self.compress_ratio = args.compress_ratios[layer_id]
        self.is_candidate_source = layer_id == args.candidate_source_layer
        self.uses_candidates = 0 <= args.candidate_source_layer < layer_id
        self.candidate_topk_blocks = args.candidate_topk_blocks
        self.candidate_block_size = args.candidate_block_size
        self.dim = args.dim
        self.n_heads = args.index_n_heads
        self.n_local_heads = args.index_n_heads // world_size
        self.index_head_dim = args.index_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.index_topk = args.index_topk
        self.index_query_stride = args.index_query_stride
        self.index_query_chunk = args.index_query_chunk
        self.q_lora_rank = args.q_lora_rank
        self.softmax_scale = self.index_head_dim**-0.5
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.index_head_dim)
        self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
        self.freqs_cis: torch.Tensor | None = None
        if self.owns_k:
            self.wk = Linear(args.head_dim, self.index_head_dim, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, args.norm_eps)
            self.k_cache: torch.Tensor
            self.register_buffer(
                "k_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // self.compress_ratio,
                    args.index_head_dim,
                    dtype=torch.bfloat16,
                ),
                persistent=False,
            )

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        latent: torch.Tensor | None,
        start_pos: int,
        offset: int,
        state: AttentionState | None = None,
        first_group: int = 0,
    ):
        """Return ``[b, seqlen, index_topk]`` local indices into the combined KV.

        ``latent`` holds compressed groups completed in this call (pre-RoPE). ``offset`` is the
        width of the sliding-window portion the indices are shifted past. Scoring is done in query
        chunks so the indexer never materialises ``[B, S, heads, S/ratio]``.
        """
        if state is None:
            state = AttentionState()
        assert self.freqs_cis is not None
        bsz, seqlen, _ = x.size()
        ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

        # Append index keys for every completed compression group.
        if self.owns_k and latent is not None and latent.size(1) > 0:
            positions = (
                torch.arange(first_group, first_group + latent.size(1), device=x.device) * ratio
            )
            k = self.k_norm(self.wk(latent))
            k = rotate_tail(k, rd, self.freqs_cis[positions])
            fp4_act_quant(k, fp4_block_size, True)
            with torch.no_grad():
                self.k_cache[:bsz, first_group : first_group + k.size(1)] = k.detach().to(
                    self.k_cache.dtype
                )
        if self.owns_k:
            state.index_k = self.k_cache[:bsz]

        n_keys = end_pos // ratio
        index_k = state.index_k
        empty = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        if index_k is None or n_keys == 0:
            return empty
        index_k = index_k[:, :n_keys].to(qr.dtype)

        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
        q = rotate_tail(q, rd, self.freqs_cis[start_pos:end_pos])
        fp4_act_quant(q, fp4_block_size, True)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)

        topk = min(self.index_topk, n_keys)
        if topk == 0:
            return empty
        stride = max(1, self.index_query_stride)
        chunk = max(1, self.index_query_chunk)
        key_positions = torch.arange(n_keys, device=x.device).view(1, 1, n_keys)
        candidate_parts: list[torch.Tensor] = []
        idx_parts: list[torch.Tensor] = []

        for q0 in range(0, seqlen, chunk):
            q1 = min(q0 + chunk, seqlen)
            qs = q[:, q0:q1:stride]
            ws = weights[:, q0:q1:stride]
            n_query = qs.size(1)

            score = torch.einsum("bqhd,btd->bqht", qs, index_k)
            score = (score.relu_() * ws.unsqueeze(-1)).sum(dim=2)
            if world_size > 1:
                dist.all_reduce(score)

            # A group is visible once the query has passed its last token.
            query_offsets = torch.arange(n_query, device=x.device) * stride
            lens = ((start_pos + q0 + query_offsets) + 1) // ratio
            lens = lens.view(1, n_query, 1)
            score = score.masked_fill(key_positions >= lens, -torch.inf)

            if self.is_candidate_source:
                keep = select_candidate_blocks(
                    score, lens, self.candidate_topk_blocks, self.candidate_block_size
                )
                candidate_parts.append(keep.repeat_interleave(stride, dim=1)[:, : q1 - q0])
            elif self.uses_candidates and state.candidates is not None:
                score = score.masked_fill(~state.candidates[:, q0:q1:stride, :n_keys], -torch.inf)

            idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            idxs = torch.where(idxs < lens, idxs + offset, -1)
            idxs = idxs.repeat_interleave(stride, dim=1)[:, : q1 - q0]
            idx_parts.append(idxs)

        if self.is_candidate_source and candidate_parts:
            state.candidates = torch.cat(candidate_parts, dim=1)
        return torch.cat(idx_parts, dim=1).int()


def select_candidate_blocks(
    logits: torch.Tensor,
    compress_lens: torch.Tensor | int,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level one of the two-level top-k: keep the `topk_blocks` highest-scoring blocks per query.

    `logits` is [..., n_positions] with positions the query cannot reach already at -inf, which is
    what makes a block score of -inf mean "not reachable yet". `compress_lens` is a plain int during
    decode, or broadcasts against logits' leading dims during prefill. Returns a bool mask shaped
    like `logits`, so the layers consuming it just mask and never think about blocks again.
    """
    width = logits.size(-1)
    # score each block by its best position; -inf pads the last one out to block_size
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)

    # the block with this query's newest position is only partly filled, so pin it in: it holds the
    # most recent tokens but could otherwise be outscored by an older, full block
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)

    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    # fewer reachable blocks than topk_blocks means leftover picks came back -inf: drop them
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class Attention(nn.Module):
    """Latent attention over two KV sources at once, concatenated into one `sparse_attn` call: a
    sliding window of raw KV, plus -- when compress_ratio > 0 -- `index_topk` compressed positions
    reaching further back. Q and the output projection are both low-rank, the latter grouped.

    compress_ratio > 0 does not mean the layer compresses its own KV: only kv_source_layers do,
    the rest read that same cache.
    """

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.n_local_heads = args.n_heads // world_size
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        self.head_dim = args.head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = args.head_dim - args.rope_head_dim
        self.n_groups = args.o_groups
        self.n_local_groups = self.n_groups // world_size
        self.window_size = args.window_size
        self.compress_ratio = args.compress_ratios[layer_id]
        self.eps = args.norm_eps

        self.attn_sink = nn.Parameter(torch.zeros(self.n_local_heads, dtype=torch.float32))
        self.wq_a = Linear(self.dim, self.q_lora_rank)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.head_dim)
        self.wkv = Linear(self.dim, self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * args.o_lora_rank,
            dtype=torch.bfloat16,
        )
        self.wo_b = RowParallelLinear(self.n_groups * args.o_lora_rank, self.dim)
        self.softmax_scale = self.head_dim**-0.5

        is_backbone = layer_id < args.n_layers
        self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
        self.is_index_source = is_backbone and layer_id in args.index_source_layers
        self.compressor: Compressor | None = None
        self.indexer: Indexer | None = None
        if self.is_kv_source and self.compress_ratio > 0:
            self.compressor = Compressor(args, layer_id)
        if self.is_index_source and self.compress_ratio > 0:
            self.indexer = Indexer(args, layer_id)

        self.window_kv_cache: torch.Tensor
        self.register_buffer(
            "window_kv_cache",
            torch.zeros(args.max_batch_size, args.window_size, self.head_dim, dtype=torch.bfloat16),
            persistent=False,
        )
        if self.is_kv_source:
            self.compress_kv_cache: torch.Tensor
            self.register_buffer(
                "compress_kv_cache",
                torch.zeros(
                    args.max_batch_size,
                    args.max_seq_len // self.compress_ratio,
                    self.head_dim,
                    dtype=torch.bfloat16,
                ),
                persistent=False,
            )
        if self.compress_ratio:
            original_seq_len, rope_theta = (
                args.original_seq_len,
                args.compress_rope_theta,
            )
        else:
            # disable YaRN and use base rope_theta in pure sliding-window attention
            original_seq_len, rope_theta = 0, args.rope_theta
        self.rope_original_seq_len = original_seq_len
        self.rope_theta = rope_theta
        self.rope_factor = args.rope_factor
        self.beta_fast = args.beta_fast
        self.beta_slow = args.beta_slow
        self.max_seq_len = args.max_seq_len
        self.freqs_cis: torch.Tensor | None = None

    def _freqs(self, device: torch.device) -> torch.Tensor:
        if self.freqs_cis is None or self.freqs_cis.device != device:
            self.freqs_cis = get_shared_freqs(
                self.rope_head_dim,
                self.max_seq_len,
                self.rope_original_seq_len,
                self.rope_theta,
                self.rope_factor,
                self.beta_fast,
                self.beta_slow,
                device,
            )
        return self.freqs_cis

    def _window_kv(self, x, start_pos):
        """This layer's sliding-window K and local window indices for every query in the chunk.

        The ring buffer is read first (so the current chunk cannot overwrite history it still
        needs) and then updated with the newest ``window_size`` rows. The returned KV is compact:
        ``[history | current chunk]``, which works for a single decode token and for a chunk.
        """
        bsz, seqlen, _ = x.size()
        win = self.window_size
        freqs_cis = self._freqs(x.device)[start_pos : start_pos + seqlen]
        kv = self.kv_norm(self.wkv(x))
        kv = rotate_tail(kv, self.rope_head_dim, freqs_cis)
        act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)

        history_len = min(start_pos, win)
        history = None
        if history_len:
            positions = torch.arange(start_pos - history_len, start_pos, device=x.device)
            history = self.window_kv_cache[:bsz].index_select(1, positions % win)

        tail = min(seqlen, win)
        if tail:
            positions = torch.arange(start_pos + seqlen - tail, start_pos + seqlen, device=x.device)
            with torch.no_grad():
                self.window_kv_cache[:bsz, positions % win] = kv[:, -tail:].detach().to(
                    self.window_kv_cache.dtype
                )

        window_kv = kv if history is None else torch.cat([history.to(kv.dtype), kv], dim=1)
        idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos, history_len).to(x.device)
        return window_kv, idxs

    def _compress_topk_idxs(self, x, qr, latent, start_pos, offset, first_group, compress_len, state):
        """Which compressed positions each query attends to. Index sources run their own indexer;
        the layers in between reuse the result their source published for this chunk."""
        bsz, seqlen, _ = x.size()
        if not self.is_index_source:
            return state.topk_idxs
        if compress_len == 0:
            idxs = torch.empty(bsz, seqlen, 0, dtype=torch.int32, device=x.device)
        else:
            assert self.indexer is not None
            self.indexer.freqs_cis = self._freqs(x.device)
            idxs = self.indexer(x, qr, latent, start_pos, offset, state, first_group)
        state.topk_idxs = idxs
        return idxs

    def _compress_kv(self, x, qr, start_pos, offset, state):
        """The shared compressed KV and the compressed positions every query may attend to. This
        layer compresses its own KV only when it is a source; otherwise it reads the source cache.
        Completed groups are always written to the persistent cache, including during prefill."""
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        compress_len = (start_pos + seqlen) // ratio
        latent, first_group = None, 0
        if self.is_kv_source:
            assert self.compressor is not None
            latent, first_group = self.compressor(x, start_pos)

        # The indexer needs the pre-RoPE latent, so index keys are produced before the cache write.
        idxs = self._compress_topk_idxs(
            x, qr, latent, start_pos, offset, first_group, compress_len, state
        )

        if latent is not None and latent.size(1) > 0:
            positions = torch.arange(first_group, first_group + latent.size(1), device=x.device) * ratio
            latent = rotate_tail(latent, self.rope_head_dim, self._freqs(x.device)[positions])
            # Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.
            fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)
            with torch.no_grad():
                self.compress_kv_cache[:bsz, first_group : first_group + latent.size(1)] = (
                    latent.detach().to(self.compress_kv_cache.dtype)
                )
            # History is detached state, but the current chunk's latents stay on the graph so
            # the compressor and its value path receive gradients within the chunk.
            history = self.compress_kv_cache[:bsz, :first_group].to(latent.dtype)
            state.compress_kv = torch.cat([history, latent.to(history.dtype)], dim=1)
        elif self.is_kv_source:
            state.compress_kv = self.compress_kv_cache[:bsz, :compress_len]
        if state.compress_kv is None:
            state.compress_kv = x.new_empty((bsz, 0, self.head_dim))
        return state.compress_kv, idxs

    def forward(self, x: torch.Tensor, start_pos: int, state: AttentionState | None = None):
        if state is None:
            state = AttentionState()
        bsz, seqlen, _ = x.size()
        freqs_cis = self._freqs(x.device)[start_pos : start_pos + seqlen]
        rd = self.rope_head_dim

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = rotate_tail(q, rd, freqs_cis)

        kv, topk_idxs = self._window_kv(x, start_pos)
        if self.compress_ratio:
            compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1), state)
            if compress_kv.size(1):
                kv = torch.cat([kv, compress_kv.to(kv.dtype)], dim=1)
            if compress_idxs.size(-1):
                topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        o = rotate_tail(o, rd, freqs_cis, True)

        # wo_a is block-diagonal over groups (each projects only its own heads), hence einsum not
        # Linear. convert.py dequantizes it to bf16; quantize/dequantize o before einsum.
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o.to(wo_a.dtype), wo_a)
        x = self.wo_b(o.flatten(2).to(x.dtype))
        return x


class Gate(nn.Module):
    """MoE gating. The correction bias steers expert selection only; the routing weights come from the
    unbiased scores. Image-span tokens use a separate bias (training `noaux_tc_for_vl`)."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.dim = args.dim
        self.topk = n_activated_experts
        self.score_func = args.score_func
        self.gate_temp = args.gate_temp
        self.norm_topk_prob = args.norm_topk_prob
        self.route_scale = args.route_scale
        self.weight = nn.Parameter(torch.empty(n_routed_experts, args.dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        self.bias = nn.Parameter(torch.zeros(n_routed_experts, dtype=torch.float32))
        self.bias_vl = nn.Parameter(torch.zeros(n_routed_experts, dtype=torch.float32)) if args.vision_enabled else None

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [n, dim]; image_mask: [n] bool, True for tokens inside an image span."""
        scores = linear(x.float(), self.weight.float()) / self.gate_temp
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
        bias = self.bias
        if image_mask is not None and self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)
        # the bias picks experts but does not scale them: weights come from the raw scores
        indices = (scores + bias).topk(self.topk, dim=-1)[1]
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights /= weights.sum(dim=-1, keepdim=True) + 1e-20  # not norm_eps, matches training
        weights *= self.route_scale
        return weights, indices


class Expert(nn.Module):
    """One SwiGLU FFN. The clamps come straight from training, where they keep fp8/fp4 activations in
    range: the up branch is clamped on both sides, the gate branch only from above."""

    def __init__(self, dim: int, inter_dim: int, dtype=None, swiglu_limit=0.0):
        super().__init__()
        self.w1 = Linear(dim, inter_dim, dtype=dtype)
        self.w2 = Linear(inter_dim, dim, dtype=dtype)
        self.w3 = Linear(dim, inter_dim, dtype=dtype)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    """Top-k routed experts plus one shared expert every token goes through. Experts are split
    across ranks, so `self.experts` is None for those another rank owns."""

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        n_routed_experts, n_activated_experts = args.get_moe_config(layer_id)
        self.layer_id = layer_id
        self.dim = args.dim
        assert n_routed_experts % world_size == 0, (
            f"Number of experts must be divisible by world size (world_size={world_size})"
        )
        self.n_routed_experts = n_routed_experts
        self.n_local_experts = n_routed_experts // world_size
        self.n_activated_experts = n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args) if self.n_routed_experts > 0 else None
        expert_dtype = torch.float4_e2m1fn_x2 if args.expert_dtype == "fp4" else None
        self.experts = nn.ModuleList(
            [
                Expert(
                    args.dim,
                    args.moe_inter_dim,
                    dtype=expert_dtype,
                    swiglu_limit=args.swiglu_limit,
                )
                if self.experts_start_idx <= i < self.experts_end_idx
                else None
                for i in range(self.n_routed_experts)
            ]
        )
        assert args.n_shared_experts == 1
        self.shared_experts = Expert(args.dim, args.moe_inter_dim, swiglu_limit=args.swiglu_limit)

    def forward(self, x: torch.Tensor, image_mask: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        if self.n_routed_experts == 0 or self.gate is None:
            return self.shared_experts(x).type_as(x).view(shape)
        weights, indices = self.gate(x, None if image_mask is None else image_mask.flatten())
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y = y.index_add(0, idx, expert(x[idx], weights[idx, top, None]).float())
        if world_size > 1:
            dist.all_reduce(y)
        y = y + self.shared_experts(x).float()
        return y.type_as(x).view(shape)


class Block(nn.Module):
    """A block whose residual stream is `hc_mult` parallel copies (Hyper-Connections).

    Attention and FFN each sit between `hc_pre` (collapse the copies into one sublayer input) and
    `hc_post` (expand back out, mixing the residual in through `comb`). `hc_mixes` derives all three
    coefficient sets from the stream itself, `comb` made doubly stochastic by Sinkhorn.

    The coefficients a sublayer computes are used by the *next* one -- see `forward`.
    """

    attention_cls = Attention
    gradient_checkpointing = False

    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        engram_layout: EngramLayout | None = None,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = args.norm_eps
        self.attn = self.attention_cls(layer_id, args)
        self.ffn = MoE(layer_id, args)
        self.engram = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(args, layer_id, engram_layout)
        self.attn_norm = RMSNorm(args.dim, self.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, self.norm_eps)
        self.hc_mult = hc_mult = args.hc_mult
        self.hc_sinkhorn_iters = args.hc_sinkhorn_iters
        self.hc_eps = args.hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * args.dim
        with set_dtype(torch.float32):
            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim))
            nn.init.normal_(self.hc_attn_fn, mean=0.0, std=0.02)
            nn.init.normal_(self.hc_ffn_fn, mean=0.0, std=0.02)
            self.hc_attn_base = nn.Parameter(torch.zeros(mix_hc))
            self.hc_ffn_base = nn.Parameter(torch.zeros(mix_hc))
            self.hc_attn_scale = nn.Parameter(torch.ones(3))
            self.hc_ffn_scale = nn.Parameter(torch.ones(3))

    def hc_mixes(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        """x: [b,s,hc,d], hc_fn: [mix_hc, hc*d], hc_scale: [3], hc_base: [mix_hc]. Returns the
        pre / post / comb coefficients, split out of one projection of the flattened stream."""
        # normalized over the whole flattened hc*d stream, one statistic per token
        x_flat = x.flatten(2)
        rsqrt = torch.rsqrt(x_flat.float().square().mean(-1, keepdim=True) + self.norm_eps).to(x_flat.dtype)
        mixes = F.linear(x_flat.to(hc_fn.dtype), hc_fn) * rsqrt.to(hc_fn.dtype)
        return hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor):
        """Collapse the hc copies into one, weighted by pre_mix. [b,s,hc,d] x [b,s,hc] -> [b,s,d]"""
        y = torch.sum(pre_mix.unsqueeze(-1) * x.float(), dim=2)
        return y.to(x.dtype)

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        """Expand the sublayer output back to hc copies and mix the residual in through `comb`.
        x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc] -> [b,s,hc,d]"""
        # einsum avoids materialising the [b, s, hc, hc, d] broadcast product.
        mixed = torch.einsum("bsij,bsjd->bsid", comb.float(), residual.float())
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + mixed
        return y.type_as(x)

    def _ffn_forward(self, x: torch.Tensor, image_mask: torch.Tensor | None) -> torch.Tensor:
        return self.ffn(x, image_mask)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        pre_mix: torch.Tensor,
        image_mask: torch.Tensor | None,
        attn_state: AttentionState | None = None,
        *attn_args,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`pre_mix` collapses the hc_mult copies down to one input for this block's attention. Each
        sub-block's own `hc_mixes` produces the mix for the *next* one, so attention uses what the
        previous layer's FFN produced and the FFN uses what this attention produced.

        image_mask: [b, s] bool, True inside image spans (selects the VL routing bias)."""
        residual = x
        attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x = self.hc_pre(x, pre_mix)
        x = self.attn_norm(x)
        x = self.attn(x, start_pos, attn_state, *attn_args)
        x = self.hc_post(x, residual, attn_post, attn_comb)

        residual = x
        ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x = self.hc_pre(x, attn_pre)
        x = self.ffn_norm(x)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # Attention is deliberately not checkpointed: recomputation would replay cache writes.
            x = torch.utils.checkpoint.checkpoint(self._ffn_forward, x, image_mask, use_reentrant=False)
        else:
            x = self._ffn_forward(x, image_mask)
        x = self.hc_post(x, residual, ffn_post, ffn_comb)
        return x, ffn_pre


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.part_vocab_size = vocab_size // world_size
        # bf16 in the checkpoint, kept as fp32 here so the logits come out in fp32 directly
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim, dtype=torch.get_default_dtype()))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, full_logits=False):
        """x: [b, s, d]. Generation only needs the last position, so that is the default."""
        if not full_logits:
            x = x[:, -1]
        x = x.to(self.weight.device)
        logits = F.linear(x.to(self.weight.dtype), self.weight)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return logits


def get_dspark_topk_idxs(window_width: int, bsz: int, block_size: int):
    """Indices into ``[packed window | current accepted state | draft keys]``.

    ``window_width`` covers existing history plus the current accepted state; draft keys follow
    immediately after it. The width grows from 1 to ``window_size`` early in a sequence, so this
    cannot be a fixed-offset table.
    """
    matrix = torch.cat(
        [
            torch.arange(window_width),
            window_width + torch.arange(block_size),
        ]
    )
    return matrix.int().view(1, 1, -1).expand(bsz, block_size, -1).contiguous()


class DSparkAttention(Attention):
    def forward(self, x: torch.Tensor, start_pos: int, main_x: torch.Tensor):
        assert self.compress_ratio == 0
        bsz, seqlen, _ = main_x.size()
        win = self.window_size
        rd = self.rope_head_dim

        main_freqs_cis = self._freqs(x.device)[start_pos : start_pos + seqlen]
        main_kv = self.kv_norm(self.wkv(main_x))
        apply_rotary_emb(main_kv[..., -rd:], main_freqs_cis)
        act_quant(main_kv, fp8_block_size, scale_fmt, scale_dtype, True)

        if start_pos == 0:
            # Cache writes are carried state: detach so a later chunk's backward never tries to
            # traverse this chunk's (already freed) graph through the buffer.
            main_kv = main_kv.detach().to(self.window_kv_cache.dtype)
            if seqlen <= win:
                self.window_kv_cache[:bsz, :seqlen] = main_kv
            else:
                cutoff = seqlen % win
                self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = main_kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            return x

        bsz, block_size, _ = x.size()
        freqs_cis = self._freqs(x.device)[start_pos + seqlen : start_pos + seqlen + block_size]

        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.kv_norm(self.wkv(x))
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        act_quant(kv, fp8_block_size, scale_fmt, scale_dtype, True)

        # Pack the ring buffer into chronological order: up to ``win - 1`` cached positions plus the
        # current accepted state. The current row stays on the autograd graph (it conditions the
        # draft and must train main_proj), while the carried cache entry is detached so a later
        # chunk never traverses this chunk's graph.
        history_len = min(start_pos, win - 1)
        if history_len:
            positions = torch.arange(start_pos - history_len, start_pos, device=x.device)
            history = self.window_kv_cache[:bsz].index_select(1, positions % win)
        else:
            history = None
        current = main_kv.squeeze(1)
        with torch.no_grad():
            self.window_kv_cache[:bsz, start_pos % win] = current.detach().to(
                self.window_kv_cache.dtype
            )
        window_kv = current.unsqueeze(1) if history is None else torch.cat(
            [history.to(current.dtype), current.unsqueeze(1)], dim=1
        )
        kv = torch.cat([window_kv, kv.to(window_kv.dtype)], dim=1)
        topk_idxs = get_dspark_topk_idxs(window_kv.size(1), bsz, block_size).to(x.device)
        o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, block_size, self.n_local_groups, -1)
        act_quant(o, fp8_block_size, scale_fmt, scale_dtype, True)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, dspark_markov_rank: int):
        super().__init__()
        self.embed = ParallelEmbedding(vocab_size, dspark_markov_rank)
        self.head = ParallelHead(vocab_size, dspark_markov_rank)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed = self.embed(token_ids)
        logits = self.head(embed, full_logits=True)
        return logits, embed


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        # proj in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for fp32 confidence score.
        self.proj = Linear(input_dim, 1, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor, markov_embed: torch.Tensor):
        hidden = torch.cat([hidden, markov_embed], dim=-1)
        return self.proj(hidden.float()).squeeze(-1)


class DSparkBlock(Block):
    """DSpark stage stored under the mtp.* checkpoint namespace."""

    attention_cls = DSparkAttention

    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        stage_id = layer_id - args.n_layers
        self.block_size = args.dspark_block_size
        self.noise_token_id = args.dspark_noise_token_id
        self.temperature = args.temperature
        if stage_id == 0:
            assert len(args.dspark_target_layer_ids) > 0, "DSpark needs target layers"
            self.main_proj = Linear(args.dim * len(args.dspark_target_layer_ids), args.dim)
            self.main_norm = RMSNorm(args.dim, args.norm_eps)
        if stage_id == args.n_mtp_layers - 1:
            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, args.dspark_markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim + args.dspark_markov_rank)
        self.embed: ParallelEmbedding | None = None
        self.head: ParallelHead | None = None

    def forward(self, x: torch.Tensor, start_pos: int, pre_mix: torch.Tensor, main_x: torch.Tensor):
        if start_pos == 0:
            self.attn(x, start_pos, main_x)  # prefill only seeds the window KV cache
            return x, pre_mix
        return super().forward(x, start_pos, pre_mix, None, main_x)  # drafts are text: no VL bias

    def forward_embed(self, main_hidden: torch.Tensor, input_ids: torch.Tensor):
        assert self.embed is not None
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_input_ids = input_ids.new_full([input_ids.size(0), self.block_size], self.noise_token_id)
        draft_input_ids[:, 0] = input_ids
        x = self.embed(draft_input_ids)
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return x, main_x

    def forward_head(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert self.head is not None
        x = self.hc_pre(x, pre_mix)
        logits = self.head(self.norm(x), full_logits=True)
        output_ids = input_ids.new_empty(input_ids.size(0), self.block_size + 1)
        output_ids[:, 0] = input_ids
        markov_embeds = []
        for i in range(self.block_size):
            logits_bias, markov_embed = self.markov_head(output_ids[:, i])
            logits[:, i].add_(logits_bias)
            markov_embeds.append(markov_embed)
            output_ids[:, i + 1] = sample(logits[:, i], self.temperature)
        markov_embed = torch.stack(markov_embeds, dim=1)
        confidence = self.confidence_head(x, markov_embed)
        return output_ids, logits, confidence

    def forward_train_head(
        self,
        x: torch.Tensor,
        pre_mix: torch.Tensor,
        input_ids: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return teacher-forced draft logits and confidence scores.

        ``input_ids`` is the token accepted by the main model.  ``targets``
        contains the following ``block_size`` tokens.  During training the
        Markov path receives the previous ground-truth target, while inference
        uses sampled drafts in :meth:`forward_head`.
        """
        assert self.head is not None
        if targets.ndim != 2 or targets.size(1) != self.block_size:
            raise ValueError("DSpark targets must be [batch, dspark_block_size]")
        x = self.hc_pre(x, pre_mix)
        logits = self.head(self.norm(x), full_logits=True)
        previous_ids = torch.cat([input_ids.unsqueeze(1), targets[:, :-1]], dim=1)
        markov_logits, markov_embed = self.markov_head(previous_ids)
        logits = logits + markov_logits
        confidence = self.confidence_head(x, markov_embed)
        return logits, confidence


def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """initial one-hot mix"""
    pre_mix = x.new_zeros(x.size(0), x.size(1), hc_mult, dtype=torch.float32)
    pre_mix[:, :, 0] = 1.0
    return pre_mix


class Transformer(nn.Module):
    """NOESIS-Σ Transformer: embed -> expand to hc_mult copies -> blocks (with mid-network
    OSCLatentAttractor continuous thinking block) -> collapse -> logits. Building
    this sets the globals at the top of the file. The tokenizer only feeds the engram token map."""

    def __init__(self, args: ModelArgs, tokenizer=None):
        global world_size, rank, default_dtype
        if not MP_LAYOUT_CONFIGURED:
            # Standalone construction (tests, inference) is logically unsharded. Distributed
            # training must call configure_model_parallel explicitly; FSDP sharding does not
            # imply model-parallel weight splitting.
            world_size, rank = 1, 0
        if world_size > 1:
            if args.n_heads % world_size:
                raise ValueError(f"n_heads={args.n_heads} is not divisible by mp_size={world_size}")
            if args.o_groups % world_size:
                raise ValueError(f"o_groups={args.o_groups} is not divisible by mp_size={world_size}")
            if args.dspark_block_size and (args.dspark_n_routed_experts or args.n_routed_experts) % world_size:
                raise ValueError("DSpark expert count is not divisible by mp_size")
        default_dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        args.validate_for_model()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.temperature = args.temperature
        self.norm_eps = args.norm_eps
        self.hc_eps = args.hc_eps
        self.engram_layout = EngramLayout.from_args(args)
        if self.engram_layout is not None:
            for layer_id, table_rows, primes in zip(
                self.engram_layout.layer_ids,
                self.engram_layout.num_embeddings,
                self.engram_layout.primes,
                strict=True,
            ):
                required_rows = sum(sum(head_buckets) for head_buckets in primes)
                if table_rows < required_rows:
                    raise ValueError(
                        f"Engram table for layer {layer_id} needs at least {required_rows} rows "
                        f"but config provides {table_rows}"
                    )
        self.engram_hash = (
            NgramHashState(args, self.engram_layout, tokenizer) if self.engram_layout is not None else None
        )
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args, self.engram_layout))
        self.osc_thinking_block = (
            OSCLatentAttractor(
                dim=args.dim,
                rank=args.osc_rank,
                icnn_hidden=args.osc_icnn_hidden,
                n_steps=args.osc_steps,
                dt=args.osc_dt,
                c_min=args.osc_c_min,
                max_norm=args.osc_max_norm,
                enable_retrograde=args.osc_retrograde,
            )
            if args.osc_enabled
            else None
        )
        self.last_osc_metrics: dict = {}
        self.norm = RMSNorm(args.dim, self.norm_eps)
        self.head = ParallelHead(args.vocab_size, args.dim, self.norm_eps, self.hc_eps)
        self.mtp = torch.nn.ModuleList()
        self.target_layer_ids = args.dspark_target_layer_ids
        if args.dspark_block_size:
            for layer_id in range(args.n_mtp_layers):
                self.mtp.append(DSparkBlock(args.n_layers + layer_id, args))
                self.mtp[-1].embed = self.embed
                self.mtp[-1].head = self.head
        self.hc_mult = args.hc_mult
        self.vision = None
        if args.vision_enabled:
            self.vision = ViT(args)
            self.aligner = Aligner(args)
            # learned embeddings for the image span delimiters
            self.image_start = nn.Parameter(torch.zeros(args.dim))
            self.image_end = nn.Parameter(torch.zeros(args.dim))
            self.image_newline = nn.Parameter(torch.zeros(args.dim))

    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(self, images, h: torch.Tensor):
        """Overwrite each image's token span in h with its ViT/aligner features. The IMAGE slots take
        the aligner rows in row-major order; the span delimiters take learned embeddings."""
        for i, sample in enumerate(images):
            for img in sample or ():
                types = img.types.to(h.device)
                span = h[i, img.start : img.start + types.numel()]
                span[types == IMAGE_START] = self.image_start.to(h.dtype)
                span[types == IMAGE_END] = self.image_end.to(h.dtype)
                span[types == IMAGE_NEW_LINE] = self.image_newline.to(h.dtype)
                embeds = self.encode_image(img.patches.to(h.device), img.n_vit_h, img.n_vit_w)
                span[types == IMAGE] = embeds.to(h.dtype)

    def _run_backbone(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        images=None,
        token_types: torch.Tensor | None = None,
        state: AttentionState | None = None,
        retro_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], dict, AttentionState, torch.Tensor | None]:
        if state is None:
            state = AttentionState()
        if input_ids.size(1) == 0:
            raise ValueError("input chunk cannot be empty")
        if start_pos < 0 or start_pos + input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"chunk [{start_pos}, {start_pos + input_ids.size(1)}) exceeds max_seq_len={self.max_seq_len}"
            )
        embed_dev = self.embed.weight.device
        if input_ids.device != embed_dev:
            input_ids = input_ids.to(embed_dev)
        image_mask = None if token_types is None else token_types >= 0  # TEXT is -1
        engram_mask = None if image_mask is None else ~image_mask
        engram_hashes = self.engram_hash(input_ids, start_pos, engram_mask) if self.engram_hash is not None else None
        h = self.embed(input_ids)
        if images is not None:
            assert start_pos == 0, "image spans must be prefilled in a single chunk"
            self.merge_image_embeddings(images, h)
        # Expand to hc_mult copies for Hyper-Connections
        h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        main_hiddens = []
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        mid_layer_idx = max(0, len(self.layers) // 2 - 1)
        osc_metrics: dict = {}
        next_retro = retro_state
        for i, layer in enumerate(self.layers):
            layer_dev = next(layer.parameters()).device
            if h.device != layer_dev:
                h = h.to(layer_dev)
            if pre_mix.device != layer_dev:
                pre_mix = pre_mix.to(layer_dev)
            if image_mask is not None and image_mask.device != layer_dev:
                image_mask = image_mask.to(layer_dev)
            if layer.engram is not None:
                h = layer.engram(h, engram_hashes[:, :, layer.engram.layer_hash_index, :].to(layer_dev), engram_mask)
            # the MTP head reads the attention input of its target layers, not their output
            if i in self.target_layer_ids:
                main_hiddens.append(h.mean(dim=2))
            h, pre_mix = layer(h, start_pos, pre_mix, image_mask, state)
            if self.osc_thinking_block is not None and i == mid_layer_idx:
                osc_dev = next(self.osc_thinking_block.parameters()).device
                h_collapsed = h.mean(dim=2).to(osc_dev)
                s_relaxed, osc_metrics, next_retro = self.osc_thinking_block.forward_with_state(
                    h_collapsed, next_retro
                )
                delta = (s_relaxed - h_collapsed).unsqueeze(2)
                h = (h.to(osc_dev) + delta).to(h.dtype)
        self.last_osc_metrics = osc_metrics
        h_out = layer.hc_pre(h, pre_mix)
        norm_dev = next(self.norm.parameters()).device
        h_norm = self.norm(h_out.to(norm_dev))
        return h_norm, main_hiddens, osc_metrics, state, next_retro

    def distribute_to_gpus(self, devices=None):
        """Pipelines NOESIS-Σ across both GPUs (cuda:0 and cuda:1) to pool all 32 GB VRAM:
        - cuda:0: embed, layers 0 to 8 (first 9 layers)
        - cuda:1: layers 9 to 17 (last 9 layers), osc_thinking_block, norm, head
        """
        if devices is None:
            assert torch.cuda.device_count() >= 2, "distribute_to_gpus requires at least 2 CUDA GPUs!"
            devices = ("cuda:0", "cuda:1")
        dev0, dev1 = torch.device(devices[0]), torch.device(devices[1])
        self.embed.to(dev0)
        for name in ("image_embed", "vit", "vision", "image_aligner", "engram_hash"):
            sub = getattr(self, name, None)
            if isinstance(sub, torch.nn.Module):
                sub.to(dev0)
        split_idx = len(self.layers) // 2
        for i in range(split_idx):
            self.layers[i].to(dev0)
        for i in range(split_idx, len(self.layers)):
            self.layers[i].to(dev1)
        if self.osc_thinking_block is not None:
            self.osc_thinking_block.to(dev1)
        self.norm.to(dev1)
        self.head.to(dev1)
        for mtp_layer in self.mtp:
            mtp_layer.to(dev1)
        print(f"[+] Pipelined NOESIS-Σ across both GPUs: Layers 0-{split_idx-1} on cuda:0, Layers {split_idx}-{len(self.layers)-1} on cuda:1 (All 32 GB Active!)")

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        images=None,
        token_types: torch.Tensor | None = None,
        state: ChunkState | None = None,
        return_chunk: bool = False,
    ):
        """Autoregressive inference / generation forward pass returning (output_ids, logits, main_hidden).

        When ``state`` is given (or ``return_chunk=True``) this dispatches to :meth:`forward_chunk`
        and returns ``(h_norm, state, osc_metrics, mtp_hiddens)``. Training must go through
        ``__call__`` (this method) rather than calling ``forward_chunk`` directly, because FSDP
        installs the root module's pre/post-forward hooks on ``__call__``.
        """
        if state is not None or return_chunk:
            return self.forward_chunk(
                input_ids,
                start_pos=start_pos,
                state=state,
                token_types=token_types,
                images=images,
            )
        h_norm, main_hiddens, _, _, _ = self._run_backbone(input_ids, start_pos, images, token_types)
        logits = self.head(h_norm, full_logits=False)
        output_ids = sample(logits, self.temperature)
        main_hidden = torch.cat(main_hiddens, dim=-1) if main_hiddens else None
        return output_ids, logits, main_hidden

    def forward_train(
        self,
        input_ids: torch.Tensor,
        images=None,
        token_types: torch.Tensor | None = None,
        return_mtp_hidden: bool = False,
    ):
        """Differentiable backbone forward pass.

        The normal return value is kept compatible with existing training and
        evaluation callers.  An active MTP profile may request the target-layer
        states required for the teacher-forced draft loss.
        """
        h_norm, main_hiddens, osc_metrics, _, _ = self._run_backbone(
            input_ids, start_pos=0, images=images, token_types=token_types
        )
        logits = self.head(h_norm, full_logits=True)
        if return_mtp_hidden:
            if not self.mtp or not main_hiddens:
                raise RuntimeError("MTP hidden states were requested but DSpark MTP is not active")
            return logits, osc_metrics, h_norm, torch.cat(main_hiddens, dim=-1)
        return logits, osc_metrics, h_norm

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Checkpoint the MoE sublayer of every backbone and MTP block."""
        for layer in (*self.layers, *self.mtp):
            layer.gradient_checkpointing = enabled

    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        state: ChunkState | None = None,
        token_types: torch.Tensor | None = None,
        images=None,
    ) -> tuple[torch.Tensor, ChunkState, dict, list[torch.Tensor]]:
        """Run one sequence chunk and return ``(h_norm, state, osc_metrics, mtp_hiddens)``.

        ``mtp_hiddens`` are the target-layer states for *this chunk only*; the trainer computes
        the draft loss before moving on. This is the memory-safe entry point for long-context
        training and chunked prefill.
        """
        if state is None:
            state = ChunkState()
        h_norm, main_hiddens, osc_metrics, attn_state, retro = self._run_backbone(
            input_ids,
            start_pos=start_pos,
            images=images,
            token_types=token_types,
            state=state.attention,
            retro_state=state.retro,
        )
        state.attention = attn_state
        state.retro = retro
        return h_norm, state, osc_metrics, main_hiddens

    def chunked_lm_loss(
        self,
        h_norm: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        token_chunk: int = 2048,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Cross-entropy without materialising ``[B, S, vocab]``.

        The head is evaluated on flat token slices, so peak logits memory is
        ``token_chunk × vocab`` rather than ``B × S × vocab``.
        """
        b, s, d = h_norm.shape
        if targets.shape != (b, s):
            raise ValueError(f"targets must be {(b, s)}, received {tuple(targets.shape)}")
        flat_h = h_norm.reshape(-1, d)
        flat_t = targets.reshape(-1)
        flat_m = None if loss_mask is None else loss_mask.reshape(-1).bool()
        total = h_norm.new_zeros((), dtype=torch.float32)
        count = 0
        for start in range(0, flat_h.size(0), token_chunk):
            end = min(start + token_chunk, flat_h.size(0))
            logits = self.head(flat_h[start:end], full_logits=True).float()
            target = flat_t[start:end].clone()
            if flat_m is not None:
                target = target.masked_fill(~flat_m[start:end], ignore_index)
            valid = target != ignore_index
            if not bool(valid.any()):
                continue
            total = total + F.cross_entropy(logits, target, reduction="sum", ignore_index=ignore_index)
            count += int(valid.sum())
        if count == 0:
            return h_norm.sum() * 0.0
        return total / count

    @torch.no_grad()
    def mtp_prefill(self, main_hidden: torch.Tensor) -> None:
        """Seed MTP attention caches from preceding backbone target states."""
        if not self.mtp:
            raise RuntimeError("MTP prefill requested while DSpark MTP is disabled")
        if main_hidden.ndim != 3 or main_hidden.size(1) == 0:
            raise ValueError("MTP prefill requires at least one preceding target-layer state")
        seed_ids = torch.full(
            (main_hidden.size(0),),
            self.args.dspark_noise_token_id,
            dtype=torch.long,
            device=main_hidden.device,
        )
        h, main_x = self.mtp[0].forward_embed(main_hidden, seed_ids)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.mtp:
            h, pre_mix = layer(h, 0, pre_mix, main_x)

    def forward_mtp_train(
        self,
        main_hidden: torch.Tensor,
        targets: torch.Tensor,
        positions_per_sequence: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Evaluate a bounded number of teacher-forced MTP positions.

        Caches are prefilled from detached preceding backbone states.  The
        current target-layer state remains differentiable, so the draft loss
        trains both the MTP parameters and its configured backbone targets
        without materialising a second full autoregressive training pass.
        """
        if not self.mtp:
            return None
        if positions_per_sequence <= 0:
            raise ValueError("positions_per_sequence must be positive when DSpark MTP is enabled")
        block_size = self.args.dspark_block_size
        last_position = targets.size(1) - block_size - 1
        if last_position < 1:
            return None
        count = min(positions_per_sequence, last_position)
        if count == 1:
            positions = (last_position,)
        else:
            positions = tuple(1 + i * (last_position - 1) // (count - 1) for i in range(count))

        logits_parts, target_parts, confidence_parts = [], [], []
        for position in positions:
            self.mtp_prefill(main_hidden[:, :position].detach())
            accepted_id = targets[:, position]
            draft_targets = targets[:, position + 1 : position + 1 + block_size]
            h, main_x = self.mtp[0].forward_embed(main_hidden[:, position : position + 1], accepted_id)
            pre_mix = make_identity_pre_mix(h, self.hc_mult)
            for layer in self.mtp:
                h, pre_mix = layer(h, position, pre_mix, main_x)
            logits, confidence = self.mtp[-1].forward_train_head(h, pre_mix, accepted_id, draft_targets)
            logits_parts.append(logits)
            target_parts.append(draft_targets)
            confidence_parts.append(confidence)
        return (
            torch.cat(logits_parts, dim=0),
            torch.cat(target_parts, dim=0),
            torch.cat(confidence_parts, dim=0),
        )


    @torch.inference_mode()
    def forward_spec(self, input_ids: torch.Tensor, main_hidden: torch.Tensor, start_pos: int = 0):
        h, main_x = self.mtp[0].forward_embed(main_hidden, input_ids)
        pre_mix = make_identity_pre_mix(h, self.hc_mult)
        for layer in self.mtp:
            h, pre_mix = layer(h, start_pos, pre_mix, main_x)
        if start_pos == 0:
            return None
        return self.mtp[-1].forward_head(h, pre_mix, input_ids)


NoesisSigmaTransformer = Transformer


def sample(logits, temperature: float = 1.0):
    """Gumbel-max trick: equivalent to multinomial sampling but faster on GPU,
    since it avoids the GPU-to-CPU sync in torch.multinomial."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-5)
    probs = torch.softmax(logits, dim=-1, dtype=torch.float32)
    return probs.div_(torch.empty_like(probs).exponential_(1)).argmax(dim=-1)


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs(dspark_block_size=6, dspark_target_layer_ids=(3, 4))
    x = torch.randint(0, args.vocab_size, (2, 150))
    model = Transformer(args)

    output_ids, logits, main_hidden = model(x[:, :128])
    model.forward_spec(output_ids, main_hidden)
    for i in range(128, 150):
        output_ids, logits, main_hidden = model(x[:, i : i + 1], i)
        result = model.forward_spec(output_ids, main_hidden, i)
        assert result is not None
        output_ids, logits, confidence = result
