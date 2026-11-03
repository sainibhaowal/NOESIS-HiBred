from typing import Tuple, Optional
import torch
import torch.nn.functional as F

HAS_TILELANG = False

FP8 = "float8_e4m3"
FP4 = "float4_e2m1fn"
FE8M0 = "float8_e8m0fnu"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP8 quantization.
    During autograd training or when TileLang is absent, preserves differentiability.
    """
    N = x.size(-1)
    assert N % block_size == 0
    if inplace:
        return x
    grouped = x.float().unflatten(-1, (-1, block_size))
    amax = grouped.abs().amax(dim=-1).clamp(min=1e-4)
    scale = amax / 448.0
    q = (grouped / scale.unsqueeze(-1)).clamp(-448.0, 448.0).flatten(-2).to(torch.float8_e4m3fn)
    # e8m0 scale dtypes have no autograd kernel. Keep fp32 scales while training so the
    # quantized activation remains differentiable; inference keeps the low-precision scale.
    if torch.is_grad_enabled():
        return q, scale.float()
    return q, scale.to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    """Block-wise FP4 quantization fallback & training pass-through."""
    if inplace:
        return x
    N = x.size(-1)
    grouped = x.float().unflatten(-1, (-1, block_size))
    amax = grouped.abs().amax(dim=-1).clamp(min=1e-4)
    scale = amax / 6.0
    if torch.is_grad_enabled():
        return x, scale.float()
    return x, scale.to(scale_dtype)


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-block FP8 scaling."""
    K = a.size(-1)
    N = b.size(0)
    a_deq = (
        a.float()
        .unflatten(-1, (-1, block_size))
        .mul(a_s.float().unsqueeze(-1))
        .flatten(-2)
    )
    b_deq = (
        b.float()
        .view((N + block_size - 1) // block_size, -1, K // block_size, block_size)
        .mul(b_s.float().unsqueeze(1).unsqueeze(-1))
        .reshape(N, K)
    )
    return F.linear(a_deq, b_deq).to(torch.get_default_dtype())


_FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """Fallback GEMM for FP4 weights (supporting packed float4_e2m1fn_x2 and unpacked tensors)."""
    a_deq = (
        a.float()
        .unflatten(-1, (-1, act_block_size))
        .mul(a_s.float().unsqueeze(-1))
        .flatten(-2)
    )
    if b.dtype == torch.float4_e2m1fn_x2:
        lut = torch.tensor(_FP4_LUT, dtype=torch.float32, device=b.device)
        raw = b.view(torch.uint8)
        low = raw & 0x0F
        high = (raw >> 4) & 0x0F
        b_unpacked = torch.stack([lut[low.long()], lut[high.long()]], dim=-1).flatten(-2)
        b_deq = (
            b_unpacked.unflatten(-1, (-1, act_block_size))
            .mul(b_s.float().unsqueeze(-1))
            .flatten(-2)
        )
    else:
        b_deq = b.float()
    return F.linear(a_deq, b_deq).to(torch.get_default_dtype())


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Differentiable Sparse Multi-Head Latent Attention with learnable attention sink.

    q: [b, m, h, d]
    kv: [b, n, d]
    attn_sink: [h] (float32)
    topk_idxs: [b, m, topk] (int32, -1 indicates unreachable/masked slot)

    Queries are processed in chunks. Gathering ``kv[b, idx]`` for every query at once costs
    ``B*S*topk*d`` and is what made 256K impossible; chunking bounds it to
    ``B*chunk_size*topk*d`` while keeping the exact same math and gradients.
    """
    if chunk_size is None or chunk_size <= 0:
        chunk_size = q.size(1)
    b, m, h, d = q.size()
    topk = topk_idxs.size(-1)
    if topk == 0 or kv.size(1) == 0:
        return torch.zeros_like(q)

    outputs: list[torch.Tensor] = []
    for start in range(0, m, chunk_size):
        end = min(start + chunk_size, m)
        q_chunk = q[:, start:end]
        idx_chunk = topk_idxs[:, start:end]
        valid = idx_chunk >= 0  # [b, q, topk]
        safe_idxs = idx_chunk.clamp(min=0).long()
        batch_idx = torch.arange(b, device=q.device)[:, None, None]

        # Gather selected KV latents per query: [b, q, topk, d]
        kv_gathered = kv[batch_idx, safe_idxs].to(dtype=q_chunk.dtype)

        # Attention scores: [b, q, h, topk]
        scores = torch.einsum("bqhd,bqkd->bqhk", q_chunk, kv_gathered).float() * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), -1e30)

        # Append learnable attention sink logit per head (value = 0 vector)
        sink = attn_sink.float().view(1, 1, h, 1).expand(b, end - start, h, 1)
        probs = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)[..., :-1]

        o = torch.einsum("bqhk,bqkd->bqhd", probs.to(dtype=q_chunk.dtype), kv_gathered)
        outputs.append(o.to(q_chunk.dtype))
    return torch.cat(outputs, dim=1)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable Hyper-Connection coefficient projection & Sinkhorn doubly-stochastic normalization.

    mixes: [b, s, (2 + hc_mult) * hc_mult]
    hc_scale: [3]
    hc_base: [(2 + hc_mult) * hc_mult]
    Returns:
      pre:  [b, s, hc_mult]
      post: [b, s, hc_mult]
      comb: [b, s, hc_mult, hc_mult]
    """
    mixes_f = mixes.float()
    scale_f = hc_scale.float()
    base_f = hc_base.float()

    pre = torch.sigmoid(mixes_f[..., :hc_mult] * scale_f[0] + base_f[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(
        mixes_f[..., hc_mult : 2 * hc_mult] * scale_f[1] + base_f[hc_mult : 2 * hc_mult]
    )

    comb_raw = mixes_f[..., 2 * hc_mult :] * scale_f[2] + base_f[2 * hc_mult :]
    comb = comb_raw.unflatten(-1, (hc_mult, hc_mult))

    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    for _ in range(max(0, sinkhorn_iters - 1)):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb
