# NOESIS-Σ Unit Tests: Differentiable Kernels, Sinkhorn Hyper-Connections & Engram Memory
# Path: tests/test_kernels_and_engram.py

from __future__ import annotations
import torch
import pytest

from kernel import (
    act_quant,
    fp4_act_quant,
    fp8_gemm,
    fp4_gemm,
    sparse_attn,
    hc_split_sinkhorn,
)
from engram import (
    find_next_prime,
    build_compressed_token_map,
    compute_hash_multipliers,
    EngramLayout,
    NgramHashState,
)
from model import ModelArgs, Engram


def test_quantization_and_gemm_kernels(device: torch.device) -> None:
    """Verify act_quant, fp4_act_quant, fp8_gemm, and fp4_gemm in-place and out-of-place."""
    torch.manual_seed(10)
    x = torch.randn(2, 4, 64, device=device, dtype=torch.bfloat16)

    # Out-of-place act_quant
    y, s = act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu, inplace=False)
    assert y.dtype == torch.float8_e4m3fn
    assert s.shape == (2, 4, 2)

    # In-place act_quant (pass-through returning x)
    x_clone = x.clone()
    y_in = act_quant(x_clone, block_size=32, scale_fmt="e4m3", scale_dtype=torch.float8_e4m3fn, inplace=True)
    assert y_in is x_clone
    assert torch.isfinite(x_clone).all()

    # fp4_act_quant (both inplace and out-of-place scale paths)
    x_fp4 = x.clone()
    q4 = fp4_act_quant(x_fp4, block_size=32, inplace=True, scale_dtype=torch.float8_e8m0fnu)
    assert q4 is x_fp4

    q4_b, s4_b = fp4_act_quant(x, block_size=16, inplace=False, scale_dtype=torch.float8_e4m3fn)
    assert q4_b.shape == x.shape and s4_b.shape == (2, 4, 4)

    # fp8_gemm & fp4_gemm
    w = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    w_scale = torch.ones(1, 2, device=device, dtype=torch.float8_e8m0fnu)
    out8 = fp8_gemm(x, s, w, w_scale, torch.float8_e8m0fnu, block_size=32)
    out4 = fp4_gemm(x, s, w, w_scale, torch.float8_e8m0fnu, act_block_size=32)
    assert out8.shape == (2, 4, 32)
    assert out4.shape == (2, 4, 32)


def test_sparse_attn_and_sinkhorn_hyper_connections(device: torch.device) -> None:
    """Verify sparse_attn causal/sink behavior and hc_split_sinkhorn doubly-stochastic matrices."""
    torch.manual_seed(11)
    B, M, N, H, D = 2, 6, 8, 4, 16
    q = torch.randn(B, M, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(B, N, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    attn_sink = torch.zeros(H, device=device, dtype=torch.float32, requires_grad=True)

    # Include both valid indices [0..N-1] and masked slots (-1)
    topk_idxs = torch.tensor(
        [[[0, 1, -1, -1], [0, 1, 2, -1], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]] * B,
        dtype=torch.int32,
        device=device,
    )
    out = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale=D**-0.5)
    assert out.shape == (B, M, H, D)
    loss = out.float().square().mean()
    loss.backward()
    assert q.grad is not None and kv.grad is not None and attn_sink.grad is not None

    # Edge case: topk == 0
    empty_idxs = torch.empty(B, M, 0, dtype=torch.int32, device=device)
    out_empty = sparse_attn(q.detach(), kv.detach(), attn_sink.detach(), empty_idxs, softmax_scale=D**-0.5)
    assert (out_empty == 0).all()

    # Verify hc_split_sinkhorn doubly-stochastic property (row sums == 1, col sums == 1)
    hc_mult = 4
    mix_dim = (2 + hc_mult) * hc_mult
    mixes = torch.randn(B, M, mix_dim, device=device, dtype=torch.float32, requires_grad=True)
    hc_scale = torch.ones(3, device=device, dtype=torch.float32, requires_grad=True)
    hc_base = torch.zeros(mix_dim, device=device, dtype=torch.float32, requires_grad=True)

    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=hc_mult, sinkhorn_iters=20, eps=1e-6)
    assert pre.shape == (B, M, hc_mult)
    assert post.shape == (B, M, hc_mult)
    assert comb.shape == (B, M, hc_mult, hc_mult)

    row_sums = comb.sum(dim=-1)
    col_sums = comb.sum(dim=-2)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
    assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)


def test_engram_layout_hash_state_and_module(device: torch.device) -> None:
    """Verify prime generation, EngramLayout, NgramHashState (prefill + decode + mask), and Engram block."""
    seen: set[int] = set()
    p1 = find_next_prime(10, seen)
    seen.add(p1)
    p2 = find_next_prime(10, seen)
    assert p1 == 11 and p2 == 13

    token_map, vsize = build_compressed_token_map(None, fallback_vocab_size=256)
    assert vsize == 256 and len(token_map) == 256

    mults = compute_hash_multipliers((1, 3), max_ngram_size=3, tokenizer_vocab_size=256)
    assert mults.shape == (2, 3)
    assert (mults % 2 == 1).all()  # all multipliers must be odd

    args = ModelArgs(
        max_batch_size=2,
        max_seq_len=64,
        vocab_size=256,
        dim=64,
        moe_inter_dim=128,
        n_layers=2,
        n_heads=4,
        head_dim=32,
        rope_head_dim=16,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        hc_mult=2,
        engram_layer_ids=(0, 1),
        engram_num_embeddings=(600, 600),
        engram_vocab_size=50,
        engram_compressed_vocab_size=256,
        engram_max_ngram_size=3,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_pad_id=0,
    )
    layout = EngramLayout.from_args(args)
    assert layout is not None
    assert layout.layer_ids == (0, 1)

    hash_state = NgramHashState(args, layout, tokenizer=None).to(device)
    input_ids = torch.randint(0, 256, (2, 8), device=device)
    token_mask = torch.ones_like(input_ids, dtype=torch.bool)
    token_mask[:, 3] = False  # simulate an image token blocking n-grams

    hashes_prefill = hash_state(input_ids, start_pos=0, token_mask=token_mask)
    assert hashes_prefill.shape == (2, 8, 2, 4)  # (max_ngram_size-1)*n_heads = 2*2 = 4

    # Decode step
    next_ids = torch.randint(0, 256, (2, 1), device=device)
    hashes_decode = hash_state(next_ids, start_pos=8)
    assert hashes_decode.shape == (2, 1, 2, 4)

    # Test Engram layer forward & backward
    engram_mod = Engram(args, layer_id=0, layout=layout).to(device)
    x = torch.randn(2, 8, args.hc_mult, args.dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = engram_mod(x, hashes_prefill[:, :, 0, :], token_mask=token_mask)
    assert y.shape == x.shape
    y.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
