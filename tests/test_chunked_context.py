# NOESIS-Σ Unit Tests: chunked long-context forward, caches, kernels, and losses
# Path: tests/test_chunked_context.py

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from kernel import sparse_attn
from model import ChunkState, Compressor, ModelArgs, Transformer
from osc_block import RetrogradeController


def tiny_args(**overrides) -> ModelArgs:
    """A tiny model with compression, indexer, OSC and MTP all enabled."""
    cfg = dict(
        max_batch_size=2,
        max_seq_len=128,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=512,
        dim=64,
        moe_inter_dim=128,
        n_layers=4,
        n_mtp_layers=1,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=32,
        window_size=16,
        compress_ratios=(0, 4, 4, 0, 0),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        candidate_source_layer=-1,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        hc_mult=2,
        engram_layer_ids=(),
        vision_n_layers=0,
        dspark_block_size=4,
        dspark_target_layer_ids=(1, 2),
        dspark_markov_rank=16,
        dspark_noise_token_id=0,
        osc_enabled=True,
        osc_rank=16,
        osc_icnn_hidden=32,
        osc_steps=2,
        osc_retrograde=True,
    )
    cfg.update(overrides)
    return ModelArgs(**cfg)


def build_fp32_model(args: ModelArgs, seed: int = 7) -> Transformer:
    torch.manual_seed(seed)
    model = Transformer(args)
    return model.float().eval()


@torch.no_grad()
def test_chunked_forward_matches_full_forward() -> None:
    """One forward over 32 tokens equals four 8-token chunks carrying ChunkState."""
    args = tiny_args()
    model = build_fp32_model(args)
    torch.manual_seed(11)
    x = torch.randint(0, args.vocab_size, (1, 32))

    full, _, _, _ = model.forward_chunk(x, start_pos=0, state=ChunkState())

    state = ChunkState()
    parts = []
    for start in range(0, 32, 8):
        h, state, _, _ = model.forward_chunk(x[:, start : start + 8], start_pos=start, state=state)
        parts.append(h)
    chunked = torch.cat(parts, dim=1)

    assert full.shape == chunked.shape == (1, 32, args.dim)
    torch.testing.assert_close(chunked, full, atol=2e-4, rtol=2e-4)


@torch.no_grad()
def test_prefill_then_decode_seeds_persistent_caches() -> None:
    """Chunked prefill plus a one-token decode equals a full forward's last-position logits."""
    args = tiny_args()
    model = build_fp32_model(args)
    torch.manual_seed(13)
    x = torch.randint(0, args.vocab_size, (1, 25))

    full_logits = model(x, start_pos=0)[1]

    state = ChunkState()
    for start in range(0, 24, 6):
        _, state, _, _ = model.forward_chunk(x[:, start : start + 6], start_pos=start, state=state)
    decode_h, _, _, _ = model.forward_chunk(x[:, 24:25], start_pos=24, state=state)
    decode_logits = model.head(decode_h)

    assert torch.argmax(decode_logits, dim=-1).item() == torch.argmax(full_logits, dim=-1).item()
    torch.testing.assert_close(decode_logits, full_logits, atol=5e-4, rtol=5e-4)


def test_sparse_attn_chunked_matches_full_with_masks_and_sink() -> None:
    """The query-chunked kernel is numerically equal to the unchunked computation, grads included."""
    torch.manual_seed(17)
    b, m, h, d, n, topk = 2, 23, 3, 8, 12, 5
    topk_idxs = torch.randint(0, n, (b, m, topk), dtype=torch.int32)
    topk_idxs[0, :4, 0] = -1  # unreachable slots must be masked, not gathered
    topk_idxs[1, 10:, -1] = -1
    scale = d**-0.5

    q0 = torch.randn(b, m, h, d, dtype=torch.float64)
    kv0 = torch.randn(b, n, d, dtype=torch.float64)
    sink0 = torch.randn(h, dtype=torch.float64)

    outs = []
    for chunk in (m, 5, 1):
        q = q0.clone().requires_grad_()
        kv = kv0.clone().requires_grad_()
        sink = sink0.clone().requires_grad_()
        out = sparse_attn(q, kv, sink, topk_idxs, scale, chunk_size=chunk)
        (out.square().mean()).backward()
        outs.append((out.detach(), q.grad, kv.grad, sink.grad))

    ref_out, ref_q, ref_kv, ref_sink = outs[0]
    for out, q_grad, kv_grad, sink_grad in outs[1:]:
        torch.testing.assert_close(out, ref_out, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(q_grad, ref_q, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(kv_grad, ref_kv, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(sink_grad, ref_sink, atol=1e-8, rtol=1e-8)


@torch.no_grad()
def test_compressor_group_boundaries_match_full_pooling() -> None:
    """A group split across two chunks pools the same latents as one full pass."""
    args = tiny_args()
    torch.manual_seed(19)
    comp = Compressor(args, layer_id=1).float().eval()
    x = torch.randn(2, 11, args.dim, dtype=torch.float32)

    full_latents, full_first = comp(x, start_pos=0)
    assert full_first == 0 and full_latents.size(1) == 11 // 4

    first_latents, first_group = comp(x[:, :6], start_pos=0)
    second_latents, second_group = comp(x[:, 6:], start_pos=6)
    # First chunk completes group 0 and buffers two tokens; the second finishes group 1
    # with those buffered tokens plus positions 6-7, then buffers positions 8-10.
    assert first_group == 0 and first_latents.size(1) == 1
    assert second_group == 1 and second_latents.size(1) == 1

    chunked = torch.cat([first_latents, second_latents], dim=1)
    assert chunked.size(1) == full_latents.size(1)
    torch.testing.assert_close(chunked, full_latents, atol=1e-5, rtol=1e-5)


def test_chunked_lm_loss_matches_full_cross_entropy() -> None:
    """The memory-bounded head/loss loop equals a single full-logit cross-entropy."""
    args = tiny_args()
    model = build_fp32_model(args)
    torch.manual_seed(23)
    h = torch.randn(2, 13, args.dim)
    targets = torch.randint(0, args.vocab_size, (2, 13))
    mask = torch.rand(2, 13) > 0.3

    token_chunk = model.chunked_lm_loss(h, targets, loss_mask=mask, token_chunk=4)

    logits = model.head(h.reshape(-1, args.dim), full_logits=True).float()
    reference = F.cross_entropy(logits, targets.reshape(-1), reduction="none")
    reference = reference[mask.reshape(-1)].mean()
    torch.testing.assert_close(token_chunk, reference, atol=1e-5, rtol=1e-5)

    # An all-masked chunk must produce a zero loss that still participates in autograd.
    h_grad = h.clone().requires_grad_()
    empty = model.chunked_lm_loss(h_grad, targets, loss_mask=torch.zeros_like(mask), token_chunk=4)
    assert empty.item() == 0.0 and empty.requires_grad
    empty.backward()
    assert h_grad.grad is not None and torch.isfinite(h_grad.grad).all()


def test_osc_scan_matches_reference_ema_and_carries_state() -> None:
    """The blockwise scan equals the token-by-token EMA, including across chunk boundaries."""
    torch.manual_seed(29)
    dim = 6
    controller = RetrogradeController(dim, alpha=0.1, decay=0.95, u_max=1e9).float()
    u = torch.randn(2, 40, dim)

    def reference(values: torch.Tensor, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            projected = controller.w_retro(values).float()
            out = []
            for t in range(values.size(1)):
                state = controller.decay * state + projected[:, t]
                out.append(controller.alpha * state)
            return torch.stack(out, dim=1), state

    with torch.no_grad():
        full, final_state = controller.scan(u)
        ref, ref_state = reference(u, torch.zeros(2, dim))
    torch.testing.assert_close(full, ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final_state, ref_state, atol=1e-5, rtol=1e-5)

    with torch.no_grad():
        head, mid_state = controller.scan(u[:, :17])
        tail, tail_state = controller.scan(u[:, 17:], mid_state)
    torch.testing.assert_close(torch.cat([head, tail], dim=1), full, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(tail_state, ref_state, atol=1e-5, rtol=1e-5)


def test_chunked_mtp_training_runs_and_backpropagates() -> None:
    """MTP draft loss computed per chunk overflows neither the graph nor the draft cache."""
    args = tiny_args()
    model = build_fp32_model(args)
    model.train()
    torch.manual_seed(31)
    ids = torch.randint(1, args.vocab_size, (1, 48))

    state = ChunkState()
    for chunk in range(3):
        start, end = chunk * 16, chunk * 16 + 16
        x = ids[:, start:end]
        targets = ids[:, start + 1 : end + 1]
        length = min(x.size(1), targets.size(1))
        x, targets = x[:, :length], targets[:, :length]
        h, state, _, mtp_hidden = model.forward_chunk(x, start_pos=start, state=state)
        assert len(mtp_hidden) == len(args.dspark_target_layer_ids)
        lm = model.chunked_lm_loss(h, targets)
        draft = model.forward_mtp_train(torch.cat(mtp_hidden, dim=-1), targets)
        assert draft is not None
        mtp_logits, mtp_targets, _ = draft
        mtp_loss = F.cross_entropy(mtp_logits.flatten(0, 1), mtp_targets.flatten())
        (lm + 0.2 * mtp_loss).backward()
        assert torch.isfinite(lm) and torch.isfinite(mtp_loss)
        state.detach()

    assert model.mtp[0].main_proj.weight.grad is not None
    assert torch.isfinite(model.mtp[0].main_proj.weight.grad).all()


def test_chunk_exceeding_max_seq_len_is_rejected() -> None:
    args = tiny_args()
    model = build_fp32_model(args)
    x = torch.randint(0, args.vocab_size, (1, 8))
    with pytest.raises(ValueError, match="max_seq_len"):
        model.forward_chunk(x, start_pos=args.max_seq_len - 4)
