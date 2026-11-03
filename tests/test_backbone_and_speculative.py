# NOESIS-Σ Unit Tests: MLA Compressor/Indexer, Speculative DSpark MTP, RoPE & Weight Conversion
# Path: tests/test_backbone_and_speculative.py

from __future__ import annotations
import torch
import pytest

from model import (
    ModelArgs,
    NoesisSigmaTransformer,
    precompute_freqs_cis,
    apply_rotary_emb,
    rotate_tail,
     select_candidate_blocks,
    Gate,
    sample,
)
from convert import cast_e2m1fn_to_e4m3fn, infer_num_experts


def test_rope_yarn_and_inverse_conjugation(device: torch.device) -> None:
    """Verify YaRN frequency precomputation and that inverse rotary embedding restores original tensor."""
    freqs_yarn = precompute_freqs_cis(
        dim=16, seqlen=64, original_seq_len=32, base=10000.0, factor=2.0, beta_fast=32, beta_slow=1
    ).to(device)
    freqs_std = precompute_freqs_cis(
        dim=16, seqlen=64, original_seq_len=0, base=10000.0, factor=1.0, beta_fast=32, beta_slow=1
    ).to(device)
    assert freqs_yarn.shape == (64, 8)
    assert freqs_std.shape == (64, 8)

    x = torch.randn(2, 10, 4, 32, device=device, dtype=torch.float32, requires_grad=True)
    y = rotate_tail(x, rd=16, freqs_cis=freqs_yarn[:10], inverse=False)
    x_rec = rotate_tail(y, rd=16, freqs_cis=freqs_yarn[:10], inverse=True)
    assert torch.allclose(x, x_rec, atol=1e-5)


def test_candidate_blocks_and_gate_routing_modes(device: torch.device) -> None:
    """Verify select_candidate_blocks two-stage sparse MLA filter and all Gate scoring functions."""
    logits = torch.randn(2, 4, 16, device=device)
    compress_lens = torch.tensor([[[4], [4], [4], [4]], [[8], [8], [8], [8]]], device=device)
    keep = select_candidate_blocks(logits, compress_lens, topk_blocks=2, block_size=4)
    assert keep.shape == (2, 4, 16)
    assert keep.dtype == torch.bool

    for score_fn in ("softmax", "sigmoid", "sqrtsoftplus"):
        args = ModelArgs(dim=64, n_routed_experts=4, n_activated_experts=2, score_func=score_fn, vision_n_layers=0)
        gate = Gate(layer_id=0, args=args).to(device)
        x = torch.randn(8, 64, device=device)
        weights, indices = gate(x)
        assert weights.shape == (8, 2) and indices.shape == (8, 2)


def test_compressed_mla_and_dspark_speculative_decoding(device: torch.device) -> None:
    """Verify full Transformer with compressed KV (Compressor + Indexer) and DSpark MTP speculative stage."""
    torch.manual_seed(30)
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    args = ModelArgs(
        max_batch_size=2,
        max_seq_len=128,
        temperature=0.0,  # test greedy argmax branch in sample() as well
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
        candidate_source_layer=1,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        candidate_topk_blocks=2,
        candidate_block_size=4,
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
    )

    model = NoesisSigmaTransformer(args).to(device)
    x = torch.randint(0, args.vocab_size, (2, 24), device=device)

    with torch.inference_mode():
        # Prefill 20 tokens
        out_ids, logits, main_hidden = model(x[:, :20], start_pos=0)
        assert out_ids.shape == (2,)
        assert logits.shape == (2, args.vocab_size)
        assert main_hidden is not None

        # Seed DSpark speculative stage on prefill
        spec_prefill = model.forward_spec(out_ids, main_hidden, start_pos=0)
        assert spec_prefill is None

        # Decode 4 tokens with speculative draft verification
        for pos in range(20, 24):
            out_ids, logits, main_hidden = model(x[:, pos : pos + 1], start_pos=pos)
            spec_res = model.forward_spec(out_ids, main_hidden, start_pos=pos)
            assert spec_res is not None
            draft_ids, draft_logits, confidence = spec_res
            assert draft_ids.shape == (2, args.dspark_block_size + 1)
            assert draft_logits.shape == (2, args.dspark_block_size, args.vocab_size)
            assert confidence.shape == (2, args.dspark_block_size)

    # MTP must also receive a differentiable teacher-forced training signal,
    # rather than being limited to its inference-only draft interface.
    model.train()
    _, _, _, mtp_hidden = model.forward_train(x[:, :20], return_mtp_hidden=True)
    mtp_outputs = model.forward_mtp_train(mtp_hidden, x[:, 1:21], positions_per_sequence=2)
    assert mtp_outputs is not None
    mtp_logits, mtp_targets, mtp_confidence = mtp_outputs
    assert mtp_logits.shape == (4, args.dspark_block_size, args.vocab_size)
    assert mtp_targets.shape == (4, args.dspark_block_size)
    assert mtp_confidence.shape == (4, args.dspark_block_size)
    mtp_loss = torch.nn.functional.cross_entropy(mtp_logits.flatten(0, 1), mtp_targets.flatten())
    mtp_loss.backward()
    assert model.mtp[0].main_proj.weight.grad is not None
    assert torch.isfinite(model.mtp[0].main_proj.weight.grad).all()


def test_convert_fp4_to_fp8_and_expert_inference(device: torch.device) -> None:
    """Verify convert.py lossless FP4-to-FP8 block casting and checkpoint expert counting."""
    names = [
        "model.layers.0.mlp.experts.0.w1.weight",
        "model.layers.0.mlp.experts.3.w1.weight",
        "mtp.0.ffn.experts.1.w1.weight",
    ]
    n_exp, mtp_exp = infer_num_experts(names)
    assert n_exp == 4 and mtp_exp == 2

    # Test cast_e2m1fn_to_e4m3fn on a 32x32 logical weight block (stored as 32x16 int8)
    x_packed = torch.randint(-128, 127, (32, 16), dtype=torch.int8, device="cpu")
    scale = torch.ones((32, 1), dtype=torch.float32, device="cpu")
    fp8_w, fp8_s = cast_e2m1fn_to_e4m3fn(x_packed, scale)
    assert fp8_w.shape == (32, 32)
    assert fp8_w.dtype == torch.float8_e4m3fn
    assert fp8_s.dtype == torch.float8_e8m0fnu


def test_dense_moe_and_ablation_modes(device: torch.device) -> None:
    """Verify that MoE seamlessly switches to a pure dense MLP when n_routed_experts is 0."""
    from model import MoE
    args = ModelArgs(dim=64, moe_inter_dim=128, n_routed_experts=0, n_shared_experts=1)
    dense_moe = MoE(layer_id=0, args=args).to(device)
    assert dense_moe.gate is None
    x = torch.randn(2, 8, 64, device=device)
    y = dense_moe(x)
    assert y.shape == (2, 8, 64)
