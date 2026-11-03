# NOESIS-Σ Unit Tests: configuration validation, state defaults, and edge paths
# Path: tests/test_validation_and_edge_paths.py

from __future__ import annotations

import pytest
import torch

import model as model_module
from kernel import act_quant, fp4_act_quant, sparse_attn
from model import AttentionState, ChunkState, ModelArgs, Transformer
from verifier_loss import SigmaVerifierLoss


VALIDATION_BASE = dict(
    vocab_size=256,
    dim=32,
    moe_inter_dim=32,
    n_layers=2,
    n_mtp_layers=0,
    n_heads=4,
    n_routed_experts=2,
    n_shared_experts=1,
    n_activated_experts=1,
    q_lora_rank=16,
    head_dim=8,
    rope_head_dim=4,
    o_groups=2,
    o_lora_rank=16,
    window_size=8,
    compress_ratios=(0, 0),
    kv_source_layers=(),
    index_source_layers=(),
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    hc_mult=1,
    dtype="bf16",
    expert_dtype=None,
    osc_enabled=False,
    osc_rank=4,
    osc_icnn_hidden=8,
)


def validation_args(**overrides) -> ModelArgs:
    cfg = dict(VALIDATION_BASE)
    cfg.update(overrides)
    return ModelArgs(**cfg)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"n_layers": 0}, "n_layers must be positive"),
        ({"index_query_stride": 0}, "index_query_stride"),
        ({"compress_ratios": (0,)}, "one entry for every"),
        ({"compress_ratios": (-1, -1)}, "negative"),
        ({"compress_ratios": (1, 1)}, "require KV and index source"),
        (
            {"compress_ratios": (1, 0), "kv_source_layers": (1,), "index_source_layers": (0,)},
            "source layers must have a positive",
        ),
        (
            {
                "n_layers": 1,
                "n_mtp_layers": 1,
                "compress_ratios": (1, 1),
                "kv_source_layers": (0,),
                "index_source_layers": (0,),
            },
            "MTP layers cannot use compressed-KV",
        ),
        (
            {"compress_ratios": (1, 1), "kv_source_layers": (1,), "index_source_layers": (1,)},
            "no earlier KV source",
        ),
        (
            {"compress_ratios": (1, 1), "kv_source_layers": (0,), "index_source_layers": (1,)},
            "no earlier index source",
        ),
        ({"kv_source_layers": (0,)}, "while every compression ratio is zero"),
        ({"engram_layer_ids": (0,)}, "one table size for each"),
        (
            {"engram_layer_ids": (0, 0), "engram_num_embeddings": (16, 16)},
            "cannot contain duplicates",
        ),
        (
            {"engram_layer_ids": (2,), "engram_num_embeddings": (16,)},
            "refer to backbone layers",
        ),
        (
            {"engram_layer_ids": (0,), "engram_num_embeddings": (0,)},
            "table sizes must be positive",
        ),
        (
            {"engram_layer_ids": (0,), "engram_num_embeddings": (16,)},
            "positive n-gram",
        ),
        (
            {
                "engram_layer_ids": (0,),
                "engram_num_embeddings": (16,),
                "engram_vocab_size": 4,
                "engram_n_heads": 1,
                "engram_head_dim": 4,
            },
            "engram_compressed_vocab_size",
        ),
        ({"engram_num_embeddings": (16,)}, "without engram_layer_ids"),
        ({"dspark_block_size": 2, "dspark_target_layer_ids": (0,)}, "at least one MTP layer"),
        (
            {"n_mtp_layers": 1, "compress_ratios": (0, 0, 0), "dspark_block_size": 2},
            "requires dspark_target_layer_ids",
        ),
        (
            {
                "n_mtp_layers": 1,
                "compress_ratios": (0, 0, 0),
                "dspark_block_size": 2,
                "dspark_target_layer_ids": (5,),
            },
            "refer to backbone layers",
        ),
        (
            {
                "n_mtp_layers": 1,
                "compress_ratios": (0, 0, 0),
                "dspark_block_size": 2,
                "dspark_target_layer_ids": (0,),
                "dspark_noise_token_id": 256,
            },
            "valid tokenizer id",
        ),
        ({"dspark_target_layer_ids": (0,)}, "while DSpark MTP is disabled"),
    ],
)
def test_model_args_validation_errors(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        validation_args(**overrides).validate_for_model()


def test_model_args_validation_accepts_valid_layout() -> None:
    validation_args().validate_for_model()


def test_configure_model_parallel_validates_arguments() -> None:
    with pytest.raises(ValueError, match="mp_size must be positive"):
        model_module.configure_model_parallel(0, 0)
    with pytest.raises(ValueError, match="mp_rank must be in"):
        model_module.configure_model_parallel(2, 2)
    model_module.configure_model_parallel(1, 0)
    assert model_module.world_size == 1 and model_module.rank == 0


def test_attention_state_reset_clears_all_fields() -> None:
    state = AttentionState(
        compress_kv=torch.zeros(1, 1, 1),
        index_k=torch.zeros(1, 1, 1),
        topk_idxs=torch.zeros(1, 1, 1, dtype=torch.int32),
        candidates=torch.zeros(1, 1, 1, dtype=torch.int32),
    )
    state.reset()
    assert state.compress_kv is None
    assert state.index_k is None
    assert state.topk_idxs is None
    assert state.candidates is None


def test_chunk_state_detach_covers_mtp_window() -> None:
    state = ChunkState(
        retro=torch.zeros(1, 4, requires_grad=True),
        mtp_window=torch.zeros(1, 4, requires_grad=True),
    )
    state.detach()
    assert state.retro is not None and not state.retro.requires_grad
    assert state.mtp_window is not None and not state.mtp_window.requires_grad


def tiny_args(**overrides) -> ModelArgs:
    cfg = dict(
        max_batch_size=2,
        max_seq_len=32,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=512,
        dim=64,
        moe_inter_dim=128,
        n_layers=3,
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
        window_size=8,
        compress_ratios=(0, 4, 4, 0),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        hc_mult=2,
        engram_layer_ids=(),
        vision_n_layers=0,
        dspark_block_size=4,
        dspark_target_layer_ids=(0,),
        dspark_markov_rank=16,
        dspark_noise_token_id=0,
        osc_enabled=True,
        osc_rank=16,
        osc_icnn_hidden=32,
        osc_steps=2,
    )
    cfg.update(overrides)
    return ModelArgs(**cfg)


def build_model(args: ModelArgs | None = None) -> Transformer:
    torch.manual_seed(3)
    return Transformer(args or tiny_args()).float()


def test_attention_and_indexer_default_state_paths() -> None:
    model = build_model()
    source = model.layers[1].attn
    reader = model.layers[2].attn
    x = torch.randn(1, 1, 64)
    assert source.is_kv_source
    assert source.indexer is not None
    source.indexer.freqs_cis = source._freqs(x.device)

    # A layer that is not the KV source starts with no published compressed KV.
    qr = reader.q_norm(reader.wq_a(x))
    compress_kv, _ = reader._compress_kv(x, qr, 0, 0, AttentionState())
    assert compress_kv.shape == (1, 0, reader.head_dim)

    # Attention and the indexer both create their own state when none is passed.
    assert source(x, 0, None).shape == x.shape
    source_qr = source.q_norm(source.wq_a(x))
    idxs = source.indexer(x, source_qr, None, 0, 0, None, 0)
    assert idxs.shape == (1, 1, 0)

    # A zero top-k budget returns an empty selection even when keys exist.
    source.indexer.index_topk = 0
    try:
        state = AttentionState(index_k=torch.zeros(1, 4, source.indexer.index_head_dim))
        idxs = source.indexer(x, source_qr, None, 4, 0, state, 0)
    finally:
        source.indexer.index_topk = 8
    assert idxs.shape == (1, 1, 0)


def test_block_gradient_checkpointing_forward_and_backward() -> None:
    model = build_model()
    model.train()
    model.enable_gradient_checkpointing(True)
    x = torch.randint(0, 512, (1, 4))
    _, _, _, _ = model.forward_chunk(x, start_pos=0, state=ChunkState())
    h, _, _, _ = model.forward_chunk(x, start_pos=0, state=ChunkState())
    h.sum().backward()
    assert any(param.grad is not None for param in model.layers[0].ffn.parameters())


def test_forward_dispatches_to_chunk_interface() -> None:
    model = build_model().eval()
    x = torch.randint(0, 512, (1, 2))
    with torch.no_grad():
        out = model(x, state=ChunkState())
    assert isinstance(out, tuple) and len(out) == 4


def test_dspark_target_shape_is_validated() -> None:
    model = build_model().eval()
    layer = model.mtp[0]
    x = torch.zeros(1, 4, 2, 64)
    pre_mix = torch.zeros(1, 4, 2)
    input_ids = torch.zeros(1, dtype=torch.long)
    with pytest.raises(ValueError, match="DSpark targets"):
        layer.forward_train_head(x, pre_mix, input_ids, torch.zeros(1, 3, dtype=torch.long))


def test_standalone_construction_resets_model_parallel_globals() -> None:
    model_module.MP_LAYOUT_CONFIGURED = False
    model_module.world_size = 4
    model_module.rank = 1
    try:
        Transformer(validation_args())
    finally:
        model_module.configure_model_parallel(1, 0)
    assert model_module.world_size == 1 and model_module.rank == 0


def test_model_parallel_divisibility_errors() -> None:
    try:
        model_module.configure_model_parallel(2, 0)
        with pytest.raises(ValueError, match="n_heads"):
            Transformer(validation_args(n_heads=3))
        with pytest.raises(ValueError, match="o_groups"):
            Transformer(validation_args(o_groups=3))
        with pytest.raises(ValueError, match="DSpark expert count"):
            Transformer(
                validation_args(
                    n_layers=2,
                    n_mtp_layers=1,
                    compress_ratios=(0, 0, 0),
                    n_routed_experts=3,
                    n_activated_experts=1,
                    dspark_block_size=2,
                    dspark_target_layer_ids=(0,),
                )
            )
    finally:
        model_module.configure_model_parallel(1, 0)


def test_engram_table_must_cover_hash_buckets() -> None:
    args = validation_args(
        n_layers=2,
        compress_ratios=(0, 0),
        engram_layer_ids=(0,),
        engram_num_embeddings=(1,),
        engram_max_ngram_size=2,
        engram_vocab_size=4,
        engram_n_heads=1,
        engram_head_dim=4,
        engram_compressed_vocab_size=256,
        osc_rank=4,
        osc_icnn_hidden=8,
    )
    with pytest.raises(ValueError, match="needs at least"):
        Transformer(args)


def test_forward_chunk_rejects_empty_and_over_budget() -> None:
    model = build_model(tiny_args(max_seq_len=8))
    with pytest.raises(ValueError, match="cannot be empty"):
        model.forward_chunk(torch.empty(1, 0, dtype=torch.long), start_pos=0, state=ChunkState())
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.forward_chunk(torch.randint(0, 512, (1, 4)), start_pos=6, state=ChunkState())


def test_chunked_lm_loss_validates_target_shape() -> None:
    model = build_model().eval()
    h = torch.randn(1, 4, 64)
    with torch.no_grad():
        with pytest.raises(ValueError, match="targets must be"):
            model.chunked_lm_loss(h, torch.zeros(1, 3, dtype=torch.long))


def test_mtp_entry_points_reject_invalid_calls() -> None:
    plain = build_model(tiny_args(dspark_block_size=0, dspark_target_layer_ids=())).eval()
    with torch.no_grad():
        with pytest.raises(RuntimeError, match="MTP is disabled"):
            plain.mtp_prefill(torch.zeros(1, 2, 64))
        with pytest.raises(RuntimeError, match="not active"):
            plain.forward_train(torch.randint(0, 512, (1, 4)), return_mtp_hidden=True)
        assert plain.forward_mtp_train(torch.zeros(1, 2, 64), torch.zeros(1, 10, dtype=torch.long)) is None

    model = build_model().eval()
    with torch.no_grad():
        with pytest.raises(ValueError, match="preceding target-layer state"):
            model.mtp_prefill(torch.zeros(1, 64))
        with pytest.raises(ValueError, match="positions_per_sequence"):
            model.forward_mtp_train(
                torch.zeros(2, 6, 64), torch.zeros(2, 12, dtype=torch.long), positions_per_sequence=0
            )
        assert model.forward_mtp_train(
            torch.zeros(2, 12, 64), torch.zeros(2, 5, dtype=torch.long)
        ) is None
        logits, targets, confidence = model.forward_mtp_train(
            torch.zeros(2, 12, 64), torch.zeros(2, 14, dtype=torch.long), positions_per_sequence=1
        )
    assert logits.shape[0] == targets.shape[0] == confidence.shape[0]


def test_distribute_to_gpus_requires_two_devices_and_moves_modules(capsys) -> None:
    model = build_model()
    model.vision = torch.nn.Linear(1, 1)
    if torch.cuda.device_count() >= 2:
        pytest.skip("the assertion branch only runs with fewer than two CUDA devices")
    with pytest.raises(AssertionError, match="at least 2 CUDA GPUs"):
        model.distribute_to_gpus()
    model.distribute_to_gpus(("cpu", "cpu"))
    assert "Pipelined NOESIS" in capsys.readouterr().out


def test_quantization_inference_branch_and_sparse_attn_default_chunk() -> None:
    x = torch.randn(2, 64)
    with torch.no_grad():
        q8, s8 = act_quant(x, block_size=32, scale_dtype=torch.float8_e8m0fnu)
        q4, s4 = fp4_act_quant(x, block_size=32)
    assert q8.dtype == torch.float8_e4m3fn
    assert s8.dtype == torch.float8_e8m0fnu
    assert s4.dtype == torch.float8_e8m0fnu

    q = torch.randn(1, 4, 2, 8)
    kv = torch.randn(1, 6, 8)
    sink = torch.zeros(2)
    idxs = torch.tensor([[[0, 1], [2, 3], [4, 5], [-1, 0]]], dtype=torch.int32)
    default_chunk = sparse_attn(q, kv, sink, idxs, 0.5, chunk_size=0)
    assert default_chunk.shape == q.shape


def test_verifier_auxiliary_loss_without_energy() -> None:
    criterion = SigmaVerifierLoss(alpha_attractor=0.1, beta_smoothness=0.2)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    total, parts = criterion.auxiliary_loss({}, hidden)
    assert parts["loss_attractor"].item() == 0.0
    assert total.item() > 0.0

    single = torch.randn(1, 1, 8)
    total, parts = criterion.auxiliary_loss({"energy_delta": torch.tensor(-1.0)}, single)
    assert parts["loss_attractor"].item() == 0.0
    assert parts["loss_smoothness"].item() == 0.0
    assert total.item() == 0.0

    total, parts = criterion.auxiliary_loss({"energy_delta": torch.tensor(2.0)}, hidden)
    assert parts["loss_attractor"].item() == pytest.approx(2.0)
    assert total.item() > 0.0
