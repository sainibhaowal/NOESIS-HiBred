# NOESIS-Σ Tests: 2-rank FSDP smoke (needs >= 2 CUDA GPUs)
# Path: tests/test_fsdp_smoke.py

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _path in (str(ROOT / "core"), str(ROOT / "core" / "encoding"), str(ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pytest
import torch

from model import Block, ModelArgs, Transformer, configure_model_parallel

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs at least 2 CUDA GPUs"),
]


TINY_CONFIG = dict(
    max_batch_size=1,
    max_seq_len=64,
    dtype="bf16",
    expert_dtype=None,
    vocab_size=512,
    dim=64,
    moe_inter_dim=128,
    n_layers=3,
    n_mtp_layers=0,
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
    compress_ratios=[0, 4, 0],
    kv_source_layers=[1],
    index_source_layers=[1],
    index_n_heads=4,
    index_head_dim=32,
    index_topk=8,
    hc_mult=2,
    engram_layer_ids=[],
    vision_n_layers=0,
    dspark_block_size=0,
    dspark_target_layer_ids=[],
    osc_enabled=True,
    osc_rank=16,
    osc_icnn_hidden=32,
    osc_steps=2,
)


def _fsdp_worker(rank: int, world_size: int, config_json: str, init_file: str, out_dir: str) -> None:
    from functools import partial

    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    from model import ChunkState

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", init_method=f"file://{init_file}", world_size=world_size, rank=rank
    )
    try:
        configure_model_parallel(1, 0)
        args = ModelArgs(**json.loads(config_json))
        torch.manual_seed(0)
        model = Transformer(args)
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.data = parameter.data.float()

        model = FSDP(
            model.to(torch.device("cuda", rank)),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
            device_id=torch.device("cuda", rank),
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
            mixed_precision=None,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        x = torch.randint(0, args.vocab_size, (1, 16), device=torch.device("cuda", rank))
        h, state, _, _ = model(x, start_pos=0, state=ChunkState())
        targets = torch.randint(0, args.vocab_size, (1, 16), device=h.device)
        loss = model.chunked_lm_loss(h, targets)
        loss.backward()
        grad_norm = model.clip_grad_norm_(1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        assert torch.isfinite(loss) and torch.isfinite(grad_norm)
        if rank == 0:
            Path(out_dir, "fsdp_step.json").write_text(json.dumps({"loss": float(loss)}))
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp_step(tmp_path: Path) -> None:
    """One FSDP step over two ranks produces finite gradients and a recorded loss."""
    import torch.multiprocessing as mp

    init_file = tmp_path / "init"
    config_json = json.dumps(TINY_CONFIG)
    mp.spawn(
        _fsdp_worker,
        args=(2, config_json, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )
    result = json.loads((tmp_path / "fsdp_step.json").read_text())
    assert result["loss"] == result["loss"]
