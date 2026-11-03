#!/usr/bin/env python3
"""Preflight for NOESIS-Σ production training and long-context generation.

Reports the hardware, validates the config against the chosen topology, and prints a memory plan:
FSDP master/gradient/optimizer shards, persistent attention caches, and a rough activation
estimate for the chosen chunk length. It never allocates the real model's parameters (they are
built on the ``meta`` device), so it is safe to run before launching an 8-GPU job.

Examples
--------
    python scripts/check_hardware.py --world_size 8
    python scripts/check_hardware.py --config core/config_1.5b.json --seq_len 262144 --chunk_len 8192
    python scripts/check_hardware.py --json

Exit status is non-zero when a hard incompatibility is found (wrong context, impossible
divisibility, more ranks than GPUs, or a memory plan that cannot fit even after shrinking).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (PROJECT_ROOT / "core", PROJECT_ROOT / "core" / "encoding"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from model import ModelArgs, Transformer, configure_model_parallel  # noqa: E402

GIB = 1024**3


def _load_active_parameter_count():
    """Import the trainer's counter without executing its CLI."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_pretrain_15b", PROJECT_ROOT / "scripts" / "pretrain_1.5b_production.py"
    )
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        spec.loader.exec_module(module)
    except SystemExit:  # pragma: no cover - the trainer's __main__ guard
        pass
    finally:
        sys.argv = old_argv
    return module.active_parameter_count


def hardware_report(world_size: int) -> dict:
    report: dict = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "nccl_available": torch.distributed.is_nccl_available(),
        "gpus": [],
    }
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report["gpus"].append(
                {
                    "index": index,
                    "name": props.name,
                    "vram_gib": props.total_memory / GIB,
                    "bf16": bool(torch.cuda.is_bf16_supported()),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
            )
        peers = []
        for i in range(min(world_size, torch.cuda.device_count())):
            for j in range(min(world_size, torch.cuda.device_count())):
                if i != j and torch.cuda.can_device_access_peer(i, j):
                    peers.append([i, j])
        report["p2p_pairs"] = len(peers)
    else:
        report["cpu_count"] = torch.get_num_threads()
    return report


def cache_bytes(args: ModelArgs, seq_len: int, batch_size: int) -> dict[str, int]:
    """Exact size of the buffers the model registers, at ``seq_len`` and ``batch_size``."""
    head_dim = args.head_dim
    bf16 = 2

    attention_layers = args.n_layers + args.n_mtp_layers
    window = attention_layers * batch_size * args.window_size * head_dim * bf16

    compressed = 0
    index = 0
    partial = 0
    for layer_id in range(args.n_layers + args.n_mtp_layers):
        ratio = args.compress_ratios[layer_id]
        if ratio <= 0 or layer_id >= args.n_layers:
            continue
        groups = math.ceil(seq_len / ratio)
        if layer_id in args.kv_source_layers:
            compressed += batch_size * groups * head_dim * bf16
            if ratio > 1:
                partial += 2 * batch_size * ratio * head_dim * 4  # fp32 kv_state + score_state
        if layer_id in args.index_source_layers:
            index += batch_size * groups * args.index_head_dim * bf16

    # Two shared RoPE tables (sliding-window theta and compressed theta), complex64.
    rope = 2 * seq_len * (args.rope_head_dim // 2) * 8
    total = window + compressed + index + partial + rope
    return {
        "window": window,
        "compressed_kv": compressed,
        "indexer": index,
        "partial_groups": partial,
        "rope_tables": rope,
        "total": total,
    }


def activation_estimate(args: ModelArgs, batch_size: int, chunk_len: int, checkpointing: bool) -> float:
    """Rough peak-activation estimate in bytes.

    This is intentionally conservative and clearly labelled as an estimate: the authoritative
    number is the peak measured by the 8-GPU memory gate.
    """
    hc = args.hc_mult
    per_token_layer = 2 * (
        4 * args.dim * hc
        + 3 * args.head_dim * args.n_heads
        + 4 * args.moe_inter_dim * (args.n_activated_experts + args.n_shared_experts)
    )
    retained = 1.0 if checkpointing else 2.2
    # Logits are chunked by the trainer, but keep one slice of the head live.
    logits = batch_size * 2048 * args.vocab_size * 4
    return batch_size * chunk_len * args.n_layers * per_token_layer * retained * 1.4 + logits


def validate(args: ModelArgs, seq_len: int, world_size: int, mp_size: int, gpu_count: int) -> list[str]:
    errors: list[str] = []
    if seq_len > args.max_seq_len:
        errors.append(
            f"seq_len={seq_len:,} exceeds the config's max_seq_len={args.max_seq_len:,}"
        )
    if mp_size > 1:
        for name, value in (
            ("n_heads", args.n_heads),
            ("o_groups", args.o_groups),
            ("n_routed_experts", args.n_routed_experts),
            ("dspark_n_routed_experts", args.dspark_n_routed_experts or 1),
        ):
            if value % mp_size:
                errors.append(f"{name}={value} is not divisible by mp_size={mp_size}")
    if gpu_count and world_size > gpu_count:
        errors.append(f"world_size={world_size} exceeds visible GPU count={gpu_count}")
    for index, ratio in enumerate(args.compress_ratios):
        if ratio < 0:
            errors.append(f"compress_ratios[{index}] is negative")
    if args.vision_n_layers:
        errors.append("vision_n_layers is non-zero; the production profile disables vision")
    return errors


def recommend(
    args: ModelArgs,
    batch_size: int,
    chunk_len: int,
    bptt_chunks: int,
    checkpointing: bool,
    per_gpu_gib: float,
    reserved_gib: float,
) -> tuple[int, int, str]:
    """Shrink ``chunk_len`` until the estimate fits; return ``(chunk_len, bptt_chunks, note)``."""
    chunk = chunk_len
    note = ""
    while chunk >= 1024:
        activation = activation_estimate(args, batch_size, chunk, checkpointing) / GIB
        if activation <= reserved_gib:
            break
        chunk //= 2
        note = f"reduced chunk_len from {chunk_len} to {chunk} to fit activations"
    if chunk < 1024:
        chunk = 1024
        note = "even chunk_len=1024 does not fit; reduce batch_size or enable offload"
    # Keep the differentiable window at roughly 32K tokens by default.
    bptt = max(1, min(bptt_chunks, max(1, 32768 // chunk)))
    return chunk, bptt, note


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="core/config_1.5b_production_256k.json")
    parser.add_argument("--seq_len", type=int, default=262144)
    parser.add_argument("--chunk_len", type=int, default=8192)
    parser.add_argument("--bptt_chunks", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1, help="micro batch per GPU")
    parser.add_argument("--world_size", type=int, default=8)
    parser.add_argument("--mp_size", type=int, default=1, help="legacy tensor-parallel size")
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer_32k")
    parser.add_argument(
        "--activation_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    opts = parser.parse_args()

    configure_model_parallel(opts.mp_size, 0)
    with open(PROJECT_ROOT / opts.config, encoding="utf-8") as handle:
        cfg = json.load(handle)
    args = ModelArgs(**cfg)

    hardware = hardware_report(opts.world_size)
    gpu_count = len(hardware["gpus"])
    errors = validate(args, opts.seq_len, opts.world_size, opts.mp_size, gpu_count)

    precision = opts.precision
    if precision == "auto":
        precision = "bf16" if hardware["gpus"] and hardware["gpus"][0]["bf16"] else (
            "fp16" if gpu_count else "fp32"
        )

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / opts.tokenizer))
        with torch.device("meta"):
            model = Transformer(args, tokenizer=tokenizer)
        params = sum(parameter.numel() for parameter in model.parameters())
        active = _load_active_parameter_count()(model)
    except Exception as exc:  # noqa: BLE001
        params = active = 0
        errors.append(f"could not build the meta model: {type(exc).__name__}: {exc}")

    caches = cache_bytes(args, opts.seq_len, opts.batch_size)
    master = params * 4 / opts.world_size
    grads = params * 4 / opts.world_size
    adam = params * 8 / opts.world_size
    gather = params * 2 / opts.world_size
    activation = activation_estimate(args, opts.batch_size, opts.chunk_len, opts.activation_checkpointing)
    total = master + grads + adam + gather + caches["total"] + activation

    per_gpu_gib = hardware["gpus"][0]["vram_gib"] if hardware["gpus"] else 0.0
    reserved_gib = per_gpu_gib * 0.85 if per_gpu_gib else 0.0
    recommended_chunk, recommended_bptt, note = recommend(
        args,
        opts.batch_size,
        opts.chunk_len,
        opts.bptt_chunks,
        opts.activation_checkpointing,
        per_gpu_gib,
        reserved_gib,
    )
    fits = not per_gpu_gib or total / GIB <= reserved_gib

    result = {
        "hardware": hardware,
        "config": str(PROJECT_ROOT / opts.config),
        "seq_len": opts.seq_len,
        "chunk_len": opts.chunk_len,
        "bptt_chunks": opts.bptt_chunks,
        "batch_size": opts.batch_size,
        "world_size": opts.world_size,
        "mp_size": opts.mp_size,
        "precision": precision,
        "parameters": params,
        "active_parameters": active,
        "memory_gib_per_rank": {
            "master_params": master / GIB,
            "gradients": grads / GIB,
            "adamw_states": adam / GIB,
            "fsdp_gather_buffer": gather / GIB,
            "attention_caches": caches["total"] / GIB,
            "cache_breakdown": {key: value / GIB for key, value in caches.items()},
            "activations_estimate": activation / GIB,
            "total_estimate": total / GIB,
            "gpu_vram": per_gpu_gib,
            "usable_budget": reserved_gib,
        },
        "fits": fits,
        "recommended": {
            "chunk_len": recommended_chunk,
            "bptt_chunks": recommended_bptt,
            "precision": precision,
            "note": note,
        },
        "errors": errors,
    }

    if opts.json:
        print(json.dumps(result, indent=2))
    else:
        print("=" * 78)
        print("NOESIS-Σ HARDWARE PREFLIGHT")
        print("=" * 78)
        print(f"torch {hardware['torch']} | CUDA: {hardware['cuda_available']} | NCCL: {hardware['nccl_available']}")
        for gpu in hardware["gpus"]:
            print(
                f"  GPU {gpu['index']}: {gpu['name']} | {gpu['vram_gib']:.1f} GiB | "
                f"bf16={gpu['bf16']} | sm_{gpu['compute_capability'].replace('.', '')}"
            )
        if hardware["gpus"]:
            print(f"  P2P-capable GPU pairs: {hardware.get('p2p_pairs', 0)}")
        print(f"\nconfig: {opts.config}")
        print(f"parameters: {params:,} ({params / 1e9:.3f}B) | active/token: {active:,} ({active / 1e9:.3f}B)")
        print(
            f"topology: world_size={opts.world_size} mp={opts.mp_size} | seq_len={opts.seq_len:,} "
            f"chunk={opts.chunk_len:,} bptt_chunks={opts.bptt_chunks} accumulation="
            f"{opts.chunk_len * opts.bptt_chunks:,} tokens | precision={precision}"
        )
        print("\nMemory plan per rank (GiB):")
        for name, value in result["memory_gib_per_rank"].items():
            if name == "cache_breakdown":
                continue
            print(f"  {name:24s} {value:7.2f}")
        print("  cache breakdown:")
        for name, value in result["memory_gib_per_rank"]["cache_breakdown"].items():
            print(f"    {name:22s} {value:7.3f}")
        verdict = "FITS" if fits else "DOES NOT FIT"
        print(f"\nverdict: {verdict} (estimate {total / GIB:.2f} GiB vs budget {reserved_gib:.2f} GiB)")
        if note:
            print(f"note: {note}")
        print(
            f"recommended: --chunk_len {recommended_chunk} --bptt_chunks {recommended_bptt} "
            f"--precision {precision}"
        )
        if errors:
            print("\nERRORS:")
            for error in errors:
                print(f"  - {error}")
        print("=" * 78)

    if errors or not fits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
