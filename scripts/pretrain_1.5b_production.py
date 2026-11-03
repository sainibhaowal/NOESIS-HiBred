#!/usr/bin/env python3
# NOESIS-Σ — Frontier 1.5B MoE Production Pretraining Engine (256K Context)
# Multi-Corpus 5-Stream Balanced Blend:
#   1. FineWeb-Edu (40%) - World Knowledge & Fluency
#   2. Cosmopedia-v2 (20%) - Textbooks & Reasoning
#   3. Python Code (20%) - Code Synthesis & Algorithmic Logic
#   4. GSM8K / Math (10%) - Multi-step Mathematical Chain-of-Thought
#   5. Function & Tool Calling (10%) - Structured JSON Tool Execution
#
# Hardware Targets:
#   - Kaggle GPU T4 x2 (32 GB) via DataParallel + 8-bit AdamW
#   - Kaggle TPU v5e-8 (128 GB) via PyTorch / XLA
#
# Path: scripts/pretrain_1.5b_production.py

from __future__ import annotations
import sys
import os
import argparse
import contextlib
import hashlib
import time
import math
import json
import itertools
from functools import partial
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.distributed as dist
from datetime import timedelta
from torch.optim import AdamW

# Resolve module paths
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CORE_DIR = PROJECT_ROOT / "core"
ENCODING_DIR = CORE_DIR / "encoding"

for p in (str(CORE_DIR), str(ENCODING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import (
    Block,
    ChunkState,
    DSparkBlock,
    ModelArgs,
    NoesisSigmaTransformer,
    ParallelEngramEmbedding,
    configure_model_parallel,
    sample,
)
from data_pipeline import PackedTokenSampler
from verifier_loss import SigmaVerifierLoss
from transformers import AutoTokenizer

# Detect TPU v5e-8 support
HAS_XLA = False
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False


def build_5corpus_stream(
    tokenizer: AutoTokenizer,
    max_tokens: int = 50_000_000,
    rank: int = 0,
    world_size: int = 1,
    allow_synthetic_fallback: bool = False,
) -> Iterator[int]:
    """Blend five corpora in the documented 40/20/20/10/10 document schedule.

    Production runs are fail-closed: a missing, exhausted, or broken remote stream
    raises an error rather than silently changing the corpus with embedded examples.
    ``allow_synthetic_fallback`` is reserved for offline smoke tests only.

    - 40% FineWeb-Edu (4 docs)
    - 20% Cosmopedia-v2 (2 docs)
    - 20% Python Code (2 scripts)
    - 10% GSM8K Chain-of-Thought Math (1 problem)
    - 10% Function & Tool Calling (1 JSON schema tool interaction)
    """
    from datasets import load_dataset
    eos_id = tokenizer.eos_token_id or 2

    print("[*] Initializing 5-Stream Multi-Corpus Pretraining Data Pipeline...")

    def unavailable(stream_name: str, exc: Exception):
        message = f"{stream_name} unavailable: {type(exc).__name__}: {exc}"
        if not allow_synthetic_fallback:
            raise RuntimeError(
                f"Production data check failed — {message}. "
                "Fix the dataset/HF connection, or use --allow_synthetic_fallback "
                "only for an offline smoke test."
            ) from exc
        print(f"  [!] {message}")
        return None

    def next_record(stream, stream_name: str):
        if stream is None:
            return None
        try:
            return next(stream)
        except StopIteration as exc:
            if not allow_synthetic_fallback:
                raise RuntimeError(
                    f"Production data check failed — {stream_name} was exhausted. "
                    "Do not continue with synthetic fallback data."
                ) from exc
            return None
        except Exception as exc:  # network / parquet failures happen lazily while streaming
            if not allow_synthetic_fallback:
                raise RuntimeError(
                    f"Production data check failed while reading {stream_name}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            print(f"  [!] {stream_name} read failed: {type(exc).__name__}: {exc}")
            return None

    def emit(text: str):
        for token in tokenizer.encode(text, add_special_tokens=False) + [eos_id]:
            yield token

    # 1. FineWeb-Edu
    try:
        ds_edu = iter(load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True))
        print("  [✓] Stream 1 (40%): HuggingFaceFW/fineweb-edu (High-grade Fluency & Knowledge)")
    except Exception as e:
        ds_edu = unavailable("Stream 1 FineWeb-Edu", e)

    # 2. Cosmopedia-v2
    try:
        ds_cosmo = iter(load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True))
        print("  [✓] Stream 2 (20%): HuggingFaceTB/cosmopedia-v2 (Synthetic Textbooks & Explanations)")
    except Exception as e:
        ds_cosmo = unavailable("Stream 2 Cosmopedia-v2", e)

    # 3. Python code.  Use direct public Parquet files, not the deprecated
    # dataset-loading script.  A single explicit shard is reliable for a canary;
    # production jobs may provide a vetted local/sharded list with NOESIS_CODE_DATA_FILES.
    code_dataset = os.environ.get("NOESIS_CODE_DATASET", "codeparrot/github-code")
    ds_code = None
    try:
        if code_dataset == "codeparrot/github-code":
            code_data_files = os.environ.get(
                "NOESIS_CODE_DATA_FILES",
                "hf://datasets/codeparrot/github-code/data/train-00000-of-01126.parquet",
            )
            raw_code = load_dataset(
                "parquet",
                data_files={"train": code_data_files},
                split="train",
                streaming=True,
            )

            def python_rows():
                for row in raw_code:
                    language = str(row.get("language", row.get("lang", ""))).lower()
                    if language == "python":
                        yield row

            ds_code = iter(python_rows())
            print(
                "  [✓] Stream 3 (20%): GitHub Python "
                f"(public Parquet, language-filtered) [{code_data_files}]"
            )
        else:
            ds_code = iter(load_dataset(code_dataset, split="train", streaming=True))
            print(f"  [✓] Stream 3 (20%): Python Code [{code_dataset}]")
    except Exception as e:
        ds_code = unavailable("Stream 3 Python code", e)

    synthetic_code_samples = [
        "def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "def binary_search(values: list[int], target: int) -> int:\n    lo, hi = 0, len(values) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if values[mid] == target:\n            return mid\n        if values[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
    ]
    syn_code_idx = rank
    if ds_code is None:
        print("  [!] Stream 3 fallback: embedded code smoke samples (offline smoke test only)")

    # 4. GSM8K Math CoT
    try:
        ds_gsm = iter(load_dataset("openai/gsm8k", "main", split="train", streaming=True))
        print("  [✓] Stream 4 (10%): OpenAI GSM8K (Step-by-Step Mathematical Reasoning)")
    except Exception as e:
        ds_gsm = unavailable("Stream 4 GSM8K", e)

    # 5. Function & Tool Calling (Glaive or synthetic tool calls)
    try:
        ds_tools = iter(load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True))
        print("  [✓] Stream 5 (10%): Glaive Function Calling v2 (Structured Tool Execution)")
    except Exception as e:
        ds_tools = unavailable("Stream 5 Glaive Function Calling v2", e)
        if ds_tools is None:
            print("  [!] Stream 5 fallback: embedded tool samples (offline smoke test only)")

    # Synthetic fallback tool call templates if dataset is offline
    synthetic_tool_samples = [
        "System: You have access to the following tools: [{'name': 'calculator', 'description': 'Evaluate math expression', 'parameters': {'expression': 'string'}}].\nUser: What is 482 * 19?\nAssistant: <tool_call>{\"name\": \"calculator\", \"parameters\": {\"expression\": \"482 * 19\"}}</tool_call>\nTool: 9158\nAssistant: The result of 482 * 19 is 9,158.",
        "System: You have access to the following tools: [{'name': 'weather_lookup', 'description': 'Get temperature for city', 'parameters': {'city': 'string'}}].\nUser: Check the weather in Tokyo.\nAssistant: <tool_call>{\"name\": \"weather_lookup\", \"parameters\": {\"city\": \"Tokyo\"}}</tool_call>\nTool: {\"temperature\": \"18C\", \"condition\": \"Clear\"}\nAssistant: The weather in Tokyo is currently 18°C and clear.",
        "System: You have access to the following tools: [{'name': 'execute_python', 'description': 'Run python snippet', 'parameters': {'code': 'string'}}].\nUser: Sort the list [5, 2, 9, 1].\nAssistant: <tool_call>{\"name\": \"execute_python\", \"parameters\": {\"code\": \"sorted([5, 2, 9, 1])\"}}</tool_call>\nTool: [1, 2, 5, 9]\nAssistant: The sorted list is [1, 2, 5, 9]."
    ]
    syn_tool_idx = 0

    # Keep the original iterators for cleanup.  ``itertools.islice`` used for
    # data-parallel partitioning has no close() method and otherwise leaves its
    # underlying HTTP/Parquet iterator to interpreter shutdown.
    streams_to_close = (ds_edu, ds_cosmo, ds_code, ds_gsm, ds_tools)

    # Data parallel: every rank keeps every world_size-th document, so GPUs never train on the same text.
    if world_size > 1:
        ds_edu, ds_cosmo, ds_code, ds_gsm = [
            itertools.islice(d, rank, None, world_size) if d is not None else None
            for d in (ds_edu, ds_cosmo, ds_code, ds_gsm)
        ]
        if ds_tools is not None:
            ds_tools = itertools.islice(ds_tools, rank, None, world_size)
        syn_tool_idx = rank

    token_count = 0
    try:
        while token_count < max_tokens:
            cycle_start = token_count
            # Stream 1: FineWeb-Edu (4 docs = 40%)
            if ds_edu is not None:
                for _ in range(4):
                    doc = next_record(ds_edu, "Stream 1 FineWeb-Edu")
                    if doc is None:
                        break
                    text = doc.get("text", "")
                    if text.strip():
                        for t in emit(text):
                            yield t
                            token_count += 1
                            if token_count >= max_tokens:
                                return

            # Stream 2: Cosmopedia-v2 (2 docs = 20%)
            if ds_cosmo is not None:
                for _ in range(2):
                    doc = next_record(ds_cosmo, "Stream 2 Cosmopedia-v2")
                    if doc is None:
                        break
                    text = doc.get("text", "")
                    if text.strip():
                        for t in emit(text):
                            yield t
                            token_count += 1
                            if token_count >= max_tokens:
                                return

            # Stream 3: Python Code (2 scripts = 20%)
            for _ in range(2):
                code_text = ""
                if ds_code is not None:
                    doc = next_record(ds_code, "Stream 3 Python code")
                    if doc is not None:
                        code_text = (
                            doc.get("text", "")
                            or doc.get("output", "")
                            or doc.get("code", "")
                            or doc.get("content", "")
                        )
                if not code_text.strip():
                    if not allow_synthetic_fallback:
                        raise RuntimeError(
                            "Production data check failed — Stream 3 returned an empty code record."
                        )
                    code_text = synthetic_code_samples[syn_code_idx % len(synthetic_code_samples)]
                    syn_code_idx += 1
                if code_text.strip():
                    for t in emit(f"```python\n{code_text}\n```"):
                        yield t
                        token_count += 1
                        if token_count >= max_tokens:
                            return

            # Stream 4: GSM8K Math CoT (1 problem = 10%)
            if ds_gsm is not None:
                item = next_record(ds_gsm, "Stream 4 GSM8K")
                if item is not None:
                    q = item.get("question", "")
                    a = item.get("answer", "")
                    cot_text = f"Question: {q}\n<think>\n{a}\n</think>\nFinal Answer: {a.split('####')[-1].strip() if '####' in a else a}\n"
                    for t in emit(cot_text):
                        yield t
                        token_count += 1
                        if token_count >= max_tokens:
                            return

            # Stream 5: Tool Calling & JSON Schemas (1 interaction = 10%)
            if ds_tools is not None:
                item = next_record(ds_tools, "Stream 5 Glaive Function Calling v2")
                if item is not None:
                    chat = item.get("chat", "") or item.get("text", "")
                    if chat.strip():
                        for t in emit(chat):
                            yield t
                            token_count += 1
                            if token_count >= max_tokens:
                                return
            else:
                sample_call = synthetic_tool_samples[syn_tool_idx % len(synthetic_tool_samples)]
                syn_tool_idx += 1
                for t in emit(sample_call):
                    yield t
                    token_count += 1
                    if token_count >= max_tokens:
                        return

            # A finite/temporarily exhausted remote iterator must not make
            # the training process spin forever without producing tokens.
            if token_count == cycle_start:
                if not allow_synthetic_fallback:
                    raise RuntimeError(
                        "Production data check failed — no usable tokens were produced by the corpus streams."
                    )
                fallback = synthetic_code_samples[syn_code_idx % len(synthetic_code_samples)]
                syn_code_idx += 1
                for t in emit(f"```python\n{fallback}\n```"):
                    yield t
                    token_count += 1
                    if token_count >= max_tokens:
                        return
    finally:
        # datasets' parquet/http iterators may own sockets or background
        # resources. Close the generator before torch.distributed teardown.
        for stream in streams_to_close:
            close = getattr(stream, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def build_batch(
    token_gen: Iterator[int],
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Collects tokens from generator and returns [B, seq_len] tensor."""
    needed = batch_size * seq_len
    tokens = []
    for t in token_gen:
        tokens.append(t)
        if len(tokens) >= needed:
            break
    if len(tokens) < needed:
        pad_id = 0
        tokens.extend([pad_id] * (needed - len(tokens)))
    return torch.tensor(tokens, dtype=torch.long, device=device).view(batch_size, seq_len)


def build_dry_stream(vocab_size: int, seed: int) -> Iterator[int]:
    """Offline random-token stream used only by --dry_data smoke tests (no network needed)."""
    gen = torch.Generator().manual_seed(seed)
    while True:
        for t in torch.randint(0, vocab_size, (4096,), generator=gen).tolist():
            yield t


def allreduce_grads(model: torch.nn.Module, world_size: int, bucket_elems: int = 25_000_000) -> None:
    """Averages gradients across data-parallel ranks in flat buckets.

    Every rank must issue identical collectives, so parameters without a gradient on this rank
    (e.g. an MoE expert that received no tokens) contribute zeros instead of being skipped.
    """
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    bucket, count = [], 0

    def flush():
        nonlocal bucket, count
        if not bucket:
            return
        flat = _flatten_dense_tensors([p.grad for p in bucket])
        dist.all_reduce(flat)
        flat.div_(world_size)
        for p, g in zip(bucket, _unflatten_dense_tensors(flat, [p.grad for p in bucket])):
            p.grad.copy_(g)
        bucket, count = [], 0

    for p in params:
        if bucket and (p.grad.dtype != bucket[0].grad.dtype or p.grad.device != bucket[0].grad.device):
            flush()
        bucket.append(p)
        count += p.numel()
        if count >= bucket_elems:
            flush()
    flush()


_QUANTIZED_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e8m0fnu", None),
        getattr(torch, "float4_e2m1fn_x2", None),
    )
    if isinstance(dtype, torch.dtype)
)


def unify_parameter_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """Give every trainable parameter the same dtype before FSDP wrapping.

    FSDP1 flattens each wrapped unit into one tensor, and a flat parameter must have a uniform
    dtype. NOESIS-Σ deliberately keeps Hyper-Connection couplings and the compressor in fp32 while
    the rest of the model is bf16, so the master weights are unified to fp32 and FSDP's
    ``MixedPrecision(param_dtype=...)`` performs the bf16/fp16 compute casts.
    """
    quantized = []
    for name, parameter in model.named_parameters():
        if parameter.dtype in _QUANTIZED_DTYPES:
            quantized.append(name)
            continue
        if not parameter.is_floating_point():
            raise RuntimeError(f"non-floating parameter {name} ({parameter.dtype}) cannot be wrapped by FSDP")
        if parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype)
    if quantized:
        raise RuntimeError(
            "FSDP runs cannot include quantized (fp8/fp4) weights; use a config with "
            f"dtype=bf16 and expert_dtype=null. Offending parameters: {quantized[:5]}"
        )


def active_parameter_count(model: torch.nn.Module) -> int:
    """Count the parameters used for one token under each MoE top-k setting.

    Routed experts are counted at the activation fraction of their owning MoE.
    Shared experts, attention, Engram, OSC, and MTP parameters are always
    counted.  This handles a DSpark MoE with a different top-k than the
    backbone instead of applying one global expert ratio to the whole model.
    """
    active = sum(parameter.numel() for name, parameter in model.named_parameters() if ".experts." not in name)
    for module in model.modules():
        experts = getattr(module, "experts", None)
        routed = getattr(module, "n_routed_experts", None)
        activated = getattr(module, "n_activated_experts", None)
        if experts is None or not isinstance(routed, int) or not isinstance(activated, int) or routed <= 0:
            continue
        routed_parameters = sum(
            parameter.numel()
            for expert in experts
            if expert is not None
            for parameter in expert.parameters()
        )
        active += round(routed_parameters * min(activated, routed) / routed)
    return active


def lr_at(step: int, total: int, peak: float, floor: float, warmup: int) -> float:
    """Linear warmup then cosine decay (from-scratch training diverges without warmup)."""
    if step <= warmup:
        return peak * step / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def save_checkpoint_atomic(state: dict, path: Path) -> None:
    """Writes to a temp file then renames, so a crash mid-save never corrupts latest.pt."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def _rng_state() -> dict:
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def run_pretrain_15b(
    config_path: str = "core/config_1.5b_production_256k.json",
    steps: int = 25000,
    batch_size: int = 1,
    seq_len: int = 262144,
    chunk_len: int = 8192,
    bptt_chunks: int = 4,
    grad_accum_steps: int = 1,
    lr: float = 3e-4,
    min_lr: float = 3e-5,
    warmup: int = 200,
    save_interval: int = 500,
    sample_interval: int = 250,
    resume_checkpoint: str | None = None,
    output_dir: str = "checkpoints/noesis_sigma_1.5b",
    pipeline: bool = False,
    dry_data: bool = False,
    allow_synthetic_fallback: bool = False,
    mtp_loss_weight: float = 0.20,
    mtp_confidence_weight: float = 0.05,
    mtp_positions_per_sequence: int = 4,
    backend: str = "auto",
    seed: int = 1234,
    use_fsdp: bool | None = None,
    activation_checkpointing: bool = True,
    precision: str = "auto",
    data_mode: str = "shards",
    token_shards_dir: str = "data/tokens",
    lm_token_chunk: int = 2048,
):
    if mtp_loss_weight < 0 or mtp_confidence_weight < 0:
        raise ValueError("MTP loss weights cannot be negative")
    if mtp_positions_per_sequence <= 0:
        raise ValueError("mtp_positions_per_sequence must be positive")
    if chunk_len <= 0 or seq_len <= 0:
        raise ValueError("chunk_len and seq_len must be positive")
    if chunk_len > seq_len:
        raise ValueError("chunk_len cannot exceed seq_len")
    if bptt_chunks <= 0:
        raise ValueError("bptt_chunks must be positive")
    if precision not in ("auto", "bf16", "fp16", "fp32"):
        raise ValueError("precision must be auto, bf16, fp16, or fp32")
    if data_mode not in ("shards", "streaming", "dry"):
        raise ValueError("data_mode must be shards, streaming, or dry")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0

    def log(*a, **k):
        if is_main:
            print(*a, **k)

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    if dry_data:
        data_mode = "dry"

    # 1. Device and process group. FSDP needs the process group before the model is wrapped.
    if HAS_XLA:
        if world_size > 1 or use_fsdp:
            raise RuntimeError("FSDP is not supported on the XLA/TPU path; use torchrun on CUDA.")
        device = xm.xla_device()
        device_name = f"TPU v5e ({xm.xla_real_device_type()})"
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        device_name = f"CUDA GPU {local_rank} ({torch.cuda.get_device_name(local_rank)})"
    else:
        device = torch.device("cpu")
        device_name = "CPU"

    fsdp_active = bool(use_fsdp) if use_fsdp is not None else world_size > 1
    if world_size > 1 or fsdp_active:
        be = backend if backend != "auto" else ("nccl" if device.type == "cuda" else "gloo")
        init_kwargs = {"backend": be, "timeout": timedelta(minutes=60)}
        if device.type == "cuda" and be == "nccl":
            init_kwargs["device_id"] = torch.device(f"cuda:{local_rank}")
        if not dist.is_initialized():
            if world_size == 1:
                # Single-process FSDP smoke runs have no torchrun environment; rendezvous with
                # itself on an ephemeral local port instead of requiring RANK/WORLD_SIZE.
                import socket

                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                dist.init_process_group(
                    backend=be,
                    init_method=f"tcp://127.0.0.1:{port}",
                    world_size=1,
                    rank=0,
                    timeout=timedelta(minutes=60),
                )
            else:
                dist.init_process_group(**init_kwargs)

    # The model is logically unsharded; FSDP owns the physical sharding.
    configure_model_parallel(1, 0)

    # FSDP/DDP *averages* gradients across ranks, so each rank must normalise by its own token
    # count. The global total is only used for throughput reporting.
    local_tokens_per_opt_step = batch_size * (seq_len - 1) * grad_accum_steps
    tokens_per_opt_step = local_tokens_per_opt_step * world_size
    log("=" * 96)
    log("NOESIS-Σ FRONTIER 1.5B FOUNDATION ENGINE — PRODUCTION CHUNKED PRETRAINING (FROM SCRATCH)")
    log(f"Accelerator: {device_name} | Data-parallel ranks (GPUs): {world_size} | FSDP: {fsdp_active}")
    log(f"Architecture Config: {config_path}")
    log(
        "Corpus: FineWeb-Edu (40%) | Cosmopedia-v2 (20%) | Code (20%) | GSM8K (10%) | Tools (10%)"
        if data_mode == "streaming"
        else f"Corpus: {data_mode}"
    )
    log(
        f"Outer context: {seq_len:,} | chunk: {chunk_len:,} | BPTT horizon: "
        f"{chunk_len * bptt_chunks:,} tokens ({bptt_chunks} chunks)"
    )
    log(f"Tokens/Step (all GPUs): {tokens_per_opt_step:,} | Steps: {steps:,} (~{tokens_per_opt_step * steps / 1e9:.3f}B tokens)")
    log("=" * 96)

    # 2. Configuration and tokenizer. Engram derives its compressed vocabulary map from this exact
    #    tokenizer, so load it before constructing the model.
    with open(PROJECT_ROOT / config_path, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    config_hash = _config_hash(cfg_dict)
    args = ModelArgs(**cfg_dict)
    args.max_batch_size = max(args.max_batch_size, batch_size)
    args.max_seq_len = max(args.max_seq_len, seq_len)
    tokenizer = AutoTokenizer.from_pretrained(
        str(PROJECT_ROOT / "checkpoints/tokenizer_32k"),
        model_max_length=args.max_seq_len,
    )
    tokenizer.model_max_length = args.max_seq_len
    if args.dspark_block_size:
        log(
            "MTP training: active "
            f"(weight={mtp_loss_weight:g}, confidence_weight={mtp_confidence_weight:g}, "
            f"positions/sequence={mtp_positions_per_sequence})"
        )

    # 3. Precision. Production bf16 where supported; fp16 + sharded scaler on T4/V100-class GPUs.
    if precision == "auto":
        if device.type == "cuda":
            precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            precision = "fp32"
    if precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but this GPU does not support it; use --precision fp16")
    compute_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]
    use_scaler = precision == "fp16"
    log(f"[+] Precision: {precision} (grad scaler: {use_scaler})")

    # 4. Model. Build on CPU in the storage dtype so every rank starts from the identical state;
    #    FSDP then shards and moves it. Building after init_process_group is safe because the
    #    logical model-parallel layout was fixed by configure_model_parallel above.
    torch.manual_seed(seed)
    log("[*] Instantiating NOESIS-Σ Transformer (random init, training from scratch)...")
    model_storage_dtype = compute_dtype if device.type != "cpu" else torch.float32
    previous_default = torch.get_default_dtype()
    torch.set_default_dtype(model_storage_dtype)
    try:
        model = NoesisSigmaTransformer(args, tokenizer=tokenizer)
    finally:
        torch.set_default_dtype(previous_default)
    if activation_checkpointing:
        model.enable_gradient_checkpointing(True)
        log("[+] Activation checkpointing: MoE sublayer only")

    scaler = None
    if fsdp_active:
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
            StateDictType,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

        if device.type == "cuda" and precision == "fp16":
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            scaler = ShardedGradScaler(enabled=True)

        # fp32 master weights + bf16/fp16 compute: FSDP requires a uniform parameter dtype, and
        # keeping the masters in fp32 is what makes AdamW's moments fp32.
        unify_parameter_dtype(model, torch.float32)
        mixed_precision = None
        if precision in ("bf16", "fp16"):
            mixed_precision = MixedPrecision(
                param_dtype=compute_dtype,
                reduce_dtype=torch.float32,
                # Buffers keep their deliberate dtypes: the compressor's fp32 pooling state and the
                # bf16 KV caches should not be silently converted by FSDP.
                buffer_dtype=None,
            )
        # Engram tables are plain bf16 trainable parameters now (see Phase 1), so they are sharded
        # and reduced like every other weight. Ignoring them would let each rank drift.
        wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block, DSparkBlock},
        )
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed_precision,
            device_id=device if device.type == "cuda" else None,
            sync_module_states=True,
            limit_all_gathers=True,
            use_orig_params=True,
        )
        log(f"[+] FSDP FULL_SHARD active across {world_size} rank(s)")
        in_dev = out_dev = device
    else:
        if use_scaler:
            scaler = torch.amp.GradScaler("cuda", enabled=True)
        is_pipeline_2gpu = bool(
            pipeline and world_size == 1 and device.type == "cuda" and torch.cuda.device_count() >= 2
        )
        if is_pipeline_2gpu:
            model.distribute_to_gpus()
            in_dev, out_dev = torch.device("cuda:0"), torch.device("cuda:1")
        else:
            model = model.to(device)
            in_dev = out_dev = device

    total_params = sum(p.numel() for p in model.parameters())
    active_total = active_parameter_count(model)
    log(f"[+] Total Parameters:       {total_params / 1e6:.2f}M ({total_params:,})")
    log(f"[+] Active per Token (MoE): {active_total / 1e6:.2f}M ({active_total:,})")
    log(f"[+] Latent Attractor:       Strang-split OSC block at mid-network layer {args.n_layers // 2} (Rank {args.osc_rank})")

    # 5. Optimizer. FSDP shards optimizer state alongside the parameters; full AdamW is used
    #    instead of bitsandbytes so quantized-optimizer/FSDP incompatibilities cannot appear.
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    loss_fn = SigmaVerifierLoss(alpha_attractor=0.05, beta_smoothness=0.01)

    start_step, history = 1, []

    def _fsdp_state_configs():
        return (
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
            FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
        )

    # 6. Resume. FSDP checkpoints hold a full, unsharded state dict saved on rank 0.
    data_state: dict | None = None
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        log(f"[*] Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location="cpu")
        saved_hash = ckpt.get("config_hash")
        if saved_hash and saved_hash != config_hash:
            raise RuntimeError(
                "Checkpoint config does not match the current config file; refusing a silent "
                "architecture mismatch. Pass the original --config or start a new run."
            )
        if fsdp_active:
            model_cfg, optim_cfg = _fsdp_state_configs()
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg):
                model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    optimizer.load_state_dict(
                        FSDP.optim_state_dict_to_load(model, optimizer, ckpt["optimizer_state_dict"])
                    )
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        history = ckpt.get("history", [])
        data_state = ckpt.get("data_state")
        if ckpt.get("rng_state"):
            torch.set_rng_state(ckpt["rng_state"]["torch"])
            if "cuda" in ckpt["rng_state"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt["rng_state"]["cuda"])
        del ckpt
        log(f"[+] Resumed at step {start_step}")
    elif resume_checkpoint:
        raise FileNotFoundError(f"--resume checkpoint not found: {resume_checkpoint}")

    # 7. Data. ``shards`` is the production path and resumable; ``streaming``/``dry`` remain for
    #    smoke tests and deliberately do not claim exact remote-position resumption.
    sampler: PackedTokenSampler | None = None
    token_gen = None

    if data_mode == "shards":
        sampler = PackedTokenSampler(
            PROJECT_ROOT / token_shards_dir,
            rank=rank,
            world_size=world_size,
            eos_token_id=tokenizer.eos_token_id or 2,
            seed=seed,
        )
        if data_state is not None:
            if isinstance(data_state, list):
                rank_state = data_state[rank] if rank < len(data_state) else None
            else:  # version-1 checkpoint compatibility
                rank_state = data_state
            if rank_state is not None:
                sampler.load_state_dict(rank_state)

        def next_micro_batch():
            ids, mask, segments = sampler.next_batch(batch_size, seq_len)
            return (
                torch.from_numpy(ids),
                torch.from_numpy(mask),
                torch.from_numpy(segments),
            )

    elif data_mode == "dry":
        token_gen = build_dry_stream(args.vocab_size, seed + rank)

        def next_micro_batch():
            ids = build_batch(token_gen, batch_size, seq_len, torch.device("cpu"))
            mask = torch.ones_like(ids, dtype=torch.bool)
            return ids, mask, torch.zeros_like(ids, dtype=torch.int32)

    else:
        token_gen = build_5corpus_stream(
            tokenizer=tokenizer,
            max_tokens=steps * tokens_per_opt_step * 2,
            rank=rank,
            world_size=world_size,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )

        def next_micro_batch():
            ids = build_batch(token_gen, batch_size, seq_len, torch.device("cpu"))
            mask = torch.ones_like(ids, dtype=torch.bool)
            return ids, mask, torch.zeros_like(ids, dtype=torch.int32)

    model.train()
    t_start = time.time()
    total_tokens_trained = 0
    n_chunks = math.ceil(seq_len / chunk_len)
    # One backward per BPTT window: chunk losses inside a window share the carried graph, so
    # calling backward per chunk would free tensors the next chunk still needs.
    total_backwards = grad_accum_steps * math.ceil(n_chunks / bptt_chunks)
    sampled_tokens_per_batch = batch_size * (seq_len - 1)

    def full_state_dicts():
        """Return ``(model_state, optimizer_state)``; FSDP returns an unsharded rank-0 copy."""
        if fsdp_active:
            model_cfg, optim_cfg = _fsdp_state_configs()
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, model_cfg, optim_cfg):
                return model.state_dict(), FSDP.optim_state_dict(model, optimizer)
        return model.state_dict(), optimizer.state_dict()

    log(f"\n{'Step':>6} | {'Loss':>8} | {'LM Loss':>8} | {'MTP Loss':>8} | {'ΔE (Attractor)':>15} | {'‖∇Φ(x)‖':>9} | {'PPL':>8} | {'Tok/s':>8} | {'VRAM(MB)':>9}")
    log("-" * 108)
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step, steps + 1):
        step_t0 = time.time()
        cur_lr = lr_at(step, steps, lr, min_lr, warmup)
        for g in optimizer.param_groups:
            g["lr"] = cur_lr

        numerator = 0.0  # loss numerator, normalised by the exact token count below
        lm_sum = 0.0
        mtp_sum = 0.0
        mtp_count = 0
        aux_sum = 0.0
        delta_e_sum = 0.0
        grad_phi_sum = 0.0
        backward_idx = 0

        for _ in range(grad_accum_steps):
            ids_cpu, mask_cpu, _segments = next_micro_batch()
            state = ChunkState()
            window_losses: list[torch.Tensor] = []

            for chunk_idx in range(n_chunks):
                start = chunk_idx * chunk_len
                end = min(start + chunk_len, seq_len)
                input_ids = ids_cpu[:, start:end].to(in_dev, non_blocking=True)
                targets = ids_cpu[:, start + 1 : end + 1].to(out_dev, non_blocking=True)
                chunk_mask = mask_cpu[:, start + 1 : end + 1].to(out_dev, non_blocking=True)
                length = min(input_ids.size(1), targets.size(1))
                input_ids = input_ids[:, :length]
                targets = targets[:, :length]
                chunk_mask = chunk_mask[:, :length]

                # ``model(...)`` (not ``model.forward_chunk``) so FSDP's root pre/post-forward
                # hooks run; ``state`` selects the chunked return contract.
                h_out, state, osc_metrics, main_hiddens = model(
                    input_ids, start_pos=start, state=state
                )
                h_out = h_out[:, :length]

                valid = int(chunk_mask.sum())
                if valid:
                    lm_chunk = model.chunked_lm_loss(
                        h_out, targets, loss_mask=chunk_mask, token_chunk=lm_token_chunk
                    )
                    aux, _ = loss_fn.auxiliary_loss(osc_metrics, h_out)
                    term = (lm_chunk + aux) * valid
                    numerator += float((lm_chunk.detach() + aux.detach()).item()) * valid
                    lm_sum += float(lm_chunk.detach().item()) * valid
                    aux_sum += float(aux.detach().item()) * valid
                else:
                    term = h_out.sum() * 0.0
                    lm_chunk = term

                delta_E = osc_metrics.get("energy_delta")
                if delta_E is not None:
                    delta_e_sum += float(delta_E.detach().float().mean().item()) * valid
                osc = getattr(model, "osc_thinking_block", None)
                if osc is not None:
                    with torch.no_grad():
                        # Between forward and backward FSDP keeps parameters in the compute dtype
                        # and activations may still be fp32. Match them explicitly instead of
                        # calling the submodule directly, which bypasses FSDP's casts.
                        param_dtype = next(osc.icnn.parameters()).dtype
                        probe = h_out.detach().to(param_dtype)
                        x_norm = torch.nn.functional.layer_norm(
                            probe,
                            (probe.size(-1),),
                            osc.norm.weight.to(param_dtype),
                            osc.norm.bias.to(param_dtype),
                            osc.norm.eps,
                        )
                        grad_phi_sum += float(
                            torch.linalg.norm(osc.icnn.grad(x_norm), dim=-1).mean()
                        ) * valid

                if args.dspark_block_size and main_hiddens:
                    main_hidden = torch.cat(main_hiddens, dim=-1)
                    mtp_outputs = model.forward_mtp_train(
                        main_hidden,
                        targets,
                        positions_per_sequence=mtp_positions_per_sequence,
                    )
                    if mtp_outputs is not None:
                        mtp_logits, mtp_targets, mtp_confidence = mtp_outputs
                        mtp_loss = nn.functional.cross_entropy(
                            mtp_logits.reshape(-1, mtp_logits.size(-1)),
                            mtp_targets.reshape(-1),
                        )
                        confidence_targets = (
                            mtp_logits.detach().argmax(dim=-1) == mtp_targets
                        ).to(mtp_confidence.dtype)
                        confidence_loss = nn.functional.binary_cross_entropy_with_logits(
                            mtp_confidence, confidence_targets
                        )
                        n_mtp = mtp_targets.numel()
                        term = term + (
                            mtp_loss_weight * mtp_loss
                            + mtp_confidence_weight * confidence_loss
                        ) * n_mtp
                        mtp_sum += float(mtp_loss.detach().item()) * n_mtp
                        mtp_count += n_mtp

                window_losses.append(term / local_tokens_per_opt_step)

                at_window_end = (
                    (chunk_idx + 1) % bptt_chunks == 0 or chunk_idx == n_chunks - 1
                )
                if not at_window_end:
                    continue
                is_last_backward = backward_idx == total_backwards - 1
                sync_ctx = (
                    model.no_sync()
                    if fsdp_active and not is_last_backward
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    torch.stack(window_losses).sum().backward()
                window_losses.clear()
                backward_idx += 1
                # Cut the truncated-BPTT graph only after the whole window has been backpropped;
                # the next window starts from a detached carried state.
                state.detach()

        if fsdp_active:
            # FSDP already reduced gradients during the final backward.
            gnorm = model.clip_grad_norm_(1.0)
        else:
            if world_size > 1:
                allreduce_grads(model, world_size)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        local_tokens = sampled_tokens_per_batch * grad_accum_steps
        step_loss = numerator / max(local_tokens, 1)
        step_loss_lm = lm_sum / max(local_tokens, 1)
        step_mtp_loss = mtp_sum / mtp_count if mtp_count else 0.0
        step_delta_e = delta_e_sum / max(local_tokens, 1)
        step_grad_norm = grad_phi_sum / max(local_tokens, 1)

        # Average loss metrics across ranks for logging. Gradients are already global.
        if world_size > 1:
            metrics = torch.tensor(
                [step_loss, step_loss_lm, step_mtp_loss, step_delta_e, step_grad_norm],
                dtype=torch.float32,
                device=device,
            )
            dist.all_reduce(metrics)
            metrics /= world_size
            step_loss, step_loss_lm, step_mtp_loss, step_delta_e, step_grad_norm = metrics.tolist()

        if not (math.isfinite(step_loss) and bool(torch.isfinite(gnorm))):
            log(f"[!] Step {step}: non-finite loss/grad — update skipped")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        elif HAS_XLA:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step_time = time.time() - step_t0
        total_tokens_trained += tokens_per_opt_step
        tokens_per_sec = tokens_per_opt_step / max(step_time, 1e-5)
        ppl = math.exp(min(step_loss_lm, 20.0))
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else 0.0

        history.append({"step": step, "loss": step_loss, "loss_lm": step_loss_lm, "loss_mtp": step_mtp_loss, "energy_delta": step_delta_e,
                        "grad_phi_norm": step_grad_norm, "perplexity": ppl, "tokens_per_sec": tokens_per_sec,
                        "vram_mb": vram_mb, "lr": cur_lr})
        log(f"{step:6d} | {step_loss:8.4f} | {step_loss_lm:8.4f} | {step_mtp_loss:8.4f} | {step_delta_e:15.4f} | {step_grad_norm:9.4f} | {ppl:8.2f} | {tokens_per_sec:8.1f} | {vram_mb:9.1f}")

        # Checkpoint: every rank participates in the FSDP full-state collectives; rank 0 writes.
        if step % save_interval == 0 or step == steps:
            model_state, optim_state = full_state_dicts()
            rank_data_states: list[dict | None] = [None] * world_size
            if world_size > 1:
                local_state = sampler.state_dict() if sampler is not None else None
                dist.all_gather_object(rank_data_states, local_state)
            elif sampler is not None:
                rank_data_states[0] = sampler.state_dict()
            if is_main:
                latest_path = out_path / "latest.pt"
                save_checkpoint_atomic({
                    "step": step,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optim_state,
                    "config": cfg_dict,
                    "config_hash": config_hash,
                    "loss": step_loss,
                    "history": history,
                    "data_state": rank_data_states,
                    "rng_state": _rng_state(),
                    "precision": precision,
                    "chunk_len": chunk_len,
                    "bptt_chunks": bptt_chunks,
                    "version": 2,
                }, latest_path)
                with open(out_path / "training_history_1.5b.json", "w", encoding="utf-8") as f:
                    json.dump(history, f)
                print(f"[+] Saved Checkpoint: {latest_path} ({os.path.getsize(latest_path) / (1024**2):.1f} MB)")

        # Generation preview: rank 0 only, and can never kill the training run
        if is_main and step % sample_interval == 0:
            try:
                model.eval()
                with torch.no_grad():
                    prompts = [
                        "Question: If x + 5 = 12, what is x?\n<think>",
                        "def fibonacci(n: int) -> int:\n    \"\"\"Compute the n-th Fibonacci number.\"\"\"\n",
                        "System: You have tool 'calculator'.\nUser: Calculate 15 * 8.\nAssistant: <tool_call>",
                    ]
                    test_prompt = prompts[(step // sample_interval) % len(prompts)]
                    prompt_ids = tokenizer.encode(test_prompt, add_special_tokens=False)
                    gen = list(prompt_ids)
                    for _ in range(32):
                        seq = torch.tensor([gen], dtype=torch.long, device=in_dev)
                        _, lg, _ = model(seq)
                        nxt = sample(lg, temperature=0.7).item()
                        gen.append(nxt)
                        if nxt == (tokenizer.eos_token_id or 2):
                            break
                    print(f"\n[Generation Preview @ Step {step}]\nPrompt: {test_prompt}\nGenerated: {tokenizer.decode(gen[len(prompt_ids):])!r}\n")
            except Exception as e:  # noqa: BLE001
                print(f"[!] Preview skipped: {type(e).__name__}: {e}")
            finally:
                model.train()

    # Close the HF streaming generator while its HTTP/parquet resources are
    # still valid.  This avoids late interpreter-shutdown "Bad file
    # descriptor" warnings after a successful training run.
    close_token_gen = getattr(token_gen, "close", None)
    if callable(close_token_gen):
        close_token_gen()
    del token_gen, tokenizer

    if history:
        log("=" * 96)
        log("[✔] NOESIS-Σ PRE-TRAINING COMPLETE!")
        log(f"Steps: {steps:,} | Tokens this run: {total_tokens_trained:,} | Elapsed: {time.time() - t_start:.1f}s")
        log(f"Final Loss: {history[-1]['loss']:.4f} | PPL: {history[-1]['perplexity']:.2f} | ΔE: {history[-1]['energy_delta']:.4f}")
        log(f"Checkpoint: {out_path / 'latest.pt'}")
        log("=" * 96)
    if world_size > 1 and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "NOESIS-Σ 1.5B pretraining. Production: torchrun --nproc_per_node=8 with FSDP, "
            "chunked 256K sequences, and memory-mapped token shards."
        )
    )
    parser.add_argument("--config", type=str, default="core/config_1.5b_production_256k.json")
    parser.add_argument("--steps", type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=1, help="Micro batch size PER GPU")
    parser.add_argument("--seq_len", type=int, default=262144, help="Total training sequence length")
    parser.add_argument(
        "--chunk_len",
        type=int,
        default=8192,
        help="Tokens processed per forward chunk; the outer seq_len is carried across chunks.",
    )
    parser.add_argument(
        "--bptt_chunks",
        type=int,
        default=4,
        help="Differentiable horizon in chunks (chunk_len * bptt_chunks tokens).",
    )
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=200, help="LR warmup steps")
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--sample_interval", type=int, default=250)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints/noesis_sigma_1.5b")
    parser.add_argument("--pipeline", action="store_true", help="Naive 2-GPU model split (legacy, not for FSDP)")
    parser.add_argument(
        "--data_mode",
        choices=("shards", "streaming", "dry"),
        default="shards",
        help="shards: memory-mapped corpus (production); streaming: legacy HF streams; dry: random tokens.",
    )
    parser.add_argument("--token_shards_dir", type=str, default="data/tokens")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="auto selects bf16 when the GPU supports it, otherwise fp16 (T4/V100) or fp32 on CPU.",
    )
    parser.add_argument(
        "--fsdp",
        dest="use_fsdp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Force FSDP on/off. Default: on for multi-rank runs, off for single-rank smoke tests.",
    )
    parser.add_argument(
        "--activation_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Checkpoint the MoE sublayer to bound activation memory (recommended at long context).",
    )
    parser.add_argument(
        "--lm_token_chunk",
        type=int,
        default=2048,
        help="Token slice for the LM head/loss; keeps logits at [lm_token_chunk, vocab].",
    )
    parser.add_argument("--dry_data", action="store_true", help="Random tokens, no network (smoke tests)")
    parser.add_argument(
        "--allow_synthetic_fallback",
        action="store_true",
        help="Offline smoke tests only. Production runs fail instead of silently replacing a missing corpus.",
    )
    parser.add_argument(
        "--mtp_loss_weight",
        type=float,
        default=0.20,
        help="Weight for teacher-forced DSpark draft cross-entropy when MTP is enabled.",
    )
    parser.add_argument(
        "--mtp_confidence_weight",
        type=float,
        default=0.05,
        help="Weight for DSpark draft-confidence calibration loss when MTP is enabled.",
    )
    parser.add_argument(
        "--mtp_positions_per_sequence",
        type=int,
        default=4,
        help="Bounded number of teacher-forced MTP positions evaluated per training sequence.",
    )
    parser.add_argument("--backend", type=str, default="auto", help="auto | nccl | gloo")
    parser.add_argument("--seed", type=int, default=1234)
    a = parser.parse_args()

    run_pretrain_15b(
        config_path=a.config, steps=a.steps, batch_size=a.batch_size, seq_len=a.seq_len,
        chunk_len=a.chunk_len, bptt_chunks=a.bptt_chunks,
        grad_accum_steps=a.grad_accum, lr=a.lr, min_lr=a.min_lr, warmup=a.warmup,
        save_interval=a.save_interval, sample_interval=a.sample_interval,
        resume_checkpoint=a.resume, output_dir=a.output_dir, pipeline=a.pipeline,
        dry_data=a.dry_data, allow_synthetic_fallback=a.allow_synthetic_fallback,
        mtp_loss_weight=a.mtp_loss_weight, mtp_confidence_weight=a.mtp_confidence_weight,
        mtp_positions_per_sequence=a.mtp_positions_per_sequence,
        backend=a.backend, seed=a.seed,
        use_fsdp=a.use_fsdp, activation_checkpointing=a.activation_checkpointing,
        precision=a.precision, data_mode=a.data_mode, token_shards_dir=a.token_shards_dir,
        lm_token_chunk=a.lm_token_chunk,
    )
