# Long-Context Training on 8 x 16 GB GPUs

This document describes the production path for training NOESIS-Σ at a
262,144-token total context on eight 16 GB CUDA GPUs. The profile is
[`core/config_1.5b_production_256k.json`](../core/config_1.5b_production_256k.json).

## Topology

- Eight ranks, one GPU per rank, NCCL process group.
- FSDP `FULL_SHARD` with `use_orig_params=True`, `sync_module_states=True`,
  `limit_all_gathers=True`.
- Logical model-parallel size is 1. FSDP sharding does not imply tensor
  parallelism, so `configure_model_parallel(1, 0)` is called before model
  construction.
- Parameters are unified to fp32 before wrapping. FSDP requires one dtype per
  flat parameter, and fp32 master weights are needed for stable AdamW updates.
  A `MixedPrecision` policy casts to bf16 for compute while reductions stay in
  fp32.

## Per-rank memory plan

The preflight script computes the plan from the live configuration:

```bash
python scripts/check_hardware.py \
  --config core/config_1.5b_production_256k.json \
  --seq_len 262144 --chunk_len 4096 --bptt_chunks 4 --world_size 8
```

At 16 GB per rank the conservative activation estimate recommends
`chunk_len=4096`. Use 8192 only when the measured memory gate shows headroom.

Representative values for 1,517,979,352 total parameters at world size 8:

| Component | GiB per rank |
| :--- | ---: |
| fp32 master weights | 0.71 |
| fp32 gradients | 0.71 |
| AdamW moments | 1.41 |
| FSDP all-gather buffer | 0.35 |
| Attention and MTP caches at 262K | 0.14 |
| Activations at `chunk_len=4096` | config and checkpointing dependent |
| Budget at 85 percent of 16 GB | 13.60 |

The activation estimate carries a safety factor and is intentionally
conservative. The authoritative value is the peak allocated memory reported by
the 8-GPU memory gate.

Cache accounting is exact and covers four groups:

- sliding-window KV for every attention and MTP layer;
- compressed KV for every source layer;
- indexer keys for every index source layer;
- fp32 partial-group state (`kv_state` and `score_state`) plus two shared RoPE
  tables.

## Chunked sequence processing

`Transformer.forward_chunk(input_ids, start_pos, state)` is the single entry
point. It processes one chunk and returns `(h_norm, state, osc_metrics,
mtp_hiddens)`. `ChunkState` carries:

- `attention.compress_kv`: compressed latents completed so far;
- `attention.index_k`: persistent indexer key cache view;
- `attention.topk_idxs` and `attention.candidates`: results published by the
  source layer for later layers in the same forward;
- `retro`: OSC retrograde state;
- `mtp_window` and `mtp_start_pos`: draft cache bookkeeping.

The attention path writes completed compression groups to persistent buffers
at absolute group indices on every path, including prefill. Index keys are
appended at absolute group positions. A group that spans a chunk boundary is
completed by the next chunk from `kv_state` and `score_state`.

All cache writes are detached. A later chunk must not traverse an earlier
chunk's freed graph, so the carried caches are functional state rather than a
differentiable path. Within a chunk, the compressor latents, window KV, and
MTP conditioning stay on the graph.

## Differentiable horizon

The trainer accumulates losses for `bptt_chunks` consecutive chunks and calls
`backward` once per window. The default window is `4 x 8192 = 32,768` tokens.
`ChunkState.detach()` cuts the graph at every window boundary.

Cross-chunk gradient flow is limited to the OSC retrograde state, which is a
small `[batch, dim]` tensor. Attention and compression caches are detached at
every boundary. Full 262K backpropagation would require context parallelism
and is not implemented. Long-range learning therefore relies on the carried
state and on the 32K differentiable horizon.

## Trainer command

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/pretrain_1.5b_production.py \
  --config core/config_1.5b_production_256k.json \
  --steps 25000 \
  --batch_size 1 \
  --seq_len 262144 \
  --chunk_len 4096 \
  --bptt_chunks 4 \
  --lr 3e-4 \
  --save_interval 500 \
  --sample_interval 250 \
  --output_dir checkpoints/noesis_sigma_1.5b
```

One optimizer step processes `batch_size * seq_len * world_size` tokens.
Chunking is internal accumulation and does not change that count. FSDP
all-reduce is skipped for every non-final window through `model.no_sync()`.

## Data pipeline

`--data_mode streaming` keeps the live Hugging Face iterators. For a
network-free production run, tokenize the blend once:

```bash
python scripts/build_tokenized_corpus.py \
  --output_dir data/tokenized \
  --tokenizer checkpoints/tokenizer_32k
```

`core/data_pipeline.py` exposes `PackedTokenSampler`, which memory-maps the
shards, inserts an EOS between documents, returns a loss mask and segment ids,
and resumes from `(shard, offset, epoch)`. Rank striding is explicit, so every
rank reads a distinct slice.

## Checkpoint export

Training checkpoints are FSDP full state dicts inside `latest.pt`. Export to a
portable directory:

```bash
python scripts/export_checkpoint.py \
  --checkpoint checkpoints/noesis_sigma_1.5b \
  --output-dir checkpoints/noesis_sigma_1.5b/exported \
  --dtype bf16
```

The exporter strips FSDP and `torch.compile` prefixes, validates every shape
against a meta-device model, rejects flat FSDP parameters, and writes
`model.safetensors`, `config.json`, and `tokenizer/`. `core/generate.py`
accepts the exported directory directly.

## Generation

The generation loop enforces `prompt + max_new_tokens <= max_seq_len`. Prefill
runs in `--prefill-chunk` token blocks, one forward per block, and decode runs
one token at a time. A 190K prompt with 64K output and a 128K prompt with 128K
output both fit the 262,144-token total budget. Over-budget requests raise
unless `--truncate-prompt` is set for text-only prompts.

```bash
python core/generate.py \
  --ckpt-path checkpoints/noesis_sigma_1.5b/exported \
  --input-file examples/example.txt \
  --max-new-tokens 65536 \
  --prefill-chunk 8192
```

## Verification gates

| Gate | Check |
| :--- | :--- |
| Unit and quality | `python -m pytest -q -m "not slow"` and `python scripts/quality_gate.py`. |
| Kernel equivalence | Chunked sparse attention, compressor pooling, indexer chunks, OSC scan, and chunked LM loss match full-sequence references. |
| Cache correctness | Chunked prefill plus one decode step equals a full forward, with compression and MTP enabled. |
| Single-GPU overfit | Loss decreases on a tiny corpus and checkpoint reload reproduces the next-step loss. |
| Two-rank FSDP | `tests/test_fsdp_smoke.py` runs one step with finite gradients. Marked slow and skipped below two CUDA GPUs. |
| Eight-GPU memory | Peak allocated memory stays below 15 GB per rank at `seq_len=262144`, `chunk_len=4096`, `bptt_chunks=4`. Try 8192 only after the 4096 gate passes with headroom. |
| Generation | 190K plus 64K and 128K plus 128K complete; over-budget requests raise. |
| Export | Exported safetensors loads in `core/generate.py` and reproduces training logits. |

## Known limitations

- Cross-chunk gradients are truncated at every BPTT window. The carried
  attention caches are not differentiable across those boundaries.
- The indexer selects keys with a discrete top-k, so its projection parameters
  receive no direct gradient from the language-model loss. Distilling the
  indexer against main-attention probabilities is a follow-up item; the
  production profile therefore treats retrieval quality as untrained.
- DSpark drafting is trained and exposed through `forward_spec`, but the
  generation loop does not implement speculative acceptance and rollback.
- Compute for 262K-token training is far beyond a 128 GB, 8-GPU budget at
  frontier token counts. The path makes the code correct and memory-safe, not
  frontier-scale training feasible. Curriculum runs at 8K to 32K should
  establish stability before 262K sequences are attempted.
- Vision is implemented but disabled in the production profile.
