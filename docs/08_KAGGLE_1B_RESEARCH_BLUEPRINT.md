# NOESIS-Σ: Kaggle Training and Evaluation Blueprint

## Scope

This blueprint describes a controlled Kaggle workflow for the current 1.5B
text model. It does not claim that a 1B configuration, a baseline suite, or
benchmark results already exist in this repository.

The Kaggle workflow is a legacy pilot path. The production target is eight
16 GB CUDA GPUs, documented in
[`docs/09_LONG_CONTEXT_TRAINING.md`](09_LONG_CONTEXT_TRAINING.md). Kaggle TPU
sessions are not supported by the current NCCL and FSDP trainer.

Current source configurations are:

| File | Purpose |
| :--- | :--- |
| [`core/config_1.5b_production_256k.json`](../core/config_1.5b_production_256k.json) | Production profile: 262,144-token total context, all text features, vision disabled. |
| [`core/config_1.5b.json`](../core/config_1.5b.json) | 1.5B text-model architecture configuration with a 1M context limit. |
| [`core/config_1.5b_256k.json`](../core/config_1.5b_256k.json) | Verified 256K-context variant used by the Kaggle runbook. |
| [`core/config_671b.json`](../core/config_671b.json) | Large reference configuration. It is not an approved training target. |

## Preconditions

Before starting a run, confirm these values in the active Kaggle notebook:

1. accelerator type, GPU count, and usable memory;
2. current session limit, weekly quota, and persistent-storage quota;
3. internet availability for streamed datasets;
4. attached dataset version and archive checksum;
5. required Python packages and tokenizer files; and
6. a configured Hugging Face token if the selected source requires one.

Do not copy quota values or storage limits from older notebooks. They are
account- and time-dependent.

## Phase 1: source and environment validation

1. Upload the current `noesis_kaggle_bundle.tar.gz` as a new Kaggle dataset
   version and attach that exact version to the notebook.
2. Extract the archive to `/kaggle/working`; Kaggle input paths are read-only.
3. Run the repository quality gate or its equivalent notebook checks.
4. Confirm that the startup log identifies the direct public GitHub Python
   Parquet source and does not contain a synthetic-data fallback.

Use the commands in [`Model_Run.md`](Model_Run.md) as the source of truth for
the current archive layout and training command.

## Phase 2: one-step real-data gate

Run a single optimizer step with the intended model configuration, corpus mix,
batch size, sequence length, gradient accumulation, and optimizer settings.

Record:

- source dataset identifiers and revisions;
- effective tokens per optimizer step;
- loss, gradient norm, energy metrics, throughput, and memory;
- whether both distributed ranks initialized; and
- the saved checkpoint path and file size.

Continue only when all values are finite, the checkpoint reloads, and no stream
uses a synthetic fallback.

## Phase 3: engineering pilot

Run a bounded pilot before selecting a long token budget. Validate at steps 10,
100, 500, and 1,000 where the available session time permits.

| Check | Acceptance criterion |
| :--- | :--- |
| Loss and gradients | Finite values with no sustained numerical divergence. |
| Data pipeline | Required streams remain available and retain the documented mixture. |
| Checkpointing | The rank-zero checkpoint reloads model and optimizer state. |
| Resume behavior | Resumption completes using the same configuration. |
| Memory | Peak allocation remains below the available per-device memory. |
| Throughput | Tokens per second is recorded with hardware and runtime settings. |

Live streamed datasets do not restore the exact remote record position during
resume. Cache indexed shards when exact-order reproducibility is required.

## Phase 4: evaluation design

Define the evaluation before scaled training:

1. Freeze a held-out validation set that is excluded from training.
2. Select benchmark tasks and use documented evaluation
   protocols.
3. Record prompt templates, decoding parameters, scoring code, and revisions.
4. Compare against baselines under matched data, token, and compute budgets.
5. Run multiple seeds before reporting a comparative conclusion.

Benchmark tables must contain measured results only. Do not include projected
accuracy, latency, memory, or perplexity values.

## Planned extensions

A separate 1B configuration or a reduced OSC reference model may be useful for
cost-controlled ablations. These are future work items. Add them only with a
configuration file, tests, quality-gate coverage, and an explicit experiment
plan.

## Operational record

For every Kaggle run, preserve:

- repository commit and archive checksum;
- configuration file and command line;
- Kaggle accelerator details and package versions;
- dataset identifiers, revisions, and licenses;
- checkpoint paths and checksums;
- training and validation metrics; and
- known failures, retries, and deviations from the planned run.
