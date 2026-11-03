# NOESIS-Σ: Research and Compute Planning

## Purpose

This document defines an evidence-based path from implementation validation to
training, evaluation, and possible publication. It does not predict benchmark
outcomes, acceptance decisions, funding decisions, hardware availability, or
training duration.

The active implementation target is the text-only 1.5B production profile in
[`core/config_1.5b_production_256k.json`](../core/config_1.5b_production_256k.json):
262,144-token total context, FSDP `FULL_SHARD` over eight 16 GB CUDA GPUs, all
text features enabled, and vision disabled. The 671B configuration is a
reference artifact and is not a training target in the current workflow.

## Compute planning principles

1. Confirm the allocated accelerator, memory, session limits, and storage in
   the provider console before starting a run. Provider quotas and hardware
   availability change over time.
2. Measure throughput, memory, and checkpoint size with a one-step real-data
   gate before selecting a token budget.
3. Treat streamed-data retries and session interruptions as operational risks.
   Keep a validated checkpoint and record the dataset revision or shard list.
4. Do not infer long-context capability from short-sequence training. A
   512-token or 1024-token run is a short-context training test.
5. Use the same model configuration, data mix, optimizer, and effective batch
   size when resuming a run unless the experiment is explicitly restarted.

## Execution stages

| Stage | Objective | Required evidence | Exit condition |
| :--- | :--- | :--- | :--- |
| Source validation | Verify the repository before remote execution. | Quality gate output and clean generated artifacts. | All local checks pass. |
| Data gate | Verify real data access and optimizer operation. | One completed step, finite loss and gradients, all required streams, checkpoint. | Data source and checkpoint are confirmed. |
| Engineering pilot | Detect stability, memory, or resume defects. | Loss history, throughput, memory, and checkpoint reload at scheduled steps. | Metrics remain finite and the run resumes correctly. |
| Evaluation pilot | Establish a baseline for later comparison. | Frozen validation set and documented evaluation command. | Baseline metrics are recorded. |
| Scaled training | Produce a model candidate. | Dataset provenance, run configuration, checkpoints, and validation history. | Token budget and evaluation plan are complete. |
| Research report | Support externally reviewable claims. | Multi-seed results, matched baselines, uncertainty, and reproducible artifacts. | Claims are limited to the reported evidence. |

## Resource selection

Select resources based on the measured requirements of the experiment:

| Resource type | Suitable use | Operational requirement |
| :--- | :--- | :--- |
| Local development hardware | Unit tests, source validation, and small CPU or GPU checks. | Do not use for unbounded training jobs. |
| Eight CUDA GPUs, 16 GB each | The supported production target for the 1.5B profile. | Use the hardware preflight, FSDP `FULL_SHARD`, `chunk_len=4096`, and the memory gate before a long run. |
| Kaggle GPU sessions | Short pilots and checkpointed experiments. | Confirm current quotas, enable required network access, and export checkpoints before session expiry. |
| Kaggle TPU sessions | Legacy and unsupported for the current trainer. The distributed path uses NCCL and FSDP rather than XLA. | Do not treat TPU sessions as a production target without a separate XLA implementation and validation. |
| Dedicated or grant-funded accelerators | Larger controlled training runs. | Specify the token budget, dataset license, evaluation plan, checkpoint retention, and reproducibility artifacts. |

## Publication evidence requirements

An architecture description and a small-corpus ablation are insufficient for a
comparative research claim. A report should include:

- a documented dataset and license;
- fixed training and validation splits;
- matched baseline configurations and token budgets;
- at least three seeds when reporting comparative metrics;
- mean, standard deviation, and failure cases;
- benchmark prompts, scoring rules, and implementation versions;
- measured latency and memory on named hardware; and
- a statement of limitations.

## Current repository gates

Run the documented local and Kaggle preparation steps in
[`Model_Run.md`](Model_Run.md). The repository's pre-push quality gate verifies
compilation, examples, tests, and coverage. It does not validate model quality
or substitute for training and benchmark evaluation.
