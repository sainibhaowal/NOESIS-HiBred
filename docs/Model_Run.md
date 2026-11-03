# NOESIS-Σ 1.5B: Verified Kaggle Runbook

This runbook defines the verified workflow for the current 1.5B NOESIS-Σ text
model. It supersedes older commands that reference files not present in this
repository.

## What the local checks prove

- Python source compiles.
- Encoding, core, production-example, and non-training test suites pass.
- The five real streaming sources can produce tokens, including the public
  language-filtered GitHub Python Parquet source.
- The trainer writes one atomic rank-0 checkpoint and can resume model and
  optimizer state. A live remote stream cannot restore its exact data position;
  use cached indexed shards when exact data-order resumption matters.

They do **not** prove that a randomly initialized model is useful. Model
quality requires a held-out evaluation set and thousands or billions of real
training tokens.

## Context configurations

- `core/config_1.5b_production_256k.json` is the production profile:
  `max_seq_len=262144`, all text features enabled, vision disabled, FSDP
  `FULL_SHARD` over 8 x 16 GB GPUs, `chunk_len=4096`, `bptt_chunks=4`.
- `core/config_1.5b.json` is the original 1M-context configuration.
- `core/config_1.5b_256k.json` is the 256K-context configuration. It changes
  only `max_seq_len`; the 1.5B architecture is otherwise identical.
- `core/config_1.5b_full_text_256k.json` is a legacy alias for older
  experiments.

The 1.5B configurations have `vision_n_layers: 0`; these training runs are
text-only. A run with `--seq_len 512` or `--seq_len 1024` tests only that
short training sequence length, not 256K-context behavior. Long-context
semantics, cache carries, and the truncated gradient horizon are documented in
`docs/09_LONG_CONTEXT_TRAINING.md`.

## Production 8-GPU run

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/check_hardware.py \
  --config core/config_1.5b_production_256k.json \
  --seq_len 262144 --chunk_len 4096 --bptt_chunks 4 --world_size 8

python -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/pretrain_1.5b_production.py \
  --config core/config_1.5b_production_256k.json \
  --steps 25000 --batch_size 1 --seq_len 262144 \
  --chunk_len 4096 --bptt_chunks 4 \
  --lr 3e-4 --save_interval 500 \
  --output_dir checkpoints/noesis_sigma_1.5b
```

Export and generate:

```bash
python scripts/export_checkpoint.py --checkpoint checkpoints/noesis_sigma_1.5b
python core/generate.py \
  --ckpt-path checkpoints/noesis_sigma_1.5b/exported \
  --input-file examples/example.txt \
  --max-new-tokens 65536 --prefill-chunk 8192
```

## Local validation

Run from the repository root:

```bash
PYTHONPYCACHEPREFIX=/tmp/noesis_pycache .venv/bin/python -m compileall -q core scripts examples tests
.venv/bin/python examples/validate_examples.py
.venv/bin/python -m pytest -q core/encoding/test_encoding.py --tb=short
.venv/bin/python -m pytest -q tests -k 'not test_verify_noesis_sigma_script_runs_end_to_end and not test_end_to_end_corpus_pretraining_checkpoint_and_generation and not test_ablation_benchmark_execution_and_results_table and not test_generate_module_vl_eos_and_cli_modes' --tb=short
.venv/bin/python -m pytest -q tests/test_pretrain_production_guards.py --tb=short
```

## Git pre-commit checks

After cloning the repository, install the development tools and Git hook once:

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pre-commit install --hook-type pre-commit --hook-type pre-push
.venv/bin/pre-commit run --all-files
```

The hook is intentionally fast: it checks syntax, structured files, merge
markers, private keys, large additions, whitespace, and critical Ruff errors.
It does not replace the full Pytest validation above; run that suite before a
release, Kaggle bundle upload, or significant training run.

Every normal `git push` also runs the non-training quality gate. It requires
source compilation, production-example validation, the full Pytest suite,
overall measured coverage of at least 80%, and `core/` coverage of at least
99%. The same gate runs in GitHub Actions. Do not use `git push --no-verify`.

## Kaggle preparation

1. Upload the newly built `noesis_kaggle_bundle.tar.gz` as a **new Kaggle
   Dataset version**, then attach that version to the notebook. Kaggle input
   datasets are read-only, so extract the archive to `/kaggle/working`.
2. Configure `HF_TOKEN` as a Kaggle Secret. Do not print the token.
3. Verify the code source before a quality run. The startup log must contain:

```text
[✓] Stream 3 (20%): GitHub Python (public Parquet, language-filtered)
```

If the log contains `fallback`, stop. Production mode deliberately fails
instead of silently replacing unavailable data with embedded samples.

## One-step real-data gate

This is a data and optimizer check, not a quality test:

```bash
cd /kaggle/working/noesis-runtime
NOESIS_CODE_DATASET=codeparrot/github-code \
NOESIS_CODE_DATA_FILES=hf://datasets/codeparrot/github-code/data/train-00000-of-01126.parquet \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/pretrain_1.5b_production.py \
  --config core/config_1.5b_256k.json \
  --steps 1 --batch_size 2 --seq_len 512 --grad_accum 16 \
  --lr 3e-4 --save_interval 1 --sample_interval 1000000 \
  --output_dir /kaggle/working/noesis_data_gate
```

Require all of the following before continuing:

- both T4 ranks initialize;
- FSDP `FULL_SHARD` wraps the model and fp32 AdamW is constructed;
- all five real streams are listed and no fallback is listed;
- loss and gradient values are finite;
- `latest.pt` is saved.

## Continuing a verified run

Use the same model configuration, data settings, batch size, sequence length,
and gradient accumulation. `--steps` is the final step number, not an
additional number of steps.

```bash
cd /kaggle/working/noesis-runtime
NOESIS_CODE_DATASET=codeparrot/github-code \
NOESIS_CODE_DATA_FILES=hf://datasets/codeparrot/github-code/data/train-00000-of-01126.parquet \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/pretrain_1.5b_production.py \
  --config core/config_1.5b_256k.json \
  --steps 3000 --batch_size 2 --seq_len 512 --grad_accum 16 \
  --lr 3e-4 --warmup 200 --save_interval 100 --sample_interval 1000000 \
  --resume /kaggle/working/noesis_data_gate/latest.pt \
  --output_dir /kaggle/working/noesis_data_gate
```

For an engineering pilot, compare loss, validation loss, data-source logs, and
checkpoint reload at steps 10, 100, 500, and 1,000. Ten steps verify that the
training system operates; they do not measure final model quality.
