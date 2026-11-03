# NOESIS-Σ: Controlled Ablation Study Report

## Status and scope

This document describes the repository's five-variant ablation harness. The
recorded values use a small shared example corpus and 50 optimization steps.
They validate harness behavior only. They are not evidence of downstream model
quality, production performance, architecture superiority, or publication
readiness.

Use the results to confirm that each configuration builds, trains, emits the
expected metrics, and writes a result table. Use held-out data, multiple seeds,
and standard benchmarks for research conclusions.

## Experimental design

The harness in [`scripts/run_ablation_study.py`](../scripts/run_ablation_study.py)
constructs the following variants from [`ModelArgs`](../core/model.py):

| Stage | Variant | Controlled change |
| :---: | :--- | :--- |
| 1 | Baseline | Dense feed-forward layers, `hc_mult=1`, no OSC, no routed experts. |
| 2 | OSC | Adds the operator-split block while retaining the baseline attention and feed-forward configuration. |
| 3 | OSC + MLA | Adds compressed latent attention. |
| 4 | OSC + MLA + HC | Adds Sinkhorn Hyper-Connections with `hc_mult=4`. |
| 5 | Full configuration | Adds routed MoE, Engram memory, and DSpark settings used by the harness. |

The harness uses a common corpus, optimizer setting, and step count for all
variants. A single small-corpus run does not provide a confidence interval or
control for seed variation.

## Recorded exploratory output

The following output was generated from
[`examples/pretraining_corpus.txt`](../examples/pretraining_corpus.txt) with
50 steps and AdamW at `lr=0.001`.

| Variant | Parameters | Initial loss | Final loss | Perplexity | KV cache | Tokens/s | Gradient residual | Energy delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Baseline | 2.84M | 7.049 | 2.861 | 17.47 | 0.125 MB | 117.1 | N/A | N/A |
| OSC | 3.09M | 7.172 | 1.851 | 6.36 | 0.125 MB | 87.6 | 14.927 | -20.187 |
| OSC + MLA | 3.27M | 7.147 | 2.035 | 7.65 | 0.098 MB | 85.7 | 16.555 | -21.132 |
| OSC + MLA + HC | 3.36M | 7.070 | 2.109 | 8.24 | 0.098 MB | 92.5 | 15.094 | -25.197 |
| Full configuration | 14.66M | 7.127 | 1.351 | 3.86 | 0.098 MB | 76.1 | 14.153 | -18.524 |

## Interpretation limits

- A lower training loss on this small corpus does not establish generalization.
- An energy delta is an internal training metric. It does not measure reasoning
  ability or safety.
- The reported KV-cache footprint is configuration-specific and must be
  remeasured for each context length, batch size, dtype, and device.
- Throughput depends on hardware, software versions, and runtime settings.
- Comparative claims require matched token budgets, fixed evaluation data,
  multiple random seeds, and reported uncertainty.

## Required evidence for a research result

1. Define a frozen training corpus and held-out validation split.
2. Run each variant with the same token budget and at least three random seeds.
3. Report mean and standard deviation for validation loss, benchmark score,
   memory, and throughput.
4. Publish the exact configurations, dataset revisions, software versions, and
   evaluation prompts.
5. Compare against external baselines under a documented compute budget.

## Commands

Run the harness:

```bash
.venv/bin/python scripts/run_ablation_study.py \
  --steps 50 \
  --corpus examples/pretraining_corpus.txt
```

Run its focused tests:

```bash
.venv/bin/python -m pytest -q tests/test_ablation_study_suite.py --tb=short
```

The generated table is written to
[`docs/ablation_results.md`](ablation_results.md). Review the run metadata
before using any generated values outside development validation.
