# NOESIS-Σ: Testing and Verification

## Current verification source

Use [`scripts/quality_gate.py`](../scripts/quality_gate.py) for the current
repository verification result. The gate runs source compilation, example
validation, the full test suite, and coverage thresholds. It is run by the
pre-push hook and by GitHub Actions.

The result is valid only for the exact commit, dependency set, and platform on
which it was executed. It verifies software behavior covered by the tests; it
does not establish trained-model quality, benchmark performance, or production
fitness.

## Required local command

```bash
.venv/bin/python scripts/quality_gate.py
```

The current gate requires:

| Check | Requirement |
| :--- | :--- |
| Python compilation | `core`, `scripts`, `examples`, and `tests` compile successfully. |
| Example validation | Harmony cases, production evaluation cases, and image assets validate. |
| Test suite | All discovered tests pass. |
| Measured `core` coverage | At least 99%. |
| Measured `core` and `scripts` coverage | At least 80%. |

Coverage thresholds are guards against regression, not evidence that every
execution path, hardware backend, or remote dataset condition has been tested.

## Test categories

| Area | Primary locations | Scope |
| :--- | :--- | :--- |
| Prompt encoding | `core/encoding/test_encoding.py`, `tests/test_encoding_suite.py` | Message formatting, tool-call parsing, reasoning effort, multimodal spans, and golden fixtures. |
| Model and kernels | `tests/test_backbone_and_speculative.py`, `tests/test_kernels_and_engram.py` | Attention, MoE, rotary embeddings, sparse kernels, Engram memory, and conversion helpers. |
| OSC and loss | `tests/test_osc_and_verifier.py` | Analytical gradient agreement, causal controller behavior, energy metrics, and loss components. |
| Vision | `tests/test_vision_and_multimodal.py` | Image processing, vision modules, and multimodal integration. |
| CLI and integration | `tests/test_cli_pipelines_and_edge_cases.py` | Generation, conversion, verification command behavior, and edge cases. |
| Production trainer guards | `tests/test_pretrain_production_guards.py` | Strict data-source behavior and 256K configuration checks. |
| Ablation harness | `tests/test_ablation_study_suite.py` | Variant construction, metrics, and generated result-table structure. |

## Additional release checks

Before a model release or a significant training run, perform these checks in
addition to the quality gate:

1. Run the one-step real-data gate in [`Model_Run.md`](Model_Run.md).
2. Reload the produced checkpoint in a fresh process.
3. Evaluate on a held-out validation set that was excluded from training.
4. Record the model configuration, dataset revision, command line, package
   versions, hardware, and random seed.
5. Review data licenses, model licenses, and known limitations.
