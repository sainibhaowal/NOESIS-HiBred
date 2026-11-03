# NOESIS-Σ Documentation Index

This directory contains technical documentation for the NOESIS-Σ source tree.
Current operational commands and verification requirements are maintained in
[`Model_Run.md`](Model_Run.md).

| Document | Purpose |
| :--- | :--- |
| [01: Architecture and System Design](01_ARCHITECTURE_AND_DESIGN.md) | Architecture, data flow, and configuration references for the implemented model. |
| [02: Mathematical Specification](02_MATHEMATICAL_SPECIFICATION_OSC.md) | Mathematical description of the operator-split block and verifier loss. |
| [03: Module Reference](03_MODULE_REFERENCE.md) | File, class, and function reference for `core/` and related scripts. |
| [04: Pretraining and Inference Guide](04_PRETRAINING_AND_INFERENCE_GUIDE.md) | Data preparation, training-loop, checkpoint, and inference examples. |
| [05: Testing and Verification](05_TESTING_AND_VERIFICATION_REPORT.md) | Current quality gate, test scope, and release checks. |
| [06: Controlled Ablation Study](06_ABLATION_STUDY_REPORT.md) | Development ablation harness, its scope, and interpretation limits. |
| [07: Research and Compute Planning](07_PUBLICATION_AND_COMPUTE_STRATEGY.md) | Evidence requirements and staged compute planning. |
| [08: Kaggle Training and Evaluation Blueprint](08_KAGGLE_1B_RESEARCH_BLUEPRINT.md) | Controlled Kaggle workflow and evaluation planning. |
| [Generated ablation output](ablation_results.md) | Development output from the small-corpus ablation harness. |
| [Model runbook](Model_Run.md) | Verified local and Kaggle procedures for the current 1.5B model. |

Documentation describes implemented behavior or explicitly labeled plans. Model
quality claims require measured results from a documented training and
evaluation run.
