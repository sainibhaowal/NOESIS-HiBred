# NOESIS-Σ Dedicated Automated Ablation Benchmark & Verification Suite
# Path: tests/test_ablation_study_suite.py

from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CORE_DIR = PROJECT_ROOT / "core"

for p in (str(SCRIPTS_DIR), str(CORE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_ablation_study import (
    run_automated_ablation_study,
    get_ablation_model_args,
    compute_kv_cache_footprint_mb,
    load_shared_corpus,
    AblationMetrics,
)


def test_ablation_study_configurations_and_footprint(device: torch.device) -> None:
    """Verify that all 5 ablation model configurations build properly and produce valid theoretical memory calculations."""
    for var_id in range(1, 6):
        name, desc, args = get_ablation_model_args(var_id)
        assert name.startswith(str(var_id))
        assert len(desc) > 0
        kv_mb = compute_kv_cache_footprint_mb(args, batch_size=2, seq_len=32)
        assert kv_mb > 0.0

        if var_id in (3, 4, 5):
            # MLA compression should strictly reduce KV cache footprint compared to Baseline
            baseline_name, _, baseline_args = get_ablation_model_args(1)
            baseline_kv = compute_kv_cache_footprint_mb(baseline_args, batch_size=2, seq_len=32)
            assert kv_mb < baseline_kv


def test_ablation_benchmark_execution_and_results_table(
    device: torch.device,
    tokenizer_vocab: dict[str, int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """Runs the 5-stage ablation benchmark on CPU/GPU, verifies loss drop & attractor convergence, and prints publication table."""
    corpus_file = "examples/pretraining_corpus.txt"
    corpus_p = PROJECT_ROOT / corpus_file
    assert corpus_p.exists(), f"Pretraining corpus missing at {corpus_p}"

    # Run the 5-stage Iso-Compute ablation benchmark (15 steps for fast CI verification)
    results, md_table, latex_table = run_automated_ablation_study(
        corpus_file=corpus_file,
        steps=15,
        lr=1e-3,
        device=str(device),
        output_dir=str(tmp_path),
    )

    assert len(results) == 5, f"Expected 5 variants, got {len(results)}"

    # Validate each configuration's metrics
    baseline = results[0]
    osc_only = results[1]
    osc_mla = results[2]
    osc_mla_hc = results[3]
    full_noesis = results[4]

    # Verify Baseline
    assert baseline.variant_id == 1
    assert baseline.init_loss > 0.0
    assert baseline.final_loss < baseline.init_loss
    assert baseline.perplexity > 1.0
    assert baseline.tokens_per_sec > 0.0

    # Verify + OSC Only (Key Novelty)
    assert osc_only.variant_id == 2
    assert osc_only.final_loss < osc_only.init_loss
    assert isinstance(osc_only.energy_delta, float)
    assert osc_only.energy_delta < 0.0, "OSC attractor must achieve energy dissipation (delta < 0)"
    assert isinstance(osc_only.grad_phi_norm, float) and osc_only.grad_phi_norm > 0.0

    # Verify + OSC + MLA (KV Memory Reduction)
    assert osc_mla.variant_id == 3
    assert osc_mla.kv_cache_mb < baseline.kv_cache_mb, "MLA must compress KV-cache memory vs baseline"

    # Verify + OSC + MLA + HC (Sinkhorn Hyper-Connections)
    assert osc_mla_hc.variant_id == 4
    assert osc_mla_hc.final_loss < osc_mla_hc.init_loss

    # Verify Full NOESIS-Σ
    assert full_noesis.variant_id == 5
    assert full_noesis.final_loss < full_noesis.init_loss
    assert full_noesis.params_m > baseline.params_m

    # Ensure tables were properly generated
    assert "| Architecture Variant |" in md_table
    assert r"\begin{table*}[t]" in latex_table
    assert r"\end{table*}" in latex_table

    # Print output tables to test stdout
    print("\n" + "=" * 80)
    print("VERIFIED PUBLICATION MARKDOWN TABLE:")
    print("=" * 80)
    print(md_table)
    print("\n" + "=" * 80)
    print("VERIFIED PUBLICATION LATEX TABLE:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)
