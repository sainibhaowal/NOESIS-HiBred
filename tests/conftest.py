# NOESIS-Σ Test Suite Configuration & Shared Fixtures
# Path: tests/conftest.py

from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT_DIR / "core"
ENCODING_DIR = CORE_DIR / "encoding"

for path_str in (str(CORE_DIR), str(ENCODING_DIR)):
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture(autouse=True)
def reset_model_globals():
    """Every test starts from a logically unsharded model and the default fp32 dtype.

    ``core/model.py`` stores the logical model-parallel layout in module globals; tests that
    exercise ``configure_model_parallel`` must not leak that state into later tests.
    """
    import model as model_module

    model_module.configure_model_parallel(1, 0)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")
    yield
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cpu")


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def config_10m_dict() -> dict:
    test_cfg = ROOT_DIR / "tests" / "fixtures" / "config_test.json"
    if test_cfg.exists():
        with open(test_cfg, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "dim": 256,
        "n_layers": 6,
        "n_heads": 8,
        "n_routed_experts": 4,
        "n_activated_experts": 2,
        "vocab_size": 32000,
        "max_seq_len": 512,
        "hc_mult": 2,
        "osc_enabled": True,
        "engram_enabled": True,
    }


@pytest.fixture(scope="session")
def tokenizer_vocab() -> dict[str, int]:
    with open(CORE_DIR / "tokenizer.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("model", {}).get("vocab", {})
