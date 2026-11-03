# NOESIS-Σ Unit Tests: 1.5B configuration profiles
# Path: tests/test_configs.py

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model import ModelArgs

CORE = Path(__file__).resolve().parent.parent / "core"
PROFILES = (
    "config_1.5b.json",
    "config_1.5b_256k.json",
    "config_1.5b_production_256k.json",
    "config_1.5b_full_text_256k.json",
)


def load(name: str) -> dict:
    return json.loads((CORE / name).read_text())


@pytest.mark.parametrize("name", PROFILES)
def test_profile_loads_and_validates(name: str) -> None:
    args = ModelArgs(**load(name))
    args.validate_for_model()
    assert args.n_layers == 18
    assert args.dim == 1536
    assert args.n_heads == 12
    assert args.vision_n_layers == 0
    assert args.osc_enabled
    assert len(args.compress_ratios) == args.n_layers + args.n_mtp_layers


def test_production_profile_is_the_long_context_profile() -> None:
    cfg = load("config_1.5b_production_256k.json")
    assert cfg["max_seq_len"] == 262_144
    assert cfg["max_batch_size"] == 1
    assert cfg["index_query_stride"] >= 1

    # Every feature stays on at once; vision stays off.
    assert cfg["hc_mult"] >= 2
    assert cfg["osc_enabled"] is True
    assert cfg["engram_layer_ids"]
    assert cfg["kv_source_layers"] and cfg["index_source_layers"]
    assert any(ratio > 0 for ratio in cfg["compress_ratios"])
    assert cfg["dspark_block_size"] > 0 and cfg["dspark_target_layer_ids"]
    assert cfg["vision_n_layers"] == 0


def test_production_profile_differs_only_in_runtime_knobs() -> None:
    base = load("config_1.5b.json")
    production = load("config_1.5b_production_256k.json")
    runtime_keys = {"max_seq_len", "max_batch_size", "index_query_stride", "index_query_chunk"}
    assert {
        key: value for key, value in base.items() if key not in runtime_keys
    } == {
        key: value for key, value in production.items() if key not in runtime_keys
    }


def test_context_profiles_share_the_architecture() -> None:
    base = load("config_1.5b.json")
    context = load("config_1.5b_256k.json")
    assert context["max_seq_len"] == 256_000
    assert {k: v for k, v in context.items() if k != "max_seq_len"} == {
        k: v for k, v in base.items() if k != "max_seq_len"
    }
