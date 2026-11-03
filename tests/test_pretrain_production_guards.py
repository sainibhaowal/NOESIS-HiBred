"""Regression tests for production pretraining data-safety guards."""

from __future__ import annotations

import importlib.util
import itertools
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "pretrain_1.5b_production.py"


class TinyTokenizer:
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [7]


def load_pretrain_module():
    spec = importlib.util.spec_from_file_location("noesis_pretrain_production", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def successful_loader(calls: list[tuple[str, dict]]):
    def load_dataset(path: str, *args, **kwargs):
        calls.append((path, kwargs))
        if path == "HuggingFaceFW/fineweb-edu":
            return itertools.cycle([{"text": "Educational document."}])
        if path == "HuggingFaceTB/cosmopedia-v2":
            return itertools.cycle([{"text": "Textbook explanation."}])
        if path == "parquet":
            return itertools.cycle([{"language": "Python", "code": "print('ok')"}])
        if path == "openai/gsm8k":
            return itertools.cycle([{"question": "2 + 2", "answer": "#### 4"}])
        if path == "glaiveai/glaive-function-calling-v2":
            return itertools.cycle([{"chat": "User: ping\nAssistant: pong"}])
        raise AssertionError(f"unexpected dataset: {path}")

    return load_dataset


def test_default_production_code_stream_uses_public_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    module = load_pretrain_module()
    calls: list[tuple[str, dict]] = []
    monkeypatch.delenv("NOESIS_CODE_DATASET", raising=False)
    monkeypatch.delenv("NOESIS_CODE_DATA_FILES", raising=False)
    monkeypatch.setattr(datasets, "load_dataset", successful_loader(calls))

    stream = module.build_5corpus_stream(TinyTokenizer(), max_tokens=40)
    assert len(list(stream)) == 40

    parquet_calls = [kwargs for name, kwargs in calls if name == "parquet"]
    assert len(parquet_calls) == 1
    assert parquet_calls[0]["data_files"]["train"] == (
        "hf://datasets/codeparrot/github-code/data/train-00000-of-01126.parquet"
    )


def test_production_stream_fails_instead_of_using_embedded_code(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    module = load_pretrain_module()
    calls: list[tuple[str, dict]] = []
    real_loader = successful_loader(calls)

    def code_fails(path: str, *args, **kwargs):
        if path == "parquet":
            raise OSError("test network failure")
        return real_loader(path, *args, **kwargs)

    monkeypatch.setattr(datasets, "load_dataset", code_fails)
    stream = module.build_5corpus_stream(TinyTokenizer(), max_tokens=20)

    with pytest.raises(RuntimeError, match="Stream 3 Python code"):
        next(stream)


def test_256k_config_changes_only_the_context_limit() -> None:
    base = json.loads((ROOT / "core" / "config_1.5b.json").read_text())
    context_256k = json.loads((ROOT / "core" / "config_1.5b_256k.json").read_text())

    assert context_256k["max_seq_len"] == 256_000
    assert {
        key: value for key, value in context_256k.items() if key != "max_seq_len"
    } == {
        key: value for key, value in base.items() if key != "max_seq_len"
    }


def test_full_text_1_5b_profile_enables_text_features_but_not_vision() -> None:
    """The opt-in full profile must not accidentally re-enable the VL tower."""
    profile = json.loads((ROOT / "core" / "config_1.5b_full_text_256k.json").read_text())

    assert profile["vision_n_layers"] == 0
    assert len(profile["compress_ratios"]) == profile["n_layers"] + profile["n_mtp_layers"]
    assert profile["compress_ratios"][2] == 8
    assert profile["kv_source_layers"] == [2]
    assert profile["index_source_layers"] == [2]
    assert profile["engram_layer_ids"] == [3, 9, 15]
    assert len(profile["engram_num_embeddings"]) == len(profile["engram_layer_ids"])
    assert profile["dspark_block_size"] == 2
    assert profile["dspark_target_layer_ids"] == [9, 15]
