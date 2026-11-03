# NOESIS-Σ Unit Tests: preflight, corpus builder, and trainer helper functions
# Path: tests/test_script_entry_points.py

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import check_hardware  # noqa: E402
import build_tokenized_corpus  # noqa: E402
from model import ModelArgs  # noqa: E402


TINY_CONFIG = dict(
    max_batch_size=1,
    max_seq_len=64,
    dtype="bf16",
    expert_dtype=None,
    vocab_size=256,
    dim=32,
    moe_inter_dim=32,
    n_layers=2,
    n_mtp_layers=0,
    n_heads=4,
    n_routed_experts=2,
    n_shared_experts=1,
    n_activated_experts=1,
    q_lora_rank=16,
    head_dim=8,
    rope_head_dim=4,
    o_groups=2,
    o_lora_rank=16,
    window_size=8,
    compress_ratios=[0, 0],
    kv_source_layers=[],
    index_source_layers=[],
    index_n_heads=2,
    index_head_dim=4,
    index_topk=2,
    hc_mult=1,
    osc_enabled=False,
    osc_rank=4,
    osc_icnn_hidden=8,
)


def load_pretrain_module():
    spec = importlib.util.spec_from_file_location(
        "noesis_pretrain_helpers", SCRIPTS / "pretrain_1.5b_production.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- check_hardware -----------------------------------------------------------


def test_hardware_report_and_memory_plan() -> None:
    report = check_hardware.hardware_report(1)
    assert "torch" in report and "gpus" in report
    if not torch.cuda.is_available():
        assert "cpu_count" in report

    args = ModelArgs(**TINY_CONFIG)
    caches = check_hardware.cache_bytes(args, seq_len=64, batch_size=1)
    assert caches["total"] == sum(
        caches[key]
        for key in ("window", "compressed_kv", "indexer", "partial_groups", "rope_tables")
    )
    assert check_hardware.activation_estimate(args, 1, 16, True) > 0
    assert check_hardware.activation_estimate(args, 1, 16, False) > check_hardware.activation_estimate(
        args, 1, 16, True
    )

    chunk, bptt, note = check_hardware.recommend(args, 1, 8192, 4, True, 16.0, 13.6)
    assert chunk == 8192 and bptt == 4 and note == ""
    chunk, bptt, note = check_hardware.recommend(args, 1, 8192, 4, True, 16.0, 1e-9)
    assert chunk >= 1024 and bptt == max(1, min(4, 32768 // chunk)) and note


def test_hardware_validate_reports_configuration_errors() -> None:
    args = ModelArgs(**TINY_CONFIG)
    assert "exceeds the config's max_seq_len" in check_hardware.validate(args, 128, 1, 1, 0)[0]
    errors = check_hardware.validate(args, 32, 9, 3, 8)
    assert any("world_size=9 exceeds" in error for error in errors)
    assert any("n_heads=4 is not divisible" in error for error in errors)
    assert any("o_groups=2 is not divisible" in error for error in errors)
    assert any("n_routed_experts=2 is not divisible" in error for error in errors)
    assert any("dspark_n_routed_experts" in error for error in errors)

    negative = ModelArgs(**{**TINY_CONFIG, "compress_ratios": [-1, 0]})
    assert "compress_ratios[0] is negative" in check_hardware.validate(negative, 32, 1, 1, 0)
    vision = ModelArgs(**{**TINY_CONFIG, "vision_n_layers": 1})
    assert "production profile disables vision" in check_hardware.validate(vision, 32, 1, 1, 0)[0]
    assert check_hardware.validate(args, 32, 1, 1, 0) == []


def test_check_hardware_main_json_and_text(tmp_path, monkeypatch, capsys) -> None:
    config_path = tmp_path / "tiny.json"
    config_path.write_text(json.dumps(TINY_CONFIG), encoding="utf-8")
    argv = [
        "check_hardware.py",
        "--config",
        str(config_path),
        "--seq_len",
        "32",
        "--chunk_len",
        "16",
        "--bptt_chunks",
        "2",
        "--world_size",
        "1",
        "--json",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    check_hardware.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["config"] == str(config_path)
    assert payload["parameters"] >= 0

    monkeypatch.setattr(sys, "argv", [arg for arg in argv if arg != "--json"])
    check_hardware.main()
    text = capsys.readouterr().out
    assert "NOESIS-Σ HARDWARE PREFLIGHT" in text
    assert "Memory plan per rank" in text

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_hardware.py",
            "--config",
            str(config_path),
            "--seq_len",
            "128",
            "--chunk_len",
            "16",
            "--world_size",
            "1",
        ],
    )
    with pytest.raises(SystemExit):
        check_hardware.main()
    assert "ERRORS:" in capsys.readouterr().out


# -- build_tokenized_corpus ---------------------------------------------------


def test_iter_text_files(tmp_path) -> None:
    path = tmp_path / "corpus.txt"
    path.write_text("first\n\nsecond\n", encoding="utf-8")
    assert list(build_tokenized_corpus.iter_text_files([str(path)], True)) == ["first", "second"]
    assert list(build_tokenized_corpus.iter_text_files([str(path)], False)) == ["first\n\nsecond\n"]
    with pytest.raises(FileNotFoundError):
        list(build_tokenized_corpus.iter_text_files([str(tmp_path / "missing.txt")], True))


def install_fake_datasets(monkeypatch, rows: dict) -> None:
    module = types.ModuleType("datasets")

    def load_dataset(dataset_name, *args, **kwargs):
        del args, kwargs
        return iter(rows.get(dataset_name, []))

    module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", module)


def test_iter_hf_blend_formats_documents(monkeypatch) -> None:
    install_fake_datasets(
        monkeypatch,
        {
            "HuggingFaceFW/fineweb-edu": [{"text": "edu"}],
            "HuggingFaceTB/cosmopedia-v2": [{"text": "cosmo"}],
            "parquet": [
                {"language": "Python", "code": "print(1)"},
                {"language": "Rust", "text": "skip"},
            ],
            "openai/gsm8k": [{"question": "q", "answer": "work #### 42"}],
            "glaiveai/glaive-function-calling-v2": [{"chat": "tool"}],
        },
    )
    stream = build_tokenized_corpus.iter_hf_blend(object())
    docs = [next(stream) for _ in range(5)]
    assert docs[0] == "edu"
    assert docs[1] == "cosmo"
    assert docs[2] == "```python\nprint(1)\n```"
    assert "Final Answer: 42" in docs[3]
    assert docs[4] == "tool"


def test_iter_hf_blend_world_size_striding_and_fallback(monkeypatch, capsys) -> None:
    install_fake_datasets(
        monkeypatch,
        {
            "HuggingFaceFW/fineweb-edu": [{"text": "a"}, {"text": "b"}],
            "HuggingFaceTB/cosmopedia-v2": [],
            "parquet": [],
            "openai/gsm8k": [],
            "glaiveai/glaive-function-calling-v2": [],
        },
    )
    stream = build_tokenized_corpus.iter_hf_blend(object(), rank=1, world_size=2)
    assert next(stream) == "b"
    capsys.readouterr()

    module = types.ModuleType("datasets")

    def unavailable(dataset_name, *args, **kwargs):
        del dataset_name, args, kwargs
        raise OSError("offline")

    module.load_dataset = unavailable
    monkeypatch.setitem(sys.modules, "datasets", module)
    with pytest.raises(RuntimeError, match="unavailable"):
        next(build_tokenized_corpus.iter_hf_blend(object()))

    stream = build_tokenized_corpus.iter_hf_blend(object(), allow_fallback=True)
    with pytest.raises(RuntimeError, match="no usable documents"):
        next(stream)


class FakeTokenizer:
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [len(word) + 1 for word in text.split()]


def test_build_tokenized_corpus_main(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("alpha beta\n\ngamma\n", encoding="utf-8")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=staticmethod(lambda path: FakeTokenizer())
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    out = tmp_path / "tokens"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_tokenized_corpus.py",
            "--text_files",
            str(corpus),
            "--line_is_document",
            "--out",
            str(out),
            "--tokens_per_shard",
            "4",
            "--max_tokens",
            "5",
            "--dtype",
            "uint16",
        ],
    )
    build_tokenized_corpus.main()
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["dtype"] == "uint16"
    assert sum(entry["tokens"] for entry in meta["shards"]) == 5

    monkeypatch.setattr(sys, "argv", ["build_tokenized_corpus.py"])
    with pytest.raises(SystemExit):
        build_tokenized_corpus.main()


# -- pretrain helpers ---------------------------------------------------------


def test_pretrain_batch_lr_and_checkpoint_helpers(tmp_path) -> None:
    module = load_pretrain_module()
    batch = module.build_batch(iter([1, 2, 3]), 2, 2, torch.device("cpu"))
    assert batch.tolist() == [[1, 2], [3, 0]]

    dry = module.build_dry_stream(10, seed=0)
    assert all(0 <= next(dry) < 10 for _ in range(20))

    assert module.lr_at(0, 100, 1e-3, 1e-5, 10) == 0.0
    assert module.lr_at(10, 100, 1e-3, 1e-5, 10) == pytest.approx(1e-3)
    assert module.lr_at(1000, 100, 1e-3, 1e-5, 10) == pytest.approx(1e-5, abs=1e-12)

    path = tmp_path / "ckpt.pt"
    module.save_checkpoint_atomic({"step": 1}, path)
    assert torch.load(path, weights_only=True) == {"step": 1}
    assert module._config_hash({"b": 1, "a": 2}) == module._config_hash({"a": 2, "b": 1})
    assert "torch" in module._rng_state()


def test_pretrain_parameter_helpers() -> None:
    module = load_pretrain_module()
    model = nn.Sequential(nn.Linear(3, 3))
    module.unify_parameter_dtype(model, torch.bfloat16)
    assert all(parameter.dtype == torch.bfloat16 for parameter in model.parameters())

    class MoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(4)])
            self.n_routed_experts = 4
            self.n_activated_experts = 2

    mixed = nn.Sequential(nn.Linear(3, 3), MoE())
    # Linear(3, 3) contributes 12 (weight + bias); 2 of the 4 routed experts are active.
    assert module.active_parameter_count(mixed) == 12 + 8

    quantized = nn.Module()
    quantized.weight = nn.Parameter(torch.zeros(2, 2, dtype=torch.float8_e4m3fn))
    with pytest.raises(RuntimeError, match="quantized"):
        module.unify_parameter_dtype(quantized, torch.bfloat16)

    integer = nn.Module()
    integer.weight = nn.Parameter(torch.zeros(2, dtype=torch.long), requires_grad=False)
    with pytest.raises(RuntimeError, match="non-floating"):
        module.unify_parameter_dtype(integer, torch.bfloat16)
