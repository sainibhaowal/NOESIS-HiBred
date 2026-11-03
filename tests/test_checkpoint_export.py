# NOESIS-Σ Unit Tests: training checkpoint export
# Path: tests/test_checkpoint_export.py

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_model as load_safetensors

import generate
from model import ModelArgs, Transformer

ROOT = Path(__file__).resolve().parent.parent


def exporter_module():
    spec = importlib.util.spec_from_file_location(
        "noesis_export_checkpoint", ROOT / "scripts" / "export_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_config(**overrides) -> dict:
    cfg = dict(
        max_batch_size=1,
        max_seq_len=64,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=512,
        dim=64,
        moe_inter_dim=128,
        n_layers=3,
        n_mtp_layers=0,
        n_heads=4,
        n_routed_experts=4,
        n_shared_experts=1,
        n_activated_experts=2,
        q_lora_rank=32,
        head_dim=32,
        rope_head_dim=16,
        o_groups=2,
        o_lora_rank=32,
        window_size=16,
        compress_ratios=[0, 4, 0],
        kv_source_layers=[1],
        index_source_layers=[1],
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        hc_mult=2,
        engram_layer_ids=[],
        vision_n_layers=0,
        dspark_block_size=0,
        dspark_target_layer_ids=[],
        osc_enabled=True,
        osc_rank=16,
        osc_icnn_hidden=32,
        osc_steps=2,
    )
    cfg.update(overrides)
    return cfg


def build_model(cfg: dict, seed: int = 3) -> Transformer:
    torch.manual_seed(seed)
    return Transformer(ModelArgs(**cfg)).float().eval()


def write_fsdp_checkpoint(path: Path, cfg: dict, model: Transformer, prefix: str) -> None:
    torch.save(
        {
            "version": 2,
            "config": cfg,
            "model_state_dict": {
                prefix + name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            },
        },
        path,
    )


def test_export_strips_fsdp_prefix_and_round_trips(tmp_path: Path) -> None:
    cfg = tiny_config()
    model = build_model(cfg)
    checkpoint = tmp_path / "latest.pt"
    write_fsdp_checkpoint(checkpoint, cfg, model, "_fsdp_wrapped_module._orig_mod.")

    out_dir = tmp_path / "exported"
    exporter = exporter_module()
    exporter.export(checkpoint, out_dir, "fp32", None, tmp_path / "none", False)

    weights = out_dir / "model.safetensors"
    assert weights.exists() and (out_dir / "config.json").exists()
    assert json.loads((out_dir / "config.json").read_text()) == cfg

    restored = Transformer(ModelArgs(**cfg)).float().eval()
    load_safetensors(restored, str(weights))
    assert all("_fsdp_wrapped_module" not in name for name in restored.state_dict())

    torch.manual_seed(9)
    x = torch.randint(0, cfg["vocab_size"], (1, 12))
    with torch.no_grad():
        torch.testing.assert_close(restored.forward_chunk(x)[0], model.forward_chunk(x)[0])


def test_export_casts_weights_to_requested_dtype(tmp_path: Path) -> None:
    cfg = tiny_config()
    model = build_model(cfg)
    checkpoint = tmp_path / "latest.pt"
    write_fsdp_checkpoint(checkpoint, cfg, model, "")

    out_dir = tmp_path / "exported_bf16"
    exporter_module().export(checkpoint, out_dir, "bf16", None, tmp_path / "none", False)
    from safetensors.torch import load_file

    tensors = load_file(str(out_dir / "model.safetensors"))
    assert tensors["embed.weight"].dtype == torch.bfloat16
    # Integer buffers keep their dtype.
    assert all(t.dtype == torch.bfloat16 for t in tensors.values() if t.is_floating_point())


def test_export_rejects_flat_fsdp_parameters(tmp_path: Path) -> None:
    cfg = tiny_config()
    checkpoint = tmp_path / "latest.pt"
    torch.save(
        {"version": 2, "config": cfg, "model_state_dict": {"layers.0._flat_param": torch.zeros(4)}},
        checkpoint,
    )
    with pytest.raises(ValueError, match="flat parameters"):
        exporter_module().export(checkpoint, tmp_path / "out", "fp32", None, tmp_path / "none", False)


def test_export_rejects_incomplete_state_dict(tmp_path: Path) -> None:
    cfg = tiny_config()
    model = build_model(cfg)
    state = model.state_dict()
    missing = next(iter(state))
    payload = {
        "version": 2,
        "config": cfg,
        "model_state_dict": {
            name: tensor for name, tensor in state.items() if name != missing
        },
    }
    checkpoint = tmp_path / "latest.pt"
    torch.save(payload, checkpoint)

    with pytest.raises(ValueError, match="missing"):
        exporter_module().export(checkpoint, tmp_path / "out", "fp32", None, tmp_path / "none", False)

    exporter_module().export(
        checkpoint, tmp_path / "partial", "fp32", None, tmp_path / "none", True
    )
    assert (tmp_path / "partial" / "model.safetensors").exists()


def test_load_payload_and_clean_state_dict_validation(tmp_path: Path) -> None:
    exporter = exporter_module()
    cfg = tiny_config()
    model = build_model(cfg)
    run = tmp_path / "run"
    run.mkdir()
    write_fsdp_checkpoint(run / "latest.pt", cfg, model, "")
    assert exporter.load_payload(run)["config"] == cfg

    with pytest.raises(FileNotFoundError, match="not found"):
        exporter.load_payload(tmp_path / "missing")
    torch.save({"step": 1}, tmp_path / "no_model.pt")
    with pytest.raises(ValueError, match="no model_state_dict"):
        exporter.load_payload(tmp_path / "no_model.pt")

    with pytest.raises(ValueError, match="duplicate"):
        exporter.clean_state_dict(
            {"a": torch.zeros(1), "_fsdp_wrapped_module.a": torch.zeros(1)}
        )


def test_export_uses_an_external_config_override(tmp_path: Path) -> None:
    cfg = tiny_config()
    model = build_model(cfg)
    checkpoint = tmp_path / "latest.pt"
    torch.save({"version": 2, "model_state_dict": model.state_dict()}, checkpoint)

    exporter = exporter_module()
    with pytest.raises(ValueError, match="no embedded config"):
        exporter.export(checkpoint, tmp_path / "out", "fp32", None, tmp_path / "none", False)

    override = tmp_path / "override.json"
    override.write_text(json.dumps({**cfg, "temperature": 0.5}), encoding="utf-8")
    exporter.export(checkpoint, tmp_path / "out", "fp32", override, tmp_path / "none", False)
    assert json.loads((tmp_path / "out" / "config.json").read_text())["temperature"] == 0.5


def test_export_copies_the_tokenizer_when_loading_returns_none(
    tmp_path: Path, monkeypatch
) -> None:
    import sys
    import types

    cfg = tiny_config()
    model = build_model(cfg)
    checkpoint = tmp_path / "latest.pt"
    write_fsdp_checkpoint(checkpoint, cfg, model, "")

    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=staticmethod(lambda path: None)
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)

    out_dir = tmp_path / "exported"
    exporter_module().export(checkpoint, out_dir, "fp32", None, tokenizer_dir, False)
    assert (out_dir / "tokenizer" / "tokenizer.json").read_text() == "{}"


def test_export_main_cli_defaults_output_to_the_run_directory(tmp_path: Path, monkeypatch) -> None:
    import sys

    cfg = tiny_config()
    model = build_model(cfg)
    run = tmp_path / "run"
    run.mkdir()
    write_fsdp_checkpoint(run / "latest.pt", cfg, model, "")

    monkeypatch.setattr(
        sys,
        "argv",
        ["export_checkpoint.py", "--checkpoint", str(run), "--tokenizer", str(tmp_path / "none")],
    )
    exporter_module().main()
    assert (run / "exported" / "model.safetensors").is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_checkpoint.py",
            "--checkpoint",
            str(run / "latest.pt"),
            "--output-dir",
            str(tmp_path / "named"),
            "--dtype",
            "fp32",
            "--tokenizer",
            str(tmp_path / "none"),
        ],
    )
    exporter_module().main()
    assert (tmp_path / "named" / "config.json").is_file()


def test_generate_resolves_exported_weights(tmp_path: Path) -> None:
    unsharded = tmp_path / "model.safetensors"
    unsharded.write_bytes(b"")
    assert generate.resolve_weights_path(str(tmp_path)) == str(unsharded)

    sharded = tmp_path / "model0-mp1.safetensors"
    sharded.write_bytes(b"")
    assert generate.resolve_weights_path(str(tmp_path)) == str(sharded)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert generate.resolve_weights_path(str(empty)).endswith("model0-mp1.safetensors")
