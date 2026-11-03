# NOESIS-Σ Unit Tests: long-context generation budget, chunked prefill, stop tokens
# Path: tests/test_generate_long_context.py

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import generate
from model import ModelArgs, Transformer


def generation_args(**overrides) -> ModelArgs:
    cfg = dict(
        max_batch_size=2,
        max_seq_len=512,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=512,
        dim=64,
        moe_inter_dim=128,
        n_layers=4,
        n_mtp_layers=1,
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
        compress_ratios=(0, 4, 4, 0, 0),
        kv_source_layers=(1,),
        index_source_layers=(1,),
        index_n_heads=4,
        index_head_dim=32,
        index_topk=8,
        hc_mult=2,
        engram_layer_ids=(),
        vision_n_layers=0,
        dspark_block_size=4,
        dspark_target_layer_ids=(1, 2),
        dspark_markov_rank=16,
        dspark_noise_token_id=0,
        osc_enabled=True,
        osc_rank=16,
        osc_icnn_hidden=32,
        osc_steps=2,
        temperature=0.0,  # greedy, so chunked and full prefill must agree exactly
    )
    cfg.update(overrides)
    return ModelArgs(**cfg)


def greedy_model(**overrides) -> Transformer:
    torch.manual_seed(5)
    model = Transformer(generation_args(**overrides)).float().eval()
    model.temperature = 0.0
    return model


def test_budget_is_enforced_before_allocating() -> None:
    model = greedy_model(max_seq_len=64, max_batch_size=1)
    with pytest.raises(ValueError, match="leaves no room"):
        generate.generate(model, [[1, 2, 3]], max_new_tokens=64, eos_id=None)
    with pytest.raises(ValueError, match="prompt budget"):
        generate.generate(model, [[1] * 40], max_new_tokens=30, eos_id=None)


def test_truncate_prompt_fits_the_budget() -> None:
    model = greedy_model(max_seq_len=64, max_batch_size=1)
    prompt = list(range(1, 51))
    out = generate.generate(
        model, [prompt], max_new_tokens=20, eos_id=None, truncate_prompt=True
    )
    assert len(out) == 1 and len(out[0]) <= 20

    with pytest.raises(ValueError, match="truncate"):
        generate.generate(model, [prompt], max_new_tokens=20, eos_id=None)


def test_chunked_prefill_matches_full_prefill() -> None:
    model = greedy_model()
    prompt = list(range(1, 18))
    reference = generate.generate(model, [prompt], max_new_tokens=12, eos_id=None, prefill_chunk=64)
    chunked = generate.generate(model, [prompt], max_new_tokens=12, eos_id=None, prefill_chunk=4)
    single = generate.generate(model, [prompt], max_new_tokens=12, eos_id=None, prefill_chunk=1)
    assert reference == chunked == single


def test_unequal_prompt_batch_decodes_like_individual_runs() -> None:
    model = greedy_model()
    prompts = [list(range(1, 6)), list(range(10, 19))]
    batch = generate.generate(model, prompts, max_new_tokens=8, eos_id=None, prefill_chunk=3)
    individual = [
        generate.generate(model, [prompt], max_new_tokens=8, eos_id=None, prefill_chunk=3)[0]
        for prompt in prompts
    ]
    assert batch == individual


def test_stop_token_ids_truncate_each_completion() -> None:
    model = greedy_model()
    calls = {"n": 0}

    def fake_forward(*_args, **_kwargs):
        calls["n"] += 1
        token = 55 if calls["n"] == 1 else 99
        return torch.tensor([token]), None, None

    with patch.object(model, "forward", side_effect=fake_forward):
        out = generate.generate(model, [[1, 2, 3]], max_new_tokens=6, eos_id=None, stop_token_ids=[99])
    assert out == [[55]]

    calls["n"] = 0
    with patch.object(model, "forward", side_effect=fake_forward):
        out = generate.generate(model, [[1, 2, 3]], max_new_tokens=6, eos_id=None)
    assert out == [[55, 99, 99, 99, 99, 99]]


def test_long_decode_stays_inside_the_budget() -> None:
    """A prompt plus a long output uses the configured context instead of the old 64K cap."""
    model = greedy_model(max_seq_len=1024, max_batch_size=1)
    prompt = list(range(1, 33))
    out = generate.generate(model, [prompt], max_new_tokens=256, eos_id=None, prefill_chunk=16)
    assert len(out[0]) <= 256
    assert len(prompt) + 256 <= model.max_seq_len


def test_cache_budget_report(capsys) -> None:
    model = greedy_model(max_seq_len=256, max_batch_size=1)
    generate.report_cache_budget(model, max_new_tokens=128)
    text = capsys.readouterr().out
    assert "attention/MTP caches" in text
    assert "128 prompt tokens" in text

    generate.report_cache_budget(model)
    generate.report_cache_budget(model, max_new_tokens=512)
    text = capsys.readouterr().out
    assert "attention/MTP caches" in text
    assert "every request will fail" in text


def test_generate_rejects_invalid_arguments() -> None:
    model = greedy_model()
    with pytest.raises(ValueError, match="non-negative"):
        generate.generate(model, [[1, 2]], max_new_tokens=-1, eos_id=None)
    with pytest.raises(ValueError, match="prefill_chunk must be positive"):
        generate.generate(model, [[1, 2]], max_new_tokens=2, eos_id=None, prefill_chunk=0)
    with pytest.raises(ValueError, match="must not be empty"):
        generate.generate(model, [], max_new_tokens=2, eos_id=None)
    with pytest.raises(ValueError, match="at least one token"):
        generate.generate(model, [[]], max_new_tokens=2, eos_id=None)
    with pytest.raises(ValueError, match="exceeds max_batch_size"):
        generate.generate(model, [[1], [2], [3]], max_new_tokens=2, eos_id=None)


def test_zero_new_tokens_returns_empty_completions() -> None:
    model = greedy_model()
    assert generate.generate(model, [[1, 2], [3]], max_new_tokens=0, eos_id=None) == [[], []]


def test_truncation_carries_token_types() -> None:
    model = greedy_model(max_seq_len=64, max_batch_size=1)
    prompt = list(range(1, 51))
    types = [0] * len(prompt)
    tokens, truncated_types = generate._truncate_prompts(
        [prompt], [types], budget=20
    )
    assert len(tokens[0]) == 20
    assert truncated_types is not None and len(truncated_types[0]) == 20


def test_stop_strings_trim_at_the_earliest_match() -> None:
    assert generate._trim_stop_strings("abcdef", ["cd", "zz"]) == "ab"
    assert generate._trim_stop_strings("abcdef", ["zz"]) == "abcdef"
