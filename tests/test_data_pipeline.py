# NOESIS-Σ Unit Tests: memory-mapped token shards and resumable sampling
# Path: tests/test_data_pipeline.py

from __future__ import annotations

import json

import numpy as np
import pytest

from data_pipeline import (
    BatchState,
    PackedTokenSampler,
    ShardCorpusError,
    batches_per_epoch,
    iter_shard_tokens,
    write_token_shards,
)


def write_corpus(tmp_path, docs, tokens_per_shard=4):
    return write_token_shards(
        tmp_path,
        docs,
        tokens_per_shard=tokens_per_shard,
        eos_token_id=9,
    )


def test_write_token_shards_splits_and_round_trips(tmp_path) -> None:
    meta = write_corpus(tmp_path, [[1, 2, 9, 3, 9], [4, 9]], tokens_per_shard=4)
    assert [entry["tokens"] for entry in meta["shards"]] == [4, 3]
    assert meta["dtype"] == "uint32"
    assert meta["eos_token_id"] == 9
    assert (tmp_path / "metadata.json").is_file()

    sampler = PackedTokenSampler(tmp_path)
    ids, mask, segments = sampler.next_batch(1, 7)
    assert ids.tolist() == [[1, 2, 9, 3, 9, 4, 9]]
    assert mask.all()
    assert segments.tolist() == [[0, 0, 1, 1, 2, 2, 3]]
    assert sampler.cursor.epoch == 1
    assert sampler.cursor.shard == 0


def test_sampler_shape_and_rank_striding(tmp_path) -> None:
    write_corpus(tmp_path, [[1, 2, 9, 3, 9], [4, 9]], tokens_per_shard=4)
    rank_one = PackedTokenSampler(tmp_path, rank=1, world_size=2)
    assert rank_one.cursor.shard == 1
    ids, _, _ = rank_one.next_batch(1, 2)
    assert ids.tolist() == [[9, 4]]


def test_resume_at_shard_boundary_advances(tmp_path) -> None:
    write_corpus(tmp_path, [[1, 2, 3, 4], [5, 6, 7, 8]], tokens_per_shard=2)
    sampler = PackedTokenSampler(tmp_path)
    sampler.load_state_dict({"shard": 0, "offset": 2, "epoch": 0})
    ids, _, _ = sampler.next_batch(1, 2)
    assert ids.tolist() == [[3, 4]]


def test_state_dict_round_trip_and_validation(tmp_path) -> None:
    write_corpus(tmp_path, [[1, 2, 3, 4]], tokens_per_shard=4)
    sampler = PackedTokenSampler(tmp_path)
    sampler.next_batch(1, 2)
    saved = sampler.state_dict()
    assert saved == {"shard": 0, "offset": 2, "epoch": 0}

    fresh = PackedTokenSampler(tmp_path)
    fresh.load_state_dict(saved)
    assert fresh.state_dict() == saved

    with pytest.raises(ShardCorpusError, match="out of range"):
        fresh.load_state_dict({"shard": 5, "offset": 0, "epoch": 0})
    with pytest.raises(ShardCorpusError, match="exceeds shard length"):
        fresh.load_state_dict({"shard": 0, "offset": 5, "epoch": 0})


def test_batch_state_from_dict_defaults() -> None:
    assert BatchState.from_dict(None) == BatchState()
    assert BatchState.from_dict({}) == BatchState()
    assert BatchState.from_dict({"shard": "1", "offset": "2", "epoch": "3"}) == BatchState(1, 2, 3)


def test_next_batch_rejects_invalid_sizes(tmp_path) -> None:
    write_corpus(tmp_path, [[1, 2]], tokens_per_shard=2)
    sampler = PackedTokenSampler(tmp_path)
    with pytest.raises(ValueError, match="must be positive"):
        sampler.next_batch(0, 2)
    with pytest.raises(ValueError, match="must be positive"):
        sampler.next_batch(1, 0)


def test_missing_metadata_raises(tmp_path) -> None:
    with pytest.raises(ShardCorpusError, match="not found"):
        PackedTokenSampler(tmp_path)


def test_metadata_without_shards_raises(tmp_path) -> None:
    (tmp_path / "metadata.json").write_text(json.dumps({"shards": []}), encoding="utf-8")
    with pytest.raises(ShardCorpusError, match="lists no shards"):
        PackedTokenSampler(tmp_path)


def test_listed_shard_missing_raises(tmp_path) -> None:
    (tmp_path / "metadata.json").write_text(
        json.dumps({"shards": [{"path": "missing.bin", "tokens": 4}]}), encoding="utf-8"
    )
    with pytest.raises(ShardCorpusError, match="listed but missing"):
        PackedTokenSampler(tmp_path)


def test_non_positive_shard_length_raises(tmp_path) -> None:
    np.arange(4, dtype=np.uint32).tofile(tmp_path / "shard-00000.bin")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"shards": [{"path": "shard-00000.bin", "tokens": 0}]}), encoding="utf-8"
    )
    with pytest.raises(ShardCorpusError, match="non-positive"):
        PackedTokenSampler(tmp_path)


def test_world_size_larger_than_shard_count_raises(tmp_path) -> None:
    write_corpus(tmp_path, [[1, 2]], tokens_per_shard=2)
    with pytest.raises(ShardCorpusError, match="exceeds the 1 available shards"):
        PackedTokenSampler(tmp_path, world_size=2)


def test_shard_size_mismatch_raises_on_read(tmp_path) -> None:
    np.arange(3, dtype=np.uint32).tofile(tmp_path / "shard-00000.bin")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"shards": [{"path": "shard-00000.bin", "tokens": 5}]}), encoding="utf-8"
    )
    sampler = PackedTokenSampler(tmp_path)
    with pytest.raises(ShardCorpusError, match="metadata says"):
        sampler.next_batch(1, 2)


def test_iter_shard_tokens_skips_blank_and_empty_encodings() -> None:
    vocab = {"alpha": [1, 2], "beta": [3], "empty": []}
    docs = ["", "   ", "alpha", "empty", "beta"]
    out = [array.tolist() for array in iter_shard_tokens(docs, vocab.__getitem__, eos_token_id=9)]
    assert out == [[1, 2, 9], [3, 9]]


def test_write_token_shards_handles_empty_input(tmp_path) -> None:
    meta = write_token_shards(tmp_path, [], tokens_per_shard=8, eos_token_id=7)
    assert meta["shards"] == []
    assert meta["eos_token_id"] == 7


def test_batches_per_epoch(tmp_path) -> None:
    write_corpus(tmp_path, [[1, 2, 3, 4, 5]], tokens_per_shard=5)
    assert batches_per_epoch(tmp_path, batch_size=1, seq_len=2) == 3
    assert batches_per_epoch(tmp_path, batch_size=8, seq_len=8) == 1
