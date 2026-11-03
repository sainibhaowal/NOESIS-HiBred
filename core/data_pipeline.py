"""Memory-mapped, resumable token packing for production pretraining.

The training corpus is stored once as ``.bin`` shards plus a ``metadata.json`` index. Training then
reads fixed-size token batches by memory map, so a run never depends on a live HF stream and can
resume at the exact ``(shard, offset)`` it stopped at.

Shard format
------------
``metadata.json``::

    {"dtype": "uint32", "eos_token_id": 2,
     "shards": [{"path": "shard-00000.bin", "tokens": 1000000000}, ...]}

Each ``.bin`` file is a flat little-endian array of token ids. Documents are separated by one EOS
token, so a packed batch needs no padding: every token is a real training token.

This module deliberately has no torch dependency on the read path. ``PackedTokenSampler`` returns
plain numpy arrays; the trainer converts to device tensors.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


class ShardCorpusError(RuntimeError):
    """Raised when a shard index is missing, malformed, or inconsistent."""


@dataclass
class BatchState:
    """Exact position of the sampler. Saving this makes data order resumable."""

    shard: int = 0
    offset: int = 0
    epoch: int = 0

    def to_dict(self) -> dict:
        return {"shard": self.shard, "offset": self.offset, "epoch": self.epoch}

    @classmethod
    def from_dict(cls, value: dict | None) -> "BatchState":
        if not value:
            return cls()
        return cls(int(value.get("shard", 0)), int(value.get("offset", 0)), int(value.get("epoch", 0)))


class PackedTokenSampler:
    """Deterministic, rank-strided batches over a set of token shards.

    Each rank starts at a different shard and walks the corpus in the same global order, which keeps
    ranks from training on the same tokens without needing a synchronised shuffle. ``next_batch``
    advances the saved cursor across shard and epoch boundaries.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        rank: int = 0,
        world_size: int = 1,
        eos_token_id: int = 2,
        seed: int = 0,
    ) -> None:
        self.root = Path(shard_dir)
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise ShardCorpusError(
                f"{meta_path} not found. Build it once with scripts/build_tokenized_corpus.py; "
                "live streaming is intentionally not used for production runs."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.dtype = np.dtype(meta.get("dtype", "uint32"))
        self.eos_token_id = int(meta.get("eos_token_id", eos_token_id))
        entries = meta.get("shards") or []
        if not entries:
            raise ShardCorpusError(f"{meta_path} lists no shards")
        self.paths: list[Path] = []
        self.lengths: list[int] = []
        for entry in entries:
            path = self.root / entry["path"]
            tokens = int(entry["tokens"])
            if not path.exists():
                raise ShardCorpusError(f"shard {path} is listed but missing")
            if tokens <= 0:
                raise ShardCorpusError(f"shard {path} declares non-positive token count")
            self.paths.append(path)
            self.lengths.append(tokens)
        if int(world_size) > len(self.paths):
            raise ShardCorpusError(
                f"world_size={world_size} exceeds the {len(self.paths)} available shards; "
                "build more shards so every rank can start on a distinct one"
            )
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.seed = int(seed)
        self._maps: dict[int, np.memmap] = {}
        self.cursor = BatchState(shard=self.rank % len(self.paths), offset=0, epoch=0)

    # -- state -----------------------------------------------------------------
    def state_dict(self) -> dict:
        return self.cursor.to_dict()

    def load_state_dict(self, value: dict | None) -> None:
        self.cursor = BatchState.from_dict(value)
        if self.cursor.shard >= len(self.paths):
            raise ShardCorpusError(f"resume shard {self.cursor.shard} is out of range")
        if self.cursor.offset > self.lengths[self.cursor.shard]:
            raise ShardCorpusError(
                f"resume offset {self.cursor.offset} exceeds shard length "
                f"{self.lengths[self.cursor.shard]}"
            )

    # -- reading ---------------------------------------------------------------
    def _map(self, index: int) -> np.memmap:
        mapped = self._maps.get(index)
        if mapped is None:
            mapped = np.memmap(self.paths[index], dtype=self.dtype, mode="r")
            if mapped.size != self.lengths[index]:
                raise ShardCorpusError(
                    f"{self.paths[index]} has {mapped.size} tokens but metadata says "
                    f"{self.lengths[index]}"
                )
            self._maps[index] = mapped
        return mapped

    def _read(self, count: int) -> np.ndarray:
        """Read ``count`` tokens sequentially from the cursor, crossing shards/epochs as needed."""
        out = np.empty(count, dtype=np.int64)
        filled = 0
        while filled < count:
            index = self.cursor.shard % len(self.paths)
            shard = self._map(index)
            start = self.cursor.offset
            take = min(count - filled, shard.size - start)
            if take <= 0:
                # End of shard: move to the next one, wrapping into a new epoch. Skipping
                # ``world_size`` shards between reads keeps ranks disjoint.
                self._advance_shard()
                continue
            out[filled : filled + take] = shard[start : start + take]
            filled += take
            self.cursor.offset = start + take
            if self.cursor.offset >= shard.size:
                self._advance_shard()
        return out

    def _advance_shard(self) -> None:
        next_shard = (self.cursor.shard + self.world_size) % len(self.paths)
        if next_shard <= self.cursor.shard:
            # Wrapped past the last shard: one full corpus pass is complete.
            self.cursor.epoch += 1
        self.cursor.shard = next_shard
        self.cursor.offset = 0

    def next_batch(self, batch_size: int, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(input_ids, loss_mask, segment_ids)`` with shapes ``[B, seq_len]``."""
        if batch_size <= 0 or seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        flat = self._read(batch_size * seq_len)
        input_ids = flat.reshape(batch_size, seq_len)
        loss_mask = np.ones_like(input_ids, dtype=bool)
        # Segment ids increment at every EOS: 0 for the first document, 1 for the next, ...
        is_eos = (input_ids == self.eos_token_id).astype(np.int32)
        segment_ids = np.cumsum(is_eos, axis=1, dtype=np.int32)
        segment_ids -= segment_ids[:, :1]
        return input_ids, loss_mask, segment_ids


def iter_shard_tokens(
    texts: Iterable[str],
    encode,
    eos_token_id: int,
) -> Iterable[np.ndarray]:
    """Yield one token array per document, with an EOS appended.

    ``encode`` is any callable ``str -> list[int]`` (typically the tokenizer). Kept separate from
    the file writer so the tokenizer dependency stays out of the training path.
    """
    for text in texts:
        if not text or not text.strip():
            continue
        ids = list(encode(text))
        if not ids:
            continue
        ids.append(eos_token_id)
        yield np.asarray(ids, dtype=np.int64)


def write_token_shards(
    out_dir: str | Path,
    document_tokens: Iterable[Iterable[int] | np.ndarray],
    *,
    tokens_per_shard: int = 250_000_000,
    eos_token_id: int = 2,
    dtype: str = "uint32",
) -> dict:
    """Write flat token shards and the metadata index. Returns the metadata dict."""
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    np_dtype = np.dtype(dtype)
    entries: list[dict] = []
    shard_index = 0
    buffer = np.empty(tokens_per_shard, dtype=np_dtype)
    used = 0

    def flush() -> None:
        nonlocal shard_index, used
        if used == 0:
            return
        name = f"shard-{shard_index:05d}.bin"
        buffer[:used].tofile(root / name)
        entries.append({"path": name, "tokens": int(used)})
        shard_index += 1
        used = 0

    for doc in document_tokens:
        tokens = np.asarray(doc, dtype=np_dtype)
        position = 0
        while position < tokens.size:
            take = min(tokens_per_shard - used, tokens.size - position)
            buffer[used : used + take] = tokens[position : position + take]
            used += take
            position += take
            if used == tokens_per_shard:
                flush()
    flush()

    meta = {"dtype": np_dtype.name, "eos_token_id": int(eos_token_id), "shards": entries}
    (root / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def batches_per_epoch(shard_dir: str | Path, batch_size: int, seq_len: int) -> int:
    """Approximate number of full batches in one corpus pass (for scheduler planning)."""
    meta = json.loads((Path(shard_dir) / "metadata.json").read_text(encoding="utf-8"))
    total = sum(int(entry["tokens"]) for entry in meta.get("shards", []))
    return max(1, math.ceil(total / (batch_size * seq_len)))
