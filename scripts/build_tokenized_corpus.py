#!/usr/bin/env python3
"""Tokenize a corpus once into memory-mapped ``.bin`` shards for production pretraining.

The trainer reads shards through ``core/data_pipeline.PackedTokenSampler``; it never tokenizes or
streams Hugging Face datasets during a training run. This script is the only place that touches the
tokenizer and the raw corpora, so it can be run on a CPU machine ahead of the GPU job.

Examples
--------
Tokens from local text files (one document per line, or ``--line_is_document``)::

    python scripts/build_tokenized_corpus.py --out data/tokens --text_files corpus/*.txt

Streaming Hugging Face datasets (same five-corpus blend as the legacy trainer)::

    python scripts/build_tokenized_corpus.py --out data/tokens --hf_blend

Use ``--max_tokens`` to bound a build. Shards are written with an EOS after every document.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (PROJECT_ROOT / "core", PROJECT_ROOT / "core" / "encoding"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pipeline import iter_shard_tokens, write_token_shards  # noqa: E402


def iter_text_files(paths: list[str], line_is_document: bool):
    for name in paths:
        path = Path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8", errors="ignore")
        if line_is_document:
            for line in text.splitlines():
                if line.strip():
                    yield line
        else:
            yield text


def iter_hf_blend(tokenizer, rank: int = 0, world_size: int = 1, allow_fallback: bool = False):
    """The documented 40/20/20/10/10 document blend, streamed once.

    This is intentionally the same schedule as the legacy trainer's streaming path. Production
    users may prefer to point ``--text_files`` at their own vetted corpora instead.
    """
    from datasets import load_dataset

    def guarded(name, factory):
        try:
            dataset = iter(factory())
            print(f"  [ok] {name}")
            return dataset
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise RuntimeError(f"{name} unavailable: {type(exc).__name__}: {exc}") from exc
            print(f"  [!] {name} unavailable: {exc}")
            return None

    edu = guarded(
        "FineWeb-Edu",
        lambda: load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True),
    )
    cosmo = guarded(
        "Cosmopedia-v2",
        lambda: load_dataset("HuggingFaceTB/cosmopedia-v2", "cosmopedia-v2", split="train", streaming=True),
    )
    code = guarded(
        "GitHub Python (parquet)",
        lambda: load_dataset(
            "parquet",
            data_files={
                "train": "hf://datasets/codeparrot/github-code/data/train-00000-of-01126.parquet"
            },
            split="train",
            streaming=True,
        ),
    )
    gsm = guarded(
        "GSM8K",
        lambda: load_dataset("openai/gsm8k", "main", split="train", streaming=True),
    )
    tools = guarded(
        "Glaive function calling",
        lambda: load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True),
    )

    if world_size > 1:
        streams = [
            itertools.islice(stream, rank, None, world_size) if stream is not None else None
            for stream in (edu, cosmo, code, gsm, tools)
        ]
        edu, cosmo, code, gsm, tools = streams

    def take(stream, count):
        if stream is None:
            return []
        out = []
        for _ in range(count):
            try:
                out.append(next(stream))
            except StopIteration:
                break
        return out

    while True:
        produced = False
        for row in take(edu, 4):
            text = row.get("text", "")
            if text.strip():
                produced = True
                yield text
        for row in take(cosmo, 2):
            text = row.get("text", "")
            if text.strip():
                produced = True
                yield text
        for row in take(code, 2):
            if str(row.get("language", row.get("lang", ""))).lower() != "python":
                continue
            text = row.get("text", "") or row.get("code", "") or row.get("content", "")
            if text.strip():
                produced = True
                yield f"```python\n{text}\n```"
        for row in take(gsm, 1):
            answer = row.get("answer", "")
            final = answer.split("####")[-1].strip() if "####" in answer else answer
            produced = True
            yield (
                f"Question: {row.get('question', '')}\n"
                f"<think>\n{answer}\n</think>\nFinal Answer: {final}\n"
            )
        for row in take(tools, 1):
            text = row.get("chat", "") or row.get("text", "")
            if text.strip():
                produced = True
                yield text
        if not produced:
            raise RuntimeError("the corpus blend produced no usable documents")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data/tokens", help="output directory for shards")
    parser.add_argument("--text_files", nargs="*", default=None, help="local text files to tokenize")
    parser.add_argument(
        "--line_is_document",
        action="store_true",
        help="treat every line of the text files as a separate document",
    )
    parser.add_argument("--hf_blend", action="store_true", help="use the five-corpus HF stream")
    parser.add_argument("--allow_synthetic_fallback", action="store_true", help="skip missing HF streams")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer_32k")
    parser.add_argument("--tokens_per_shard", type=int, default=250_000_000)
    parser.add_argument("--max_tokens", type=int, default=0, help="0 means no limit")
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint32")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    if not args.text_files and not args.hf_blend:
        parser.error("choose --text_files or --hf_blend")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / args.tokenizer))
    eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2

    if args.text_files:
        texts = iter_text_files(args.text_files, args.line_is_document)
    else:
        texts = iter_hf_blend(tokenizer, args.rank, args.world_size, args.allow_synthetic_fallback)

    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    docs = iter_shard_tokens(texts, encode, eos)

    if args.max_tokens > 0:
        def limited(iterator):
            emitted = 0
            for doc in iterator:
                if emitted >= args.max_tokens:
                    return
                doc = doc[: args.max_tokens - emitted]
                emitted += int(doc.size)
                yield doc

        docs = limited(docs)
        print(f"[*] Limiting corpus to {args.max_tokens:,} tokens")

    print(f"[*] Writing shards to {args.out} (dtype={args.dtype}, {args.tokens_per_shard:,} tokens/shard)")
    meta = write_token_shards(
        args.out,
        docs,
        tokens_per_shard=args.tokens_per_shard,
        eos_token_id=eos,
        dtype=args.dtype,
    )
    total = sum(int(entry["tokens"]) for entry in meta["shards"])
    print(f"[+] Wrote {len(meta['shards'])} shard(s), {total:,} tokens -> {Path(args.out) / 'metadata.json'}")


if __name__ == "__main__":
    main()
