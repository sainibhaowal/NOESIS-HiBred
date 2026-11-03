import os
import json
import sys
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from safetensors.torch import load_model
except ImportError:
    load_model = None

from model import Transformer, ModelArgs

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "encoding")))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../encoding")))

from encoding import (
    encode_case,
    encode_messages,
    load_cases,
    parse_message_from_completion_text,
    parse_tagged_text,
    to_json,
)
from image_processor import TEXT, prepare_vl_inputs


def cache_bytes(model: Transformer) -> int:
    """Bytes held by the non-persistent attention/MTP caches."""
    return sum(
        buf.numel() * buf.element_size()
        for name, buf in model.named_buffers()
        if "cache" in name
    )


def report_cache_budget(model: Transformer, max_new_tokens: int | None = None) -> None:
    """Print the cache footprint and the prompt budget for a requested output length."""
    print(
        f"[*] attention/MTP caches: {cache_bytes(model) / (1024 ** 3):.2f} GiB "
        f"at max_seq_len={model.max_seq_len:,}"
    )
    if max_new_tokens is None:
        return
    prompt_budget = model.max_seq_len - max_new_tokens
    if prompt_budget > 0:
        print(f"[*] --max-new-tokens={max_new_tokens:,} leaves {prompt_budget:,} prompt tokens")
    else:
        print(
            f"[!] --max-new-tokens={max_new_tokens:,} exceeds "
            f"max_seq_len={model.max_seq_len:,}; every request will fail"
        )


def _truncate_prompts(prompt_tokens, prompt_token_types, budget):
    prompt_tokens = [t[-budget:] if len(t) > budget else t for t in prompt_tokens]
    if prompt_token_types is not None:
        prompt_token_types = [x[-budget:] if len(x) > budget else x for x in prompt_token_types]
    return prompt_tokens, prompt_token_types


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    prompt_token_types: List[List[int]] | None = None,
    images=None,
    stop_token_ids: List[int] | None = None,
    prefill_chunk: int = 8192,
    truncate_prompt: bool = False,
) -> List[List[int]]:
    """Batch generation with a hard ``prompt + output <= max_seq_len`` budget.

    Prefill runs in ``prefill_chunk``-token chunks (``start_pos`` advances); decode then runs one
    token at a time. For positions still inside a prompt, the ground-truth token overrides the
    model's prediction, which is what lets a batch with unequal prompt lengths decode in lockstep.

    ``prompt + max_new_tokens`` must fit in ``model.max_seq_len``; over-budget prompts raise unless
    ``truncate_prompt`` is set (text-only left truncation).

    ``prompt_token_types`` / ``images`` come from ``image_processor.prepare_vl_inputs``. Image spans
    are only visible to the first prefill pass, so they must fit inside the shortest prompt and the
    whole image span is forced into that single first chunk.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if prefill_chunk <= 0:
        raise ValueError("prefill_chunk must be positive")
    if not prompt_tokens:
        raise ValueError("prompt_tokens must not be empty")
    if any(len(t) == 0 for t in prompt_tokens):
        raise ValueError("prompts must contain at least one token")
    if len(prompt_tokens) > model.max_batch_size:
        raise ValueError(
            f"batch of {len(prompt_tokens)} exceeds max_batch_size={model.max_batch_size}; "
            "generate in smaller batches"
        )

    prompt_lens = [len(t) for t in prompt_tokens]
    max_prompt_len = max(prompt_lens)
    prompt_budget = model.max_seq_len - max_new_tokens
    if prompt_budget <= 0:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} leaves no room in max_seq_len={model.max_seq_len}"
        )
    if max_prompt_len > prompt_budget:
        if not truncate_prompt or images is not None:
            raise ValueError(
                f"prompt length {max_prompt_len} + max_new_tokens {max_new_tokens} exceeds "
                f"max_seq_len {model.max_seq_len} (prompt budget is {prompt_budget}); "
                "pass --truncate-prompt to left-truncate text-only prompts"
            )
        prompt_tokens, prompt_token_types = _truncate_prompts(
            prompt_tokens, prompt_token_types, prompt_budget
        )
        prompt_lens = [len(t) for t in prompt_tokens]
        max_prompt_len = max(prompt_lens)

    if max_new_tokens == 0:
        return [[] for _ in prompt_tokens]

    total_len = max_prompt_len + max_new_tokens
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    token_types = None
    if images is not None:
        token_types = torch.full((len(prompt_tokens), total_len), TEXT, dtype=torch.long)
        for i, types in enumerate(prompt_token_types):
            token_types[i, : len(types)] = torch.tensor(types, dtype=torch.long)
        for sample in images:
            for img in sample or ():
                assert img.start + img.types.numel() <= min(prompt_lens), "image spans must fit in the prefill chunk"

    min_prompt_len = min(prompt_lens)
    stop_ids = {int(t) for t in (stop_token_ids or ())}
    if eos_id is not None:
        stop_ids.add(int(eos_id))
    stop_ids_t = torch.tensor(sorted(stop_ids), dtype=torch.long) if stop_ids else None
    finished = torch.zeros(len(prompt_tokens), dtype=torch.bool)
    prompt_mask = tokens != -1

    prev_pos = 0
    cur_pos = 0
    while cur_pos < total_len - 1:
        if cur_pos < min_prompt_len:
            # Positions below min_prompt_len are ground truth for every sample, so prefill them in
            # one memory-bounded chunk. Image spans must be consumed by the first pass.
            step = min_prompt_len if images is not None and prev_pos == 0 else prefill_chunk
            cur_pos = min(cur_pos + step, min_prompt_len)
        else:
            # From here on at least one sample is decoding, so each step must be single-token.
            cur_pos += 1
        with_images = images is not None and prev_pos == 0
        next_token = model.forward(
            tokens[:, prev_pos:cur_pos],
            prev_pos,
            images=images if with_images else None,
            token_types=token_types[:, prev_pos:cur_pos] if with_images else None,
        )[0]
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        if stop_ids_t is not None:
            hit = (next_token.unsqueeze(1) == stop_ids_t.to(next_token.device)).any(dim=1)
            finished |= torch.logical_and(~prompt_mask[:, cur_pos], hit)
        prev_pos = cur_pos
        if finished.all():
            break

    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        cut = len(toks)
        for sid in stop_ids:
            if sid in toks:
                cut = min(cut, toks.index(sid))
        completion_tokens.append(toks[:cut])
    return completion_tokens


def prepare_case(case, thinking_mode, tokenizer, args):
    """Encode one message case and expand any image placeholders."""
    if case.get("context"):
        raise ValueError("Standalone inference does not support context without a prefilled KV cache")
    prompt, image_records = encode_case(case, thinking_mode)
    tokens, token_types, images = prepare_vl_inputs(prompt, image_records, tokenizer, args)
    return prompt, tokens, token_types, images


def _trim_stop_strings(text: str, stop_strings: list[str]) -> str:
    cut = len(text)
    for stop in stop_strings:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut]


def resolve_weights_path(ckpt_path: str, rank: int = 0, world_size: int = 1) -> str:
    """Per-rank sharded weights, or the unsharded ``model.safetensors`` from an export."""
    per_rank = os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    if os.path.exists(per_rank):
        return per_rank
    fallback = os.path.join(ckpt_path, "model.safetensors")
    if os.path.exists(fallback):
        return fallback
    return per_rank


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    thinking_mode: str = "chat",
    prefill_chunk: int = 8192,
    truncate_prompt: bool = False,
    stop_token_ids: List[int] | None = None,
    stop_strings: List[str] | None = None,
) -> None:
    stop_strings = list(stop_strings or ())
    stop_token_ids = list(stop_token_ids or ())
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    # Rank-local logger; a global `print` override here used to leak across repeated calls.
    def log(*args, **kwargs):
        if rank == 0:
            print(*args, **kwargs)

    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    if device.type == "cuda" and hasattr(torch._C, "_accelerator_setAllocatorSettings"):
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    if not config:
        config = os.path.join(ckpt_path, "config.json")
    with open(config) as f:
        args = ModelArgs(**json.load(f))
        args.temperature = temperature
    if interactive:
        # Single interactive request: cache one sequence, but keep the full configured context.
        args.max_batch_size = 1
    log(args)
    # Exported checkpoints keep the tokenizer in a subdirectory so the root config.json is
    # unambiguously the model config.
    tokenizer_path = os.path.join(ckpt_path, "tokenizer")
    if not os.path.exists(tokenizer_path):
        tokenizer_path = ckpt_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    log("build model")
    with torch.device(device):
        model = Transformer(args, tokenizer)
    log("load model")
    load_model(model, resolve_weights_path(ckpt_path, rank, world_size))
    torch.set_default_device(device)
    if rank == 0:
        report_cache_budget(model, max_new_tokens)
    log("I'm NOESIS-Σ 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode=thinking_mode))
            completions = generate(
                model,
                [prompt_tokens],
                max_new_tokens,
                tokenizer.eos_token_id,
                stop_token_ids=stop_token_ids,
                prefill_chunk=prefill_chunk,
                truncate_prompt=truncate_prompt,
            )
            completion = _trim_stop_strings(tokenizer.decode(completions[0]), stop_strings)
            log(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode=thinking_mode))
    else:
        if input_file.endswith(".json"):
            # Harmony input: a JSON file with one or more OpenAI-format cases
            # ({"messages": [...], "tools": [...]} or a bare message list).
            cases = load_cases(input_file)
            raw_prompts = [to_json(case["messages"]) for case in cases]
        else:
            # Plain-text input: blank-line-separated prompts, optionally with
            # <image>path</image> tags.
            with open(input_file) as f:
                raw_prompts = f.read().rstrip("\n").split("\n\n")
            cases = [{"messages": [{"role": "user", "content": parse_tagged_text(prompt)}]} for prompt in raw_prompts]

        prompt_tokens, prompt_token_types, images = [], [], []
        for case in cases:
            _, tokens, token_types, image_inputs = prepare_case(case, thinking_mode, tokenizer, args)
            prompt_tokens.append(tokens)
            prompt_token_types.append(token_types)
            images.append(image_inputs)

        if any(images):
            # image spans must be prefilled in one chunk, so VL prompts are generated one at a time
            completion_tokens = [
                generate(
                    model,
                    [tok],
                    max_new_tokens,
                    tokenizer.eos_token_id,
                    [types],
                    [image],
                    stop_token_ids=stop_token_ids,
                    prefill_chunk=prefill_chunk,
                    truncate_prompt=truncate_prompt,
                )[0]
                for tok, types, image in zip(prompt_tokens, prompt_token_types, images)
            ]
        else:
            completion_tokens = generate(
                model,
                prompt_tokens,
                max_new_tokens,
                tokenizer.eos_token_id,
                stop_token_ids=stop_token_ids,
                prefill_chunk=prefill_chunk,
                truncate_prompt=truncate_prompt,
            )
        completions = [
            _trim_stop_strings(text, stop_strings) for text in tokenizer.batch_decode(completion_tokens)
        ]
        for raw_prompt, completion in zip(raw_prompts, completions):
            log("Prompt:", raw_prompt)
            log("Completion:", completion)
            log()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="model config; defaults to <ckpt-path>/config.json for exported checkpoints",
    )
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="output budget; long answers (64K-128K) must be requested explicitly and must fit "
        "together with the prompt inside the config's max_seq_len",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--thinking-mode", type=str, default="chat", choices=["chat", "thinking"])
    parser.add_argument(
        "--prefill-chunk",
        type=int,
        default=8192,
        help="tokens per prefill forward; bounds activation memory for long prompts",
    )
    parser.add_argument(
        "--truncate-prompt",
        action="store_true",
        help="left-truncate text prompts that do not fit the prompt budget",
    )
    parser.add_argument("--stop-token-ids", type=int, nargs="*", default=[])
    parser.add_argument("--stop-strings", type=str, nargs="*", default=[])
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(
        args.ckpt_path,
        args.config,
        args.input_file,
        args.interactive,
        args.max_new_tokens,
        args.temperature,
        args.thinking_mode,
        args.prefill_chunk,
        args.truncate_prompt,
        args.stop_token_ids,
        args.stop_strings,
    )
