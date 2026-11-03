#!/usr/bin/env python3
"""Export a training checkpoint into a ``generate.py``-ready directory.

``scripts/pretrain_1.5b_production.py`` writes ``latest.pt`` containing FSDP
full state dicts (prefixed keys, possibly fp32 master weights). This script:

  1. loads ``latest.pt`` (or a ``--checkpoint`` file / run directory),
  2. strips FSDP/compile wrappers from the parameter names,
  3. validates shapes against a meta-device model built from the checkpoint config,
  4. casts floating parameters to bf16 (default), and
  5. writes ``model.safetensors``, ``config.json`` and the tokenizer into ``--output-dir``.

The result loads directly in ``core/generate.py --ckpt-path <output-dir>``.

Examples
--------
    python scripts/export_checkpoint.py --checkpoint checkpoints/noesis_sigma_1.5b
    python scripts/export_checkpoint.py --checkpoint run/latest.pt --output-dir exported --dtype fp32
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for path in (PROJECT_ROOT / "core", PROJECT_ROOT / "core" / "encoding"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from model import ModelArgs, Transformer, configure_model_parallel  # noqa: E402

_STRIP_PREFIXES = (
    "_fsdp_wrapped_module.",
    "_orig_mod.",
    "_checkpoint_wrapped_module.",
)

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def clean_param_name(name: str) -> str:
    """Remove FSDP/``torch.compile`` wrapper prefixes from a state-dict key."""
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def load_payload(checkpoint: Path) -> dict:
    if checkpoint.is_dir():
        checkpoint = checkpoint / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in payload:
        raise ValueError(f"{checkpoint} has no model_state_dict (is this a training checkpoint?)")
    return payload


def clean_state_dict(model_state: dict) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    flat = [name for name in model_state if "_flat_param" in name]
    if flat:
        raise ValueError(
            "checkpoint holds FSDP flat parameters; save with FULL_STATE_DICT before exporting"
        )
    for raw_name, tensor in model_state.items():
        name = clean_param_name(raw_name)
        if name in cleaned:
            raise ValueError(f"duplicate parameter after prefix stripping: {name}")
        cleaned[name] = tensor
    return cleaned


def validate_against_model(
    state: dict[str, torch.Tensor],
    args: ModelArgs,
    tokenizer,
    allow_partial: bool,
) -> tuple[list[str], list[str], list[str]]:
    with torch.device("meta"):
        model = Transformer(args, tokenizer=tokenizer)
    expected = model.state_dict()
    missing = [name for name in expected if name not in state]
    unexpected = [name for name in state if name not in expected]
    mismatched = [
        f"{name}: checkpoint {tuple(state[name].shape)} != model {tuple(expected[name].shape)}"
        for name in expected
        if name in state and tuple(state[name].shape) != tuple(expected[name].shape)
    ]
    if (missing or unexpected or mismatched) and not allow_partial:
        problems = []
        if missing:
            problems.append(f"{len(missing)} missing (e.g. {missing[:3]})")
        if unexpected:
            problems.append(f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
        if mismatched:
            problems.append(f"{len(mismatched)} shape mismatches (e.g. {mismatched[:3]})")
        raise ValueError("state dict does not match the config: " + "; ".join(problems))
    return missing, unexpected, mismatched


def export(
    checkpoint: Path,
    output_dir: Path,
    dtype_name: str,
    config_path: Path | None,
    tokenizer_path: Path | None,
    allow_partial: bool,
) -> None:
    payload = load_payload(checkpoint)
    config = payload.get("config")
    if config is None:
        if config_path is None:
            raise ValueError(
                "checkpoint has no embedded config; pass --config to select the architecture"
            )
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)
    elif config_path is not None:
        with open(config_path, encoding="utf-8") as handle:
            config = json.load(handle)

    configure_model_parallel(1, 0)
    args = ModelArgs(**config)

    tokenizer = None
    tokenizer_dir = tokenizer_path or (PROJECT_ROOT / "checkpoints" / "tokenizer_32k")
    if tokenizer_dir.exists():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))

    state = clean_state_dict(payload["model_state_dict"])
    missing, unexpected, mismatched = validate_against_model(
        state, args, tokenizer, allow_partial
    )
    if missing:
        print(f"[!] exporting partial checkpoint: missing {len(missing)} keys")
    if unexpected:
        print(f"[!] dropping {len(unexpected)} unexpected keys")
    if mismatched:
        print(f"[!] dropping {len(mismatched)} mismatched keys")

    dtype = _DTYPES[dtype_name]
    dropped = set(unexpected) | {m.split(":")[0] for m in mismatched}
    exported: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name in dropped:
            continue
        tensor = tensor.detach().contiguous()
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)
        exported[name] = tensor.clone()

    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / "model.safetensors"
    save_file(exported, str(weights_path))

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    # Keep the tokenizer in a subdirectory: `config.json` at the root is the model config,
    # and transformers would otherwise try to parse it as a tokenizer's `PretrainedConfig`.
    tokenizer_out = output_dir / "tokenizer"
    if tokenizer is not None:
        tokenizer.save_pretrained(str(tokenizer_out))
    elif tokenizer_dir.exists():
        tokenizer_out.mkdir(exist_ok=True)
        for item in tokenizer_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, tokenizer_out / item.name)

    total = sum(t.numel() for t in exported.values())
    print(f"[+] exported {len(exported)} tensors ({total:,} params, {dtype_name})")
    print(f"[+] weights: {weights_path} ({weights_path.stat().st_size / (1024**3):.2f} GiB)")
    print(f"[+] config:  {output_dir / 'config.json'}")
    if tokenizer_out.exists():
        print(f"[+] tokenizer: {tokenizer_out}")
    print(f"[+] load with: python core/generate.py --ckpt-path {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/noesis_sigma_1.5b",
        help="latest.pt file or a run directory containing latest.pt",
    )
    parser.add_argument("--output-dir", default=None, help="defaults to <checkpoint dir>/exported")
    parser.add_argument("--dtype", choices=tuple(_DTYPES), default="bf16")
    parser.add_argument("--config", default=None, help="override the checkpoint's embedded config")
    parser.add_argument("--tokenizer", default=None, help="tokenizer directory to copy")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="export the keys that match instead of failing on missing/unexpected keys",
    )
    opts = parser.parse_args()

    checkpoint = Path(opts.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    output_dir = opts.output_dir
    if output_dir is None:
        base = checkpoint.parent if checkpoint.is_file() else checkpoint
        output_dir = base / "exported"
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    export(
        checkpoint,
        output_dir,
        opts.dtype,
        Path(opts.config) if opts.config else None,
        Path(opts.tokenizer) if opts.tokenizer else None,
        opts.allow_partial,
    )


if __name__ == "__main__":
    main()
