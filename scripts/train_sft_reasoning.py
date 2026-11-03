#!/usr/bin/env python3
# NOESIS-Σ — Post-Training Supervised Fine-Tuning (SFT) Reasoning Alignment
# Aligns NOESIS-Σ 250M on Multi-Step Reasoning Traces (ChatML format)
#
# Path: scripts/train_sft_reasoning.py

from __future__ import annotations
import sys
import os
import argparse
import time
import math
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
CORE_DIR = PROJECT_ROOT / "core"
ENCODING_DIR = CORE_DIR / "encoding"

for p in (str(CORE_DIR), str(ENCODING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import ModelArgs, NoesisSigmaTransformer, sample
from verifier_loss import SigmaVerifierLoss
from transformers import AutoTokenizer


def run_sft_reasoning(
    base_checkpoint: str = "checkpoints/noesis_sigma_1.5b/latest.pt",
    config_path: str = "core/config_1.5b.json",
    steps: int = 500,
    batch_size: int = 2,
    seq_len: int = 256,
    lr: float = 1e-4,
    output_dir: str = "checkpoints/noesis_sigma_1.5b",
):
    from datasets import load_dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("NOESIS-Σ POST-TRAINING: SUPERVISED REASONING ALIGNMENT (SFT)")
    print(f"Device: {device} | Base Checkpoint: {base_checkpoint}")
    print(f"Target Steps: {steps:,} | Learning Rate: {lr}")
    print("=" * 88)

    # 1. Load Config & Model
    cfg_file = PROJECT_ROOT / config_path
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)
    args = ModelArgs(**cfg_dict)
    args.max_batch_size = max(args.max_batch_size, batch_size)
    args.max_seq_len = max(args.max_seq_len, seq_len)

    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.set_default_dtype(torch.bfloat16)
    with torch.device(device):
        model = NoesisSigmaTransformer(args)
        if device.type == "cuda":
            model = model.to(torch.bfloat16)
    torch.set_default_dtype(torch.float32)

    ckpt_file = PROJECT_ROOT / base_checkpoint
    if ckpt_file.exists():
        print(f"[*] Loading weights from {ckpt_file}...")
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print("[+] Checkpoint loaded successfully!")
    else:
        print(f"[!] Warning: Checkpoint {ckpt_file} not found. Training from initialization.")

    # 2. Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "checkpoints/tokenizer_32k"))
    eos_id = tokenizer.eos_token_id or 2

    print("[*] Streaming GSM8K Reasoning Dataset...")
    ds = iter(load_dataset("openai/gsm8k", "main", split="train", streaming=True))

    try:
        import bitsandbytes as bnb
        if device.type == "cuda":
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
        else:
            optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    except ImportError:
        optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-5)
    loss_fn = SigmaVerifierLoss(alpha_attractor=0.05, beta_smoothness=0.01)

    model.train()
    print(f"\n{'Step':>6} | {'SFT Loss':>10} | {'Attractor ΔE':>15} | {'‖∇Φ(x)‖':>9} | {'PPL':>8}")
    print("-" * 65)

    for step in range(1, steps + 1):
        # Build ChatML batch
        tokens_list = []
        needed = batch_size * seq_len
        while len(tokens_list) < needed:
            try:
                item = next(ds)
                prompt = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n{item['answer']}<|im_end|>\n"
                t_ids = tokenizer.encode(prompt, add_special_tokens=False) + [eos_id]
                tokens_list.extend(t_ids)
            except StopIteration:
                ds = iter(load_dataset("openai/gsm8k", "main", split="train", streaming=True))

        tokens_tensor = torch.tensor(tokens_list[:needed], dtype=torch.long, device=device).view(batch_size, seq_len)
        input_ids = tokens_tensor[:, :-1].contiguous()
        targets = tokens_tensor[:, 1:].contiguous()

        logits, osc_metrics, h_out = model.forward_train(input_ids)
        delta_E = osc_metrics.get("energy_delta", torch.tensor(0.0, device=device)).mean()
        if hasattr(model, "osc_thinking_block") and model.osc_thinking_block is not None:
            x_norm = model.osc_thinking_block.norm(h_out)
            grad_phi = model.osc_thinking_block.icnn.grad(x_norm)
            grad_norm = torch.linalg.norm(grad_phi, dim=-1).mean()
        else:
            grad_norm = torch.tensor(0.0, device=device)

        loss, _ = loss_fn(
            logits=logits,
            targets=targets,
            osc_metrics={"energy_delta": delta_E},
            hidden_states=h_out,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        ppl = math.exp(min(loss.item(), 20.0))
        print(f"{step:6d} | {loss.item():10.4f} | {delta_E.item():15.4f} | {grad_norm.item():9.4f} | {ppl:8.2f}")

        if step % 100 == 0 or step == steps:
            instruct_path = out_path / "instruct_latest.pt"
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "config": cfg_dict,
                "loss": loss.item(),
            }, instruct_path)
            print(f"[+] Saved Instruct Aligned Model: {instruct_path}")

    print("\n[✔] SFT Reasoning Alignment Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOESIS-Σ SFT Alignment")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/noesis_sigma_1.5b/latest.pt")
    args = parser.parse_args()

    run_sft_reasoning(
        base_checkpoint=args.checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
    )
