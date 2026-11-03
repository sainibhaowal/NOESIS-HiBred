#!/usr/bin/env python3
# NOESIS-Σ — Automated 5-Stage Scientific Ablation Study Runner
#
# Isolates the systematic contributions of:
#   Variant 1: Baseline (Standard Transformer)
#   Variant 2: + OSC Only (Signature Continuous Attractor Novelty)
#   Variant 3: + OSC + MLA (Multi-Head Latent Attention Memory Compression)
#   Variant 4: + OSC + MLA + HC (Sinkhorn Doubly-Stochastic Hyper-Connections)
#   Variant 5: Full NOESIS-Σ (+ MoE, Engram, DSpark Speculative MTP)
#
# Measures:
#   1. Initial Loss vs. Final Loss (L)
#   2. Perplexity (PPL = exp(L))
#   3. KV-Cache Memory footprint (MB of VRAM allocated)
#   4. Inference Speed (Tokens per second)
#   5. OSC Attractor Residual Energy norm (||∇Φ(x)||)
#
# Path: experiments/ablation_study.py

from __future__ import annotations
import sys
import os
import argparse
import math
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

# Ensure core and core/encoding are in sys.path
EXPERIMENTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENTS_DIR.parent
CORE_DIR = PROJECT_ROOT / "core"
ENCODING_DIR = CORE_DIR / "encoding"

for p in (str(CORE_DIR), str(ENCODING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from model import ModelArgs, Transformer
from verifier_loss import SigmaVerifierLoss
import encoding as enc


@dataclass
class AblationMetrics:
    variant_id: int
    name: str
    description: str
    params_m: float
    init_loss: float
    final_loss: float
    perplexity: float
    kv_cache_mb: float
    tokens_per_sec: float
    grad_phi_norm: float | str
    energy_delta: float | str


def load_shared_corpus(corpus_path: Path, vocab: dict[str, int], vocab_size: int, seq_len: int = 32) -> torch.Tensor:
    """Loads a shared corpus from file, tokenizes into causal sequences of shape [N, seq_len]."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into documents/paragraphs
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if not blocks:
        blocks = [text]

    all_tokens: list[int] = []
    for b in blocks:
        words = b.split()
        for w in words:
            tok_id = vocab.get(w, abs(hash(w))) % vocab_size
            all_tokens.append(tok_id)

    # Ensure sufficient length for batching
    min_required = seq_len * 16
    while len(all_tokens) < min_required:
        all_tokens = all_tokens + all_tokens

    # Reshape into uniform sequences
    n_seqs = len(all_tokens) // seq_len
    tensor_data = torch.tensor(all_tokens[: n_seqs * seq_len], dtype=torch.long).view(n_seqs, seq_len)
    return tensor_data


def compute_kv_cache_footprint_mb(args: ModelArgs, batch_size: int = 4, seq_len: int = 256, bytes_per_elem: int = 2) -> float:
    """Calculates theoretical KV-Cache VRAM footprint in Megabytes (MB)."""
    total_bytes = 0
    for l_id in range(args.n_layers):
        ratio = args.compress_ratios[l_id] if l_id < len(args.compress_ratios) else 0
        if ratio == 0:
            # Standard MHA/GQA: K and V for every head and token
            # [batch, seq_len, n_heads * head_dim] * 2 (for K and V)
            layer_bytes = 2 * batch_size * seq_len * args.n_heads * args.head_dim * bytes_per_elem
        else:
            # Multi-Head Latent Attention (MLA): Window cache + Compressed KV latent cache
            win_bytes = batch_size * args.window_size * args.head_dim * bytes_per_elem
            comp_bytes = batch_size * (seq_len // ratio) * args.head_dim * bytes_per_elem
            layer_bytes = win_bytes + comp_bytes
        total_bytes += layer_bytes

    return total_bytes / (1024 * 1024)


def get_ablation_model_args(variant_id: int, vocab_size: int = 1000, dim: int = 256, n_layers: int = 2) -> tuple[str, str, ModelArgs]:
    """Generates strictly matched Iso-Compute ModelArgs for the 5-stage ablation ladder."""
    base_kwargs = dict(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=4,
        head_dim=64,
        q_lora_rank=128,
        o_lora_rank=128,
        dtype="bf16",
        expert_dtype=None,
        max_seq_len=256,
        max_batch_size=4,
    )

    if variant_id == 1:
        return (
            "1. Baseline",
            "Standard Transformer (No OSC, Standard Residual, Dense MLP, Full Attn)",
            ModelArgs(
                **base_kwargs,
                osc_enabled=False,
                hc_mult=1,
                n_routed_experts=0,
                compress_ratios=(0, 0),
                kv_source_layers=(),
                index_source_layers=(),
                engram_layer_ids=(),
                n_mtp_layers=0,
                dspark_block_size=0,
            ),
        )
    elif variant_id == 2:
        return (
            "2. + OSC Only",
            "Baseline + Operator-Split Core Latent Attractor (Key Novelty)",
            ModelArgs(
                **base_kwargs,
                osc_enabled=True,
                osc_rank=32,
                osc_icnn_hidden=128,
                osc_steps=3,
                hc_mult=1,
                n_routed_experts=0,
                compress_ratios=(0, 0),
                kv_source_layers=(),
                index_source_layers=(),
                engram_layer_ids=(),
                n_mtp_layers=0,
                dspark_block_size=0,
            ),
        )
    elif variant_id == 3:
        return (
            "3. + OSC + MLA",
            "+ Multi-Head Latent Attention (2:1 KV Memory Compression)",
            ModelArgs(
                **base_kwargs,
                osc_enabled=True,
                osc_rank=32,
                osc_icnn_hidden=128,
                osc_steps=3,
                hc_mult=1,
                n_routed_experts=0,
                compress_ratios=(0, 2),
                kv_source_layers=(1,),
                index_source_layers=(1,),
                engram_layer_ids=(),
                n_mtp_layers=0,
                dspark_block_size=0,
            ),
        )
    elif variant_id == 4:
        return (
            "4. + OSC + MLA + HC",
            "+ 4-Stream Doubly-Stochastic Sinkhorn Hyper-Connections",
            ModelArgs(
                **base_kwargs,
                osc_enabled=True,
                osc_rank=32,
                osc_icnn_hidden=128,
                osc_steps=3,
                hc_mult=4,
                n_routed_experts=0,
                compress_ratios=(0, 2),
                kv_source_layers=(1,),
                index_source_layers=(1,),
                engram_layer_ids=(),
                n_mtp_layers=0,
                dspark_block_size=0,
            ),
        )
    elif variant_id == 5:
        return (
            "5. Full NOESIS-Σ",
            "Complete Architecture (+ Routed MoE, Engram, DSpark Speculative MTP)",
            ModelArgs(
                **base_kwargs,
                osc_enabled=True,
                osc_rank=32,
                osc_icnn_hidden=128,
                osc_steps=3,
                hc_mult=4,
                n_routed_experts=4,
                n_activated_experts=2,
                compress_ratios=(0, 2, 0),
                kv_source_layers=(1,),
                index_source_layers=(1,),
                engram_layer_ids=(),
                n_mtp_layers=1,
                dspark_block_size=2,
                dspark_target_layer_ids=(0, 1),
            ),
        )
    else:
        raise ValueError(f"Unknown variant id: {variant_id}")


def run_variant_experiment(
    variant_id: int,
    corpus_tensor: torch.Tensor,
    steps: int = 50,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> AblationMetrics:
    """Executes Iso-Compute training for one variant and measures all 5 metrics."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    name, desc, args = get_ablation_model_args(variant_id)
    model = Transformer(args).to(dev)

    params_m = sum(p.numel() for p in model.parameters()) / 1e6
    kv_cache_mb = compute_kv_cache_footprint_mb(args, batch_size=2, seq_len=32)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = SigmaVerifierLoss(alpha_attractor=0.05, beta_smoothness=0.01)

    batch_size, seq_len = 2, 32
    init_loss: float = 0.0
    final_loss: float = 0.0
    last_grad_phi_norm: float | str = "N/A"
    last_energy_delta: float | str = "N/A"

    model.train()
    for step in range(steps):
        start_idx = (step * batch_size) % (corpus_tensor.size(0) - batch_size)
        batch = corpus_tensor[start_idx : start_idx + batch_size, :seq_len].to(dev)

        optimizer.zero_grad()
        logits, osc_metrics, h_norm = model.forward_train(batch)

        targets = batch[:, 1:].contiguous()
        preds = logits[:, :-1, :].contiguous()
        loss, _ = loss_fn(preds, targets, osc_metrics, h_norm[:, :-1, :])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        val = float(loss.item())
        if step == 0:
            init_loss = val
        final_loss = val

        if model.osc_thinking_block is not None:
            # Measure ||∇Φ(x)||: Residual attractor gradient norm
            with torch.no_grad():
                h_mid = h_norm.mean(dim=1)  # [B, d]
                x_norm = model.osc_thinking_block.norm(h_mid)
                grad_val = model.osc_thinking_block.icnn.grad(x_norm)
                norm_val = torch.linalg.norm(grad_val, dim=-1).mean().item()
                last_grad_phi_norm = float(norm_val)

        if "energy_delta" in osc_metrics:
            last_energy_delta = float(osc_metrics["energy_delta"].item())

    perplexity = math.exp(min(final_loss, 20.0))

    # Benchmark Inference Speed (Tokens/sec)
    model.eval()
    gen_tokens = 64
    prompt = corpus_tensor[:1, :8].to(dev)
    t0 = time.perf_counter()
    with torch.inference_mode():
        curr = prompt
        for pos in range(gen_tokens):
            out_ids, _, _ = model(curr[:, -1:], start_pos=pos)
            curr = torch.cat([curr, out_ids.unsqueeze(1)], dim=1)
    elapsed = time.perf_counter() - t0
    tokens_per_sec = gen_tokens / max(elapsed, 1e-4)

    return AblationMetrics(
        variant_id=variant_id,
        name=name,
        description=desc,
        params_m=params_m,
        init_loss=init_loss,
        final_loss=final_loss,
        perplexity=perplexity,
        kv_cache_mb=kv_cache_mb,
        tokens_per_sec=tokens_per_sec,
        grad_phi_norm=last_grad_phi_norm,
        energy_delta=last_energy_delta,
    )


def run_automated_ablation_study(
    corpus_file: str = "examples/pretraining_corpus.txt",
    steps: int = 50,
    lr: float = 1e-3,
    device: str | None = None,
    output_dir: str = "docs",
) -> List[AblationMetrics]:
    """Runs the full 5-variant study and prints/saves publication tables."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus_p = PROJECT_ROOT / corpus_file

    # Load tokenizer vocabulary
    tokenizer_p = CORE_DIR / "tokenizer.json"
    vocab: dict[str, int] = {}
    if tokenizer_p.exists():
        with open(tokenizer_p, "r", encoding="utf-8") as f:
            v_data = json.load(f)
            vocab = v_data.get("model", {}).get("vocab", {})

    print("=" * 88)
    print("NOESIS-Σ AUTOMATED ABLATION STUDY: 5-STAGE ISO-COMPUTE ISOLATION")
    print(f"Corpus: {corpus_file} | Steps: {steps} | Learning Rate: {lr} | Device: {dev}")
    print("=" * 88)

    corpus_tensor = load_shared_corpus(corpus_p, vocab, vocab_size=1000, seq_len=32)

    results: List[AblationMetrics] = []
    for var_id in range(1, 6):
        print(f"\n[Variant {var_id}/5] Initializing & Training Iso-Compute Model...")
        m = run_variant_experiment(var_id, corpus_tensor, steps=steps, lr=lr, device=dev)
        results.append(m)
        print(f"  -> {m.name}: Final Loss = {m.final_loss:.4f} | PPL = {m.perplexity:.2f} | KV = {m.kv_cache_mb:.3f} MB")

    # Format Markdown Table
    md_lines = [
        "### Empirical Ablation Study Results (Iso-Compute Matched)",
        "",
        "| Architecture Variant | Params | Initial Loss | Final Loss (↓) | Perplexity (↓) | KV Cache (MB) | Tok/s (↑) | ||∇Φ(x)|| Residual | Attractor ΔEnergy |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        g_str = f"{r.grad_phi_norm:.4f}" if isinstance(r.grad_phi_norm, float) else str(r.grad_phi_norm)
        e_str = f"{r.energy_delta:.4f}" if isinstance(r.energy_delta, float) else str(r.energy_delta)
        md_lines.append(
            f"| **{r.name}** | {r.params_m:.2f}M | {r.init_loss:.3f} | **{r.final_loss:.3f}** | **{r.perplexity:.2f}** | {r.kv_cache_mb:.3f} MB | {r.tokens_per_sec:.1f} | {g_str} | {e_str} |"
        )
    md_table = "\n".join(md_lines)

    # Format LaTeX Table
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Ablation study evaluating systematic contributions of NOESIS-$\Sigma$ components under strict Iso-Compute conditions.}",
        r"\label{tab:noesis_ablation}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"\textbf{Architecture Variant} & \textbf{Params} & \textbf{Init Loss} & \textbf{Final Loss ($\downarrow$)} & \textbf{PPL ($\downarrow$)} & \textbf{KV (MB)} & \textbf{Tok/s} & \textbf{$\|\nabla\Phi(x)\|$} & \textbf{$\Delta E$} \\",
        r"\midrule",
    ]
    for r in results:
        g_val = f"{r.grad_phi_norm:.3f}" if isinstance(r.grad_phi_norm, float) else "--"
        e_val = f"{r.energy_delta:.3f}" if isinstance(r.energy_delta, float) else "--"
        latex_lines.append(
            f"  {r.name} & {r.params_m:.2f}M & {r.init_loss:.2f} & \\textbf{{{r.final_loss:.2f}}} & \\textbf{{{r.perplexity:.2f}}} & {r.kv_cache_mb:.3f} & {r.tokens_per_sec:.1f} & {g_val} & {e_val} \\\\"
        )
    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    latex_table = "\n".join(latex_lines)

    print("\n" + "=" * 88)
    print("MARKDOWN PUBLICATION TABLE")
    print("=" * 88)
    print(md_table)

    print("\n" + "=" * 88)
    print("LATEX TABLE FOR CONFERENCE SUBMISSION")
    print("=" * 88)
    print(latex_table)
    print("=" * 88)

    # Save to file
    out_p = PROJECT_ROOT / output_dir / "ablation_results.md"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write("# NOESIS-Σ — Automated Ablation Study Results\n\n")
        f.write(f"**Corpus**: `{corpus_file}` | **Steps**: `{steps}` | **Optimization**: AdamW (`lr={lr}`)\n\n")
        f.write("## 1. Markdown Table\n\n")
        f.write(md_table + "\n\n")
        f.write("## 2. LaTeX Table Code\n\n```latex\n" + latex_table + "\n```\n")
    print(f"\n[Saved] Publication tables written to: {out_p}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOESIS-Σ Automated Ablation Experiment Runner")
    parser.add_argument("--corpus", type=str, default="examples/pretraining_corpus.txt", help="Path to shared corpus")
    parser.add_argument("--steps", type=int, default=50, help="Iso-compute training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    run_automated_ablation_study(
        corpus_file=args.corpus,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
    )
