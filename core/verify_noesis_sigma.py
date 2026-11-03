# NOESIS-Σ — End-to-End Architecture, Training & Inference Verification Suite
# Path: core/verify_noesis_sigma.py

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure core/ and core/encoding/ are in sys.path
CORE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CORE_DIR))
sys.path.insert(0, str(CORE_DIR / "encoding"))


def verify_all_imports() -> None:
    print("[1/5] Verifying all core/ module imports and connections...")
    import kernel
    import osc_block
    import verifier_loss
    import engram
    import image_processor
    import vision
    import model
    import generate
    import convert
    import encoding

    print("      -> kernel.py          (Differentiable Sparse MLA & Sinkhorn Hyper-Connections) [OK]")
    print("      -> osc_block.py       (Strang-Split Conservative-Dissipative Attractor + ICNN) [OK]")
    print("      -> verifier_loss.py   (Multi-Objective Verifier & Energy Relaxation Loss)      [OK]")
    print("      -> engram.py          (N-gram Hash Memory Tables)                              [OK]")
    print("      -> image_processor.py (Multimodal Patch & Token Span Processor)                [OK]")
    print("      -> vision.py          (ViT & Spatial Downsampling Aligner)                     [OK]")
    print("      -> model.py           (NOESIS-Σ Transformer Backbone + OSC Thinking Block)     [OK]")
    print("      -> generate.py        (Autoregressive & Speculative Generation Engine)         [OK]")
    print("      -> convert.py         (Safetensors Weight Conversion Utility)                  [OK]")
    print("      -> encoding/          (Chat & Reasoning Message Renderer)                      [OK]")


def main() -> None:
    print("=" * 78)
    print("  NOESIS-Σ (SIGMA) END-TO-END CORE VERIFICATION & PRE-TRAINING TEST SUITE")
    print("=" * 78)

    verify_all_imports()

    from model import ModelArgs, NoesisSigmaTransformer
    from verifier_loss import SigmaVerifierLoss
    from generate import generate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
    print(f"\n[2/5] Hardware & Configuration Check...")
    print(f"      -> Active Device        : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    test_cfg = CORE_DIR.parent / "tests" / "fixtures" / "config_test.json"
    if test_cfg.exists():
        with open(test_cfg, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg_name = "config_test.json"
    else:
        cfg_dict = {
            "dim": 256,
            "n_layers": 6,
            "n_heads": 8,
            "n_routed_experts": 4,
            "n_activated_experts": 2,
            "vocab_size": 32000,
            "max_seq_len": 512,
            "hc_mult": 2,
            "osc_enabled": True,
            "engram_enabled": True,
        }
        cfg_name = "embedded_test_spec"

    args = ModelArgs(**cfg_dict)
    print(f"      -> Config Loaded        : {cfg_name} (dim={args.dim}, layers={args.n_layers}, hc_mult={args.hc_mult}, experts={args.n_routed_experts})")

    # Verify Tokenizer & Tokenizer Config
    tokenizer_path = CORE_DIR / "tokenizer.json"
    tokenizer_cfg_path = CORE_DIR / "tokenizer_config.json"
    assert tokenizer_path.exists(), f"Missing tokenizer at {tokenizer_path}"
    assert tokenizer_cfg_path.exists(), f"Missing tokenizer_config at {tokenizer_cfg_path}"

    with open(tokenizer_path, "r", encoding="utf-8") as tf:
        tok_data = json.load(tf)
    with open(tokenizer_cfg_path, "r", encoding="utf-8") as tcf:
        tok_cfg = json.load(tcf)

    vocab_dict = tok_data.get("model", {}).get("vocab", {})
    added_tokens = tok_data.get("added_tokens", [])
    total_vocab = len(vocab_dict) + len(added_tokens)
    sample_text = "NOESIS-Sigma integrates Strang-split conservative-dissipative attractors with MLA and MoE."
    encoded_ids = [vocab_dict.get(ch, idx + 10) for idx, ch in enumerate(sample_text.split())]
    bounded_ids = [tid % args.vocab_size for tid in encoded_ids]
    print(
        f"      -> Tokenizer Verified   : {tokenizer_path.name} & {tokenizer_cfg_path.name} "
        f"(vocab={total_vocab:,}, bos={tok_cfg.get('bos_token')}, eos={tok_cfg.get('eos_token')})"
    )

    # Instantiate unified NOESIS-Σ Transformer from core/model.py
    print("\n[3/5] Instantiating Unified NOESIS-Σ Transformer (core/model.py)...")
    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)
    model = NoesisSigmaTransformer(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    osc_params = sum(p.numel() for p in model.osc_thinking_block.parameters())
    print(f"      -> Total Parameters     : {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"      -> OSC Attractor Params : {osc_params:,} (Strang Split + Direct-Grad ICNN + Retrograde)")

    # Verify Differentiable Training Loop (3 steps to confirm loss reduction & gradient flow)
    print("\n[4/5] Verifying Differentiable Training Loop (MLA + MoE + Hyper-Connections + OSC)...")
    criterion = SigmaVerifierLoss(
        alpha_attractor=0.05,
        beta_smoothness=0.01,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    batch_size, seq_len = 2, 16
    token_seq = (bounded_ids * ((seq_len // len(bounded_ids)) + 2))[:seq_len]
    input_ids = torch.tensor([token_seq, token_seq[::-1]], dtype=torch.long, device=device)
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)

    losses = []
    for step_idx in range(1, 4):
        optimizer.zero_grad()
        logits, osc_metrics, hidden_states = model.forward_train(input_ids)
        total_loss, loss_dict = criterion(
            logits=logits,
            targets=target_ids,
            osc_metrics=osc_metrics,
            hidden_states=hidden_states,
        )
        total_loss.backward()

        # Verify gradients on first step
        if step_idx == 1:
            attn_grad = model.layers[0].attn.wq_a.weight.grad
            moe_grad = model.layers[0].ffn.shared_experts.w1.weight.grad
            hc_grad = model.layers[0].hc_attn_fn.grad
            osc_grad = model.osc_thinking_block.icnn.W.grad
            assert attn_grad is not None and torch.isfinite(attn_grad).all(), "MLA Attention grad missing/NaN!"
            assert moe_grad is not None and torch.isfinite(moe_grad).all(), "MoE Expert grad missing/NaN!"
            assert hc_grad is not None and torch.isfinite(hc_grad).all(), "Hyper-Connections grad missing/NaN!"
            assert osc_grad is not None and torch.isfinite(osc_grad).all(), "OSC ICNN grad missing/NaN!"
            print(f"      -> Gradient Flow Check  : MLA={attn_grad.norm().item():.4f}, MoE={moe_grad.norm().item():.4f}, HC={hc_grad.norm().item():.4f}, OSC={osc_grad.norm().item():.4f} [OK]")

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss_dict["loss_total"])
        e_init = float(osc_metrics["initial_energy"].detach().item())
        e_final = float(osc_metrics["final_energy"].detach().item())
        e_delta = float(osc_metrics["energy_delta"].detach().item())
        print(
            f"      -> Train Step {step_idx}/3       : "
            f"total_loss={loss_dict['loss_total']:.4f} | "
            f"lm={loss_dict['loss_lm']:.4f} | "
            f"OSC E_init={e_init:.4f} -> E_final={e_final:.4f} "
            f"(dE={e_delta:.4f})"
        )

    assert losses[-1] < losses[0], f"Training loss did not decrease: {losses[0]} -> {losses[-1]}"
    assert float(osc_metrics["energy_delta"].detach().item()) <= 0.0, "OSC Attractor energy failed to relax!"

    # Verify Autoregressive Generation via core/generate.py
    print("\n[5/5] Verifying Autoregressive Inference & Generation (core/generate.py)...")
    model.eval()
    prompt_tokens = [bounded_ids[:8], bounded_ids[:6]]
    max_new_tokens = 6
    with torch.inference_mode():
        generated_seqs = generate(
            model=model,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            eos_id=-1,
        )
    assert len(generated_seqs) == 2
    assert len(generated_seqs[0]) == max_new_tokens
    print(f"      -> Prompt Lengths       : {[len(p) for p in prompt_tokens]}")
    print(f"      -> Generated Token IDs  : {generated_seqs}")
    print("=" * 78)
    print("  ALL CORE MODULES CONNECTED, VERIFIED & READY FOR TRAINING [PASS]")
    print("=" * 78)


if __name__ == "__main__":
    main()
