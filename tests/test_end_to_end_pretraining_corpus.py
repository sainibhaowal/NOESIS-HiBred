# NOESIS-Σ End-to-End Real Corpus Pre-Training, Checkpoint & Generation Readiness Suite
# Path: tests/test_end_to_end_pretraining_corpus.py

from __future__ import annotations
from pathlib import Path
import math

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

from model import ModelArgs, NoesisSigmaTransformer
from verifier_loss import SigmaVerifierLoss
from generate import generate
import encoding as enc


SAMPLE_CORPUS = [
    "NOESIS-Sigma unifies conservative skew-symmetric Hamiltonian dynamics with dissipative ICNN attractors.",
    "Multi-Head Latent Attention compresses key-value states while preserving long-horizon reasoning fidelity.",
    "Doubly-stochastic Sinkhorn Hyper-Connections route residual streams across parallel width channels.",
    "Top-k Mixture-of-Experts with shared expert pathways scales model capacity with sub-linear compute.",
    "Second-order symmetric Strang operator splitting guarantees Lyapunov stability during latent thinking.",
    "Retrograde causal feedback bounds token-to-token transitions under Input-to-State Stability constraints.",
    "The multi-objective SigmaVerifierLoss enforces cross-entropy prediction, attractor basin descent, and smoothness.",
    "Autoregressive prefill and ring-buffer KV decoding enable fast speculative and interactive generation.",
]


class CorpusPretrainingDataset(Dataset):
    """Tokenizes a text corpus using the NOESIS-Σ vocabulary and produces fixed-length causal blocks."""

    def __init__(self, texts: list[str], vocab: dict[str, int], vocab_size: int, seq_len: int = 24) -> None:
        self.seq_len = seq_len
        self.samples: list[ tuple[torch.Tensor, torch.Tensor] ] = []

        for doc in texts:
            # Render in chat format via core/encoding and map tokens through tokenizer.json vocab
            rendered = enc.encode_messages([{"role": "user", "content": doc}], thinking_mode="thinking")
            raw_tokens = [vocab.get(tok, abs(hash(tok))) % vocab_size for tok in rendered.split()]
            # Pad or tile to seq_len + 1 for causal next-token prediction
            while len(raw_tokens) < seq_len + 1:
                raw_tokens = raw_tokens + raw_tokens
            token_Ids = raw_tokens[: seq_len + 1]
            x = torch.tensor(token_Ids[:-1], dtype=torch.long)
            y = torch.tensor(token_Ids[1:], dtype=torch.long)
            self.samples.append((x, y))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def test_end_to_end_corpus_pretraining_checkpoint_and_generation(
    device: torch.device,
    config_10m_dict: dict,
    tokenizer_vocab: dict[str, int],
    tmp_path: Path,
) -> None:
    """Full end-to-end pre-training test on a multi-document corpus:
     Corpus -> Tokenization -> DataLoader -> Forward/Backward/Optimizer -> Checkpoint Save/Load -> Generation."""
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    torch.set_default_device(device)
    torch.set_default_dtype(torch.bfloat16)

    args = ModelArgs(**config_10m_dict)
    dataset = CorpusPretrainingDataset(
        texts=SAMPLE_CORPUS,
        vocab=tokenizer_vocab,
        vocab_size=args.vocab_size,
        seq_len=16,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        generator=torch.Generator(device=device),
    )

    model = NoesisSigmaTransformer(args).to(device)
    criterion = SigmaVerifierLoss(alpha_attractor=0.05, beta_smoothness=0.01).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.95), weight_decay=0.01)

    epoch_losses: list[float] = []
    model.train()

    for epoch in range(3):
        step_losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits, osc_metrics, hidden_states = model.forward_train(x_batch)
            total_loss, loss_dict = criterion(
                logits=logits,
                targets=y_batch,
                osc_metrics=osc_metrics,
                hidden_states=hidden_states,
            )
            total_loss.backward()

            # Verify every major architectural subsystem has finite non-zero gradients
            if epoch == 0 and len(step_losses) == 0:
                subsystems = {
                    "embed": model.embed.weight.grad,
                    "mla_wq_a": model.layers[0].attn.wq_a.weight.grad,
                    "mla_wkv": model.layers[0].attn.wkv.weight.grad,
                    "moe_gate": model.layers[0].ffn.gate.weight.grad,
                    "moe_shared": model.layers[0].ffn.shared_experts.w1.weight.grad,
                    "hc_sinkhorn": model.layers[0].hc_attn_fn.grad,
                    "osc_skew_u": model.osc_thinking_block.u_proj.weight.grad,
                    "osc_icnn_W": model.osc_thinking_block.icnn.W.grad,
                    "osc_icnn_c": model.osc_thinking_block.icnn.c.grad,
                    "osc_retro": model.osc_thinking_block.retrograde.w_retro.weight.grad,
                    "head": model.head.weight.grad,
                }
                for name, grad in subsystems.items():
                    assert grad is not None, f"Subsystem '{name}' received None gradient!"
                    assert torch.isfinite(grad).all(), f"Subsystem '{name}' has NaN/Inf gradient!"
                    assert grad.norm().item() > 0.0, f"Subsystem '{name}' has zero gradient norm!"

            # Verify OSC Lyapunov relaxation at every step
            assert float(osc_metrics["energy_delta"].detach().item()) <= 0.0

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step_losses.append(loss_dict["loss_total"])

        epoch_losses.append(sum(step_losses) / len(step_losses))

    # Loss must strictly decrease across epochs on the training corpus
    assert epoch_losses[-1] < epoch_losses[0], (
        f"Pre-training corpus loss failed to converge: {epoch_losses}"
    )

    # Save checkpoint and reload into a fresh model instance to verify serialization parity
    ckpt_file = tmp_path / "noesis_sigma_pretrained.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_10m_dict,
            "final_loss": epoch_losses[-1],
        },
        ckpt_file,
    )
    assert ckpt_file.exists()

    loaded_ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    reloaded_model = NoesisSigmaTransformer(ModelArgs(**loaded_ckpt["config"])).to(device)
    reloaded_model.load_state_dict(loaded_ckpt["model_state_dict"])

    model.eval()
    reloaded_model.eval()
    sample_x, _ = dataset[0]
    sample_x = sample_x.unsqueeze(0).to(device)
    with torch.inference_mode():
        logits_orig, _, _ = model.forward_train(sample_x)
        logits_reloaded, _, _ = reloaded_model.forward_train(sample_x)
        assert torch.allclose(logits_orig, logits_reloaded, atol=1e-4, rtol=1e-4)

        # Verify autoregressive generation from the reloaded pre-trained model
        prompts = [sample_x[0, :8].tolist(), sample_x[0, :6].tolist()]
        completions = generate(
            model=reloaded_model,
            prompt_tokens=prompts,
            max_new_tokens=6,
            eos_id=-1,
        )
        assert len(completions) == 2
        assert all(len(c) == 6 for c in completions)
