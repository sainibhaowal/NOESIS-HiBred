"""NOESIS-Σ Verifier Loss & Constraint Gate for Training.

Computes multi-objective training losses:
  L_total = L_LM + alpha * L_attractor + beta * L_verifier_smoothness

Enforces:
  1. Attractor Basin Energy Minimization (OSC convergence)
  2. Latent State Lipschitz / Smoothness Constraint (anti-hallucination bounds)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmaVerifierLoss(nn.Module):
    """Auxiliary training losses enforcing NOESIS-Σ verification constraints."""

    def __init__(
        self,
        alpha_attractor: float = 0.05,
        beta_smoothness: float = 0.01,
        max_energy_budget: float = 16.0,
    ) -> None:
        super().__init__()
        self.alpha = alpha_attractor
        self.beta = beta_smoothness
        self.max_budget = max_energy_budget

    def auxiliary_loss(
        self,
        osc_metrics: dict[str, torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Attractor + smoothness terms only.

        Long-context training computes cross-entropy chunk by chunk (to avoid ``[B, S, V]``
        logits), so it must be able to add the auxiliary terms without recomputing CE here.
        """
        energy_delta = osc_metrics.get("energy_delta")
        if energy_delta is None:
            attractor_loss = hidden_states.new_zeros(())
        else:
            attractor_loss = F.relu(energy_delta.float()).mean()
        if hidden_states.size(1) > 1:
            diff = hidden_states[:, 1:, :].float() - hidden_states[:, :-1, :].float()
            smoothness_penalty = torch.mean(torch.linalg.norm(diff, dim=-1))
        else:
            smoothness_penalty = hidden_states.new_zeros(())
        total = self.alpha * attractor_loss + self.beta * smoothness_penalty
        return total, {"loss_attractor": attractor_loss, "loss_smoothness": smoothness_penalty}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        osc_metrics: dict[str, torch.Tensor],
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        # 1. Primary Cross-Entropy Language Modeling Loss
        # logits: [B, T, V], targets: [B, T]
        lm_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100,
        )

        # 2. NOESIS Attractor Basin Loss (penalizes states diverging from stability)
        energy_delta = osc_metrics.get("energy_delta", torch.tensor(0.0, device=logits.device))
        attractor_loss = F.relu(energy_delta)  # Penalize energy increases during relaxation

        # 3. Verifier Smoothness Constraint (Lipschitz bound across sequence steps)
        # Bounded temporal variation prevents hallucination spikes
        diff = hidden_states[:, 1:, :] - hidden_states[:, :-1, :]
        smoothness_penalty = torch.mean(torch.norm(diff, dim=-1))

        # Combined Total Loss
        total_loss = (
            lm_loss
            + self.alpha * attractor_loss
            + self.beta * smoothness_penalty
        )

        loss_dict = {
            "loss_total": float(total_loss.detach().item()),
            "loss_lm": float(lm_loss.detach().item()),
            "loss_attractor": float(attractor_loss.detach().item()),
            "loss_smoothness": float(smoothness_penalty.detach().item()),
        }

        return total_loss, loss_dict
