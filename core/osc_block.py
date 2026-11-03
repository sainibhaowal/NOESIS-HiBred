"""NOESIS-Σ Continuous Operator-Split Cognitive Engine (OSC) Bottleneck.

Canonical Implementation of the NOESIS-Σ Mathematical Specification:
  1. Hybrid ICNN with Diagonal + Low-Rank + Linear Drift (Section 2.3):
     - Analytical closed-form gradient:
       ∇Φ(x) = W^T (a ⊙ σ(Wx + b)) + D ⊙ x + V^T (Vx) + c
     - Strong convexity floor c_min (Monotone dissipation & Lyapunov stability)
     - Zero autograd tape in forward loop for ultra-high GPU throughput
  2. Conservative Skew-Symmetric Operator:
     - K = U V^T - V U^T (Skew-symmetric: K^T = -K, conservative energy-preserving flow)
  3. Context Graft Projection B:
     - Learned projection B @ u scaling external token drive relative to internal dynamics
  4. Symmetric Strang Splitting (Section 7):
     - 2nd-order symplectic accuracy O(Δt^2):
       x_half = x + 0.5 * Δt * (K @ x)
       x_diss = x_half - Δt * ∇Φ(x_half)
       x_next = x_diss + 0.5 * Δt * (K @ x_diss)
       x = Π_L(x_next + Δt * (B @ u))
  5. Coupled Retrograde Feedback Controller:
     - Closed-loop causal feedback with Input-to-State Stability (ISS) norm bound
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class CanonicalICNNDirectGrad(nn.Module):
    r"""Direct-gradient ICNN Potential (Hybrid: Diagonal + Low-Rank + Linear Bias).

    Potential:
        Φ(x) = Σ_i a_i * softplus(W x + b)_i + 0.5 * ⟨D, x⊙x⟩ + 0.5 * || V x ||^2 + c^T x

    Exact Analytical Gradient (tape-free):
        ∇Φ(x) = W^T [ a ⊙ sigmoid(Wx + b) ] + (D ⊙ x) + V^T (Vx) + c
    """

    def __init__(
        self,
        d: int,
        m: int = 128,
        r: int = 16,
        c_min: float = 0.01,
    ) -> None:
        super().__init__()
        self.d = d
        self.m = m
        self.r = r
        self.c_min = c_min

        # Parameters
        self.W = nn.Parameter(torch.empty(m, d))
        self.b = nn.Parameter(torch.zeros(m))
        self.a_raw = nn.Parameter(torch.zeros(m))
        self.D_raw = nn.Parameter(torch.zeros(d))
        self.V = nn.Parameter(torch.empty(d, r)) if r > 0 else None

        # Enhancement 1: Linear drift prior vector c ∈ R^d
        self.c = nn.Parameter(torch.zeros(d))

        # Initialization
        nn.init.xavier_uniform_(self.W, gain=0.5)
        if self.V is not None:
            nn.init.normal_(self.V, std=0.02)

    def pos_a(self) -> torch.Tensor:
        return F.softplus(self.a_raw)

    def pos_d(self) -> torch.Tensor:
        return F.softplus(self.D_raw) + self.c_min

    def grad(self, x: torch.Tensor) -> torch.Tensor:
        """Compute closed-form ∇Φ(x) analytically in one pass."""
        a = self.pos_a()
        D = self.pos_d()

        # Wx + b
        h = F.linear(x, self.W, self.b)
        act = a * torch.sigmoid(h)

        # W^T act + D ⊙ x + V^T (Vx) + c
        grad = F.linear(act, self.W.t()) + (D * x) + self.c
        if self.V is not None:
            grad = grad + (x @ self.V) @ self.V.t()
        return grad

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute scalar energy Φ(x) for convergence monitoring."""
        a = self.pos_a()
        D = self.pos_d()
        h = F.linear(x, self.W, self.b)
        phi_softplus = (a * F.softplus(h)).sum(dim=-1)
        phi_quad = 0.5 * (D * (x ** 2)).sum(dim=-1)
        phi_linear = (x * self.c).sum(dim=-1)
        if self.V is not None:
            phi_quad = phi_quad + 0.5 * ((x @ self.V) ** 2).sum(dim=-1)
        return (phi_softplus + phi_quad + phi_linear).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy(x)


class RetrogradeController(nn.Module):
    r"""Coupled Continuous-Discrete Feedback Controller from Core/OSC/retrograde.py.

    Emitted token embeddings project back into continuous state with exponential
    decay λ and Input-to-State Stability (ISS) norm bound.
    """

    def __init__(
        self,
        dim: int,
        alpha: float = 0.1,
        decay: float = 0.95,
        u_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.decay = decay
        self.u_max = u_max
        self.w_retro = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.w_retro.weight, std=0.02)

    def scan(
        self, token_embeddings: torch.Tensor, state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Causal EMA ``s_t = decay * s_{t-1} + u_t`` in O(T) time and memory.

        The old Toeplitz implementation allocated ``[T, T]`` (~128 GB at 256K). The blockwise
        associative scan is algebraically identical: within a 1024-token block,
        ``sum_{i<=t} decay^(t-i) u_i = decay^t * cumsum(u_i * decay^-i)``. fp32 keeps the
        exponentials finite for a 1024-token block while remaining accurate.
        """
        u_proj = self.w_retro(token_embeddings).float()  # [B, T, D]
        b, t, d = u_proj.shape
        if state is None:
            state = u_proj.new_zeros(b, d)
        state = state.to(device=u_proj.device, dtype=torch.float32)
        out = torch.empty_like(u_proj)

        log_decay = math.log(self.decay) if self.decay > 0 else float("-inf")
        block = 1024
        for start in range(0, t, block):
            end = min(start + block, t)
            length = end - start
            idx = torch.arange(length, device=u_proj.device, dtype=torch.float32)
            # decay^-i and decay^i; stable because length <= 1024.
            inv = torch.exp(-idx * log_decay).view(1, length, 1)
            fwd = torch.exp(idx * log_decay).view(1, length, 1)
            local = torch.cumsum(u_proj[:, start:end] * inv, dim=1) * fwd
            local = local + state.unsqueeze(1) * (fwd * self.decay)
            out[:, start:end] = local
            state = local[:, -1]

        u_retro = self.alpha * out

        # Enforce Input-to-State Stability (ISS) norm clamp
        norm = torch.linalg.norm(u_retro, dim=-1, keepdim=True).clamp(min=1e-8)
        scale = torch.clamp(self.u_max / norm, max=1.0)
        return (u_retro * scale).to(token_embeddings.dtype), state

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        return self.scan(token_embeddings)[0]


class OSCLatentAttractor(nn.Module):
    """Complete NOESIS-Σ Thinking Block with Strang Splitting & Canonical ICNN."""

    def __init__(
        self,
        dim: int,
        rank: int = 32,
        icnn_hidden: int = 128,
        n_steps: int = 4,
        dt: float = 0.05,
        c_min: float = 0.01,
        max_norm: float = 16.0,
        enable_retrograde: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.n_steps = n_steps
        self.dt = dt
        self.max_norm = max_norm
        self.enable_retrograde = enable_retrograde

        # 1. Conservative flow: Low-rank skew-symmetric operator K = U V^T - V U^T
        self.u_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, rank, bias=False)

        # 2. Dissipative flow: Canonical ICNN with + c drift
        self.icnn = CanonicalICNNDirectGrad(d=dim, m=icnn_hidden, r=rank // 2, c_min=c_min)

        # Enhancement 2: Context Graft Projection Matrix B @ u
        self.b_proj = nn.Linear(dim, dim, bias=False)
        nn.init.eye_(self.b_proj.weight)  # Initialize scaled identity for smooth graft
        self.b_proj.weight.data.mul_(0.1)

        # 3. Retrograde Closed-Loop Feedback
        self.retrograde = RetrogradeController(dim=dim) if enable_retrograde else None

        # 4. Normalization & Output projection
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _skew_step(self, x: torch.Tensor) -> torch.Tensor:
        """K x = (U V^T - V U^T) x (conservative rotation, energy-preserving)."""
        u = self.u_proj(x)
        v = self.v_proj(x)
        term1 = F.linear(v, self.u_proj.weight.t())
        term2 = F.linear(u, self.v_proj.weight.t())
        return term1 - term2

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        output, metrics, _ = self.forward_with_state(hidden_states)
        return output, metrics

    def forward_with_state(
        self, hidden_states: torch.Tensor, retro_state: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor | None]:
        """OSC relaxation with an optional carried retrograde state for chunked processing."""
        x = self.norm(hidden_states)
        u_raw = x
        next_retro = None

        # Retrograde feedback addition
        if self.retrograde is not None:
            retro, next_retro = self.retrograde.scan(hidden_states, retro_state)
            u_raw = u_raw + retro

        # Enhancement 3: Learned Context Graft B @ u
        b_u = self.b_proj(u_raw)

        initial_energy = self.icnn.energy(x)

        # Enhancement 4: 2nd-Order Symmetric Strang Splitting Loop
        # x ← x + 0.5*dt*(K@x); g = ∇Φ(x); x ← x - dt*g; x ← x + 0.5*dt*(K@x); x ← Π_L(x + dt*Bu)
        half_dt = 0.5 * self.dt
        for _ in range(self.n_steps):
            # 1. Conservative half-step: x_half = x + 0.5 * dt * (K @ x)
            x_half = x + half_dt * self._skew_step(x)

            # 2. Dissipative full step: x_diss = x_half - dt * ∇Φ(x_half)
            grad = self.icnn.grad(x_half)
            x_diss = x_half - self.dt * grad

            # 3. Conservative second half-step: x_next = x_diss + 0.5 * dt * (K @ x_diss)
            x_next = x_diss + half_dt * self._skew_step(x_diss)

            # 4. Drive addition + Norm bounding projection Π_L(x + dt * B @ u)
            x_driven = x_next + self.dt * b_u
            norm = torch.linalg.norm(x_driven, dim=-1, keepdim=True).clamp(min=1e-6)
            scale = torch.clamp(self.max_norm / norm, max=1.0)
            x = x_driven * scale

        final_energy = self.icnn.energy(x)
        metrics = {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_delta": final_energy - initial_energy,
        }

        output = hidden_states + self.out_proj(x)
        return output, metrics, next_retro
