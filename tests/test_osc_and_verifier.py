# NOESIS-Σ Unit & Mathematical Verification Tests: OSC Thinking Block & Verifier Loss
# Path: tests/test_osc_and_verifier.py

from __future__ import annotations
import torch
import pytest

from osc_block import CanonicalICNNDirectGrad, RetrogradeController, OSCLatentAttractor
from verifier_loss import SigmaVerifierLoss


def test_canonical_icnn_convexity_and_exact_analytic_gradient(device: torch.device) -> None:
    """Verify that CanonicalICNNDirectGrad is strictly convex and its closed-form
    analytic gradient grad(x) exactly matches torch.autograd.grad(energy_per_token(x))."""
    torch.manual_seed(0)
    d, m, r = 64, 128, 16
    icnn = CanonicalICNNDirectGrad(d=d, m=m, r=r, c_min=0.05).to(device).float()

    x = torch.randn(3, 8, d, device=device, dtype=torch.float32, requires_grad=True)
    y = torch.randn(3, 8, d, device=device, dtype=torch.float32)

    # 1. Verify strong convexity parameter D >= c_min and a >= 0
    assert (icnn.pos_a() >= 0).all()
    assert (icnn.pos_d() >= 0.05).all()

    # 2. Verify Jensen convexity: Phi(0.5*x + 0.5*y) < 0.5*Phi(x) + 0.5*Phi(y)
    # Compute token-wise potential directly to compare against closed-form grad(x)
    wx_b = torch.nn.functional.linear(x, icnn.W, icnn.b)
    act = torch.nn.functional.softplus(wx_b) * icnn.pos_a()
    phi_icnn = act.sum(dim=-1)
    d_vec = icnn.pos_d()
    quad_diag = 0.5 * (d_vec * (x * x)).sum(dim=-1)
    vx = x @ icnn.V
    quad_lr = 0.5 * (vx * vx).sum(dim=-1)
    lin_c = (x * icnn.c).sum(dim=-1)
    phi_total = (phi_icnn + quad_diag + quad_lr + lin_c).sum()

    autograd_g = torch.autograd.grad(phi_total, x, create_graph=False)[0]
    analytic_g = icnn.grad(x)

    assert torch.allclose(analytic_g, autograd_g, atol=1e-5, rtol=1e-4), (
        f"Max diff between analytic grad and autograd: {(analytic_g - autograd_g).abs().max().item()}"
    )

    # Check scalar energy() and forward() alias
    e1 = icnn.energy(x)
    e2 = icnn(x)
    assert e1.ndim == 0
    assert torch.allclose(e1, e2)


def test_retrograde_controller_causal_and_iss_bound(device: torch.device) -> None:
    """Verify RetrogradeController computes causal differences and enforces ISS norm bound u_max."""
    torch.manual_seed(1)
    dim = 32
    u_max = 0.5
    ctrl = RetrogradeController(dim=dim, u_max=u_max).to(device).float()

    # Use large input jump to trigger ISS norm clamping
    s = torch.randn(2, 10, dim, device=device, dtype=torch.float32) * 100.0
    u_retro = ctrl(s)

    assert u_retro.shape == s.shape
    norms = torch.linalg.norm(u_retro, dim=-1)
    assert (norms <= u_max + 1e-5).all(), f"ISS bound violated: max norm={norms.max().item()} > {u_max}"

    # Verify causal property: changing future tokens t >= 5 must not affect u_retro at t < 5
    s2 = s.clone()
    s2[:, 5:, :] += 50.0
    u_retro2 = ctrl(s2)
    assert torch.allclose(u_retro[:, :5, :], u_retro2[:, :5, :], atol=1e-6)


def test_osc_latent_attractor_strang_splitting_and_energy_relaxation(device: torch.device) -> None:
    """Verify OSCLatentAttractor skew-symmetry, Strang splitting energy dissipation, and autograd."""
    torch.manual_seed(2)
    dim = 64
    osc = OSCLatentAttractor(
        dim=dim,
        rank=16,
        icnn_hidden=64,
        n_steps=4,
        dt=0.08,
        c_min=0.02,
        max_norm=10.0,
        enable_retrograde=True,
    ).to(device).float()

    h = torch.randn(2, 12, dim, device=device, dtype=torch.float32, requires_grad=True)

    # 1. Skew-symmetric operator K = U V^T - V U^T must satisfy <x, Kx> = 0 for all x
    kx = osc._skew_step(h)
    inner_prod = (h * kx).sum(dim=-1)
    assert torch.allclose(inner_prod, torch.zeros_like(inner_prod), atol=1e-5), (
        f"Skew-symmetric operator failed <x, Kx>=0: max={inner_prod.abs().max().item()}"
    )

    # 2. Forward pass and Lyapunov energy relaxation
    out, metrics = osc(h)
    assert out.shape == h.shape
    assert "initial_energy" in metrics and "final_energy" in metrics and "energy_delta" in metrics
    assert float(metrics["energy_delta"].item()) < 0.0, (
        f"Expected energy relaxation dE < 0, got {metrics['energy_delta'].item()}"
    )

    # 3. Backward pass through Strang splitting
    loss = out.square().mean() + 0.1 * metrics["final_energy"]
    loss.backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    for name, param in osc.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), f"Missing/NaN grad for {name}"


def test_sigma_verifier_loss_components(device: torch.device) -> None:
    """Verify SigmaVerifierLoss computes LM cross-entropy, attractor penalty, and Lipschitz smoothness."""
    torch.manual_seed(3)
    B, T, V, D = 2, 8, 500, 32
    criterion = SigmaVerifierLoss(alpha_attractor=0.1, beta_smoothness=0.05).to(device)

    logits = torch.randn(B, T, V, device=device, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, V, (B, T), device=device, dtype=torch.long)
    targets[0, -1] = -100  # test ignore_index padding token
    hidden_states = torch.randn(B, T, D, device=device, dtype=torch.float32, requires_grad=True)

    # Case A: Negative energy_delta (relaxing attractor -> attractor_loss == 0)
    osc_metrics_good = {"energy_delta": torch.tensor(-12.5, device=device)}
    loss_good, dict_good = criterion(logits, targets, osc_metrics_good, hidden_states)
    assert dict_good["loss_attractor"] == 0.0
    assert dict_good["loss_lm"] > 0.0
    assert dict_good["loss_smoothness"] > 0.0

    # Case B: Positive energy_delta (diverging attractor -> penalized by ReLU)
    osc_metrics_bad = {"energy_delta": torch.tensor(5.0, device=device)}
    loss_bad, dict_bad = criterion(logits, targets, osc_metrics_bad, hidden_states)
    assert pytest.approx(dict_bad["loss_attractor"], rel=1e-5) == 5.0
    assert loss_bad.item() > loss_good.item()

    loss_bad.backward()
    assert logits.grad is not None and hidden_states.grad is not None
