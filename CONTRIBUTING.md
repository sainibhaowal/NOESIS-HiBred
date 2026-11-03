# Contributing to NOESIS-Σ

Thank you for your interest in contributing to **NOESIS-Σ**! We welcome contributions from researchers, engineers, and community members.

---

## Code of Conduct & Principles
- **Documentation Integrity**: Preserve existing docstrings and comments.
- **Verification First**: Every architectural modification or kernel change must pass the test suite (`pytest`) with 100% pass rate.
- **Mathematical Rigor**: When touching continuous dynamical blocks (`core/osc_block.py`, `core/verifier_loss.py`), ensure Lyapunov stability bounds, energy conservation invariants ($\langle x, Kx \rangle = 0$), and Input-to-State Stability (ISS) invariants remain mathematically sound.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<OWNER>/<REPO>.git
   cd noesis-sigma
   ```

2. **Create a virtual environment & install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r core/requirements.txt pytest pytest-cov
   ```

3. **Run the full test suite**:
   ```bash
   pytest tests -v
   ```

4. **Verify the complete standalone system**:
   ```bash
   python3 core/verify_noesis_sigma.py
   ```

---

## Submitting Pull Requests

1. Create a feature branch: `git checkout -b feature/your-feature-name`.
2. Ensure all tests pass: `pytest -q`.
3. Keep pull requests focused on a single capability or optimization.
4. Open a Pull Request on GitHub with a clear description of changes and test results.

---

## License
By contributing to NOESIS-Σ, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

