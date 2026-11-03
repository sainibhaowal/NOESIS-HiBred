# Third-Party Notices & Open Source Attribution

NOESIS-Σ is licensed under the Apache License 2.0 (see `LICENSE`). It adapts code and
architectural primitives from the following third-party projects, whose licenses are reproduced below.

---

## 1. DeepSeek-V3 / DeepSeek-V3.2 Inference Code

- **Source:** https://github.com/deepseek-ai/DeepSeek-V3
- **License:** MIT License
- **Adapted in:** `core/model.py`, `core/kernel.py`, `core/convert.py`, `core/generate.py`
  (Multi-Head Latent Attention, MoE gating/experts, parallel embedding/head,
  FP8 kernels, checkpoint conversion). These files have been modified by the NOESIS-Σ contributors.

Full license text: see **Appendix A** below (verbatim from upstream `LICENSE-CODE`).

---

## 2. Original NOESIS-Σ Components

The following components were developed by the NOESIS-Σ contributors and are released under Apache 2.0:

- **Operator-Split Continuous (OSC) Latent Attractor** (`core/osc_block.py`): 2nd-order symmetric
  Strang splitting of a skew-symmetric conservative flow and a dissipative gradient flow −∇Φ(x).
- **Direct-gradient Input-Convex Neural Network (ICNN)** potential Φ(x) with closed-form ∇Φ.
- **Retrograde controller** with input-to-state-stability (ISS) norm bound.
- **SigmaVerifierLoss** (`core/verifier_loss.py`): LM loss + attractor energy (ReLU(ΔΦ)) + temporal smoothness.

---

## 3. Runtime Dependencies

Installed separately via `core/requirements.txt` and not redistributed in this repository.
Their licenses apply as published by each project:

| Project | License | License text |
|---|---|---|
| PyTorch | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| Transformers | Apache-2.0 | https://github.com/huggingface/transformers/blob/main/LICENSE |
| Datasets | Apache-2.0 | https://github.com/huggingface/datasets/blob/main/LICENSE |
| Tokenizers | Apache-2.0 | https://github.com/huggingface/tokenizers/blob/main/LICENSE |
| Safetensors | Apache-2.0 | https://github.com/huggingface/safetensors/blob/main/LICENSE |
| bitsandbytes | MIT | https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/LICENSE |
| NumPy | BSD-3-Clause | https://github.com/numpy/numpy/blob/main/LICENSE.txt |
| SymPy | BSD-3-Clause | https://github.com/sympy/sympy/blob/master/LICENSE |
| Pillow | MIT-CMU | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| TileLang | MIT | https://github.com/tile-ai/tilelang/blob/main/LICENSE |
| tqdm | MPL-2.0 / MIT | https://github.com/tqdm/tqdm/blob/master/LICENCE |

## 4. Datasets

Pretraining streams datasets from the Hugging Face Hub at runtime; no dataset content is included in this
repository. Users are responsible for complying with each dataset's license (FineWeb-Edu, Cosmopedia-v2,
codeparrot/github-code, GSM8K, Glaive Function Calling v2).

---

# Appendix A — DeepSeek-V3 License (verbatim)

The MIT License requires its text to accompany copies of the adapted code, so it is reproduced in full.

## DeepSeek-V3 `LICENSE-CODE`

Source: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-CODE

````text
MIT License

Copyright (c) 2023 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
````
