# NOESIS-Σ: Generated Ablation Harness Output

> **Development output only.** These values were produced on the small example
> corpus with 50 steps. They validate the harness and are not benchmark,
> generalization, production-performance, or publication results.

**Corpus**: `examples/pretraining_corpus.txt` | **Steps**: `50` | **Optimization**: AdamW (`lr=0.001`)

## 1. Markdown Table

### Exploratory Iso-Compute Harness Output

| Architecture Variant | Params | Initial Loss | Final Loss (↓) | Perplexity (↓) | KV Cache (MB) | Tok/s (↑) | ||∇Φ(x)|| Residual | Attractor ΔEnergy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Baseline** | 2.84M | 7.049 | **2.861** | **17.47** | 0.125 MB | 117.1 | N/A | N/A |
| **2. + OSC Only** | 3.09M | 7.172 | **1.851** | **6.36** | 0.125 MB | 87.6 | 14.9269 | -20.1874 |
| **3. + OSC + MLA** | 3.27M | 7.147 | **2.035** | **7.65** | 0.098 MB | 85.7 | 16.5555 | -21.1321 |
| **4. + OSC + MLA + HC** | 3.36M | 7.070 | **2.109** | **8.24** | 0.098 MB | 92.5 | 15.0935 | -25.1967 |
| **5. Full NOESIS-Σ** | 14.66M | 7.127 | **1.351** | **3.86** | 0.098 MB | 76.1 | 14.1528 | -18.5236 |

## 2. LaTeX Table Code

```latex
\begin{table*}[t]
\centering
\caption{Ablation study evaluating systematic contributions of NOESIS-$\Sigma$ components under strict Iso-Compute conditions.}
\label{tab:noesis_ablation}
\begin{tabular}{lcccccccc}
\toprule
\textbf{Architecture Variant} & \textbf{Params} & \textbf{Init Loss} & \textbf{Final Loss ($\downarrow$)} & \textbf{PPL ($\downarrow$)} & \textbf{KV (MB)} & \textbf{Tok/s} & \textbf{$\|\nabla\Phi(x)\|$} & \textbf{$\Delta E$} \\
\midrule
  1. Baseline & 2.84M & 7.05 & \textbf{2.86} & \textbf{17.47} & 0.125 & 117.1 & -- & -- \\
  2. + OSC Only & 3.09M & 7.17 & \textbf{1.85} & \textbf{6.36} & 0.125 & 87.6 & 14.927 & -20.187 \\
  3. + OSC + MLA & 3.27M & 7.15 & \textbf{2.03} & \textbf{7.65} & 0.098 & 85.7 & 16.555 & -21.132 \\
  4. + OSC + MLA + HC & 3.36M & 7.07 & \textbf{2.11} & \textbf{8.24} & 0.098 & 92.5 & 15.094 & -25.197 \\
  5. Full NOESIS-Σ & 14.66M & 7.13 & \textbf{1.35} & \textbf{3.86} & 0.098 & 76.1 & 14.153 & -18.524 \\
\bottomrule
\end{tabular}
\end{table*}
```
