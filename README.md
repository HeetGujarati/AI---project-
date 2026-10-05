# LoRA+: Efficient Low Rank Adaptation of Large Models (RTX 4060 Implementation)

An empirical research implementation and benchmark of **LoRA+: Efficient Low Rank Adaptation of Large Models** ([Hayou, Ghosh, & Yu, ICML 2024](https://proceedings.mlr.press/v235/hayou24a.html)), based on the official Berkeley reference implementation ([nikhil-ghosh-berkeley/loraplus](https://github.com/nikhil-ghosh-berkeley/loraplus)).

Engineered specifically for **NVIDIA GeForce RTX 4060 (8GB VRAM)** under Windows using `Qwen/Qwen2.5-1.5B` and `distilgpt2`.

---

## 1. Official References & Citations
- **Official LoRA+ GitHub**: [https://github.com/nikhil-ghosh-berkeley/loraplus](https://github.com/nikhil-ghosh-berkeley/loraplus)
- **Official Reference Implementation**: [https://github.com/nikhil-ghosh-berkeley/loraplus/blob/main/lora_plus.py](https://github.com/nikhil-ghosh-berkeley/loraplus/blob/main/lora_plus.py)
- **ICML 2024 Paper**: [https://proceedings.mlr.press/v235/hayou24a.html](https://proceedings.mlr.press/v235/hayou24a.html)
- **ArXiv Preprint**: [https://arxiv.org/abs/2402.12354](https://arxiv.org/abs/2402.12354)

> *Attribution: Implementation inspired by and based on the official LoRA+ implementation by Soufiane Hayou, Nikhil Ghosh, and Bin Yu.*

---

## 2. Core Algorithm

### Standard LoRA:
$$W' = W + \Delta W = W + \frac{\alpha}{r} B A$$
where $W$ is frozen, $A \sim \mathcal{N}(0, \sigma^2)$, and $B = 0$.
In standard LoRA:
$$\eta_A = \eta_B = \eta$$

### LoRA+:
LoRA+ corrects the asymptotic imbalance between matrices $A$ and $B$ in large models by setting:
$$\eta_A = \eta$$
$$\eta_B = \lambda \times \eta_A \quad (\lambda = \text{loraplus\_lr\_ratio} \ge 1)$$

Example:
$$\eta_A = 10^{-4}, \quad \lambda = 16 \implies \eta_B = 1.6 \times 10^{-3}$$

---

## 3. Comparison: LoRA+ Paper vs Our Experiment

| Property | LoRA+ Paper (Hayou et al., 2024) | Our Experiment |
|---|---|---|
| **Venue** | ICML 2024 | Independent Research Reproduction |
| **Base Models** | RoBERTa-base, RoBERTa-large, GPT-2, LLaMA-7B | Qwen/Qwen2.5-1.5B (Fallback: distilgpt2) |
| **Hardware** | Large Compute Cluster (A100/H100) | NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) |
| **Dataset** | GLUE Benchmark (MNLI, SST-2), GSM8k | Alpaca-Cleaned Instruction Dataset |
| **LoRA Rank ($r$)** | 8 | 8 |
| **LoRA Alpha ($\alpha$)** | 16 | 16 |
| **LR Ratios Evaluated** | 2, 4, 8, 16, 32, 64 | 1 (Standard LoRA), 4, 8, 16, 32 |
| **Precision** | FP16 / BF16 | FP16 Mixed Precision + Gradient Checkpointing |
| **Evaluation** | Task Accuracy & Perplexity | Validation Loss, Perplexity, Peak VRAM, Step Timing |

---

## 4. Hardware Optimization for RTX 4060 (8GB)
- **Precision**: FP16 Mixed Precision
- **Batch Size**: 1 per device
- **Gradient Accumulation**: 8 steps (Effective Batch Size = 8)
- **Sequence Length**: 512 tokens with prompt masking
- **Gradient Checkpointing**: Enabled
- **VRAM Footprint**: ~3.5 – 4.5 GB peak (well below 8GB ceiling)

---

## 5. Quick Start & Reproduction

### Smoke Test (Phase 6 Verification)
```powershell
python scripts/train.py --method loraplus --ratio 16 --max_train_samples 50 --max_eval_samples 20
```

### Run Master Benchmark Suite (5 Experiments)
```powershell
python scripts/run_experiments.py
```
This automatically runs:
1. Standard LoRA ($\lambda=1$)
2. LoRA+ ($\lambda=4$)
3. LoRA+ ($\lambda=8$)
4. LoRA+ ($\lambda=16$)
5. LoRA+ ($\lambda=32$)
and generates all 8 publication plots and the research report.

### Fallback Model (distilgpt2)
```powershell
python scripts/train.py --model distilgpt2 --method loraplus --ratio 16
```

### Interactive Inference
```powershell
python scripts/inference.py --model Qwen/Qwen2.5-1.5B --adapter experiments/checkpoints/loraplus_16
```

### Generate 5-Page IEEE Research Paper & Report Suite
```powershell
python scripts/generate_ieee_report.py
```
This automatically compiles:
- `experiments/results/ieee_report_5pages.pdf`: Publication-ready 5-page IEEE two-column paper (compiled with embedded high-res experiment plots and hardware telemetry).
- `experiments/results/ieee_report.tex`: Complete IEEEtran conference LaTeX manuscript ready for submission / Overleaf.
- `experiments/results/ieee_report.html`: Two-column IEEE formatted print layout.
- `experiments/results/final_report.md`: Comprehensive 5-page Markdown research report.

**Authors**: Heet Gujarati (202451069), Yash Jagani (202451077), Brahmesh Italiya (202451038)  
**Affiliation**: Department of Computer Science and Engineering, Indian Institute of Information Technology, Vadodara, India

### Launch Interactive Research Dashboard
```powershell
python scripts/run_dashboard.py
```
Serves the dark-themed Apache ECharts interactive research dashboard on `http://localhost:8080/` with live telemetry, loss curves, and hardware metrics.

---

## 6. Project Layout
```
lora-plus-project/
├── config/             # Base, LoRA, LoRA+, and Master experiment YAMLs
├── src/                # Modular implementation (optimizer, dataset, model, trainer, viz)
├── scripts/            # Training, evaluation, automated runner, and inference CLIs
├── experiments/        # Results (JSON), logs, checkpoints, and generated plots
└── tests/              # Pytest unit test suite
```
