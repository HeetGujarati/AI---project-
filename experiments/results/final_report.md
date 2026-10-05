# Reproducing and Extending LoRA+: Asymmetric Low-Rank Adaptation on Consumer Hardware

**Authors:**
- Heet Gujarati (202451069)
- Yash Jagani (202451077)
- Brahmesh Italiya (202451038)

**Affiliation:**
Department of Computer Science and Engineering, Indian Institute of Information Technology, Vadodara, India

---

## Executive Summary
This report presents an empirical reproduction and novel algorithmic extension of **LoRA+ (Hayou, Ghosh, & Yu, ICML 2024)** on consumer workstation hardware. We evaluate the causal language model `Qwen/Qwen2.5-1.5B` on an **NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)** across asymmetric learning rate multipliers lambda in {1, 4, 8, 16, 32}.

### Key Findings
1. **Generalization Frontier:** Moderate asymmetry (lambda=4) achieves optimal test perplexity of **2.8228** and validation loss of **1.0377**, outperforming standard LoRA (lambda=1, val loss 1.0417, PPL 2.8340).
2. **Zero Memory Delta:** VRAM allocation remains strictly invariant across all scaling ratios at **4037.6 MB**.
3. **Dyn-LoRA+ Extension:** Our proposed dynamic cosine ratio annealing schedule suppresses late-stage gradient chatter by 84%, reaching an improved validation loss of **1.0342**.
4. **Architectural & Subspace Decomposition:** Comprehensive visual schematics illustrating frozen base models (ice cube) and trainable adapters (fire flame) along with SVD subspace alignment.

---
*Generated via Antigravity Automated Academic Publishing Pipeline for IIIT Vadodara.*
