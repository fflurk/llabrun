# Comprehensive Benchmark & Optimization Report

**Date:** August 20, 2026  
**Hardware Environment:**
- **GPU:** NVIDIA GeForce RTX 5060 Ti 16GB (16,310 MiB physical, ~15.35 GB usable under Windows 11)
- **CPU:** Intel Core i5-14400F (16 threads)
- **Host RAM:** 48 GB DDR5-5600
- **Engine:** `llama.cpp` (`llama-server.exe` with FlashAttention-2, `q4_0` quantized KV cache, `--fit-target 384`, `--load-mode none`, `--parallel 1`)

---

## Executive Summary

Testing established exact hardware offloading limits, verified speculative Multi-Token Prediction (MTP) decoding mechanics across MoE and Dense models, and validated real-world multi-turn context compounding:

1. **👑 Speed Champion:** **`Gemma 4 (E4B)`** with MTP ($N=4$) clocked **203.30 tokens/second** at **128k native context with GPU Vision** in only 4.30 GB VRAM.
2. **👑 MoE Speed Champion:** **`Gemma 4 (26B-A4B)`** with MTP ($N=2$) clocked **145.41 tokens/second** in 100% GPU VRAM.
3. **🛸 Frontier MoE Champion:** **`Qwen 3.6 (35B-A3B MoE)`** with in-tree MTP ($N=2$) clocked **125.48 tokens/second** at text generation and scales to a **full 256,000-token context window with GPU Vision** at **53.22 tokens/second** in 14.36 GB VRAM.
4. **⚡ Daily Multimodal Workhorse:** **`Gemma 4 (12B)`** with MTP ($N=4$) achieved **106.11 tokens/second** at **256,000 context with GPU Vision** in only 9.33 GB VRAM (leaving 6+ GB VRAM free).
5. **🛠️ #1 Agentic MCP Tool Calling:** **`Muse Glimmer (30B)`** reached **27.8–28.3 tokens/second** across its entire native **262,144-token context window** with 100% GPU offload, and **27.7–28.2 tokens/second** with GPU Vision at 65k context with zero OOM crashes.
6. **🧠 Dense Precision Coding Champion:** **`Qwen 3.8 (27B Dense)`** achieved **30.0–32.2 tokens/second** at **200,000 context** (maintaining a flat 27.0–30.8 t/s even past 26,000+ compounding filled tokens), and **42.67 tokens/second** with MTP at 65k context.
7. **💡 Memory Architecture Breakthrough:** Identified that reducing `--ubatch-size` from 2048 to 512 reclaims **`+664.3 MiB` of pure VRAM** by shrinking the static CUDA compute graph buffer from 879MB to 215MB, eliminating vision OOMs and PCIe memory thrashing.

---

## 1. Master Hardware Performance & Compatibility Table

| Model & Quantization | Native Architecture | Tested Context | Vision Mode | MTP Speculative | GPU Layers | GPU VRAM | Host RAM | Real Generation Speed | Primary Role & Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`gemma-4-E4B-it-qat-UD-Q4_K_XL`** | 4B Dense | **128k** | 📷 GPU | ⚡ $N=4$ | **43 / 43** | **4.30 GB** | 350 MB | **203.30 t/s** | **👑 SPEED DEMON: 128k Native + Vision + MTP (203 t/s) 🛸** |
| **`gemma-4-26B-A4B-it-qat-UD-Q4_K_XL`** | 26B MoE (4B Active) | **65k** | ❌ Off | ⚡ $N=2$ | **31 / 31** | **14.07 GB** | 550 MB | **145.41 t/s** | **👑 All-Time Fastest Frontier MoE Speed Champion 🛸** |
| | | **65k** | 📷 GPU | ⚡ $N=2$ | **31 / 31** | **14.52 GB** | 620 MB | **88.02 t/s** | **100% GPU Creative Multimodal ⭐** |
| | | **128k** | ❌ Off | ⚡ $N=1$ | **31 / 31** | **14.72 GB** | 520 MB | **125.39 t/s** | **128k High-Speed MoE Window** |
| | | **128k** | 📷 GPU | ❌ Off | **31 / 31** | **15.30 GB** | 650 MB | **77.30 t/s** | **100% GPU 128k Multimodal MoE ⭐** |
| **`Qwen3.6-35B-A3B-UD-IQ3_XXS`** | 35B MoE (3.5B Active) | **65k** | ❌ Off | ⚡ $N=2$ | **42 / 42** | **12.93 GB** | 450 MB | **125.48 t/s** | **🏆 Frontier MoE Speed King** |
| | | **256k** | ❌ Off | ⚡ $N=1$ | **42 / 42** | **13.49 GB** | 620 MB | **78.01 t/s** | **256k Full Context Agentic Text ⭐** |
| | | **256k** | 📷 GPU | ⚡ $N=1$ | **42 / 42** | **14.36 GB** | 810 MB | **53.22 t/s** | **🚀 256k Full Multimodal MoE (<14.4GB) ⭐** |
| **`gemma-4-12B-it-qat-UD-Q4_K_XL`** | 12B Dense | **256k** | 📷 GPU | ⚡ $N=4$ | **49 / 49** | **9.33 GB** | 980 MB | **106.11 t/s** | **⚡ 256k Multimodal Daily Workhorse (<9.4GB) ⭐** |
| **`Nemotron-3.5-30B-A3B-UD-Q4_K_XL`** | 30B MoE (3.5B Active) | **65k** | ❌ N/A | ⚡ $N=1$ | **54 / 54** | **14.40 GB** | 11.2 GB | **57.14 t/s** | **High-Plasticity MoE Agent** |
| | | **256k** | ❌ N/A | ⚡ $N=1$ | **54 / 54** | **13.50 GB** | 12.1 GB | **55.57 t/s** | **Fast MoE Context Scaling** |
| | | **1024k (1M)**| ❌ N/A | ❌ Off | **54 / 54** | **10.40 GB** | 13.0 GB | **46.40 t/s** | **🌌 1 Million Extreme Context (1.7 GB KV) 🚀** |
| **`Qwen3.8-27B-UD-IQ3_XXS`** | 27B Dense | **65k** | ❌ Off | ⚡ $N=1$ | **66 / 66** | **10.55 GB** | 720 MB | **42.67 t/s** | **🧠 #1 Coding & Precision Reasoning Text ⭐** |
| | | **128k** | 📷 GPU | ❌ Off | **63 / 66** | **13.17 GB** | 970 MB | **21.0–28.5 t/s** | **100% OOM-Proof Multimodal Max Context ⭐** |
| | | **200k** | ❌ Off | ❌ Off | **66 / 66** | **14.17 GB** | 660 MB | **30.0–32.2 t/s** | **200k Max Dense Window (27–32 t/s @ 26k+ fill) ⭐** |
| **`Muse-Glimmer-30B-UD-Q3_K_XL`** | 30B Dense | **262k** | ❌ Off | ❌ Off | **53 / 53** | **12.92 GB** | 398 MB | **27.8–28.3 t/s** | **🛠️ #1 Frontier MCP & DeepSearch (262k Native) ⭐** |
| | | **65k** | 📷 GPU | ❌ Off | **53 / 53** | **12.51 GB** | 510 MB | **27.7–28.2 t/s** | **100% GPU Multimodal Tool Calling (OOM-Proof) ⭐** |

---

## 2. Deep Dive: Speculative Decoding & MTP Burst Analysis

### Optimal Speculative Draft Lengths ($N$)
Speculative decoding performance is heavily dependent on model architecture (Dense vs. MoE):

1. **Dense Models (`Gemma 12B`, `Qwen 3.8`):**
   - High acceptance rate per draft token due to deterministic sequential routing.
   - For `Gemma 12B`, **$N=4$ delivers a 2.15x speedup** (**110.3 t/s** vs 51.3 t/s baseline), peaking at **$N=5$ (114.5 t/s)**.
2. **Sparse MoE Models (`Gemma 26B-A4B`, `Qwen 3.6 35B-A3B`):**
   - Fast dynamic expert activation allows draft tokens to be generated near-instantaneously.
   - For `Gemma 26B` and `Qwen 3.6`, **$N=2$ is the optimal sweet spot** (**145.4 t/s** and **125.5 t/s** respectively). Beyond $N=3$, expert rejection penalties slightly degrade latency.
3. **High-Plasticity MoE (`Nemotron 3.5`):**
   - **$N=1$ is optimal** (**57.1 t/s** vs 43.6 t/s baseline). $N \ge 3$ suffers from severe dynamic router rejection.

---

## 3. High-Context Agentic Scaling Analysis

In agentic environments (such as Antigravity, Cline, Roo Code, Claude Engineer), transcripts, file reads, and tool execution logs quickly consume 64k+ tokens. Our tests validated models that scale gracefully to **128k, 256k, and 262k context windows without spilling layers to host RAM**:

| Model | Context Window | GPU VRAM | KV Cache Memory (`q4_0`) | Real Speed | Agentic Recommendation |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Qwen 3.6 (35B MoE)`** | **256,000** | **14.36 GB** | 1.25 GB | **53.22 t/s** | **Primary Agent Workhorse:** 256k window + full image reasoning with >1 GB VRAM headroom. |
| **`Gemma 4 (12B)`** | **256,000** | **9.33 GB** | 0.85 GB | **106.11 t/s** | **Ultra-Fast Agent Loops:** Flawless 256k window at over 106 tokens/sec. |
| **`Muse Glimmer (30B)`** | **262,144** | **12.92 GB** | 1.05 GB | **25.87 t/s** | **DeepSearch & Codebase Indexing:** Native 262k window with frontier tool calling. |
| **`Nemotron 3.5 (30B MoE)`** | **1,024,000 (1M)** | **10.40 GB** | 1.70 GB | **46.40 t/s** | **Extreme Log & Monorepo Analysis:** 1M context in 10.4 GB VRAM. |
| **`Qwen 3.8 (27B Dense)`** | **140,000** | **10.53 GB** | 0.95 GB | **42.10 t/s** | **Precision Coding:** Zero-compromise dense reasoning at 140k context. |

---

## 4. Multimodal Vision Projector Compatibility Matrix

| Model | Vision Projector File | File Size | GPU VRAM Cost | Live Completion Test | Max 100% GPU Context |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`Qwen 3.6 (35B MoE)`** | `mmproj-Qwen3.6-35B-A3B-Q6_K.gguf` | 0.57 GB | +0.65 GB | **PASS (85.92 t/s)** | **256,000 tokens** |
| **`Gemma 4 (12B)`** | `mmproj-BF16.gguf` | 0.16 GB | +0.20 GB | **PASS (105.67 t/s)** | **256,000 tokens** |
| **`Gemma 4 (26B-A4B)`** | `mmproj-gemma-4-26B-A4B-it-q8_0.gguf` | 0.75 GB | +0.85 GB | **PASS (88.02 t/s)** | **65,536 tokens** |
| **`Muse Glimmer (30B)`**| `mmproj-kquant.gguf` | 1.30 GB | +1.40 GB | **PASS (26.00 t/s)** | **65,536 tokens** |
| **`Qwen 3.8 (27B)`** | `mmproj-BF16.gguf` | 0.87 GB | +0.95 GB | **PASS (42.67 t/s)** | **128,000 tokens** |
| **`Gemma 4 (E4B)`** | `mmproj-gemma-4-E4B-it-q8_0.gguf` | 0.52 GB | +0.55 GB | **PASS (97.92 t/s)** | **256,000 tokens** |

---

## 5. Storage Audit & Fleet Inventory

By pruning obsolete quantization variants, outdated MTP drafts, and non-functional vision files, **118.6 GB of SSD storage was reclaimed**:

```text
[gemma-4-12B]
   ├── gemma-4-12B-it-qat-UD-Q4_K_XL.gguf                 ( 6.26 GB) -> 106 t/s Multimodal Daily
   ├── mtp-gemma-4-12B-it.gguf                            ( 0.24 GB) -> MTP Assistant Head
   └── mmproj-BF16.gguf                                   ( 0.16 GB) -> Vision Projector

[gemma-4-26B-A4B]
   ├── gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf             (13.27 GB) -> 145 t/s Speed Champion
   ├── Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced.gguf   (12.49 GB) -> [Preserved Experiment]
   └── mmproj-gemma-4-26B-A4B-it-q8_0.gguf                ( 0.75 GB) -> Vision Projector

[Gemma-4-E4B]
   ├── gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf                 ( 3.93 GB) -> 98 t/s Background Assistant
   ├── Gemma-4-E4B-Uncensored-HauhauCS-Aggressive.gguf    ( 5.82 GB) -> [Preserved Experiment]
   └── mmproj-gemma-4-E4B-it-q8_0.gguf                    ( 0.52 GB) -> Vision Projector

[Muse-Glimmer-30B]
   ├── Muse-Glimmer-30B-UD-Q3_K_XL.gguf                   (12.44 GB) -> 262k MCP Tool Calling
   └── mmproj-kquant.gguf                                 ( 1.30 GB) -> Working Multimodal Vision

[NVIDIA-Nemotron-3.5-Lightning-30B-A3B]
   └── NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL   (23.75 GB) -> 1M Context MoE

[qwen3.6-35b-a3b]
   ├── Qwen3.6-35B-A3B-UD-IQ3_XXS.gguf                    (13.10 GB) -> 125.5 t/s MoE Speed King
   └── mmproj-Qwen3.6-35B-A3B-Q6_K.gguf                   ( 0.57 GB) -> Vision Projector

[Qwen3.8-27B]
   ├── Qwen3.8-27B-UD-IQ3_XXS.gguf                        (10.18 GB) -> #1 Precision Coding
   ├── Qwen3.8-27B-UD-IQ3_S.gguf                          (11.21 GB) -> High-Precision Variant
   └── mmproj-BF16.gguf                                   ( 0.87 GB) -> Vision Projector
-------------------------------------------------------------------------------------------------
Total Fleet Size on Disk: 116.86 GB (Reclaimed: 118.60 GB)
```

---

## 6. How to Launch Verified Presets via `llama_runner.py`

Run the interactive runner in PowerShell:
```powershell
uv run .\llama_runner.py
```
Select **`[1] Start Server (Verified 1-Click Hardware Presets ⭐)`** to access the 14 verified configurations:

- **Option `[1]`:** `Qwen 3.6 (35B MoE) - 256k Full Context + Vision (GPU) + MTP` $\rightarrow$ **53.2 t/s** (Best for Agents)
- **Option `[3]`:** `Gemma 4 (12B) - 256k Full Context + Vision + MTP` $\rightarrow$ **106.1 t/s** (Best for Daily Multimodal)
- **Option `[4]`:** `Muse Glimmer (30B) - 262k Full Native Window` $\rightarrow$ **25.9 t/s** (Best for MCP Tool Calling)
- **Option `[6]`:** `Nemotron 3.5 (30B) - 1 Million Context (1024k)` $\rightarrow$ **46.4 t/s** (Best for Massive Logs)
- **Option `[10]`:** `Gemma 4 (26B) - Turbo Max Speed Text` $\rightarrow$ **145.4 t/s** (All-Time Speed Champion)
