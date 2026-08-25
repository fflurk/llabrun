# llabrun — Llama.cpp Lab Builder & Runner

A production-tuned local LLM orchestration, building, and benchmarking suite built on `llama.cpp`. Designed for deterministic layer offloading, micro-batch memory optimization, direct RAM loading, speculative decoding (MTP), and instant portability across any hardware (NVIDIA, Intel Arc, AMD) via declarative model presets.

---

## 🚀 Key Features

*   **Zero-Overhead Runner CLI ([llama_runner.py](llama_runner.py)):** Interactive model selector, benchmark orchestrator, and Router INI preset generator.
*   **Modular Presets Configuration ([presets.json](presets.json)):** Declarative favorite models and verified presets decoupled from code for instant portability across different hardware and VRAM budgets.
*   **Multi-Backend Engine Updater ([update_llama.py](update_llama.py)):**
    *   **Lazy Mode (`--download`):** Instant download of official precompiled GitHub releases (CUDA or Vulkan). Zero compiler or CUDA toolkit prerequisites (runs directly on GPU drivers).
    *   **Full Optimized Mode (`--build`):** Bleeding-edge master compilation with CMake + MSVC supporting **CUDA (NVIDIA)**, **Vulkan (Intel Arc / AMD / Universal)**, or **Hybrid (CUDA + Vulkan)** dual-backend builds.
    *   **Automated `--help` Option Diffing & Categorized Changelogs:** Instant visibility into new upstream features, Vulkan/CUDA optimizations, and fixes.
*   **Micro-Batch Optimization (`--ubatch-size 512`):** Cuts CUDA compute graph buffer from 879MB to 215MB, recovering **+664MB pure VRAM** while maintaining 805–886 t/s prompt prefill speed.
*   **Deterministic Layer Offloading (`-ngl <N>`):** Replaces brittle auto-fit with calibrated layer allocations for 100% stable startup immune to desktop VRAM fluctuations.
*   **Direct Physical RAM Loading (`--load-mode none`):** Completely bypasses OS mmap disk paging, eliminating runtime page faults, SSD wear, and MoE CPU tensor latency.
*   **Multi-Turn Memory Containment (`--cache-ram 0`, `--ctx-checkpoints 4`, `--context-shift`):** Prevents slot memory ballooning and provides smooth context rolling.
*   **Native Multi-Token Prediction (MTP):** Speculative draft verification accelerating text generation speeds up to **145–203 tokens/sec**.

---

## 💻 System Requirements & Prerequisites

On a new system, toolchain dependencies (**Python 3.14, uv, CMake, Ninja**) are managed automatically by **[`mise`](mise.toml)** with a single command: `mise install`.

| Requirement | Required to **Run** (`llama_runner.py`) | Required to **Lazy Update** (`update_llama.py --download`) | Required to **Source Build** (`update_llama.py --build`) | Provisioned By |
| :--- | :---: | :---: | :---: | :--- |
| **`mise` CLI** | ✅ Yes | ✅ Yes | ✅ Yes | [mise.jdx.dev](https://mise.jdx.dev) (auto-provisions Python 3.14, `uv`, `cmake`, `ninja`) |
| **GPU Display Driver** | ✅ Yes (NVIDIA or Intel/AMD) | ✅ Yes | ✅ Yes | GPU Vendor / Windows Update (CUDA & Vulkan runtimes) |
| **Visual Studio Build Tools** | ❌ No | ❌ No | ✅ Yes (`vcvars64.bat` / MSVC C++) | [VS Build Tools](https://aka.ms/vs/17/release/vs_BuildTools.exe) (Desktop C++) |
| **NVIDIA CUDA Toolkit** | ❌ No | ❌ No | ⚠️ Only for CUDA/Hybrid builds | [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) (*Not needed for pure Vulkan builds*) |
| **LunarG Vulkan SDK** | ❌ No | ❌ No | ⚠️ Only for Vulkan/Hybrid builds | `winget install KhronosGroup.VulkanSDK` (*Not needed for Lazy Mode*) |
| **Git** | ❌ No | ❌ No | ✅ Yes | Git for Windows / mise |

> 💡 **New System Quickstart:**
> 1. Run `mise install` (installs Python, uv, CMake, and Ninja into project environment).
> 2. Run `uv run update_llama.py --doctor` to verify your environment.

---

## 📊 Reference Performance Matrix (16GB VRAM Example Profile)

> 💡 **Hardware Adaptation:** The benchmarks below illustrate verified real-world performance on a reference **16GB GPU (NVIDIA RTX 5060 Ti + 48GB DDR5 Host RAM)**. To calibrate for other systems (e.g. 24GB RTX 3090/4090, Intel Arc 140T, or 8GB laptops), adjust the layer offload count (`ngl`) and context limits in [`presets.json`](presets.json).

| Base Model & Quant | Context | Vision | MTP Speculative | Layers in GPU | GPU VRAM | Host DDR5 RAM | Real Generation Speed | Notes / Recommendation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.8-27B-UD-IQ3_XXS`** | **96k** | 📷 GPU | ⚡ n=1 | **66 / 66** | **14.29 GB** | 720 MB | **45.1–49.3 t/s** | **100% GPU Daily Driver + Instant GPU Vision ⭐** |
| **`Qwen3.8-27B-UD-IQ3_XXS`** | **128k** | 📷 GPU | ❌ Off | **66 / 66** | **14.82 GB** | 970 MB | **32.1–32.4 t/s** | **100% GPU Max Multimodal Long-Context (OOM-Proof) ⭐** |
| **`Qwen3.8-27B-UD-IQ3_XXS`** | **200k** | 🖥️ CPU | ❌ Off | **66 / 66** | **14.94 GB** | 660 MB | **32.1–32.4 t/s** | **100% GPU Max Dense Window (200k Context) ⭐** |
| **`Qwen3.8-27B-UD-IQ3_XXS`** | **256k** | 📷 GPU | ⚡ n=1 | **66 / 66** | **13.81 GB** | 4.8 GB | **16.1–16.9 t/s** | **256k Full Window via Host RAM KV (`-nkvo`) 🚀** |
| **`Muse-Glimmer-30B-UD-Q3_K_XL`** | **262k** | ❌ Off | ❌ Off | **53 / 53** | **12.92 GB** | 398 MB | **27.8–28.3 t/s** | **#1 MCP & DeepSearch (Full Native Window) ⭐** |
| | **65k** | 📷 GPU | ❌ Off | **53 / 53** | **12.51 GB** | 510 MB | **27.7–28.2 t/s** | **100% GPU Multimodal Tool Calling (OOM-Proof) ⭐** |
| **`Nemotron-30B-A3B-Q4_K_XL`**| **65k** | ❌ N/A | ⚡ n=1 | **54 / 54** | **14.40 GB** | 11.2 GB | **57.14 t/s** | **3.5B Active MoE ⭐** |
| **`Nemotron-30B-A3B-Q4_K_XL`**| **256k**| ❌ N/A | ⚡ n=1 | **54 / 54** | **13.50 GB** | 12.1 GB | **55.57 t/s** | **Fast MoE Scaling** |
| **`Nemotron-30B-A3B-Q4_K_XL`**| **1024k (1M)**| ❌ N/A | ❌ Off | **54 / 54** | **10.40 GB** | 13.0 GB | **46.40 t/s** | **1M Context (1.7 GB KV) 🚀** |
| **`gemma-4-26B-A4B-Q4_K_XL`** | **65k** | ❌ Off | ⚡ n=2 | **31 / 31** | **14.07 GB** | 550 MB | **145.41 t/s** | **👑 ALL-TIME SPEED CHAMPION 🛸** |
| **`gemma-4-26B-A4B-Q4_K_XL`** | **65k** | 📷 GPU | ⚡ n=2 | **31 / 31** | **14.52 GB** | 620 MB | **88.02 t/s** | **100% GPU Ultra-Fast Multimodal ⭐** |
| **`gemma-4-26B-A4B-Q4_K_XL`** | **128k** | ❌ Off | ⚡ n=1 | **31 / 31** | **14.72 GB** | 520 MB | **125.39 t/s** | **100% GPU 128k High Speed MoE ⭐** |
| **`gemma-4-26B-A4B-Q4_K_XL`** | **128k** | 📷 GPU | ❌ Off | **31 / 31** | **15.30 GB** | 650 MB | **77.30 t/s** | **100% GPU 128k Multimodal MoE ⭐** |
| **`Qwen3.6-35B-A3B-UD-IQ3_XXS`**| **65k**| ❌ Off | ⚡ n=2 | **42 / 42** | **12.93 GB** | 450 MB | **125.48 t/s** | **🏆 All-Time Fastest Frontier MoE ⭐** |
| | **256k**| ❌ Off | ⚡ n=1 | **42 / 42** | **13.49 GB** | 620 MB | **78.01 t/s** | **256k Full Context Agentic Text ⭐** |
| | **256k**| 📷 GPU | ⚡ n=1 | **42 / 42** | **14.36 GB** | 810 MB | **53.22 t/s** | **🚀 256k Full Multimodal MoE (<14.4GB) ⭐** |
| **`gemma-4-12B (Q4_K_XL)`** | **256k** | 📷 GPU | ⚡ n=4 | **49 / 49** | **9.33 GB** | 980 MB | **106.11 t/s** | **100% GPU 256k Multimodal (<9.4GB) ⭐** |
| **`gemma-4-E4B (Q4_K_XL)`** | **128k** | 📷 GPU | ⚡ n=4 | **43 / 43** | **4.30 GB** | 350 MB | **203.30 t/s** | **👑 SPEED DEMON: 128k Native + Vision + MTP (203 t/s) 🛸** |

> 📖 **Full In-Depth Benchmark Report:** See [benchmark_results.md](benchmark_results.md) for the complete multi-turn burst curves, context scaling sweeps, and hardware telemetry analysis.

---

## 🛠️ Usage Guide

### 1. Interactive Server
Launch the interactive runner to host an OpenAI-compatible API server on `http://127.0.0.1:8080/v1`:
```powershell
mise run serve
# or: uv run .\llama_runner.py
```
Select **[1] Start Server (Verified 1-Click Hardware Presets ⭐)** to run any tested configuration, or choose **[2] Start Server (Custom)** to configure manually.

### 2. Updating the Engine
```powershell
# Option A: Lazy Mode (Fast download of official release binaries + GPU runtime)
mise run update
# or: uv run .\update_llama.py --download               # Defaults to CUDA (NVIDIA)
# or: uv run .\update_llama.py --download --vulkan      # Vulkan release (Intel Arc / AMD / Universal)

# Option A + Source Sync (Download prebuilt binaries AND sync Source/llama.cpp for agent analysis)
uv run .\update_llama.py --download --sync-source

# Option B: Full Optimized Mode (Compile from source with CMake & MSVC)
mise run build
# or: uv run .\update_llama.py --build                  # CUDA build (NVIDIA)
# or: uv run .\update_llama.py --build --vulkan         # Vulkan build (Intel Arc / AMD, no CUDA needed)
# or: uv run .\update_llama.py --build --hybrid         # Hybrid build (Both CUDA + Vulkan)

# System Toolchain Readiness Check (Doctor)
mise run doctor
# or: uv run .\update_llama.py --doctor

# Restore Previous Version from Backup
mise run rollback
# or: uv run .\update_llama.py --rollback
```

> 💡 **Agent Workflow Tip:** When collaborating with AI coding agents to optimize flags, benchmark performance, or inspect new CLI features, passing `--sync-source` (or cloning `Source/llama.cpp`) gives the agent direct access to upstream C++/CUDA/Vulkan kernel implementations and default values without requiring a local C++ compiler.

### 3. Downloading Models from HuggingFace
Download models directly into structured `models/<Family>/` subfolders with automatic vision projector deduplication and MTP sidefile exclusion:
```powershell
# Interactive Downloader (Pick from curated verified repos or enter custom HF repo)
mise run download
# or: uv run --with huggingface_hub python llama_runner.py --download

# Direct Repo Download
uv run --with huggingface_hub python llama_runner.py --download unsloth/Qwen3.8-27B-GGUF
```

### 4. Customizing Presets for Your Hardware
All presets and favorite models are declared in [`presets.json`](presets.json). You can edit existing entries or add custom ones:
```json
{
  "id": "my-custom-model",
  "category": "Agentic Long-Context",
  "label": "🚀 My Model [{variant}] - 128k Full Context",
  "match_family": "my-model-folder-name",
  "preferred_quants": ["Q4_K_M", "IQ4_XS"],
  "context": "128k",
  "vision": "auto",
  "reasoning": "Thinking (Natural / Medium - Recommended)",
  "mtp": "Off (Standard)",
  "engine_overrides": {
    "ngl": 48
  }
}
```
You can also specify a custom preset file path when launching the runner:
```powershell
uv run .\llama_runner.py --presets-file .\custom_presets.json
```

### 5. Automated Benchmark Sweeps
To run automated multi-turn evaluation sweeps across prompt lengths and temperature grids:
```powershell
uv run .\llama_runner.py
```
Select **[3] Run Benchmark**.

---

## 📖 Deep Documentation
*   [settings.md](settings.md): Detailed parameter rationale, reasoning budgets, vision configurations, and quantization guidelines.
*   [templates.md](templates.md): Jinja prompt template overrides, tool-calling definitions, and channel tags.
