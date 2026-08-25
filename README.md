# llabrun — Llama.cpp Lab Builder & Runner

A production-tuned local LLM orchestration, building, and benchmarking suite built on `llama.cpp`. Designed for deterministic layer offloading, micro-batch memory optimization, direct RAM loading, speculative decoding (MTP), and instant cross-platform portability across **macOS (Apple Silicon Metal)**, **Linux / WSL2 (CUDA & Vulkan)**, and **Windows (CUDA, Vulkan, Hybrid)** via declarative model presets.

---

## 🚀 Key Features

*   **Zero-Overhead Runner CLI ([llama_runner.py](llama_runner.py)):** Interactive model selector, benchmark orchestrator, and Router INI preset generator.
*   **Modular Presets Configuration ([presets.example.json](presets.example.json)):** Declarative favorite models and verified presets decoupled from code for instant portability across different hardware and VRAM budgets.
*   **Cross-Platform Engine Updater ([update_llama.py](update_llama.py)):**
    *   **macOS (Apple Silicon M1–M4):** Automatic Apple Metal acceleration (both Lazy precompiled releases and source builds via Apple Clang).
    *   **Linux & WSL2:** NVIDIA CUDA Toolkit compilation, Vulkan compute, and CPU OpenMP acceleration.
    *   **Windows:** Dual-backend Hybrid (CUDA + Vulkan), standalone CUDA (NVIDIA), or standalone Vulkan (Intel Arc / AMD).
    *   **Lazy Mode (`--download`):** Instant download of official precompiled GitHub releases (`.zip` on Windows, `.tar.gz` on macOS/Linux).
    *   **Full Optimized Mode (`--build`):** Master branch source compilation with CMake + native compilers (`clang` on macOS, `gcc` on Linux, `MSVC` on Windows).
    *   **Automated `--help` Option Diffing & Categorized Changelogs:** Instant visibility into upstream changes.
*   **Micro-Batch Optimization (`--ubatch-size 512`):** Recovers **+664MB pure VRAM** while maintaining blisteringly fast prompt prefill.
*   **Deterministic Layer Offloading (`-ngl <N>`):** Calibrated layer allocations for 100% stable startup immune to desktop VRAM fluctuations.
*   **Direct Physical RAM Loading (`--load-mode none`):** Completely bypasses OS disk paging, eliminating page faults, SSD wear, and MoE CPU tensor latency.
*   **Multi-Turn Memory Containment (`--cache-ram 0`, `--ctx-checkpoints 4`, `--context-shift`):** Prevents slot memory ballooning and provides smooth context rolling.
*   **Native Multi-Token Prediction (MTP):** Speculative draft verification accelerating text generation speeds up to **145–203 tokens/sec**.

---

## 💻 Cross-Platform Requirements & Prerequisites

On any platform, toolchain dependencies (**Python 3.14, uv, CMake, Ninja**) are managed automatically by **[`mise`](mise.toml)** with a single command: `mise install`.

| Platform | Primary Acceleration | Lazy Mode (Zero SDK) | Source Build Toolchain |
| :--- | :---: | :--- | :--- |
| **macOS (Apple Silicon)** | **Apple Metal** ⚡ | `llama-b*-bin-macos-arm64.tar.gz` | Apple Clang (`xcode-select --install`) |
| **Linux & WSL2** | **CUDA** ⚡ / **Vulkan** | `llama-b*-bin-ubuntu-vulkan-x64.tar.gz` | GCC / G++ + CMake + NVIDIA CUDA Toolkit |
| **Windows** | **CUDA** / **Vulkan** / **Hybrid** ⚡ | `llama-b*-bin-win-*-x64.zip` | Visual Studio MSVC (`vcvars64.bat`) + CUDA / Vulkan SDKs |

> 💡 **New System Quickstart:**
> 1. Run `mise install` (installs Python, uv, CMake, and Ninja).
> 2. Run `uv run update_llama.py --doctor` to verify your environment.
> 3. Run `mise run update` (or `uv run update_llama.py`) to install your engine.

---

## 📊 Reference Performance Matrix (16GB VRAM Example Profile)

> 💡 **Hardware Adaptation:** The benchmarks below illustrate verified real-world performance on a reference **16GB GPU (NVIDIA RTX 5060 Ti + 48GB DDR5 Host RAM)**. To calibrate for other systems (e.g. 24GB RTX 3090/4090, Intel Arc 140T, or 8GB laptops), adjust the layer offload count (`ngl`) and context limits in [`presets.example.json`](presets.example.json) or your custom `presets.json`.

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

### 4. Configuration & Settings Architecture
`llabrun` decouples personal system settings from model presets to prevent git merge conflicts and protect private API keys:

*   **`settings.json` (System & Network Config):**
    Copy [`settings.example.json`](settings.example.json) to `settings.json` (ignored by git). Configure server port, host, private API key, directories, and default hardware threads:
    ```json
    {
      "server": {
        "host": "127.0.0.1",
        "port": 8080,
        "api_key": "sk-my-secret-key",
        "timeout": 600
      },
      "paths": {
        "models_dir": "models",
        "presets_file": "presets.json"
      }
    }
    ```
    > 🔒 **Security:** When `api_key` is set, `llama-server` automatically enforces `Authorization: Bearer <KEY>` authentication on all `/v1/*` endpoints. Because `settings.json` is in `.gitignore`, your secret keys are never committed to git.

*   **`presets.json` (Model & Hardware Profiles):**
    Copy [`presets.example.json`](presets.example.json) to `presets.json` (ignored by git) to customize your 1-click model profiles, layer offload counts (`ngl`), and context sizes without conflicts:
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
