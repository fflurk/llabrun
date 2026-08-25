# Optimal Serving Configurations & Architectural Principles for llama.cpp

This document outlines high-performance serving standards, memory optimizations, and vendor-aligned sampling parameters for **Qwen 3.8**, **NVIDIA Nemotron 3.5 Lightning**, **Gemma 4**, and **Qwen 3.6** models via `llama.cpp` (`llama-server.exe`). The parameters represent a synthesis of official vendor guidelines, deep engine profiling, and universal memory management strategies, accompanied by a verified 16GB reference hardware baseline (NVIDIA RTX 5060 Ti with 48GB DDR5 host RAM).

These configurations employ a **"defense-in-depth" architecture**: we pair hardened Jinja templates with native engine-level reasoning switches, direct memory-mapped RAM management (`--load-mode none`), micro-batch optimization (`--ubatch-size 512` which saves +664MB VRAM), prompt-cache containment (`--cache-ram 0`), deterministic layer offload (`-ngl <N>`), and calibrated Multi-Token Prediction (MTP) speculative decoding.

---

## Target Architecture & Model Summary

*   **Qwen 3.8 (27B Dense):** Flagship dense reasoning model. Using **Unsloth Dynamic `UD-IQ3_XXS`** (11.09 GB), it achieves **100% GPU offload (66/66 layers)** across all modes:
    *   **Daily Driver (96k + GPU Vision + MTP n=1):** **45.1–49.3 tokens/sec** with instant **<1.0s GPU image encoding**.
    *   **Max Long-Context GPU Vision (128k, MTP Off):** **32.1–32.4 tokens/sec** with instant **<1.0s GPU image encoding** (1.36 GB safety headroom).
    *   **Max Dense Window (200k, CPU Vision, MTP Off):** **32.1–32.4 tokens/sec** (flat memory footprint).
    *   **Full 256k Window via Host RAM KV (`-nkvo` + MTP n=1):** **16.1–16.9 tokens/sec** across the entire 262,144-token native context window.
*   **Muse Glimmer (30B Dense):** #1 MCP and tool-calling model. Achieves **100% GPU offload (53/53 layers)** at **262k native context** for pure text (**27.8–28.3 tokens/sec**), and **27.7–28.2 tokens/sec** with GPU Vision at 65k context with zero memory crashes.
*   **NVIDIA Nemotron 3.5 Lightning (30B-A3B MoE):** High-throughput hybrid model featuring **47 layers of Gated DeltaNet SSM** and **6 layers of Attention** with 128 sparse experts (6 active per token $\approx$ 3.5B active params). Delivers **56–57 tokens/sec** and scales seamlessly up to **1,000,000 tokens (1M context)** with a tiny 1.72 GB KV cache.
*   **Gemma 4 (26B-A4B & E4B):** High-entropy creative and conversational models (`Temp: 1.15`). E4B hits **203.3 tokens/sec** at 128k native context with GPU Vision and MTP (n=4). 26B-A4B delivers **125–145 tokens/sec** with MTP.
*   **Qwen 3.6 (35B-A3B MoE):** Long-context agentic workhorse. Runs at **256k full context** with GPU Vision + MTP at **53.2 tokens/sec** (100% GPU).

---

## 1. Engine & Memory Architecture Standards

### Micro-Batch Size Optimization (`--ubatch-size 512`, `-ub 512`)
*   **Why:** `llama.cpp` sizes its static CUDA compute graph buffer for batch kernel execution based on `ubatch-size`. Setting `ubatch 2048` permanently locks **`879.4 MiB`** of VRAM into the compute buffer.
*   **The Fix:** Setting `--ubatch-size 512` cuts the compute buffer to just **`215.1 MiB`**, **instantly freeing up `+664.3 MiB` of pure physical VRAM**.
*   **Performance:** Achieves **805–886 tokens/sec** prompt prefill ingestion speed (over 96% of max throughput) while reclaiming critical memory headroom needed for CLIP vision encoders.

### Host System RAM KV Cache Offload (`--no-kv-offload`, `-nkvo`)
*   **Why:** When running extreme long-context windows (200k–262k+ tokens) on dense models like Qwen 3.8 (27B), allocating the full KV cache in GPU VRAM requires either dropping 8+ layers to CPU (causing severe compute degradation down to 13 t/s) or crashing with CUDA OOM.
*   **The Solution:** `--no-kv-offload` instructs `llama.cpp` to store the Key/Value cache buffers entirely in host DDR5 system RAM while keeping **100% of the transformer layers (66/66) on the GPU**.
*   **Empirical Performance:** On PCIe 4.0/5.0 buses, streaming attention heads across the bus yields **12.2 t/s standard** and **16.8 t/s with MTP (n=1)** across a massive 256k context window, while freezing GPU VRAM usage at a static 13.8 GB.

### Hybrid Vision Offloading (`--no-mmproj-offload`)
*   **Why:** Vision projectors (`mmproj`) take ~950 MB of VRAM. For workflows where 90% of turns are text/code and images are only sent occasionally, dedicating GPU VRAM to the projector can crowd out MTP or force layer offloading.
*   **The Solution:** `--no-mmproj-offload` keeps the vision projector on CPU.
*   **Empirical Trade-Off:** Encoding a new image on CPU takes **~3.9s TTFT** (only ~2.3s longer than GPU). Once encoded, tokens generate at full **47.5 tokens/sec** (2.3x faster than partial layer offload). Multi-turn follow-ups on the same image experience **zero vision delay (<0.2s TTFT)** as visual tokens remain in the GPU KV cache.

### Direct Heap RAM Loading (`--load-mode none`)
*   **Why:** `llama.cpp` historically defaulted to `mmap` (memory-mapping from disk). On hybrid MoE models (where some expert layers reside in CPU RAM), `mmap` causes constant OS virtual memory page faults, micro-stutters during generation, and continuous SSD wear.
*   **The Fix:** `--load-mode none` replaces deprecated `--no-mmap` flags. It reads CPU-bound tensors sequentially into direct, contiguous DDR5 physical heap RAM at boot.
*   **Benefits:** Completely eliminates runtime disk I/O, prevents Windows Pagefile write churn, and boosts MoE draft speculative throughput.

### Prompt Cache & Context Checkpoint Containment (`--cache-ram 0`, `--ctx-checkpoints 4`)
*   **Why:** `llama-server` defaults to an 8,192 MiB prompt cache and 32 context checkpoints, which can accumulate hundreds of megabytes of stale memory across multi-turn chats.
*   **The Fix:** `--cache-ram 0` stops background slot memory ballooning, while `--ctx-checkpoints 4` caps checkpoint buffers.

### Context Shift Rolling (`--context-shift`)
*   **Why:** Prevents hard `HTTP 400 Bad Request` crashes when conversations reach the context ceiling by gracefully discarding the oldest middle tokens while preserving the system prompt.

### Deterministic Layer Offloading (`-ngl <N>`) vs. Auto-Fit
*   **Why:** Auto-fit (`--fit-target`) only measures static weights at cold boot and is blind to dynamic CLIP vision encoders (1.7 GB), multi-turn checkpoints, and Windows desktop fluctuations.
*   **The Fix:** Explicitly passing calibrated `-ngl <N>` values guarantees 100% deterministic boot and zero unexpected PCIe memory spills.

### Slot & Parallelism Containment (`--parallel 1`)
*   **Why:** When `--parallel` is omitted, `llama-server` automatically allocates 4 concurrent slots, multiplying KV cache allocations 4x and wasting over 2.6 GB of VRAM.
*   **The Fix:** Explicitly pass `--parallel 1` for single-user interactive and agentic workflows.

### KV Cache Quantization (`--cache-type-k q4_0`, `--cache-type-v q4_0`)
*   Standardized across all models. Slashes KV cache memory by **50%** compared to FP16 with virtually zero measurable perplexity loss, enabling 128k–256k context windows on 16GB cards.

---

## 2. Model Family Configurations

### A. Qwen 3.8 (27B Dense)

#### Baseline Settings
*   **Recommended Quantization:** `Qwen3.8-27B-UD-IQ3_XXS.gguf` (Unsloth Dynamic imatrix quantization, 11.09 GB).
*   **Template Override:** `--chat-template-file templates/chat_template_qwen.jinja`
*   **Chat Kwargs:** `{"preserve_thinking": true, "system_prompt": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.", "auto_disable_thinking_with_tools": false}`
*   **Engine Flags:** `-ngl 66 --load-mode none --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0 --batch-size 2048 --ubatch-size 512 --cache-ram 0 --ctx-checkpoints 4 --context-shift`
*   **Thinking Mode (ON):** `--reasoning on --reasoning-budget 8192`
*   **Sampling:** `Temp: 0.6`, `Top-P: 0.95`, `Top-K: 20`, `Presence Penalty: 0.0`, `Repeat Penalty: 1.0`.

#### Verified Serving Profiles (100% GPU Offload)
1.  **Daily Driver: 96k Context + Fast GPU Vision + MTP (n=1) ⭐:**
    *   Flags: `-ngl 66 --ctx-size 98304 --mmproj mmproj-BF16.gguf --spec-type draft-mtp --spec-draft-n-max 1`
    *   Performance: **45.1–49.3 tokens/sec** with instant **<1.0s GPU image encoding** and 1.03 GB safety headroom.
2.  **Max GPU Vision: 128k Context + Fast GPU Vision (MTP Off) ⭐:**
    *   Flags: `-ngl 66 --ctx-size 131072 --mmproj mmproj-BF16.gguf`
    *   Performance: **32.1–32.4 tokens/sec** with instant **<1.0s GPU image encoding** and 1.36 GB safety headroom.
3.  **Max Dense Window: 200k Context + Hybrid Vision (MTP Off) 📚:**
    *   Flags: `-ngl 66 --ctx-size 204800 --mmproj mmproj-BF16.gguf --no-mmproj-offload`
    *   Performance: **32.1–32.4 tokens/sec** with 0.95 GB safety headroom.
4.  **Full 256k Window: 256k Context + Host RAM KV (`-nkvo`) + MTP (n=1) 🌌:**
    *   Flags: `-ngl 66 --ctx-size 262144 --mmproj mmproj-BF16.gguf --no-kv-offload --spec-type draft-mtp --spec-draft-n-max 1`
    *   Performance: **16.1–16.9 tokens/sec** across the entire 262,144-token window with 1.18 GB VRAM headroom.

---

### B. NVIDIA Nemotron 3.5 Lightning (30B-A3B MoE)

#### Architecture Overview
*   **32.9B Total Parameters**, with only **3.5B active parameters per token** (6 active out of 128 experts).
*   **Hybrid Recurrent SSM Backbone:** 47 layers use linear Gated DeltaNet SSM ($O(1)$ constant 45 MiB state); only 6 layers use traditional Attention.

#### Baseline Settings
*   **Recommended Quantization:** `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_XL.gguf` (23.75 GB).
*   **Chat Kwargs:** `{"preserve_thinking": true, "system_prompt": "You are a helpful AI assistant."}`
*   **Engine Flags:** `--fit-target 384 --load-mode none --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0`
*   **Thinking Mode (ON):** `--reasoning on --reasoning-budget 8192`
*   **Sampling:** `Temp: 0.6`, `Top-P: 0.95`, `Top-K: 20`, `Min-P: 0.01`.
*   **Speculative MTP:** `--spec-type draft-mtp --spec-draft-n-max 1`

#### Context Scaling Performance (Live Measured)
| Context Length | MTP Setting | KV Cache VRAM | GPU Weights | Host DDR5 RAM | Generation Speed |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **65k (65,536)** | ⚡ MTP (n=1) | 108 MiB | 13.1 GB | 11.2 GB | **57.14 t/s** ⭐ |
| **128k (131,072)** | ⚡ MTP (n=1) | 216 MiB | 12.7 GB | 11.6 GB | **56.16 t/s** |
| **256k (262,144)** | ⚡ MTP (n=1) | 432 MiB | 12.2 GB | 12.1 GB | **55.57 t/s** |
| **512k (524,288)** | ⚡ MTP (n=1) | 864 MiB | 10.9 GB | 13.4 GB | **48.51 t/s** |
| **1M (1,048,576)** | ❌ MTP Off | 1.72 GB | 10.4 GB | 13.0 GB | **46.40 t/s** 🚀 |

---

### C. Gemma 4 Configurations (26B & E4B)

### Base Engine Settings
*   **Jinja Templates (`--jinja`):** Required for all models to properly process `--chat-template-kwargs`. Without this flag, the server uses a simpler C++ template parser that may silently ignore template variables.
*   **Custom Template Override (`--chat-template-file templates/chat_template_gemma.jinja`):** The embedded GGUF templates are significantly outdated compared to our local override (16,934 chars embedded vs 19,570 chars local). Key differences identified via extraction and diffing:
    *   **Tool response routing:** The embedded template only handles the legacy Gemma-native `tool_responses` array on assistant messages. Our override adds full OpenAI Chat Completions compatibility — it forward-scans consecutive `role: tool` messages, resolves `tool_call_id` back to function names, and handles content-parts arrays. Without this, any agent framework using the standard OpenAI tool-calling API will silently drop tool results.
    *   **Schema handling:** The embedded template has a simplistic type parser that only checks `value['type'] | upper == 'STRING'`. Our override handles union types (`anyOf`, `oneOf`, `allOf`), `$ref`, `const`, `null`, and array-of-types — critical for complex function schemas.
    *   **Non-thinking mode suppression:** The embedded template has no mechanism to suppress `<|channel>thought` tags when thinking is disabled. Our override injects an empty `<|channel>thought\n<channel|>` block when `enable_thinking` is false, preventing the model from spontaneously entering a thinking channel during instruct mode.
    *   **Image/video token formatting:** The E4B embedded template wraps media tokens in double newlines (`\n\n<|image|>\n\n`). Our override strips these to `<|image|>`, preventing excessive whitespace from consuming context budget.
*   **Memory Mapping (`--load-mode none`):** We load models directly into physical DDR5 heap RAM across all models, bypassing OS page faults and eliminating disk thrashing.
*   **KV Cache (`--cache-type-k q4_0`, `--cache-type-v q4_0`):** While `q8_0` provides maximum speed at small contexts, benchmark sweeps revealed that at massive contexts (128k), `q8_0` saturates VRAM and plunges generation speeds (down to ~8 t/s). Asymmetric quantization to `q4_0` completely resolves this bottleneck, maintaining a stable ~18 t/s even at 128k tokens with virtually zero visible degradation in output quality. `q4_0` is the strict baseline for daily driving.
*   **System Prompt Injection:** We inject the system prompt directly via `--chat-template-kwargs '{"system_prompt":"You are Gemma, a large language model. Think extra hard..."}'`. Bypassing the standard API message array guarantees the system prompt is structurally pinned at the absolute top of the context window by the Jinja parser, preventing it from being degraded or discarded during long multi-turn sessions.

### Thinking Mode (ON)
*Target: Generic reasoning, complex math, creative ideation.*
*   **Engine Flags:** `--reasoning on`, `--reasoning-budget 8192`, `--reasoning-budget-message "\nI will now provide my response.\n\n"`. (See [Reasoning Budget Strategy](#3-reasoning-budget-strategy) for rationale).
*   **Template Note:** `enable_thinking` is no longer needed in `--chat-template-kwargs`. The `--reasoning on` flag natively sets `thinking = 1` and triggers the Jinja template injection.
*   **Sampling:** `Temp: 1.15`, `Top-P: 0.95`, `Top-K: 64`. 

### Instruct Mode (OFF)
*Target: Fast generic chat, direct Q&A, formatting tasks.*
*   **Engine Flags:** `--reasoning off`, `--reasoning-budget 0`, `--reasoning-format none`.
*   **Sampling:** `Temp: 1.0`, `Top-P: 0.95`, `Top-K: 64`.

---

### D. Muse Glimmer Configurations (30B Dense)

#### Architecture & Benchmark Dominance
*   **30B Dense Frontier Reasoning Model:** Outstanding performance in Model Context Protocol (MCP) tool calling (**75.5 on MCP Atlas vs 62.5 on Qwen**), deep document retrieval (**74.6 on DeepSearch QA**), and mathematics (**94.7 on AIME 2026**).
*   **16GB GPU Fit:** 53 of 53 layers offloaded **100% to GPU** with `--fit-target 384` and `--parallel 1`.
*   **Real Speed:** **28.85 tokens/sec** at 65k context in thinking mode.

#### Baseline Settings
*   **Recommended Quantization:** `Muse-Glimmer-30B-UD-Q3_K_XL.gguf` (13.36 GB).
*   **Template Override:** `--chat-template-file templates/chat_template_muse_glimmer.jinja`
*   **Chat Kwargs:** `{"preserve_thinking": true, "system_prompt": "You are a helpful AI assistant."}`
*   **Engine Flags:** `--fit-target 384 --load-mode none --flash-attn on --cache-type-k q4_0 --cache-type-v q4_0`
*   **Thinking Mode (ON):** `--reasoning on --reasoning-budget 8192`
*   **Sampling:** `Temp: 0.6`, `Top-P: 0.95`, `Top-K: 20`, `Min-P: 0.01`.

#### Multimodal & Projector Characteristics (from Live Benchmarks)
*   **Text-Only (Vision Disabled):** 53/53 layers in VRAM $\rightarrow$ **25.87 t/s generation across full 262k native context window** (12.92 GB VRAM).
*   **Vision on GPU (`--mmproj mmproj-kquant.gguf`):**
    *   **65k Context:** **53/53 layers offloaded (100% in GPU)** in **12.51 GB VRAM** $\rightarrow$ **26.00 t/s generation**.
    *   **128k+ Context:** Spills 1 layer to RAM (52/53 @ 22.9 t/s) due to projector + large KV cache.
*   **Recommendation:** For agentic multimodal tool-calling, **65k Context** gives 100% GPU speed with fast vision encoding. For purely textual deep codebase search, scale to the full **262k native window** in text-only mode.

---

### E. Qwen 3.6 Configurations (35B & 27B)

### Base Engine Settings
*   **Jinja Templates (`--jinja`):** Required for proper processing of `preserve_thinking` and `system_prompt` template kwargs. Must be set for all Qwen entries.
*   **Custom Template Override (`--chat-template-file templates/chat_template_qwen.jinja`):** Load our community v22.2 override for error escalation and reasoning preservation.
    *   **Tool error detection and escalation:** The embedded template blindly relays tool responses. Our override actively inspects `role: tool` content for error signatures (`"error":`, `traceback`, `command not found`, etc.), tracks consecutive failures, and injects escalating `⚠️ SYSTEM WARNING` messages that force the model to abandon a failing strategy after 2+ consecutive errors. Without this, agentic loops can waste hundreds of tokens retrying the same broken command.
    *   **`preserve_thinking` handling:** The embedded template has a basic `enable_thinking` check. Our override adds full `preserve_thinking` support with a `ns_flags` namespace that correctly strips or preserves `<think>` blocks across multi-turn conversations, and cleans up tool_call content that leaks into the visible response.
    *   **Tool call serialization:** The embedded template uses string concatenation (`+`) for tool call formatting. Our override uses the safer tilde operator (`~`), adds `is defined` guards, handles arguments passed as raw JSON strings (not just mappings), and prevents empty parameter blocks from being emitted.
    *   **Consecutive-failure thinking bypass:** When 2+ tool calls fail in a row, our override forces `<think>\n</think>\n` (empty thinking block) to skip the reasoning phase entirely, pushing the model to immediately output a corrected action instead of wasting budget reasoning about the same failure.
*   **Memory Mapping (`--load-mode none`):** Direct DDR5 heap allocation prevents performance penalties when tensors are split between GPU and CPU.
*   **KV Cache (`--cache-type-k q4_0`, `--cache-type-v q4_0`):** Standardized across all models to prevent VRAM exhaustion and catastrophic speed drops at large contexts (65k-128k).
*   **System Prompt Injection:** Similar to Gemma, we hardcode the baseline instruction via `--chat-template-kwargs '{"system_prompt":"You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Think extra hard..."}'`. This anchors the strict coding behavior globally, ensuring the model doesn't drift into generic chatting over extremely long context windows.

### Thinking Mode (ON)
*Target: Precise Algorithmic Generation, WebDev, and Agentic Software Engineering.*
*   **Template:** `--chat-template-kwargs '{"preserve_thinking":true}'`.
*   **Why:** `enable_thinking` is deprecated and handled automatically by `--reasoning on`. `preserve_thinking` is *optional* but recommended for long-horizon agentic tasks; it ensures the cognitive trace is kept in the context window across turns (provided your front-end template safely handles think markup without breaking tool calls).
*   **Engine Flags:** `--reasoning on`, `--reasoning-budget 8192`, `--reasoning-budget-message "\n\n"`. (See [Reasoning Budget Strategy](#3-reasoning-budget-strategy) for rationale).
*   **Sampling:** `Temp: 0.6`, `Top-P: 0.95`, `Top-K: 20`. 
*   **Penalties:** `Presence Penalty: 0.0`, `Repeat Penalty: 1.0`.
*   **Why:** This matches Alibaba's official strict preset for coding. The **0.0 Presence Penalty is highly recommended**. If Qwen is penalized for repeating syntax (e.g., repeating variable names or `def` statements) while thinking, it may drift into alternative (often incorrect) solutions to avoid the penalty.

### Instruct Mode (OFF)
*Target: High-throughput metadata extraction, simple queries, and summarization.*
*   **Template:** `--chat-template-kwargs '{"preserve_thinking":false}'`.
*   **Engine Flags:** `--reasoning off`, `--reasoning-budget 0`, `--reasoning-format none`.
*   **Sampling:** `Temp: 0.7`, `Top-P: 0.8`, `Top-K: 20`.
*   **Penalties:** `Presence Penalty: 1.5`.
*   **Why:** Reasoning is disabled at the engine level. `preserve_thinking:false` ensures that even if earlier turns had thought tags, they are not kept across subsequent conversational turns. The high presence penalty (1.5) aggressively forces the model to avoid repeating itself, yielding highly concise and fast natural language. `--reasoning-format none` is an additional hard switch consistent with the defense-in-depth approach.

---

## 3. Reasoning Budget Strategy

### Current Setting: Fixed 8192 tokens (all models)

The reasoning budget serves a dual purpose: allowing sufficient depth for complex tasks while **preventing Qwen's known infinite reasoning loop** where the model repeats the same sentences until context is exhausted (a well-documented behavioral issue with Qwen 3.5/3.6 models).

**Why 8192:**
*   At 80 t/s (Gemma 26B): ~102 seconds of thinking — generous for multi-step reasoning.
*   At 55 t/s (Qwen 35B): ~149 seconds of thinking — sufficient for complex algorithmic problems.
*   At 28 t/s (Qwen 27B Dense): ~293 seconds of thinking — proportionally the most generous.
*   Uses only ~6.3% of the 128K context window, leaving ample room for the response.
*   Previous values (Gemma: 4096, Qwen: 6000) were too conservative — Gemma would frequently hit the cap and leak truncated reasoning into the visible response.

**Why not unlimited (`-1`):**
*   Qwen models are documented to enter infinite reasoning loops with certain system prompts or user messages, repeating sentences indefinitely until context exhaustion. A fixed budget is the only reliable safeguard against this.
*   Community reports confirm this persists across quantization levels and platforms (llama.cpp, SGLang, LM Studio).

### Budget Message (`--reasoning-budget-message`)

When the budget is exhausted, llama.cpp injects this text into the reasoning stream as if the model wrote it, then forces a transition to the response. This provides a **graceful landing** instead of a hard cut:

*   **Gemma:** `"\nI will now provide my response.\n\n"` — explicit natural-language nudge, needed because Gemma is less disciplined at the thinking→response boundary.
*   **Qwen:** `"\n\n"` — minimal double-newline, sufficient because Qwen responds well to whitespace cues and its template handles the transition more reliably.

### Future: Per-Request Dynamic Budget (Pi Coding Agent)

llama.cpp supports per-request budget control via the `thinking_budget_tokens` field in the API request body (e.g., `{"thinking_budget_tokens": 16384}`). **This only works when the CLI `--reasoning-budget` is set to `-1`.**

Relevant code from `server-common.cpp`:
```cpp
int reasoning_budget = opt.reasoning_budget;
if (reasoning_budget == -1 && body.contains("thinking_budget_tokens")) {
    reasoning_budget = json_value(body, "thinking_budget_tokens", -1);
}
```

**Migration plan for Pi coding agent:**
1.  Change `--reasoning-budget` to `-1` in `serve.ps1` (unlocks per-request control).
2.  Keep `--reasoning-budget-message` on the CLI (still applies when any budget fires).
3.  Have the Pi agent send `thinking_budget_tokens` per request:
    *   Quick classification/extraction: `2048`
    *   Standard coding tasks: `8192`
    *   Complex multi-file refactoring: `16384`–`32768`
4.  For requests without `thinking_budget_tokens`, budget defaults to unlimited — acceptable because a programmatic agent controls `max_tokens` and can detect/abort runaway loops.

> **Note:** The built-in llama-server Web UI does not expose `thinking_budget_tokens`, so the fixed CLI budget (8192) is required while using it as the primary interface.

---

## 4. Vision Projectors (mmproj)

### Current Projector Mappings & Precision

The fleet utilizes tailored mmproj files matching the exact model families on disk:

| Model Family | mmproj File Path | Precision / Quant | Purpose & Notes |
|---|---|---|---|
| **Gemma 4 26B** | `models/gemma-4-26B-A4B/mmproj-gemma-4-26B-A4B-it-q8_0.gguf` | Q8_0 (769 MB) | Shared across 26B Official & Uncensored |
| **Gemma 4 E4B** | `models/Gemma-4-E4B/mmproj-gemma-4-E4B-it-q8_0.gguf` | Q8_0 (520 MB) | High-throughput lightweight vision |
| **Gemma 4 12B** | `models/gemma-4-12B/mmproj-BF16.gguf` | Native BF16 (167 MB) | Encoder-free direct patch projection |
| **Qwen 3.6 35B** | `models/qwen3.6-35b-a3b/mmproj-Qwen3.6-35B-A3B-Q6_K.gguf` | Q6_K (570 MB) | Ultra-fast multimodal MoE projector |
| **Qwen 3.8 27B** | `models/Qwen3.8-27B/mmproj-BF16.gguf` | Native BF16 (870 MB) | Precision document & coding vision |
| **Muse Glimmer** | `models/Muse-Glimmer-30B/mmproj-kquant.gguf` | K-Quant (1.30 GB) | Multimodal tool-calling projector |

**Why Q8_0 and Native BF16 over F16:**
*   These models are trained natively in **BF16**. Converting BF16→F16 introduces clipping damage (truncated mantissa/exponent range) that manifests as visual noise.
*   Q8_0 and native BF16 avoid clipping entirely. Testing consistently confirms high fidelity across OCR, diagrams, and UI screenshots.
*   **Important:** Qwen 3.8 27B and 35B-A3B have different text model embedding dimensions (`n_embd = 5120` for 27B vs `n_embd = 2048` for 35B). They cannot share a projector and require distinct mmproj files to avoid load mismatch errors.

### Gemma 4 12B Encoder-Free Architecture & QAT Text Backbone

*   **Encoder-Free Design:** Unlike Gemma 4 26B-A4B and Qwen 3.6 (which have dedicated 27+ layer Vision Transformers), **Gemma 4 12B is an encoder-free unified multimodal model**. Its `mmproj-BF16.gguf` file is only ~167 MiB because it contains only 9 linear patch projection and positional embedding tensors (`vision_embedder.*`).
*   **The Text Backbone IS the Vision Encoder:** Because 12B lacks a separate vision tower, **the 48-layer language model decoder performs 100% of the visual feature extraction and spatial reasoning**.
*   **Unsloth Dynamic QAT (`UD-Q4_K_XL`):** Quantizing with Unsloth's QAT calibration preserves critical visual attention weights while shrinking the model to **6.26 GB**, enabling **106.1 t/s multimodal generation at 256k context in <9.4 GB VRAM**.

### GPU vs CPU Offloading

*   **Gemma 4 Models:** The vision architecture is relatively lightweight. Offloading the projector to the CPU takes ~8 seconds to process an image. This is a viable trade-off if you strictly need to preserve VRAM for text generation.
*   **Qwen 3.6 Models:** Qwen uses an exceptionally heavy Vision Transformer. Offloading to the CPU results in a **massive ~40-second penalty** per image. **Always load the Qwen mmproj on the GPU.**
### Gemma 4 Variable Resolution & Token Budget (`1120` Max Soft Tokens)

Gemma 4 models ship with **Variable Image Resolution** governed by discrete token budget buckets: **`70, 140, 280, 560, 1120`**.
*   **Default (280 tokens):** Optimized strictly for token efficiency and fast benchmark evaluation (~645K pixels). However, at this resolution, fine-grained OCR and small object detection fall apart.
*   **Maximum Detail (`1120` tokens / 2.51MP):** As confirmed by Google and official Gemma 4 documentation, manually bumping `max_soft_tokens` to **`1120`** allows the model to process up to 10,080 initial image patches (which are compressed 3x3 into 1120 final visual embeddings). This provides sharp OCR and state-of-the-art visual reasoning.
*   **`llama.cpp` Configuration:** In our runner (`llama_runner.py`), when vision mode is enabled (`mode != 'No'`), we configure:
    *   `--image-min-tokens 560`: Ensures images are never downscaled below medium detail.
    *   `--image-max-tokens 1120`: Sets the high-detail ceiling to Google's official maximum 2.51MP bucket.
    *   `--mtmd-batch-max-tokens 1120`: Elevates `llama.cpp`'s multimodal encoder batch limit above the default `1024` so 1120-token image encoding succeeds in a single pass.
    *   `--ubatch-size 2048`: Elevates the physical ubatch size so full-image causal attention is never split across ubatches.

### Prompt Caching Limitation
**Critical Note:** Loading a vision projector currently **disables prompt caching** (`cache_reuse`) in llama.cpp. When vision is active, every request must re-process the entire prompt history from scratch. For deep agentic coding sessions that do not require images, it is highly recommended to run with vision **Not Loaded** to benefit from massive prompt processing speedups across conversational turns.

---

## 5. Logging & Diagnostics

### Verbosity (`-lv 3`)

Set to level 3 (info), which is the llama-server default. This captures model loading details essential for diagnosing memory and layer offloading:

| Level | What it captures |
|---|---|
| 0 | Generic output only |
| 1 | + Errors |
| 2 | + Warnings |
| **3** | **+ Info: model size, CUDA buffer sizes, layer offloading, KV cache allocation** |
| 4 | + Trace: per-request details |
| 5 | + Debug: everything |

> **Note:** A previous setting of `-lv 1` (errors only) resulted in permanently empty log files, since no errors occurred during normal operation. Level 3 adds zero inference overhead — it only affects log file verbosity.

### Post-Load Memory Diagnostic

A background PowerShell process automatically reports VRAM usage after the server is healthy:
*   **nvidia-smi:** GPU name, VRAM used/total/free.
*   **Server log parsing:** Extracts `buffer size`, `offload.*layer`, `model size`, `CUDA`, `host buffer`, `kv_cache`, and `mmproj` lines from the timestamped debug log.

---

## 6. Architecture & Benchmark Insights (August 2026)

Systematic sweeps established several critical operational principles:

### Dense vs MoE Code Generation & SWE-Bench Calibration
*   **Dense Models (`Qwen 3.8 27B`, `Gemma 4 26B`)**: Exhibit exceptional deterministic reasoning and syntax integrity when bridging complex synchronous CLI frameworks with asynchronous backends (e.g., `Typer` + `Playwright`).
*   **Frontier MoE (`Qwen 3.6 35B-A3B`)**: When calibrated with Alibaba's thinking template at `Temp 0.6`, it matches 27B dense coding capabilities (SWE-bench 73.4, Terminal-Bench 51.5) while running at **125.5 t/s** ($N=2$) and scaling up to **256k context**.
*   **High-Entropy Sampling Warning**: If temperature drifts above `0.8` on MoE models, expert activation becomes unstable and causes hallucinated method signatures. **Keep MoE coding at Temp 0.6**.

### Context Shift and OpenThoughts
Models leveraging the OpenThoughts reasoning trace generate large amounts of internal monologue—often **4,000 to 8,000 tokens** per response. 
*   **Context Management**: For long multi-turn sessions, the `--context-shift` flag acts as a pressure valve to keep the engine operating without OOM crashes.
*   **Speed Stability**: With `q4_0` KV cache and FlashAttention-2, generation speed remains flat and responsive even beyond 128k context.

### QAT/PTQ Stability and Sampler Pipeline
When serving Quantization-Aware Training (QAT) or Post-Training Quantization (PTQ) models (e.g., Gemma E4B or Qwen A4B variants), **high temperatures (e.g., 1.15) cause catastrophic syntax failure**.
*   **The Cause**: The `llama.cpp` sampler pipeline applies Temperature *last* (after `min_p`, `top_p`, `top_k`). High temperature multipliers amplify underlying quantization noise.
*   **The Solution**: A lower temperature setting of `0.6` effectively suppresses this quantization noise, yielding stable, coherent code generation even from 4B models. **Temp 0.6 is mandatory for heavily quantized QAT/PTQ models**.

---

## 7. Master Hardware Compatibility Matrix (16GB RTX 5060 Ti)

System verified with **16,310 MiB VRAM (15.35 GB usable)** and **48GB DDR5 Host RAM** using `--fit-target 384 --load-mode none`:

| Base Model & Quant | Context | Vision | MTP | Layers in GPU | GPU VRAM | Host DDR5 RAM | Real Generation Speed | Verified Status & Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`Qwen3.8-27B-UD-IQ3_XXS`** | **65k** | 📷 GPU | ⚡ n=1 | **66 / 66** | **10.55 GB** | 720 MB | **42.67 t/s** | **100% GPU Daily Driver ⭐** |
| | **128k** | 📷 GPU | ❌ Off | **66 / 66** | **13.17 GB** | 970 MB | **28.50 t/s** | **100% GPU Multimodal Max Context ⭐** |
| | **140k** | ❌ Off | ⚡ n=1 | **66 / 66** | **10.53 GB** | 530 MB | **42.10 t/s** | **100% GPU Turbo Long-Context ⭐** |
| | **200k** | ❌ Off | ❌ Off | **66 / 66** | **14.17 GB** | 660 MB | **29.40 t/s** | **100% GPU Max Dense Ceiling ⭐** |
| **`Qwen3.8-27B-UD-IQ3_S`** | **65k** | ❌ Off | ⚡ n=1 | **66 / 66** | **11.07 GB** | 590 MB | **43.22 t/s** | **100% GPU High Precision ⭐** |
| | **128k** | ❌ Off | ❌ Off | **66 / 66** | **13.47 GB** | 650 MB | **30.17 t/s** | **100% GPU Text Long-Context ⭐** |
| **`Muse-Glimmer-30B-UD-Q3_K_XL`** | **262k** | ❌ Off | ❌ Off | **53 / 53** | **12.92 GB** | 398 MB | **25.87 t/s** | **#1 MCP & DeepSearch (Full Native Window) ⭐** |
| | **65k** | 📷 GPU | ❌ Off | **53 / 53** | **12.51 GB** | 510 MB | **26.00 t/s** | **100% GPU Multimodal Tool Calling** |
| **`Nemotron-30B-A3B-Q4_K_XL`** | **65k** | ❌ N/A | ⚡ n=1 | **54 / 54** | **14.40 GB** | 11.2 GB | **57.14 t/s** | **MoE Fast Path (3.5B active) ⭐** |
| | **256k** | ❌ N/A | ⚡ n=1 | **54 / 54** | **13.50 GB** | 12.1 GB | **55.57 t/s** | **Fast MoE Context Scaling** |
| | **1024k (1M)**| ❌ N/A | ❌ Off | **54 / 54** | **10.40 GB** | 13.0 GB | **46.40 t/s** | **1M Context (1.7 GB KV) 🚀** |
| **`gemma-4-26B-A4B-Q4_K_XL`** | **65k** | ❌ Off | ⚡ n=2 | **31 / 31** | **14.07 GB** | 550 MB | **145.41 t/s** | **👑 ALL-TIME FASTEST SPEED CHAMPION 🛸** |
| | **65k** | 📷 GPU | ⚡ n=2 | **31 / 31** | **14.52 GB** | 620 MB | **88.02 t/s** | **100% GPU Ultra-Fast Multimodal ⭐** |
| | **128k** | ❌ Off | ⚡ n=1 | **31 / 31** | **14.72 GB** | 520 MB | **125.39 t/s** | **100% GPU 128k High Speed MoE ⭐** |
| | **256k** | ❌ Off | ❌ Off | **31 / 31** | **16.05 GB** | 650 MB | **70.70 t/s** | **100% GPU 256k Window ⭐** |
| **`Qwen3.6-35B-A3B-UD-IQ3_XXS`**| **65k**| ❌ Off | ⚡ n=2 | **42 / 42** | **12.93 GB** | 450 MB | **125.48 t/s** | **🏆 All-Time Fastest Frontier MoE ⭐** |
| | **65k** | 📷 GPU | ⚡ n=1 | **42 / 42** | **13.24 GB** | 580 MB | **85.92 t/s** | **100% GPU Hyper-Speed Multimodal ⭐** |
| | **140k**| ❌ Off | ⚡ n=1 | **42 / 42** | **13.15 GB** | 510 MB | **97.36 t/s** | **140k Turbo Context** |
| | **256k**| ❌ Off | ⚡ n=1 | **42 / 42** | **13.49 GB** | 620 MB | **78.01 t/s** | **256k Full Context Agentic Text ⭐** |
| | **256k**| 📷 GPU | ⚡ n=1 | **42 / 42** | **14.36 GB** | 810 MB | **53.22 t/s** | **🚀 256k Full Multimodal MoE (<14.4GB) ⭐** |
| **`gemma-4-12B (Q4_K_XL)`** | **65k** | ❌ Off | ⚡ n=4 | **49 / 49** | **6.90 GB** | 300 MB | **110.31 t/s** | **2.15x MTP Speedup ⭐** |
| | **256k** | 📷 GPU | ⚡ n=4 | **49 / 49** | **9.33 GB** | 980 MB | **106.11 t/s** | **100% GPU 256k Multimodal (<9.4GB) ⭐** |
| **`gemma-4-E4B (Q4_K_XL)`** | **65k** | ❌ Off | ⚡ n=5 | **43 / 43** | **3.12 GB** | 250 MB | **186.21 t/s** | **👑 ALL-TIME FASTEST SPEED DEMON (186 t/s) 🛸** |
| | **65k** | 📷 GPU | ⚡ n=2 | **43 / 43** | **3.54 GB** | 310 MB | **161.86 t/s** | **100% GPU Multimodal Speed Champion** |
| | **256k** | ❌ Off | ⚡ n=4 | **43 / 43** | **4.89 GB** | 1.8 GB | **163.12 t/s** | **Ultra-Light 256k Long-Context (163 t/s) 🚀** |

> 📖 **Full In-Depth Benchmark Report:** See [benchmark_results.md](benchmark_results.md) for the complete multi-turn burst curves, context scaling sweeps, and hardware telemetry analysis.

