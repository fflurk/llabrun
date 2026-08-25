# Chat Template Analysis: Embedded GGUF vs Local Overrides

This document provides a line-by-line analysis of the Jinja chat templates embedded inside our GGUF model files versus the local override templates we load via `--chat-template-file`. Templates were extracted using `extract_templates.py` on 2026-06-03. **Updated 2026-07-27** with new upstream templates from both the Qwen community and Google.

---

## Inventory

| GGUF Source | Embedded Size | Local Override | Override Size |
|---|---|---|---|
| Gemma 26B Official (UD-IQ4_XS) | 16,934 chars (354 lines) | `chat_template_gemma.jinja` | 27,132 chars (515 lines) |
| Gemma 26B Uncensored (HauhauCS) | 16,934 chars (354 lines) | `chat_template_gemma.jinja` | (same as above) |
| Gemma E4B Uncensored (HauhauCS) | 11,926 chars (263 lines) | `chat_template_gemma.jinja` | (same as above) |
| **New Official (Google 2026-07-09)** | **18,683 chars (391 lines)** | **`new_google_chat_template_gemma.jinja`** | **Retained as upstream reference** |
| **New Official E4B (Google 2026-07-09)** | **18,569 chars (387 lines)** | **`new_google_chat_template_gemma_e4b.jinja`** | **Retained as upstream reference** |
| Qwen 27B Official (UD-IQ3_XXS) | 8,057 chars (158 lines) | `chat_template_qwen.jinja` | 16,289 chars (329 lines) |
| Qwen 27B Uncensored (HauhauCS) | 7,764 chars (154 lines) | `chat_template_qwen.jinja` | 16,289 chars (329 lines) |
| Qwen 27B OT (NEO-CODE-HERE) | 7,764 chars (154 lines) | `chat_template_qwen.jinja` | 16,289 chars (329 lines) |
| Qwen 35B Official (MXFP4_MOE) | 8,057 chars (158 lines) | `chat_template_qwen.jinja` | 16,289 chars (329 lines) |
| Qwen 35B Uncensored (all 3) | 7,764 chars (154 lines) | `chat_template_qwen.jinja` | 16,289 chars (329 lines) |

> [!IMPORTANT]
> The mmproj files (`mmproj-*.gguf`) contain **no chat template at all** — they are pure vision encoder weights with no text formatting logic.

---

## 1. Gemma Templates

### 1.1 Override Strategy: Rebuilt From Google's 2026-07-09 Base

As of 2026-07-27, our `chat_template_gemma.jinja` override is **rebuilt from Google's latest canonical template** (`new_google_chat_template_gemma.jinja`, published 2026-07-09) as the base, with our custom fixes backported on top. Every modification is marked with `{#- BACKPORT: ... -#}` inline comments for clean diffing against upstream.

This replaces the previous approach of maintaining a heavily diverged fork. The new structure makes future Google updates easy to merge and our fixes easy to isolate for a potential upstream PR.

#### Thinking defaults are controlled via llama.cpp arguments

Google's defaults (`enable_thinking=false`, `preserve_thinking=false`) are left as-is in the template. We control these via `--jinja-enable-thinking` and `--jinja-preserve-thinking` flags in llama.cpp instead of hardcoding them in the template.

#### What Google Fixed in Their 2026-07-09 Update

| Fix | Description |
|---|---|
| **`format_argument` null handling** | Added `{%- if argument is none -%} 'null'` — previously null values fell through to `else` |
| **`strip_thinking` macro** | New macro to strip `<\|channel\>thought...<channel\|\>` blocks from content |
| **O(1) continuation detection** | Uses `prev_non_tool_role` tracking instead of O(n) backward scan |
| **Forward-scan for tool responses** | OpenAI Chat Completions `role: tool` messages handled via forward-scan with `tool_call_id` resolution |
| **Content-parts + media tokens** | Handles content-parts arrays with `image_url`, `input_audio` type aliases |
| **`.get()` syntax** | Safer key access throughout — returns `None` instead of crashing on missing keys |
| **Tool-response thinking injection** | At generation prompt, opens `<\|channel\>thought\n` after tool responses when thinking is enabled |
| **`continues_into_next` logic** | `(not message.get('tool_calls') or ns_tr_out.flag)` — allows continuation when tool calls are fully resolved |

#### What Google Still Lacks (Our Backported Fixes)

| # | Backport Tag | Fix | Why It Matters |
|---|---|---|---|
| 1 | `Build type_names` | Union type support: `["string","null"]` | OpenAI APIs commonly send nullable params as type arrays — `value['type'] \| upper` crashes on lists |
| 2 | `anyOf/oneOf/allOf/$ref/const` | JSON Schema union keywords | Schemas using composition keywords were silently dropped |
| 3 | `enum outside STRING check` | Enum for any type | Google only checked enum inside `STRING` branch |
| 4 | `nullable guard` | `is defined and` guard | `{%- if value['nullable'] %}` crashes if key absent |
| 5 | `required guard` | `is defined and` guard | `{%- if value['required'] %}` crashes if key absent |
| 6 | `type emission` | String vs array branching | `type:<\|"\|\>{{ value['type'] \| upper }}<\|"\|\>` produces garbage on array types |
| 7 | `$defs serialization` | `params['$defs']` in function declarations | Schemas using `$defs`+`$ref` compiled to nothing |
| 8 | `is_tool_error` macro | Error pattern detection | No error recovery without this |
| 9 | `consecutive_failures` | Failure counter + `⚠️ SYSTEM WARNING` injection | Prevents infinite retry loops in agentic workflows |
| 10 | `dynamic_thinking` | `<\|think_off\|\>` / `<\|think_on\|\>` toggle | Mid-conversation thinking control |
| 11 | `failure bypass` | Empty `<\|channel\>thought\n<channel\|\>` on 2+ failures | Forces immediate corrective action without reasoning |

> [!TIP]
> **Fixes 1–7 are pure bug fixes** that Google should accept in an upstream PR. Fixes 8–11 are our agentic extensions.

---

### 1.2 New E4B Template — Now Matches 26B

Google's new E4B template (`new_google_chat_template_gemma_e4b.jinja`, 387 lines) is now **nearly identical** to the 26B version (391 lines). The only difference:

| Feature | 26B Template | E4B Template |
|---|---|---|
| **Non-thinking suppression** | Injects `<\|channel\>thought\n<channel\|\>` when `enable_thinking` is false | **Missing** — no suppression at generation prompt |

This is a massive upgrade from the old E4B template (263 lines) which was missing tool role handling, content capture, continuation detection, and essentially everything needed for agentic use. **Our single `chat_template_gemma.jinja` override now serves both E4B and 26B/12B models.**

---

### 1.3 Gemma 26B Embedded (Both Variants) — Unchanged

The two Gemma 26B GGUFs (official UD and HauhauCS uncensored) still contain **byte-identical** templates (16,934 chars). These are the same as previously documented and are superseded by Google's 2026-07-09 update. Our override is mandatory.

---

### 1.4 Gemma E4B Embedded — Oldest, Most Limited

The E4B embedded template (263 lines) remains unchanged in the GGUF. It predates tool role handling, content capture, and continuation detection entirely. Our override is **mandatory** for E4B.

> [!CAUTION]
> The E4B embedded template is the most dangerous to use unmodified. It lacks tool role handling, non-thinking suppression, and content capture — making it fundamentally incompatible with any agentic framework.

---

## 2. Qwen Templates

### 2.1 Override Strategy: Adopted Community v22.2 (`qwen3.8-froggeric-v22.2`)

Our Qwen override is the **community froggeric v22.2 template** (`chat_template_qwen.jinja`, version string `qwen3.8-froggeric-v22.2`) loaded via `--chat-template-file templates/chat_template_qwen.jinja` (and automatically synchronized from HuggingFace via `update_llama.py`). 

The v22.2 template is specifically tailored for **Qwen 3.8 and 3.6 models**, fixing reasoning traps, false error triggers, and client aliases across local coding harnesses (Cline, Claude Code, Cursor, OpenCode).

#### What v22.2 Delivers & Changes for Us:

| Feature / Upgrade | Details in v22.2 | Value for Our Local Harness |
|---|---|---|
| **Code & Grep Error Disambiguation** | Adds `_is_code_or_grep` filter that ignores code containing `throw new Error`, `console.error`, `logger.error`, `import `, `def `, etc. | **Eliminates false `⚠️ SYSTEM WARNING` loops** when tools read files or search codebases containing standard error-handling syntax. |
| **Extended Reasoning Aliases** | Maps OpenAI/Claude aliases: `"high"`, `"max"`, `"ultracode"`, `"extreme"` $\to$ `"xhigh"`; `"minimal"` $\to$ `"low"`; `"none"`, `"off"` $\to$ disabled. | **Zero breakage** when calling local endpoints from diverse coding clients. |
| **Multi-System Message Merging** | Merges multiple leading `system` and `developer` turns into a single unified block separated by `\n\n`. | Properly preserves developer preambles without creating invalid fragmented ChatML turns. |
| **Zero-Token `medium` Baseline** | Defaults to `medium` with zero extra prompt tokens injected. | Retains 100% Prefix KV Cache hit rate across conversational turns. |
| **Native `--reasoning-preserve`** | Supports `preserve_reasoning` as an alias for `preserve_thinking`. | Full compatibility with llama.cpp's latest CLI flags. |
| **Inline Dynamic Chat Tags** | Supports `<\|think_off\|>`, `<\|think_low\|>`, `<\|think_medium\|>`, `<\|think_xhigh\|>`, `<\|think_ultracode\|>`, `<\|think_extreme\|>`. | Allows per-message reasoning depth adjustments on the fly. |

---

### 2.2 Embedded Template Variants (Unchanged)

There are two distinct embedded templates across the Qwen GGUFs:

| Variant | Found In | Size | Key Difference |
|---|---|---|---|
| **Official (Unsloth)** | UD-IQ3_XXS, MXFP4_MOE | 8,057 chars (158 lines) | Has `developer` role support, dual-system-message merging, does NOT raise on missing user query |
| **Uncensored (HauhauCS)** | All HauhauCS models, NEO-CODE-HERE-OT | 7,764 chars (154 lines) | No `developer` role, raises exception `'No user query found'` if all user messages are tool responses |

---

### 2.3 Key Gaps in All Embedded Qwen Templates (Why v22.2 Override is Mandatory)

All Qwen embedded GGUF templates lack the critical agentic safeguards present in our v22.2 override:

| Feature | Embedded GGUF Template | v22.2 Override |
|---|---|---|
| **Code/Grep Error Disambiguation** | ❌ None | ✅ `_is_code_or_grep` filter prevents false retries |
| **Two-Tier Error Warning Escalation** | ❌ None | ✅ Injects `⚠️ SYSTEM WARNING` after 1 and 2 failures |
| **Reasoning Bypass on Repeated Failures** | ❌ None | ✅ Forces `<think>\n</think>` to break infinite retry loops |
| **Dynamic Inline Thinking Tags** | ❌ None | ✅ Full `<\|think_*\|>` support |
| **JSON & XML Tool Calling Support** | ❌ Crashing on stringified JSON | ✅ Robust parsing of both Dict and Stringified JSON |
| **Oversized Tool Payload Truncation** | ❌ None | ✅ `max_tool_arg_chars` & `max_tool_response_chars` guards |

> [!WARNING]
> The embedded Qwen templates have **zero error recovery**. In an agentic loop, a single tool failure can cause the model to retry the exact same broken command indefinitely, wasting the entire context window. The v21.3 override's failure detection and escalation warnings are the only safeguard against this.

---

## 3. Cross-Family Comparison

### Template Architecture

| Aspect | Gemma (all) | Qwen (all) |
|---|---|---|
| **Special tokens** | `<\|turn\>`, `<\|channel\>`, `<\|tool_call\>`, `<\|tool_response\>`, `<\|think\|\>` | `<\|im_start\|\>`, `<\|im_end\|\>`, `<think>`, `<tool_call>`, `<tool_response>` |
| **System prompt location** | Inside `<\|turn\>system\n...<turn\|\>` block | Inside `<\|im_start\|\>system\n...<\|im_end\|\>` block |
| **Thinking activation** | `enable_thinking` kwarg → injects `<\|think\|\>` token in system turn | `enable_thinking` kwarg → controls whether `<think>` block is opened |
| **Tool schema format** | Custom compact notation: `declaration:name{description:...,parameters:{...}}` | Full JSON: `tool \| tojson` with XML-style `<function>` blocks |
| **Tool response format** | `<\|tool_response\>response:name{key:value}<tool_response\|\>` | `<tool_response>\ncontent\n</tool_response>` in user turn |

### What Each Override Adds

| Override Feature | Gemma Override | Qwen Override |
|---|---|---|
| OpenAI tool compatibility | ✅ Forward-scan `role: tool` messages | N/A (already uses `role: tool` natively) |
| Error detection & warnings | ✅ `is_tool_error` macro + `⚠️ SYSTEM WARNING` | ✅ Pattern matching + `⚠️ SYSTEM WARNING` |
| Thinking bypass on failures | ✅ Empty `<\|channel\>thought` block after 2+ failures | ✅ Empty `<think>` block after 2+ failures |
| Dynamic thinking toggle | ✅ `<\|think_off\|\>` / `<\|think_on\|\>` tags | ✅ `<\|think_off\|\>` / `<\|think_on\|\>` tags |
| Non-thinking suppression | ✅ Empty `<\|channel\>thought` block | ✅ Empty `<think>` block |
| Union type schemas | ✅ `anyOf`, `oneOf`, `allOf`, `$ref` | N/A (uses JSON, not compact notation) |
| Continuation detection | ✅ O(1) `prev_non_tool_role` tracking | N/A (ChatML format doesn't need this) |
| Content capture | ✅ `captured_content` + `has_content` check | N/A (ChatML has explicit `<\|im_end\|\>`) |
| Null value serialization | ✅ Explicit `null` output | N/A (uses `tojson` filter) |
| JSON tool format | N/A (uses compact notation) | ✅ `tool_call_format='json'` option |
| Tool response truncation | ❌ Not implemented | ✅ `max_tool_response_chars` |

---

## 4. Recommendations

> [!IMPORTANT]
> **Never remove the `--chat-template-file` overrides.** The embedded templates are fundamentally incompatible with agentic workflows. The Gemma E4B embedded template is especially dangerous — it predates the OpenAI tool-calling convention entirely.

### When to Re-Extract and Re-Diff
Run `extract_templates.py` any time you:
- Download a new GGUF model or a new quantization of an existing model
- Update `llama.cpp` to a new build (the server itself doesn't modify templates, but new GGUF uploads from model authors might include updated ones)
- Switch to a different model publisher's quant of the same base model

### Updating the Gemma Override
Since our override is now built from Google's canonical base:
1. Download the new Google template
2. Diff it against `new_google_chat_template_gemma.jinja` (our upstream reference)
3. Apply Google's changes to `chat_template_gemma.jinja`, preserving all `{#- BACKPORT: ... -#}` blocks
4. Update `new_google_chat_template_gemma.jinja` with the new upstream version

### Updating the Qwen Override
Since we use the community v21.3 directly:
1. Download the new community template
2. Diff against `chat_template_qwen.jinja`
3. Replace if all our features are preserved (check error detection, warnings, thinking bypass, truncation)
4. Retain old version as `fixed_chat_template_qwen_vXX.jinja` for reference

### Future Improvements
1. **Upstream PR**: Submit schema fixes 1–7 from our Gemma backport list to Google — these are pure bug fixes they should accept.
2. **E4B non-thinking suppression**: Consider PR to Google to add the missing non-thinking suppression to the E4B template.

---

## 5. Architectural Stability: Why Qwen Outperforms Gemma

A major finding in our testing is that Qwen models are fundamentally more stable and easier to write templates for than Gemma models, especially for tool calling. This difference stems entirely from how the creators chose to format tool schemas during fine-tuning.

### Qwen: Native JSON Simplicity
Alibaba fine-tuned Qwen to understand raw JSON. When providing available tools to the model, the template simply dumps the JSON array directly into the prompt:
```jinja
{% for tool in tools %}
    {{ tool | tojson }}
{% endfor %}
```
Because the model understands standard JSON natively, the Jinja template is just a thin wrapper.

**The v21.3 community template** builds on this solid foundation with:
1. **JSON tool_call format**: Optional `tool_call_format='json'` for JSON-native tool call serialization.
2. **AST Flattening**: Optimized loop nesting to prevent `llama.cpp`'s Minja parser from bottlenecking on deep AST trees.
3. **Replace Bug Fix**: Swapped `replace()` filters for `.split() | join('')` to bypass a severe C++ bug that randomly dropped text payloads.
4. **Auto-Disable Thinking**: Automatically skips the `<think>` block during tool calls (`auto_disable_thinking_with_tools`), preventing the model from hallucinating plain text instead of structured JSON.
5. **Payload Truncation**: Slices massive tool responses and appends `[TRUNCATED]` to prevent context window explosions.

### Gemma: The Custom Compiler Curse
Google fine-tuned Gemma to expect a highly specific, proprietary string syntax that looks like this:
```text
declaration:function_name{description:<|"|>...<|"|>,parameters:{properties:{...}}}
```
Because no standard API sends data in this format, the Gemma Jinja template is forced to act as a real-time cross-compiler. It relies on a monstrous (100+ line) recursive macro (`format_parameters`) to translate standard JSON into Google's custom dialect on the fly. 

This causes massive downstream issues:
- **Parser Bottlenecks**: `llama.cpp` struggles to execute the deep recursive macros quickly.
- **Fragility**: One missing key in your JSON schema can cause the entire template to crash.
- **Catastrophic Forgetting**: Community fine-tunes (like Uncensored/Aggressive models) easily forget how to format their outputs into this strict proprietary syntax, breaking tool calls completely.

Even Google's latest 2026-07-09 template still relies on this massive recursive compiler. Until Google switches to standard JSON for their base model fine-tuning, Gemma templates will always be inherently more brittle than Qwen templates.

---

## Reference Files

| File | Purpose |
|---|---|
| `chat_template_gemma.jinja` | **Active override** — Google 2026-07-09 base + all backported fixes |
| `chat_template_gemma_no_error_heuristics.jinja` | **Clean override variant** — Google 2026-07-09 base + schema guards + truncation (error heuristics removed) |
| `chat_template_qwen.jinja` | **Active override** — Community froggeric v21.3, used as-is |
| `new_google_chat_template_gemma.jinja` | Upstream reference — Google's 26B canonical (2026-07-09) |
| `new_google_chat_template_gemma_e4b.jinja` | Upstream reference — Google's E4B canonical (2026-07-09) |
| `fixed_chat_template_qwen_v20.jinja` | Historical reference — our previous custom Qwen override |
| `unsloth_chat_template_qwen.jinja` | Reference — stock Unsloth embedded template |
