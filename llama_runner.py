#!/usr/bin/env python3
"""
llama_runner.py -- llabrun: Llama.cpp Lab Runner & Orchestrator
Interactive model selector, benchmark orchestrator, and Router INI preset generator.
"""
import argparse
import copy
import csv
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


# ── Last Run State & Presets Persistence ──────────────────────────────────
LAST_RUN_FILE = Path("bin/.llama_last_run.json")


def save_last_run(run_data: Dict[str, Any]) -> None:
    try:
        LAST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_RUN_FILE.write_text(json.dumps(run_data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        log(f"Warning: Failed to save last run state: {e}")


def load_last_run() -> Optional[Dict[str, Any]]:
    if LAST_RUN_FILE.exists():
        try:
            return json.loads(LAST_RUN_FILE.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def save_preset_to_file(preset_data: Dict[str, Any], presets_path: Optional[Path] = None) -> bool:
    if presets_path is None:
        presets_path = Path("presets.json")

    presets_config: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "version": 1,
        "description": "User and custom 1-click model presets for llabrun.",
        "presets": []
    }

    if presets_path.exists():
        try:
            presets_config = json.loads(presets_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    elif Path("presets.example.json").exists():
        try:
            example_cfg = json.loads(Path("presets.example.json").read_text(encoding="utf-8"))
            presets_config["defaults"] = example_cfg.get("defaults", {})
            presets_config["reasoning_prefixes"] = example_cfg.get("reasoning_prefixes", {})
            presets_config["families"] = example_cfg.get("families", {})
            presets_config["presets"] = []
        except Exception:
            pass

    if "presets" not in presets_config:
        presets_config["presets"] = []

    # Check if duplicate ID exists, replace or append
    existing_idx = next((i for i, p in enumerate(presets_config["presets"]) if p.get("id") == preset_data.get("id")), None)
    if existing_idx is not None:
        presets_config["presets"][existing_idx] = preset_data
    else:
        presets_config["presets"].append(preset_data)

    presets_path.write_text(json.dumps(presets_config, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


# ── Benchmark Tool Definitions ─────────────────────────────────────────────

BENCHMARK_TOOLS = []

BENCHMARK_SYSTEM_PROMPT = 'You are an expert Senior Frontend Developer.'


def log(msg: str) -> None:
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f'[{ts}] {msg}')


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def context_tokens(label: str) -> int:
    mapping = {
        'auto': 0, '32k': 32768, '65k': 65536, '96k': 98304, '128k': 131072,
        '140k': 143360, '160k': 163840, '200k': 204800,
        '256k': 262144, '262k': 262144, '512k': 524288, '1024k': 1048576, '1M': 1048576
    }
    if label not in mapping:
        raise ValueError(f'Unsupported context label: {label}')
    return mapping[label]


def detect_family(name: str) -> str:
    low = name.lower()
    if 'qwen' in low:
        return 'Qwen'
    if 'gemma' in low:
        return 'Gemma'
    if 'muse' in low or 'glimmer' in low:
        return 'Muse'
    if 'nemotron' in low:
        return 'Nemotron'
    return 'Generic'


def scan_models(models_root: Path) -> List[Dict[str, Any]]:
    if not models_root.exists():
        raise FileNotFoundError(f'Models root not found: {models_root}')
    families: List[Dict[str, Any]] = []
    for folder in sorted([p for p in models_root.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        ggufs = sorted(folder.glob('*.gguf'), key=lambda p: p.name.lower())
        if not ggufs:
            continue
        mmproj = next((p for p in ggufs if p.name.lower().startswith('mmproj-') or '.mmproj-' in p.name.lower()), None)
        variants = [p for p in ggufs if p != mmproj and not p.name.lower().startswith('mmproj-')]
        if not variants:
            continue
        families.append({
            'base_model': folder.name,
            'base_path': str(folder),
            'family': detect_family(folder.name),
            'mmproj': str(mmproj) if mmproj else None,
            'variants': [
                {
                    'variant': p.stem,
                    'model_path': str(p),
                }
                for p in variants
            ],
        })
    return families


def resolve_template_file(filename: str, models_root: Path) -> str:
    templates_root = models_root.parent / 'templates'
    if (templates_root / filename).exists():
        return str(templates_root / filename)
    if (models_root / filename).exists():
        return str(models_root / filename)
    return str(templates_root / filename)


def load_settings(settings_path: Optional[Path] = None) -> Dict[str, Any]:
    default_settings: Dict[str, Any] = {
        'server': {
            'host': '127.0.0.1', 'port': 8080, 'api_key': '', 'ui': True,
            'metrics': True, 'slots_endpoint': True, 'timeout': 600, 'verbosity': 4, 'predict': -1
        },
        'paths': {
            'models_dir': 'models', 'presets_file': 'presets.json',
            'prompt_logs_dir': 'prompt_logs', 'bench_results_dir': 'bench-results'
        },
        'engine_defaults': {
            'threads': 8, 'threads_batch': 8, 'batch_size': 2048, 'ubatch_size': 512,
            'cache_type_k': 'q4_0', 'cache_type_v': 'q4_0', 'flash_attn': 'on',
            'cache_ram': 0, 'ctx_checkpoints': 4, 'context_shift': True,
            'reasoning': 'on', 'reasoning_budget': 8192,
            'temp': 0.7, 'top_p': 0.95, 'top_k': 40, 'min_p': 0.0,
            'presence_penalty': 0.0, 'repeat_penalty': 1.0, 'load_mode': 'none'
        }
    }

    script_dir = Path(__file__).resolve().parent
    candidates = [settings_path] if settings_path else [
        Path('settings.json'),
        script_dir / 'settings.json',
        Path('settings.example.json'),
        script_dir / 'settings.example.json'
    ]
    for cand in candidates:
        if cand and cand.exists():
            try:
                data = json.loads(cand.read_text(encoding='utf-8'))
                res = deep_merge(default_settings, data)
                if 'server' in res and 'port' in res['server']:
                    try:
                        res['server']['port'] = int(res['server']['port'])
                    except (ValueError, TypeError):
                        pass
                return res
            except Exception as e:
                log(f"Warning: Failed to load settings from {cand}: {e}")
    return default_settings


def load_runner_config(config_path: Optional[Path] = None, settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if settings is None:
        settings = load_settings()

    srv_defaults = copy.deepcopy(settings.get('server', {}))
    eng_defaults = copy.deepcopy(settings.get('engine_defaults', {}))

    default_config: Dict[str, Any] = {
        'defaults': {
            'server': srv_defaults,
            'engine': eng_defaults
        },
        'reasoning_prefixes': {
            'Low': "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.\n\n",
            'High': "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n",
        },
        'families': {
            'Gemma': {
                'system_prompt': 'You are Gemma, a large language model created by Google.',
                'template_file': 'chat_template_gemma.jinja',
                'sampling': {'temp': 0.6, 'top_p': 0.95, 'top_k': 20},
                'vision_engine_overrides': {'image_min_tokens': 560, 'image_max_tokens': 1120, 'mtmd_batch_max_tokens': 1120, 'ubatch_size': 2048},
                'reasoning_budget_message': '\nI will now provide my response.\n\n'
            },
            'Qwen': {
                'system_prompt': 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
                'template_file': 'chat_template_qwen.jinja',
                'sampling': {'temp': 0.6, 'top_p': 0.95, 'top_k': 20},
                'vision_engine_overrides': {'image_min_tokens': 1024, 'image_max_tokens': 1024, 'mtmd_batch_max_tokens': 1024, 'ubatch_size': 2048},
                'reasoning_budget_message': '\n\n',
                'auto_disable_thinking_with_tools': False,
                'non_thinking_presence_penalty': 1.5
            },
            'Muse': {
                'system_prompt': 'You are a helpful AI assistant.',
                'sampling': {'temp': 0.6, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.01}
            },
            'Nemotron': {
                'system_prompt': 'You are a helpful AI assistant.',
                'sampling': {'temp': 0.6, 'top_p': 0.95, 'min_p': 0.01}
            },
            'Generic': {
                'system_prompt': 'You are a helpful assistant.',
                'sampling': {'temp': 0.7, 'top_p': 0.95, 'top_k': 40}
            }
        },
        'presets': []
    }

    candidates = [config_path] if config_path else [Path('presets.json')]
    for cand in candidates:
        if cand and cand.exists():
            try:
                data = json.loads(cand.read_text(encoding='utf-8'))
                return deep_merge(default_config, data)
            except Exception as e:
                log(f"Warning: Failed to load config from {cand}: {e}")
    return default_config


def family_baseline(family: str, models_root: Path, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_runner_config(models_root.parent / 'presets.json')

    srv_defaults = copy.deepcopy(config.get('defaults', {}).get('server', {}))
    eng_defaults = copy.deepcopy(config.get('defaults', {}).get('engine', {}))

    base = {
        'server': srv_defaults,
        'engine': eng_defaults,
        'vision': {
            'mode': 'No',
            'mmproj_path': None,
            'mmproj_offload': None,
        },
        'identity': {
            'base_model': None,
            'family': family,
            'variant': None,
            'model_path': None,
            'context_label': '32k',
            'context_tokens': 32768,
            'reasoning_mode': 'Thinking',
            'mtp_profile': 'Off',
        },
    }

    fam_cfg = config.get('families', {}).get(family, config.get('families', {}).get('Generic', {}))
    if fam_cfg.get('sampling'):
        base['engine'].update(fam_cfg['sampling'])

    chat_kwargs = {'preserve_thinking': True}
    if fam_cfg.get('system_prompt'):
        chat_kwargs['system_prompt'] = fam_cfg['system_prompt']
    if fam_cfg.get('auto_disable_thinking_with_tools') is not None:
        chat_kwargs['auto_disable_thinking_with_tools'] = fam_cfg['auto_disable_thinking_with_tools']
    base['engine']['chat_template_kwargs'] = chat_kwargs

    if fam_cfg.get('template_file'):
        base['engine']['chat_template_file'] = resolve_template_file(fam_cfg['template_file'], models_root)

    return base


def reasoning_override(family: str, mode: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_runner_config()

    prefixes = config.get('reasoning_prefixes', {})
    SI_LOW = prefixes.get('Low', "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.\n\n")
    SI_XHIGH = prefixes.get('High', "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.\n\n")

    prefix = SI_LOW if 'Low' in mode else (SI_XHIGH if 'xHigh' in mode or 'High' in mode else '')
    is_thinking = 'Non-Thinking' not in mode and 'NonThinking' not in mode

    fam_cfg = config.get('families', {}).get(family, config.get('families', {}).get('Generic', {}))
    base_sys = fam_cfg.get('system_prompt', 'You are a helpful assistant.')
    sys_prompt = prefix + base_sys

    if not is_thinking:
        presence_pen = fam_cfg.get('non_thinking_presence_penalty', 0.0)
        return {
            'identity': {'reasoning_mode': 'NonThinking'},
            'engine': {
                'reasoning': 'off', 'reasoning_budget': 0, 'reasoning_format': 'none',
                'temp': 0.7, 'top_p': 0.8, 'top_k': 20,
                'presence_penalty': presence_pen,
                'repeat_penalty': 1.0,
                'chat_template_kwargs': {'preserve_thinking': False, 'system_prompt': sys_prompt}
            }
        }

    cfg = {
        'identity': {'reasoning_mode': mode},
        'engine': {
            'reasoning': 'on',
            'reasoning_budget': 8192,
            'temp': 0.6, 'top_p': 0.95, 'top_k': 20,
            'chat_template_kwargs': {'preserve_thinking': True, 'system_prompt': sys_prompt}
        }
    }
    if fam_cfg.get('reasoning_budget_message'):
        cfg['engine']['reasoning_budget_message'] = fam_cfg['reasoning_budget_message']
    if fam_cfg.get('auto_disable_thinking_with_tools') is not None:
        cfg['engine']['chat_template_kwargs']['auto_disable_thinking_with_tools'] = fam_cfg['auto_disable_thinking_with_tools']

    return cfg


def vision_override(mode: str, mmproj_path: Optional[str], family: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if config is None:
        config = load_runner_config()

    cfg = {'vision': {'mode': mode, 'mmproj_path': mmproj_path, 'mmproj_offload': None}}
    if mode == 'GPU':
        cfg['vision']['mmproj_offload'] = True
    elif mode == 'CPU':
        cfg['vision']['mmproj_offload'] = False
        cfg['engine'] = {'no_mmap': True}

    if family == 'Qwen' and mode == 'CPU':
        cfg['vision']['warning'] = 'Qwen CPU mmproj is strongly discouraged due to heavy image latency.'

    if mode != 'No':
        if 'engine' not in cfg:
            cfg['engine'] = {}
        fam_cfg = config.get('families', {}).get(family, {})
        if fam_cfg.get('vision_engine_overrides'):
            cfg['engine'].update(fam_cfg['vision_engine_overrides'])

    return cfg



def temp_override(profile: str) -> dict:
    if profile in ('Family Default', 'Model Default'):
        return {'identity': {'temp_profile': 'Family Default'}}
    elif 'Custom (' in profile:
        m = re.search(r'Custom \(([0-9\.]+)\)', profile)
        if m:
            val = float(m.group(1))
            return {'identity': {'temp_profile': f'Custom ({val})'}, 'engine': {'temp': val}}
    elif profile == 'Low / Deterministic (0.2)' or profile == 'Low (0.2)':
        return {'identity': {'temp_profile': 'Low (0.2)'}, 'engine': {'temp': 0.2, 'top_k': 20}}
    elif profile in ('Balanced (0.6)', 'Low (0.6)'):
        return {'identity': {'temp_profile': 'Balanced (0.6)'}, 'engine': {'temp': 0.6, 'top_k': 20}}
    elif profile == 'High (1.0)':
        return {'identity': {'temp_profile': 'High (1.0)'}, 'engine': {'temp': 1.0, 'top_k': 40}}
    elif profile in ('High / Creative (1.15)', 'High (1.15)'):
        return {'identity': {'temp_profile': 'High (1.15)'}, 'engine': {'temp': 1.15, 'top_k': 64}}
    try:
        val = float(profile)
        return {'identity': {'temp_profile': f'Custom ({val})'}, 'engine': {'temp': val}}
    except ValueError:
        return {'identity': {'temp_profile': profile}}

def mtp_override(profile: str) -> dict:
    if profile == 'MTP (draft-mtp, n=1)':
        return {'identity': {'mtp_profile': 'MTP (n=1)'}, 'engine': {'spec_type': 'draft-mtp', 'spec_draft_n_max': 1}}
    elif profile == 'MTP (draft-mtp, n=2)':
        return {'identity': {'mtp_profile': 'MTP (n=2)'}, 'engine': {'spec_type': 'draft-mtp', 'spec_draft_n_max': 2}}
    elif profile == 'MTP (draft-mtp, n=4)':
        return {'identity': {'mtp_profile': 'MTP (n=4)'}, 'engine': {'spec_type': 'draft-mtp', 'spec_draft_n_max': 4}}
    return {'identity': {'mtp_profile': 'Off'}, 'engine': {'spec_type': None, 'spec_draft_n_max': None, 'spec_draft_model': None}}

def context_override(label: str) -> Dict[str, Any]:
    return {'identity': {'context_label': label, 'context_tokens': context_tokens(label)}}


def testcase_catalog() -> List[Dict[str, Any]]:
    return [
        {'name': 'baseline', 'engine': {}},  # Uses family_baseline defaults (serve.ps1 match)
        {'name': 'two_slots', 'engine': {'cache_idle_slots': True, 'cache_ram': 8192, 'parallel': 2, 'slot_prompt_similarity': 0.50}},
        {'name': 'context_shift_on', 'engine': {'context_shift': True}},
        {'name': 'q8_0_kv', 'engine': {'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0'}},
        {'name': 'ubatch_256', 'engine': {'ubatch_size': 256}},
        {'name': 'ubatch_1024', 'engine': {'ubatch_size': 1024}},
    ]


def choose_many(items: List[Any], label_fn) -> List[Any]:
    for i, item in enumerate(items, start=1):
        print(f'[{i}] {label_fn(item)}')
    print('[A] All')
    raw = input('Choose one or more (comma-separated numbers, or A): ').strip()
    if not raw or raw.upper() == 'A':
        return list(items)
    selected: List[Any] = []
    for part in raw.split(','):
        part = part.strip()
        if part.isdigit():
            idx = int(part)
            if 1 <= idx <= len(items):
                selected.append(items[idx - 1])
    return selected


def choose_one(values: List[str], title: str) -> str:
    print(f'\n{title}')
    for i, value in enumerate(values, start=1):
        print(f'[{i}] {value}')
    while True:
        raw = input('Choose one number: ').strip()
        if raw.isdigit() and 1 <= int(raw) <= len(values):
            return values[int(raw) - 1]


def json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def build_args(cfg: Dict[str, Any], port: int) -> List[str]:
    ident = cfg['identity']
    server = cfg.get('server', {})
    eng = cfg.get('engine', {})
    vis = cfg.get('vision', {})

    # Core required parameters
    args = [
        '--model', ident['model_path'],
        '--port', str(port),
        '--ctx-size', str(ident['context_tokens']),
    ]

    # Server options (only emit if non-default)
    if server.get('host') and server['host'] != '127.0.0.1':
        args += ['--host', str(server['host'])]
    if server.get('timeout') and server['timeout'] not in (3600, 600):
        args += ['--timeout', str(server['timeout'])]
    elif server.get('timeout') == 600:
        args += ['--timeout', '600']
    if server.get('ui') is False:
        args.append('--no-ui')
    if server.get('metrics'):
        args.append('--metrics')
    if server.get('slots_endpoint'):
        args.append('--slots')
    if server.get('verbosity') is not None and server['verbosity'] != 3:
        args += ['-lv', str(server['verbosity'])]
    if server.get('log_prompts_dir'):
        args += ['--log-prompts-dir', server['log_prompts_dir']]
    if server.get('api_key'):
        args += ['--api-key', str(server['api_key'])]
    elif os.environ.get('LLAMA_API_KEY'):
        args += ['--api-key', os.environ['LLAMA_API_KEY']]

    # Device placement (Multi-GPU / Hybrid backends)
    if eng.get('device'):
        args += ['--device', str(eng['device'])]
    if eng.get('split_mode'):
        args += ['--split-mode', str(eng['split_mode'])]
    if eng.get('tensor_split'):
        args += ['--tensor-split', str(eng['tensor_split'])]
    if vis.get('device') or eng.get('mmproj_device'):
        args += ['--mmproj-device', str(vis.get('device') or eng.get('mmproj_device'))]
    if eng.get('spec_draft_device'):
        args += ['--spec-draft-device', str(eng['spec_draft_device'])]

    # Engine compute & KV cache options
    if eng.get('cache_type_k') and eng['cache_type_k'] != 'f16':
        args += ['--cache-type-k', eng['cache_type_k']]
    if eng.get('cache_type_v') and eng['cache_type_v'] != 'f16':
        args += ['--cache-type-v', eng['cache_type_v']]
    if eng.get('flash_attn') and eng['flash_attn'] != 'auto':
        args += ['--flash-attn', eng['flash_attn']]
    if eng.get('threads') is not None and eng['threads'] != -1:
        args += ['--threads', str(eng['threads'])]
    if eng.get('threads_batch') is not None and eng['threads_batch'] != -1 and eng['threads_batch'] != eng.get('threads'):
        args += ['--threads-batch', str(eng['threads_batch'])]
    if eng.get('batch_size') is not None:
        args += ['--batch-size', str(eng['batch_size'])]
    if eng.get('ubatch_size') is not None:
        args += ['--ubatch-size', str(eng['ubatch_size'])]
    if eng.get('cache_ram') is not None:
        args += ['--cache-ram', str(eng['cache_ram'])]
    if eng.get('ctx_checkpoints') is not None:
        args += ['--ctx-checkpoints', str(eng['ctx_checkpoints'])]
    if eng.get('context_shift'):
        args.append('--context-shift')
    if eng.get('no_kv_offload'):
        args.append('--no-kv-offload')
    # Explicitly pass --parallel to prevent server auto-defaulting to 4 slots (which quad-multiplies VRAM)
    args += ['--parallel', str(eng.get('parallel', 1))]
    if eng.get('ngl') is not None and str(eng['ngl']) not in ('auto', 'all', '-1'):
        args += ['-ngl', str(eng['ngl'])]
    elif eng.get('fit_target') is not None:
        args += ['--fit-target', str(eng['fit_target'])]
    if eng.get('load_mode') is not None:
        args += ['--load-mode', str(eng['load_mode'])]
    if eng.get('predict') is not None and eng['predict'] != -1:
        args += ['--predict', str(eng['predict'])]

    # Sampling parameters
    if eng.get('temp') is not None:
        args += ['--temp', str(eng['temp'])]
    if eng.get('top_p') is not None and eng['top_p'] < 1.0:
        args += ['--top-p', str(eng['top_p'])]
    if eng.get('top_k') is not None and eng['top_k'] not in (0, -1, 40):
        args += ['--top-k', str(eng['top_k'])]
    if eng.get('min_p') is not None and eng['min_p'] > 0.0:
        args += ['--min-p', str(eng['min_p'])]
    if eng.get('presence_penalty') is not None and eng['presence_penalty'] != 0.0:
        args += ['--presence-penalty', str(eng['presence_penalty'])]
    if eng.get('repeat_penalty') is not None and eng['repeat_penalty'] != 1.0:
        args += ['--repeat-penalty', str(eng['repeat_penalty'])]

    # Reasoning parameters
    if eng.get('reasoning') and eng['reasoning'] != 'auto':
        args += ['--reasoning', eng['reasoning']]
    if eng.get('reasoning_budget') is not None and eng['reasoning_budget'] != -1:
        args += ['--reasoning-budget', str(eng['reasoning_budget'])]
    if eng.get('reasoning_format'):
        args += ['--reasoning-format', eng['reasoning_format']]
    if eng.get('reasoning_budget_message') is not None:
        args += ['--reasoning-budget-message', eng['reasoning_budget_message']]

    # Templates & kwargs
    if eng.get('jinja') is False:
        args.append('--no-jinja')
    if eng.get('chat_template_file'):
        args += ['--chat-template-file', eng['chat_template_file']]
    if eng.get('chat_template_kwargs') is not None:
        args += ['--chat-template-kwargs', json_compact(eng['chat_template_kwargs'])]

    # Multimodal / Vision
    if vis.get('mode') != 'No' and vis.get('mmproj_path'):
        args += ['--mmproj', vis['mmproj_path']]
        if vis.get('mmproj_offload') is False:
            args.append('--no-mmproj-offload')
    if eng.get('image_min_tokens') is not None:
        args += ['--image-min-tokens', str(eng['image_min_tokens'])]
    if eng.get('image_max_tokens') is not None:
        args += ['--image-max-tokens', str(eng['image_max_tokens'])]
    if eng.get('mtmd_batch_max_tokens') is not None:
        args += ['--mtmd-batch-max-tokens', str(eng['mtmd_batch_max_tokens'])]

    # Speculative / MTP
    draft_model = eng.get('spec_draft_model')
    if not draft_model and eng.get('spec_type') == 'draft-mtp' and cfg.get('identity', {}).get('model_path'):
        mp = Path(cfg['identity']['model_path'])
        candidates = list(mp.parent.glob('mtp*.gguf'))
        if candidates:
            draft_model = str(candidates[0])

    if draft_model:
        args += ['--spec-draft-model', str(draft_model)]
    if eng.get('spec_type'):
        args += ['--spec-type', str(eng['spec_type'])]
    if eng.get('spec_draft_n_max') is not None:
        args += ['--spec-draft-n-max', str(eng['spec_draft_n_max'])]

    # Advanced caching & slot flags (only if explicitly overridden)
    if eng.get('no_mmap'):
        args += ['--load-mode', 'none']
    if eng.get('cache_prompt') is False:
        args.append('--no-cache-prompt')
    if eng.get('kv_unified') is False:
        args.append('--no-kv-unified')
    if eng.get('cache_idle_slots'):
        args.append('--cache-idle-slots')
    if eng.get('slot_prompt_similarity') is not None and eng['slot_prompt_similarity'] != 0.10:
        args += ['--slot-prompt-similarity', str(eng['slot_prompt_similarity'])]
    if eng.get('cont_batching') is False:
        args.append('--no-cont-batching')
    if eng.get('kv_offload') is False:
        args.append('--no-kv-offload')
    if eng.get('context_shift') is False:
        args.append('--no-context-shift')
    if eng.get('no_host'):
        args.append('--no-host')
    if eng.get('poll') is not None:
        args += ['--poll', str(eng['poll'])]

    return args


def format_command(exe: Path, args: List[str]) -> str:
    parts = [shlex.quote(str(exe))] + [shlex.quote(a) for a in args]
    return ' '.join(parts)


def wait_ready(base_url: str, timeout_sec: int = 180) -> None:
    deadline = time.time() + timeout_sec
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'{base_url}/health', timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception as e:
            last_err = e
            time.sleep(0.75)
    raise RuntimeError(f'Server did not become ready at {base_url}: {last_err}')


def _api_request(base_url: str, payload: Dict[str, Any], timeout: int = 1200) -> Dict[str, Any]:
    """Send a request to the chat completions API."""
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        f'{base_url}/v1/chat/completions', data=data,
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8'))


def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract tool calls from an API response."""
    choices = response.get('choices', [])
    if not choices:
        return []
    msg = choices[0].get('message', {})
    raw = msg.get('tool_calls', [])
    parsed = []
    for tc in raw:
        fn = tc.get('function', {})
        args_raw = fn.get('arguments', '{}')
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {'_raw': args_raw}
        parsed.append({'id': tc.get('id', ''), 'name': fn.get('name', ''), 'args': args})
    return parsed


def get_message_content(response: Dict[str, Any]) -> str:
    choices = response.get('choices', [])
    if not choices:
        return ''
    msg = choices[0].get('message', {})
    content = msg.get('content', '')
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if not isinstance(item, dict):
                parts.append(str(item))
            elif item.get('type') == 'text':
                parts.append(item.get('text', ''))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return ''.join(parts)
    return str(content or '')


def get_raw_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
    choices = response.get('choices', [])
    if not choices:
        return []
    msg = choices[0].get('message', {})
    return msg.get('tool_calls', []) or []


def make_tool_message(name: str, payload: Dict[str, Any], tool_call_id: str = '') -> Dict[str, Any]:
    msg = {'role': 'tool', 'content': json.dumps(payload, ensure_ascii=False)}
    if tool_call_id:
        msg['tool_call_id'] = tool_call_id
    msg['name'] = name
    return msg


def summarize_tool_call(tc: Dict[str, Any]) -> Dict[str, Any]:
    fn = tc.get('function', {}) if isinstance(tc, dict) else {}
    return {
        'id': tc.get('id', ''),
        'type': tc.get('type', ''),
        'name': fn.get('name', ''),
        'arguments_raw': fn.get('arguments', ''),
    }


def extract_code_block(text: str) -> str:
    if not text:
        return ''
    m = re.search(r"```(?:[a-zA-Z0-9_+.-]+)?\n([\s\S]*?)```", text)
    return m.group(1) if m else ''


def infer_file_content_from_text(text: str) -> str:
    if not text:
        return ''
    code = extract_code_block(text)
    candidate = code or text
    low = candidate.lower()
    if '<!doctype' in low or '<html' in low or '</html>' in low:
        return candidate
    if filename_like_source(candidate):
        return candidate
    return ''


def filename_like_source(text: str) -> bool:
    low = text.lower()
    indicators = ['function ', 'const ', 'import ', 'export ', '<html', '<!doctype', 'body {', 'def ']
    return any(tok in low for tok in indicators)


def normalize_for_fuzzy_match(text: str) -> str:
    return '\n'.join(line.rstrip() for line in text.replace('\r\n', '\n').replace('\r', '\n').strip().split('\n'))


def apply_edit_safely(content: str, old_text: str, new_text: str) -> tuple[str, bool, str]:
    if not old_text:
        return content, False, 'Edit target not found in file (empty old_content)'
    if old_text in content:
        return content.replace(old_text, new_text, 1), True, 'Exact edit applied'

    norm_content = normalize_for_fuzzy_match(content)
    norm_old = normalize_for_fuzzy_match(old_text)
    if norm_old and norm_old in norm_content:
        return content, False, f'Edit target found with different whitespace but not applied (fuzzy match, old_len={len(old_text)})'

    old_lines = [line.strip() for line in old_text.splitlines() if line.strip()]
    content_lines = content.splitlines()
    if len(old_lines) >= 2:
        for i in range(0, max(0, len(content_lines) - len(old_lines) + 1)):
            window = [line.strip() for line in content_lines[i:i + len(old_lines)]]
            if window == old_lines:
                return content, False, f'Edit target found with line-normalized match but not applied (old_len={len(old_text)})'

    return content, False, f'Edit target not found in file ({len(old_text)} chars)'


def validate_syntax(content: str, filename: str) -> List[str]:
    """Run basic syntax checks based on file extension. Returns a list of issues found."""
    issues = []
    if not content.strip():
        issues.append('Empty content')
        return issues
        
    low = content.lower()
    filename_low = filename.lower()
    
    if filename_low.endswith('.html') or '<html' in low:
        # Check script/style closure for HTML
        if '<script' in low:
            opens = low.count('<script')
            closes = low.count('</script>')
            if opens != closes:
                issues.append(f'Script tag mismatch: {opens} opens vs {closes} closes')
        if '<style' in low:
            opens = low.count('<style')
            closes = low.count('</style>')
            if opens != closes:
                issues.append(f'Style tag mismatch: {opens} opens vs {closes} closes')
                
    # We could add generic bracket balancing here if needed, but it's prone 
    # to false positives with strings/comments. We'll stick to tag balancing.
    return issues


def apply_edit(content: str, old_text: str, new_text: str) -> tuple[str, bool, str]:
    """Apply an edit_file operation. Returns (new_content, success, message)."""
    if old_text in content:
        return content.replace(old_text, new_text, 1), True, 'Edit applied successfully'
    # Try fuzzy match — strip leading/trailing whitespace per line
    old_stripped = '\n'.join(l.strip() for l in old_text.splitlines())
    content_stripped = '\n'.join(l.strip() for l in content.splitlines())
    if old_stripped in content_stripped:
        return content, False, f'Edit target found with different whitespace but not applied (fuzzy match, old_len={len(old_text)})'
    return content, False, f'Edit target not found in file ({len(old_text)} chars)' if old_text else 'Edit target not found in file (empty old_content)'


def benchmark_conversation(base_url: str, engine_cfg: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    """Run a single-turn benchmark conversation.
    Captures full raw responses and exact content.
    """
    messages = [
        {'role': 'system', 'content': BENCHMARK_SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt_text},
    ]
    
    payload = {
        'model': 'default',
        'messages': messages,
        'temperature': engine_cfg.get('temp', 0.8),
        'top_p': engine_cfg.get('top_p', 0.95),
        'top_k': engine_cfg.get('top_k', 40),
        'min_p': engine_cfg.get('min_p', 0.0),
        'presence_penalty': engine_cfg.get('presence_penalty', 0.0),
        'frequency_penalty': max(0, engine_cfg.get('repeat_penalty', 1.0) - 1.0),
    }

    t0 = time.perf_counter()
    resp1 = _api_request(base_url, payload)
    t1_wall = time.perf_counter() - t0
    
    timings1 = resp1.get('timings', {})
    usage1 = resp1.get('usage', {})
    assistant_content = get_message_content(resp1)
    
    file_content = infer_file_content_from_text(assistant_content)

    return {
        'turn1': {
            'wall_s': round(t1_wall, 2),
            'prompt_t_s': round(timings1.get('prompt_per_second', 0), 1),
            'gen_t_s': round(timings1.get('predicted_per_second', 0), 1),
            'prompt_tokens': usage1.get('prompt_tokens', 0),
            'completion_tokens': usage1.get('completion_tokens', 0),
            'response': resp1,
        },
        'assistant_content_t1': assistant_content,
        'file_v1': file_content,
        'html_issues_v1': validate_syntax(file_content, 'file.html'),
    }


def filter_server_log(pipe, filepath: Path) -> None:
    exclude_phrases = [
        'res          send: sending result for task id',
        'res          send: task id',
        'slot process_toke: id',
        'srv  update_slots: run slots completed',
        'que    start_loop: waiting for new tasks',
        'que    start_loop: processing new tasks',
        'que    start_loop: processing task',
        'que    start_loop: update slots',
        'srv  update_slots: posting NEXT_RESPONSE',
        'que          post: new task',
        'slot update_batch: id',
        'srv  update_slots: decoding batch',
        'set_adapters_lora: adapters',
        'adapters_lora_are_same: adapters',
        'set_embeddings: value =',
        'http: streamed chunk: event: content_block_delta',
        'http: streamed chunk: event: message_delta',
        'http: streamed chunk: event: ping',
        'http: streamed chunk: event: content_block_start',
        'http: streamed chunk: event: content_block_stop',
        'http: streamed chunk: event: message_start',
        'http: streamed chunk: event: message_stop',
    ]
    with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
        skip_next = False
        for line in pipe:
            if skip_next:
                skip_next = False
                continue
            
            skip = False
            for p in exclude_phrases:
                if p in line:
                    skip = True
                    if 'streamed chunk: event: ' in line:
                        skip_next = True
                    break
            
            if not skip:
                f.write(line)
                f.flush()


def parse_server_log(log_path: Path) -> Dict[str, Any]:
    """Parse a verbose (level 4) server log for memory and offloading info."""
    info: Dict[str, Any] = {
        'gpu_name': '', 'gpu_vram_total': '', 'gpu_vram_free': '',
        'gpu_allocated_total': '', 'gpu_vram_used_est': '', 'gpu_vram_free_est': '', 'gpu_os_used': '',
        'model_file': '', 'mmproj_file': '', 'model_name': '', 'model_file_type': '', 'mmproj_usage': '',
        'model_buffer_gpu': '', 'model_buffer_cpu': '', 'model_buffer_cuda_host': '',
        'layers_offloaded': '', 'kv_cache_size': '',
        'kv_cache_k': '', 'kv_cache_v': '',
        'model_size': '', 'model_type': '', 'model_params': '',
        'compute_gpu': '', 'compute_cpu': '',
        'memory_breakdown': '',
    }
    if not log_path.exists():
        return info
    try:
        lines = log_path.read_text(encoding='utf-8', errors='replace').splitlines()
    except Exception:
        return info

    for line in lines:
        # GGUF Model and Multimodal Projector filenames
        if 'srv    load_model: loading model' in line:
            m = re.search(r"loading model\s*'([^']+)'", line)
            if m:
                info['model_file'] = Path(m.group(1)).name
        if 'srv    load_model: loaded multimodal model' in line:
            m = re.search(r"loaded multimodal model,\s*'([^']+)'", line)
            if m:
                info['mmproj_file'] = Path(m.group(1)).name
        if 'print_info: file type' in line:
            m = re.search(r'file type\s*=\s*(.+)', line)
            if m:
                info['model_file_type'] = m.group(1).strip()
        if 'general.name' in line and 'str' in line:
            m = re.search(r'general\.name\s+str\s+=\s+(.+)', line)
            if m:
                info['model_name'] = m.group(1).strip()
        # Multimodal projector memory usage
        if 'estimated worst-case memory usage of mmproj is' in line:
            m = re.search(r'usage of mmproj is\s*([\d.]+\s*MiB)', line)
            if m:
                info['mmproj_usage'] = m.group(1).strip()
        # MTP context estimation
        if '[spec] estimated memory usage of MTP context' in line:
            m = re.search(r'is\s*([\d.]+\s*MiB)', line)
            if m:
                info['mtp_usage'] = m.group(1).strip()
        # Device info: GPU name and free memory
        if 'CUDA0' in line and 'MiB' in line and 'common_param:' in line:
            m = re.search(r'(NVIDIA[^(]+)\((\d+)\s*MiB,\s*(\d+)\s*MiB free\)', line)
            if m:
                info['gpu_name'] = m.group(1).strip()
                info['gpu_vram_total'] = f'{m.group(2)} MiB'
                info['gpu_vram_free'] = f'{m.group(3)} MiB'
        # Server fit report (supports dense models and MoE models with overflowing expert layers)
        if 'common_params_fit_impl:' in line and 'CUDA0' in line and 'layers' in line and 'MiB used' in line:
            m = re.search(r'(\d+)\s*layers(?:[^,]*),\s*([\d.]+)\s*MiB used,\s*([\d.]+)\s*MiB free', line)
            if m:
                info['fit_layers'] = m.group(1)
                info['fit_gpu_used'] = f'{m.group(2)} MiB'
                info['fit_gpu_free'] = f'{m.group(3)} MiB'
        # Memory breakdown table
        if 'common_memory_breakdown_print' in line and 'CUDA0' in line:
            info['memory_breakdown'] = line.split('common_memory_breakdown_print:')[-1].strip() if 'common_memory_breakdown_print:' in line else line.strip()
        if 'common_memory_breakdown_print' in line and 'Host' in line:
            host_part = line.split('common_memory_breakdown_print:')[-1].strip() if 'common_memory_breakdown_print:' in line else line.strip()
            info['memory_breakdown'] = (info.get('memory_breakdown', '') + '\n' + host_part).strip()
        # Layer offloading
        if 'load_tensors: offloaded' in line:
            m = re.search(r'offloaded (\d+/\d+) layers', line)
            if m:
                info['layers_offloaded'] = m.group(1)
        # Buffer sizes
        if 'load_tensors:' in line and 'CUDA0 model buffer size' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                info['model_buffer_gpu'] = m.group(1)
        if 'load_tensors:' in line and 'CUDA_Host model buffer size' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                info['model_buffer_cuda_host'] = m.group(1)
        if 'load_tensors:' in line and ('CPU model buffer size' in line or 'CPU_Mapped model buffer size' in line):
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                info['model_buffer_cpu'] = m.group(1)
        # KV cache (track main vs MTP and totals)
        if 'llama_kv_cache:' in line and 'CUDA0 KV buffer size' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                val = float(m.group(1).replace('MiB', '').strip())
                info['kv_cuda_total'] = f"{float(info.get('kv_cuda_total', '0').replace('MiB', '').strip()) + val:.2f} MiB"
        if 'llama_kv_cache:' in line and 'CPU KV buffer size' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                val = float(m.group(1).replace('MiB', '').strip())
                info['kv_cpu_total'] = f"{float(info.get('kv_cpu_total', '0').replace('MiB', '').strip()) + val:.2f} MiB"
        if 'llama_kv_cache:' in line and 'size =' in line:
            m = re.search(r'size =\s*([\d.]+\s*MiB)', line)
            if m:
                info['kv_cache_size'] = m.group(1)
            mk = re.search(r'K \(([^)]+)\):\s*([\d.]+\s*MiB)', line)
            mv = re.search(r'V \(([^)]+)\):\s*([\d.]+\s*MiB)', line)
            if mk and 'kv_cache_k' not in info:
                info['kv_cache_k'] = f'{mk.group(1)}: {mk.group(2)}'
            if mv and 'kv_cache_v' not in info:
                info['kv_cache_v'] = f'{mv.group(1)}: {mv.group(2)}'
        # Recurrent state buffers (SSM / DeltaNet / RWKV)
        if 'llama_memory_recurrent:' in line and 'CUDA0 RS buffer size' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                info['rs_gpu'] = m.group(1)
        if 'llama_memory_recurrent:' in line and 'CPU RS buffer size' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                info['rs_cpu'] = m.group(1)
        # Compute buffers
        if 'sched_reserve:' in line and 'CUDA0 compute buffer' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                val = float(m.group(1).replace('MiB', '').strip())
                curr = float(info['compute_gpu'].replace('MiB', '').strip()) if info.get('compute_gpu') else 0.0
                info['compute_gpu'] = f'{max(val, curr):.2f} MiB'
        if 'sched_reserve:' in line and 'CUDA_Host compute buffer' in line:
            m = re.search(r'=\s*([\d.]+\s*MiB)', line)
            if m:
                val = float(m.group(1).replace('MiB', '').strip())
                curr = float(info['compute_cpu'].replace('MiB', '').strip()) if info.get('compute_cpu') else 0.0
                info['compute_cpu'] = f'{max(val, curr):.2f} MiB'
        # Model info
        if 'print_info: file size' in line:
            m = re.search(r'file size\s*=\s*(.+)', line)
            if m:
                info['model_size'] = m.group(1).strip()
        if 'print_info: model type' in line:
            m = re.search(r'model type\s*=\s*(.+)', line)
            if m:
                info['model_type'] = m.group(1).strip()
        if 'print_info: model params' in line:
            m = re.search(r'model params\s*=\s*(.+)', line)
            if m:
                info['model_params'] = m.group(1).strip()

    # Calculate estimated actual GPU footprint and post-load free VRAM
    def _parse_mib(val: Any) -> float:
        if not val:
            return 0.0
        try:
            return float(str(val).replace('MiB', '').strip())
        except (ValueError, TypeError):
            return 0.0

    try:
        model_gpu = _parse_mib(info.get('model_buffer_gpu'))
        kv_gpu = _parse_mib(info.get('kv_cuda_total')) or _parse_mib(info.get('kv_cache_size'))
        rs_gpu = _parse_mib(info.get('rs_gpu'))
        comp_gpu = _parse_mib(info.get('compute_gpu'))
        mmproj_gpu = _parse_mib(info.get('mmproj_usage'))
        
        total_gpu_alloc = model_gpu + kv_gpu + rs_gpu + comp_gpu + mmproj_gpu
        if total_gpu_alloc > 0:
            info['gpu_allocated_total'] = f'{total_gpu_alloc:.2f} MiB'

        total_vram_m = re.search(r'([\d.]+)\s*MiB', str(info.get('gpu_vram_total', '')))
        free_pre_m = re.search(r'([\d.]+)\s*MiB', str(info.get('gpu_vram_free', '')))
        if total_vram_m and free_pre_m:
            total_vram = float(total_vram_m.group(1))
            free_pre = float(free_pre_m.group(1))
            os_used = max(0.0, total_vram - free_pre)
            est_free = max(0.0, total_vram - (os_used + total_gpu_alloc))
            info['gpu_vram_used_est'] = f'{(os_used + total_gpu_alloc):.0f} MiB'
            info['gpu_vram_free_est'] = f'~{est_free:.0f} MiB'
            info['gpu_os_used'] = f'{os_used:.0f} MiB'
    except Exception:
        pass

    return info


def print_results_table(results: List[Dict[str, Any]]) -> None:
    """Print a formatted results table after sweep completes."""
    if not results:
        return
    print(f'\n  ========== BENCHMARK RESULTS ==========')
    print(f'  {"Variant":<28} {"Ctx":<5} {"Vis":<4} {"MTP":<9} {"Temp":<11} {"T1 t/s":<8} {"T2 t/s":<8} {"T1 tok":<8} {"T2 tok":<8} {"Wall":<8} {"Tools":<6} {"Layers":<8}')
    print(f'  {"-" * 134}')
    for r in results:
        status = 'OK' if r['status'] == 'ok' else 'FAIL'
        variant = str(r.get('variant', ''))[:27]
        ctx = str(r.get('context', ''))
        vis = str(r.get('vision', 'No'))[:3]
        mtp = str(r.get('mtp', 'Off'))[:8]
        temp = str(r.get('temp_profile', '-'))[:11]
        t1_tg = str(r.get('t1_gen_t_s', '-'))
        t2_tg = str(r.get('t2_gen_t_s', '-'))
        t1_tok = str(r.get('t1_tokens', '-'))
        t2_tok = str(r.get('t2_tokens', '-'))
        wall = str(r.get('total_wall_s', '-'))
        tools = 'YES' if r.get('used_tools') else 'NO'
        layers = str(r.get('layers_offloaded', '-'))
        print(f'  {variant:<28} {ctx:<5} {vis:<4} {mtp:<9} {temp:<11} {t1_tg:<8} {t2_tg:<8} {t1_tok:<8} {t2_tok:<8} {wall:<8} {tools:<6} {layers:<8}')
    print(f'  {"=" * 134}\n')


def fetch_optional(url: str) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except Exception:
        return None


def print_summary(cfg: Dict[str, Any], testcase_name: str, command: str) -> None:
    ident = cfg['identity']
    eng = cfg['engine']
    vis = cfg['vision']
    print(f'\n=== Resolved Configuration ===')
    print(f"Base model:        {ident.get('base_model', 'Unknown')}")
    print(f"Family:            {ident.get('family', 'Unknown')}")
    print(f"Variant:           {ident.get('variant', 'Unknown')}")
    print(f"Model path:        {ident.get('model_path', 'Unknown')}")
    print(f"Context:           {ident.get('context_label', 'auto')} ({ident.get('context_tokens', 'auto')})")
    print(f"Vision:            {vis.get('mode', 'No')}")
    print(f"mmproj:            {vis.get('mmproj_path') or 'None'}")
    print(f"mmproj offload:    {vis.get('mmproj_offload')}")
    print(f"Reasoning mode:    {ident.get('reasoning_mode', 'Unknown')}")
    print(f"Reasoning:         {eng.get('reasoning', 'auto')}")
    print(f"Reasoning budget:  {eng.get('reasoning_budget', 'default')}")
    print(f"Reasoning format:  {eng.get('reasoning_format')}")
    print(f"Chat kwargs:       {eng.get('chat_template_kwargs')}")
    print(f"Temp / TopP / TopK:{eng.get('temp', 0.6)} / {eng.get('top_p', 0.95)} / {eng.get('top_k', 20)}")
    print(f"Presence / Repeat: {eng.get('presence_penalty', 0.0)} / {eng.get('repeat_penalty', 1.0)}")
    if eng.get('spec_type'):
        print(f"Speculative / MTP: {eng['spec_type']} (draft_n_max={eng.get('spec_draft_n_max', 1)})")
    else:
        print(f"Speculative / MTP: Off")
    print(f"Cache K/V:         {eng.get('cache_type_k', 'q4_0')} / {eng.get('cache_type_v', 'q4_0')}")
    print(f"FlashAttn:         {eng.get('flash_attn', 'on')}")
    print(f"CachePrompt:       {eng.get('cache_prompt', 'on (default)')}")
    print(f"KV Unified:        {eng.get('kv_unified', 'on (default)')}")
    print(f"Cache Idle Slots:  {eng.get('cache_idle_slots', 'off (default)')}")
    print(f"Cache RAM:         {eng.get('cache_ram', 'auto')}")
    print(f"Slot similarity:   {eng.get('slot_prompt_similarity', '0.1 (default)')}")
    print(f"Cont batching:     {eng.get('cont_batching', 'on (default)')}")
    print(f"KV offload:        {eng.get('kv_offload', 'on (default)')}")
    print(f"Context shift:     {eng.get('context_shift', 'off')}")
    print(f"Test case:         {testcase_name}")
    print(f'\nFULL COMMAND:')
    print(command)
    print()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames_set = set()
    for r in rows:
        fieldnames_set.update(r.keys())
    
    # Keep some standard columns first, then append the rest
    standard = ['variant', 'context', 'vision', 'mtp', 'temp_profile', 'status', 'wall_time_s', 'prompt_t_s', 'gen_t_s', 'vram_used', 'vram_free', 'completion_tokens', 'error']
    fieldnames = [f for f in standard if f in fieldnames_set] + [f for f in fieldnames_set if f not in standard]

    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval='')
        writer.writeheader()
        writer.writerows(rows)


def kill_existing_servers() -> None:
    """Gracefully terminate any running llama-server processes with force fallback."""
    try:
        if sys.platform == 'win32':
            # Try graceful taskkill first to allow socket & memory cleanup
            res = subprocess.run(['taskkill', '/IM', 'llama-server.exe'], capture_output=True, text=True)
            if res.returncode == 0:
                time.sleep(1)
            # Ensure full cleanup if still lingering
            subprocess.run(['taskkill', '/F', '/IM', 'llama-server.exe'], capture_output=True, text=True)
        else:
            subprocess.run(['pkill', '-15', '-f', 'llama-server'], capture_output=True)
            time.sleep(0.5)
            subprocess.run(['pkill', '-9', '-f', 'llama-server'], capture_output=True)
        time.sleep(0.5)
    except Exception:
        pass


def run_diagnostics(port: int, log_file: str) -> None:
    """Background thread: wait for server health, then print log-parsed memory info."""
    time.sleep(5)
    try:
        wait_ready(f'http://127.0.0.1:{port}', 180)
    except Exception:
        return
    time.sleep(2)

    print(f'\n  ========== MEMORY AFTER LOAD ==========')
    info = parse_server_log(Path(str(log_file)))
    if info.get('gpu_name'):
        total_str = f' ({info["gpu_vram_total"]} Total)' if info.get('gpu_vram_total') else ''
        print(f'  GPU:            {info["gpu_name"]}{total_str}')
    if info.get('model_file'):
        ft_str = f'  ({info["model_file_type"]})' if info.get('model_file_type') else ''
        print(f'  Model GGUF:     {info["model_file"]}{ft_str}')
    if info.get('mmproj_file'):
        print(f'  MMProj GGUF:    {info["mmproj_file"]}')
    if info.get('model_name'):
        print(f'  Model Name:     {info["model_name"]}')
    if info.get('layers_offloaded'):
        offloaded_note = f' (Split between GPU & CPU RAM)' if info.get('model_buffer_cpu') else ' (Fully on GPU)'
        print(f'  Layers:         {info["layers_offloaded"]}{offloaded_note}')
    if info.get('model_params'):
        print(f'  Params:         {info["model_params"]}')
    if info.get('model_size'):
        print(f'  File:           {info["model_size"]}')
    if info.get('model_buffer_gpu'):
        cpu_m_str = f' | {info["model_buffer_cpu"]} (CPU RAM)' if info.get('model_buffer_cpu') else ''
        host_str = f' | {info["model_buffer_cuda_host"]} (CUDA Host)' if info.get('model_buffer_cuda_host') else ''
        print(f'  Model Weights:  {info["model_buffer_gpu"]} (GPU){cpu_m_str}{host_str}')
    if info.get('mmproj_usage'):
        print(f'  MMProj / Vision:{info["mmproj_usage"]} (GPU worst-case est.)')
    if info.get('mtp_usage'):
        print(f'  MTP Draft Spec: {info["mtp_usage"]}')
    if info.get('kv_cuda_total'):
        cpu_kv_str = f' | {info["kv_cpu_total"]} (CPU)' if info.get('kv_cpu_total') else ''
        print(f'  KV Cache:       {info["kv_cuda_total"]} (GPU){cpu_kv_str}  [Format: {info.get("kv_cache_k", "K: q4_0")}, {info.get("kv_cache_v", "V: q4_0")}]')
    elif info.get('kv_cache_size'):
        print(f'  KV Cache:       {info["kv_cache_size"]}  (K: {info.get("kv_cache_k", "")}, V: {info.get("kv_cache_v", "")})')
    if info.get('rs_gpu'):
        rs_cpu_str = f' | {info["rs_cpu"]} (CPU)' if info.get('rs_cpu') else ''
        print(f'  Recurrent (SSM):{info["rs_gpu"]} (GPU){rs_cpu_str}')
    if info.get('compute_gpu'):
        cpu_comp_str = f' | {info["compute_cpu"]} (CPU)' if info.get('compute_cpu') else ''
        print(f'  Compute Buffer: {info["compute_gpu"]} (GPU){cpu_comp_str}')
    if info.get('fit_gpu_used'):
        print(f'  ----------------------------------------')
        print(f'  Server GPU Load:{info["fit_gpu_used"]} (Allocated on GPU by llama-server)')
    elif info.get('gpu_allocated_total'):
        os_str = f' | OS Baseline: {info["gpu_os_used"]}' if info.get('gpu_os_used') else ''
        print(f'  ----------------------------------------')
        print(f'  Total GPU Used: ~{info.get("gpu_vram_used_est", info["gpu_allocated_total"])}  (Server Load: {info["gpu_allocated_total"]}{os_str})')
    if info.get('fit_gpu_free'):
        print(f'  Available VRAM: ~{info["fit_gpu_free"]} free')
    elif info.get('gpu_vram_free_est'):
        print(f'  Est. Free VRAM: {info["gpu_vram_free_est"]}')
    print('  ========================================\n')


def write_summary(out_path: Path, results: List[Dict[str, Any]], prompt_text: str) -> None:
    """Generate a summary.md combining all run stats and generated content."""
    lines = []
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    lines.append(f'# Benchmark Summary')
    lines.append(f'')
    lines.append(f'**Generated:** {ts}  ')
    lines.append(f'**Total Runs:** {len(results)}  ')
    lines.append(f'')
    lines.append(f'## Comparison Table')
    lines.append(f'')
    lines.append(f'| Variant | Ctx | Vision | MTP | Temp | Gen t/s | Tokens | Total wall | Layers | HTML Issues |')
    lines.append(f'|---|---|---|---|---|---|---|---|---|---|')
    for r in results:
        v1_issues = r.get('html_issues_v1', 0)
        lines.append(f'| {r.get("variant", "")} | {r.get("context", "")} | {r.get("vision", "No")} | {r.get("mtp", "Off")} | {r.get("temp_profile", "-")} | {r.get("gen_t_s", "-")} | {r.get("completion_tokens", "-")} | {r.get("wall_time_s", "-")}s | {r.get("layers_offloaded", "-")} | {v1_issues} |')
    lines.append(f'')

    # Per-run details
    for i, r in enumerate(results, 1):
        lines.append(f'---')
        lines.append(f'')
        lines.append(f'## Run {i}: {r.get("variant", "unknown")}')
        lines.append(f'')
        lines.append(f'### Settings')
        lines.append(f'')
        lines.append(f'| Setting | Value |')
        lines.append(f'|---|---|')
        lines.append(f'| Context | {r.get("context", "")} |')
        lines.append(f'| Vision | {r.get("vision", "No")} |')
        lines.append(f'| MTP / Speculative | {r.get("mtp", "Off")} |')
        lines.append(f'| Reasoning | {r.get("reasoning_mode", "")} |')
        lines.append(f'| Status | {r.get("status", "")} |')
        lines.append(f'')

        # Memory info
        mem = r.get('memory_info', {})
        if mem:
            lines.append(f'### Memory')
            lines.append(f'')
            lines.append(f'| Component | Value |')
            lines.append(f'|---|---|')
            if mem.get('gpu_name'): lines.append(f'| GPU | {mem["gpu_name"]} |')
            if mem.get('gpu_vram_total'): lines.append(f'| VRAM Total | {mem["gpu_vram_total"]} |')
            if mem.get('gpu_vram_used_est'): lines.append(f'| Est. Total GPU Used | {mem["gpu_vram_used_est"]} |')
            if mem.get('gpu_allocated_total'): lines.append(f'| Model + KV + Compute | {mem["gpu_allocated_total"]} |')
            if mem.get('gpu_vram_free_est'): lines.append(f'| Est. Free VRAM | {mem["gpu_vram_free_est"]} |')
            elif mem.get('gpu_vram_free'): lines.append(f'| VRAM Free (pre-load) | {mem["gpu_vram_free"]} |')
            if mem.get('model_file'): lines.append(f'| Model GGUF | {mem["model_file"]} |')
            if mem.get('mmproj_file'): lines.append(f'| MMProj GGUF | {mem["mmproj_file"]} |')
            if mem.get('model_name'): lines.append(f'| Model Name | {mem["model_name"]} |')
            if mem.get('model_params'): lines.append(f'| Parameters | {mem["model_params"]} |')
            if mem.get('model_size'): lines.append(f'| File Size | {mem["model_size"]} |')
            if mem.get('layers_offloaded'): lines.append(f'| Layers Offloaded | {mem["layers_offloaded"]} |')
            if mem.get('model_buffer_gpu'): lines.append(f'| GPU Model Buffer | {mem["model_buffer_gpu"]} |')
            if mem.get('model_buffer_cpu'): lines.append(f'| CPU Model Buffer | {mem["model_buffer_cpu"]} |')
            if mem.get('mmproj_usage'): lines.append(f'| MMProj / Vision GPU | {mem["mmproj_usage"]} |')
            if mem.get('kv_cache_size'): lines.append(f'| KV Cache | {mem["kv_cache_size"]} (K: {mem["kv_cache_k"]}, V: {mem["kv_cache_v"]}) |')
            if mem.get('compute_gpu'): lines.append(f'| GPU Compute | {mem["compute_gpu"]} |')
            if mem.get('compute_cpu'): lines.append(f'| CPU Compute | {mem["compute_cpu"]} |')
            lines.append(f'')

        # Performance
        lines.append(f'### Performance')
        lines.append(f'')
        lines.append(f'| Metric | Generate |')
        lines.append(f'|---|---|')
        lines.append(f'| Wall time | {r.get("wall_time_s", "-")}s |')
        lines.append(f'| Prompt t/s | {r.get("prompt_t_s", "-")} |')
        lines.append(f'| Gen t/s | {r.get("gen_t_s", "-")} |')
        lines.append(f'| Tokens | {r.get("completion_tokens", "-")} |')
        lines.append(f'')

        # Syntax validation
        v1_issues = r.get('html_issues_v1_list', [])
        lines.append(f'### Syntax Validation')
        lines.append(f'')
        lines.append(f'- **HTML issues ({len(v1_issues)}):** {(", ".join(v1_issues)) if v1_issues else "None ✅"}')
        lines.append(f'')

        final_content = r.get('final_file_content', '')
        if final_content:
            lines.append(f'### Output File')
            lines.append(f'')
            lines.append(f'```html')
            lines.append(final_content)
            lines.append(f'```')
            lines.append(f'')

    # Write
    out_path.write_text('\n'.join(lines), encoding='utf-8')


def generate_router_ini_from_presets(presets: List[Dict[str, Any]], out_path: Path) -> None:
    skip_args = {'--host', '--port', '--timeout', '--no-ui', '--metrics', '--slots', '--verbosity'}
    
    def args_to_ini(args_list: List[str]) -> str:
        lines = []
        i = 0
        while i < len(args_list):
            arg = args_list[i]
            if arg in skip_args:
                if i + 1 < len(args_list) and not args_list[i+1].startswith('-'):
                    i += 2
                else:
                    i += 1
                continue
            key = arg.lstrip('-')
            if i + 1 < len(args_list) and not args_list[i+1].startswith('-'):
                val = args_list[i+1]
                val = val.replace('\r\n', '\\n').replace('\n', '\\n')
                lines.append(f"{key} = {val}")
                i += 2
            else:
                lines.append(f"{key} = true")
                i += 1
        return "\n".join(lines)

    def make_clean_alias(cfg: Dict[str, Any], label: str) -> str:
        ident = cfg.get('identity', {})
        base = ident.get('base_model', '').lower()
        ctx = ident.get('context_label', '').lower()
        vis = cfg.get('vision', {}).get('mode', 'No').lower()
        mtp = ident.get('mtp_profile', 'Off')
        
        if '35b' in base or '35b' in label.lower():
            tag = 'qwen3.6-35b-moe'
        elif 'qwen3.8' in base or 'qwen 3.8' in label.lower():
            tag = 'qwen3.8-27b'
        elif 'gemma-4-12b' in base or '12b' in label.lower():
            tag = 'gemma4-12b'
        elif 'gemma-4-26b' in base or '26b' in label.lower():
            tag = 'gemma4-26b-a4b'
        elif 'e4b' in base or 'e4b' in label.lower():
            tag = 'gemma4-e4b'
        elif 'muse' in base or 'muse' in label.lower():
            tag = 'muse-glimmer-30b'
        elif 'nemotron' in base or 'nemotron' in label.lower():
            tag = 'nemotron-3.5-30b'
        else:
            tag = ident.get('family', 'model').lower()
            
        parts = [tag, ctx]
        if vis == 'gpu':
            parts.append('vision')
        elif 'text' in label.lower() or vis == 'no':
            if 'speed' in label.lower() or 'turbo' in label.lower():
                parts.append('speed')
            elif 'dense' in label.lower():
                parts.append('dense')
            elif 'text' in label.lower():
                parts.append('text')
                
        if 'mtp' in mtp.lower() and 'speed' not in parts:
            parts.append('mtp')
            
        return '-'.join(parts)

    ini_sections = ["version = 1\n"]
    seen_aliases = {}

    for idx, p in enumerate(presets, start=1):
        alias = make_clean_alias(p['cfg'], p['label'])
        if alias in seen_aliases:
            seen_aliases[alias] += 1
            alias = f"{alias}-{seen_aliases[alias]}"
        else:
            seen_aliases[alias] = 1
            
        args_list = build_args(p['cfg'], 8080)
        ini_body = args_to_ini(args_list)
        ini_sections.append(f"# [{idx:2d}] {p['label']}\n[{alias}]\n{ini_body}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ini_content = "\n".join(ini_sections)
    out_path.write_text(ini_content, encoding='utf-8')
    
    # Also keep the workspace root router.ini synchronized
    root_router = Path('router.ini')
    if out_path.resolve() != root_router.resolve():
        try:
            root_router.write_text(ini_content, encoding='utf-8')
        except Exception:
            pass

    print(f"\nRouter preset saved to: {out_path} (and synchronized with router.ini)")
    print(f"To run router mode: llama-server.exe --port 8080 --models-preset router.ini\n")
    print("Exported Model Aliases:")
    for s in ini_sections:
        for l in s.splitlines():
            if l.startswith('[') and l.endswith(']'):
                print(f"  - {l[1:-1]}")


def generate_router_ini(families: List[Dict[str, Any]], out_path: Path, models_root: Path) -> None:
    skip_args = {'--host', '--port', '--timeout', '--no-ui', '--metrics', '--slots', '--verbosity'}
    
    def args_to_ini(args_list: List[str]) -> str:
        lines = []
        i = 0
        while i < len(args_list):
            arg = args_list[i]
            if arg in skip_args:
                if i + 1 < len(args_list) and not args_list[i+1].startswith('-'):
                    i += 2
                else:
                    i += 1
                continue
            key = arg.lstrip('-')
            if i + 1 < len(args_list) and not args_list[i+1].startswith('-'):
                val = args_list[i+1]
                val = val.replace('\r\n', '\\n').replace('\n', '\\n')
                lines.append(f"{key} = {val}")
                i += 2
            else:
                lines.append(f"{key} = true")
                i += 1
        return "\n".join(lines)

    ini_sections = ["version = 1\n"]
    
    for fam in families:
        base_name = fam['base_model']
        family_type = fam['family']
        mmproj = fam['mmproj']
        
        for var in fam['variants']:
            variant_name = var['variant']
            is_e4b = 'e4b' in variant_name.lower() or 'e4b' in base_name.lower()
            
            profiles = []
            if family_type == 'Gemma':
                if is_e4b:
                    profiles.append({'name': f"{variant_name}-Thinking", 'reasoning': 'Thinking', 'temp': 'Low (0.6)'})
                else:
                    profiles.append({'name': f"{variant_name}-Thinking", 'reasoning': 'Thinking', 'temp': 'High (1.15)'})
                    profiles.append({'name': f"{variant_name}-Instruct", 'reasoning': 'NonThinking', 'temp_custom': 1.0, 'top_k_custom': 64})
            elif family_type == 'Qwen':
                profiles.append({'name': f"{variant_name}-Thinking", 'reasoning': 'Thinking', 'temp': 'Low (0.6)'})
                profiles.append({'name': f"{variant_name}-Instruct", 'reasoning': 'NonThinking', 'temp_custom': None})
            elif family_type == 'Nemotron':
                profiles.append({'name': f"{variant_name}-Instruct", 'reasoning': 'NonThinking', 'temp_custom': 0.6})
            elif family_type == 'Muse':
                profiles.append({'name': f"{variant_name}-Instruct", 'reasoning': 'NonThinking', 'temp_custom': 1.0, 'top_k_custom': 64})
            else:
                profiles.append({'name': f"{variant_name}-Generic", 'reasoning': 'NonThinking', 'temp_custom': 0.8})
                
            for prof in profiles:
                base_cfg = family_baseline(family_type, models_root)
                base_cfg = deep_merge(base_cfg, {'identity': {
                    'base_model': base_name, 'family': family_type, 
                    'variant': variant_name, 'model_path': var['model_path']
                }})
                
                vis_mode = 'GPU' if mmproj else 'No'
                base_cfg = deep_merge(base_cfg, vision_override(vis_mode, mmproj, family_type))
                base_cfg = deep_merge(base_cfg, context_override('128k'))
                base_cfg = deep_merge(base_cfg, reasoning_override(family_type, prof['reasoning']))
                
                if prof.get('temp'):
                    base_cfg = deep_merge(base_cfg, temp_override(prof['temp']))
                if prof.get('temp_custom') is not None:
                    base_cfg['engine']['temp'] = prof['temp_custom']
                if prof.get('top_k_custom') is not None:
                    base_cfg['engine']['top_k'] = prof['top_k_custom']
                    
                args_list = build_args(base_cfg, 8080)
                ini_body = args_to_ini(args_list)
                
                ini_sections.append(f"[{prof['name']}]\n{ini_body}\n")
                
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(ini_sections), encoding='utf-8')
    print(f"\nRouter preset saved to: {out_path}")
    print(f"To run router mode: llama-server.exe --port 8080 --models-preset {out_path}")



def build_verified_presets(families: List[Dict[str, Any]], models_root: Path, presets_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    presets = []

    def find_var(match_str: str, preferred_quant_substrings: List[str]):
        m_low = match_str.lower()
        fam = next((f for f in families if m_low == f['family'].lower() or m_low in f['base_model'].lower()), None)
        if not fam:
            return None, None
        for substr in preferred_quant_substrings:
            v = next((v for v in fam['variants'] if substr.lower() in v['variant'].lower()), None)
            if v:
                return fam, v
        if fam['variants']:
            return fam, fam['variants'][0]
        return fam, None

    if presets_path is None:
        if (models_root.parent / 'presets.json').exists():
            presets_path = models_root.parent / 'presets.json'
        elif Path('presets.json').exists():
            presets_path = Path('presets.json')

    if presets_path is None or not presets_path.exists():
        return presets

    config = load_runner_config(presets_path)
    raw_presets = config.get('presets', [])

    for item in raw_presets:
        match_family = item.get('match_family', '')
        pref_quants = item.get('preferred_quants', [])
        fam, var = find_var(match_family, pref_quants)
        if not fam or not var:
            continue

        base = family_baseline(fam['family'], models_root, config=config)
        cfg = deep_merge(base, {'identity': {
            'base_model': fam['base_model'],
            'family': fam['family'],
            'variant': var['variant'],
            'model_path': var['model_path']
        }})

        # Context
        ctx_label = item.get('context', '65k')
        cfg = deep_merge(cfg, context_override(ctx_label))

        # Vision
        vis_setting = item.get('vision', 'No')
        if vis_setting == 'auto':
            vis_mode = 'GPU' if fam.get('mmproj') else 'No'
            cfg = deep_merge(cfg, vision_override(vis_mode, fam.get('mmproj'), fam['family'], config=config))
        else:
            mmproj_path = fam.get('mmproj') if vis_setting != 'No' else None
            cfg = deep_merge(cfg, vision_override(vis_setting, mmproj_path, fam['family'], config=config))

        # Reasoning
        reasoning_mode = item.get('reasoning', 'Thinking (Natural / Medium - Recommended)')
        cfg = deep_merge(cfg, reasoning_override(fam['family'], reasoning_mode, config=config))

        # Temperature / Sampling
        if item.get('temp_profile'):
            cfg = deep_merge(cfg, temp_override(item['temp_profile']))
        if item.get('temp') is not None:
            cfg['engine']['temp'] = item['temp']
        if item.get('top_k') is not None:
            cfg['engine']['top_k'] = item['top_k']
        if item.get('top_p') is not None:
            cfg['engine']['top_p'] = item['top_p']
        if item.get('min_p') is not None:
            cfg['engine']['min_p'] = item['min_p']

        # Speculative / MTP
        mtp_setting = item.get('mtp', 'Off (Standard)')
        cfg = deep_merge(cfg, mtp_override(mtp_setting))

        draft_pattern = item.get('draft_model_pattern')
        if draft_pattern:
            # Check model directory and base model directory for draft file
            draft_file = Path(var['model_path']).parent / draft_pattern
            if not draft_file.exists():
                draft_file = models_root / fam['base_model'] / draft_pattern
            if draft_file.exists():
                cfg['engine']['spec_draft_model'] = str(draft_file)

        # Arbitrary engine overrides (e.g. ngl, ubatch_size)
        if item.get('engine_overrides'):
            cfg['engine'] = deep_merge(cfg.get('engine', {}), item['engine_overrides'])

        # Label formatting
        label_tmpl = item.get('label', '{variant}')
        label = label_tmpl.replace('{variant}', var['variant'])

        presets.append({
            'id': item.get('id', ''),
            'category': item.get('category', 'General'),
            'label': label,
            'cfg': cfg
        })

    return presets


def derive_family_folder(repo_id: str) -> str:
    repo_name = repo_id.split('/')[-1]
    clean = re.sub(r'[-_]gguf$', '', repo_name, flags=re.IGNORECASE)
    low = clean.lower()
    if 'qwen3.8' in low or 'qwen-3.8' in low:
        return 'Qwen3.8-27B'
    if 'qwen3.6' in low or 'qwen-3.6' in low:
        return 'Qwen3.6-35B'
    if 'nemotron' in low:
        return 'Nemotron-30B'
    if 'muse' in low or 'glimmer' in low:
        return 'Muse-Glimmer-30B'
    if 'gemma' in low and '26b' in low:
        return 'Gemma4-26B'
    if 'gemma' in low and ('e4b' in low or '4b' in low):
        return 'Gemma4-E4B'
    return clean


def download_models_interactive(models_root: Path, target_repo: Optional[str] = None) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("\n[!] 'huggingface_hub' package is required for downloading.")
        print("    Running via mise/uv: uv run --with huggingface_hub python llama_runner.py --download")
        return 1

    curated_repos = [
        ("Qwen 3.8 (27B Dense) ⭐", "unsloth/Qwen3.8-27B-GGUF"),
        ("Gemma 4 (26B-A4B MoE) 🛸", "unsloth/gemma-4-26B-A4B-it-GGUF"),
        ("Gemma 4 (E4B MoE) ⚡", "unsloth/gemma-4-E4B-it-GGUF"),
        ("NVIDIA Nemotron 3.5 (30B-A3B MoE) 🚀", "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF"),
        ("Qwen 3.6 (35B-A3B MoE) 📚", "unsloth/Qwen3.6-35B-A3B-GGUF"),
        ("Muse Glimmer (30B Dense) 🛠️", "unsloth/Muse-Glimmer-30B-GGUF"),
    ]

    print("\n" + "=" * 60)
    print("  HuggingFace GGUF Model Downloader")
    print("=" * 60)

    repo_id = target_repo
    if not repo_id:
        print("\nSelect a Model Repository to Download:")
        for idx, (label, repo) in enumerate(curated_repos, start=1):
            print(f"[{idx}] {label:<40} ({repo})")
        print(f"[{len(curated_repos) + 1}] Custom HuggingFace Repository (enter user/repo)")
        
        while True:
            choice = input(f"Choose [1-{len(curated_repos) + 1}]: ").strip()
            if choice.isdigit():
                c_idx = int(choice)
                if 1 <= c_idx <= len(curated_repos):
                    repo_id = curated_repos[c_idx - 1][1]
                    break
                elif c_idx == len(curated_repos) + 1:
                    repo_id = input("Enter HuggingFace repo ID (e.g. bartowski/Qwen2.5-Coder-32B-Instruct-GGUF): ").strip()
                    if repo_id:
                        break

    if not repo_id:
        print("No repository specified. Aborting.")
        return 1

    default_family = derive_family_folder(repo_id)
    print(f"\nDefault target folder: models/{default_family}/")
    custom_folder = input(f"Enter folder name under models/ [Enter for '{default_family}']: ").strip()
    family_folder_name = custom_folder if custom_folder else default_family
    target_dir = models_root / family_folder_name
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nQuerying repository '{repo_id}' on HuggingFace Hub...")
    api = HfApi()
    try:
        repo_files = api.list_repo_tree(repo_id)
    except Exception as e:
        print(f"Failed to query repository '{repo_id}': {e}")
        return 1

    model_files = []
    mmproj_files = []

    for f in repo_files:
        p_str = getattr(f, 'path', str(f))
        if not p_str.endswith('.gguf'):
            continue
        # Rule 1: Never download MTP draft sidefiles
        if 'mtp' in p_str.lower():
            continue
        # Rule 2: Separate mmproj files
        if 'mmproj' in p_str.lower():
            mmproj_files.append(f)
        else:
            model_files.append(f)

    if not model_files:
        print(f"No valid GGUF model files found in '{repo_id}'.")
        return 1

    # Check if target folder already contains an mmproj file
    existing_mmprojs = list(target_dir.glob('mmproj-*.gguf')) + list(target_dir.glob('*.mmproj-*.gguf'))
    if existing_mmprojs:
        print(f"  ℹ️ Found existing multimodal projector: {existing_mmprojs[0].name}. (Skipping mmproj download)")
    elif mmproj_files:
        # Suggest downloading mmproj-BF16 if available, else first mmproj
        bf16_mm = next((m for m in mmproj_files if 'bf16' in getattr(m, 'path', '').lower()), mmproj_files[0])
        mm_path = getattr(bf16_mm, 'path', str(bf16_mm))
        mm_size_mb = getattr(bf16_mm, 'size', 0) / (1024 * 1024)
        print(f"\nMultimodal projector available: {mm_path} ({mm_size_mb:.1f} MB)")
        dl_mm = input("Download vision projector (mmproj) for this family? [Y/n]: ").strip().lower()
        if dl_mm != 'n':
            print(f"Downloading {mm_path} to models/{family_folder_name}/ ...")
            try:
                hf_hub_download(repo_id=repo_id, filename=mm_path, local_dir=str(target_dir))
                print(f"  ✅ Saved vision projector to models/{family_folder_name}/{Path(mm_path).name}")
            except Exception as e:
                print(f"  [!] Failed to download mmproj: {e}")

    # List model quantization files
    print(f"\nAvailable Model Quantizations in '{repo_id}':")
    for idx, mf in enumerate(model_files, start=1):
        m_path = getattr(mf, 'path', str(mf))
        m_size_gb = getattr(mf, 'size', 0) / (1024 * 1024 * 1024)
        size_str = f"({m_size_gb:.2f} GB)" if m_size_gb > 0 else ""
        print(f"[{idx}] {m_path:<48} {size_str}")

    chosen = input(f"\nSelect quantizations to download (e.g. 1 or comma-separated, or 'q' to cancel): ").strip()
    if not chosen or chosen.lower() == 'q':
        print("Download cancelled.")
        return 0

    selected_indices = []
    for part in chosen.split(','):
        part = part.strip()
        if part.isdigit():
            i = int(part)
            if 1 <= i <= len(model_files):
                selected_indices.append(i - 1)

    if not selected_indices:
        print("No valid files selected.")
        return 0

    for idx in selected_indices:
        target_file = getattr(model_files[idx], 'path', str(model_files[idx]))
        print(f"\nDownloading '{target_file}' to models/{family_folder_name}/ ...")
        try:
            hf_hub_download(repo_id=repo_id, filename=target_file, local_dir=str(target_dir))
            print(f"  ✅ Download complete: models/{family_folder_name}/{Path(target_file).name}")
        except Exception as e:
            print(f"  [!] Download failed for {target_file}: {e}")

    print(f"\n🎉 Model ready in models/{family_folder_name}/!")
    return 0


def start_server(server_path: Path, resolved: Dict[str, Any], port: int, out_dir: Path, log_prompts_dir: Optional[str] = None) -> int:
    resolved_copy = copy.deepcopy(resolved)
    resolved_copy['server']['ui'] = True
    if log_prompts_dir:
        resolved_copy['server']['log_prompts_dir'] = log_prompts_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = f'server_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = run_dir / 'server.log'
    args_list = build_args(resolved_copy, port)
    args_list += ['--log-file', str(log_file)]
    command = format_command(server_path, args_list)

    (run_dir / 'resolved-config.json').write_text(json.dumps(resolved_copy, indent=2, ensure_ascii=False), encoding='utf-8')
    (run_dir / 'command-line.txt').write_text(command + '\n', encoding='utf-8')

    print_summary(resolved_copy, 'baseline', command)
    kill_existing_servers()

    ident = resolved_copy.get('identity', {})
    vis_cfg = resolved_copy.get('vision', {})
    has_auth = bool(resolved_copy.get('server', {}).get('api_key') or os.environ.get('LLAMA_API_KEY'))

    # Validate model file existence
    model_path_str = ident.get('model_path')
    if model_path_str and not Path(model_path_str).exists():
        print(f"\n[!] Error: Model file not found at: {model_path_str}")
        return 1

    print(f'  Starting: {ident.get("variant", ident.get("family", "Model"))}')
    print(f'  Context: {ident.get("context_label", "default")} ({ident.get("context_tokens", "auto")})')
    print(f'  Vision: {vis_cfg.get("mode", "No")}')
    print(f'  MTP: {ident.get("mtp_profile", "Off")}')
    print(f'  Reasoning: {ident.get("reasoning_mode", "default")}')
    print(f'  Auth: {"API Key Protected" if has_auth else "Disabled (Local Public)"}')
    print(f'  API: http://{resolved_copy.get("server", {}).get("host", "127.0.0.1")}:{port}/v1')
    print(f'  Debug log: {log_file}')
    print(f'  Press Ctrl+C to stop\n')

    diag_thread = threading.Thread(target=run_diagnostics, args=(port, log_file), daemon=True)
    diag_thread.start()

    try:
        proc = subprocess.Popen([str(server_path)] + args_list)
        proc.wait()
    except KeyboardInterrupt:
        log('Stopping server...')
        proc.kill()
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    return 0


def main() -> int:
    settings = load_settings()
    srv_cfg = settings.get('server', {})
    paths_cfg = settings.get('paths', {})

    default_port = int(srv_cfg.get('port', 8080))
    default_models_dir = paths_cfg.get('models_dir', str(Path.cwd() / 'models'))
    default_presets_file = paths_cfg.get('presets_file', str(Path.cwd() / 'presets.json'))
    default_out_dir = paths_cfg.get('bench_results_dir', str(Path.cwd() / 'bench-results'))

    exe_suffix = '.exe' if sys.platform.startswith('win') else ''
    parser = argparse.ArgumentParser(description='llabrun: Llama.cpp Lab Runner & Orchestrator')
    parser.add_argument('--bin-dir', default=str(Path.cwd() / 'bin'))
    parser.add_argument('--models-root', default=str(default_models_dir))
    parser.add_argument('--presets-file', default=str(default_presets_file), help='Path to presets.json (or presets.example.json) configuration file')
    parser.add_argument('--out-dir', default=str(default_out_dir))
    parser.add_argument('--server-exe', default=f'llama-server{exe_suffix}')
    parser.add_argument('--base-port', type=int, default=default_port)
    parser.add_argument('--download', nargs='?', const='interactive', help='Download model from HuggingFace (repo_id or interactive)')
    args = parser.parse_args()

    bin_dir = Path(args.bin_dir)
    models_root = Path(args.models_root)
    presets_file = Path(args.presets_file) if args.presets_file else None
    out_dir = Path(args.out_dir)
    server_path = bin_dir / args.server_exe

    if args.download:
        target_repo = None if args.download == 'interactive' else args.download
        return download_models_interactive(models_root, target_repo)

    if not server_path.exists() and sys.platform != 'win32':
        alt_path = bin_dir / 'llama-server'
        if alt_path.exists():
            server_path = alt_path

    if not server_path.exists():
        raise FileNotFoundError(f'Server executable not found: {server_path}')

    if sys.platform != 'win32':
        try:
            server_path.chmod(server_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except Exception:
            pass
    out_dir.mkdir(parents=True, exist_ok=True)
    port = args.base_port

    families = scan_models(models_root)
    if not families:
        print(f'\n[!] No model families discovered under {models_root}')
        dl_choice = input('Would you like to download a model from HuggingFace now? [Y/n]: ').strip().lower()
        if dl_choice != 'n':
            download_models_interactive(models_root)
            families = scan_models(models_root)
            if not families:
                return 0
        else:
            return 0

    last_run = load_last_run()
    has_valid_last_run = False
    if last_run and last_run.get('cfg'):
        ident = last_run.get('cfg', {}).get('identity', {})
        model_p = ident.get('model_path')
        if model_p and Path(model_p).exists():
            has_valid_last_run = True

    print('\n' + '=' * 60)
    print('  llabrun — Llama.cpp Lab Runner & Orchestrator')
    print('=' * 60)

    if has_valid_last_run:
        last_var = last_run.get('variant', 'Model')
        last_ctx = last_run.get('context', 'default')
        last_vis = last_run.get('vision', 'No')
        last_mtp = last_run.get('mtp', 'Off')
        print(f"  Last Run: {last_var} [{last_ctx} | Vision: {last_vis} | MTP: {last_mtp}]")
        print('\n1. Select Action')
        print(f'[1] Quick Start: Run Last Configuration ({last_var}) ⭐ [Press Enter]')
        print(f'[2] Save Last Run to Presets (presets.json)')
        print('[3] Start Server (Interactive Setup / Custom Model)')
        print('[4] Start Server (Verified 1-Click Hardware Presets)')
        print('[5] Run Benchmark (Automated evaluation loop)')
        print('[6] Generate Router INI Presets')
        print('[7] Download Models (from HuggingFace ⭐)')

        while True:
            action_raw = input('\nChoose [1-7] (default [1]): ').strip() or '1'
            if action_raw in ['1', '2', '3', '4', '5', '6', '7']:
                action = {'1': 'last', '2': 'save_last', '3': 'start', '4': 'preset', '5': 'benchmark', '6': 'router', '7': 'download'}[action_raw]
                break
    else:
        print('\n1. Select Action')
        print('[1] Start Server (Interactive Setup / Custom Model ⭐) [Press Enter]')
        print('[2] Start Server (Verified 1-Click Hardware Presets)')
        print('[3] Run Benchmark (Automated evaluation loop)')
        print('[4] Generate Router INI Presets')
        print('[5] Download Models (from HuggingFace ⭐)')

        while True:
            action_raw = input('\nChoose [1-5] (default [1]): ').strip() or '1'
            if action_raw in ['1', '2', '3', '4', '5']:
                action = {'1': 'start', '2': 'preset', '3': 'benchmark', '4': 'router', '5': 'download'}[action_raw]
                break

    if action == 'last':
        print(f"\n🚀 Quick-Starting last configuration: {last_run.get('variant')}...")
        cfg = copy.deepcopy(last_run['cfg'])
        # Dynamically apply the latest server settings from settings.json
        if 'server' not in cfg:
            cfg['server'] = {}
        cfg['server'] = deep_merge(cfg['server'], settings.get('server', {}))
        cfg['server']['port'] = args.base_port

        # Dynamically apply latest engine_defaults (e.g. threads, batch size) as the base
        if 'engine' in cfg and settings.get('engine_defaults'):
            cfg['engine'] = deep_merge(settings.get('engine_defaults', {}), cfg['engine'])

        return start_server(server_path, cfg, args.base_port, out_dir, cfg.get('server', {}).get('log_prompts_dir'))

    if action == 'save_last':
        default_label = f"{last_run.get('variant')} ({last_run.get('context', 'default')})"
        preset_label = input(f'\nEnter preset label (default: "{default_label}"): ').strip() or default_label
        clean_id = re.sub(r'[^a-zA-Z0-9_\-]', '-', last_run.get('variant', 'model').lower()).strip('-') + f"-{last_run.get('context', 'default').lower()}"
        preset_item = {
            'id': clean_id,
            'category': 'Custom Presets',
            'label': f'⭐ {preset_label}',
            'match_family': last_run.get('family', ''),
            'preferred_quants': [last_run.get('variant', '')],
            'context': last_run.get('context', 'auto'),
            'vision': last_run.get('vision', 'No'),
            'reasoning': last_run.get('reasoning', 'Thinking'),
            'temp_profile': last_run.get('temp_profile', 'Family Default'),
            'mtp': last_run.get('mtp', 'Off')
        }
        target_presets_file = Path(args.presets_file) if args.presets_file else Path('presets.json')
        save_preset_to_file(preset_item, target_presets_file)
        print(f"  ✅ Saved preset '{preset_label}' to {target_presets_file.name}!")
        print(f"  You can now launch it anytime directly from 1-Click Hardware Presets.\n")

        launch_now = input(f"Launch {preset_label} now? [Y/n]: ").strip().lower()
        if launch_now != 'n':
            return start_server(server_path, last_run['cfg'], args.base_port, out_dir, last_run.get('cfg', {}).get('server', {}).get('log_prompts_dir'))
        return 0

    if action == 'download':
        return download_models_interactive(models_root)

    if action == 'preset':
        active_presets_file = presets_file if (presets_file and presets_file.exists()) else Path('presets.json')
        if not active_presets_file.exists():
            print("\n[!] No active 'presets.json' file found.")
            print("Verified 1-Click Presets are optional hardware-specific profiles (e.g. 16GB RTX reference picks).")
            print("\nOptions:")
            print("  [1] Copy & activate 16GB Reference Profile (presets.example.json -> presets.json)")
            print("  [2] Start Server with Interactive Setup (Auto GPU layer offload / Any hardware ⭐)")
            p_choice = input("\nChoose [1-2] (default 2): ").strip() or "2"
            if p_choice == "1":
                if Path('presets.example.json').exists():
                    shutil.copy2('presets.example.json', 'presets.json')
                    print("  ✅ Created presets.json from presets.example.json")
                    presets_file = Path('presets.json')
                else:
                    print("  [!] presets.example.json not found.")
                    action = 'start'
            else:
                action = 'start'

        if action == 'preset':
            presets = build_verified_presets(families, models_root, presets_path=presets_file)
            if not presets:
                print("\n[!] No models in models/ folder match the active presets.")
                print("Switching to custom model launcher...\n")
                action = 'start'
            else:
                print("\n======================= VERIFIED 1-CLICK PRESETS =======================")
                for idx, p in enumerate(presets, start=1):
                    print(f"[{idx}] {p['label']}")
                print("========================================================================")
                
                while True:
                    choice = input(f"Choose preset [1-{len(presets)}]: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(presets):
                        selected_preset = presets[int(choice) - 1]
                        break
                        
                print('\nEnable Prompt Logging?')
                print('[1] No (default)')
                print(f'[2] Yes (to {Path.cwd() / "prompt_logs"})')
                log_choice = input('Choose [1] or [2] (default 1): ').strip()
                log_prompts_dir = str(Path.cwd() / "prompt_logs") if log_choice == '2' else None
                
                return start_server(server_path, selected_preset['cfg'], args.base_port, out_dir, log_prompts_dir)

    if action == 'router':
        print(f'\n2. Router Preset Mode')
        print('[1] Generate Router INI from Verified 1-Click Hardware Presets ⭐ (Recommended)')
        print('[2] Generate from all discovered model files (Auto Scan)')
        print('[3] Select specific models to include')
        while True:
            rmode = input('Choose [1], [2], [3]: ').strip()
            if rmode in ['1', '2', '3']: break
        
        if rmode == '1':
            presets = build_verified_presets(families, models_root, presets_path=presets_file)
            out_file = out_dir / f"router_{time.strftime('%Y%m%d_%H%M%S')}.ini"
            generate_router_ini_from_presets(presets, out_file)
            return 0

        target_families = families
        if rmode == '3':
            selected_fams = choose_many(families, lambda x: x['base_model'])
            if not selected_fams:
                print("No models selected. Exiting.")
                return 1
            target_families = []
            for fam in selected_fams:
                print(f"\nSelect variant(s) for {fam['base_model']}")
                chosen_vars = choose_many(fam['variants'], lambda x: x['variant'])
                if chosen_vars:
                    new_fam = fam.copy()
                    new_fam['variants'] = chosen_vars
                    target_families.append(new_fam)
            if not target_families:
                print("No variants selected. Exiting.")
                return 1
                
        out_file = out_dir / f"router_{time.strftime('%Y%m%d_%H%M%S')}.ini"
        generate_router_ini(target_families, out_file, models_root)
        return 0

    print(f'\n2. Select base model(s)')
    if action == 'start':
        fam = choose_one([f['base_model'] for f in families], 'Base Model')
        selected_families = [f for f in families if f['base_model'] == fam]
    else:
        selected_families = choose_many(families, lambda x: x['base_model'])
    if not selected_families:
        raise RuntimeError('No base models selected')

    selected_variants = []
    for fam in selected_families:
        variants = [
            {'base_model': fam['base_model'], 'family': fam['family'], 'variant': v['variant'], 'model_path': v['model_path'], 'mmproj': fam['mmproj']}
            for v in fam['variants']
        ]
        if action == 'start':
            var_name = choose_one([v['variant'] for v in variants], f"3. Select variant for {fam['base_model']}")
            chosen = [v for v in variants if v['variant'] == var_name]
        else:
            print(f"\n3. Select variant(s) for {fam['base_model']}")
            chosen = choose_many(variants, lambda x: x['variant'])
        selected_variants.extend(chosen)
    if not selected_variants:
        raise RuntimeError('No variants selected')

    if action == 'start':
        ctx = choose_one(['auto', '32k', '65k', '128k', '140k', '200k', '256k', '512k', '1M'], '4. Context size')
        contexts = [ctx]
    else:
        print(f'\n4. Select context(s)')
        contexts = choose_many(['auto', '32k', '65k', '128k', '140k', '200k', '256k', '512k', '1M'], lambda x: x)

    if action == 'start':
        vision_modes = [choose_one(['No', 'GPU', 'CPU'], '5. Vision mode')]
    else:
        print(f'\n5. Select vision mode(s)')
        vision_modes = choose_many(['No', 'GPU', 'CPU'], lambda x: x)

    THINKING_CHOICES = [
        'Thinking (Natural / Medium - Recommended)',
        'Thinking (Low Effort)',
        'Thinking (xHigh Effort)',
        'Non-Thinking (Instruct)'
    ]

    if action == 'start':
        reasoning_modes = [choose_one(THINKING_CHOICES, '6. Thinking mode & effort')]
    else:
        print(f'\n6. Select thinking mode(s) & effort')
        reasoning_modes = choose_many(THINKING_CHOICES, lambda x: x)

    if action == 'start':
        print('\n7. Temperature & Sampling Profile')
        print('[1] Family Default (Calibrated for model & reasoning mode ⭐)')
        print('[2] Low / Deterministic (0.2)')
        print('[3] Balanced (0.6)')
        print('[4] High / Creative (1.15)')
        print('[5] Custom Temperature (enter float)')
        while True:
            t_choice = input('Choose [1-5] (default 1): ').strip() or '1'
            if t_choice == '1':
                temp_profiles = ['Family Default']
                break
            elif t_choice == '2':
                temp_profiles = ['Low (0.2)']
                break
            elif t_choice == '3':
                temp_profiles = ['Balanced (0.6)']
                break
            elif t_choice == '4':
                temp_profiles = ['High (1.15)']
                break
            elif t_choice == '5':
                val = input('Enter temperature value (e.g. 0.35): ').strip()
                try:
                    t_float = float(val)
                    temp_profiles = [f'Custom ({t_float})']
                    break
                except ValueError:
                    print('Invalid float value. Please try again.')
    else:
        print(f'\n7. Select temperature profile(s)')
        temp_profiles = choose_many(['Family Default', 'Low (0.2)', 'Balanced (0.6)', 'High (1.15)'], lambda x: x)

    if action == 'start':
        mtp_profiles = [choose_one(['Off (Standard)', 'MTP (draft-mtp, n=1)', 'MTP (draft-mtp, n=2)', 'MTP (draft-mtp, n=4)'], '8. Multi-Token Prediction (MTP / Speculative)')]
    else:
        print(f'\n8. Select Multi-Token Prediction mode(s)')
        mtp_profiles = choose_many(['Off (Standard)', 'MTP (draft-mtp, n=1)', 'MTP (draft-mtp, n=2)', 'MTP (draft-mtp, n=4)'], lambda x: x)

    # 9. Prompt Logging (start) or Prompt Input (benchmark)
    log_prompts_val = None
    if action == 'start':
        print(f'\n9. Enable Prompt Logging?')
        print('[1] No (default)')
        default_log_dir = str(Path.cwd() / 'prompt_logs')
        print('[2] Yes (to ./prompt_logs)')
        while True:
            pl_choice = input('Choose [1] or [2] (default 1): ').strip() or '1'
            if pl_choice in ['1', '2']:
                break
        if pl_choice == '2':
            log_prompts_val = default_log_dir

    # Load prompt text
    prompt_text = ''
    if action == 'benchmark':
        prompt_file = Path.cwd() / 'prompt.txt'
        if prompt_file.exists():
            prompt_text = prompt_file.read_text(encoding='utf-8')
            log(f'Loaded prompt from {prompt_file} ({len(prompt_text)} chars)')
        else:
            prompt_input = input('\n9. Enter custom prompt text (or absolute path to a .txt file):\n').strip()
            if os.path.exists(prompt_input) and os.path.isfile(prompt_input):
                try:
                    with open(prompt_input, 'r', encoding='utf-8') as f:
                        prompt_text = f.read()
                except Exception as e:
                    print(f"Error reading prompt file: {e}")
                    return 1
            else:
                prompt_text = prompt_input

        if not prompt_text.strip():
            prompt_text = "Explain the difference between mutable and immutable data structures in modern programming languages, with concrete code examples."
            log(f"Using standard benchmark prompt ({len(prompt_text)} chars)")

    # ---- Action logic ----
    config = load_runner_config(Path(args.presets_file) if args.presets_file else models_root.parent / 'presets.json')

    if action == 'start':
        variant = selected_variants[0]
        ctx = contexts[0]
        vis_mode = vision_modes[0]
        r_mode = reasoning_modes[0]
        t_prof = temp_profiles[0]
        m_prof = mtp_profiles[0]

        base_cfg = family_baseline(variant['family'], models_root, config=config)
        base_cfg['server']['ui'] = True
        base_cfg = deep_merge(base_cfg, {'identity': {'base_model': variant['base_model'], 'family': variant['family'], 'variant': variant['variant'], 'model_path': variant['model_path']}})
        base_cfg = deep_merge(base_cfg, vision_override(vis_mode, variant['mmproj'], variant['family'], config=config))
        base_cfg = deep_merge(base_cfg, reasoning_override(variant['family'], r_mode, config=config))
        base_cfg = deep_merge(base_cfg, context_override(ctx))
        base_cfg = deep_merge(base_cfg, temp_override(t_prof))
        base_cfg = deep_merge(base_cfg, mtp_override(m_prof))
        if log_prompts_val:
            base_cfg['server']['log_prompts_dir'] = log_prompts_val
        resolved = base_cfg

        # Save to last run state
        run_record = {
            'variant': variant['variant'],
            'family': variant['family'],
            'base_model': variant['base_model'],
            'model_path': variant['model_path'],
            'context': ctx,
            'vision': vis_mode,
            'reasoning': r_mode,
            'temp_profile': t_prof,
            'mtp': m_prof,
            'cfg': resolved,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        save_last_run(run_record)

        return start_server(server_path, resolved, port, out_dir, log_prompts_val)

    # ---- Benchmark logic ----
    run_ts = time.strftime("%Y%m%d_%H%M%S")
    bench_out_dir = out_dir / f"bench_{run_ts}"
    bench_out_dir.mkdir(parents=True, exist_ok=True)
    bench_results = []
    port = args.base_port

    total_runs = len(selected_variants) * len(contexts) * len(vision_modes) * len(reasoning_modes) * len(temp_profiles) * len(mtp_profiles)
    run_idx = 1
    log(f'Starting benchmark: {total_runs} total runs planned.')
    (bench_out_dir / 'prompt.txt').write_text(prompt_text, encoding='utf-8')

    interrupt_flag = False
    for variant in selected_variants:
        if interrupt_flag: break
        for ctx in contexts:
            if interrupt_flag: break
            for vis_mode in vision_modes:
                if interrupt_flag: break
                for r_mode in reasoning_modes:
                    if interrupt_flag: break
                    for t_prof in temp_profiles:
                        if interrupt_flag: break
                        for m_prof in mtp_profiles:
                            if interrupt_flag: break
                            base_cfg = family_baseline(variant['family'], models_root, config=config)
                            base_cfg = deep_merge(base_cfg, {'identity': {'base_model': variant['base_model'], 'family': variant['family'], 'variant': variant['variant'], 'model_path': variant['model_path']}})
                            base_cfg = deep_merge(base_cfg, vision_override(vis_mode, variant['mmproj'], variant['family'], config=config))
                            base_cfg = deep_merge(base_cfg, reasoning_override(variant['family'], r_mode, config=config))
                            resolved = deep_merge(base_cfg, context_override(ctx))
                            resolved = deep_merge(resolved, temp_override(t_prof))
                            resolved = deep_merge(resolved, mtp_override(m_prof))

                        log(f'  Run {run_idx}/{total_runs}: {variant["variant"]} @ {ctx} | vision={vis_mode} | mtp={m_prof} | temp={t_prof}')

                        run_name = '__'.join([
                            resolved['identity']['base_model'],
                            resolved['identity']['variant'],
                            resolved['identity']['context_label'],
                            vis_mode,
                            resolved['identity']['mtp_profile'].replace(' ', '_').replace('(', '').replace(')', '').replace('=', '_'),
                            resolved['identity']['reasoning_mode'],
                            t_prof.replace(' ', '_').replace('(', '').replace(')', '')
                        ])
                        safe_run_name = ''.join(ch if ch.isalnum() or ch in '._-' else '_' for ch in run_name)
                        run_dir = bench_out_dir / safe_run_name
                        run_dir.mkdir(parents=True, exist_ok=True)

                        log_file = run_dir / 'server.log'
                        args_list = build_args(resolved, port)
                        command = format_command(server_path, args_list)

                        (run_dir / 'resolved-config.json').write_text(json.dumps(resolved, indent=2, ensure_ascii=False), encoding='utf-8')
                        (run_dir / 'command-line.txt').write_text(command + '\n', encoding='utf-8')

                        print(f'\n  ====== Run {run_idx}/{total_runs}: {variant["variant"]} | {ctx} | vision={vis_mode} | mtp={m_prof} ======')
                        run_idx += 1
                        kill_existing_servers()

                        proc = None
                        log_thread = None
                        result_row = {
                            'variant': variant['variant'], 'context': ctx, 'vision': vis_mode,
                            'mtp': resolved['identity'].get('mtp_profile', 'Off'),
                            'reasoning_mode': resolved['identity']['reasoning_mode'], 'temp_profile': resolved['identity'].get('temp_profile', '-'), 'status': 'ok',
                        }
                        try:
                            proc = subprocess.Popen(
                                [str(server_path)] + args_list,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1, encoding='utf-8', errors='replace'
                            )
                            log_thread = threading.Thread(target=filter_server_log, args=(proc.stderr, log_file), daemon=True)
                            log_thread.start()
                            log(f'  Waiting for server to load...')
                            wait_ready(f'http://127.0.0.1:{port}', 300)
        
                            # Parse server log for memory info
                            mem_info = parse_server_log(log_file)
                            result_row['memory_info'] = mem_info
                            result_row['layers_offloaded'] = mem_info.get('layers_offloaded', '')
                            gpu_buf = mem_info.get('model_buffer_gpu', '')
                            cpu_buf = mem_info.get('model_buffer_cpu', '')
                            log(f'  Server ready. GPU: {gpu_buf}, CPU: {cpu_buf}, Layers: {result_row["layers_offloaded"]}')
        
                            # Run two-turn benchmark conversation
                            log(f'  Running two-turn benchmark conversation...')
                            conv = benchmark_conversation(
                                f'http://127.0.0.1:{port}', resolved['engine'], prompt_text,
                            )
        
                            # Populate result row from conversation
                            t1 = conv['turn1']
                            result_row['wall_time_s'] = t1.get('wall_s', 0)
                            result_row['prompt_t_s'] = t1.get('prompt_t_s', 0)
                            result_row['gen_t_s'] = t1.get('gen_t_s', 0)
                            result_row['completion_tokens'] = t1.get('completion_tokens', 0)
                            result_row['html_issues_v1'] = len(conv['html_issues_v1'])
                            result_row['html_issues_v1_list'] = conv['html_issues_v1']
                            result_row['final_file_content'] = conv.get('file_v1', '')
        
                            # Save artifacts
                            (run_dir / 'response_t1.json').write_text(
                                json.dumps(t1.get('response', {}), indent=2, ensure_ascii=False), encoding='utf-8')
                            if conv['file_v1']:
                                (run_dir / 'file_v1.html').write_text(conv['file_v1'], encoding='utf-8')
        
                            slots_data = fetch_optional(f'http://127.0.0.1:{port}/slots')
                            if slots_data:
                                (run_dir / 'slots.json').write_text(slots_data, encoding='utf-8')
                            metrics_data = fetch_optional(f'http://127.0.0.1:{port}/metrics')
                            if metrics_data:
                                (run_dir / 'metrics.txt').write_text(metrics_data, encoding='utf-8')
        
                            log(f'  Done: {result_row["gen_t_s"]} t/s ({result_row["completion_tokens"]} tok), wall={result_row["wall_time_s"]}s')
        
                        except Exception as exc:
                            result_row['status'] = 'FAILED'
                            result_row['error'] = str(exc)
                            log(f'  FAILED: {exc}')
                        except KeyboardInterrupt:
                            log('\nInterrupted by user (Ctrl+C). Saving partial results and aborting...')
                            result_row['status'] = 'INTERRUPTED'
                            interrupt_flag = True
                        finally:
                            if proc is not None and proc.poll() is None:
                                try:
                                    proc.terminate()
                                    proc.wait(timeout=2.0)
                                except Exception:
                                    proc.kill()
                                    try:
                                        proc.wait(timeout=5.0)
                                    except Exception:
                                        pass
                            if proc is not None and proc.stderr:
                                try:
                                    proc.stderr.close()
                                except Exception:
                                    pass
                            if log_thread is not None and log_thread.is_alive():
                                try:
                                    log_thread.join(timeout=2.0)
                                except Exception:
                                    pass
                            # Port cycling with wrap-around to prevent high-port exhaustion
                            port = args.base_port + ((port - args.base_port + 1) % 50)
        
                        bench_results.append(result_row)

    print_results_table(bench_results)
    write_csv(bench_out_dir / 'bench-results.csv', bench_results)
    write_summary(bench_out_dir / 'summary.md', bench_results, prompt_text)
    log(f'Benchmark complete. Results saved to {bench_out_dir}')
    log(f'Summary: {bench_out_dir / "summary.md"}')
    return 0

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f'\nCancelled by user. Exiting.', file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        raise SystemExit(1)
