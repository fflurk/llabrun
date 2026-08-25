#!/usr/bin/env python3
"""
update_llama.py -- llabrun Engine Builder & Updater for llama.cpp (Windows CUDA / Vulkan / Hybrid)

Modes:
  1. Lazy Mode (Prebuilt Release):
     - Fast download of official GitHub release binaries (CUDA or Vulkan)
     - Zero compiler prerequisites (runs directly with GPU drivers)
  2. Full Optimized Mode (Source Build):
     - Native compilation from source using CMake, MSVC, and CUDA / Vulkan
     - Maximum hardware-specific performance and bleeding-edge commits
     - Supports building single backend (CUDA or Vulkan) or dual Hybrid (CUDA + Vulkan)

Features:
  - Version tracking with automatic skip if already up-to-date
  - Captures llama-server --help before/after and diffs for new options
  - Categorized commit changelog (CUDA, Vulkan, Server, Models, Core)
  - Automatic HuggingFace chat template synchronization
  - Safe backup and one-command rollback
  - Comprehensive system diagnostics / doctor
"""

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

# ── Configuration ──────────────────────────────────────────────────────────
REPO = "ggml-org/llama.cpp"
SOURCE_DIR = Path("Source/llama.cpp")
BIN_DIR = Path("bin")
BACKUP_DIR = Path("bin_backup")
VERSION_FILE = Path("bin/.llama_version")
BUILD_INFO_FILE = Path("bin/.llama_build_info.json")
HELP_BEFORE_FILE = Path("bin/.llama_help_snapshot.txt")
CURL_EXE = r"C:\Windows\System32\curl.exe"  # Real Windows curl
SERVER_EXE = BIN_DIR / "llama-server.exe"


# ── ANSI Terminal Colors ───────────────────────────────────────────────────
class Col:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"


def _print_safe(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


def info(msg: str) -> None:
    _print_safe(f"{Col.CYAN}> {msg}{Col.RESET}")


def ok(msg: str) -> None:
    _print_safe(f"{Col.GREEN}+ {msg}{Col.RESET}")


def warn(msg: str) -> None:
    _print_safe(f"{Col.YELLOW}! {msg}{Col.RESET}")


def err(msg: str) -> None:
    _print_safe(f"{Col.RED}X {msg}{Col.RESET}")


# ── HTTP & GitHub Helpers ──────────────────────────────────────────────────
def fetch_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "update-llama/2.0"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_with_curl(url: str, output_path: str) -> None:
    cmd = [
        CURL_EXE,
        "-L",                # Follow redirects
        "-#",                # Progress bar
        "--retry", "3",      # Retry on transient failures
        "--retry-delay", "2",
        "-o", output_path,
        url,
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"curl download failed with exit code {result.returncode}")


def download_with_python(url: str, output_path: str) -> None:
    from urllib.request import urlretrieve
    info("Downloading with Python fallback...")

    def progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)

    urlretrieve(url, output_path, reporthook=progress)
    print()


def download_file(url: str, output_path: str) -> None:
    if Path(CURL_EXE).exists():
        download_with_curl(url, output_path)
    else:
        download_with_python(url, output_path)


# ── Version & Snapshot Tracking ────────────────────────────────────────────
def load_build_info() -> Dict[str, Any]:
    if BUILD_INFO_FILE.exists():
        try:
            return json.loads(BUILD_INFO_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    ver = get_installed_version()
    if ver:
        return {
            "version": ver,
            "mode": "release" if ver.startswith("b") or ver.startswith("v") else "build",
            "backend": "cuda",
            "timestamp": "unknown"
        }
    return {}


def save_build_info(version: str, mode: str, backend: str) -> None:
    BUILD_INFO_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": version,
        "mode": mode,         # "release" or "build"
        "backend": backend,   # "cuda", "vulkan", "hybrid"
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    BUILD_INFO_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    VERSION_FILE.write_text(version, encoding="utf-8")


def get_installed_version() -> Optional[str]:
    if VERSION_FILE.exists():
        ver = VERSION_FILE.read_text(encoding="utf-8-sig").strip()
        if ver:
            return ver
    if SERVER_EXE.exists():
        try:
            result = subprocess.run(
                [str(SERVER_EXE), "--version"],
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout + result.stderr
            for line in output.splitlines():
                if "version:" in line.lower():
                    # e.g., "version: 9469 (d178a1181)" or "version: b10603"
                    return line.strip()
        except Exception:
            pass
    return None


def clean_version_tag(ver_str: Optional[str]) -> Optional[str]:
    if not ver_str:
        return None
    cleaned = ver_str.strip().strip("\ufeff")
    m = re.search(r'\(([a-fA-F0-9]+)\)', cleaned)
    if m:
        return m.group(1)
    if cleaned.startswith("version: "):
        cleaned = cleaned.replace("version: ", "").strip()
    return cleaned


def capture_help() -> str:
    if not SERVER_EXE.exists():
        return ""
    try:
        result = subprocess.run(
            [str(SERVER_EXE), "--help"],
            capture_output=True, text=True, timeout=15
        )
        return (result.stdout + result.stderr).strip()
    except Exception:
        return ""


def save_help_snapshot(help_text: str) -> None:
    if help_text:
        HELP_BEFORE_FILE.write_text(help_text, encoding="utf-8")


def load_help_snapshot() -> str:
    if HELP_BEFORE_FILE.exists():
        return HELP_BEFORE_FILE.read_text(encoding="utf-8")
    return ""


def diff_help(before: str, after: str) -> Optional[str]:
    if not before or not after:
        return None
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff = list(difflib.unified_diff(
        before_lines, after_lines,
        fromfile="previous version", tofile="new version",
        lineterm=""
    ))
    if not diff:
        return None
    added = [line[1:].strip() for line in diff if line.startswith("+") and not line.startswith("+++")]
    removed = [line[1:].strip() for line in diff if line.startswith("-") and not line.startswith("---")]
    if not added and not removed:
        return None
    return "\n".join(diff)


def fetch_changelog(old_tag: str, new_tag: str) -> Optional[List[Dict[str, str]]]:
    old_tag = old_tag.strip().strip("\ufeff")
    new_tag = new_tag.strip().strip("\ufeff")
    url = f"https://api.github.com/repos/{REPO}/compare/{old_tag}...{new_tag}"
    try:
        data = fetch_json(url)
    except Exception as e:
        warn(f"Could not fetch changelog: {e}")
        return None
    commits = data.get("commits", [])
    if not commits:
        return None
    entries = []
    for c in commits:
        msg = c.get("commit", {}).get("message", "").split("\n")[0].strip()
        sha = c.get("sha", "")[:7]
        author = c.get("commit", {}).get("author", {}).get("name", "unknown")
        if msg:
            entries.append({"sha": sha, "msg": msg, "author": author})
    return entries


def print_changelog(entries: List[Dict[str, str]], old_tag: str, new_tag: str) -> None:
    ignored_prefixes = {
        "opencl", "metal", "hexagon", "sycl", "[sycl]", "ggml-webgpu",
        "ggml", "ggml-cpu", "ui", "webui", "app", "vocab", "ci", "docs", "test",
        "tests", "nix", "vendor", "sync", "readme", "security", "meta", "build",
        "chore", "ngram-mod", "graph", "download", "tp"
    }

    groups: Dict[str, List[Dict[str, str]]] = {
        "cuda": [], "vulkan": [], "server": [], "model": [], "core": [], "other": [], "ignored": []
    }

    for entry in entries:
        msg = entry["msg"]
        prefix = ""
        if ":" in msg:
            prefix = msg.split(":")[0].strip().lower()
            if "(" in prefix and ")" in prefix:
                prefix = prefix.split("(")[1].split(")")[0].strip()

        if prefix in ignored_prefixes or msg.lower().startswith("bump") or msg.lower().startswith("update"):
            groups["ignored"].append(entry)
        elif "cuda" in prefix or "kv-cache" in prefix:
            groups["cuda"].append(entry)
        elif "vulkan" in prefix:
            groups["vulkan"].append(entry)
        elif "server" in prefix:
            groups["server"].append(entry)
        elif "model" in prefix or "mtmd" in prefix or "qwen" in prefix or "gemma" in prefix:
            groups["model"].append(entry)
        elif prefix in ["llama", "common", "arg", "fix", "feat", "perf", "refactor"]:
            groups["core"].append(entry)
        else:
            groups["other"].append(entry)

    print()
    _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
    _print_safe(f"{Col.BOLD}  Changelog: {old_tag} -> {new_tag} ({len(entries)} commits){Col.RESET}")
    _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")

    display = [
        ("cuda", "CUDA & KV-Cache", Col.GREEN),
        ("vulkan", "Vulkan GPU Backend", Col.MAGENTA),
        ("server", "Server API", Col.CYAN),
        ("model", "Models & Multimodal", Col.YELLOW),
        ("core", "Core Llama & Features", Col.RESET),
        ("other", "Other Updates", Col.DIM),
    ]

    for key, label, color in display:
        items = groups[key]
        if items:
            _print_safe(f"\n  {Col.BOLD}{label} ({len(items)}){Col.RESET}")
            for item in items:
                _print_safe(f"  {color}  {item['sha']} {item['msg']}{Col.RESET}")

    num_ignored = len(groups["ignored"])
    if num_ignored > 0:
        _print_safe(f"\n  {Col.DIM}  + {num_ignored} skipped (other backends, UI, docs, tests, etc.){Col.RESET}")

    _print_safe(f"\n{Col.BOLD}{'=' * 60}{Col.RESET}")


def show_help_diff_report(help_before: str, help_after: str) -> None:
    if help_before and help_after:
        diff_result = diff_help(help_before, help_after)
        if diff_result:
            print()
            _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
            _print_safe(f"{Col.BOLD}  New/Changed Options Since Last Version{Col.RESET}")
            _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
            for line in diff_result.splitlines():
                if line.startswith("+") and not line.startswith("+++"):
                    _print_safe(f"  {Col.GREEN}{line}{Col.RESET}")
                elif line.startswith("-") and not line.startswith("---"):
                    _print_safe(f"  {Col.RED}{line}{Col.RESET}")
                elif line.startswith("@@"):
                    _print_safe(f"  {Col.CYAN}{line}{Col.RESET}")
                else:
                    _print_safe(f"  {Col.DIM}{line}{Col.RESET}")
            _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
        else:
            ok("No changes detected in --help options.")


def download_hf_template() -> None:
    api_url = "https://huggingface.co/api/models/froggeric/Qwen-Fixed-Chat-Templates/commits/main"
    download_url = "https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja"
    output_path = "templates/chat_template_qwen.jinja"
    version_file = Path("templates/.template_version")

    info("Checking for community Qwen template updates on HuggingFace...")
    try:
        commits = fetch_json(api_url)
        if not commits:
            warn("No commits found for Qwen template.")
            return

        latest_commit = commits[0]
        latest_sha = latest_commit["id"]

        installed_sha = None
        if version_file.exists():
            installed_sha = version_file.read_text(encoding="utf-8").strip()

        if installed_sha == latest_sha:
            ok(f"Community Qwen template is up-to-date ({latest_sha[:7]})")
            return

        Path("templates").mkdir(exist_ok=True)
        download_file(download_url, output_path)
        version_file.write_text(latest_sha, encoding="utf-8")
        ok(f"Downloaded latest community Qwen template ({latest_sha[:7]}) -> {output_path}")

        if installed_sha:
            print()
            _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
            _print_safe(f"{Col.BOLD}  Qwen Template Changelog: {installed_sha[:7]} -> {latest_sha[:7]}{Col.RESET}")
            _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
            for c in commits:
                if c["id"] == installed_sha:
                    break
                date_str = c.get("date", "")[:10]
                _print_safe(f"  {Col.CYAN}{date_str}{Col.RESET} {c.get('title', '')}")
            _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
            print()
    except Exception as e:
        warn(f"Failed to fetch community Qwen template from HuggingFace: {e}")


# ── Backup & Rollback ──────────────────────────────────────────────────────
def backup_bin() -> None:
    if not BIN_DIR.exists() or not any(BIN_DIR.iterdir()):
        return
    if BACKUP_DIR.exists():
        shutil.rmtree(BACKUP_DIR)
    info(f"Backing up {BIN_DIR} -> {BACKUP_DIR}")
    shutil.copytree(BIN_DIR, BACKUP_DIR)


def rollback() -> None:
    if not BACKUP_DIR.exists():
        err("No backup found in bin_backup/ to restore from!")
        sys.exit(1)
    info(f"Rolling back: {BACKUP_DIR} -> {BIN_DIR}")
    if BIN_DIR.exists():
        shutil.rmtree(BIN_DIR)
    shutil.move(str(BACKUP_DIR), str(BIN_DIR))
    ok("Rollback complete.")


# ── Diagnostics & Doctor ───────────────────────────────────────────────────
def check_prerequisites() -> Dict[str, Any]:
    info("Checking system prerequisites...")
    status: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "uv": None,
        "git": None,
        "cmake": None,
        "vcvars": None,
        "cuda_nvcc": None,
        "nvidia_gpu": None,
        "vulkan_runtime": None,
        "vulkan_sdk": None,
    }

    # uv
    try:
        res = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            status["uv"] = res.stdout.strip()
    except Exception:
        pass

    # git
    try:
        res = subprocess.run(["git", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            status["git"] = res.stdout.strip()
    except Exception:
        pass

    # cmake
    try:
        res = subprocess.run(["cmake", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            status["cmake"] = res.stdout.splitlines()[0]
    except Exception:
        pass

    # Visual Studio C++ Compiler (vswhere -> vcvars64.bat)
    try:
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            res = subprocess.run([
                str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"
            ], capture_output=True, text=True)
            path_out = res.stdout.strip()
            if path_out:
                vcvars = Path(path_out) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if vcvars.exists():
                    status["vcvars"] = str(vcvars)
    except Exception:
        pass

    # CUDA Toolkit (nvcc)
    try:
        res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if "release" in line.lower():
                    status["cuda_nvcc"] = line.strip()
                    break
    except Exception:
        pass
    if not status["cuda_nvcc"] and os.environ.get("CUDA_PATH"):
        status["cuda_nvcc"] = f"Found in CUDA_PATH: {os.environ.get('CUDA_PATH')}"

    # NVIDIA GPU & Driver (nvidia-smi)
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if res.returncode == 0 and res.stdout.strip():
            status["nvidia_gpu"] = res.stdout.splitlines()[0].strip()
    except Exception:
        pass

    # Vulkan Support (vulkaninfo or standard system dll)
    vulkan_dll = Path(os.environ.get("SystemRoot", "C:\\Windows")) / "System32" / "vulkan-1.dll"
    if vulkan_dll.exists():
        status["vulkan_runtime"] = "vulkan-1.dll present in System32 (GPU driver supported)"
    try:
        res = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True)
        if res.returncode == 0:
            status["vulkan_runtime"] = "vulkaninfo CLI functional"
    except Exception:
        pass

    # Vulkan SDK (glslc & headers required for compiling from source)
    try:
        res = subprocess.run(["glslc", "--version"], capture_output=True, text=True)
        if res.returncode == 0:
            status["vulkan_sdk"] = res.stdout.splitlines()[0].strip()
    except Exception:
        pass
    if not status["vulkan_sdk"] and os.environ.get("VULKAN_SDK"):
        status["vulkan_sdk"] = f"Found in VULKAN_SDK: {os.environ.get('VULKAN_SDK')}"
    if not status["vulkan_sdk"]:
        vulkan_root = Path("C:/VulkanSDK")
        if vulkan_root.exists():
            sdk_dirs = sorted(vulkan_root.glob("*"), reverse=True)
            if sdk_dirs and (sdk_dirs[0] / "Bin" / "glslc.exe").exists():
                status["vulkan_sdk"] = f"Found in {sdk_dirs[0]}"

    print()
    _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")
    _print_safe(f"{Col.BOLD}  System Prerequisite Diagnostics (Doctor){Col.RESET}")
    _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}")

    def p_line(name: str, val: Optional[str], needed_for_build: bool = True):
        if val:
            _print_safe(f"  {Col.GREEN}+ {name:<22}: {val}{Col.RESET}")
        else:
            req_str = " (Required for Source Build)" if needed_for_build else " (Optional)"
            _print_safe(f"  {Col.YELLOW}! {name:<22}: Not Found{req_str}{Col.RESET}")

    p_line("Python Runtime", status["python"], False)
    p_line("uv Package Manager", status["uv"], False)
    p_line("NVIDIA GPU / Driver", status["nvidia_gpu"], False)
    p_line("Vulkan Driver / DLL", status["vulkan_runtime"], False)
    p_line("Git", status["git"], True)
    p_line("CMake", status["cmake"], True)
    p_line("MSVC C++ Toolchain", "Found (vcvars64.bat)" if status["vcvars"] else None, True)
    p_line("CUDA Toolkit (nvcc)", status["cuda_nvcc"], False)
    p_line("Vulkan SDK (glslc)", status["vulkan_sdk"], False)

    print()
    can_build_base = all([status["git"], status["cmake"] or status["vcvars"], status["vcvars"]])
    can_build_cuda = can_build_base and status["cuda_nvcc"] is not None
    can_build_vulkan = can_build_base and status["vulkan_sdk"] is not None
    can_build_hybrid = can_build_cuda and can_build_vulkan

    _print_safe(f"  {Col.BOLD}Mode Compatibility:{Col.RESET}")
    _print_safe(f"    - {Col.GREEN}Lazy Mode (Prebuilt CUDA):{Col.RESET} READY (Requires only NVIDIA driver)")
    _print_safe(f"    - {Col.GREEN}Lazy Mode (Prebuilt Vulkan):{Col.RESET} READY (Works with NVIDIA, Intel Arc, AMD)")
    
    if can_build_cuda:
        _print_safe(f"    - {Col.GREEN}Source Build (CUDA):{Col.RESET} READY (All MSVC & CUDA tools found)")
    else:
        _print_safe(f"    - {Col.YELLOW}Source Build (CUDA):{Col.RESET} MISSING PREREQUISITES (CUDA Toolkit needed)")

    if can_build_vulkan:
        _print_safe(f"    - {Col.GREEN}Source Build (Vulkan):{Col.RESET} READY (MSVC + CMake + Vulkan SDK ready)")
    else:
        _print_safe(f"    - {Col.YELLOW}Source Build (Vulkan):{Col.RESET} MISSING PREREQUISITES (LunarG Vulkan SDK needed)")
        if not status["vulkan_sdk"]:
            _print_safe(f"        -> Install via winget: {Col.CYAN}winget install KhronosGroup.VulkanSDK{Col.RESET}")

    if can_build_hybrid:
        _print_safe(f"    - {Col.GREEN}Source Build (Hybrid):{Col.RESET} READY (CUDA + Vulkan SDK ready)")
    else:
        _print_safe(f"    - {Col.YELLOW}Source Build (Hybrid):{Col.RESET} MISSING PREREQUISITES (Requires both CUDA & Vulkan SDKs)")

    _print_safe(f"{Col.BOLD}{'=' * 60}{Col.RESET}\n")
    return status


# ── Lazy Mode: Prebuilt Release Downloader ──────────────────────────────────
def fetch_release_info(preferred_tag: Optional[str] = None) -> Dict[str, Any]:
    if preferred_tag:
        tag = preferred_tag if preferred_tag.startswith("b") or preferred_tag.startswith("v") else f"b{preferred_tag}"
        url = f"https://api.github.com/repos/{REPO}/releases/tags/{tag}"
        return fetch_json(url)

    # Fetch recent releases list to find the latest release with compiled assets
    url = f"https://api.github.com/repos/{REPO}/releases?per_page=10"
    releases = fetch_json(url)
    if isinstance(releases, list):
        for r in releases:
            assets = r.get("assets", [])
            if assets and any("win" in a.get("name", "").lower() for a in assets):
                return r
    raise RuntimeError("No suitable Windows releases found on GitHub.")


def select_release_assets(
    release_data: Dict[str, Any],
    cuda_version: Optional[str] = None,
    backend: str = "cuda"
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    assets = release_data.get("assets", [])
    win_assets = [a for a in assets if "win" in a.get("name", "").lower() and a.get("name", "").endswith(".zip")]

    if backend == "vulkan":
        # Match llama-b*-bin-win-vulkan-x64.zip
        vulkan_asset = next((
            a for a in win_assets
            if "vulkan" in a["name"].lower() and "x64" in a["name"].lower() and "arm64" not in a["name"].lower()
        ), None)
        if not vulkan_asset:
            raise RuntimeError(f"Could not find Vulkan asset in release {release_data.get('tag_name')}.")
        return vulkan_asset, None

    # CUDA: Find main llama-b*-bin-win-cuda-*-x64.zip + cudart-llama-bin-win-cuda-*-x64.zip
    cuda_candidates = [
        a for a in win_assets
        if "cuda" in a["name"].lower() and "x64" in a["name"].lower() and "arm64" not in a["name"].lower()
    ]

    if cuda_version:
        main_bin = next((a for a in cuda_candidates if a["name"].startswith("llama-") and cuda_version in a["name"]), None)
        cudart_bin = next((a for a in cuda_candidates if a["name"].startswith("cudart-") and cuda_version in a["name"]), None)
    else:
        # Preference: 13.3 > 12.4 > 12.2 > any cuda
        preferred_vers = ["13.3", "12.4", "12.2", "11.8"]
        main_bin = None
        cudart_bin = None
        for ver in preferred_vers:
            m = next((a for a in cuda_candidates if a["name"].startswith("llama-") and ver in a["name"]), None)
            c = next((a for a in cuda_candidates if a["name"].startswith("cudart-") and ver in a["name"]), None)
            if m and c:
                main_bin, cudart_bin = m, c
                break

        if not main_bin or not cudart_bin:
            # Fallback to first matching pair
            for a in cuda_candidates:
                if a["name"].startswith("llama-"):
                    m = re.search(r'cuda-([0-9\.]+)', a["name"])
                    if m:
                        c_ver = m.group(1)
                        c = next((cand for cand in cuda_candidates if cand["name"].startswith("cudart-") and c_ver in cand["name"]), None)
                        if c:
                            main_bin, cudart_bin = a, c
                            break

    if not main_bin or not cudart_bin:
        raise RuntimeError(f"Could not find matching CUDA binary + cudart asset pair in release {release_data.get('tag_name')}.")

    return main_bin, cudart_bin


def update_via_release(
    tag: Optional[str] = None,
    cuda_ver: Optional[str] = None,
    backend: str = "cuda",
    sync_source: bool = False,
    force: bool = False,
    no_backup: bool = False,
    no_diff: bool = False
) -> None:
    backend_label = "Vulkan" if backend == "vulkan" else "CUDA"
    info(f"Querying GitHub for llama.cpp release ({backend_label} backend)...")
    release_data = fetch_release_info(tag)
    release_tag = release_data.get("tag_name", "unknown")
    installed_ver = get_installed_version()

    print(f"\n  {Col.BOLD}Target Release:   {Col.GREEN}{release_tag}{Col.RESET} ({backend_label})")
    if installed_ver:
        print(f"  {Col.BOLD}Installed Binary: {Col.YELLOW}{installed_ver}{Col.RESET}")
    print()

    if installed_ver and (installed_ver == release_tag or release_tag in installed_ver) and not force:
        ok(f"Already up-to-date on release {release_tag}")
        print(f"  {Col.DIM}Use --force to redownload and reinstall.{Col.RESET}")
        return

    main_asset, cudart_asset = select_release_assets(release_data, cuda_ver, backend=backend)
    info(f"Selected Main Asset:   {Col.BOLD}{main_asset['name']}{Col.RESET}")
    if cudart_asset:
        info(f"Selected CUDA Runtime: {Col.BOLD}{cudart_asset['name']}{Col.RESET}")

    # Capture help before
    help_before = ""
    if not no_diff:
        help_before = load_help_snapshot()
        if not help_before:
            help_before = capture_help()

    # Backup
    if not no_backup:
        backup_bin()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        main_zip = tmp_path / main_asset["name"]

        info(f"Downloading {main_asset['name']}...")
        download_file(main_asset["browser_download_url"], str(main_zip))

        if cudart_asset:
            cudart_zip = tmp_path / cudart_asset["name"]
            info(f"Downloading {cudart_asset['name']}...")
            download_file(cudart_asset["browser_download_url"], str(cudart_zip))

        BIN_DIR.mkdir(parents=True, exist_ok=True)
        info("Extracting binaries to bin/...")

        with zipfile.ZipFile(main_zip, "r") as z:
            z.extractall(BIN_DIR)

        if cudart_asset:
            with zipfile.ZipFile(cudart_zip, "r") as z:
                z.extractall(BIN_DIR)

    ok(f"Successfully extracted release {release_tag} ({backend_label}) into bin/")
    save_build_info(release_tag, mode="release", backend=backend)

    # Capture help after & diff
    if not no_diff:
        help_after = capture_help()
        save_help_snapshot(help_after)
        show_help_diff_report(help_before, help_after)

    # Changelog
    if installed_ver and installed_ver != release_tag:
        old_clean = clean_version_tag(installed_ver)
        if old_clean and old_clean != release_tag:
            info("Fetching release changelog...")
            changelog = fetch_changelog(old_clean, release_tag)
            if changelog:
                print_changelog(changelog, old_clean, release_tag)

    # HuggingFace Community Qwen Template
    print()
    download_hf_template()

    # Optional source repo synchronization for agent code inspection & tuning
    if sync_source or (SOURCE_DIR.exists() and (SOURCE_DIR / ".git").exists()):
        try:
            print()
            info("Syncing Source/llama.cpp for agent code inspection & parameter tuning...")
            git_pull_or_clone(SOURCE_DIR)
            ok("Source repository synchronized.")
        except Exception as e:
            warn(f"Could not sync source repository: {e}")

    print()
    _print_safe(f"{Col.GREEN}{'=' * 60}{Col.RESET}")
    _print_safe(f"{Col.GREEN}{Col.BOLD}  + Lazy Update complete! Running {release_tag} ({backend_label}){Col.RESET}")
    _print_safe(f"{Col.GREEN}{'=' * 60}{Col.RESET}")
    if not no_backup:
        _print_safe(f"  {Col.DIM}Rollback available: uv run update_llama.py --rollback{Col.RESET}")
    print()


# ── Full Optimized Mode: Source Builder ─────────────────────────────────────
def git_pull_or_clone(repo_dir: Path) -> None:
    if not repo_dir.exists() or not (repo_dir / ".git").exists():
        info(f"Source repository not found at {repo_dir}. Cloning from GitHub...")
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", f"https://github.com/{REPO}.git", str(repo_dir)], check=True)
    else:
        info(f"Running git pull in {repo_dir}...")
        subprocess.run(["git", "pull"], cwd=repo_dir, check=True)


def git_get_sha(repo_dir: Path) -> Optional[str]:
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def git_get_closest_tag(repo_dir: Path, sha: str = "HEAD") -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0", sha],
            cwd=repo_dir, capture_output=True, text=True, check=True
        )
        return res.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def run_cmake_cmd(cmd_list: List[str], cwd: Path) -> None:
    vcvars_prefix = ""
    try:
        vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if vswhere.exists():
            result = subprocess.run([
                str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath"
            ], capture_output=True, text=True, check=True)

            install_path = result.stdout.strip()
            if install_path:
                vcvars = Path(install_path) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
                if vcvars.exists():
                    vcvars_prefix = f'call "{vcvars}" >nul && '
    except Exception:
        pass

    # Auto-detect Vulkan SDK if installed in standard path but not in session PATH
    vulkan_env_prefix = ""
    if not os.environ.get("VULKAN_SDK"):
        vulkan_root = Path("C:/VulkanSDK")
        if vulkan_root.exists():
            sdk_dirs = sorted(vulkan_root.glob("*"), reverse=True)
            if sdk_dirs and (sdk_dirs[0] / "Bin" / "glslc.exe").exists():
                sdk_path = sdk_dirs[0]
                vulkan_env_prefix = f'set "VULKAN_SDK={sdk_path}" && set "PATH={sdk_path}\\Bin;%PATH%" && '

    cmd_str = " ".join(cmd_list)
    full_cmd = f"{vcvars_prefix}{vulkan_env_prefix}{cmd_str}"
    subprocess.run(full_cmd, cwd=cwd, check=True, shell=True)


def build_llama_from_source(backend: str = "cuda", all_targets: bool = False) -> None:
    cmake_flags = ["cmake", "-B", "build", "-DBUILD_SHARED_LIBS=OFF"]

    if backend == "cuda":
        cmake_flags.append("-DGGML_CUDA=ON")
        info("Configuring CMake for CUDA build...")
    elif backend == "vulkan":
        cmake_flags.append("-DGGML_VULKAN=ON")
        info("Configuring CMake for Vulkan build...")
    elif backend in ["hybrid", "cuda+vulkan"]:
        cmake_flags.extend(["-DGGML_CUDA=ON", "-DGGML_VULKAN=ON"])
        info("Configuring CMake for Hybrid (CUDA + Vulkan) build...")
    else:
        raise ValueError(f"Unknown backend: {backend}")

    run_cmake_cmd(cmake_flags, cwd=SOURCE_DIR)

    targets = ["llama-server"]
    if all_targets:
        targets.extend(["llama-cli", "llama-mtmd-cli", "llama-gguf-split"])

    info(f"Building targets: {', '.join(targets)}...")
    run_cmake_cmd(
        ["cmake", "--build", "build", "--config", "Release", "-j", "--target"] + targets,
        cwd=SOURCE_DIR
    )

    info("Copying built binaries to bin/...")
    BIN_DIR.mkdir(exist_ok=True)
    release_bin_dir = SOURCE_DIR / "build" / "bin" / "Release"
    if not release_bin_dir.exists():
        release_bin_dir = SOURCE_DIR / "build" / "bin"

    copied = 0
    for ext in ["*.exe", "*.dll", "*.spv"]:
        for file_path in release_bin_dir.glob(ext):
            shutil.copy2(file_path, BIN_DIR / file_path.name)
            copied += 1

    if copied == 0:
        warn("No .exe or .dll files were found to copy!")
    else:
        ok(f"Copied {copied} binaries/assets successfully.")


def update_via_build(
    backend: str = "cuda",
    all_targets: bool = False,
    force: bool = False,
    no_backup: bool = False,
    no_diff: bool = False
) -> None:
    # Check source build prerequisites
    diag = check_prerequisites()
    if not diag["git"]:
        err("Git is required to pull source code. Please install Git.")
        sys.exit(1)
    if not diag["vcvars"]:
        err("Visual Studio C++ Build Tools (vcvars64.bat) not found. Required to compile C++ on Windows.")
        sys.exit(1)
    if backend in ["cuda", "hybrid"] and not diag["cuda_nvcc"]:
        err("NVIDIA CUDA Toolkit (nvcc) not found. Required for CUDA & Hybrid source builds.")
        sys.exit(1)
    if backend in ["vulkan", "hybrid"] and not diag["vulkan_sdk"]:
        err("LunarG Vulkan SDK (glslc) not found. Required to compile Vulkan compute shaders from source.")
        _print_safe(f"  {Col.CYAN}-> Install via winget: winget install KhronosGroup.VulkanSDK{Col.RESET}")
        sys.exit(1)

    installed_ver = get_installed_version()
    installed_sha = clean_version_tag(installed_ver)

    git_pull_or_clone(SOURCE_DIR)
    new_sha = git_get_sha(SOURCE_DIR)

    if not new_sha:
        err("Could not determine git commit SHA from source directory.")
        sys.exit(1)

    if installed_sha == new_sha and not force:
        ok(f"Already up-to-date with commit: {new_sha[:7]}")
        print(f"  {Col.DIM}Use --force to rebuild anyway.{Col.RESET}")
        return

    new_tag = git_get_closest_tag(SOURCE_DIR, new_sha) or "unknown"
    backend_label = backend.upper()
    print(f"\n  {Col.BOLD}Building Commit: {Col.GREEN}{new_sha[:7]}{Col.RESET} (release {new_tag}+) [{backend_label}]")
    if installed_sha:
        installed_tag = git_get_closest_tag(SOURCE_DIR, installed_sha) or "unknown"
        print(f"  {Col.BOLD}Installed:       {Col.YELLOW}{installed_sha[:7]}{Col.RESET} (release {installed_tag}+)")
    print()

    # Capture help before
    help_before = ""
    if not no_diff:
        help_before = load_help_snapshot()
        if not help_before:
            help_before = capture_help()

    # Backup
    if not no_backup:
        backup_bin()

    try:
        t0 = time.monotonic()
        build_llama_from_source(backend=backend, all_targets=all_targets)
        elapsed = time.monotonic() - t0
        ok(f"Built successfully in {elapsed:.1f}s")
    except subprocess.CalledProcessError as e:
        err(f"Build failed with exit code {e.returncode}")
        if not no_backup and BACKUP_DIR.exists():
            warn("Restoring from backup...")
            rollback()
        sys.exit(1)

    save_build_info(new_sha, mode="build", backend=backend)

    # Capture help after & diff
    if not no_diff:
        help_after = capture_help()
        save_help_snapshot(help_after)
        show_help_diff_report(help_before, help_after)

    # Changelog
    if installed_sha and new_sha != installed_sha:
        info("Fetching changelog...")
        changelog = fetch_changelog(installed_sha, new_sha)
        if changelog:
            print_changelog(changelog, installed_sha[:7], new_sha[:7])

    # HuggingFace Community Qwen Template
    print()
    download_hf_template()

    print()
    _print_safe(f"{Col.GREEN}{'=' * 60}{Col.RESET}")
    _print_safe(f"{Col.GREEN}{Col.BOLD}  + Build complete! Now running {new_sha[:7]} [{backend_label}]{Col.RESET}")
    _print_safe(f"{Col.GREEN}{'=' * 60}{Col.RESET}")
    if not no_backup:
        _print_safe(f"  {Col.DIM}Rollback available: uv run update_llama.py --rollback{Col.RESET}")
    print()


# ── Main Entrypoint & Interactive Menu ─────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update or build llama.cpp (Windows CUDA / Vulkan / Hybrid) with Lazy Release and Optimized Source modes."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--download", "--release", dest="mode_download", action="store_true",
        help="Lazy Mode: Download official prebuilt GitHub release binaries (Zero compiler required)."
    )
    group.add_argument(
        "--build", "--source", dest="mode_build", action="store_true",
        help="Full Optimized Mode: Compile latest master with CMake + MSVC + (CUDA / Vulkan / Hybrid)."
    )
    group.add_argument(
        "--doctor", "--check-prereqs", dest="mode_doctor", action="store_true",
        help="Inspect system toolchain and check prerequisite readiness."
    )
    group.add_argument(
        "--rollback", dest="mode_rollback", action="store_true",
        help="Restore the previous version from bin_backup/."
    )

    parser.add_argument(
        "--backend", type=str, choices=["cuda", "vulkan", "hybrid"], default=None,
        help="Target acceleration backend: 'cuda', 'vulkan', or 'hybrid' (defaults to last configured backend)."
    )
    parser.add_argument(
        "--vulkan", dest="flag_vulkan", action="store_true",
        help="Convenience shortcut for --backend vulkan."
    )
    parser.add_argument(
        "--hybrid", dest="flag_hybrid", action="store_true",
        help="Convenience shortcut for --backend hybrid (builds with both CUDA and Vulkan)."
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        help="Pin release download or changelog to a specific tag (e.g. b10603)."
    )
    parser.add_argument(
        "--cuda", type=str, default=None,
        help="Specify preferred CUDA version for release download (e.g. 13.3, 12.4)."
    )
    parser.add_argument(
        "--all-targets", action="store_true",
        help="Build all tools (llama-cli, llama-mtmd-cli, llama-gguf-split) during source build."
    )
    parser.add_argument(
        "--sync-source", action="store_true",
        help="Pull/clone Source/llama.cpp for local agent code inspection & parameter tuning."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force update/rebuild even if already up-to-date."
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip creating a backup before updating."
    )
    parser.add_argument(
        "--no-diff", action="store_true",
        help="Skip the --help diff after updating."
    )

    args = parser.parse_args()

    # Load last build metadata
    last_info = load_build_info()
    last_backend = last_info.get("backend", "cuda")
    last_mode = last_info.get("mode", "release")
    last_ver = last_info.get("version")

    # Check if user specified backend explicitly or if we have a saved state
    has_explicit_backend = bool(args.flag_vulkan or args.flag_hybrid or args.backend)
    has_saved_backend = bool(BUILD_INFO_FILE.exists() and last_info.get("backend"))

    # Determine effective backend (CLI flag overrides saved state)
    if args.flag_vulkan:
        backend = "vulkan"
    elif args.flag_hybrid:
        backend = "hybrid"
    elif args.backend:
        backend = args.backend
    else:
        backend = last_backend

    # Direct flag dispatch
    if args.mode_doctor:
        check_prerequisites()
        return

    if args.mode_rollback:
        rollback()
        return

    if args.mode_download:
        if not has_explicit_backend and not has_saved_backend:
            print(f"\n{Col.BOLD}First-time release download detected! Select your target GPU backend:{Col.RESET}")
            print(f"  {Col.CYAN}[1]{Col.RESET} CUDA (NVIDIA GPU only ⭐)")
            print(f"  {Col.CYAN}[2]{Col.RESET} Vulkan (Intel Arc / AMD / Universal)")
            while True:
                d_choice = input("\nChoose [1-2] (default 1): ").strip() or "1"
                if d_choice in ["1", "2"]:
                    break
            backend = {"1": "cuda", "2": "vulkan"}[d_choice]

        if backend == "hybrid":
            warn("Hybrid mode is only available for source builds. Falling back to CUDA prebuilt release.")
            backend = "cuda"
        update_via_release(
            tag=args.tag, cuda_ver=args.cuda, backend=backend, sync_source=args.sync_source,
            force=args.force, no_backup=args.no_backup, no_diff=args.no_diff
        )
        return

    if args.mode_build:
        if not has_explicit_backend and not has_saved_backend:
            print(f"\n{Col.BOLD}First-time source build detected! Select your target GPU backend:{Col.RESET}")
            print(f"  {Col.CYAN}[1]{Col.RESET} CUDA (NVIDIA GPU only ⭐)")
            print(f"  {Col.CYAN}[2]{Col.RESET} Vulkan (Intel Arc / AMD / Universal)")
            print(f"  {Col.CYAN}[3]{Col.RESET} Hybrid (CUDA + Vulkan — Dual NVIDIA & Intel Arc)")
            while True:
                b_choice = input("\nChoose [1-3] (default 1): ").strip() or "1"
                if b_choice in ["1", "2", "3"]:
                    break
            backend = {"1": "cuda", "2": "vulkan", "3": "hybrid"}[b_choice]

        update_via_build(
            backend=backend, all_targets=args.all_targets, force=args.force,
            no_backup=args.no_backup, no_diff=args.no_diff
        )
        return

    # Interactive menu if no mode specified
    default_choice = "1"
    if last_mode == "release":
        default_choice = "1" if last_backend == "cuda" else "2"
    elif last_mode == "build":
        if last_backend == "cuda":
            default_choice = "3"
        elif last_backend == "vulkan":
            default_choice = "4"
        elif last_backend == "hybrid":
            default_choice = "5"

    mode_labels = {
        "1": "Lazy Release (CUDA / NVIDIA)",
        "2": "Lazy Release (Vulkan / Intel Arc / AMD)",
        "3": "Source Build (CUDA / NVIDIA)",
        "4": "Source Build (Vulkan / Intel Arc)",
        "5": "Source Build Hybrid (CUDA + Vulkan)"
    }

    print(f"\n{Col.BOLD}llabrun — llama.cpp Engine Builder & Updater{Col.RESET}")
    if last_ver:
        _print_safe(f"  {Col.DIM}Currently Installed:{Col.RESET} {Col.BOLD}{last_ver}{Col.RESET} [{mode_labels.get(default_choice, 'Custom')}]")

    print(f"  {Col.CYAN}[1]{Col.RESET} Lazy Mode: Download latest {Col.BOLD}CUDA{Col.RESET} release (NVIDIA ⭐)")
    print(f"  {Col.CYAN}[2]{Col.RESET} Lazy Mode: Download latest {Col.BOLD}Vulkan{Col.RESET} release (Intel Arc / AMD / Universal)")
    print(f"  {Col.CYAN}[3]{Col.RESET} Full Optimized Mode: Build from source with {Col.BOLD}CUDA{Col.RESET} (NVIDIA)")
    print(f"  {Col.CYAN}[4]{Col.RESET} Full Optimized Mode: Build from source with {Col.BOLD}Vulkan{Col.RESET} (Intel Arc / Universal)")
    print(f"  {Col.CYAN}[5]{Col.RESET} Full Optimized Mode: Build {Col.BOLD}Hybrid (CUDA + Vulkan){Col.RESET} from source")
    print(f"  {Col.CYAN}[6]{Col.RESET} System Diagnostics & Prerequisites (Doctor)")
    print(f"  {Col.CYAN}[7]{Col.RESET} Rollback to previous binary backup")

    while True:
        prompt_suffix = f" (default [{default_choice}] {mode_labels.get(default_choice, '')})" if last_ver else ""
        choice = input(f"\nChoose [1-7]{prompt_suffix}: ").strip()
        if not choice and last_ver:
            choice = default_choice
        if choice in ["1", "2", "3", "4", "5", "6", "7"]:
            break

    if choice == "1":
        update_via_release(
            tag=args.tag, cuda_ver=args.cuda, backend="cuda", sync_source=args.sync_source,
            force=args.force, no_backup=args.no_backup, no_diff=args.no_diff
        )
    elif choice == "2":
        update_via_release(
            tag=args.tag, cuda_ver=args.cuda, backend="vulkan", sync_source=args.sync_source,
            force=args.force, no_backup=args.no_backup, no_diff=args.no_diff
        )
    elif choice == "3":
        update_via_build(
            backend="cuda", all_targets=args.all_targets, force=args.force,
            no_backup=args.no_backup, no_diff=args.no_diff
        )
    elif choice == "4":
        update_via_build(
            backend="vulkan", all_targets=args.all_targets, force=args.force,
            no_backup=args.no_backup, no_diff=args.no_diff
        )
    elif choice == "5":
        update_via_build(
            backend="hybrid", all_targets=args.all_targets, force=args.force,
            no_backup=args.no_backup, no_diff=args.no_diff
        )
    elif choice == "6":
        check_prerequisites()
    elif choice == "7":
        rollback()


if __name__ == "__main__":
    main()
