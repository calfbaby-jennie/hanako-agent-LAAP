#!/usr/bin/env python3
"""
LAAP AGENT CLI — 终端交互界面
===============================
设计参考 Hermes CLI，但更简洁、模型最新。
用法:
    laap                    # 启动交互模式
    laap --model deepseek-v4  # 指定模型
    laap --list-models      # 列出可用模型
"""
import argparse, json, logging, os, re, sys, textwrap, time, threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from laap.config.paths import get_aris_brain_dir

os.environ["LAAP_QUIET"] = "1"
logger = logging.getLogger(__name__)

# ─── HACK: suppress noisy loggers ──────────────────
for _l in ["httpcore", "httpx", "urllib3", "openai", "anthropic"]:
    logging.getLogger(_l).setLevel(logging.WARNING)

# ─── prompt_toolkit TUI ────────────────────────────
from prompt_toolkit import PromptSession, print_formatted_text as pt_print
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style as PTStyle
from prompt_toolkit.formatted_text import ANSI as PT_ANSI
from prompt_toolkit.patch_stdout import patch_stdout

# ═══════════════════════════════════════════════════════════
# 模型目录 — 2026年最新版
# ═══════════════════════════════════════════════════════════

MODEL_CATALOG = {
    "deepseek": {
        "name": "DeepSeek",
        "models": [
            "deepseek-v4", "deepseek-v4-thinking", "deepseek-r2", "deepseek-chat",
        ],
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default": "deepseek-v4",
    },
    "openai": {
        "name": "OpenAI",
        "models": [
            "gpt-5.5", "gpt-5.5-turbo", "gpt-5.1-mini", "gpt-5.1-mini-high",
            "o4-mini", "o4", "o4-pro",
        ],
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default": "gpt-5.5",
    },
    "anthropic": {
        "name": "Anthropic",
        "models": [
            "claude-opus-4.5", "claude-sonnet-4.5", "claude-sonnet-4",
            "claude-haiku-3.5",
        ],
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "default": "claude-sonnet-4.5",
    },
    "gemini": {
        "name": "Google Gemini",
        "models": [
            "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite",
        ],
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com",
        "default": "gemini-2.5-pro",
    },
    "openrouter": {
        "name": "OpenRouter",
        "models": [
            "openrouter/auto", "anthropic/claude-sonnet-4.5",
            "openai/gpt-5.5", "deepseek/deepseek-v4",
            "google/gemini-2.5-pro",
        ],
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "default": "openrouter/auto",
    },
    "xai": {
        "name": "xAI",
        "models": [
            "grok-4", "grok-4-mini", "grok-4-vision",
        ],
        "api_key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "default": "grok-4",
    },
    "ollama": {
        "name": "Ollama (本地)",
        "models": [
            "llama-4", "llama-4-scout", "qwen-3", "qwen-3-32b",
            "mistral-4", "deepseek-v4-local",
        ],
        "api_key_env": "",
        "base_url": "http://localhost:11434",
        "default": "llama-4",
    },
}

STYLES = {
    "brand": "ansicyan bold",
    "cmd": "ansigreen bold",
    "url": "ansiblue underline",
    "dim": "ansibrightblack",
    "ok": "ansigreen",
    "warn": "ansiyellow",
    "err": "ansired bold",
    "num": "ansimagenta",
}

# ─── 横幅 ──────────────────────────────────────────

BANNER = f"""
  {chr(27)}[36m╔══════════════════════════════════════════════╗
  ║     {chr(27)}[1mLAAP AGENT{chr(27)}[0m{chr(27)}[36m     Token-Efficient AI Agent     ║
  ║  {chr(27)}[33m零LLM优先 · 有色Petri网 · AGI引擎{chr(27)}[36m      ║
  ╚══════════════════════════════════════════════╝{chr(27)}[0m"""


def print_banner():
    print(BANNER)
    print(f"  {chr(27)}[90mType '/' for commands, 'Ctrl+C' to exit{chr(27)}[0m")
    print()


# ─── 命令处理器 ────────────────────────────────────

COMMANDS = {
    "/help": "显示帮助信息",
    "/model": "切换模型 (例: /model gpt-5.5)",
    "/provider": "切换 Provider (例: /provider openai)",
    "/tools": "列出可用工具",
    "/skills": "列出可用技能",
    "/memory": "查看记忆状态",
    "/cost": "显示 Token 消耗统计",
    "/clear": "清屏",
    "/exit": "退出",
}


def handle_command(cmd: str, state: dict) -> Optional[str]:
    parts = cmd.strip().split(maxsplit=1)
    base = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if base in ("/exit", "/quit", "/q"):
        print(f"\n  {chr(27)}[33mLAAP AGENT 已退出{chr(27)}[0m")
        sys.exit(0)

    elif base == "/help":
        out = [f"\n  {chr(27)}[36m可用命令:{chr(27)}[0m"]
        for c, h in COMMANDS.items():
            out.append(f"    {chr(27)}[32m{c:12}{chr(27)}[0m {h}")
        out.append(f"    {chr(27)}[32m/exit{'':8}{chr(27)}[0m 退出")
        return "\n".join(out)

    elif base == "/model":
        if not arg:
            models = MODEL_CATALOG[state["provider"]]["models"]
            return f"  当前模型: {chr(27)}[33m{state['model']}{chr(27)}[0m\n  可用: {', '.join(models)}"
        state["model"] = arg
        return f"  {chr(27)}[32m✓{chr(27)}[0m 模型已切换: {chr(27)}[33m{arg}{chr(27)}[0m"

    elif base == "/provider":
        if not arg:
            return f"  当前 Provider: {chr(27)}[33m{state['provider']}{chr(27)}[0m\n  可用: {', '.join(MODEL_CATALOG.keys())}"
        if arg not in MODEL_CATALOG:
            return f"  {chr(27)}[31m✗{chr(27)}[0m 未知 Provider: {arg}"
        state["provider"] = arg
        state["model"] = MODEL_CATALOG[arg]["default"]
        return f"  {chr(27)}[32m✓{chr(27)}[0m 已切换至 {arg}, 模型: {chr(27)}[33m{state['model']}{chr(27)}[0m"

    elif base == "/tools":
        return f"  {chr(27)}[36m工具系统已加载{chr(27)}[0m (运行时查看)"

    elif base == "/cost":
        return f"  本次会话 Token: {chr(27)}[35m{state.get('tokens', 0):,}{chr(27)}[0m"

    elif base == "/clear":
        print("\033[2J\033[H", end="")
        print_banner()
        return None

    return f"  {chr(27)}[31m未知命令: {base}{chr(27)}[0m  输入 /help 查看可用命令"


# ─── 主循环 ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LAAP AGENT CLI")
    parser.add_argument("--model", default=None, help="指定模型")
    parser.add_argument("--provider", default=None, help="指定 Provider")
    parser.add_argument("--list-models", action="store_true", help="列出所有模型")
    parser.add_argument("--toolsets", default=None, help="工具集 (逗号分隔)")
    parser.add_argument("text", nargs="*", help="单轮对话文本")
    args = parser.parse_args()

    # List models mode
    if args.list_models:
        print(f"\n  {chr(27)}[36mLAAP AGENT — 可用模型{chr(27)}[0m\n")
        for key, info in MODEL_CATALOG.items():
            print(f"  {chr(27)}[33m{key:15}{chr(27)}[0m {info['name']}")
            for m in info["models"]:
                def_ = " ← 默认" if m == info["default"] else ""
                print(f"    {chr(27)}[32m{m:30}{chr(27)}[0m{chr(27)}[90m{def_}{chr(27)}[0m")
            print()
        return

    # State
    provider = args.provider or "deepseek"
    if provider not in MODEL_CATALOG:
        provider = "deepseek"
    model = args.model or MODEL_CATALOG[provider]["default"]

    state = {
        "provider": provider,
        "model": model,
        "tokens": 0,
        "messages": [],
    }

    # Single-turn mode
    if args.text:
        text = " ".join(args.text)
        state["messages"].append({"role": "user", "content": text})
        print(f"\n  {chr(27)}[33m[{model}]{chr(27)}[0m 处理中...")
        print(f"  {chr(27)}[32m{text}{chr(27)}[0m")
        print()
        return

    # Interactive mode
    print_banner()

    history_path = Path.home() / ".laap" / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    session = PromptSession(
        history=FileHistory(str(history_path / "cli_history.txt")),
        style=PTStyle.from_dict({
            "prompt": "#44ddbb bold",
            "trailing_input": "#666666",
        }),
    )

    # Tooltip showing current model
    _model_display = f"{provider}/{model}"

    print(f"  {chr(27)}[90m模型: {_model_display}  |  输入 /help 查看命令{chr(27)}[0m")
    print()

    while True:
        try:
            with patch_stdout():
                user_input = session.prompt(
                    [("class:prompt", "λ ")],
                    vi_mode=False,
                )
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {chr(27)}[33mLAAP AGENT 已退出{chr(27)}[0m")
            break

        text = user_input.strip()
        if not text:
            continue

        # Command handling
        if text.startswith("/"):
            result = handle_command(text, state)
            if result:
                print(result)
            # Update model display if changed
            _model_display = f"{state['provider']}/{state['model']}"
            # Restore prompt hint
            continue

        # Chat message
        state["messages"].append({"role": "user", "content": text})
        state["tokens"] += len(text) // 2

        # Show user message
        print(f"\n  {chr(27)}[36m┌─ {chr(27)}[1m你{chr(27)}[0m{chr(27)}[36m{chr(27)}[0m")
        print(f"  │ {text}")
        print(f"  {chr(27)}[36m└─ {chr(27)}[33m[{_model_display}]{chr(27)}[0m")

        # Try zero-LLM path first
        try:
            sys.path.insert(0, str(get_aris_brain_dir()))
            from aether_agent_loop import get_agent
            agent = get_agent()
            t0 = time.time()
            result = agent.process(text)
            ms = (time.time() - t0) * 1000

            mode_tag = f"{chr(27)}[32m零LLM{chr(27)}[0m" if result.direct else f"{chr(27)}[33mLLM{chr(27)}[0m"
            print(f"\n  {chr(27)}[36m┌─ {chr(27)}[1mLAAP{chr(27)}[0m{chr(27)}[36m{chr(27)}[0m  {mode_tag}  {chr(27)}[90m{ms:.0f}ms{chr(27)}[0m")
            for line in result.output.split("\n"):
                print(f"  │ {line}")
            print(f"  {chr(27)}[36m└─{chr(27)}[0m")
        except Exception as e:
            print(f"\n  {chr(27)}[31m✗ 错误: {e}{chr(27)}[0m")

        state["tokens"] += result.tokens_used if 'result' in dir() else 0

    print()


if __name__ == "__main__":
    main()
