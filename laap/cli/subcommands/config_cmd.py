"""
LAAP Config — 模型与系统配置命令
==================================
用法:
    laap config list           # 列出所有 Provider
    laap config set <key> <val>  # 设置配置项
    laap config show           # 显示当前配置
    laap config model <name>   # 选择默认模型
"""
import json, logging, os, sys
from pathlib import Path

logger = logging.getLogger(__name__)

from laap.config.paths import get_aris_brain_dir

CONFIG_DIR = get_aris_brain_dir()
CONFIG_FILE = CONFIG_DIR / "laap_config.json"
ENV_FILE = CONFIG_DIR / ".env"

DEFAULT_CONFIG = {
    "llm": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "agent": {
        "max_steps": 10,
        "memory_enabled": True,
        "tools_enabled": True,
    },
}

# 已知 Provider
KNOWN_PROVIDERS = {
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": ["deepseek-v4", "deepseek-v4-thinking", "deepseek-r2", "deepseek-chat"],
        "base_url": "https://api.deepseek.com",
        "docs": "https://platform.deepseek.com/api_keys",
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "models": ["gpt-5.5", "gpt-5.5-turbo", "gpt-5.1-mini", "o4-mini", "o4", "o4-pro"],
        "base_url": "https://api.openai.com/v1",
        "docs": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": ["claude-opus-4.5", "claude-sonnet-4.5", "claude-sonnet-4", "claude-haiku-3.5"],
        "base_url": "https://api.anthropic.com",
        "docs": "https://console.anthropic.com/",
    },
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "models": ["openrouter/auto", "anthropic/claude-sonnet-4.5", "openai/gpt-5.5", "deepseek/deepseek-v4"],
        "base_url": "https://openrouter.ai/api/v1",
        "docs": "https://openrouter.ai/keys",
    },
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "models": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "base_url": "https://generativelanguage.googleapis.com",
        "docs": "https://aistudio.google.com/app/apikey",
    },
    "xai": {
        "api_key_env": "XAI_API_KEY",
        "models": ["grok-4", "grok-4-mini", "grok-4-vision"],
        "base_url": "https://api.x.ai/v1",
        "docs": "https://console.x.ai/",
    },
    "ollama": {
        "api_key_env": "",
        "models": ["llama-4", "llama-4-scout", "qwen-3", "qwen-3-32b", "mistral-4", "deepseek-v4-local"],
        "base_url": "http://localhost:11434",
        "docs": "https://ollama.ai/",
    },
}


def _load_config() -> dict:
    """加载配置。"""
    config = dict(DEFAULT_CONFIG)
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text("utf-8"))
            config.update(data)
        except Exception:
            pass
    return config


def _save_config(config: dict):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_env() -> dict:
    """读取 .env 文件。"""
    env = {}
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text("utf-8", errors="replace").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    return env


def _write_env(env: dict):
    content = "\n".join(f"{k}={v}" for k, v in sorted(env.items()) if v)
    ENV_FILE.write_text(content + "\n", encoding="utf-8")


# ─── 命令实现 ──────────────────────────────────────

def cmd_list(args):
    """列出所有支持的 Provider。"""
    print(f"\n  可用 Provider:")
    print(f"  {'='*50}")
    for name, info in KNOWN_PROVIDERS.items():
        key_status = "已设置" if os.environ.get(info["api_key_env"]) or _read_env().get(info["api_key_env"]) else "未设置"
        models = ", ".join(info["models"][:3])
        print(f"\n  {name:15} [{key_status}]")
        print(f"  {'':15} 模型: {models}")
        print(f"  {'':15} 文档: {info['docs']}")
    print()


def cmd_set(args):
    """设置配置项。"""
    key = args.key
    val = args.value
    config = _load_config()
    
    # 支持点号路径: llm.provider deepseek
    if "." in key:
        parts = key.split(".")
        target = config
        for p in parts[:-1]:
            target = target.setdefault(p, {})
        target[parts[-1]] = val
    else:
        config[key] = val
    
    _save_config(config)
    
    # 如果是 API Key，也写入 .env
    env = _read_env()
    for pname, pinfo in KNOWN_PROVIDERS.items():
        if pinfo["api_key_env"] and key == pinfo["api_key_env"]:
            env[key] = val
            os.environ[key] = val
            _write_env(env)
            print(f"  ✅ {key} 已设置并写入 .env")
            return
    
    print(f"  ✅ {key} = {val}")
    print(f"  重启后生效")


def cmd_show(args):
    """显示当前配置。"""
    config = _load_config()
    env = _read_env()
    
    print(f"\n  LAAP 配置:")
    print(f"  {'='*50}")
    print(f"  LLM Provider:  {config.get('llm', {}).get('provider', '?')}")
    print(f"  LLM Model:     {config.get('llm', {}).get('model', '?')}")
    print(f"  温度:          {config.get('llm', {}).get('temperature', '?')}")
    print(f"  最大 Token:    {config.get('llm', {}).get('max_tokens', '?')}")
    print(f"  记忆:          {'开启' if config.get('agent', {}).get('memory_enabled', True) else '关闭'}")
    print(f"  工具:          {'开启' if config.get('agent', {}).get('tools_enabled', True) else '关闭'}")
    print()
    print(f"  API Keys:")
    for name, info in KNOWN_PROVIDERS.items():
        key_name = info["api_key_env"]
        if key_name:
            val = env.get(key_name, "") or os.environ.get(key_name, "")
            status = f"已设置 ({val[:8]}...)" if val else "未设置"
            print(f"    {name:15} {key_name:25} {status}")
    print()


def cmd_model(args):
    """选择默认模型。"""
    provider = args.provider
    model = args.model
    
    config = _load_config()
    config.setdefault("llm", {})["provider"] = provider
    config["llm"]["model"] = model
    _save_config(config)
    
    print(f"  ✅ 默认模型已切换: {provider}/{model}")


def add_subparser(sub):
    """注册 config 子命令。"""
    p = sub.add_parser("config", help="模型与系统配置")
    subsub = p.add_subparsers(dest="config_cmd")
    
    subsub.add_parser("list", help="列出所有 Provider")
    
    set_p = subsub.add_parser("set", help="设置配置项")
    set_p.add_argument("key", help="配置键 (如 llm.provider, DEEPSEEK_API_KEY)")
    set_p.add_argument("value", help="配置值")
    
    subsub.add_parser("show", help="显示当前配置")
    
    model_p = subsub.add_parser("model", help="选择默认模型")
    model_p.add_argument("provider", help="Provider 名称 (deepseek/openai/anthropic/...)")
    model_p.add_argument("model", help="模型名称")


def main(args):
    if args.config_cmd == "list":
        cmd_list(args)
    elif args.config_cmd == "set":
        cmd_set(args)
    elif args.config_cmd == "show":
        cmd_show(args)
    elif args.config_cmd == "model":
        cmd_model(args)
    else:
        cmd_show(args)
