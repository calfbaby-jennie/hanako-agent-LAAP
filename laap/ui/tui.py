"""LAAP - Hermes-level Golden Dragon TUI.

A full-screen Textual terminal interface for LAAP. Displays a Hermes-inspired
welcome banner, dynamic tool/skill/MCP inventory, slash-command completion,
real-time system metrics, PSI cognitive state, gateway status, interactive
approval modal for dangerous commands, rich chat messaging, tool/MCP progress
panels, and coding progress indicators.

If Textual is not installed the public symbols still exist but ``run_tui()``
and ``LAAP_TUI.run()`` raise a clear runtime error so callers can fall back to
the REPL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from laap.config.paths import get_video_dir

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore

from rich.console import Group
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich.spinner import Spinner
from rich.progress import BarColumn, Progress, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown

logger = logging.getLogger(__name__)

# ── TUI error logging ────────────────────────────────────────────
_TUI_LOG_DIR = Path.home() / ".laap" / "logs"
_TUI_LOG_FILE = _TUI_LOG_DIR / "tui.log"
try:
    _TUI_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _tui_file_handler = logging.FileHandler(_TUI_LOG_FILE, encoding="utf-8")
    _tui_file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logger.addHandler(_tui_file_handler)
    logger.setLevel(logging.DEBUG)
except Exception:
    _tui_file_handler = None

# ── Textual availability ─────────────────────────────────────────
try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
    from textual.widgets import Static, Input, Button, Label, ListView, ListItem
    from textual.reactive import reactive
    from textual.screen import Screen, ModalScreen
    from textual.binding import Binding
    from textual.color import Color
    from textual.message import Message
    from textual.worker import Worker, get_current_worker
    HAS_TEXTUAL = True
except ImportError as _textual_import_err:  # pragma: no cover
    HAS_TEXTUAL = False
    App = object  # type: ignore
    Screen = object  # type: ignore
    ModalScreen = object  # type: ignore
    Static = Input = Button = Label = ListView = ListItem = object  # type: ignore
    Container = Horizontal = Vertical = ScrollableContainer = object  # type: ignore
    Binding = lambda *a, **kw: None  # type: ignore
    Color = lambda *a: None  # type: ignore
    Worker = get_current_worker = None  # type: ignore

    class _ReactiveStub:
        def __call__(self, default, *args, **kwargs):
            return property(lambda self: default)
    reactive = _ReactiveStub()  # type: ignore

    class Message:  # type: ignore
        pass


# ── Color Scheme ─────────────────────────────────────────────────
class Colors:
    GOLD = "#FFD700"
    GOLD_BRIGHT = "#FFE55C"
    GOLD_DARK = "#B8960C"
    GOLD_LIGHT = "#FFED80"
    CRITICAL = "#DC143C"
    SUCCESS = "#00D68F"
    INFO = "#4FC1FF"
    WARNING = "#FFA500"
    ORANGE = "#FF8C00"
    # Pure black / dark palette
    BG_PURE = "#000000"
    BG_DARK = "#0A0A0A"
    BG_INPUT = "#0A0A0A"
    BG_SIDEBAR = "#0D0D0D"
    BG_PANEL = "#111111"
    BG_MEDIUM = "#1A1A1A"
    BG_LIGHT = "#252525"
    BG_HIGHLIGHT = "#2A2A2A"
    TEXT = "#C8C8C8"
    TEXT_DIM = "#888888"
    TEXT_BRIGHT = "#FFFFFF"
    BRONZE = "#CD7F32"
    BRONZE_DARK = "#8B6914"


# ── Windows terminal compatibility ───────────────────────────────
if sys.platform == "win32":
    try:
        import ctypes
        k32 = ctypes.windll.kernel32
        STD_OUTPUT_HANDLE = -11
        handle = k32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        k32.GetConsoleMode(handle, ctypes.byref(mode))
        k32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception as e:
        logger.debug("[WARN] Windows console mode setup failed: %s", e)


# ── Agent singleton (preserved for backward compatibility) ───────
_agent_instance = None
_avatar_bridge = None


def get_avatar_bridge():
    global _avatar_bridge
    if _avatar_bridge is None:
        try:
            from laap.ui.avatar_3d import AvatarBridge
            _avatar_bridge = AvatarBridge()
        except Exception as e:
            logger.debug("[WARN] Avatar bridge unavailable: %s", e)
            return None
    return _avatar_bridge


def _active_config_summary() -> Dict[str, str]:
    """Return the currently active provider/model/key/base_url from ConfigManager."""
    try:
        from laap.cli.config_manager import config_manager
        summary = config_manager.get_summary()
        return {
            "provider": summary.get("provider") or "deepseek",
            "model": summary.get("model") or "deepseek-v4-flash",
            "api_key": summary.get("api_key") or "",
            "base_url": summary.get("base_url") or "",
        }
    except Exception as e:
        logger.debug("[WARN] config summary failed: %s", e)
        return {
            "provider": os.environ.get("LAAP_PROVIDER", "deepseek"),
            "model": os.environ.get("LAAP_MODEL", "deepseek-v4-flash"),
            "api_key": os.environ.get("LAAP_API_KEY", ""),
            "base_url": os.environ.get("LAAP_BASE_URL", ""),
        }


def _sync_env_from_config():
    """Sync LAAP_* environment variables with ConfigManager's active config."""
    try:
        from laap.cli.config_manager import config_manager
        config_manager.ensure_config_dir()
        config_manager.load_env_overrides()
        config_manager.apply_to_environment()
    except Exception as e:
        logger.debug("[WARN] env sync failed: %s", e)


def get_or_create_agent(force_reinit: bool = False):
    global _agent_instance
    if _agent_instance is not None and not force_reinit:
        return _agent_instance
    try:
        _sync_env_from_config()
        cfg = _active_config_summary()
        provider = cfg["provider"]
        model = cfg["model"]
        api_key = cfg["api_key"]
        base_url = cfg["base_url"]
        from laap.llm.factory import LLMFactory
        factory = LLMFactory(default_provider=provider, default_model=model or None)
        from laap.agent.lifelike import LifelikeAgent, LifelikeConfig
        agent_cfg = LifelikeConfig(
            name="Ao", verbose=False,
            llm_provider=provider,
            llm_model=model,
        )
        _agent_instance = LifelikeAgent(
            config=agent_cfg, llm_factory=factory, show_banner=False
        )
        # If the factory couldn't initialize the LLM (missing key etc.) but we
        # have explicit credentials in the config, inject them now.
        if _agent_instance and _agent_instance.llm is None and api_key:
            try:
                _agent_instance.llm = factory.get(
                    name=provider, model=model,
                    api_key=api_key, base_url=base_url or None,
                )
            except Exception as e:
                logger.debug("[WARN] explicit LLM injection failed: %s", e)
        return _agent_instance
    except Exception as e:
        logging.getLogger("laap.tui").error("[ERROR] Agent init: %s", e)
        return None


def reinitialize_agent() -> bool:
    """Reinitialize the singleton agent with the current environment config."""
    global _agent_instance
    _agent_instance = None
    return get_or_create_agent(force_reinit=True) is not None


# ── Markdown rendering helpers ───────────────────────────────────
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_CODE = re.compile(r"`([^`]+)`")
_RE_ANSI = re.compile(r"\x1b\[.*?m")
_RE_CODE_BLOCK = re.compile(r"```(\w*)\n?(.*?)```", re.DOTALL)


def strip_ansi(text: str) -> str:
    return _RE_ANSI.sub("", text)


def _detect_language(text: str) -> str:
    """Best-effort language detection for code blocks."""
    lowered = text.lower()
    if any(k in lowered for k in ("def ", "import ", "class ", "print(")):
        return "python"
    if any(k in lowered for k in ("function", "const ", "let ", "=>")):
        return "javascript"
    if any(k in lowered for k in ("<html", "<div", "<!doctype")):
        return "html"
    if any(k in lowered for k in ("select ", "from ", "where ", "insert ")):
        return "sql"
    if any(k in lowered for k in ("#include", "int main")):
        return "cpp"
    return "text"


def md_render(t: str) -> Text:
    t = strip_ansi(t)
    r = Text()
    i = 0
    while i < len(t):
        m = _RE_BOLD.search(t, i)
        n = _RE_CODE.search(t, i)
        next_match = None
        if m and n:
            next_match = m if m.start() < n.start() else n
        elif m:
            next_match = m
        elif n:
            next_match = n
        if not next_match:
            r.append(t[i:])
            break
        if next_match.start() > i:
            r.append(t[i:next_match.start()])
        g = next_match.group(1)
        if next_match.lastindex == 1:
            r.append(g, style=Style(bold=True, color=Colors.GOLD))
        else:
            r.append(g, style=Style(color=Colors.GOLD_LIGHT, bgcolor="#1A1A1A"))
        i = next_match.end()
    return r


def md_inline(t: str) -> Any:
    """Render markdown with syntax-highlighted code blocks."""
    t = strip_ansi(t)
    parts = []
    last = 0
    for m in _RE_CODE_BLOCK.finditer(t):
        if m.start() > last:
            parts.append(md_render(t[last:m.start()]))
        lang = m.group(1).strip() or _detect_language(m.group(2))
        code = m.group(2)
        parts.append(Syntax(code, lang, theme="monokai", background_color="#0A0A0A", line_numbers=False))
        last = m.end()
    if last < len(t):
        parts.append(md_render(t[last:]))
    if len(parts) == 1:
        return parts[0]
    return Group(*parts)


def md_plain(t: str) -> Text:
    return md_render(t)


# ── Markdown 公开导出（P0.5）─────────────────────────────────────
# 提供 rich_markdown 与 StreamingMarkdown 公开符号，供 session_search 等模块
# 通过 `from laap.ui.tui import rich_markdown, StreamingMarkdown` 导入使用。
# 简化策略：剥离常见 markdown 标记后包装为 rich.text.Text（含 .plain 属性），
# 这样测试与 UI 都能通过 .plain 访问纯文本内容。
_RE_MD_EXP_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_MD_EXP_ITALIC = re.compile(r"\*(.+?)\*")
_RE_MD_EXP_CODE = re.compile(r"`([^`]+)`")
_RE_MD_EXP_HEADING = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)
_RE_MD_EXP_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _strip_markdown_markers(text: str) -> str:
    """剥离常见 markdown 标记（**bold**、*italic*、`code`、# heading、[link](url)）。"""
    if not text:
        return ""
    # 顺序很重要：先 code（避免 code 内的 * 被吃掉），再 bold（在 italic 前以处理 **），
    # 再 italic、heading、link。
    result = _RE_MD_EXP_CODE.sub(r"\1", text)
    result = _RE_MD_EXP_BOLD.sub(r"\1", result)
    result = _RE_MD_EXP_ITALIC.sub(r"\1", result)
    result = _RE_MD_EXP_HEADING.sub(r"\1", result)
    result = _RE_MD_EXP_LINK.sub(r"\1", result)
    return result


def rich_markdown(text: str) -> Text:
    """将 markdown 文本渲染为 rich Text 对象（含 .plain 属性）。

    基础实现：剥离 markdown 标记后包装为 Text，便于测试与 UI 通过
    `.plain` 访问纯文本内容。
    """
    return Text(_strip_markdown_markers(text))


class StreamingMarkdown:
    """流式 markdown 渲染器。

    提供静态渲染与纯文本提取接口（与 tests/test_session_search.py 对齐）：
      - ``StreamingMarkdown.render(text)``：静态方法，返回 rich Text（含 .plain）
      - ``StreamingMarkdown.strip(text)``：静态方法，返回去除标记的纯字符串
    """

    def __init__(self, text: str = ""):
        self._text = text

    def append(self, text: str) -> None:
        """增量追加文本（流式场景）。"""
        self._text += text

    def __str__(self) -> str:
        return self._text

    @staticmethod
    def render(text: str) -> Text:
        """静态渲染：将 markdown 文本转为 rich Text（含 .plain 属性）。"""
        return Text(_strip_markdown_markers(text))

    @staticmethod
    def strip(text: str) -> str:
        """静态方法：剥离 markdown 标记，返回纯文本。"""
        return _strip_markdown_markers(text)


# ── Logo rendering (delegates to hermes_banner when possible) ────
def render_logo() -> str:
    """Render the LAAP ASCII logo containing the literal text 'LAAP'."""
    lines = [
        f"  [{Colors.GOLD}]██╗      █████╗  █████╗ █████╗ █████╗ ██████╗  █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]",
        f"  [{Colors.GOLD}]██║     ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]",
        f"  [{Colors.GOLD_DARK}]██║     ███████║███████║███████║██████╔╝███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]",
        f"  [{Colors.GOLD_DARK}]██║     ██╔══██║██╔══██║██╔══██║██╔═══╝ ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]",
        f"  [{Colors.GOLD_LIGHT}]███████╗██║  ██║██║  ██║██║  ██║██║     ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]",
        f"  [{Colors.GOLD_LIGHT}]╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]",
        f"  [{Colors.BRONZE}]                 LAAP — Lifeform Autonomous Adaptive Protocol [/]",
    ]
    return "\n".join(lines)


# ── Utility collectors ───────────────────────────────────────────
_BUILT_IN_SKILL_CATEGORIES = [
    "autonomous-ai-agents", "backend", "creative", "data-science", "design",
    "devops", "email", "frontend", "general", "github", "laap", "media",
    "mlops", "note-taking", "productivity", "red-teaming", "research",
    "smart-home", "software-development",
]


def _load_tool_registry() -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Return (tools_by_category, all_tool_names) from the real registry."""
    try:
        from laap.tools.tool_registry import TOOL_REGISTRY, discover_and_register
        discover_and_register()
    except Exception as e:
        logger.debug("[WARN] Tool registry load failed: %s", e)
        return {}, []

    by_cat: Dict[str, List[Dict[str, Any]]] = {}
    names: List[str] = []
    for name, meta in TOOL_REGISTRY.items():
        cat = meta.get("category") or "general"
        by_cat.setdefault(cat, []).append(meta)
        names.append(name)

    # Merge MCP servers as mcp category pseudo-tools without blocking IO
    try:
        mcp_servers = _load_mcp_servers()
        if mcp_servers:
            mcp_tools = []
            for sname, transport, enabled in mcp_servers:
                if not enabled:
                    continue
                meta = {
                    "name": f"mcp.{sname}",
                    "category": "mcp",
                    "description": f"MCP server ({transport})",
                    "schema": {"type": "object", "properties": {}, "required": []},
                }
                mcp_tools.append(meta)
            if mcp_tools:
                by_cat.setdefault("mcp", []).extend(mcp_tools)
                names.extend(m["name"] for m in mcp_tools)
    except Exception as e:
        logger.debug("[WARN] MCP merge into tool registry failed: %s", e)

    return by_cat, sorted(names)


def _load_mcp_servers() -> List[Tuple[str, str, bool]]:
    """Return [(name, transport, enabled), ...] from MCP config."""
    try:
        from laap.mcp.config import load_config
        cfg = load_config()
        out = []
        for name, data in cfg.items():
            transport = data.get("transport", "stdio")
            enabled = bool(data.get("enabled", True))
            out.append((name, transport, enabled))
        return out
    except Exception as e:
        logger.debug("[WARN] MCP config load failed: %s", e)
        return []


def _load_mcp_tools() -> List[Tuple[str, str, str]]:
    """Return [(server, tool_name, description), ...] by discovering MCP tools.

    Best-effort: tries to connect to each enabled stdio server and list tools.
    Falls back to server-name-only entries if discovery fails.
    """
    out: List[Tuple[str, str, str]] = []
    try:
        from laap.mcp.config import load_config
        cfg = load_config()
        if not cfg:
            return out
        for sname, data in cfg.items():
            if not data.get("enabled", True):
                continue
            # Try non-blocking discovery
            try:
                from laap.mcp.client import MCPClient
                from laap.mcp.transports import StdioTransport
                command = data.get("command", "")
                args = data.get("args", [])
                env = data.get("env", {})
                if not command:
                    continue
                transport = asyncio.run(StdioTransport.spawn(command, args, env))
                client = MCPClient(transport)
                tools = asyncio.run(client.list_tools())
                for tool in tools:
                    tname = tool.get("name") or tool.get("function", {}).get("name") or "unknown"
                    tdesc = tool.get("description") or tool.get("function", {}).get("description") or ""
                    out.append((sname, tname, tdesc))
            except Exception:
                out.append((sname, sname, f"MCP server ({data.get('transport', 'stdio')})"))
    except Exception as e:
        logger.debug("[WARN] MCP tool discovery failed: %s", e)
    return out


def _load_skills() -> Dict[str, List[str]]:
    """Return skills grouped by category."""
    by_cat: Dict[str, List[str]] = {}
    from laap.config.paths import get_laap_home

    skills_dir = get_laap_home() / "skills"
    if skills_dir.is_dir():
        for entry in skills_dir.iterdir():
            if entry.is_dir():
                by_cat.setdefault(entry.name, []).append(entry.name)
            elif entry.suffix.lower() in (".md", ".py", ".json"):
                by_cat.setdefault("general", []).append(entry.stem)
    # Fallback to package skills
    if not by_cat:
        pkg_skills = Path(__file__).resolve().parent.parent / "skills"
        if pkg_skills.is_dir():
            for entry in pkg_skills.iterdir():
                if entry.is_dir() and not entry.name.startswith("_"):
                    by_cat.setdefault(entry.name, []).append(entry.name)
    if not by_cat:
        for cat in _BUILT_IN_SKILL_CATEGORIES:
            by_cat[cat] = [f"{cat}_skill"]
    return by_cat


def _load_active_profile() -> str:
    try:
        from laap.cli.profiles import profile_manager
        active = profile_manager.get_active()
        return active.get("name", "default") if active else "default"
    except Exception:
        return "default"


def _git_status_warnings() -> List[str]:
    warnings: List[str] = []
    git_dir = Path(".git")
    if git_dir.is_dir():
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-list", "HEAD..@{u}", "--count"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                count = result.stdout.strip()
                if count and count != "0":
                    warnings.append(f"git commit behind ({count})")
        except Exception:
            pass
    return warnings


def _api_key_warning() -> List[str]:
    warnings: List[str] = []
    try:
        from laap.cli.config_manager import config_manager
        if not config_manager.is_configured():
            warnings.append("no api key configured")
    except Exception:
        warnings.append("no api key configured")
    return warnings


def _make_light_bar_text(width: int = 120) -> Text:
    """Render a horizontal golden light-bar separator."""
    # Use a gradient-like sequence of block characters in gold tones
    chars = "▂▄▆█▆▄▂"
    repeat = max(1, width // len(chars))
    bar = (chars * repeat)[:width]
    return Text(bar, style=Style(color=Colors.GOLD, bgcolor=Colors.BG_PURE))


# ── Textual widgets/screens ──────────────────────────────────────
if HAS_TEXTUAL:

    class StatusBar(Static):
        """Bottom status bar with model/provider/metrics/context fill."""
        model = reactive("")
        provider = reactive("")
        status = reactive("idle")
        generation_mode = reactive("idle")
        encoding = reactive("UTF-8")
        tokens = reactive(0)
        elapsed = reactive(0.0)
        cpu_usage = reactive(0)
        memory_usage = reactive(0)
        context_fill = reactive(0.0)

        def render(self):
            r = Text()
            r.append(f"  {self.model or '?'} ", style=Style(color=Colors.GOLD_DARK))
            r.append(f"| {self.provider or '?'} ", style=Style(color=Colors.INFO))

            mode_display = self.generation_mode or self.status or "idle"
            st_map = {
                "idle": "Idle",
                "thinking": "Thinking...",
                "working": "Working...",
                "coding": "Coding...",
            }
            st = st_map.get(mode_display, mode_display)
            c = Colors.SUCCESS if mode_display == "idle" else Colors.GOLD
            if mode_display.startswith("tool:"):
                st = f"Tool:{mode_display.split(':', 1)[1]}"
                c = Colors.WARNING
            r.append(st, style=Style(color=c, bold=True))

            r.append(f" | CPU: {self.cpu_usage}%", style=Style(color=Colors.TEXT_DIM))
            r.append(f" | MEM: {self.memory_usage}%", style=Style(color=Colors.TEXT_DIM))
            r.append(f" | Tokens: {self.tokens}", style=Style(color=Colors.GOLD_DARK))
            if self.elapsed:
                r.append(f" | {self.elapsed:.1f}s", style=Style(color=Colors.GOLD_DARK))

            # Context fill indicator (green -> yellow -> orange -> red)
            fill = self.context_fill
            if fill < 0.5:
                fill_color = Colors.SUCCESS
            elif fill < 0.75:
                fill_color = Colors.WARNING
            elif fill < 0.9:
                fill_color = Colors.ORANGE
            else:
                fill_color = Colors.CRITICAL
            r.append(f" | CTX: {fill * 100:.0f}%", style=Style(color=fill_color))
            r.append(f" | {self.encoding}", style=Style(color=Colors.TEXT_DIM))
            return r

    class ToolStatusPanel(Static):
        """Left sidebar: real tools grouped by category."""

        def __init__(self, id=None):
            self.tools_by_category, self.tool_names = _load_tool_registry()
            super().__init__(self._build_content(), id=id)

        def _build_content(self) -> Text:
            content = Text()
            content.append("Tools\n", style=Style(color=Colors.GOLD, bold=True))
            if not self.tools_by_category:
                content.append("  No tools registered", style=Style(color=Colors.TEXT_DIM))
                return content
            for cat in sorted(self.tools_by_category.keys()):
                tools = self.tools_by_category[cat]
                line = Text(f"  [{cat}] ")
                line.append(f"{len(tools)}", style=Style(color=Colors.TEXT_DIM))
                line.append(" ")
                line.append("●", style=Style(color=Colors.SUCCESS))
                content.append(line)
                content.append("\n")
            return content

        def refresh_tools(self):
            self.tools_by_category, self.tool_names = _load_tool_registry()
            self.update(self._build_content())

    class SystemMetricsPanel(Static):
        """Left sidebar: CPU / memory bars."""

        def __init__(self, id=None):
            self._update_interval = 2
            self._running = True
            super().__init__(self._build_content(0.0, 0.0), id=id)

        def on_mount(self):
            self.set_interval(self._update_interval, self._tick)

        def _tick(self):
            try:
                if psutil is None:
                    return
                cpu = psutil.cpu_percent(interval=0.1)
                mem = psutil.virtual_memory().percent
                self.update_metrics(cpu, mem)
            except Exception as e:
                logger.debug("[WARN] Metrics update failed: %s", e)

        def update_metrics(self, cpu, mem):
            self.update(self._build_content(cpu, mem))

        def _make_bar(self, percent):
            filled = int(percent / 5)
            empty = 20 - filled
            return f"[{'█' * filled}{'░' * empty}]"

        def _build_content(self, cpu, mem) -> Text:
            content = Text()
            content.append("System\n", style=Style(color=Colors.GOLD, bold=True))
            content.append("CPU:\n", style=Style(color=Colors.TEXT_DIM))
            content.append(f"{cpu}%  {self._make_bar(cpu)}\n", style=Style(color=Colors.INFO))
            content.append("Memory:\n", style=Style(color=Colors.TEXT_DIM))
            content.append(f"{mem}%  {self._make_bar(mem)}", style=Style(color=Colors.WARNING))
            return content

    class CognitiveStatePanel(Static):
        """Left sidebar: PSI / Aether cognitive state."""

        def __init__(self, id=None):
            self.dominant_feeling = "neutral"
            self.arousal = 0.5
            self.valence = 0.0
            self.actor_count = 0
            super().__init__(self._build_content(), id=id)

        def on_mount(self):
            self.set_interval(3, self._poll)

        def _poll(self):
            try:
                state = _fetch_psi_state()
                self.dominant_feeling = state.get("dominant_feeling", "neutral")
                self.arousal = state.get("arousal", 0.5)
                self.valence = state.get("valence", 0.0)
                self.actor_count = state.get("actor_count", 0)
                self.update(self._build_content())
            except Exception as e:
                logger.debug("[WARN] PSI poll failed: %s", e)

        def _build_content(self) -> Text:
            content = Text()
            content.append("Cognitive State\n", style=Style(color=Colors.GOLD, bold=True))
            content.append("Feeling: ", style=Style(color=Colors.TEXT_DIM))
            content.append(f"{self.dominant_feeling}\n", style=Style(color=Colors.INFO))
            content.append("Arousal: ", style=Style(color=Colors.TEXT_DIM))
            content.append(f"{self.arousal:.2f}\n", style=Style(color=Colors.INFO))
            content.append("Valence: ", style=Style(color=Colors.TEXT_DIM))
            content.append(f"{self.valence:.2f}\n", style=Style(color=Colors.INFO))
            content.append("Aether actors: ", style=Style(color=Colors.TEXT_DIM))
            content.append(str(self.actor_count), style=Style(color=Colors.WARNING))
            return content

    class VideoPlayerPanel(Static):
        """Left sidebar: video playback launcher for the LAAP interactive video."""

        VIDEO_PATH = get_video_dir() / "互动视频.mp4"

        def __init__(self, id=None):
            video_path = str(self.VIDEO_PATH)
            self._exists = os.path.exists(video_path)
            self._video_size = "0 B"
            if self._exists:
                try:
                    self._video_size = self._format_size(os.path.getsize(video_path))
                except OSError as e:
                    logger.warning("VideoPlayerPanel: cannot read size of %s: %s", video_path, e)
            else:
                logger.warning("VideoPlayerPanel: interactive video not found at %s", video_path)
            super().__init__(self._build_content(), id=id)
            self.styles.cursor = "pointer"

        def _format_size(self, size: int) -> str:
            for unit in ("B", "KB", "MB", "GB"):
                if size < 1024:
                    return f"{size:.1f} {unit}"
                size /= 1024
            return f"{size:.1f} TB"

        def _build_content(self) -> Text:
            video_path = str(self.VIDEO_PATH)
            content = Text()
            content.append("Interactive Video\n", style=Style(color=Colors.GOLD, bold=True))
            if self._exists:
                content.append("▶ ", style=Style(color=Colors.SUCCESS, bold=True))
                content.append("Click to play\n", style=Style(color=Colors.TEXT_DIM))
                content.append("File: ", style=Style(color=Colors.TEXT_DIM))
                content.append("互动视频.mp4\n", style=Style(color=Colors.INFO))
                content.append("Size: ", style=Style(color=Colors.TEXT_DIM))
                content.append(self._video_size, style=Style(color=Colors.WARNING))
            else:
                content.append("  Video not found\n", style=Style(color=Colors.CRITICAL))
                content.append(video_path, style=Style(color=Colors.TEXT_DIM))
            return content

        def on_click(self):
            video_path = str(self.VIDEO_PATH)
            if not self._exists:
                return
            try:
                if sys.platform == "win32":
                    os.startfile(video_path)
                else:
                    import subprocess
                    subprocess.call(["xdg-open", video_path])
            except Exception as e:
                logger.warning("VideoPlayerPanel: video open failed: %s", e)

    class VitalSignsPanel(Static):
        """Left sidebar: digital lifeform vital signs (PSI, curiosity, trust, competence)."""

        def __init__(self, id=None):
            self.psi_loop = "idle"
            self.curiosity = 0.62
            self.trust = 0.74
            self.competence = 0.81
            self.energy = 0.88
            self.coherence = 0.79
            super().__init__(self._build_content(), id=id)

        def on_mount(self):
            self.set_interval(2, self._poll)

        def _poll(self):
            try:
                state = _fetch_psi_state()
                # Derive vital signs from PSI / agent state
                self.psi_loop = state.get("psi_loop", "idle")
                self.curiosity = state.get("curiosity", self._drift(self.curiosity))
                self.trust = state.get("trust", self._drift(self.trust))
                self.competence = state.get("competence", self._drift(self.competence))
                self.energy = state.get("energy", self._drift(self.energy))
                self.coherence = state.get("coherence", self._drift(self.coherence))
                self.update(self._build_content())
            except Exception as e:
                logger.debug("[WARN] Vital signs poll failed: %s", e)

        def _drift(self, value: float) -> float:
            import random
            delta = (random.random() - 0.5) * 0.04
            return max(0.0, min(1.0, value + delta))

        def _bar(self, value: float, color: str) -> Text:
            filled = int(value * 10)
            empty = 10 - filled
            t = Text()
            t.append(f"[{'█' * filled}{'░' * empty}] ", style=Style(color=color))
            t.append(f"{value:.2f}", style=Style(color=color, bold=True))
            return t

        def _build_content(self) -> Text:
            content = Text()
            content.append("Lifeform Vitals\n", style=Style(color=Colors.GOLD, bold=True))
            content.append("PSI loop: ", style=Style(color=Colors.TEXT_DIM))
            content.append(f"{self.psi_loop}\n", style=Style(color=Colors.INFO, bold=True))
            content.append("Curiosity  ", style=Style(color=Colors.TEXT_DIM))
            content.append(self._bar(self.curiosity, Colors.INFO))
            content.append("\nTrust      ", style=Style(color=Colors.TEXT_DIM))
            content.append(self._bar(self.trust, Colors.SUCCESS))
            content.append("\nCompetence ", style=Style(color=Colors.TEXT_DIM))
            content.append(self._bar(self.competence, Colors.WARNING))
            content.append("\nEnergy     ", style=Style(color=Colors.TEXT_DIM))
            content.append(self._bar(self.energy, Colors.GOLD))
            content.append("\nCoherence  ", style=Style(color=Colors.TEXT_DIM))
            content.append(self._bar(self.coherence, Colors.INFO))
            return content

    class GatewayStatusPanel(Static):
        """Left sidebar: Telegram / Feishu gateway status."""

        def __init__(self, id=None):
            self.telegram_status = "disabled"
            self.feishu_status = "disabled"
            super().__init__(self._build_content(), id=id)

        def on_mount(self):
            self.set_interval(5, self._poll)

        def _poll(self):
            try:
                telegram_ok = bool(os.environ.get("TELEGRAM_BOT_TOKEN", ""))
                feishu_ok = bool(os.environ.get("FEISHU_APP_ID", "") or os.environ.get("FEISHU_WEBHOOK_URL", ""))
                self.telegram_status = "configured" if telegram_ok else "disabled"
                self.feishu_status = "configured" if feishu_ok else "disabled"
                self.update(self._build_content())
            except Exception as e:
                logger.debug("[WARN] Gateway poll failed: %s", e)

        def _build_content(self) -> Text:
            content = Text()
            content.append("Gateways\n", style=Style(color=Colors.GOLD, bold=True))
            content.append("Telegram: ", style=Style(color=Colors.TEXT_DIM))
            tg_color = Colors.SUCCESS if self.telegram_status == "configured" else Colors.TEXT_DIM
            content.append(f"{self.telegram_status}\n", style=Style(color=tg_color))
            content.append("Feishu: ", style=Style(color=Colors.TEXT_DIM))
            fs_color = Colors.SUCCESS if self.feishu_status == "configured" else Colors.TEXT_DIM
            content.append(self.feishu_status, style=Style(color=fs_color))
            return content

    class ChatLog(ScrollableContainer):
        """Chat log with user/assistant/system/tool messages."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._typing_task = None
            self._last_assistant_text = ""
            self._messages: List[Tuple[str, str, float]] = []

        def compose(self):
            yield Static("", id="msgs")

        def add_welcome(self, banner):
            try:
                old = self.query_one("#welcome", Static)
                old.remove()
            except Exception:
                pass
            m = Static(banner, id="welcome")
            m.styles.margin = (0, 1, 1, 1)
            self.mount(m)
            self.scroll_end()

        def remove_welcome(self):
            try:
                w = self.query_one("#welcome", Static)
                w.remove()
            except Exception:
                pass

        def clear_all(self):
            try:
                for w in list(self.query("Static")):
                    if w.id != "msgs":
                        w.remove()
                self._messages.clear()
                self._last_assistant_text = ""
            except Exception as e:
                logger.debug("[WARN] clear_all failed: %s", e)

        def _ts(self) -> str:
            return datetime.now().strftime("%H:%M")

        def add_user(self, text: str):
            r = md_inline(text)
            ts = self._ts()
            self._messages.append(("user", text, time.time()))
            m = Static(r, classes="user_msg")
            m.styles.margin = (1, 2, 0, 8)
            m.styles.padding = (0, 2, 0, 2)
            m.styles.background = Colors.BG_PANEL
            m.styles.border = ("solid", Colors.GOLD)
            m.styles.border_left = ("none", Colors.GOLD)
            m.styles.text_align = "right"
            m.tooltip = f"{ts}  User"
            self.mount(m)
            self.scroll_end()

        def stream_assistant(self, text: str, append: bool = False):
            if append:
                self._last_assistant_text += text
            else:
                self._last_assistant_text = text
            display = self._last_assistant_text
            try:
                resp = self.query_one("#response", Static)
                resp.update(md_inline(display))
                resp.tooltip = f"{self._ts()}  Assistant  (press 'c' to copy)"
                self.scroll_end()
                return
            except Exception:
                pass
            try:
                m = Static(md_inline(display), id="response", classes="assistant_msg")
                m.styles.margin = (1, 8, 0, 2)
                m.styles.padding = (0, 2, 0, 2)
                m.styles.background = Colors.BG_MEDIUM
                m.styles.border = ("solid", Colors.BRONZE_DARK)
                m.styles.border_right = ("none", Colors.BRONZE_DARK)
                m.tooltip = f"{self._ts()}  Assistant  (press 'c' to copy)"
                self.mount(m)
                self.scroll_end()
            except Exception as e:
                logger.debug("[WARN] stream_assistant failed: %s", e)

        def add_assistant(self, text: str):
            self._last_assistant_text = text
            self._messages.append(("assistant", text, time.time()))
            r = md_inline(text)
            m = Static(r, classes="assistant_msg")
            m.styles.margin = (1, 8, 0, 2)
            m.styles.padding = (0, 2, 0, 2)
            m.styles.background = Colors.BG_MEDIUM
            m.styles.border = ("solid", Colors.BRONZE_DARK)
            m.styles.border_right = ("none", Colors.BRONZE_DARK)
            m.tooltip = f"{self._ts()}  Assistant  (press 'c' to copy)"
            self.mount(m)
            self.scroll_end()

        def add_tool(self, name: str, ok: bool = True):
            marker = "[OK]" if ok else "[ERROR]"
            color = Colors.SUCCESS if ok else Colors.CRITICAL
            r = Text(f"  {marker} {name}", style=Style(color=color))
            m = Static(r)
            m.styles.margin = (0, 2, 0, 4)
            self.mount(m)
            self.scroll_end()

        def add_tool_progress(self, tool_name: str, args: Dict[str, Any]):
            """Show a Rich panel/spinner for an in-flight tool call."""
            summary_items = list(args.items())[:3]
            summary = " ".join(f"{k}={v!r}" for k, v in summary_items)
            if len(args) > 3:
                summary += " ..."
            call_id = f"tool_{uuid.uuid4().hex[:8]}"
            try:
                spinner = Spinner("dots", text=f" {tool_name} {summary}")
                panel = Panel(spinner, title=f"[bold {Colors.GOLD}]{tool_name}[/]", border_style=Colors.BRONZE)
                m = Static(panel, classes="tool_progress", id=call_id)
                m.styles.margin = (0, 2, 0, 4)
                self.mount(m)
                self.scroll_end()
                return m
            except Exception as e:
                logger.debug("[WARN] tool progress render failed: %s", e)
                return None

        def update_tool_progress(self, widget, ok: bool, result_text: str = "", duration: float = 0.0):
            if widget is None:
                return
            try:
                status = "[OK]" if ok else "[ERROR]"
                color = Colors.SUCCESS if ok else Colors.CRITICAL
                title = ""
                try:
                    title = widget.content.title
                except Exception:
                    title = "Tool"
                dur = f" ({duration:.2f}s)" if duration > 0 else ""
                panel = Panel(
                    Text(f"{status} {result_text}{dur}"),
                    title=f"[bold {color}]{title}[/]",
                    border_style=color,
                )
                widget.update(panel)
            except Exception as e:
                logger.debug("[WARN] tool progress update failed: %s", e)

        def add_coding_progress(self, file_path: str = "", lines_changed: Optional[int] = None):
            """Show a glowing 'Coding...' progress panel."""
            try:
                progress = Progress(
                    TextColumn("[bold #FFD700]Coding...[/]"),
                    BarColumn(bar_width=30, complete_style=Colors.GOLD, finished_style=Colors.SUCCESS),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    transient=False,
                )
                task = progress.add_task("coding", total=100)
                progress.update(task, completed=30)
                desc = f"  Editing {file_path}" if file_path else "  Writing code..."
                if lines_changed is not None:
                    desc += f" ({lines_changed} lines changed)"
                panel = Panel(
                    Group(progress, Text(desc, style=Style(color=Colors.TEXT_DIM))),
                    title=f"[bold {Colors.GOLD}]Coding[/]",
                    border_style=Colors.GOLD,
                )
                call_id = f"coding_{uuid.uuid4().hex[:8]}"
                m = Static(panel, classes="coding_progress", id=call_id)
                m.styles.margin = (0, 2, 0, 4)
                self.mount(m)
                self.scroll_end()
                return m
            except Exception as e:
                logger.debug("[WARN] coding progress render failed: %s", e)
                return None

        def update_coding_progress(self, widget, completed: int = 100, file_path: str = "", lines_changed: Optional[int] = None):
            if widget is None:
                return
            try:
                title = ""
                try:
                    title = widget.content.title
                except Exception:
                    title = "Coding"
                status = "[OK]" if completed >= 100 else f"{completed}%"
                color = Colors.SUCCESS if completed >= 100 else Colors.GOLD
                desc = f"  {file_path}" if file_path else "  Writing code..."
                if lines_changed is not None:
                    desc += f" ({lines_changed} lines changed)"
                panel = Panel(
                    Text(f"{status} {desc}"),
                    title=f"[bold {color}]{title}[/]",
                    border_style=color,
                )
                widget.update(panel)
            except Exception as e:
                logger.debug("[WARN] coding progress update failed: %s", e)

        def add_sys(self, text: str):
            r = Text(f"  {text}", style=Style(color=Colors.GOLD_DARK, italic=True))
            m = Static(r, classes="sys_msg")
            m.styles.margin = (0, 2, 0, 2)
            m.styles.italic = True
            self.mount(m)
            self.scroll_end()

        def add_thinking(self):
            self._stop_typing_animation()
            r = Text("  Ao is thinking...", style=Style(color=Colors.GOLD_DARK, italic=True))
            m = Static(r, id="thinking")
            m.styles.margin = (0, 2, 0, 2)
            self.mount(m)
            self.scroll_end()
            self._typing_task = self.app.set_interval(0.4, self._animate_thinking)

        def _animate_thinking(self):
            try:
                w = self.query_one("#thinking", Static)
                frames = ["Ao is thinking.", "Ao is thinking..", "Ao is thinking..."]
                current = w.renderable.plain.strip() if hasattr(w.renderable, "plain") else ""
                idx = 0
                for i, f in enumerate(frames):
                    if current.startswith(f.rstrip(".")) and i == (len(frames) - 1 if "." * (current.count(".") + 1) > "..." else current.count(".") - 1):
                        idx = (i + 1) % len(frames)
                        break
                    if current == f:
                        idx = (i + 1) % len(frames)
                        break
                # Simpler rotation: count trailing dots
                dots = current.count(".") if current else 0
                idx = (dots) % 3
                frame = f"  {frames[idx]}"
                w.update(Text(frame, style=Style(color=Colors.GOLD_DARK, italic=True)))
            except Exception:
                pass

        def _stop_typing_animation(self):
            if self._typing_task:
                try:
                    self._typing_task.stop()
                except Exception:
                    pass
                self._typing_task = None

        def start_response(self):
            self._last_assistant_text = ""
            try:
                try:
                    old = self.query_one("#response")
                    old.remove()
                except Exception:
                    pass
                m = Static("", id="response", classes="assistant_msg")
                m.styles.margin = (1, 8, 0, 2)
                m.styles.padding = (0, 2, 0, 2)
                m.styles.background = Colors.BG_MEDIUM
                m.styles.border = ("solid", Colors.BRONZE_DARK)
                m.styles.border_right = ("none", Colors.BRONZE_DARK)
                self.mount(m)
                self.scroll_end()
                return True
            except Exception:
                return False

        def remove_thinking(self):
            self._stop_typing_animation()
            try:
                w = self.query_one("#thinking")
                w.remove()
            except Exception as e:
                logger.debug("[WARN] remove_thinking failed: %s", e)

        def get_last_assistant_text(self) -> str:
            return self._last_assistant_text

        def finalize_assistant_response(self, text: str):
            """Replace the streaming response widget with a persisted assistant message."""
            self._last_assistant_text = text
            self._messages.append(("assistant", text, time.time()))
            try:
                resp = self.query_one("#response", Static)
                resp.remove()
            except Exception:
                pass
            self.add_assistant(text)

        def pop_last_assistant_turn(self) -> bool:
            """Remove the last assistant message from the chat log and internal history."""
            try:
                for i in range(len(self._messages) - 1, -1, -1):
                    if self._messages[i][0] == "assistant":
                        self._messages.pop(i)
                        # Remove rendered assistant widgets (best-effort)
                        for w in list(self.query(".assistant_msg")):
                            w.remove()
                        return True
                return False
            except Exception as e:
                logger.debug("[WARN] pop_last_assistant_turn failed: %s", e)
                return False

    class SlashCommandOverlay(Vertical):
        """Overlay above the input showing matching slash commands."""

        def __init__(self, commands: List[Dict[str, str]], id=None):
            super().__init__(id=id)
            self.commands = commands
            self._selected = 0
            self._items: List[Static] = []
            self._matches: List[Dict[str, Any]] = []

        def compose(self):
            yield Static("Commands", id="overlay_title")

        def update_matches(self, partial: str):
            for w in list(self.query(".cmd_item")):
                w.remove()
            self._items.clear()
            p = partial.lower().lstrip("/")
            matches = [
                c for c in self.commands
                if c["name"].startswith(p)
                or any(a.startswith(p) for a in c.get("aliases", []))
            ]
            self._matches = matches[:8]
            for cmd in self._matches:
                aliases = f" ({', '.join(cmd.get('aliases', []))})" if cmd.get("aliases") else ""
                line = Text(f"  /{cmd['name']}{aliases}  — ")
                line.append(cmd.get("desc", ""), style=Style(color=Colors.TEXT_DIM))
                m = Static(line, classes="cmd_item")
                self.mount(m)
                self._items.append(m)
            self._selected = 0
            self._highlight()

        def _highlight(self):
            for i, item in enumerate(self._items):
                item.styles.background = Colors.BG_HIGHLIGHT if i == self._selected else Colors.BG_PANEL

        def move_selection(self, delta: int):
            if not self._items:
                return
            self._selected = (self._selected + delta) % len(self._items)
            self._highlight()

        def selected_command(self) -> Optional[str]:
            if not self._items or self._selected >= len(self._matches):
                return None
            return self._matches[self._selected]["name"]

    class ToolPickerOverlay(Vertical):
        """Overlay above the input showing available tools for selection."""

        def __init__(self, tools: List[Tuple[str, str, str]], id=None):
            super().__init__(id=id)
            self.all_tools = tools
            self._selected = 0
            self._items: List[Static] = []
            self._matches: List[Tuple[str, str, str]] = []

        def compose(self):
            yield Static("Available Tools (Tab to insert, Esc to close)", id="tool_overlay_title")

        def update_matches(self, partial: str):
            for w in list(self.query(".tool_item")):
                w.remove()
            self._items.clear()
            p = partial.lower().strip()
            matches = [
                t for t in self.all_tools
                if p in t[0].lower() or p in (t[1] or "").lower() or p in (t[2] or "").lower()
            ]
            self._matches = matches[:12]
            for name, cat, desc in self._matches:
                line = Text(f"  {name}", style=Style(color=Colors.GOLD))
                line.append(f"  [{cat}] ", style=Style(color=Colors.TEXT_DIM))
                if desc:
                    line.append(desc[:50], style=Style(color=Colors.TEXT_DIM, italic=True))
                m = Static(line, classes="tool_item")
                self.mount(m)
                self._items.append(m)
            self._selected = 0
            self._highlight()

        def _highlight(self):
            for i, item in enumerate(self._items):
                item.styles.background = Colors.BG_HIGHLIGHT if i == self._selected else Colors.BG_PANEL

        def move_selection(self, delta: int):
            if not self._items:
                return
            self._selected = (self._selected + delta) % len(self._items)
            self._highlight()

        def selected_tool(self) -> Optional[str]:
            if not self._items or self._selected >= len(self._matches):
                return None
            return self._matches[self._selected][0]

    class ApprovalModal(ModalScreen):
        """Modal screen for approving dangerous commands."""

        BINDINGS = [
            Binding("escape", "dismiss_modal", "Deny"),
        ]

        def __init__(self, command: str, pattern: str, severity: str = "high"):
            super().__init__()
            self.command = command
            self.pattern = pattern
            self.severity = severity
            self._result = "deny"

        def compose(self):
            with Vertical(id="approval_dialog"):
                yield Static("[WARNING] Dangerous command detected", id="approval_title")
                yield Static(f"Command: {self.command}", id="approval_command")
                yield Static(f"Pattern matched: {self.pattern} [{self.severity}]", id="approval_pattern")
                with Horizontal(id="approval_buttons"):
                    yield Button("Allow once", id="btn_once", variant="primary")
                    yield Button("Allow session", id="btn_session")
                    yield Button("Allow always", id="btn_always")
                    yield Button("Deny", id="btn_deny", variant="error")

        def on_button_pressed(self, event: Button.Pressed):
            mapping = {
                "btn_once": "once",
                "btn_session": "session",
                "btn_always": "always",
                "btn_deny": "deny",
            }
            self._result = mapping.get(event.button.id, "deny")
            self.dismiss(self._result)

        def action_dismiss_modal(self):
            self._result = "deny"
            self.dismiss("deny")

    class MainScreen(Screen):
        """Main application screen with Hermes-inspired layout."""

        BINDINGS = [
            Binding("ctrl+c", "quit", "Quit"),
            Binding("ctrl+l", "clear", "Clear"),
            Binding("ctrl+w", "toggle_sidebar", "Toggle Sidebar"),
            Binding("ctrl+enter", "submit", "Submit"),
            Binding("shift+enter", "newline", "Newline"),
            Binding("tab", "autocomplete", "Autocomplete"),
            Binding("up", "select_up", "Up"),
            Binding("down", "select_down", "Down"),
            Binding("escape", "close_overlays_or_cancel", "Close Overlays / Cancel"),
            Binding("c", "copy_last", "Copy last response"),
        ]

        def __init__(self, **kw):
            super().__init__(**kw)
            self._buf = ""
            self._buf_lock = threading.Lock()
            self._history: List[str] = []
            self._history_index = 0
            self._sidebar_visible = True
            self._slash_overlay_visible = False
            self._tool_overlay_visible = False
            self._commands = _build_slash_commands()
            self._tool_progress_widgets: Dict[str, Any] = {}
            self._coding_progress_widgets: Dict[str, Any] = {}
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._profile_name = _load_active_profile()
            self._yolo = False
            self._welcome_visible = True
            self._cancelled = False
            self._debug = os.environ.get("LAAP_DEBUG", "").lower() in ("1", "true", "yes")
            self._attachments: List[str] = []
            self._last_response_text = ""

        def compose(self) -> ComposeResult:
            with Horizontal():
                with Vertical(id="sidebar", classes="sidebar"):
                    yield ToolStatusPanel(id="tool_panel")
                    yield SystemMetricsPanel(id="metrics_panel")
                    yield CognitiveStatePanel(id="cognitive_panel")
                    yield VitalSignsPanel(id="vitals_panel")
                    yield VideoPlayerPanel(id="video_panel")
                    yield GatewayStatusPanel(id="gateway_panel")

                with Vertical(id="main_content"):
                    yield ChatLog(id="chat_log")
                    with Vertical(id="input_container"):
                        yield SlashCommandOverlay(self._commands, id="slash_overlay")
                        yield ToolPickerOverlay(self._all_tool_entries(), id="tool_overlay")
                        yield Static(_make_light_bar_text(), id="light_bar")
                        yield Input(
                            placeholder="Type your message or /help for commands...",
                            id="input_box",
                        )
                    yield StatusBar(id="status_bar")

        def _all_tool_entries(self) -> List[Tuple[str, str, str]]:
            by_cat, names = _load_tool_registry()
            out = []
            for cat, items in by_cat.items():
                for meta in items:
                    out.append((meta.get("name", ""), cat, meta.get("description", "")))
            return out

        def on_mount(self):
            chat_log = self.query_one("#chat_log", ChatLog)
            banner = self._build_welcome_banner()
            chat_log.add_welcome(banner)

            sb = self.query_one("#status_bar", StatusBar)
            sb.model = self._safe_model_name()
            sb.provider = self._safe_provider_name()

            self.query_one("#slash_overlay", SlashCommandOverlay).styles.display = "none"
            self.query_one("#tool_overlay", ToolPickerOverlay).styles.display = "none"
            self.query_one("#input_box", Input).focus()

            # Wire approval callback
            try:
                from laap.agent_core.tools.approval_tool import get_approver
                approver = get_approver()
                approver.set_prompt_callback(self._approval_callback)
            except Exception as e:
                logger.debug("[WARN] Approval callback registration failed: %s", e)

            # First-launch setup prompt
            self._maybe_show_setup_welcome(chat_log)

            # Loading shimmer cleanup
            self.set_interval(1.2, self._remove_loading_shimmer)

        def _maybe_show_setup_welcome(self, chat_log: ChatLog):
            try:
                from laap.cli.config_manager import config_manager
                if not config_manager.is_configured():
                    chat_log.add_sys(
                        "Welcome! No LLM configured yet. Type /config wizard to set up your first model."
                    )
            except Exception as e:
                logger.debug("[WARN] Setup welcome check failed: %s", e)

        def _remove_loading_shimmer(self):
            try:
                w = self.query_one("#loading_shimmer", Static)
                w.remove()
            except Exception:
                pass

        def _safe_model_name(self) -> str:
            try:
                from laap.cli.config_manager import config_manager
                model = config_manager.get_summary().get("model")
                if model:
                    return model
            except Exception:
                pass
            try:
                agent = get_or_create_agent()
                if agent and agent.llm:
                    return agent.llm.model
            except Exception:
                pass
            return os.environ.get("LAAP_MODEL", "deepseek-v4-flash")

        def _safe_provider_name(self) -> str:
            try:
                from laap.cli.config_manager import config_manager
                provider = config_manager.get_summary().get("provider")
                if provider:
                    return provider
            except Exception:
                pass
            try:
                agent = get_or_create_agent()
                if agent and agent.llm:
                    return agent.llm.provider_name or type(agent.llm).__name__.replace("Provider", "")
            except Exception:
                pass
            return os.environ.get("LAAP_PROVIDER", "deepseek")

        def _build_welcome_banner(self):
            try:
                from laap.ui.hermes_banner import build_banner
                from laap import __version__
                tools_by_cat, tool_names = _load_tool_registry()
                skills = _load_skills()
                total_skills = sum(len(v) for v in skills.values())
                mcp_servers = _load_mcp_servers()
                mcp_count = sum(1 for _n, _t, enabled in mcp_servers if enabled)
                warnings = _git_status_warnings() + _api_key_warning()

                tool_entries = [(m.get("name", ""), m.get("category", ""), m.get("description", ""))
                                for items in tools_by_cat.values() for m in items]
                skill_entries = [(n, c, "") for c, names in skills.items() for n in names]

                banner = build_banner(
                    model=self._safe_model_name(),
                    provider=self._safe_provider_name(),
                    cwd=os.getcwd(),
                    session_id=self._session_id,
                    tools=tool_entries,
                    skills=skill_entries,
                    show_tip=False,
                    show_warnings=False,
                )
                # Build a custom footer with version, stats, warnings, tips
                footer = Text()
                footer.append(Text.from_markup(
                    rf"\n[bold {Colors.GOLD}]<\|_/\_/\_/\_/\|>  Welcome to LAAP Agent![/] "
                    f"[dim {Colors.GOLD_DARK}]Type your message or /help for commands.[/]\n"
                ))
                footer.append(Text.from_markup(
                    f"[dim {Colors.BRONZE}]Tip: The status bar turns yellow, then orange, then red as context fills up.[/]\n"
                ))
                activated = ", ".join(sorted(skills.keys())[:5]) or "none"
                footer.append(Text.from_markup(
                    f"[dim {Colors.GOLD_DARK}]Profile: {self._profile_name} · Activated skills: {activated}[/]\n"
                ))
                footer.append(Text.from_markup(
                    f"[dim {Colors.GOLD_DARK}]{len(tool_names)} tools · {mcp_count} MCP servers · {total_skills} skills · /help for commands[/]"
                ))
                if warnings:
                    footer.append(Text.from_markup("\n"))
                for w in warnings:
                    footer.append(Text.from_markup(f"[bold yellow][WARN] {w}[/]\n"))
                shimmer = Text("Loading...", style=Style(color=Colors.GOLD_DARK, italic=True))
                return Group(banner, footer, shimmer)
            except Exception as e:
                logger.debug("[WARN] Welcome banner build failed: %s", e)
                return Text("Welcome to LAAP Agent!")

        def _approval_callback(self, command: str, pattern: str, severity: str) -> str:
            """Synchronous callback used by the approval system.

            Returns one of: once, session, always, deny.
            """
            future: asyncio.Future = asyncio.run_coroutine_threadsafe(
                self._show_approval_modal(command, pattern, severity),
                loop=self.app._loop,
            )
            try:
                return future.result(timeout=120)
            except Exception as e:
                logger.error("[ERROR] Approval modal failed: %s", e)
                return "deny"

        async def _show_approval_modal(self, command: str, pattern: str, severity: str) -> str:
            future: asyncio.Future = asyncio.get_running_loop().create_future()

            def _on_result(result: str) -> None:
                if not future.done():
                    future.set_result(result)

            await self.app.push_screen(
                ApprovalModal(command, pattern, severity), callback=_on_result
            )
            return await future

        def on_input_changed(self, event: Input.Changed):
            text = event.value
            slash_overlay = self.query_one("#slash_overlay", SlashCommandOverlay)
            tool_overlay = self.query_one("#tool_overlay", ToolPickerOverlay)
            if text.startswith("/"):
                slash_overlay.styles.display = "block"
                slash_overlay.update_matches(text)
                tool_overlay.styles.display = "none"
                self._slash_overlay_visible = True
                self._tool_overlay_visible = False
            elif text.startswith("@tool") or text.startswith("#tool"):
                slash_overlay.styles.display = "none"
                tool_overlay.styles.display = "block"
                tool_overlay.update_matches(text.lstrip("@tool").lstrip("#tool").strip())
                self._slash_overlay_visible = False
                self._tool_overlay_visible = True
            else:
                slash_overlay.styles.display = "none"
                tool_overlay.styles.display = "none"
                self._slash_overlay_visible = False
                self._tool_overlay_visible = False

        async def action_submit(self):
            inp = self.query_one("#input_box", Input)
            if inp.value:
                await inp.action_submit()

        def action_newline(self):
            inp = self.query_one("#input_box", Input)
            inp.insert_text_at_cursor("\n")

        def action_copy_last(self):
            chat_log = self.query_one("#chat_log", ChatLog)
            text = chat_log.get_last_assistant_text() or self._last_response_text
            if text:
                try:
                    import pyperclip
                    pyperclip.copy(text)
                    chat_log.add_sys("[OK] Copied last response to clipboard")
                except Exception:
                    chat_log.add_sys("[WARN] pyperclip not available; install it to use /copy")
            else:
                chat_log.add_sys("[WARN] No assistant response to copy")

        def on_input_submitted(self, event: Input.Submitted):
            text = event.value.strip()
            event.input.value = ""
            if not text:
                return

            chat_log = self.query_one("#chat_log", ChatLog)
            status_bar = self.query_one("#status_bar", StatusBar)
            slash_overlay = self.query_one("#slash_overlay", SlashCommandOverlay)
            tool_overlay = self.query_one("#tool_overlay", ToolPickerOverlay)
            slash_overlay.styles.display = "none"
            tool_overlay.styles.display = "none"
            self._slash_overlay_visible = False
            self._tool_overlay_visible = False

            # Hide welcome on first real interaction
            if self._welcome_visible:
                chat_log.remove_welcome()
                self._welcome_visible = False

            if text.startswith("/"):
                self._history.append(text)
                self._history_index = len(self._history)
                self._handle_command(text, chat_log, status_bar)
                return

            full_text = text
            if self._attachments:
                attached = "\n\n".join(self._attachments)
                full_text = f"{text}\n\n[Attached context]\n{attached}"
                self._attachments.clear()

            chat_log.add_user(text)
            status_bar.status = "thinking"
            self._buf = ""
            chat_log.add_thinking()
            self._history.append(text)
            self._history_index = len(self._history)
            self._cancelled = False
            self.run_worker(self._chat(full_text, chat_log, status_bar), name="chat_worker")

        def action_autocomplete(self):
            if self._slash_overlay_visible:
                overlay = self.query_one("#slash_overlay", SlashCommandOverlay)
                if overlay._items:
                    cmd = overlay.selected_command()
                    if cmd:
                        self.query_one("#input_box", Input).value = f"/{cmd} "
                        overlay.styles.display = "none"
                        self._slash_overlay_visible = False
            elif self._tool_overlay_visible:
                overlay = self.query_one("#tool_overlay", ToolPickerOverlay)
                if overlay._items:
                    tool = overlay.selected_tool()
                    if tool:
                        self.query_one("#input_box", Input).value = f"@tool {tool} "
                        overlay.styles.display = "none"
                        self._tool_overlay_visible = False

        def action_close_overlays_or_cancel(self):
            if self._slash_overlay_visible or self._tool_overlay_visible:
                try:
                    self.query_one("#slash_overlay", SlashCommandOverlay).styles.display = "none"
                    self.query_one("#tool_overlay", ToolPickerOverlay).styles.display = "none"
                    self._slash_overlay_visible = False
                    self._tool_overlay_visible = False
                except Exception:
                    pass
            else:
                self._cmd_cancel("", self.query_one("#chat_log", ChatLog), self.query_one("#status_bar", StatusBar))

        def action_select_up(self):
            if self._slash_overlay_visible:
                self.query_one("#slash_overlay", SlashCommandOverlay).move_selection(-1)
            elif self._tool_overlay_visible:
                self.query_one("#tool_overlay", ToolPickerOverlay).move_selection(-1)
            else:
                self._cycle_history(-1)

        def action_select_down(self):
            if self._slash_overlay_visible:
                self.query_one("#slash_overlay", SlashCommandOverlay).move_selection(1)
            elif self._tool_overlay_visible:
                self.query_one("#tool_overlay", ToolPickerOverlay).move_selection(1)
            else:
                self._cycle_history(1)

        def _cycle_history(self, delta: int):
            if not self._history:
                return
            self._history_index = max(0, min(len(self._history), self._history_index + delta))
            inp = self.query_one("#input_box", Input)
            if self._history_index == len(self._history):
                inp.value = ""
            else:
                inp.value = self._history[self._history_index]

        def _handle_command(self, cmd: str, chat_log: ChatLog, status_bar: StatusBar):
            parts = cmd.split()
            c = parts[0].lower() if parts else ""
            rest = " ".join(parts[1:]) if len(parts) > 1 else ""

            handler = self._commands_map().get(c.lstrip("/"))
            if handler:
                handler(rest, chat_log, status_bar)
            else:
                chat_log.add_sys(f"Unknown command: {c}")

        def _commands_map(self) -> Dict[str, Callable]:
            return {
                "help": self._cmd_help,
                "h": self._cmd_help,
                "?": self._cmd_help,
                "new": self._cmd_new,
                "clear": self._cmd_new,
                "config": self._cmd_config,
                "model": self._cmd_model,
                "models": self._cmd_model_list,
                "tools": self._cmd_tools,
                "sidebar": self._cmd_sidebar,
                "profile": self._cmd_profile,
                "sessions": self._cmd_sessions,
                "resume": self._cmd_resume,
                "branch": self._cmd_branch,
                "save": self._cmd_save,
                "exit": self._cmd_exit,
                "quit": self._cmd_exit,
                "psi": self._cmd_psi,
                "aether": self._cmd_aether,
                "gateways": self._cmd_gateways,
                "mcp": self._cmd_mcp,
                "yolo": self._cmd_yolo,
                "cancel": self._cmd_cancel,
                # Hermes-level additions
                "reset": self._cmd_reset,
                "undo": self._cmd_undo,
                "redo": self._cmd_redo,
                "copy": self._cmd_copy,
                "edit": self._cmd_edit,
                "file": self._cmd_file,
                "image": self._cmd_image,
                "url": self._cmd_url,
                "search": self._cmd_search,
                "approve": self._cmd_approve,
                "status": self._cmd_status,
                "logs": self._cmd_logs,
                "debug": self._cmd_debug,
                "compact": self._cmd_compact,
                "compress": self._cmd_compress,
                "tokens": self._cmd_tokens,
                "usage": self._cmd_usage,
            }

        def _cmd_help(self, rest, chat_log, status_bar):
            help_text = """**LAAP Commands**

`/new` or `/clear` - New session
`/reset` - Hard reset agent and conversation
`/config` - Show configuration
`/config set <key> <value>` - Set config key (api_key, model, provider, base_url)
`/config wizard` - Interactive setup
`/model` - Current model info
`/model list` - List available models
`/model use <model>` or `/model set <model>` - Switch model
`/model provider <provider>` - Switch provider
`/tools` - Open tool picker
`/sidebar` - Toggle sidebar
`/profile` - Show active profile
`/sessions` - List recent sessions
`/resume <id>` - Resume session
`/branch <name>` - Create branch session
`/save` - Save session
`/exit` or `/quit` - Quit
`/psi` - Show PSI cognitive state
`/aether` - Show Aether orchestration status
`/gateways` - Show gateway status
`/mcp` - List MCP servers and tools
`/yolo` - Toggle YOLO mode
`/cancel` - Cancel current operation
`/undo` - Remove last assistant turn
`/redo` - Placeholder (restore last removed turn)
`/copy` - Copy last assistant response to clipboard
`/edit` - Open last assistant response in editor
`/file <path>` - Attach file to next message
`/image <path>` - Attach image to next message
`/url <url>` - Fetch URL and attach as context
`/search <query>` - Run web search
`/approve` - Show approval status/settings
`/status` - Show system status
`/logs` - Show recent logs
`/debug` - Toggle debug mode
`/compact` or `/compress` - Compress conversation context
`/tokens` - Show token usage estimate
`/usage` - Show model/provider/token/context usage summary

**Keyboard Shortcuts**
Ctrl+Enter - Submit
Shift+Enter - Newline in input
Ctrl+L - Clear screen
Ctrl+C - Quit
Ctrl+W - Toggle sidebar
Tab - Autocomplete slash command
Esc - Cancel / close overlays
C - Copy focused assistant message"""
            chat_log.add_assistant(help_text)

        def _cmd_new(self, rest, chat_log, status_bar):
            chat_log.clear_all()
            agent = get_or_create_agent()
            if agent:
                try:
                    agent._conversation = []
                except Exception:
                    pass
            chat_log.add_sys("Session cleared. Started new conversation.")
            self._buf = ""
            self._welcome_visible = False
            try:
                old = chat_log.query_one("#response")
                old.remove()
            except Exception:
                pass

        def _cmd_config(self, rest, chat_log, status_bar):
            parts = rest.strip().split(None, 2)
            sub = parts[0].lower() if parts else ""
            try:
                from laap.cli.config_manager import config_manager
                if sub == "wizard":
                    ok = config_manager.interactive_wizard()
                    if ok:
                        reinitialize_agent()
                        status_bar.model = self._safe_model_name()
                        status_bar.provider = self._safe_provider_name()
                        chat_log.add_sys("[OK] Configuration saved. Agent reinitialized.")
                    else:
                        chat_log.add_sys("[WARN] Wizard did not complete; configuration may be incomplete")
                    return
                if sub == "set" and len(parts) >= 3:
                    key, value = parts[1], parts[2]
                    if config_manager.set_key(key, value):
                        reinitialize_agent()
                        status_bar.model = self._safe_model_name()
                        status_bar.provider = self._safe_provider_name()
                        chat_log.add_sys(f"[OK] Set {key} and reinitialized agent")
                    else:
                        chat_log.add_sys(f"[ERROR] Failed to set {key}")
                    return
                summary = config_manager.get_summary()
                lines = [
                    "Configuration:",
                    f"  Provider: {summary.get('provider', '?')}",
                    f"  Name: {summary.get('provider_name', '?')}",
                    f"  Model: {summary.get('model', '?')}",
                    f"  Base URL: {summary.get('base_url', '?') or 'default'}",
                    f"  API Key: {'set' if summary.get('api_key_set') else 'not set'}",
                    f"  Configured: {summary.get('configured', False)}",
                ]
                chat_log.add_sys("\n".join(lines))
            except Exception as e:
                logger.exception("[ERROR] config command failed")
                chat_log.add_sys(f"[ERROR] Config command failed: {e}")

        def _cmd_model(self, rest, chat_log, status_bar):
            parts = rest.strip().split()
            sub = parts[0].lower() if parts else ""
            try:
                from laap.cli.config_manager import config_manager
                if sub in ("list", "ls") or (not rest.strip() and len(parts) == 1 and parts[0].lower() in ("list", "ls")):
                    self._cmd_model_list(rest, chat_log, status_bar)
                    return
                if sub in ("use", "set") and len(parts) >= 2:
                    model = parts[1]
                    if config_manager.set_model(model):
                        reinitialize_agent()
                        status_bar.model = self._safe_model_name()
                        status_bar.provider = self._safe_provider_name()
                        chat_log.add_sys(f"[OK] Model switched to {model}")
                    else:
                        chat_log.add_sys(f"[ERROR] Failed to switch to {model}")
                    return
                if sub == "provider" and len(parts) >= 2:
                    provider = parts[1]
                    if config_manager.switch_provider(provider):
                        reinitialize_agent()
                        status_bar.model = self._safe_model_name()
                        status_bar.provider = self._safe_provider_name()
                        chat_log.add_sys(f"[OK] Provider switched to {provider}")
                    else:
                        chat_log.add_sys(f"[ERROR] Unknown provider: {provider}")
                    return
                # Default: show current
                summary = config_manager.get_summary()
                chat_log.add_sys(
                    f"Model: {summary.get('model', '?')} | Provider: {summary.get('provider', '?')} | Key: {'OK' if summary.get('api_key_set') else 'NONE'}"
                )
            except Exception as e:
                logger.exception("[ERROR] model command failed")
                chat_log.add_sys(f"[ERROR] Model command failed: {e}")

        def _cmd_model_list(self, rest, chat_log, status_bar):
            try:
                from laap.cli.config_manager import config_manager
                models = config_manager.list_available_models()
                if not models:
                    # Fallback built-in catalog
                    models = [
                        {"id": "deepseek-v4-flash", "provider": "deepseek", "label": "DeepSeek V4 Flash"},
                        {"id": "deepseek-v4-pro", "provider": "deepseek", "label": "DeepSeek V4 Pro"},
                        {"id": "claude-sonnet-4-6", "provider": "anthropic", "label": "Claude Sonnet 4.6"},
                        {"id": "claude-opus-4-8", "provider": "anthropic", "label": "Claude Opus 4.8"},
                        {"id": "gpt-4o", "provider": "openai", "label": "GPT-4o"},
                        {"id": "gpt-4o-mini", "provider": "openai", "label": "GPT-4o Mini"},
                        {"id": "gemini-2.5-flash", "provider": "google", "label": "Gemini 2.5 Flash"},
                        {"id": "grok-3", "provider": "xai", "label": "Grok 3"},
                        {"id": "ollama/llama4", "provider": "ollama", "label": "Ollama Llama 4 (local)"},
                    ]
                lines = ["Available models:"]
                for m in models[:40]:
                    mid = m.get("id", "?")
                    prov = m.get("provider", "?")
                    label = m.get("label", mid)
                    lines.append(f"  {mid} [{prov}] - {label}")
                chat_log.add_sys("\n".join(lines))
            except Exception as e:
                logger.exception("[ERROR] model list failed")
                chat_log.add_sys(f"[ERROR] Failed to list models: {e}")

        def _cmd_tools(self, rest, chat_log, status_bar):
            overlay = self.query_one("#tool_overlay", ToolPickerOverlay)
            overlay.all_tools = self._all_tool_entries()
            overlay.update_matches(rest)
            overlay.styles.display = "block"
            self._tool_overlay_visible = True
            self.query_one("#input_box", Input).focus()

        def _cmd_sidebar(self, rest, chat_log, status_bar):
            self._toggle_sidebar()

        def _cmd_profile(self, rest, chat_log, status_bar):
            chat_log.add_sys(f"Profile: {self._profile_name or 'default'}")

        def _cmd_sessions(self, rest, chat_log, status_bar):
            try:
                from laap.store.session_manager import FileSessionStore, SessionManager
                mgr = SessionManager(FileSessionStore())
                sessions = mgr.store.list_sessions(limit=10)
                lines = ["Recent sessions:"]
                for s in sessions:
                    lines.append(f"  {s.session_id} - {s.state.value} ({s.turn_count} turns)")
                chat_log.add_sys("\n".join(lines) if len(lines) > 1 else "No saved sessions")
            except Exception as e:
                chat_log.add_sys(f"Sessions error: {e}")

        def _cmd_resume(self, rest, chat_log, status_bar):
            if not rest:
                chat_log.add_sys("Usage: /resume <session_id>")
                return
            try:
                from laap.store.session_manager import FileSessionStore, SessionManager
                mgr = SessionManager(FileSessionStore())
                msgs = mgr.store.load_messages(rest, limit=100)
                chat_log.clear_all()
                for msg in msgs:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        chat_log.add_user(content)
                    elif role == "assistant":
                        chat_log.add_assistant(content)
                self._session_id = rest
                chat_log.add_sys(f"Resumed session: {rest}")
            except Exception as e:
                chat_log.add_sys(f"Resume error: {e}")

        def _cmd_branch(self, rest, chat_log, status_bar):
            name = rest.strip() or f"{self._session_id}_branch"
            self._session_id = name
            chat_log.add_sys(f"Branched to session: {name}")

        def _cmd_save(self, rest, chat_log, status_bar):
            try:
                from laap.store.session_manager import FileSessionStore, SessionManager, SessionRecord
                mgr = SessionManager(FileSessionStore())
                session = mgr.create(self._session_id)
                mgr.store.save(session)
                chat_log.add_sys(f"Session saved: {self._session_id}")
            except Exception as e:
                chat_log.add_sys(f"Save error: {e}")

        def _cmd_exit(self, rest, chat_log, status_bar):
            self.app.exit()

        def _cmd_psi(self, rest, chat_log, status_bar):
            state = _fetch_psi_state()
            lines = [
                "PSI Cognitive State:",
                f"  Dominant feeling: {state.get('dominant_feeling', 'neutral')}",
                f"  Arousal: {state.get('arousal', 0.5):.2f}",
                f"  Valence: {state.get('valence', 0.0):.2f}",
                f"  Dominance: {state.get('dominance', 0.5):.2f}",
                f"  Confidence: {state.get('confidence', 0.5):.2f}",
            ]
            chat_log.add_sys("\n".join(lines))

        def _cmd_aether(self, rest, chat_log, status_bar):
            state = _fetch_aether_state()
            lines = [
                "Aether Orchestration Status:",
                f"  Actors: {state.get('actor_count', 0)}",
                f"  Running: {state.get('running', False)}",
                f"  Pending messages: {state.get('pending', 0)}",
            ]
            chat_log.add_sys("\n".join(lines))

        def _cmd_gateways(self, rest, chat_log, status_bar):
            telegram = bool(os.environ.get("TELEGRAM_BOT_TOKEN", ""))
            feishu = bool(os.environ.get("FEISHU_APP_ID", "") or os.environ.get("FEISHU_WEBHOOK_URL", ""))
            lines = [
                "Gateway Status:",
                f"  Telegram: {'configured' if telegram else 'disabled'}",
                f"  Feishu: {'configured' if feishu else 'disabled'}",
            ]
            chat_log.add_sys("\n".join(lines))

        def _cmd_mcp(self, rest, chat_log, status_bar):
            servers = _load_mcp_servers()
            if not servers:
                chat_log.add_sys("No MCP servers configured")
                return
            lines = ["MCP Servers:"]
            for name, transport, enabled in servers:
                status = "enabled" if enabled else "disabled"
                lines.append(f"  {name} ({transport}) - {status}")
            chat_log.add_sys("\n".join(lines))
            # Best-effort tool discovery
            try:
                mcp_tools = _load_mcp_tools()
                if mcp_tools:
                    lines = ["MCP Tools:"]
                    for server, tname, tdesc in mcp_tools[:20]:
                        lines.append(f"  {server}.{tname} - {tdesc[:50]}")
                    chat_log.add_sys("\n".join(lines))
            except Exception as e:
                chat_log.add_sys(f"MCP tool discovery error: {e}")

        def _cmd_yolo(self, rest, chat_log, status_bar):
            self._yolo = not self._yolo
            os.environ["LAAP_YOLO_MODE"] = "1" if self._yolo else "0"
            chat_log.add_sys(f"YOLO mode: {'ON' if self._yolo else 'OFF'}")

        def _cmd_cancel(self, rest, chat_log, status_bar):
            self._cancelled = True
            status_bar.status = "idle"
            chat_log.add_sys("Cancelled current operation.")

        # ── Hermes-level command implementations ─────────────────────

        def _cmd_reset(self, rest, chat_log, status_bar):
            try:
                chat_log.clear_all()
                global _agent_instance
                _agent_instance = None
                get_or_create_agent(force_reinit=True)
                chat_log.add_sys("[OK] Hard reset complete. Agent and conversation cleared.")
            except Exception as e:
                logger.exception("[ERROR] reset failed")
                chat_log.add_sys(f"[ERROR] Reset failed: {e}")

        def _cmd_undo(self, rest, chat_log, status_bar):
            if chat_log.pop_last_assistant_turn():
                chat_log.add_sys("[OK] Removed last assistant turn")
            else:
                chat_log.add_sys("[WARN] No assistant turn to undo")

        def _cmd_redo(self, rest, chat_log, status_bar):
            chat_log.add_sys("[INFO] Redo is a placeholder; removed turns are not kept in history")

        def _cmd_copy(self, rest, chat_log, status_bar):
            text = chat_log.get_last_assistant_text()
            if not text:
                chat_log.add_sys("[WARN] No assistant response to copy")
                return
            try:
                import pyperclip
                pyperclip.copy(text)
                chat_log.add_sys("[OK] Copied last response to clipboard")
            except Exception:
                chat_log.add_sys("[WARN] pyperclip not available; install it to use /copy")

        def _cmd_edit(self, rest, chat_log, status_bar):
            text = chat_log.get_last_assistant_text()
            if not text:
                chat_log.add_sys("[WARN] No assistant response to edit")
                return
            editor = os.environ.get("EDITOR", "notepad" if sys.platform == "win32" else "nano")
            import tempfile
            import subprocess
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
                    f.write(text)
                    path = f.name
                subprocess.run([editor, path], check=False)
                edited = Path(path).read_text(encoding="utf-8")
                chat_log.add_assistant(edited)
                chat_log.add_sys("[OK] Edited response inserted")
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Edit failed: {e}")

        def _cmd_file(self, rest, chat_log, status_bar):
            path = rest.strip()
            if not path:
                chat_log.add_sys("Usage: /file <path>")
                return
            p = Path(path)
            if not p.exists():
                chat_log.add_sys(f"[ERROR] File not found: {path}")
                return
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
                self._attachments.append(f"File: {path}\n```\n{content[:4000]}\n```")
                chat_log.add_sys(f"[OK] Attached file: {path}")
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Failed to read {path}: {e}")

        def _cmd_image(self, rest, chat_log, status_bar):
            path = rest.strip()
            if not path:
                chat_log.add_sys("Usage: /image <path>")
                return
            p = Path(path)
            if not p.exists():
                chat_log.add_sys(f"[ERROR] Image not found: {path}")
                return
            self._attachments.append(f"Image: {path}")
            chat_log.add_sys(f"[OK] Attached image: {path}")

        def _cmd_url(self, rest, chat_log, status_bar):
            url = rest.strip()
            if not url:
                chat_log.add_sys("Usage: /url <url>")
                return
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "LAAP/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")[:6000]
                self._attachments.append(f"URL: {url}\n```\n{html}\n```")
                chat_log.add_sys(f"[OK] Fetched URL: {url}")
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Failed to fetch {url}: {e}")

        def _cmd_search(self, rest, chat_log, status_bar):
            query = rest.strip()
            if not query:
                chat_log.add_sys("Usage: /search <query>")
                return
            try:
                from laap.tools.web import web_search
                results = web_search(query)
                self._attachments.append(f"Web search: {query}\n{results}")
                chat_log.add_sys(f"[OK] Search results attached for: {query}")
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Search failed: {e}")

        def _cmd_approve(self, rest, chat_log, status_bar):
            chat_log.add_sys(f"Approval system: YOLO mode is {'ON' if self._yolo else 'OFF'}")

        def _cmd_status(self, rest, chat_log, status_bar):
            agent = get_or_create_agent()
            lines = ["System Status:"]
            if agent:
                lines.append(f"  Agent ID: {agent.id}")
                lines.append(f"  Tools: {agent.tool_registry.count}")
                lines.append(f"  Steps: {agent.step_count}")
            lines.append(f"  Session: {self._session_id}")
            lines.append(f"  Profile: {self._profile_name}")
            lines.append(f"  Debug: {'ON' if self._debug else 'OFF'}")
            chat_log.add_sys("\n".join(lines))

        def _cmd_logs(self, rest, chat_log, status_bar):
            try:
                log_path = _TUI_LOG_FILE
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8").splitlines()[-30:]
                    chat_log.add_sys("Recent TUI logs:\n" + "\n".join(lines))
                else:
                    chat_log.add_sys("[WARN] No TUI log file found")
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Failed to read logs: {e}")

        def _cmd_debug(self, rest, chat_log, status_bar):
            self._debug = not self._debug
            os.environ["LAAP_DEBUG"] = "1" if self._debug else "0"
            chat_log.add_sys(f"Debug mode: {'ON' if self._debug else 'OFF'}")

        def _cmd_compact(self, rest, chat_log, status_bar):
            try:
                agent = get_or_create_agent()
                if agent and hasattr(agent, "_conversation") and agent._conversation:
                    before = len(agent._conversation)
                    # Keep system + first user + last 6 messages
                    system_msgs = [m for m in agent._conversation if getattr(m, "role", m.get("role")) == "system"]
                    tail = agent._conversation[-6:]
                    agent._conversation = system_msgs + tail
                    after = len(agent._conversation)
                    chat_log.add_sys(f"[OK] Compressed conversation: {before} -> {after} messages")
                else:
                    chat_log.add_sys("[WARN] No conversation to compress")
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Compact failed: {e}")

        def _cmd_tokens(self, rest, chat_log, status_bar):
            agent = get_or_create_agent()
            conv_len = 0
            try:
                if agent and hasattr(agent, "_conversation"):
                    conv_len = len(agent._conversation)
            except Exception:
                pass
            est = status_bar.tokens
            chat_log.add_sys(f"Token usage estimate: {est} | Conversation turns: {conv_len}")

        def _cmd_compress(self, rest, chat_log, status_bar):
            """Hermes-style alias for /compact."""
            self._cmd_compact(rest, chat_log, status_bar)

        def _cmd_usage(self, rest, chat_log, status_bar):
            """Hermes-style usage summary: model, provider, tokens, context, time."""
            try:
                from laap.cli.config_manager import config_manager
                summary = config_manager.get_summary()
                lines = [
                    "Usage:",
                    f"  Provider: {summary.get('provider', '?')}",
                    f"  Model: {summary.get('model', '?')}",
                    f"  Base URL: {summary.get('base_url') or 'default'}",
                    f"  API Key: {'set' if summary.get('api_key_set') else 'not set'}",
                    f"  Tokens (session estimate): {status_bar.tokens}",
                    f"  Context fill: {status_bar.context_fill * 100:.0f}%",
                    f"  Last elapsed: {status_bar.elapsed:.1f}s",
                    f"  Encoding: {status_bar.encoding}",
                ]
                chat_log.add_sys("\n".join(lines))
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Usage command failed: {e}")

        def _cmd_vitals(self, rest, chat_log, status_bar):
            """Show digital lifeform vital signs in the chat log."""
            try:
                state = _fetch_psi_state()
                lines = [
                    "Lifeform Vitals:",
                    f"  PSI loop: {state.get('psi_loop', 'idle')}",
                    f"  Curiosity:  {state.get('curiosity', 0):.2f}",
                    f"  Trust:      {state.get('trust', 0):.2f}",
                    f"  Competence: {state.get('competence', 0):.2f}",
                    f"  Energy:     {state.get('energy', 0):.2f}",
                    f"  Coherence:  {state.get('coherence', 0):.2f}",
                ]
                chat_log.add_sys("\n".join(lines))
            except Exception as e:
                chat_log.add_sys(f"[ERROR] Vitals command failed: {e}")

        def _toggle_sidebar(self):
            sidebar = self.query_one("#sidebar")
            if self._sidebar_visible:
                sidebar.styles.display = "none"
            else:
                sidebar.styles.display = "block"
            self._sidebar_visible = not self._sidebar_visible

        def action_clear(self):
            self.query_one("#chat_log", ChatLog).clear_all()

        def action_toggle_sidebar(self):
            self._toggle_sidebar()

        async def _chat(self, text: str, chat_log: ChatLog, status_bar: StatusBar):
            worker = get_current_worker()
            agent = get_or_create_agent()
            if not agent or not agent.llm:
                chat_log.remove_thinking()
                cfg = _active_config_summary()
                if not cfg.get("api_key"):
                    chat_log.add_sys("No API key configured. Run: /config set api_key <key>")
                else:
                    chat_log.add_sys("No LLM configured. Run: laap --config")
                status_bar.status = "idle"
                return

            start = time.time()
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(chat_log.start_response)
            self._buf = ""
            self._cancelled = False

            from laap.ui.stream_handler import StreamHandler
            sh = StreamHandler(verbose=False, use_spinner=False)

            def on_tok(tok: str):
                self._buf += tok
                try:
                    loop.call_soon_threadsafe(chat_log.stream_assistant, tok, True)
                except Exception as e:
                    logger.debug("[WARN] on_tok callback failed: %s", e)

            def on_tool_start(name: str, args: dict):
                try:
                    widget = chat_log.add_tool_progress(name, args or {})
                    key = f"{name}_{uuid.uuid4().hex[:6]}"
                    self._tool_progress_widgets[key] = widget
                    loop.call_soon_threadsafe(chat_log.add_tool, name, True)
                    status_bar.generation_mode = f"tool:{name}"
                except Exception as e:
                    logger.debug("[WARN] on_tool_start callback failed: %s", e)

            def on_tool_end(name: str, result: Any, duration: float = 0.0, success: bool = True):
                try:
                    key = None
                    for k in list(self._tool_progress_widgets.keys()):
                        if k.startswith(name + "_"):
                            key = k
                            break
                    widget = self._tool_progress_widgets.pop(key, None) if key else None
                    ok = success
                    if hasattr(result, "success"):
                        ok = bool(result.success)
                    loop.call_soon_threadsafe(
                        chat_log.update_tool_progress, widget, ok, str(result)[:80], duration
                    )
                    status_bar.generation_mode = "thinking" if not self._cancelled else "idle"
                except Exception as e:
                    logger.debug("[WARN] on_tool_end callback failed: %s", e)

            sh.on_token = on_tok
            sh.on_tool_call = on_tool_start
            if hasattr(sh, "on_tool_result"):
                sh.on_tool_result = on_tool_end

            full_response = ""
            try:
                status_bar.generation_mode = "thinking"
                # Run the blocking chat call in a worker thread so the event loop stays alive.
                future = asyncio.ensure_future(asyncio.to_thread(agent.chat, text, "", None, None, sh))
                def _worker_cancelled() -> bool:
                    if not worker:
                        return False
                    cancelled = getattr(worker, "is_cancelled", False)
                    # Defensive: MagicMock is truthy in tests; treat unknown types as False.
                    if type(cancelled).__name__ in ("MagicMock", "Mock"):
                        return False
                    return bool(cancelled)

                while not future.done():
                    if self._cancelled or _worker_cancelled():
                        sh.stop()
                        self._cancelled = True
                        future.cancel()
                        break
                    await asyncio.sleep(0.05)
                full_response = future.result() if not self._cancelled else ""
            except asyncio.CancelledError:
                self._cancelled = True
                full_response = ""
            except Exception as e:
                logger.exception("[ERROR] chat failed")
                loop.call_soon_threadsafe(chat_log.remove_thinking)
                err_msg = str(e) or "Unknown error"
                friendly = err_msg
                lower_err = err_msg.lower()
                if any(k in lower_err for k in ("api key", "authentication", "unauthorized", "401")):
                    friendly = "Authentication failed. Check your API key with /config set api_key <key>"
                elif any(k in lower_err for k in ("connection", "timeout", "connect", "network", "dns")):
                    friendly = "Connection failed. Check your network or base URL with /config set base_url <url>"
                elif any(k in lower_err for k in ("model", "not found", "does not exist", "invalid")):
                    friendly = f"Model error ({err_msg}). Switch models with /model use <model_id>"
                loop.call_soon_threadsafe(chat_log.add_sys, f"[ERROR] {friendly}")
                status_bar.status = "idle"
                status_bar.generation_mode = "idle"
                return

            loop.call_soon_threadsafe(chat_log.remove_thinking)
            if self._cancelled:
                loop.call_soon_threadsafe(chat_log.add_sys, "Operation cancelled by user.")
                status_bar.status = "idle"
                status_bar.generation_mode = "idle"
                return
            if full_response:
                self._buf = full_response
                self._last_response_text = full_response
                loop.call_soon_threadsafe(chat_log.finalize_assistant_response, full_response)
            else:
                loop.call_soon_threadsafe(chat_log.add_sys, "[WARN] Model returned an empty response")
            status_bar.status = "idle"
            status_bar.generation_mode = "idle"
            status_bar.tokens += max(1, len(full_response or "") // 4)
            status_bar.elapsed = time.time() - start
            status_bar.context_fill = min(1.0, status_bar.tokens / 8000)

        def q(self, widget):
            if isinstance(widget, str):
                return self.query_one("#" + widget)
            return self.query_one(widget)

    class LAAPApp(App):
        """LAAP Application with Hermes-inspired design."""

        TITLE = "LAAP - Golden Dragon Agent"

        CSS = """
        Screen {
            background: #000000;
            color: #C8C8C8;
        }

        #main_content {
            width: 1fr;
            height: 100%;
            layout: vertical;
            background: #000000;
        }

        #sidebar {
            width: 26;
            height: 100%;
            background: #0D0D0D;
            border-right: solid #333333;
            padding: 1;
        }

.tool_progress, .coding_progress {
            background: #0A0A0A;
            border: solid #B8860B;
            text-style: bold;
            tint: #FFD700 5%;
        }

        .coding_progress {
            border: solid #FFD700;
            text-style: bold italic;
            tint: #FFD700 10%;
        }

        ToolStatusPanel, SystemMetricsPanel, CognitiveStatePanel, GatewayStatusPanel, VitalSignsPanel, VideoPlayerPanel {
            height: auto;
            background: #111111;
            padding: 1;
            margin-bottom: 1;
            border: solid #1A1A1A;
        }

        VideoPlayerPanel:hover {
            border: solid #FFD700;
            tint: #FFD700 8%;
        }

        #panel_title {
            color: #FFD700;
            margin-bottom: 1;
            padding-bottom: 1;
            border-bottom: solid #333333;
        }

        #chat_log {
            height: 1fr;
            background: #000000;
            padding: 0;
        }

        #welcome {
            background: #000000;
        }

        #input_container {
            height: auto;
            layout: vertical;
            background: #000000;
        }

        #light_bar {
            height: 1;
            background: #000000;
            color: #FFD700;
            content-align: center middle;
        }

        #light_bar Static {
            color: #FFD700;
        }

        #input_box {
            height: 3;
            background: #0A0A0A;
            color: #FFD700;
            border: solid #333333;
            padding: 0 2;
            margin: 1;
        }

        #input_box:focus {
            border: solid #FFD700;
            background: #111111;
            text-style: bold;
        }

        #slash_overlay {
            height: auto;
            max-height: 12;
            background: #0D0D0D;
            border: solid #333333;
            padding: 0 1;
            margin: 0 1;
            display: none;
        }

        #tool_overlay {
            height: auto;
            max-height: 14;
            background: #0D0D0D;
            border: solid #333333;
            padding: 0 1;
            margin: 0 1;
            display: none;
        }

        #overlay_title, #tool_overlay_title {
            color: #FFD700;
            border-bottom: solid #333333;
            margin-bottom: 1;
        }

        .cmd_item, .tool_item {
            padding: 0 1;
        }

        .user_msg {
            text-align: right;
        }

        .assistant_msg {
            text-align: left;
        }

        .sys_msg {
            text-style: italic;
        }

        .tool_progress, .coding_progress {
            background: #0A0A0A;
            border: solid #B8860B;
            text-style: bold;
            tint: #FFD700 5%;
        }

        .coding_progress {
            border: solid #FFD700;
            text-style: bold italic;
            tint: #FFD700 10%;
        }

        #status_bar {
            height: 1;
            background: #0D0D0D;
            border-top: solid #333333;
            padding: 0 2;
        }

        Static {
            color: #C8C8C8;
        }

        .dim {
            color: #888888;
        }

        .info {
            color: #4FC1FF;
        }

        .warning {
            color: #FFA500;
        }

        ScrollableContainer {
            scrollbar-color: #FFD700;
            scrollbar-size: 1 1;
        }

        Button {
            background: #111111;
            border: solid #333333;
            color: #C8C8C8;
            margin: 0 1;
        }

        Button:hover {
            background: #1A1A1A;
            border: solid #FFD700;
        }

        #approval_dialog {
            width: 70;
            height: auto;
            background: #0D0D0D;
            border: solid #B8960C;
            padding: 1 2;
        }

        #approval_title {
            color: #FFA500;
            text-style: bold;
            margin-bottom: 1;
        }

        #approval_command {
            color: #FFD700;
            margin-bottom: 1;
        }

        #approval_pattern {
            color: #888888;
            margin-bottom: 1;
        }

        #approval_buttons {
            height: auto;
            align: center middle;
        }
        """

        def get_default_screen(self):
            return MainScreen()

else:
    # Stubs when Textual is not installed. Public symbols remain importable
    # so callers can detect availability via HAS_TEXTUAL and fall back.
    class StatusBar:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class ToolStatusPanel:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class SystemMetricsPanel:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class CognitiveStatePanel:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class VitalSignsPanel:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class VideoPlayerPanel:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class GatewayStatusPanel:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class ChatLog:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class SlashCommandOverlay:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class ToolPickerOverlay:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class ApprovalModal:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class MainScreen:  # type: ignore
        def __init__(self, *args, **kwargs): pass

    class LAAPApp:  # type: ignore
        def __init__(self, *args, **kwargs): pass


# ── Shared helpers (do not depend on Textual) ────────────────────
def _build_slash_commands() -> List[Dict[str, Any]]:
    """Combine the CLI command registry with LAAP-specific TUI commands."""
    try:
        from laap.cli.commands import COMMAND_REGISTRY
    except Exception:
        COMMAND_REGISTRY = []

    base = {c["name"]: dict(c) for c in COMMAND_REGISTRY}
    extra = [
        {"name": "psi", "aliases": [], "desc": "Show PSI cognitive state", "category": "Cognitive"},
        {"name": "aether", "aliases": [], "desc": "Show Aether orchestration status", "category": "Cognitive"},
        {"name": "gateways", "aliases": [], "desc": "Show gateway status", "category": "Gateway"},
        {"name": "mcp", "aliases": [], "desc": "List MCP servers and tools", "category": "Tools & Skills"},
        {"name": "yolo", "aliases": [], "desc": "Toggle YOLO mode", "category": "Safety"},
        {"name": "cancel", "aliases": [], "desc": "Cancel current operation", "category": "Control"},
        # Hermes-level additions
        {"name": "reset", "aliases": [], "desc": "Hard reset agent and conversation", "category": "Control"},
        {"name": "undo", "aliases": [], "desc": "Remove last assistant turn", "category": "Control"},
        {"name": "redo", "aliases": [], "desc": "Restore last removed turn", "category": "Control"},
        {"name": "copy", "aliases": [], "desc": "Copy last assistant response to clipboard", "category": "Utility"},
        {"name": "edit", "aliases": [], "desc": "Open last assistant response in editor", "category": "Utility"},
        {"name": "file", "aliases": [], "desc": "Attach file to next message", "category": "Utility"},
        {"name": "image", "aliases": [], "desc": "Attach image to next message", "category": "Utility"},
        {"name": "url", "aliases": [], "desc": "Fetch URL and attach as context", "category": "Utility"},
        {"name": "search", "aliases": [], "desc": "Run web search", "category": "Utility"},
        {"name": "approve", "aliases": [], "desc": "Show approval status/settings", "category": "Safety"},
        {"name": "status", "aliases": [], "desc": "Show system status", "category": "System"},
        {"name": "logs", "aliases": [], "desc": "Show recent logs", "category": "System"},
        {"name": "debug", "aliases": [], "desc": "Toggle debug mode", "category": "System"},
        {"name": "compact", "aliases": [], "desc": "Compress conversation context", "category": "System"},
        {"name": "compress", "aliases": [], "desc": "Compress conversation context (Hermes alias)", "category": "System"},
        {"name": "tokens", "aliases": [], "desc": "Show token usage estimate", "category": "System"},
        {"name": "usage", "aliases": [], "desc": "Show model/provider/token/context usage summary", "category": "System"},
    ]
    for e in extra:
        base.setdefault(e["name"], e)
    return sorted(base.values(), key=lambda x: x["name"])


def _fetch_psi_state() -> Dict[str, Any]:
    """Return PSI state from the global cognitive bus or agent, or mock values."""
    try:
        agent = get_or_create_agent()
        if agent and hasattr(agent, "emotion_gradient"):
            eg = agent.emotion_gradient
            arousal = getattr(eg.state, "arousal", 0.5)
            valence = getattr(eg.state, "valence", 0.0)
            dominance = getattr(eg.state, "dominance", 0.5)
            confidence = getattr(eg.state, "confidence", 0.5)
            psi_loop = "active" if getattr(agent, "alive", False) else "idle"
            return {
                "dominant_feeling": getattr(eg.state, "dominant_feeling", getattr(eg.state, "name", "neutral")),
                "arousal": arousal,
                "valence": valence,
                "dominance": dominance,
                "confidence": confidence,
                "psi_loop": psi_loop,
                "curiosity": 0.3 + arousal * 0.6 + confidence * 0.1,
                "trust": 0.5 + valence * 0.4 + dominance * 0.1,
                "competence": 0.4 + dominance * 0.4 + confidence * 0.2,
                "energy": 0.2 + arousal * 0.7 + confidence * 0.1,
                "coherence": 0.5 + (1.0 - abs(valence)) * 0.3 + confidence * 0.2,
            }
    except Exception:
        pass
    return {
        "dominant_feeling": "neutral",
        "arousal": 0.5,
        "valence": 0.0,
        "dominance": 0.5,
        "confidence": 0.5,
        "psi_loop": "idle",
        "curiosity": 0.62,
        "trust": 0.74,
        "competence": 0.81,
        "energy": 0.88,
        "coherence": 0.79,
    }


def _fetch_aether_state() -> Dict[str, Any]:
    """Return Aether actor system state or mock values."""
    try:
        agent = get_or_create_agent()
        if agent and hasattr(agent, "cognitive_bus"):
            bus = agent.cognitive_bus
            system = getattr(bus, "system", None)
            if system:
                return {
                    "actor_count": len(getattr(system, "actors", {})),
                    "running": getattr(bus, "_running", False),
                    "pending": len(getattr(system, "mailbox", [])),
                }
    except Exception:
        pass
    return {"actor_count": 0, "running": False, "pending": 0}


# ── TUI runner classes/functions ─────────────────────────────────
class LAAP_TUI:
    """LAAP TUI runner class (backward compatible)."""

    def __init__(self, agent=None, config_manager=None):
        self.agent = agent
        self.config_manager = config_manager
        if agent is not None:
            global _agent_instance
            _agent_instance = agent

    def run(self):
        """Run the TUI application."""
        if not HAS_TEXTUAL:
            raise RuntimeError(
                "Textual is not installed. Install it with: pip install textual"
            )
        LAAPApp().run()


def run_tui():
    """Run the LAAP TUI."""
    if not HAS_TEXTUAL:
        raise RuntimeError(
            "Textual is not installed. Install it with: pip install textual"
        )
    LAAPApp().run()


if __name__ == "__main__":
    run_tui()
