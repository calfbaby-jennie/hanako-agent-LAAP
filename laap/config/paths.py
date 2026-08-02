"""LAAP 统一路径解析模块。

所有业务代码应当通过本模块获取项目根目录、Hermes 根目录以及各类运行时目录，
而不是继续使用 ``D:\\LAAP``、``D:\\hermes-agent-main (1)\\hermes-agent-main`` 等硬编码路径。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def get_laap_root() -> Path:
    """返回 LAAP 项目根目录。

    优先级：
        1. ``LAAP_ROOT`` 环境变量。
        2. 由当前文件位置推导（``laap/config/paths.py`` 的上三级目录）。
        3. 当前工作目录。
    """
    env_root = os.environ.get("LAAP_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    # 从 laap/config/paths.py -> laap/config -> laap -> <project_root>
    derived = Path(__file__).resolve().parent.parent.parent
    if derived.exists():
        return derived

    return Path.cwd().resolve()


def get_laap_home() -> Path:
    """Return the writable LAAP data directory.

    ``LAAP_ROOT`` identifies the source checkout and must not be used as the
    default persistence directory. ``LAAP_HOME`` is the single override for
    user-owned LAAP data on every supported platform.
    """
    env_home = os.environ.get("LAAP_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".laap").resolve()


def get_hana_home() -> Path:
    """Return the writable HanaAgent data directory."""
    env_home = os.environ.get("HANA_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return (Path.home() / ".hana").resolve()


def get_aris_brain_dir() -> Path:
    """Return the writable compatibility directory for legacy Aris state."""
    env_brain = os.environ.get("ARIS_BRAIN_DIR")
    if env_brain:
        return Path(env_brain).expanduser().resolve()
    return get_laap_home() / "aris_brain"


def get_hermes_root() -> Path | None:
    """返回 Hermes 根目录。

    优先级：
        1. ``HERMES_ROOT`` 环境变量。
        2. 在 ``PATH`` 中查找 ``hermes`` / ``hermes.exe`` 可执行文件所在目录。
        3. 默认子目录 ``./hermes-agent-main`` 或 ``./external/hermes-agent-main``。
        4. 未找到时返回 ``None``，不会抛出异常。
    """
    env_hermes = os.environ.get("HERMES_ROOT")
    if env_hermes:
        return Path(env_hermes).expanduser().resolve()

    for exe_name in ("hermes", "hermes.exe"):
        exe_path = shutil.which(exe_name)
        if exe_path:
            return Path(exe_path).expanduser().resolve().parent

    laap_root = get_laap_root()
    for sub in ("hermes-agent-main", Path("external") / "hermes-agent-main"):
        candidate = laap_root / sub
        if candidate.exists():
            return candidate

    return None


def _platform_app_dir(kind: str) -> Path:
    """返回当前平台的应用级目录（state/cache/data）。"""
    home = Path.home()
    if sys.platform == "win32":
        local_appdata = os.environ.get("LOCALAPPDATA", home / "AppData" / "Local")
        return Path(local_appdata) / "laap" / kind
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / "laap" / kind
    # Linux / 其他 Unix
    return home / ".local" / kind / "laap"


def get_state_dir() -> Path:
    """返回 LAAP 状态目录。

    优先级：
        1. ``LAAP_STATE_DIR`` 环境变量。
        2. ``LAAP_HOME/state``。
        3. ``XDG_STATE_HOME/laap``（若设置）。
        4. ``~/.laap/state``。
    """
    env_state = os.environ.get("LAAP_STATE_DIR")
    if env_state:
        return Path(env_state).expanduser().resolve()

    if os.environ.get("LAAP_HOME"):
        return get_laap_home() / "state"

    xdg_state_home = os.environ.get("XDG_STATE_HOME")
    if xdg_state_home:
        return Path(xdg_state_home).expanduser().resolve() / "laap"

    return get_laap_home() / "state"


def get_cache_dir() -> Path:
    """返回 LAAP 缓存目录。

    优先级：
        1. ``LAAP_CACHE_DIR`` 环境变量。
        2. ``LAAP_HOME/cache``。
        3. ``XDG_CACHE_HOME/laap``（若设置）。
        4. ``~/.laap/cache``。
    """
    env_cache = os.environ.get("LAAP_CACHE_DIR")
    if env_cache:
        return Path(env_cache).expanduser().resolve()

    if os.environ.get("LAAP_HOME"):
        return get_laap_home() / "cache"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home).expanduser().resolve() / "laap"

    return get_laap_home() / "cache"


def get_models_dir() -> Path:
    """返回模型目录。

    优先级：
        1. ``LAAP_MODELS_DIR`` 环境变量。
        2. ``<laap_root>/public/models``。
    """
    env_models = os.environ.get("LAAP_MODELS_DIR")
    if env_models:
        return Path(env_models).expanduser().resolve()
    return get_laap_root() / "public" / "models"


def get_logs_dir() -> Path:
    """返回日志目录。

    优先级：
        1. ``LAAP_LOGS_DIR`` 环境变量。
        2. ``LAAP_HOME/logs``。
    """
    env_logs = os.environ.get("LAAP_LOGS_DIR")
    if env_logs:
        return Path(env_logs).expanduser().resolve()
    return get_laap_home() / "logs"


def get_video_dir() -> Path:
    """返回 TUI 视频所在目录。

    优先级：
        1. ``LAAP_VIDEO_DIR`` 环境变量。
        2. ``<laap_root>/.trae/specs/integrate-aether-orchestration-into-laap``。
    """
    env_video = os.environ.get("LAAP_VIDEO_DIR")
    if env_video:
        return Path(env_video).expanduser().resolve()
    return (
        get_laap_root()
        / ".trae"
        / "specs"
        / "integrate-aether-orchestration-into-laap"
    )
