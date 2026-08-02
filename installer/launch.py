#!/usr/bin/env python3
"""LAAP Hanako 启动脚本.

启动顺序：
1. 端口冲突检测（2668/11521/8765）
2. MCP server（后台）
3. sidecar（后台）
4. hanako desktop（前台）

用法：
    python installer/launch.py           # 启动全部
    python installer/launch.py --check   # 仅检测端口
    python installer/launch.py --sidecar # 仅启动 sidecar
    python installer/launch.py --mcp     # 仅启动 MCP server
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── 路径定位 ──────────────────────────────────────────────────
INSTALLER_DIR = Path(__file__).parent.resolve()
LAAP_ROOT = INSTALLER_DIR.parent

# 各组件脚本（相对 LAAP_ROOT）
MCP_SERVER_SCRIPT = LAAP_ROOT / "scripts" / "laap_cognitive_mcp.py"
SIDECAR_SCRIPT = LAAP_ROOT / "hanako" / "aris-bridge" / "aris-engine" / "sidecar.py"
HANAKO_DIR = LAAP_ROOT / "hanako"


def build_runtime_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build one canonical environment for every LAAP/Hanako child process.

    ``HOME`` remains owned by the operating system.  Project code, LAAP data,
    and Hanako data use separate explicit variables so launching from Finder,
    an IDE, or a shell resolves the same locations.
    """
    env = dict(base_env if base_env is not None else os.environ)
    user_home = Path.home()
    laap_home = Path(env.get("LAAP_HOME", user_home / ".laap")).expanduser().resolve()
    hana_home = Path(env.get("HANA_HOME", user_home / ".hana")).expanduser().resolve()

    env.setdefault("LAAP_ROOT", str(LAAP_ROOT))
    env["LAAP_HOME"] = str(laap_home)
    env["HANA_HOME"] = str(hana_home)
    env.setdefault("LAAP_STATE_DIR", str(laap_home / "state"))
    env.setdefault("LAAP_CACHE_DIR", str(laap_home / "cache"))
    env.setdefault("LAAP_LOGS_DIR", str(laap_home / "logs"))
    env.setdefault("ARIS_BRAIN_DIR", str(laap_home / "aris_brain"))
    env.setdefault("LAAP_PYTHON", sys.executable)

    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
    if str(LAAP_ROOT) not in pythonpath_entries:
        pythonpath_entries.insert(0, str(LAAP_ROOT))
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return env


def ensure_runtime_dirs(env: Optional[Dict[str, str]] = None) -> None:
    """Create writable runtime directories immediately before launching."""
    runtime_env = env or RUNTIME_ENV
    for path_var in (
        "LAAP_HOME",
        "HANA_HOME",
        "LAAP_STATE_DIR",
        "LAAP_CACHE_DIR",
        "LAAP_LOGS_DIR",
        "ARIS_BRAIN_DIR",
    ):
        Path(runtime_env[path_var]).mkdir(parents=True, exist_ok=True)


RUNTIME_ENV = build_runtime_env()

# ── 端口定义（与 check_env.py 保持一致）──────────────────────
PORT_COGNITIVE_BUS = 2668   # Hanako 后端 / CognitiveBus
PORT_SIDECAR = 11521        # Aris sidecar
PORT_MCP = 8765             # LAAP MCP server


# ── 端口检测 ──────────────────────────────────────────────────

def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """端口被占用返回 True。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except (OSError, socket.error):
        return True
    finally:
        sock.close()
    return False


def check_ports() -> bool:
    """检测三个关键端口，全部空闲返回 True。"""
    print("=" * 56)
    print("  端口冲突检测")
    print("=" * 56)
    all_free = True
    for port, label in [
        (PORT_COGNITIVE_BUS, "CognitiveBus / Hanako 后端"),
        (PORT_SIDECAR, "Aris sidecar"),
        (PORT_MCP, "LAAP MCP server"),
    ]:
        if port_in_use(port):
            print(f"  [FAIL] 端口 {port:<6} 被占用 ({label})")
            all_free = False
        else:
            print(f"  [OK]   端口 {port:<6} 空闲 ({label})")
    print("=" * 56)
    if all_free:
        print("  全部端口空闲，可以启动。")
    else:
        print("  存在端口冲突，请先释放被占用的端口。")
        # 给出排查提示
        print()
        print("  排查方法：")
        if os.name == "nt":
            print(f"    netstat -ano | findstr :{PORT_COGNITIVE_BUS}")
            print(f"    netstat -ano | findstr :{PORT_SIDECAR}")
            print(f"    netstat -ano | findstr :{PORT_MCP}")
        else:
            print(f"    lsof -i :{PORT_COGNITIVE_BUS}")
            print(f"    lsof -i :{PORT_SIDECAR}")
            print(f"    lsof -i :{PORT_MCP}")
    print("=" * 56)
    return all_free


# ── 进程管理 ──────────────────────────────────────────────────

class ProcessManager:
    """管理后台子进程，确保退出时一并清理。"""

    def __init__(self) -> None:
        self._children: List[subprocess.Popen] = []

    def start_background(self, cmd: List[str], cwd: Optional[Path] = None,
                         name: str = "") -> Optional[subprocess.Popen]:
        """启动后台进程，输出重定向到 DEVNULL（避免污染前台）。"""
        try:
            kwargs = dict(
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(cwd) if cwd else None,
                env=RUNTIME_ENV,
            )
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            proc = subprocess.Popen(cmd, **kwargs)
            self._children.append(proc)
            print(f"  [OK]   {name} 启动 (PID {proc.pid})")
            return proc
        except Exception as e:
            print(f"  [FAIL] {name} 启动失败：{e}")
            return None

    def stop_all(self) -> None:
        """停止所有后台子进程。"""
        for proc in reversed(self._children):
            if proc.poll() is not None:
                continue
            try:
                if os.name == "nt":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
            except Exception:
                pass
        # 等待退出，超时强杀
        for proc in self._children:
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass


# ── 组件启动 ──────────────────────────────────────────────────

def _python() -> str:
    """当前 Python 解释器。"""
    return sys.executable


def start_mcp_server(pm: ProcessManager) -> Optional[subprocess.Popen]:
    """启动 LAAP MCP server（SSE 模式，端口 8765）。"""
    if not MCP_SERVER_SCRIPT.exists():
        print(f"  [SKIP] MCP server 脚本不存在：{MCP_SERVER_SCRIPT}")
        return None
    cmd = [_python(), str(MCP_SERVER_SCRIPT), "--sse", "--port", str(PORT_MCP)]
    return pm.start_background(cmd, cwd=LAAP_ROOT, name="MCP server")


def start_sidecar(pm: ProcessManager) -> Optional[subprocess.Popen]:
    """启动 Aris sidecar（端口 11521，脚本内硬编码）。"""
    if not SIDECAR_SCRIPT.exists():
        print(f"  [SKIP] sidecar 脚本不存在：{SIDECAR_SCRIPT}")
        return None
    cmd = [_python(), str(SIDECAR_SCRIPT)]
    return pm.start_background(cmd, cwd=SIDECAR_SCRIPT.parent, name="Aris sidecar")


def start_hanako_desktop() -> int:
    """前台启动 Hanako 桌面端（npm run start）。

    返回退出码。
    """
    if not HANAKO_DIR.exists():
        print(f"  [FAIL] hanako 目录不存在：{HANAKO_DIR}")
        return 1
    print(f"  [INFO] 启动 Hanako 桌面端（前台）...")
    print("         按 Ctrl+C 退出会同时停止后台组件。")
    print()
    try:
        # 前台运行：继承 stdio，用户可直接交互
        return subprocess.call(
            ["npm", "run", "start"],
            cwd=str(HANAKO_DIR),
            env=RUNTIME_ENV,
            shell=os.name == "nt",
        )
    except KeyboardInterrupt:
        print("\n  收到中断信号，正在退出...")
        return 0


def start_all() -> int:
    """启动全部组件：端口检测 -> MCP -> sidecar -> hanako desktop。"""
    print()
    print("=" * 56)
    print("  LAAP Hanako 启动")
    print("=" * 56)
    print()

    # 1. 端口冲突检测
    if not check_ports():
        return 1

    ensure_runtime_dirs()
    pm = ProcessManager()

    # 注册信号处理：CtrlC 时清理后台进程
    def _cleanup(signum=None, frame=None):
        pm.stop_all()
        sys.exit(0)
    signal.signal(signal.SIGINT, _cleanup)
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _cleanup)

    # 2. MCP server（后台）
    print()
    print("  [1/3] 启动 MCP server...")
    start_mcp_server(pm)
    time.sleep(1.0)

    # 3. sidecar（后台）
    print()
    print("  [2/3] 启动 Aris sidecar...")
    start_sidecar(pm)
    time.sleep(1.0)

    # 4. hanako desktop（前台）
    print()
    print("  [3/3] 启动 Hanako 桌面端...")
    print()
    rc = start_hanako_desktop()

    # 桌面端退出后清理后台
    pm.stop_all()
    return rc


def start_only(name: str) -> int:
    """仅启动单个组件。"""
    pm = ProcessManager()

    def _cleanup(signum=None, frame=None):
        pm.stop_all()
        sys.exit(0)
    signal.signal(signal.SIGINT, _cleanup)
    if os.name == "posix":
        signal.signal(signal.SIGTERM, _cleanup)

    if name == "mcp":
        print()
        print("  仅启动 MCP server（前台日志可见，Ctrl+C 退出）...")
        proc = start_mcp_server(pm)
        if proc is None:
            return 1
        proc.wait()
        return proc.returncode
    elif name == "sidecar":
        print()
        print("  仅启动 Aris sidecar（前台日志可见，Ctrl+C 退出）...")
        proc = start_sidecar(pm)
        if proc is None:
            return 1
        proc.wait()
        return proc.returncode
    else:
        print(f"  未知组件：{name}")
        return 1


# ── CLI ───────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LAAP Hanako 启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python installer/launch.py             启动全部组件
  python installer/launch.py --check     仅检测端口
  python installer/launch.py --sidecar   仅启动 sidecar
  python installer/launch.py --mcp       仅启动 MCP server
        """,
    )
    parser.add_argument("--check", action="store_true", help="仅检测端口冲突")
    parser.add_argument("--sidecar", action="store_true", help="仅启动 Aris sidecar")
    parser.add_argument("--mcp", action="store_true", help="仅启动 LAAP MCP server")
    args = parser.parse_args()

    if args.check:
        ok = check_ports()
        return 0 if ok else 1
    if args.sidecar:
        return start_only("sidecar")
    if args.mcp:
        return start_only("mcp")
    return start_all()


if __name__ == "__main__":
    sys.exit(main())
