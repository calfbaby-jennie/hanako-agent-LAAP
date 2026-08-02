#!/usr/bin/env python3
"""LAAP Hanako 环境依赖检测脚本.

检测：
- Python 3.11+
- Node.js >= 24.12.0, < 25
- Git
- uv 或 pip 可用性
- npm 可用性
- 端口 2668/11521/8765 是否被占用
- 环境变量 HANA_HOME / LAAP_HOME 是否已设置

用法：
    python installer/check_env.py          # 人类可读输出
    python installer/check_env.py --json   # JSON 输出（脚本用）
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from typing import Any, Dict, List

# ── 待检测端口（与 launch.py 保持一致）────────────────────────
# 2668  CognitiveBus / Hanako 后端
# 11521 Aris sidecar
# 8765  LAAP MCP server
REQUIRED_PORTS = [2668, 11521, 8765]

# ── 版本要求 ──────────────────────────────────────────────────
PYTHON_MIN = (3, 11)
NODE_MIN = (24, 12, 0)
NODE_MAX_MAJOR = 25


def _run(cmd: List[str], timeout: float = 10.0) -> "subprocess.CompletedProcess[str] | None":
    """安全地运行外部命令，失败返回 None。

    Windows 上 npm/pip 等可能是 .cmd 批处理文件，CreateProcess 无法直接定位，
    需要 shell=True 让 cmd.exe 按 PATHEXT 解析（与 launch.py 一致）。
    """
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=os.name == "nt",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def check_python() -> Dict[str, Any]:
    """检测 Python 版本（>= 3.11）。"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    ok = version >= PYTHON_MIN
    return {
        "name": "Python",
        "status": "ok" if ok else "fail",
        "detail": f"当前 {version_str}，要求 >= {PYTHON_MIN[0]}.{PYTHON_MIN[1]}",
        "version": version_str,
    }


def check_node() -> Dict[str, Any]:
    """检测 Node.js 版本（与 hanako/package.json engines 一致）。"""
    result = _run(["node", "--version"])
    if result is None or result.returncode != 0:
        return {
            "name": "Node.js",
            "status": "fail",
            "detail": "未检测到 Node.js，请安装 Node.js >= 24.12.0, < 25",
        }
    raw = (result.stdout or "").strip().lstrip("v")
    try:
        parsed = tuple(int(part) for part in raw.split("-", 1)[0].split("."))
        version = (parsed + (0, 0, 0))[:3]
    except (ValueError, IndexError):
        return {
            "name": "Node.js",
            "status": "fail",
            "detail": f"无法解析 Node.js 版本：{raw!r}",
        }
    ok = version >= NODE_MIN and version[0] < NODE_MAX_MAJOR
    return {
        "name": "Node.js",
        "status": "ok" if ok else "fail",
        "detail": f"当前 v{raw}，要求 >= v24.12.0, < v25",
        "version": raw,
    }


def check_git() -> Dict[str, Any]:
    """检测 Git 是否可用。"""
    result = _run(["git", "--version"])
    if result is None or result.returncode != 0:
        return {
            "name": "Git",
            "status": "warn",
            "detail": "未检测到 Git，部分功能（如版本管理、克隆）可能受限",
        }
    version = (result.stdout or "").strip()
    return {
        "name": "Git",
        "status": "ok",
        "detail": version,
    }


def check_python_package_manager() -> Dict[str, Any]:
    """检测 uv 或 pip 是否可用。

    ``uv venv`` 默认不在环境中安装 pip，但 ``uv pip`` 仍可以直接
    管理该环境，因此不应将“没有 pip”误报为安装失败。
    """
    uv_result = _run(["uv", "--version"])
    if uv_result is not None and uv_result.returncode == 0:
        version = (uv_result.stdout or "").strip().split("\n")[0]
        return {
            "name": "Python 包管理器",
            "status": "ok",
            "detail": version,
            "provider": "uv",
        }

    result = _run([sys.executable, "-m", "pip", "--version"])
    if result is None or result.returncode != 0:
        # 退而求其次：直接调用 pip
        result = _run(["pip", "--version"])
    if result is None or result.returncode != 0:
        return {
            "name": "Python 包管理器",
            "status": "fail",
            "detail": "uv 和 pip 均不可用，无法安装 Python 依赖",
        }
    version = (result.stdout or "").strip().split("\n")[0]
    return {
        "name": "Python 包管理器",
        "status": "ok",
        "detail": version,
        "provider": "pip",
    }


def check_npm() -> Dict[str, Any]:
    """检测 npm 是否可用。"""
    result = _run(["npm", "--version"])
    if result is None or result.returncode != 0:
        return {
            "name": "npm",
            "status": "fail",
            "detail": "npm 不可用，无法安装 Hanako 依赖",
        }
    version = (result.stdout or "").strip().split("\n")[0]
    return {
        "name": "npm",
        "status": "ok",
        "detail": f"npm {version}",
        "version": version,
    }


def _port_in_use(port: int) -> bool:
    """尝试 bind 端口，bind 失败说明端口被占用。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(("127.0.0.1", port))
    except (OSError, socket.error):
        return True
    finally:
        sock.close()
    return False


def check_ports() -> List[Dict[str, Any]]:
    """检测三个关键端口是否空闲。"""
    results = []
    for port in REQUIRED_PORTS:
        in_use = _port_in_use(port)
        results.append({
            "name": f"端口 {port}",
            "status": "fail" if in_use else "ok",
            "port": port,
            "detail": "被占用，请释放后重试" if in_use else "空闲",
        })
    return results


def check_env_vars() -> List[Dict[str, Any]]:
    """检测 HANA_HOME / LAAP_HOME 环境变量。"""
    results = []
    for var in (
        "LAAP_ROOT",
        "HANA_HOME",
        "LAAP_HOME",
        "LAAP_STATE_DIR",
        "LAAP_CACHE_DIR",
        "LAAP_LOGS_DIR",
        "ARIS_BRAIN_DIR",
    ):
        value = os.environ.get(var, "").strip()
        if value:
            results.append({
                "name": var,
                "status": "ok",
                "detail": f"已设置：{value}",
            })
        else:
            results.append({
                "name": var,
                "status": "warn",
                "detail": "未设置，安装脚本将自动写入默认值",
            })
    return results


def run_all_checks() -> Dict[str, Any]:
    """运行全部检测，返回结构化报告。"""
    port_results = check_ports()
    env_results = check_env_vars()
    items = [
        check_python(),
        check_node(),
        check_git(),
        check_python_package_manager(),
        check_npm(),
        *port_results,
        *env_results,
    ]
    # 总体判定：fail 项决定退出码；warn 不影响
    has_fail = any(item["status"] == "fail" for item in items)
    return {
        "ok": not has_fail,
        "items": items,
    }


def _print_human(report: Dict[str, Any]) -> None:
    """人类可读输出。"""
    print("=" * 56)
    print("  LAAP Hanako 环境依赖检测")
    print("=" * 56)
    for item in report["items"]:
        status = item["status"].upper()
        marker = {"OK": "[OK]  ", "FAIL": "[FAIL]", "WARN": "[WARN]"}.get(status, "[ -- ]")
        print(f"  {marker} {item['name']:<14} {item['detail']}")
    print("=" * 56)
    if report["ok"]:
        print("  环境检测通过。")
    else:
        print("  环境检测未通过，请修复 FAIL 项后重试。")
    print("=" * 56)


def main() -> int:
    parser = argparse.ArgumentParser(description="LAAP Hanako 环境依赖检测")
    parser.add_argument("--json", action="store_true", help="输出 JSON（供脚本解析）")
    args = parser.parse_args()

    report = run_all_checks()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
