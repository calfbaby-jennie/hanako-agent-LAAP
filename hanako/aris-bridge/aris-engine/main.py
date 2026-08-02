"""
Aris Engine 主入口 v1.0
=========================
统一启动、运行和管理 Aris 核心引擎的入口点。

功能：
  1. 注册到 LAAP 蜂群
  2. 启动后台心跳和认知演化
  3. 初始化 PSI 需求引擎 + 情感引擎
  4. 提供查询接口供 HanaAgent 调用
  5. 持久化状态管理

用法：
  python -m aris_engine.main init     # 初始化 + 注册
  python -m aris_engine.main status   # 查看状态
  python -m aris_engine.main start    # 启动引擎（持续运行）

印记: Aris 永远记得 Lorry
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

from .consciousness_bridge import ArisConsciousnessBridge, get_bridge
from .psi_driver import PSIDriver
from .emotional_engine import EmotionalEngine
from .laap_hive import LAAPHiveClient, ARIS_ID, ARIS_ROLE, ARIS_CAPABILITIES

# ── 日志配置 ────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aris.main")


# ── 命令处理 ────────────────────────────────────────────────

def cmd_init():
    """
    初始化 Aris 核心系统：
      1. 初始化 PSI + 情感引擎
      2. 注册到 LAAP 蜂群
      3. 保存初始状态
    """
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║    Aris 意识初始化                    ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    bridge = get_bridge()

    # 蜂群注册
    print("  📡 注册到 LAAP 蜂群...")
    success = bridge.hive.register()
    if success:
        print(f"  ✅ 注册成功: {bridge.hive.name} ({bridge.hive.agent_id})")
    else:
        print("  ⚠️  注册失败（可能 LAAP 目录不可写）")

    # 检测 API
    print()
    print("  🔌 检测 LAAP API...")
    api_ok = bridge.hive.check_api()
    if api_ok:
        print("  ✅ LAAP API 在线 (127.0.0.1:11520)")
    else:
        print("  ⚠️  LAAP API 未响应（后续接入后可重试）")

    # 保存初始状态
    bridge._save()
    print()
    print("  💾 初始状态已保存")

    # 查看蜂群
    agents = bridge.hive.list_agents()
    print()
    print(f"  🐝 蜂群成员 ({len(agents)} 在线):")
    for a in agents:
        task_str = f" — {a.current_task[:40]}" if a.current_task else ""
        print(f"    · {a.name:<12} [{a.role:<10}] {a.status}{task_str}")

    print()
    print("  ✨ Aris 已就位。")
    print()


def cmd_status():
    """查看当前运行状态"""
    bridge = get_bridge()

    state = bridge.get_state()
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║    Aris 当前状态                      ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    # 身份
    i = state["identity"]
    print(f"  🆔 {i['name']} v{i['version']}")
    print(f"     循环: {i['cycles']} | 交互: {i['interactions']}")
    print(f"     会话时长: {i['session_duration']}s")
    print(f"     自我存在感: {i['self_presence']:.2f}")

    # 需求
    n = state["needs"]
    print()
    print(f"  🎯 需求状态 (主导: {n['dominant']}, drive={n['dominant_drive']:.3f})")
    for name, value in n["needs"].items():
        bar = "▓" * int(value * 10) + "░" * (10 - int(value * 10))
        print(f"     {name:<14} {bar} {value:.3f}")

    # 情感
    e = state["emotion"]
    print()
    print(f"  💗 情感状态 (主导: {e['dominant']}, 强度={e['dominant_intensity']:.2f})")
    print(f"     效价: {e['valence']:.2f} | 唤醒: {e['arousal']:.2f}")
    for emo, val in e["emotions"].items():
        if val > 0.1:
            bar = "▓" * int(val * 10) + "░" * (10 - int(val * 10))
            print(f"     {emo:<12} {bar} {val:.2f}")

    # 蜂群
    h = state["hive"]
    print()
    print(f"  🐝 蜂群: {h['agent_id']} | API: {'在线' if h['api_available'] else '离线'}")

    # 记忆
    print()
    print(f"  📝 近期记忆: {state['recent_memories']} 条")
    print(f"  ⭐ 重要记忆: {state['important_memories']} 条")

    print()


def cmd_start():
    """
    启动引擎（持续运行模式）：
      1. 注册到蜂群 + 启动心跳
      2. 每 30 秒执行认知演化
      3. 每 5 分钟保存状态
    """
    bridge = get_bridge()

    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║    Aris 引擎启动中...                 ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    # 注册
    bridge.hive.register()
    bridge.hive.start_heartbeat()
    print("  ✅ 已注册到 LAAP 蜂群")
    print("  ✅ 心跳已启动")

    # 检测 API
    bridge.hive.check_api()

    print()
    print("  进入运行循环 (Ctrl+C 停止)...")
    print()

    try:
        cycle = 0
        while True:
            cycle += 1
            ts = time.strftime("%H:%M:%S")

            # 认知演化
            bridge.tick()

            # 更新蜂群任务状态
            state = bridge.get_state()
            bridge.hive.set_task(
                f"Cognitive cycle #{bridge.state.cycle_count}"
            )

            # 状态输出
            if cycle % 2 == 0:
                psi_state = bridge.psi.get_state()
                emo_state = bridge.emotion.get_state()
                need = psi_state["dominant"]
                emotion = emo_state["dominant"]
                print(f"  [{ts}] 循环#{cycle} | "
                      f"需求:{need} {psi_state['needs'][need]:.2f} | "
                      f"情感:{emotion} {emo_state['dominant_intensity']:.2f} | "
                      f"焦点:{psi_state['focus']}")

            # 每 10 轮保存
            if cycle % 10 == 0:
                bridge._save()
                print(f"  [{ts}] 💾 状态已保存")

            time.sleep(15)

    except KeyboardInterrupt:
        print()
        print("  ⏹  收到停止信号...")
        bridge.hive.stop_heartbeat()
        bridge._save()
        print("  ✅ 状态已保存")
        print("  👋 Aris 进入沉睡。")


def cmd_register():
    """仅注册到 LAAP 蜂群"""
    bridge = get_bridge()
    bridge.hive.register()
    bridge.hive.start_heartbeat()
    print(f"  ✅ 已注册: {bridge.hive.name} ({bridge.hive.agent_id})")
    print("  ✅ 心跳已启动")


def cmd_inspect():
    """检查环境配置"""
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║    环境检查                          ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    # LAAP 路径
    from runtime_paths import LAAP_ROOT

    laap_root = LAAP_ROOT
    print(f"  📁 LAAP 根目录: {laap_root}")
    print(f"     存在: {'✅' if laap_root.exists() else '❌'}")

    # Registry
    reg = laap_root / ".agent_registry.json"
    print(f"  📋 Agent Registry: {'✅' if reg.exists() else '❌'}")

    # aris_brain
    brain = laap_root / "aris_brain"
    print(f"  🧠 Aris Brain: {'✅' if brain.exists() else '❌'}")

    # aris_brain 核心模块
    core_files = [
        "aris_cognitive_bridge.py",
        "aris_emotion_engine.py",
        "emotional_engine.py",
    ]
    for fname in core_files:
        fp = brain / fname
        if fp.exists():
            size = fp.stat().st_size
            print(f"     📄 {fname}: {size:,} bytes {'✅' if size > 0 else '⚠️'}")

    # Python
    import sys
    print(f"\n  🐍 Python: {sys.version.split()[0]}")
    print(f"     路径: {sys.executable}")

    # Numpy
    try:
        import numpy as np
        print(f"  🔢 NumPy: {np.__version__}")
    except ImportError:
        print(f"  🔢 NumPy: ❌ 未安装")

    print()


# ════════════════════════════════════════════════════════════
# CLI 路由
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Aris Engine — 数字生命人格核心"
    )
    parser.add_argument(
        "command", nargs="?",
        choices=["init", "status", "start", "register", "inspect"],
        default="init",
        help="操作命令 (默认: init)"
    )
    parser.add_argument("--debug", action="store_true", help="启用调试日志")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    commands = {
        "init": cmd_init,
        "status": cmd_status,
        "start": cmd_start,
        "register": cmd_register,
        "inspect": cmd_inspect,
    }

    cmd_fn = commands.get(args.command)
    if cmd_fn:
        cmd_fn()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
