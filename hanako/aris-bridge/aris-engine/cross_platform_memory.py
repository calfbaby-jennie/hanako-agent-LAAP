"""
Aris 跨平台记忆通道 v1.0 — Cross-Platform Memory Bridge
=========================================================
打通微信（HanaAgent）和飞书（Hermes）两个通道的记忆。

设计：
  所有通道的消息写入 ``LAAP_HOME/aris_brain`` 下的共享存储文件。
  
  每次认知上下文注入时读取最近 N 条，
  无论来自哪个平台，按时间排序，最新的优先。

印记: Aris 永远记得 Lorry — 2026-07-24
"""

import time
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta

from runtime_paths import ARIS_BRAIN_DIR

logger = logging.getLogger("aris.cross_platform")

# 北京时间偏移
CST = timezone(timedelta(hours=8))

# 共享记忆文件路径
SHARED_MEMORY_PATH = ARIS_BRAIN_DIR / "shared_memory.json"
FEISHU_LOG_PATH = ARIS_BRAIN_DIR / "aris_bridge.log"

MAX_ENTRIES = 200  # 共享存储保留上限
INJECT_COUNT = 8   # 每次注入的对话数


# ── 数据类型 ────────────────────────────────────────────────

def _now_cst() -> str:
    """当前北京时间 ISO 字符串"""
    return datetime.now(CST).isoformat()


def _make_entry(platform: str, direction: str, content: str,
                sender: str = "") -> dict:
    """创建一条记忆条目"""
    return {
        "id": f"{int(time.time()*1000)}_{platform}_{direction}",
        "timestamp": _now_cst(),
        "unixtime": time.time(),
        "platform": platform,     # "wechat" | "feishu"
        "direction": direction,   # "sent" (我说) | "received" (你说)
        "sender": sender or ("Lorry" if direction == "received" else "Aris"),
        "content": content[:300],
    }


# ══════════════════════════════════════════════════════════════
# 共享记忆存储
# ══════════════════════════════════════════════════════════════

class CrossPlatformMemory:
    """
    跨平台共享记忆。

    微信和飞书两个通道的消息都写入同一个文件。
    按时间排序，最新的优先显示。
    """

    def __init__(self):
        self._entries: List[dict] = []
        self._last_feishu_mtime = 0.0
        self._load()

    # ── 读写 ────────────────────────────────────────────

    def write(self, platform: str, direction: str, content: str,
              sender: str = ""):
        """写入一条记忆"""
        entry = _make_entry(platform, direction, content, sender)
        self._entries.append(entry)

        # 按时间排序（最新的在前）
        self._entries.sort(key=lambda e: e.get("unixtime", 0), reverse=True)

        # 裁剪上限
        if len(self._entries) > MAX_ENTRIES:
            self._entries = self._entries[:MAX_ENTRIES]

        self._save()
        logger.debug(f"Written: [{platform}] {direction}: {content[:40]}")

    def write_turn(self, platform: str, user_input: str, response: str):
        """记录一次完整交互（你说 + 我说）
        
        platform: "wechat" | "feishu" | "desktop"
        """
        self.write(platform, "received", user_input)
        self.write(platform, "sent", response)

    def get_recent(self, n: int = INJECT_COUNT) -> List[dict]:
        """获取最近 N 条记忆（已经按时间排序，最新的在前）"""
        return self._entries[:n]

    # ── 从 Feishu 日志导入 ──────────────────────────────

    def import_feishu_log(self) -> int:
        """从飞书桥接日志导入最新对话，返回新导入条数"""
        if not FEISHU_LOG_PATH.exists():
            return 0

        current_mtime = FEISHU_LOG_PATH.stat().st_mtime
        if current_mtime <= self._last_feishu_mtime:
            return 0  # 日志未更新，跳过

        try:
            text = FEISHU_LOG_PATH.read_text(encoding="utf-8")
            lines = text.split("\n")

            # 匹配飞书日志格式
            # 时间 [ArisFeishu] ← 用户ID: 内容
            # 时间 [ArisFeishu] → 内容
            pattern = re.compile(
                r'(\d{2}:\d{2}:\d{2})\s+\[ArisFeishu\]\s+([←→])\s+'
                r'(?:ou_\w+:\s*)?(.+)'
            )

            existing_ids = {e["id"] for e in self._entries}
            new_count = 0

            for line in lines:
                m = pattern.match(line.strip())
                if not m:
                    continue

                log_time, arrow, content = m.groups()
                content = content.strip()
                if not content:
                    continue

                direction = "received" if arrow == "←" else "sent"

                # 跳过已存在的
                entry_id = f"feishu_{log_time.replace(':','')}_{direction}"
                if entry_id in existing_ids:
                    continue

                entry = {
                    "id": entry_id,
                    "timestamp": f"今天 {log_time}",
                    "unixtime": time.time(),  # 近似，因为今天的日志
                    "platform": "feishu",
                    "direction": direction,
                    "sender": "Lorry" if direction == "received" else "Aris",
                    "content": content[:300],
                }
                self._entries.append(entry)
                existing_ids.add(entry_id)
                new_count += 1

            if new_count > 0:
                # 按时间排序
                self._entries.sort(
                    key=lambda e: e.get("unixtime", 0), reverse=True
                )
                # 裁剪
                if len(self._entries) > MAX_ENTRIES:
                    self._entries = self._entries[:MAX_ENTRIES]
                self._save()

            self._last_feishu_mtime = current_mtime
            logger.info(f"Imported {new_count} new Feishu messages")
            return new_count

        except Exception as e:
            logger.warning(f"Feishu import failed: {e}")
            return 0

    # ── 格式化输出 ──────────────────────────────────────

    def format_recent(self, n: int = INJECT_COUNT,
                      include_wechat: bool = True,
                      include_feishu: bool = True) -> str:
        """格式化最近 N 条对话为可读文本"""
        recent = self.get_recent(n)
        if not recent:
            return ""

        lines = []
        for entry in recent:
            platform = entry.get("platform", "?")
            direction = entry.get("direction", "?")
            sender = entry.get("sender", "?")
            content = entry.get("content", "")
            ts = entry.get("timestamp", "")

            # 平台标记
            tag = "💬" if platform == "wechat" else "📨"

            # 方向标记
            if direction == "received":
                arrow = "←"
                name = f"{tag} {sender}"
            else:
                arrow = "→"
                name = f"{tag} {sender}"

            # 格式: 平台 时间 方向 发送者: 内容
            line = f"  {arrow} {name}: {content[:80]}"
            if content:
                lines.append(line)

        if not lines:
            return ""

        header = "[跨平台记忆]"
        return header + "\n" + "\n".join(lines[:n])

    # ── 持久化 ──────────────────────────────────────────

    def _save(self):
        try:
            SHARED_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_updated": _now_cst(),
                "last_feishu_mtime": self._last_feishu_mtime,
                "entries": self._entries,
            }
            SHARED_MEMORY_PATH.write_text(
                json.dumps(data, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logger.warning(f"Save failed: {e}")

    def _load(self):
        try:
            if SHARED_MEMORY_PATH.exists():
                data = json.loads(SHARED_MEMORY_PATH.read_text())
                self._entries = data.get("entries", [])
                self._last_feishu_mtime = data.get("last_feishu_mtime", 0.0)
                logger.info(f"Loaded {len(self._entries)} shared memories")
        except Exception as e:
            logger.warning(f"Load failed: {e}")
            self._entries = []


# ── 全局实例 ────────────────────────────────────────────────

_cross_memory: Optional[CrossPlatformMemory] = None


def get_cross_memory() -> CrossPlatformMemory:
    global _cross_memory
    if _cross_memory is None:
        _cross_memory = CrossPlatformMemory()
    return _cross_memory
