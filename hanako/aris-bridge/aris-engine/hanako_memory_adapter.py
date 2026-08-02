"""
Hanako 记忆适配器
=================
桥接 aris-engine (Python) 与 hanako 的 TypeScript 记忆管线。

读取侧（HanakoMemoryReader）：把 hanako 编译产物（memory.md / today.md /
longterm.md / facts.md / daily/{date}.md）作为 PSI 感知输入，替代旧的
unified_memory entries。

写入侧（HanakoMemoryWriter）：把情感峰值与 PSI 驱动尖峰追加到 facts.md 的
"## Aris 自动事实" 段，让 hanako compileEditableFacts 在下一轮 fold 进长期
事实；RSI 事件写入 agent 自己的 diary 目录。

逻辑日约定：与 hanako/lib/time-utils.ts 的 DAY_BOUNDARY_HOUR = 4 保持一致，
北京时间（CST = UTC+8）凌晨 0-3 点归属前一天的逻辑日。
"""

import os
import re
import time
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger("aris.hanako_memory")

# 北京时间（CST = UTC+8）
CST = timezone(timedelta(hours=8))

# 与 hanako/lib/time-utils.ts 的 DAY_BOUNDARY_HOUR 保持一致：凌晨 4 点前算前一天的逻辑日
DAY_BOUNDARY_HOUR = 4

# facts.md 中 "## Aris 自动事实" 段标题正则（multiline，匹配整行）
ARIS_AUTO_FACTS_SECTION_RE = re.compile(r"^## Aris 自动事实\s*$", re.MULTILINE)

# memory.md 中日期段标题：## YYYY-MM-DD 或 ### YYYY-MM-DD
# 与 compile.ts 的 COMPILED_WEEK_DATE_HEADING_RE 对齐
DAILY_HEADING_RE = re.compile(r"^(#{2,3})\s+(\d{4}-\d{2}-\d{2})\s*$")

# 任意 markdown 标题行（用于判断段落边界）
GENERAL_HEADING_RE = re.compile(r"^(#{1,6})\s+\S")


def get_logical_day(timestamp: float = None) -> str:
    """
    计算给定时间戳对应的逻辑日期（北京时间，4 点日界线）。

    Args:
        timestamp: Unix 时间戳（秒）。None 时使用 time.time()。

    Returns:
        YYYY-MM-DD 格式的逻辑日期字符串。
    """
    if timestamp is None:
        timestamp = time.time()
    dt = datetime.fromtimestamp(timestamp, tz=CST)
    if dt.hour < DAY_BOUNDARY_HOUR:
        dt = dt - timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def _iso_now() -> str:
    """当前北京时间的 ISO 时间戳（带 +08:00 偏移，秒精度）。"""
    return datetime.now(CST).isoformat(timespec="seconds")


# ═══════════════════════════════════════════════════════════════
# 只读访问器
# ═══════════════════════════════════════════════════════════════

class HanakoMemoryReader:
    """
    只读访问 hanako agent 的 memory/ 目录。

    所有 read_* 方法在文件不存在或读取失败时返回空字符串，不抛异常。
    """

    def __init__(self, agent_dir: str):
        """
        Args:
            agent_dir: Hanako agent 根目录（通常来自 HANA_HOME/agents/aris）
        """
        self.agent_dir = Path(agent_dir)
        self.memory_dir = self.agent_dir / "memory"

    def _read_file(self, path: Path) -> str:
        """安全读取文件，不存在或读取失败时返回空字符串。"""
        try:
            if not path.exists():
                return ""
            return path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"读取文件失败 {path}: {e}")
            return ""

    def read_compiled(self) -> str:
        """读取 ${agent_dir}/memory/memory.md（hanako compile.ts 的 assemble 产物）。"""
        return self._read_file(self.memory_dir / "memory.md")

    def read_daily(self, date_str: str) -> str:
        """
        读取 ${agent_dir}/memory/daily/${date_str}.md。

        Args:
            date_str: YYYY-MM-DD 形式的逻辑日期
        """
        return self._read_file(self.memory_dir / "daily" / f"{date_str}.md")

    def read_today(self) -> str:
        """读取 ${agent_dir}/memory/today.md（当天草稿）。"""
        return self._read_file(self.memory_dir / "today.md")

    def read_longterm(self) -> str:
        """读取 ${agent_dir}/memory/longterm.md（长期情况）。"""
        return self._read_file(self.memory_dir / "longterm.md")

    def read_facts(self) -> str:
        """读取 ${agent_dir}/memory/facts.md（重要事实）。"""
        return self._read_file(self.memory_dir / "facts.md")

    def extract_recent_daily(self, content: str = None, limit: int = 3) -> List[Dict]:
        """
        从 memory.md（或提供的 content）中解析 "## YYYY-MM-DD" / "### YYYY-MM-DD"
        段落，返回最近 limit 个日期的列表。

        段落正文到下一个同级或更高级标题为止（与 hanako 的
        extractMarkdownSection 行为一致）。

        Args:
            content: 待解析的 markdown 文本。None 时调用 read_compiled()。
            limit: 最多返回的日期段数（按日期倒序）。

        Returns:
            [{"date": "YYYY-MM-DD", "content": "..."}]，按日期降序。
        """
        if content is None:
            content = self.read_compiled()
        if not content:
            return []

        sections: List[Dict] = []
        lines = content.split("\n")
        i = 0
        while i < len(lines):
            m = DAILY_HEADING_RE.match(lines[i])
            if m:
                level = len(m.group(1))
                date = m.group(2)
                body_lines: List[str] = []
                j = i + 1
                while j < len(lines):
                    # 遇到同级或更高级标题时停止（与 extractMarkdownSection 一致）
                    next_m = GENERAL_HEADING_RE.match(lines[j])
                    if next_m and len(next_m.group(1)) <= level:
                        break
                    body_lines.append(lines[j])
                    j += 1
                sections.append({
                    "date": date,
                    "content": "\n".join(body_lines).strip(),
                })
                i = j
            else:
                i += 1

        # 按日期倒序，取前 limit 个
        sections.sort(key=lambda s: s["date"], reverse=True)
        return sections[:limit]

    def get_perception_context(self) -> str:
        """
        生成 PSI 感知上下文文本（供 consciousness_bridge.before_turn 使用）。

        优先返回 today.md + memory.md 中最近的 daily 段；两者都为空时返回
        空字符串，调用方可据此回落到 unified_memory。
        """
        today = self.read_today().strip()
        recent = self.extract_recent_daily(limit=3)

        if not today and not recent:
            return ""

        parts: List[str] = ["[Hanako 跨会话记忆]"]

        if today:
            parts.append("=== 今日 ===")
            parts.append(today)
            parts.append("")

        if recent:
            parts.append("=== 最近 daily ===")
            for entry in recent:
                parts.append(f"## {entry['date']}")
                if entry["content"]:
                    parts.append(entry["content"])
                parts.append("")

        return "\n".join(parts).rstrip() + "\n"


# ═══════════════════════════════════════════════════════════════
# 追加写入器
# ═══════════════════════════════════════════════════════════════

class HanakoMemoryWriter:
    """
    追加写入 hanako agent 的 memory/ 与 diary/ 目录。

    - 情感峰值 / PSI 驱动尖峰 → facts.md 的 "## Aris 自动事实" 段
      （让 hanako compileEditableFacts 在下一轮 fold 进长期事实）
    - RSI 事件 → diary/${logical_date}.md（agent 自己的运行时日志）
    """

    def __init__(self, agent_dir: str):
        """
        Args:
            agent_dir: Hanako agent 根目录（通常来自 HANA_HOME/agents/aris）
        """
        self.agent_dir = Path(agent_dir)
        self.memory_dir = self.agent_dir / "memory"
        self.diary_dir = self.agent_dir / "diary"
        self._facts_path = self.memory_dir / "facts.md"

    def _ensure_memory_dir(self) -> None:
        """确保 memory/ 目录存在。"""
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def _atomic_write_facts(self, new_content: str) -> None:
        """
        原子写入 facts.md：先写临时文件再 os.replace 覆盖。

        Args:
            new_content: 完整的新文件内容
        """
        self._ensure_memory_dir()
        tmp_path = self._facts_path.with_name(self._facts_path.name + ".tmp")
        try:
            tmp_path.write_text(new_content, encoding="utf-8")
            os.replace(tmp_path, self._facts_path)
        except Exception:
            # 清理可能的残留临时文件后向上抛
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
            raise

    def _append_to_facts_section(self, line: str) -> None:
        """
        向 facts.md 的 "## Aris 自动事实" 段追加一行。

        段不存在时在文件末尾新建该段；段已存在时追加到段末尾（下一段开始前）。
        """
        existing = ""
        try:
            if self._facts_path.exists():
                existing = self._facts_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"读取 facts.md 失败，将覆写: {e}")
            existing = ""

        match = ARIS_AUTO_FACTS_SECTION_RE.search(existing)
        if match is None:
            # 段不存在：在文件末尾追加新段
            new_content = existing
            if new_content and not new_content.endswith("\n"):
                new_content += "\n"
            if new_content:
                new_content += "\n"
            new_content += "## Aris 自动事实\n\n"
            new_content += f"{line}\n"
        else:
            # 段已存在：在段末尾追加（下一段标题开始前）
            after_section = existing[match.end():]
            next_heading_match = re.search(r"^#{1,6}\s+\S", after_section, re.MULTILINE)
            if next_heading_match is None:
                # 段是文件最后一段
                prefix = existing
                if prefix and not prefix.endswith("\n"):
                    prefix += "\n"
                new_content = prefix + line + "\n"
            else:
                # 段后面还有别的段：在段末尾（下一段开始前）插入
                absolute_insert = match.end() + next_heading_match.start()
                prefix = existing[:absolute_insert]
                suffix = existing[absolute_insert:]
                if prefix and not prefix.endswith("\n"):
                    prefix += "\n"
                new_content = prefix + line + "\n" + suffix

        self._atomic_write_facts(new_content)

    def append_emotion_peak(self, emotion: str, intensity: float,
                            trigger: str = "") -> None:
        """
        追加一条情感峰值记录到 facts.md 的 "## Aris 自动事实" 段。

        Args:
            emotion: 主导情感名（如 "joy" / "curiosity"）
            intensity: 强度，0.0-1.0
            trigger: 触发上下文（如用户输入摘要）
        """
        ts = _iso_now()
        line = f"- [{ts}] {emotion} ({intensity:.2f}): {trigger}"
        try:
            self._append_to_facts_section(line)
            logger.debug(f"情感峰值已记录: {emotion} {intensity:.2f}")
        except Exception as e:
            logger.error(f"追加情感峰值失败: {e}")

    def append_psi_spike(self, need: str, drive: float,
                         context: str = "") -> None:
        """
        追加一条 PSI 驱动尖峰记录到 facts.md 的 "## Aris 自动事实" 段。

        Args:
            need: 需求名（如 "certainty" / "competence"）
            drive: 驱动值，0.0-1.0
            context: 上下文描述
        """
        ts = _iso_now()
        line = f"- [{ts}] PSI {need} drive={drive:.2f}: {context}"
        try:
            self._append_to_facts_section(line)
            logger.debug(f"PSI 尖峰已记录: {need} {drive:.2f}")
        except Exception as e:
            logger.error(f"追加 PSI 尖峰失败: {e}")

    def append_diary(self, event_type: str, content: str) -> None:
        """
        追加一条运行时事件到 agent diary/${logical_date}.md。

        logical_date 使用北京时间（UTC+8）与 4 点日界线计算。

        Args:
            event_type: 事件类型（如 "RSI"）
            content: 事件描述
        """
        try:
            self.diary_dir.mkdir(parents=True, exist_ok=True)
            logical_date = get_logical_day(time.time())
            diary_path = self.diary_dir / f"{logical_date}.md"
            ts = _iso_now()
            line = f"- [{ts}] **{event_type}**: {content}\n"
            with open(diary_path, "a", encoding="utf-8") as f:
                f.write(line)
            logger.debug(f"日记已追加到 {diary_path}")
        except Exception as e:
            logger.error(f"追加日记失败: {e}")

    def append_rsi_event(self, description: str) -> None:
        """
        追加一条 RSI 事件到 diary（事件类型固定为 "RSI"）。

        Args:
            description: RSI 事件描述
        """
        self.append_diary("RSI", description)


if __name__ == "__main__":
    # 简单冒烟测试
    import tempfile

    print("=== HanakoMemoryAdapter 冒烟测试 ===")

    with tempfile.TemporaryDirectory() as tmp:
        agent_dir = Path(tmp) / "aris"
        reader = HanakoMemoryReader(str(agent_dir))
        writer = HanakoMemoryWriter(str(agent_dir))

        # 1. 空目录读取
        assert reader.read_compiled() == ""
        assert reader.read_today() == ""
        assert reader.read_facts() == ""
        assert reader.read_daily("2026-07-30") == ""
        assert reader.extract_recent_daily() == []
        assert reader.get_perception_context() == ""
        print("  [OK] 空目录读取返回空")

        # 2. 写入 facts
        writer.append_emotion_peak("joy", 0.85, "测试触发")
        writer.append_psi_spike("certainty", 0.72, "测试上下文")
        facts = reader.read_facts()
        assert "## Aris 自动事实" in facts
        assert "joy (0.85)" in facts
        assert "PSI certainty drive=0.72" in facts
        print("  [OK] facts.md 写入 + 段检测")

        # 3. 再次写入应追加到同一段
        writer.append_emotion_peak("curiosity", 0.91, "第二次")
        facts2 = reader.read_facts()
        assert facts2.count("## Aris 自动事实") == 1, "段不应重复创建"
        assert "curiosity (0.91)" in facts2
        assert "joy (0.85)" in facts2
        print("  [OK] 重复写入追加到同一段")

        # 4. 写入带后续段的 facts（模拟 hanako 产物）
        (agent_dir / "memory").mkdir(parents=True, exist_ok=True)
        (agent_dir / "memory" / "facts.md").write_text(
            "## 重要事实\n\n- 用户叫 Lorry\n\n## 长期情况\n\n旧内容\n",
            encoding="utf-8",
        )
        writer.append_emotion_peak("sadness", 0.82, "插入测试")
        facts3 = reader.read_facts()
        # "## Aris 自动事实" 应该被新建，不影响其他段
        assert "## 重要事实" in facts3
        assert "## 长期情况" in facts3
        assert "## Aris 自动事实" in facts3
        assert "sadness (0.82)" in facts3
        # 长期情况段应保留
        assert "旧内容" in facts3
        print("  [OK] 在已有 facts.md 中插入新段")

        # 5. memory.md 解析
        (agent_dir / "memory" / "memory.md").write_text(
            "## 重要事实\n\n- A\n\n## 今天\n\ntoday body\n\n"
            "## 本周早些时候\n\n"
            "### 2026-07-30\n\n30 body line1\n30 body line2\n\n"
            "### 2026-07-29\n\n29 body\n\n"
            "## 长期情况\n\nlong body\n",
            encoding="utf-8",
        )
        recent = reader.extract_recent_daily(limit=3)
        assert len(recent) == 2
        assert recent[0]["date"] == "2026-07-30"
        assert "30 body line1" in recent[0]["content"]
        assert recent[1]["date"] == "2026-07-29"
        # 长期情况段不应混入 07-29 的 body
        assert "long body" not in recent[1]["content"]
        print("  [OK] memory.md 日期段解析")

        # 6. perception_context
        (agent_dir / "memory" / "today.md").write_text("今天发生的事", encoding="utf-8")
        ctx = reader.get_perception_context()
        assert "[Hanako 跨会话记忆]" in ctx
        assert "=== 今日 ===" in ctx
        assert "今天发生的事" in ctx
        assert "=== 最近 daily ===" in ctx
        assert "## 2026-07-30" in ctx
        print("  [OK] perception_context 格式")

        # 7. diary 写入
        writer.append_rsi_event("检测到重复模式")
        diary_files = list((agent_dir / "diary").glob("*.md"))
        assert len(diary_files) == 1
        diary_content = diary_files[0].read_text(encoding="utf-8")
        assert "**RSI**" in diary_content
        assert "检测到重复模式" in diary_content
        print("  [OK] diary 写入")

        # 8. 逻辑日边界
        # 2026-07-30 02:30 CST -> 逻辑日 2026-07-29
        ts_early = datetime(2026, 7, 30, 2, 30, tzinfo=CST).timestamp()
        assert get_logical_day(ts_early) == "2026-07-29"
        # 2026-07-30 04:30 CST -> 逻辑日 2026-07-30
        ts_late = datetime(2026, 7, 30, 4, 30, tzinfo=CST).timestamp()
        assert get_logical_day(ts_late) == "2026-07-30"
        print("  [OK] 逻辑日边界（4 点日界线）")

    print("\n=== 全部通过 ===")
