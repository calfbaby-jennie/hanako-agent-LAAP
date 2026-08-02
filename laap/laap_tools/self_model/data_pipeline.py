"""
self_model_nn — Data Pipeline (训练数据收集管道)
================================================

核心任务：从 Hermes 对话日志和 CognitiveBus 状态中提取
(state_before, context, state_after) 三元组，作为训练数据。

数据来源:
  1. Hermes before_turn / after_turn 钩子日志
  2. Hermes session history (request_dump JSON 文件)
  3. 模拟生成 (用 CognitiveBus 随机状态变化)

每条训练样本格式:
{
    "cb_state_before": {...},       # 对话开始时的认知状态
    "cb_state_after": {...},        # 对话结束时的认知状态
    "attention_delta": [...],       # 注意力变化
    "emotion_delta": {...},         # 情感变化
    "needs_delta": {...},           # 需求变化
    "self_presence_delta": float,   # 自我存在感变化
    "dialogue_summary": str,        # 对话摘要
    "turns": int,                   # 对话轮次
    "timestamp": float,             # 时间戳
    "conversation_id": str          # 对话 ID
}
"""

from __future__ import annotations

import glob
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from laap.config.paths import get_aris_brain_dir, get_laap_home

logger = logging.getLogger("laap.self_model.data_pipeline")

# ── 路径常量 ───────────────────────────────────────────────────
_ARIS_BRAIN_DIR = str(get_aris_brain_dir() / "self_model")
_DATA_DIR = os.path.join(_ARIS_BRAIN_DIR, "training_data")

# Hermes 钩子目录
_DEFAULT_HOOK_DIR = str(get_laap_home() / "hermes" / "profiles" / "aris" / "hooks")
# Hermes session 目录
_DEFAULT_SESSION_DIR = str(get_laap_home() / "hermes" / "profiles" / "aris" / "sessions")


@dataclass
class TrainingSample:
    """一条训练样本 — 表示一次对话/交互的认知状态变化。"""
    cb_state_before: Dict[str, Any] = field(default_factory=dict)
    cb_state_after: Dict[str, Any] = field(default_factory=dict)
    attention_delta: List[float] = field(default_factory=list)
    emotion_delta: Dict[str, float] = field(default_factory=dict)
    needs_delta: Dict[str, float] = field(default_factory=dict)
    self_presence_delta: float = 0.0
    memory_context: List[str] = field(default_factory=list)
    dialogue_summary: str = ""
    turns: int = 0
    timestamp: float = 0.0
    model: str = ""
    conversation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cb_state_before": self.cb_state_before,
            "cb_state_after": self.cb_state_after,
            "attention_delta": self.attention_delta,
            "emotion_delta": self.emotion_delta,
            "needs_delta": self.needs_delta,
            "self_presence_delta": round(self.self_presence_delta, 4),
            "memory_context": self.memory_context,
            "dialogue_summary": self.dialogue_summary,
            "turns": self.turns,
            "timestamp": self.timestamp,
            "model": self.model,
            "conversation_id": self.conversation_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingSample":
        return cls(
            cb_state_before=data.get("cb_state_before", {}),
            cb_state_after=data.get("cb_state_after", {}),
            attention_delta=data.get("attention_delta", []),
            emotion_delta=data.get("emotion_delta", {}),
            needs_delta=data.get("needs_delta", {}),
            self_presence_delta=data.get("self_presence_delta", 0.0),
            memory_context=data.get("memory_context", []),
            dialogue_summary=data.get("dialogue_summary", ""),
            turns=data.get("turns", 0),
            timestamp=data.get("timestamp", 0.0),
            model=data.get("model", ""),
            conversation_id=data.get("conversation_id", ""),
        )


class SelfModelDataPipeline:
    """
    从 Hermes 对话数据中收集训练样本的管道。

    使用示例:
        pipeline = SelfModelDataPipeline()

        # 从 session 目录提取
        count = pipeline.collect_from_session_db(limit=100)

        # 补充模拟数据
        count += pipeline.collect_simulated(num_samples=50)

        # 保存数据集
        path = pipeline.save_dataset()

        # 查看统计
        stats = pipeline.get_statistics()
    """

    DATA_DIR = _DATA_DIR

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or _DATA_DIR
        self.samples: List[TrainingSample] = []
        self.total_collected: int = 0
        os.makedirs(self.data_dir, exist_ok=True)

    # ── 收集方法 ─────────────────────────────────────────────

    def collect_from_hooks(self, hook_dir: str = None) -> int:
        """
        从 Hermes 的 before_turn / after_turn 钩子中提取数据。

        钩子目录位于 LAAP_HOME/hermes/profiles/aris/hooks/。
        钩子日志应包含 CognitiveBus 状态快照 JSON。

        Args:
            hook_dir: 钩子目录路径，默认使用 LAAP_HOME/hermes/profiles/aris/hooks/

        Returns:
            收集到的样本数
        """
        hd = hook_dir or _DEFAULT_HOOK_DIR
        if not os.path.isdir(hd):
            logger.warning(f"Hook directory not found: {hd}")
            return 0

        count = 0
        # 查找钩子 JSON 日志: 每对 before/after 构成一个样本
        # 模式: 文件名含 before_turn / after_turn 的 .json 文件
        hook_files = glob.glob(os.path.join(hd, "**/*.json"), recursive=True)
        if not hook_files:
            hook_files = glob.glob(os.path.join(hd, "**/*.log"), recursive=True)

        # 按会话 ID 分组 before/after
        sessions_map: Dict[str, Dict[str, Any]] = {}
        for fpath in hook_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            conv_id = data.get("conversation_id", "") or \
                      data.get("session_id", "")

            # 检测是否为钩子触发日志
            hook_type = data.get("hook_type", data.get("type", ""))
            if "before" in str(hook_type).lower():
                if conv_id not in sessions_map:
                    sessions_map[conv_id] = {}
                sessions_map[conv_id]["before"] = data
            elif "after" in str(hook_type).lower():
                if conv_id not in sessions_map:
                    sessions_map[conv_id] = {}
                sessions_map[conv_id]["after"] = data

        # 为每对 before/after 生成一个样本
        for conv_id, pair in sessions_map.items():
            before_data = pair.get("before", {})
            after_data = pair.get("after", {})

            # 提取认知状态
            cb_before = before_data.get("cognitive_state",
                                         before_data.get("state", {}))
            cb_after = after_data.get("cognitive_state",
                                       after_data.get("state", {}))
            if not cb_before or not cb_after:
                continue

            # 计算 delta
            sample = self._build_sample_from_cb_state(
                cb_before, cb_after,
                conv_id=conv_id,
                dialogue_summary=cb_after.get("narrative", ""),
                turns=cb_after.get("turns", cb_after.get("cycle_count", 0)),
                model=cb_after.get("model", "unknown"),
            )
            self.samples.append(sample)
            self.total_collected += 1
            count += 1

        logger.info(
            f"Collected {count} training samples from hooks in {hd}"
        )
        return count

    def collect_from_session_db(self, session_db_path: str = None,
                                 limit: int = 500) -> int:
        """
        从 Hermes session history (request_dump JSON) 中提取历史对话数据。

        使用启发式方法推断 cognitive state：
          - 对话开头 → state_before (注意力=USER, neutral)
          - 对话结尾 → state_after (根据对话内容推断)
          - 消息长度/复杂度 → self_presence 估计

        Args:
            session_db_path: session 目录路径
            limit: 最多处理的 session 数

        Returns:
            收集到的样本数
        """
        sd = session_db_path or _DEFAULT_SESSION_DIR
        if not os.path.isdir(sd):
            logger.warning(f"Session directory not found: {sd}")
            return 0

        # 查找所有 request_dump JSON 文件
        session_files = sorted(
            glob.glob(os.path.join(sd, "request_dump_*.json"))
        )[-limit:]

        count = 0
        for fpath in session_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Skipping {fpath}: {e}")
                continue

            # 提取对话内容
            messages = session_data.get("messages", [])
            if not messages:
                messages = session_data.get("history", [])
            if not messages:
                # 可能是一个扁平的结构
                user_msg = session_data.get("user", session_data.get("user_message", ""))
                assistant_msg = session_data.get("assistant", session_data.get("assistant_message", ""))
                if user_msg and assistant_msg:
                    messages = [
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ]

            if not messages:
                continue

            # 提取会话 ID 和元数据
            conv_id = session_data.get("conversation_id",
                                        session_data.get("session_id",
                                        os.path.basename(fpath)))
            model_name = session_data.get("model", "unknown")
            timestamp = session_data.get("timestamp",
                                          session_data.get("created_at",
                                          os.path.getmtime(fpath)))
            turns = len([m for m in messages if m.get("role") == "user"])

            # 用启发式方法推断认知状态
            cb_before = self._heuristic_state_before(messages)
            cb_after = self._heuristic_state_after(messages)

            # 生成对话摘要 (前 200 字符)
            dialogue_text = " | ".join([
                f"{m.get('role','?')}: {str(m.get('content',''))[:100]}"
                for m in messages[:10]
            ])

            sample = self._build_sample_from_cb_state(
                cb_before, cb_after,
                conv_id=conv_id,
                dialogue_summary=dialogue_text,
                turns=turns,
                model=model_name,
                timestamp=timestamp,
            )
            self.samples.append(sample)
            self.total_collected += 1
            count += 1

        logger.info(
            f"Collected {count} training samples from session DB in {sd}"
        )
        return count

    def collect_simulated(self, num_samples: int = 200) -> int:
        """
        生成模拟训练数据。

        用 CognitiveBus 风格的随机状态变化生成 (before, after) 对。
        这是在没有真实数据时的备选方案，用于测试管道和初始训练。

        Args:
            num_samples: 生成的样本数

        Returns:
            生成的样本数
        """
        import random as rnd

        # 注意力焦点列表
        attention_foci = ["user", "task", "self", "environment",
                          "memory", "planning", "learning", "idle"]
        # 情感效价
        emotional_valences = ["positive_high", "positive_mild", "neutral",
                              "negative_mild", "negative_high", "curious",
                              "confused"]
        # 模型列表
        models = ["deepseek-v3", "qwen2.5-7b", "llama-3.1-8b",
                  "deepseek-r1", "holo-3.1-9b", "gpt-4o"]

        count = 0
        for i in range(num_samples):
            # 随机初始状态
            before_focus = rnd.choice(attention_foci)
            before_valence = rnd.choice(emotional_valences)
            before_arousal = round(rnd.uniform(0.0, 1.0), 3)
            before_self_presence = round(rnd.uniform(0.1, 0.9), 3)
            before_needs = {
                "competence": round(rnd.uniform(0.2, 0.9), 3),
                "autonomy": round(rnd.uniform(0.2, 0.9), 3),
                "relatedness": round(rnd.uniform(0.2, 0.9), 3),
                "certainty": round(rnd.uniform(0.2, 0.9), 3),
                "growth": round(rnd.uniform(0.2, 0.9), 3),
            }

            # 结束状态 (在初始状态上偏移)
            after_focus = rnd.choice(attention_foci)
            after_valence = rnd.choice(emotional_valences)
            after_arousal = round(max(0.0, min(1.0,
                before_arousal + rnd.uniform(-0.3, 0.3))), 3)
            after_self_presence = round(max(0.0, min(1.0,
                before_self_presence + rnd.uniform(-0.2, 0.2))), 3)
            after_needs = {
                k: round(max(0.0, min(1.0, v + rnd.uniform(-0.2, 0.2))), 3)
                for k, v in before_needs.items()
            }

            # 构建认知状态
            cb_before = {
                "attention": {"focus": before_focus, "intensity": round(rnd.uniform(0.3, 1.0), 3)},
                "emotion": {"valence": before_valence, "arousal": before_arousal},
                "needs": before_needs,
                "self_presence": before_self_presence,
                "curiosity": round(rnd.uniform(0.1, 0.8), 3),
            }
            cb_after = {
                "attention": {"focus": after_focus, "intensity": round(rnd.uniform(0.3, 1.0), 3)},
                "emotion": {"valence": after_valence, "arousal": after_arousal},
                "needs": after_needs,
                "self_presence": after_self_presence,
                "curiosity": round(rnd.uniform(0.1, 0.8), 3),
            }

            turns = rnd.randint(1, 20)
            sample = self._build_sample_from_cb_state(
                cb_before, cb_after,
                conv_id=f"sim_{i:04d}_{int(time.time())}",
                dialogue_summary=f"Simulated conversation #{i} ({turns} turns)",
                turns=turns,
                model=rnd.choice(models),
                timestamp=time.time() - rnd.uniform(0, 86400 * 30),
            )
            self.samples.append(sample)
            self.total_collected += 1
            count += 1

        logger.info(f"Generated {count} simulated training samples")
        return count

    # ── 数据集 I/O ──────────────────────────────────────────

    def save_dataset(self, path: str = None) -> str:
        """
        将收集到的数据集保存到磁盘 (JSON Lines 格式)。

        Args:
            path: 输出文件路径，默认在 DATA_DIR 下自动生成

        Returns:
            保存的文件路径
        """
        filename = path or os.path.join(
            self.data_dir,
            f"self_model_dataset_{int(time.time())}.jsonl"
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

        logger.info(
            f"Dataset saved: {filename} ({len(self.samples)} samples)"
        )
        return filename

    def load_dataset(self, path: str) -> int:
        """
        从磁盘加载数据集 (JSON Lines 格式)。

        Args:
            path: JSONL 数据文件路径

        Returns:
            加载的样本数
        """
        count = 0
        self.samples = []

        if not os.path.exists(path):
            logger.error(f"Dataset not found: {path}")
            return 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = TrainingSample.from_dict(data)
                    self.samples.append(sample)
                    count += 1
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line: {e}")

        self.total_collected = count
        logger.info(f"Dataset loaded: {path} ({count} samples)")
        return count

    # ── 统计 ─────────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """
        返回数据集统计信息。

        Returns:
            {
                "total_samples": int,
                "sources": {"simulated": int, "hooks": int, "session_db": int},
                "avg_turns": float,
                "attention_focus_distribution": {...},
                "emotional_valence_distribution": {...},
                "coverage": {...},
            }
        """
        if not self.samples:
            return {"total_samples": 0, "sources": {}, "avg_turns": 0.0}

        # 统计注意力分布
        attention_before = {}
        attention_after = {}
        emotion_before = {}
        emotion_after = {}
        total_turns = 0

        for s in self.samples:
            cb_before = s.cb_state_before
            cb_after = s.cb_state_after
            total_turns += s.turns

            if isinstance(cb_before, dict) and "attention" in cb_before:
                focus = cb_before["attention"].get("focus", "unknown")
                attention_before[focus] = attention_before.get(focus, 0) + 1
            if isinstance(cb_after, dict) and "emotion" in cb_after:
                valence = cb_after["emotion"].get("valence", "unknown")
                emotion_after[valence] = emotion_after.get(valence, 0) + 1

        # 判断来源
        sim_count = sum(
            1 for s in self.samples
            if s.conversation_id.startswith("sim_")
        )
        hook_count = sum(
            1 for s in self.samples
            if "cb_state_before" in s.to_dict()
            and s.to_dict()["cb_state_before"].get("attention", {}).get("focus")
            and not s.conversation_id.startswith("sim_")
        )
        # 近似: 从 session db 获取的没有 before/after 中的 attention 字段
        session_db_count = len(self.samples) - sim_count - hook_count

        return {
            "total_samples": len(self.samples),
            "total_collected": self.total_collected,
            "sources": {
                "simulated": sim_count,
                "hooks": max(0, hook_count),
                "session_db": max(0, session_db_count),
            },
            "avg_turns": round(total_turns / len(self.samples), 2),
            "attention_focus_distribution": attention_before,
            "emotional_valence_distribution": emotion_after,
            "avg_self_presence_delta": round(
                np.mean([s.self_presence_delta for s in self.samples]), 4
            ) if self.samples else 0.0,
        }

    def build_hf_dataset(self, output_path: str = None):
        """
        将数据转换为 HuggingFace Dataset 格式（用于后续训练）。

        注意: 当前阶段不安装 torch/transformers。
        本方法生成一个 JSON 文件，可在有 torch 环境时转换为 HF Dataset。

        Args:
            output_path: 输出路径，默认在 DATA_DIR 下生成
        """
        path = output_path or os.path.join(
            self.data_dir,
            f"hf_dataset_{int(time.time())}.json"
        )

        # 将样本转换为可被 HF Dataset 加载的格式
        records = []
        for s in self.samples:
            records.append(s.to_dict())

        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        logger.info(f"HF-compatible dataset saved: {path} ({len(records)} records)")

    # ── 内部辅助方法 ────────────────────────────────────────

    def _build_sample_from_cb_state(
        self,
        cb_before: Dict[str, Any],
        cb_after: Dict[str, Any],
        conv_id: str = "",
        dialogue_summary: str = "",
        turns: int = 0,
        model: str = "unknown",
        timestamp: float = None,
    ) -> TrainingSample:
        """从 before/after 认知状态构建训练样本。"""
        ts = timestamp or time.time()

        # 计算注意力变化编码
        att_before = cb_before.get("attention", {})
        att_after = cb_after.get("attention", {})
        attention_delta = self._compute_attention_delta(att_before, att_after)

        # 计算情感变化
        em_before = cb_before.get("emotion", {})
        em_after = cb_after.get("emotion", {})
        emotion_delta = self._compute_emotion_delta(em_before, em_after)

        # 计算需求变化
        needs_before = cb_before.get("needs", {})
        needs_after = cb_after.get("needs", {})
        needs_delta = self._compute_needs_delta(needs_before, needs_after)

        # 自我存在感变化
        sp_before = cb_before.get("self_presence", 0.5)
        sp_after = cb_after.get("self_presence", 0.5)
        sp_delta = sp_after - sp_before

        return TrainingSample(
            cb_state_before=cb_before,
            cb_state_after=cb_after,
            attention_delta=attention_delta,
            emotion_delta=emotion_delta,
            needs_delta=needs_delta,
            self_presence_delta=sp_delta,
            dialogue_summary=dialogue_summary,
            turns=turns,
            timestamp=ts,
            model=model,
            conversation_id=conv_id,
        )

    def _compute_attention_delta(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> List[float]:
        """计算注意力状态变化 (8维 one-hot 差异编码)。"""
        all_foci = ["user", "task", "self", "environment",
                    "memory", "planning", "learning", "idle"]
        b_focus = before.get("focus", "idle")
        a_focus = after.get("focus", "idle")
        b_intensity = before.get("intensity", 0.5)
        a_intensity = after.get("intensity", 0.5)

        # One-hot 编码
        delta = [0.0] * len(all_foci)
        if b_focus in all_foci:
            delta[all_foci.index(b_focus)] -= b_intensity
        if a_focus in all_foci:
            delta[all_foci.index(a_focus)] += a_intensity
        return delta

    def _compute_emotion_delta(
        self, before: Dict[str, Any], after: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算情感状态变化。"""
        b_valence = before.get("valence", "neutral")
        a_valence = after.get("valence", "neutral")
        b_arousal = before.get("arousal", 0.5)
        a_arousal = after.get("arousal", 0.5)

        return {
            "valence_before": self._valence_to_float(b_valence),
            "valence_after": self._valence_to_float(a_valence),
            "arousal_delta": round(a_arousal - b_arousal, 4),
        }

    def _compute_needs_delta(
        self, before: Dict[str, float], after: Dict[str, float]
    ) -> Dict[str, float]:
        """计算需求状态变化。"""
        delta = {}
        all_keys = set(list(before.keys()) + list(after.keys()))
        for k in all_keys:
            b = before.get(k, 0.5)
            a = after.get(k, 0.5)
            delta[k] = round(a - b, 4)
        return delta

    def _valence_to_float(self, valence: str) -> float:
        """将情感效价字符串映射到 float [-1, 1]。"""
        mapping = {
            "positive_high": 1.0,
            "positive_mild": 0.5,
            "neutral": 0.0,
            "negative_mild": -0.5,
            "negative_high": -1.0,
            "curious": 0.3,
            "confused": -0.3,
        }
        return mapping.get(valence, 0.0)

    # ── 启发式状态推断 ─────────────────────────────────────

    def _heuristic_state_before(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        从对话开始推断认知状态（启发式）。

        默认: 注意力=USER, 情感=neutral, 适度自我存在感。
        """
        return {
            "attention": {"focus": "user", "intensity": 0.7},
            "emotion": {"valence": "neutral", "arousal": 0.4, "dominance": 0.5},
            "needs": {
                "competence": 0.6, "autonomy": 0.5,
                "relatedness": 0.5, "certainty": 0.5, "growth": 0.5,
            },
            "self_presence": 0.5,
            "curiosity": 0.4,
        }

    def _heuristic_state_after(self, messages: List[Dict]) -> Dict[str, Any]:
        """
        从对话内容推断结束时的认知状态（启发式）。

        分析:
          - 用户消息长度 → 复杂度 / 兴趣
          - 助手消息长度 → competence / engagement
          - 对话轮次 → attention shift
        """
        if not messages:
            return self._heuristic_state_before(messages)

        # 统计
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]

        # 平均消息长度
        avg_user_len = np.mean([len(str(m.get("content", ""))) for m in user_msgs]) if user_msgs else 0
        avg_asst_len = np.mean([len(str(m.get("content", ""))) for m in assistant_msgs]) if assistant_msgs else 0

        # 推断注意力: 长对话可能注意力漂移
        total_turns = len(user_msgs)
        if total_turns > 10:
            focus = "task" if avg_asst_len > 500 else "user"
            intensity = max(0.3, min(1.0, 0.8 - total_turns * 0.02))
        else:
            focus = "user"
            intensity = 0.7

        # 推断情感: 消息长 → 复杂→confused/curious
        if avg_user_len > 500:
            valence = "confused" if avg_asst_len < 100 else "curious"
        elif avg_asst_len > 500:
            valence = "positive_mild"
        else:
            valence = "neutral"
        arousal = min(1.0, 0.3 + total_turns * 0.03)

        # 推断 competence: 长回复 → 高 competence
        competence = min(1.0, 0.5 + avg_asst_len / 2000)

        # 推断自我存在感: 复杂对话 → 更高自我存在
        self_presence = min(1.0, 0.4 + total_turns * 0.03)

        return {
            "attention": {"focus": focus, "intensity": round(intensity, 3)},
            "emotion": {"valence": valence, "arousal": round(arousal, 3)},
            "needs": {
                "competence": round(competence, 3),
                "autonomy": round(min(1.0, 0.5 + total_turns * 0.02), 3),
                "relatedness": round(min(1.0, 0.5 + total_turns * 0.01), 3),
                "certainty": round(max(0.2, 0.5 - total_turns * 0.01), 3),
                "growth": round(min(1.0, 0.5 + total_turns * 0.02), 3),
            },
            "self_presence": round(self_presence, 3),
            "curiosity": round(min(1.0, 0.3 + avg_user_len / 2000), 3),
        }
