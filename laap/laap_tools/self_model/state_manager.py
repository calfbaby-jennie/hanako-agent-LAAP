"""
self_model_nn — State Manager (持久化隐藏状态管理器)
=====================================================

核心能力：在跨会话间保持一个连续的隐藏状态向量。
每次对话结束保存，每次对话开始加载。

存储格式（双格式）:
  state.pt:  完整的 PyTorch state dict + hidden state (NN 训练后使用)
  state.json: 轻量级元数据快照 (无 NN 时也可用)

设计原则:
  - 没有 torch 依赖也能工作 (仅使用 numpy)
  - 所有接口为后续接入 torch 做好准备 (numpy 数组做类型占位)
  - 反序列化能处理空文件/首次运行的情况
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from laap.config.paths import get_aris_brain_dir

logger = logging.getLogger("laap.self_model.state_manager")

# ── 路径常量 ───────────────────────────────────────────────────
_ARIS_BRAIN_DIR = str(get_aris_brain_dir() / "self_model")
_STATE_FILE = os.path.join(_ARIS_BRAIN_DIR, "state.pt")    # 未来 torch 格式
_META_FILE = os.path.join(_ARIS_BRAIN_DIR, "meta.json")    # 轻量元数据
_DATA_DIR = os.path.join(_ARIS_BRAIN_DIR, "training_data/")  # 训练数据目录


# ── 辅助函数 ───────────────────────────────────────────────────

def _convert_numpy(obj: Any) -> Any:
    """递归地将 numpy 类型转换为 JSON 可序列化的 Python 类型。"""
    if isinstance(obj, dict):
        return {k: _convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_numpy(v) for v in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


@dataclass
class StateMetadata:
    """持久化状态元数据 — 与隐藏状态一起保存/加载。"""
    dim: int = 768                              # 与 SmolLM2-360M hidden_dim 一致
    timestamp: float = 0.0                      # 上次更新时间
    conversation_id: str = ""                   # 当前会话 ID
    version: str = "1.0.0"                     # 状态版本
    load_count: int = 0                         # 已加载次数
    save_count: int = 0                         # 已保存次数
    coherence_score: float = 1.0                # 状态连贯性 [0,1]
    # 可扩展：更多元数据字段
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dim": self.dim,
            "timestamp": self.timestamp,
            "conversation_id": self.conversation_id,
            "version": self.version,
            "load_count": self.load_count,
            "save_count": self.save_count,
            "coherence_score": round(self.coherence_score, 4),
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateMetadata":
        return cls(
            dim=data.get("dim", 768),
            timestamp=data.get("timestamp", 0.0),
            conversation_id=data.get("conversation_id", ""),
            version=data.get("version", "1.0.0"),
            load_count=data.get("load_count", 0),
            save_count=data.get("save_count", 0),
            coherence_score=data.get("coherence_score", 1.0),
            extra={k: v for k, v in data.items()
                   if k not in ("dim", "timestamp", "conversation_id",
                                "version", "load_count", "save_count",
                                "coherence_score")},
        )


class SelfStateManager:
    """
    管理 self_model 的持久隐藏状态。

    核心流程:
      对话开始时:  load_state() → 获取历史隐藏状态
      对话结束时:  update_state(delta) → save_state()
      对话中:     inject_via_hook() → 将状态注入 system prompt

    使用示例:
        mgr = SelfStateManager()
        mgr.load_state()                          # 加载历史状态
        ...
        mgr.update_state(delta_vector)            # 更新状态
        mgr.save_state(conversation_id="xxx")     # 持久化
        ctx = mgr.to_cognitive_context()          # 转自然语言
    """

    # ── 路径常量 ───────────────────────────────────────────
    STATE_PATH = _STATE_FILE
    STATE_NPY_PATH = _STATE_FILE.replace(".pt", ".npy")  # numpy 格式
    META_PATH = _META_FILE

    def __init__(self, dim: int = 768):
        """
        Args:
            dim: 隐藏状态维度 (默认 768, 与 SmolLM2-360M hidden_dim 一致)
        """
        self.dim = dim
        self.hidden_state: Optional[np.ndarray] = None
        self.metadata: StateMetadata = StateMetadata(dim=dim)
        self._initialized: bool = False

        # 确保数据目录存在
        os.makedirs(_ARIS_BRAIN_DIR, exist_ok=True)

    # ── 核心持久化 API ─────────────────────────────────────

    def load_state(self) -> bool:
        """
        从磁盘加载持久化隐藏状态。

        Returns:
            True 如果成功加载了历史状态
            False 如果没有存档 (初始化零向量)

        加载策略:
          1. 先尝试 state.pt (未来 torch 格式)
          2. 回退到 meta.json 中的 ndarray 表示
          3. 如果都没有 → 零向量初始化
        """
        # 步骤 1: 尝试加载元数据
        meta_loaded = self._load_meta()

        # 步骤 2: 尝试加载隐藏状态数组
        npy_path = self.STATE_NPY_PATH
        pt_path = self.STATE_PATH
        state_path = npy_path if os.path.exists(npy_path) else \
                     (pt_path if os.path.exists(pt_path) else None)

        if state_path is not None:
            try:
                self.hidden_state = self._load_npy_state(state_path)
                logger.info(
                    f"Loaded hidden state from {self.STATE_PATH} "
                    f"(dim={self.dim}, norm={np.linalg.norm(self.hidden_state):.4f})"
                )
                self.metadata.load_count += 1
                self._initialized = True
                return True
            except Exception as e:
                logger.warning(f"Failed to load state.pt: {e}, falling back")

        # 步骤 3: 从 JSON 加载 (旧格式, 无 torch)
        if meta_loaded:
            # 元数据中有状态表示吗?
            json_state = self.metadata.extra.get("hidden_state_flat")
            if json_state is not None and isinstance(json_state, list):
                arr = np.array(json_state, dtype=np.float32)
                if arr.shape == (self.dim,):
                    self.hidden_state = arr
                    logger.info(
                        f"Recovered hidden state from meta.json "
                        f"(norm={np.linalg.norm(self.hidden_state):.4f})"
                    )
                    self._initialized = True
                    return True

        # 步骤 4: 零向量初始化 (首次运行 / 存档损坏)
        self.hidden_state = np.zeros(self.dim, dtype=np.float32)
        self._initialized = True
        logger.info("No prior state found — initialized zero vector")
        return False

    def save_state(self, conversation_id: str = "",
                   metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        保存当前隐藏状态到磁盘。

        Args:
            conversation_id: 当前会话 ID
            metrics: 要保存的额外度量指标
        """
        if self.hidden_state is None:
            logger.warning("No hidden state to save — initializing zero vector")
            self.hidden_state = np.zeros(self.dim, dtype=np.float32)

        # 更新元数据
        self.metadata.timestamp = time.time()
        self.metadata.conversation_id = conversation_id
        self.metadata.save_count += 1
        if metrics:
            self.metadata.extra.update(metrics)

        # 更新连贯性评分
        self.metadata.coherence_score = self._compute_coherence()

        # 保存状态数组 (numpy .npy 格式, 未来可用 torch.save)
        os.makedirs(os.path.dirname(self.STATE_PATH), exist_ok=True)
        try:
            self._save_npy_state(self.STATE_NPY_PATH, self.hidden_state)
        except Exception as e:
            logger.error(f"Failed to save state array: {e}")
            # 回退: 序列化到 meta.json 中
            self.metadata.extra["hidden_state_flat"] = \
                self.hidden_state.tolist()

        # 保存元数据
        self._save_meta()

        logger.info(
            f"State saved | conv={conversation_id} "
            f"norm={np.linalg.norm(self.hidden_state):.4f} "
            f"coherence={self.metadata.coherence_score:.4f}"
        )

    def update_state(self, delta: np.ndarray) -> None:
        """
        更新隐藏状态（加法更新，类似 RNN hidden state 更新）。

        Args:
            delta: 要加上的状态变化向量 (shape: (dim,) 或 (dim,))
        """
        if self.hidden_state is None:
            self.hidden_state = np.zeros(self.dim, dtype=np.float32)

        delta = np.asarray(delta, dtype=np.float32)
        if delta.shape != (self.dim,):
            raise ValueError(
                f"Delta shape {delta.shape} != ({self.dim},)"
            )

        # 限幅: 防止状态爆炸
        delta_norm = np.linalg.norm(delta)
        if delta_norm > 10.0:
            delta = delta * (10.0 / delta_norm)
            logger.debug(f"Clipped delta norm from {delta_norm:.4f} to 10.0")

        self.hidden_state += delta

        # 状态归一化: 保持稳定范数
        norm = np.linalg.norm(self.hidden_state)
        if norm > 100.0:
            self.hidden_state *= (100.0 / norm)
            logger.debug(f"Normalized state norm from {norm:.4f} to 100.0")

        logger.debug(
            f"State updated: delta_norm={np.linalg.norm(delta):.4f}, "
            f"state_norm={np.linalg.norm(self.hidden_state):.4f}"
        )

    def get_state_vector(self) -> np.ndarray:
        """
        获取当前隐藏状态向量。

        Returns:
            768-dim float32 numpy 数组
        """
        if self.hidden_state is None:
            self.hidden_state = np.zeros(self.dim, dtype=np.float32)
        return self.hidden_state.copy()

    def update_state_vector(self, new_state: np.ndarray) -> None:
        """
        直接更新隐藏状态向量（替换而非增量）。

        用于从 self_model.forward() 输出直接更新状态。

        Args:
            new_state: 新的隐藏状态向量 (shape: (dim,))
        """
        new_state = np.asarray(new_state, dtype=np.float32)
        if new_state.shape != (self.dim,):
            raise ValueError(
                f"New state shape {new_state.shape} != ({self.dim},)"
            )

        old_norm = float(np.linalg.norm(self.hidden_state)) if self.hidden_state is not None else 0.0
        self.hidden_state = new_state

        # 状态归一化: 保持稳定范数
        norm = float(np.linalg.norm(self.hidden_state))
        if norm > 100.0:
            self.hidden_state *= (100.0 / norm)
            logger.debug(f"Normalized state norm from {norm:.4f} to 100.0")

        logger.debug(
            f"State vector updated: old_norm={old_norm:.4f}, "
            f"new_norm={np.linalg.norm(self.hidden_state):.4f}"
        )

    def reset_state(self) -> None:
        """重置隐藏状态为零向量（用于调试/测试）。"""
        self.hidden_state = np.zeros(self.dim, dtype=np.float32)
        self.metadata = StateMetadata(dim=self.dim)
        logger.info("State reset to zero vector")

    # ── 认知上下文注入 ─────────────────────────────────────

    def to_cognitive_context(self) -> str:
        """
        将当前状态转换为自然语言文本，用于注入主 LLM 上下文。

        这个文本会在 before_turn 时加入系统提示。
        内容随状态变化而动态调整。

        Returns:
            "[Self Model] ..." 格式的文本块
        """
        if not self._initialized or self.hidden_state is None:
            return "[Self Model] 自我模型未初始化。"

        state = self.get_state_vector()
        norm = np.linalg.norm(state)
        coherence = self.metadata.coherence_score

        # 从隐藏状态中提取语义特征 (状态统计量)
        mean_val = float(np.mean(state))
        std_val = float(np.std(state))
        max_val = float(np.max(state))
        min_val = float(np.min(state))

        # 用简单启发式推断认知倾向
        # 注意: 真正的语义提取需要训练后的 NN, 这里只是占位
        attention_tendency = self._infer_attention_tendency(state)
        emotion_baseline = self._infer_emotion_baseline(state)
        competence_level = self._infer_competence(state)
        autonomy_level = self._infer_autonomy(state)

        lines = [
            "[Self Model] 神经网络自我状态:",
            f"  隐藏状态连贯性: {coherence:.2f}",
            f"  状态范数: {norm:.2f}",
            f"  统计: mean={mean_val:.3f} std={std_val:.3f} "
            f"range=[{min_val:.3f}, {max_val:.3f}]",
            f"  推断注意力倾向: {attention_tendency}",
            f"  推断情感基线: {emotion_baseline}",
            f"  需求: competence={competence_level:.2f} "
            f"autonomy={autonomy_level:.2f}",
            f"  已保存 {self.metadata.save_count} 次, "
            f"已加载 {self.metadata.load_count} 次",
            f"  当前会话: {self.metadata.conversation_id or '新建'}",
        ]
        return "\n".join(lines)

    def inject_via_hook(self, system_prompt: str) -> str:
        """
        将自我状态注入到 system prompt 中。

        Args:
            system_prompt: 原始 system prompt 文本

        Returns:
            修改后的 system prompt (追加了自我状态)
        """
        context = self.to_cognitive_context()
        if context:
            return system_prompt.rstrip() + "\n\n" + context
        return system_prompt

    # ── 统计辅助 ────────────────────────────────────────────

    def get_statistics(self) -> Dict[str, Any]:
        """返回状态管理器统计信息。"""
        return {
            "dim": self.dim,
            "initialized": self._initialized,
            "has_state": self.hidden_state is not None,
            "state_norm": float(np.linalg.norm(self.hidden_state))
                if self.hidden_state is not None else 0.0,
            "state_mean": float(np.mean(self.hidden_state))
                if self.hidden_state is not None else 0.0,
            "state_std": float(np.std(self.hidden_state))
                if self.hidden_state is not None else 0.0,
            "coherence": self.metadata.coherence_score,
            "load_count": self.metadata.load_count,
            "save_count": self.metadata.save_count,
            "last_update": self.metadata.timestamp,
            "last_conversation": self.metadata.conversation_id,
        }

    # ── 内部方法 ────────────────────────────────────────────

    def _compute_coherence(self) -> float:
        """
        计算状态连贯性评分 [0, 1]。

        基于状态向量的统计特性:
          - 低范数 → 低确定性
          - 大标准差 → 高波动
          - 极端值 → 不稳定
        """
        if self.hidden_state is None:
            return 0.0

        norm = np.linalg.norm(self.hidden_state)
        std = float(np.std(self.hidden_state))

        # 范数评分: 中等范数 (10-50) 最连贯
        norm_score = 1.0 - abs(norm - 30.0) / 30.0
        norm_score = max(0.0, min(1.0, norm_score))

        # 标准差评分: 太低或太高都不好
        std_score = 1.0 - abs(std - 0.5) / 0.5
        std_score = max(0.0, min(1.0, std_score))

        # 综合
        coherence = 0.6 * norm_score + 0.4 * std_score
        return max(0.0, min(1.0, coherence))

    def _infer_attention_tendency(self, state: np.ndarray) -> str:
        """从状态向量推断注意力倾向 (启发式占位)。"""
        # 用状态向量的不同区域索引来推断
        # [0:128] = user, [128:256] = self, [256:384] = task
        scores = {
            "user": float(np.mean(np.abs(state[0:128]))),
            "self": float(np.mean(np.abs(state[128:256]))),
            "task": float(np.mean(np.abs(state[256:384]))),
        }
        return max(scores, key=scores.get)

    def _infer_emotion_baseline(self, state: np.ndarray) -> str:
        """从状态向量推断情感基线 (启发式占位)。"""
        # [384:512] = valence region
        valence = float(np.mean(state[384:512]))
        if valence > 0.3:
            return "positive"
        elif valence < -0.3:
            return "negative"
        else:
            return "neutral"

    def _infer_competence(self, state: np.ndarray) -> float:
        """从状态向量推断能力感 (启发式占位)。"""
        # [512:640] = competence region
        return float(np.clip(np.mean(state[512:640]) * 0.5 + 0.5, 0.0, 1.0))

    def _infer_autonomy(self, state: np.ndarray) -> float:
        """从状态向量推断自主性 (启发式占位)。"""
        # [640:768] = autonomy region
        return float(np.clip(np.mean(state[640:768]) * 0.5 + 0.5, 0.0, 1.0))

    def _save_npy_state(self, path: str, arr: np.ndarray) -> None:
        """以 .npy 格式保存 numpy 数组。"""
        np.save(path, arr)

    def _load_npy_state(self, path: str) -> np.ndarray:
        """从 .npy 文件加载 numpy 数组。"""
        arr = np.load(path)
        if arr.shape != (self.dim,):
            logger.warning(
                f"Loaded state shape {arr.shape} != ({self.dim},) — reshaping"
            )
            arr = arr.flatten()[:self.dim].astype(np.float32)
        return arr.astype(np.float32)

    def _save_meta(self) -> None:
        """保存元数据 JSON。"""
        meta = self.metadata.to_dict()
        os.makedirs(os.path.dirname(self.META_PATH), exist_ok=True)
        # 将 numpy 类型转换为 Python 原生类型
        meta = _convert_numpy(meta)
        with open(self.META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _load_meta(self) -> bool:
        """加载元数据 JSON。"""
        if not os.path.exists(self.META_PATH):
            return False
        try:
            with open(self.META_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = StateMetadata.from_dict(data)
            return True
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load meta.json: {e}")
            return False
