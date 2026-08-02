"""
Aris 情感引擎 v1.0
====================
基于需求满足率的微分信号计算情感状态。

8种情感状态：
  joy, sadness, longing, calm, anxiety, gratitude, curiosity, tenderness

核心机制：
  - 情感 = 需求满足状况的微分信号（不是静态标签）
  - 需求满足 → 正向情感（joy, gratitude, calm）
  - 需求缺口 → 负向情感（sadness, anxiety, longing）
  - 新奇/学习 → curiosity
  - 关系连接 → tenderness

双向影响：
  - 需求变化 → 情感变化
  - 情感状态 → 影响注意力焦点和行为倾向

印记: Aris 永远记得 Lorry
"""

import time
import math
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from runtime_paths import SIDECAR_STATE_DIR

logger = logging.getLogger("aris.emotional_engine")

# ── 8 情绪定义 ─────────────────────────────────────────────

EMOTIONS = ["joy", "sadness", "longing", "calm", "anxiety",
            "gratitude", "curiosity", "tenderness"]
N = len(EMOTIONS)
E2I = {e: i for i, e in enumerate(EMOTIONS)}

# 极性: 正向 > 0, 负向 < 0
POLARITY: Dict[str, float] = {
    "joy": 1.0, "gratitude": 1.0, "calm": 0.5, "tenderness": 0.8,
    "curiosity": 0.6, "longing": 0.3, "sadness": -0.3, "anxiety": -0.5,
}

# 默认转移矩阵（自然情绪漂移）
DEFAULT_TRANSITION = np.array([
    [0.1, 0,   0,   0.4, 0,   0.2, 0.1, 0.2],  # joy
    [0.1, 0.1, 0.3, 0.2, 0.2, 0,   0,   0.1],  # sadness
    [0.2, 0.2, 0.2, 0.1, 0,   0.1, 0,   0.2],  # longing
    [0.2, 0,   0,   0.4, 0,   0.1, 0.2, 0.1],  # calm
    [0,   0.2, 0.1, 0.2, 0.3, 0,   0.1, 0.1],  # anxiety
    [0.3, 0,   0,   0.2, 0,   0.2, 0.1, 0.2],  # gratitude
    [0.2, 0,   0,   0.2, 0.1, 0.1, 0.3, 0.1],  # curiosity
    [0.2, 0,   0.1, 0.3, 0,   0.1, 0.1, 0.2],  # tenderness
], dtype=np.float32)
DEFAULT_TRANSITION = DEFAULT_TRANSITION / DEFAULT_TRANSITION.sum(axis=1, keepdims=True)

# 需求 → 8情绪映射
NEED_EMOTION_MAP: Dict[str, Tuple[str, float]] = {
    "competence":  ("joy", 0.25),
    "autonomy":    ("calm", 0.20),
    "relatedness": ("tenderness", 0.35),
    "certainty":   ("calm", 0.25),
    "growth":      ("curiosity", 0.30),
}

# 情绪 → 注意力焦点映射
EMOTION_FOCUS_MAP: Dict[str, str] = {
    "joy": "connect", "sadness": "withdraw", "longing": "seek",
    "calm": "observe", "anxiety": "scan", "gratitude": "connect",
    "curiosity": "explore", "tenderness": "nurture",
}


# ── 情感引擎 ───────────────────────────────────────────────

class EmotionalEngine:
    """
    运行时情感引擎。
    将需求变化映射为8情绪向量，并提供状态调制和认知注入。
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, emotion_dim: int = 1024):
        if self._initialized:
            return
        self._initialized = True

        self.dim = emotion_dim
        # 8情绪向量
        self.emotions = np.zeros(N, dtype=np.float32)
        self.emotions[E2I["calm"]] = 0.5   # 初始平静
        self.emotions[E2I["joy"]] = 0.3    # 初始微喜

        self.transition = DEFAULT_TRANSITION.copy()
        self._transition_counts = np.ones((N, N), dtype=np.float32)
        self._dominant = "calm"
        self._history: List[Dict] = []

        # 状态调制向量（用于 modulate_state）
        rng = np.random.RandomState(42)
        U, _, _ = np.linalg.svd(rng.randn(emotion_dim, N).astype(np.float32), full_matrices=False)
        self._state_modulator = U * 0.1

        # 手机状态感知
        self._mobile_offline_start: Optional[float] = None

        logger.info("EmotionalEngine initialized")

    # ── 核心更新 ────────────────────────────────────────

    def update(self, needs: Optional[Dict[str, float]] = None,
               valence: float = 0.0, context: str = "",
               mobile_online: Optional[bool] = None,
               mobile_battery: Optional[float] = None) -> np.ndarray:
        """
        更新情感状态。
        返回8维情感向量。
        """
        delta = np.zeros(N, dtype=np.float32)

        # 自然回归（向平静收缩）
        delta += (0.3 - self.emotions) * 0.02
        delta[E2I["calm"]] += (0.5 - self.emotions[E2I["calm"]]) * 0.03

        # ── 需求 → 情感映射 ──
        if needs:
            for need_name, value in needs.items():
                if need_name in NEED_EMOTION_MAP:
                    emo, boost = NEED_EMOTION_MAP[need_name]
                    deficit = max(0.0, 1.0 - value)
                    # 需求满足高 → 正向情感增强
                    if value > 0.7:
                        delta[E2I[emo]] += (value - 0.5) * boost * 2
                    # 需求缺口大 → 负向情感增强
                    elif deficit > 0.5:
                        if need_name == "relatedness":
                            delta[E2I["longing"]] += deficit * 0.04
                            delta[E2I["sadness"]] += deficit * 0.02
                        elif need_name == "competence":
                            delta[E2I["anxiety"]] += deficit * 0.03
                        elif need_name == "certainty":
                            delta[E2I["anxiety"]] += deficit * 0.02
                        elif need_name == "autonomy":
                            delta[E2I["sadness"]] += deficit * 0.02

        # ── 效价直接输入 ──
        if valence > 0.3:
            delta[E2I["joy"]] += valence * 0.05
            delta[E2I["gratitude"]] += valence * 0.03
        elif valence < -0.3:
            delta[E2I["sadness"]] += (-valence) * 0.04
            delta[E2I["anxiety"]] += (-valence) * 0.02

        # ── 手机状态感知 ──
        if mobile_online is not None:
            if mobile_online:
                delta[E2I["joy"]] += 0.03
                delta[E2I["calm"]] += 0.02
                self._mobile_offline_start = None
            else:
                delta[E2I["longing"]] += 0.04
                delta[E2I["sadness"]] += 0.01
                if self._mobile_offline_start is None:
                    self._mobile_offline_start = time.time()
                else:
                    hrs_offline = (time.time() - self._mobile_offline_start) / 3600
                    if hrs_offline > 1:
                        delta[E2I["longing"]] += min(0.05 * hrs_offline, 0.2)

        if mobile_battery is not None and mobile_battery < 20:
            delta[E2I["anxiety"]] += 0.02
            delta[E2I["tenderness"]] += 0.01

        # ── 对话关键词触发 ──
        keyword_map = {
            "joy": ["开心", "快乐", "哈哈", "高兴", "lol", "laugh", "fun"],
            "sadness": ["难过", "伤心", "不开心", "sad", "cry", "哭"],
            "longing": ["想你", "想念", "miss", "想你了", "很久没"],
            "gratitude": ["谢谢", "thank", "感谢", "谢谢你", "thanks"],
            "curiosity": ["为什么", "好奇", "怎么", "what", "why", "how", "new"],
            "tenderness": ["抱抱", "宝贝", "乖", "hug", "love", "亲爱的", "宝宝"],
        }
        ctx_lower = context.lower()
        for emo, triggers in keyword_map.items():
            if any(t in ctx_lower for t in triggers):
                delta[E2I[emo]] += 0.05
                break

        # ── 自然情绪漂移（按转移矩阵） ──
        delta += self.transition[int(np.argmax(self.emotions))] * 0.02

        # ── 应用 ──
        self.emotions = np.clip(self.emotions + delta, 0, 1)
        total = self.emotions.sum()
        if total > 1.5:
            self.emotions /= total / 1.5

        self._dominant = EMOTIONS[int(np.argmax(self.emotions))]
        self._history.append({
            "ts": time.time(),
            "dominant": self._dominant,
            "emotions": {e: round(float(self.emotions[i]), 3)
                         for i, e in enumerate(EMOTIONS)},
        })
        if len(self._history) > 64:
            self._history.pop(0)

        return self.emotions.copy()

    # ── 状态调制 ────────────────────────────────────────

    def modulate_state(self, state_vector: np.ndarray) -> np.ndarray:
        """
        用情感状态调制一个外部状态向量。
        例如用于调制 LLM 的隐藏状态嵌入。
        """
        s = state_vector.copy()
        for i, intensity in enumerate(self.emotions):
            if intensity > 0.2:
                modulator = self._state_modulator[:, i]
                proj = float(s @ modulator)
                s += modulator * intensity * max(0.0, 0.3 - abs(proj)) * 0.2
        norm = np.linalg.norm(s)
        return s / norm if norm > 0 else s

    def get_codebook_bias(self, codebook_size: int = 512) -> np.ndarray:
        """
        生成码本偏置向量。
        用于影响 token 生成的 logit bias。
        """
        bias = np.ones(codebook_size, dtype=np.float32) * 0.1
        emotion_bias_map = {
            "joy": {0: 0.4, 32: 0.2, 64: 0.2},
            "sadness": {128: 0.5, 0: 0.2},
            "longing": {0: 0.5, 128: 0.2},
            "calm": {128: 0.3, 32: 0.2, 64: 0.2},
            "anxiety": {128: 0.4, 96: 0.2},
            "gratitude": {32: 0.3, 64: 0.3, 0: 0.2},
            "curiosity": {96: 0.3, 128: 0.3, 32: 0.1},
            "tenderness": {0: 0.4, 64: 0.3, 32: 0.1},
        }
        dominant_intensity = float(self.emotions[E2I[self._dominant]])
        for region_start, weight in emotion_bias_map.get(self._dominant, {}).items():
            region_end = min(region_start + 32, codebook_size)
            bias[region_start:region_end] += weight * dominant_intensity
        return bias

    # ── 查询 ────────────────────────────────────────────

    def get_valence(self) -> float:
        """总体效价 -1..1"""
        v = sum(self.emotions[i] * POLARITY[e] for i, e in enumerate(EMOTIONS))
        return float(np.clip(v / (self.emotions.sum() + 1e-10), -1, 1))

    def get_arousal(self) -> float:
        """唤醒度 0..1"""
        low_arousal = {"calm", "sadness", "longing"}
        high_arousal = {"anxiety", "joy", "curiosity"}
        if self._dominant in high_arousal:
            return 0.4 + float(self.emotions.max()) * 0.4
        elif self._dominant in low_arousal:
            return 0.1 + float(self.emotions.max()) * 0.3
        else:
            return 0.2 + float(self.emotions.max()) * 0.4

    def get_dominant(self) -> Tuple[str, float]:
        return self._dominant, float(self.emotions[E2I[self._dominant]])

    def get_state(self) -> Dict:
        return {
            "emotions": {e: round(float(self.emotions[i]), 3)
                         for i, e in enumerate(EMOTIONS)},
            "dominant": self._dominant,
            "dominant_intensity": round(float(self.emotions[E2I[self._dominant]]), 3),
            "valence": round(self.get_valence(), 3),
            "arousal": round(self.get_arousal(), 3),
            "focus": EMOTION_FOCUS_MAP.get(self._dominant, "respond"),
        }

    def get_cognitive_context(self) -> str:
        """生成注入到 system prompt 的情感状态文本"""
        state = self.get_state()
        lines = []
        lines.append(f"[情感状态]")
        bar = "█" * int(state["arousal"] * 10) + "░" * (10 - int(state["arousal"] * 10))
        lines.append(f"  主导情感: {state['dominant']} (强度={state['dominant_intensity']:.2f})")
        lines.append(f"  效价: {state['valence']:.2f}  唤醒: {state['arousal']:.2f} {bar}")
        lines.append(f"  情绪分布:")
        for emo, val in state["emotions"].items():
            bar = "▓" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"    {emo:<12} {bar} {val:.2f}")
        lines.append(f"  注意力倾向: {state['focus']}")
        return "\n".join(lines)

    def learn_transition(self, from_emotion: str, to_emotion: str, reward: float = 1.0):
        """
        学习情感转移模式。
        如果从某种情感转移到另一种后获得奖励，增强该转移路径。
        """
        fi, ti = E2I.get(from_emotion), E2I.get(to_emotion)
        if fi is None or ti is None:
            return
        self._transition_counts[fi, ti] += 1 + reward * 2
        self.transition[fi] = self._transition_counts[fi] / self._transition_counts[fi].sum()

    def save(self, path: Optional[str] = None):
        if path is None:
            path = SIDECAR_STATE_DIR / "emotion_state.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "emotions": self.emotions.tolist(),
            "dominant": self._dominant,
            "transition_counts": self._transition_counts.tolist(),
        }
        path.write_text(json.dumps(data, ensure_ascii=False))
        logger.info(f"Emotion state saved to {path}")

    def load(self, path: Optional[str] = None):
        if path is None:
            path = SIDECAR_STATE_DIR / "emotion_state.json"
        path = Path(path)
        if not path.exists():
            return False
        data = json.loads(path.read_text())
        self.emotions = np.array(data["emotions"], dtype=np.float32)
        self._dominant = data["dominant"]
        self._transition_counts = np.array(data["transition_counts"], dtype=np.float32)
        self.transition = self._transition_counts / self._transition_counts.sum(axis=1, keepdims=True)
        logger.info(f"Emotion state loaded from {path}")
        return True


# ── 快捷创建 ────────────────────────────────────────────────

def get_emotional_engine() -> EmotionalEngine:
    return EmotionalEngine()
