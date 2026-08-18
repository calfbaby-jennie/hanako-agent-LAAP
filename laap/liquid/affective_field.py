"""LAAP Liquid Affective Field — 液态情感动力学场

基于 LTC (Liquid Time-Constant) 神经元的情感状态演化。

核心设计：
  - 情感状态作为 LTCCell 隐藏态 h(t) 演化
  - 时间常数 τ 由情绪强度驱动：强情绪 → τ 小 → 快速涌现；平静期 → τ 大 → 缓慢消退
  - 人格敏感度 [2.5, 2.0, 1.5, 2.0, 2.2] 映射为 5 维 τ_base（敏感度高 → τ_base 小 → 反应快）
  - 5 个情感维度：joy, trust, fear, surprise, sadness

LAAP 日志风格：[OK]/[INFO]/[WARN]/[ERROR]
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np

from laap.liquid.neurons import LTCCell

logger = logging.getLogger("laap.liquid.affective_field")

# ── 默认人格敏感度（LAAP 工程约定）──
DEFAULT_PERSONALITY_SENSITIVITIES: List[float] = [2.5, 2.0, 1.5, 2.0, 2.2]

# ── 5 个情感维度名称 ──
DIMENSION_NAMES: List[str] = ["joy", "trust", "fear", "surprise", "sadness"]

# 每维 LTCCell 的隐藏态维度
_HID_DIM_PER = 4


class LiquidAffectiveField:
    """液态情感动力学场

    每个情感维度用一个独立 LTCCell 演化，tau_sys = 1.0 / sensitivity。
    强情绪事件的 intensity 驱动输入信号，使 τ_eff 减小、情感快速涌现。
    无事件时按 dt 自然衰减（输入为 0），τ_eff ≈ τ_sys，缓慢消退。
    """

    def __init__(
        self,
        personality_sensitivities: Optional[List[float]] = None,
    ) -> None:
        if personality_sensitivities is None:
            personality_sensitivities = list(DEFAULT_PERSONALITY_SENSITIVITIES)
        if len(personality_sensitivities) != len(DIMENSION_NAMES):
            raise ValueError(
                f"[ERROR] personality_sensitivities 长度应为 {len(DIMENSION_NAMES)}，"
                f"实际 {len(personality_sensitivities)}"
            )

        self.sensitivities = [float(s) for s in personality_sensitivities]
        self.dimension_names = list(DIMENSION_NAMES)

        # 每维创建 LTCCell，tau_sys = 1/sensitivity（敏感度高→tau小→反应快）
        self._cells: Dict[str, LTCCell] = {}
        self._h: Dict[str, np.ndarray] = {}
        self._tau_base: Dict[str, float] = {}

        for name, sens in zip(self.dimension_names, self.sensitivities):
            tau_sys = 1.0 / max(sens, 1e-3)  # 防止除零
            self._cells[name] = LTCCell(input_dim=1, hidden_dim=_HID_DIM_PER, tau_sys=tau_sys)
            self._h[name] = np.zeros(_HID_DIM_PER, dtype=np.float64)
            self._tau_base[name] = tau_sys

        self._last_intensity: Dict[str, float] = {name: 0.0 for name in self.dimension_names}
        self.last_t: float = time.time()

        logger.info(
            f"[OK] LiquidAffectiveField 初始化，维度={self.dimension_names}，"
            f"tau_base={{ {', '.join(f'{n}={t:.3f}' for n, t in self._tau_base.items())} }}"
        )

    # ── 核心方法 ──────────────────────────────────────────────

    def process_emotion_event(
        self,
        event: dict,
        t_now: Optional[float] = None,
    ) -> np.ndarray:
        """处理情感事件，演化对应维度的 LTCCell。

        参数：
            event  : {"dimension": str, "intensity": float(0-1), "valence": float(-1..1)}
            t_now  : 当前时间戳，默认 time.time()

        返回：
            该维度的新隐状态 (hidden_dim_per,)
        """
        if t_now is None:
            t_now = time.time()

        dim = event.get("dimension", "")
        if dim not in self._cells:
            logger.warning(f"[WARN] 未知情感维度: {dim}，已知: {self.dimension_names}")
            return np.zeros(_HID_DIM_PER)

        intensity = float(np.clip(event.get("intensity", 0.5), 0.0, 1.0))
        valence = float(np.clip(event.get("valence", 1.0), -1.0, 1.0))
        dt = max(t_now - self.last_t, 0.0)

        # 输入信号 = intensity * valence（强度 × 效价）
        x = np.array([intensity * valence], dtype=np.float64)
        new_h = self._cells[dim].forward(self._h[dim], x, dt)
        self._h[dim] = new_h
        self._last_intensity[dim] = intensity
        self.last_t = t_now

        return new_h

        # 解码增益：LTC 驱动项经半隐式欧拉稀释后峰值偏低，
    # 在读出层放大而不改 ODE 动力学（有界性证明不受影响）
    _DECODE_GAIN = 3.0

    def decode_affective_state(self) -> dict:
        """从 h(t) 解码 5 维情感状态，值域 [-1, 1]（tanh 压缩）。"""
        result = {}
        for name in self.dimension_names:
            h_mean = float(np.mean(self._h[name]))
            result[name] = float(np.tanh(h_mean * self._DECODE_GAIN))  # 增益 + tanh 压缩到 [-1, 1]
        return result

    def get_tau_base(self, dimension: str) -> float:
        """返回指定维度的 τ_base（= tau_sys）。"""
        return self._tau_base.get(dimension, 0.0)

    def get_tau_effective(self, dimension: str) -> float:
        """返回指定维度当前有效时间常数（取均值）。"""
        if dimension not in self._cells:
            return 0.0
        return float(np.mean(self._cells[dimension].get_tau_effective()))

    def evolve_idle(self, t_now: Optional[float] = None) -> np.ndarray:
        """无新事件时按 dt 自然衰减演化所有维度。

        空闲时情感应指数衰减到零（无输入=无驱动），而非用 LTC forward
        （LTC 在 x=0 时仍有 sigmoid(0)=0.5 的驱动项，会导致状态增长）。
        因此采用纯指数衰减：h *= exp(-dt / tau_base)。
        """
        if t_now is None:
            t_now = time.time()
        dt = max(t_now - self.last_t, 0.0)

        parts = []
        for name in self.dimension_names:
            tau = self._tau_base[name]
            decay = float(np.exp(-dt / tau))
            self._h[name] = self._h[name] * decay
            parts.append(self._h[name])

        self.last_t = t_now
        return np.concatenate(parts)

    def step(self, t_now: Optional[float] = None) -> None:
        """单演接口，供 LiquidCognitiveCore.evolve_all 调度。"""
        self.evolve_idle(t_now)

    def evolve(self, t_now: Optional[float] = None) -> np.ndarray:
        """统一演化接口（LiquidCognitiveCore.register_field 要求）。

        等价于 evolve_idle，按 dt 自然衰减所有维度。
        """
        return self.evolve_idle(t_now)

    def get_h_summary(self) -> dict:
        """返回情感场状态摘要。"""
        all_h = np.concatenate([self._h[n] for n in self.dimension_names])
        taus = {n: self.get_tau_effective(n) for n in self.dimension_names}
        return {
            "h_norm": float(np.linalg.norm(all_h)),
            "taus": taus,
            "dimensions": len(self.dimension_names),
        }

    def __repr__(self) -> str:
        return (
            f"LiquidAffectiveField(dimensions={self.dimension_names}, "
            f"tau_base={{ {', '.join(f'{n}={t:.3f}' for n, t in self._tau_base.items())} }})"
        )
