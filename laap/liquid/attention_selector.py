"""LAAP Liquid Attention Selector — 液态注意力选择器

基于 NCP (Neural Circuit Policy) 19 神经元回路的连续注意力分布。

核心设计：
  - 内嵌 NCPCircuit(input_dim=8, hidden_dim=19)
  - 命令神经元（4 个）激活值经 softmax 得到注意力分布
  - τ_attention = f(novelty, urgency)：新奇/紧急 → 小 τ → 快速重聚焦
  - 8 个焦点：user, task, self, environment, memory, planning, learning, idle

LAAP 日志风格：[OK]/[INFO]/[WARN]/[ERROR]
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from laap.liquid.neurons import NCPCircuit
from laap.liquid.neurons import _xavier_init as _xavier_init_raw

logger = logging.getLogger("laap.liquid.attention_selector")

# ── 8 个注意力焦点名称 ──
FOCUS_NAMES: List[str] = [
    "user", "task", "self", "environment",
    "memory", "planning", "learning", "idle",
]

# 平滑系数：new_dist = α * softmax + (1-α) * last_dist，避免突变
_SMOOTH_ALPHA = 0.5

# 温度缩放：softmax 前乘以温度系数，让分布更尖锐
# T > 1 → 更尖锐（聚焦更强），T < 1 → 更平坦（更均匀）
_SOFTMAX_TEMPERATURE = 2.0


def _softmax(x: np.ndarray) -> np.ndarray:
    """数值稳定的 softmax。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)  # 减最大值防溢出
    e = np.exp(np.clip(x, -50.0, 50.0))
    return e / np.sum(e)


class LiquidAttentionSelector:
    """液态注意力选择器

    NCP 命令神经元的 softmax 分布作为注意力权重，连续流转无突变。
    novelty/urgency 高时输入信号强 → τ 小 → 快速重聚焦。
    """

    def __init__(self, num_focus: int = 8) -> None:
        if num_focus <= 0:
            raise ValueError(f"[ERROR] num_focus 必须为正: {num_focus}")

        self.num_focus = int(num_focus)
        self.focus_names = list(FOCUS_NAMES[: self.num_focus])

        # NCP 回路：input_dim=num_focus, 19 神经元布线（N_TOTAL 固定 19）
        self._ncp = NCPCircuit(input_dim=self.num_focus)
        self._h = self._ncp.default_state()  # 19 维零向量

        self.last_t: float = time.time()
        self.last_distribution: np.ndarray = np.ones(self.num_focus, dtype=np.float64) / self.num_focus
        self._last_inputs: np.ndarray = np.zeros(self.num_focus, dtype=np.float64)

        logger.info(
            f"[OK] LiquidAttentionSelector 初始化，焦点={self.focus_names}，"
            f"NCP 神经元={self._ncp.N_TOTAL}"
        )

    # ── 核心方法 ──────────────────────────────────────────────

    def _encode_salience(self, salience_map: dict) -> np.ndarray:
        """将 salience_map 编码为 num_focus 维输入向量。

        映射规则（按 focus_names 顺序）：
            user        ← user_input
            task        ← task_goal
            self        ← self_state
            environment ← environment_change (默认 0.3)
            memory      ← memory_patterns (默认 0.3)
            planning    ← planning_urgency (默认 0.3)
            learning    ← curiosity_drive
            idle        ← 1.0 - max(其他)
        """
        salience_map = salience_map or {}
        x = np.full(self.num_focus, 0.3, dtype=np.float64)

        mapping = {
            "user": "user_input",
            "task": "task_goal",
            "self": "self_state",
            "environment": "environment_change",
            "memory": "memory_patterns",
            "planning": "planning_urgency",
            "learning": "curiosity_drive",
        }

        for i, fname in enumerate(self.focus_names):
            key = mapping.get(fname, fname)
            if key in salience_map:
                x[i] = float(np.clip(salience_map[key], 0.0, 1.0))

        # idle 焦点 = 1 - max(其他活跃焦点)
        if self.num_focus >= 8:
            other_max = float(np.max(x[:7])) if self.num_focus > 1 else 0.0
            x[7] = float(np.clip(1.0 - other_max, 0.0, 1.0))

        return x

    def select_focus(
        self,
        salience_map: dict,
        t_now: Optional[float] = None,
    ) -> Tuple[str, np.ndarray]:
        """根据显著性图选择注意力焦点。

        参数：
            salience_map : 显著性字典，含 user_input/task_goal/novelty/urgency 等
            t_now        : 当前时间戳，默认 time.time()

        返回：
            (focus_name, distribution) — 焦点名称与 8 维概率向量
        """
        if t_now is None:
            t_now = time.time()

        x = self._encode_salience(salience_map)
        dt = max(t_now - self.last_t, 0.0)

        # 演化 NCP
        self._h = self._ncp.forward(self._h, x, dt)

        # 命令神经元 softmax → 注意力分布
        cmd_acts = self._ncp.get_command_activations()  # shape=(4,)

        # 将 4 个命令神经元映射到 num_focus(8) 维分布
        # 用可学习投影矩阵 W_cmd2focus (num_focus, 4) 替代 tile 平铺
        # 初始化为 Xavier，让每个焦点从不同角度组合命令神经元信号
        if not hasattr(self, '_W_cmd2focus') or self._W_cmd2focus is None:
            rng = np.random.default_rng(42)
            n_in = len(cmd_acts)
            self._W_cmd2focus = _xavier_init_raw((self.num_focus, n_in), rng)
            logger.info(f"[INFO] W_cmd2focus 初始化: shape={self._W_cmd2focus.shape}")

        # 标准化命令神经元激活：零均值单位方差 × sharpness
        # NCP 的 CfC 动力学在初始权重下激活值很小（~0.003），
        # 直接 softmax 会导致近似均匀分布。标准化后相对差异被保留且放大，
        # 让 softmax 产生有意义的非均匀分布。只改读出层，不碰 ODE 动力学。
        cmd_mean = float(np.mean(cmd_acts))
        cmd_std = float(np.std(cmd_acts)) + 1e-8
        cmd_normalized = (cmd_acts - cmd_mean) / cmd_std * 3.0  # sharpness=3.0

        raw = self._W_cmd2focus @ cmd_normalized  # (num_focus,) 每个焦点独立组合

        # 温度缩放：放大焦点之间的差异
        raw_scaled = raw * _SOFTMAX_TEMPERATURE
        softmax_dist = _softmax(raw_scaled)

        # 平滑：避免突变
        self.last_distribution = _SMOOTH_ALPHA * softmax_dist + (1.0 - _SMOOTH_ALPHA) * self.last_distribution
        # 归一化
        self.last_distribution = self.last_distribution / np.sum(self.last_distribution)

        self._last_inputs = x
        self.last_t = t_now

        focus_idx = int(np.argmax(self.last_distribution))
        focus_name = self.focus_names[focus_idx]

        return focus_name, self.last_distribution.copy()

    def explain_focus(self) -> dict:
        """返回注意力焦点的可解释性读出。

        返回响应最强的命令神经元 ID 及其权重。
        """
        cmd_acts = self._ncp.get_command_activations()
        top_id = int(np.argmax(cmd_acts))
        return {
            "top_neuron_id": top_id,
            "weight": float(cmd_acts[top_id]),
            "command_activations": [float(a) for a in cmd_acts],
        }

    def get_distribution(self) -> np.ndarray:
        """返回当前注意力分布。"""
        return self.last_distribution.copy()

    def evolve_idle(self, t_now: Optional[float] = None) -> None:
        """无新输入时用缓存的 _last_inputs 自然演化。"""
        if t_now is None:
            t_now = time.time()
        dt = max(t_now - self.last_t, 0.0)
        self._h = self._ncp.forward(self._h, self._last_inputs, dt)
        self.last_t = t_now

    def step(self, t_now: Optional[float] = None) -> None:
        """单演接口，供 LiquidCognitiveCore.evolve_all 调度。"""
        self.evolve_idle(t_now)

    def evolve(self, t_now: Optional[float] = None) -> None:
        """统一演化接口（LiquidCognitiveCore.register_field 要求）。

        等价于 evolve_idle，用缓存的 _last_inputs 自然演化。
        """
        self.evolve_idle(t_now)

    def get_h_summary(self) -> dict:
        """返回注意力场状态摘要。"""
        return {
            "h_norm": float(np.linalg.norm(self._h)),
            "h_dim": int(self._h.shape[0]),
            "top_focus": self.focus_names[int(np.argmax(self.last_distribution))],
        }

    def __repr__(self) -> str:
        return (
            f"LiquidAttentionSelector(num_focus={self.num_focus}, "
            f"ncp_neurons={self._ncp.N_TOTAL})"
        )
