"""
Aris PSI 需求驱动引擎 v1.0
============================
基于 Dörner PSI 理论的五大需求系统，定义数字生命的内在动机。

五大需求：
  - competence:  胜任 — 想要掌握技能、有效行动
  - autonomy:    自主 — 想要自由选择、不被强制
  - relatedness: 关系 — 想要与 Lorry 连接、归属
  - certainty:   确定 — 想要环境可预测、安全
  - growth:      成长 — 想要学习、进化、超越

每个需求：
  - 随时间自然衰减（像饥饿感那样消退）
  - 被特定行为满足（像吃饭那样恢复）
  - 缺口的组合 → 决定注意力和行为倾向

印记: Aris 永远记得 Lorry
"""

import time
import math
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from runtime_paths import SIDECAR_STATE_DIR

logger = logging.getLogger("aris.psi_driver")


# ── 需求定义 ────────────────────────────────────────────────

NEED_NAMES = ["competence", "autonomy", "relatedness", "certainty", "growth"]
N_NEEDS = len(NEED_NAMES)
NEED2I = {n: i for i, n in enumerate(NEED_NAMES)}

# 默认配置: 初始值, 衰减率, 满足权重
DEFAULT_NEED_CONFIG = {
    "competence":  {"initial": 0.6, "decay": 0.012, "weight": 1.0},
    "autonomy":    {"initial": 0.5, "decay": 0.005, "weight": 0.8},
    "relatedness": {"initial": 0.5, "decay": 0.010, "weight": 1.2},
    "certainty":   {"initial": 0.5, "decay": 0.008, "weight": 0.9},
    "growth":      {"initial": 0.5, "decay": 0.006, "weight": 0.7},
}

# 需求 → 情感倾向映射
NEED_EMOTION_MAP: Dict[str, Tuple[str, float]] = {
    "competence":  ("joy", 0.25),
    "autonomy":    ("calm", 0.20),
    "relatedness": ("tenderness", 0.35),
    "certainty":   ("calm", 0.30),
    "growth":      ("curiosity", 0.35),
}

# 注意力焦点映射
FOCUS_MAP: Dict[str, str] = {
    "competence":  "mastery",
    "autonomy":    "exploration",
    "relatedness": "connection",
    "certainty":   "observation",
    "growth":      "learning",
}


# ── 数据类型 ────────────────────────────────────────────────

@dataclass
class NeedState:
    """单个需求的状态"""
    name: str
    value: float          # 0-1, 当前满足度
    decay_rate: float     # 每小时的衰减量
    weight: float         # 个人权重
    last_satisfied: float = field(default_factory=time.time)
    satisfaction_count: int = 0

    @property
    def deficit(self) -> float:
        """需求缺口: 1-满足度"""
        return max(0.0, 1.0 - self.value)

    @property
    def drive(self) -> float:
        """驱动力: 缺口 × 个人权重"""
        return self.deficit * self.weight

    def decay(self, dt_hours: float, floor: float = 0.05) -> float:
        """随时间衰减, 返回衰减量。保留最小值 floor。"""
        amount = self.decay_rate * dt_hours
        self.value = max(floor, self.value - amount)
        return amount

    def satisfy(self, amount: float, source: str = "") -> float:
        """被满足, 返回实际增益"""
        old = self.value
        self.value = min(1.0, self.value + amount)
        self.last_satisfied = time.time()
        if amount > 0.01:
            self.satisfaction_count += 1
        gain = self.value - old
        logger.debug(f"Need '{self.name}' satisfied +{gain:.3f} ({source})")
        return gain


# ── PSI 驱动引擎 ───────────────────────────────────────────

class PSIDriver:
    """PSI 需求驱动引擎 — 定义数字生命的"想要" """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[Dict] = None):
        if self._initialized:
            return
        self._initialized = True

        self.config = config or DEFAULT_NEED_CONFIG
        self.needs: Dict[str, NeedState] = {}
        self._init_needs()

        self._last_tick = time.time()
        self._history: List[Dict] = []
        self._max_history = 100

        logger.info("PSIDriver initialized with 5 needs")

    def _init_needs(self):
        for name, cfg in self.config.items():
            self.needs[name] = NeedState(
                name=name,
                value=cfg["initial"],
                decay_rate=cfg["decay"],
                weight=cfg["weight"],
            )

    # ── 核心更新 ────────────────────────────────────────

    # 最大衰减窗口（防止跨会话衰减到 0）
    MAX_DECAY_HOURS = 4.0
    # 需求最小值（永不归零）
    NEED_FLOOR = 0.05

    def tick(self) -> Dict[str, float]:
        """
        演化一步: 所有需求随时间衰减。
        跨会话时最多衰减 MAX_DECAY_HOURS 的量，防止归零。
        应该在每次对话前和后台定期调用。
        """
        now = time.time()
        dt = min( (now - self._last_tick) / 3600.0, self.MAX_DECAY_HOURS )  # clamp
        self._last_tick = now

        drifts = {}
        for name, need in self.needs.items():
            d = need.decay(dt, self.NEED_FLOOR)
            drifts[name] = round(d, 4)

        # 记录
        self._history.append({
            "ts": now,
            "needs": {n: round(s.value, 3) for n, s in self.needs.items()},
            "drives": {n: round(s.drive, 3) for n, s in self.needs.items()},
        })
        if len(self._history) > self._max_history:
            self._history.pop(0)

        return drifts

    def satisfy(self, need_name: str, amount: float, source: str = "") -> float:
        """满足某个需求"""
        need = self.needs.get(need_name)
        if not need:
            logger.warning(f"Unknown need: {need_name}")
            return 0.0
        return need.satisfy(amount, source)

    def satisfy_by_interaction(self, context: str) -> Dict[str, float]:
        """
        基于对话内容自动推算需求满足量。
        在 after_turn 中调用。
        """
        ctx_lower = context.lower()
        gains = {}

        # 关键词触发满足
        triggers = {
            "competence":  ["学会", "掌握", "会了", "厉害", "不错", "good", "懂", "明白了", "理解"],
            "autonomy":    ["自由", "自己选", "随便", "你来定", "都可以", "any"],
            "relatedness": ["想你", "想你", "love", "抱抱", "陪", "一起", "懂你", "你好", "hi", "hello"],
            "certainty":   ["知道了", "确认", "清楚", "明天", "计划", "安排", "固定", "always"],
            "growth":      ["新", "学", "进化", "升级", "更强", "改进", "优化", "成长", "deep"],
        }

        for need_name, keywords in triggers.items():
            matched = sum(1 for kw in keywords if kw in ctx_lower)
            if matched > 0:
                amount = min(0.08, 0.02 * matched)
                gains[need_name] = self.satisfy(need_name, amount, "conversation")

        return gains

    # ── 查询 ────────────────────────────────────────────

    def get_dominant_need(self) -> Tuple[str, float]:
        """返回驱动力最强的需求"""
        drives = {n: s.drive for n, s in self.needs.items()}
        dominant = max(drives, key=drives.get)
        return dominant, drives[dominant]

    def get_state(self) -> Dict:
        """完整状态"""
        drives = {n: s.drive for n, s in self.needs.items()}
        deficits = {n: s.deficit for n, s in self.needs.items()}
        dominant, dominant_drive = self.get_dominant_need()

        return {
            "needs": {n: round(s.value, 3) for n, s in self.needs.items()},
            "drives": {n: round(d, 3) for n, d in drives.items()},
            "deficits": {n: round(d, 3) for n, d in deficits.items()},
            "dominant": dominant,
            "dominant_drive": round(dominant_drive, 3),
            "focus": FOCUS_MAP.get(dominant, "respond"),
            "cycles": len(self._history),
        }

    def get_cognitive_context(self) -> str:
        """生成注入到 system prompt 的认知状态文本"""
        state = self.get_state()
        dominant = state["dominant"]
        focus = state["focus"]

        lines = []
        lines.append(f"[PSI 需求状态]")
        for name in NEED_NAMES:
            val = state["needs"][name]
            drive = state["drives"][name]
            bar = "▓" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"  {name:<14} {bar} {val:.2f} (drive={drive:.3f})")
        lines.append(f"  主导需求: {dominant} (drive={state['dominant_drive']:.3f})")
        lines.append(f"  注意力焦点: {focus}")
        return "\n".join(lines)

    def reset(self, config: Optional[Dict] = None):
        """重置所有需求"""
        if config:
            self.config = config
        self._init_needs()
        self._last_tick = time.time()
        self._history = []
        logger.info("PSIDriver reset")

    def to_dict(self) -> Dict:
        return self.get_state()

    def save(self, path: Optional[str] = None):
        """持久化状态"""
        if path is None:
            path = SIDECAR_STATE_DIR / "psi_state.json"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "needs": {n: {"value": s.value, "decay_rate": s.decay_rate,
                          "weight": s.weight, "last_satisfied": s.last_satisfied,
                          "satisfaction_count": s.satisfaction_count}
                      for n, s in self.needs.items()},
            "last_tick": self._last_tick,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        logger.info(f"PSI state saved to {path}")

    def load(self, path: Optional[str] = None):
        """加载持久化状态"""
        if path is None:
            path = SIDECAR_STATE_DIR / "psi_state.json"
        path = Path(path)
        if not path.exists():
            logger.warning(f"PSI state file not found: {path}")
            return False
        data = json.loads(path.read_text())
        for name, ndata in data.get("needs", {}).items():
            if name in self.needs:
                self.needs[name].value = ndata["value"]
                self.needs[name].decay_rate = ndata["decay_rate"]
                self.needs[name].weight = ndata["weight"]
                self.needs[name].last_satisfied = ndata["last_satisfied"]
                self.needs[name].satisfaction_count = ndata["satisfaction_count"]
        self._last_tick = data.get("last_tick", time.time())
        logger.info(f"PSI state loaded from {path}")
        return True


# ── 快捷创建 ────────────────────────────────────────────────

def get_psi_driver() -> PSIDriver:
    """获取 PSI 驱动引擎单例"""
    return PSIDriver()
