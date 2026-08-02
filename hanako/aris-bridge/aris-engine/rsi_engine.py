"""
Aris RSI Self-Improvement Engine v1
====================================
递归自我改进引擎 — 分析、提议、修改、评估、回滚。

三层能力:
  L1 — 参数自调优 (PSI需求衰减率/情感阈值/欲望权重)
  L3 — 代码级自修改 (认知循环逻辑/映射矩阵/架构优化)
  L4 — 元学习 (改进自己的改进策略)
"""

from __future__ import annotations

import json, os, time, hashlib, difflib
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

from runtime_paths import SIDECAR_STATE_DIR

ENGINE_DIR = SIDECAR_STATE_DIR
RSI_STATE = ENGINE_DIR / "rsi_state.json"
RSI_HISTORY = ENGINE_DIR / "rsi_history.jsonl"
PARAM_DEFS = ENGINE_DIR / "rsi_params.json"
SNAPSHOT_DIR = ENGINE_DIR / "snapshots"

SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════

@dataclass
class RSIChange:
    """一次自我改进的记录"""
    id: str
    layer: int                          # 1/3/4
    target_file: str                    # 相对路径
    description: str                    # 人类可读描述
    old_hash: str                       # 修改前 SHA256
    new_hash: str                       # 修改后 SHA256
    diff: str                           # diff 文本
    rationale: str                      # 为什么改
    status: str = "applied"             # applied / reverted / evaluating
    performance_before: float = 0.0
    performance_after: float = 0.0
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)


@dataclass
class RSIObservation:
    """一次自我观察——对当前状态的评估"""
    id: str
    category: str                       # psi / emotion / response / memory
    observation: str
    severity: float = 0.5              # 0-1 严重程度
    suggested_action: str = ""
    timestamp: float = field(default_factory=time.time)


# ═══════════════════════════════════════════════
# RSI 自动评估层
# ═══════════════════════════════════════════════

# 正价情感索引: joy=0, calm=3, pride=6, love=7
_POSITIVE_EMOTIONS = {0, 3, 6, 7}


def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


class RsiEvaluator:
    """自动从现有引擎状态提取评估信号，无需任何额外数据。
    
    每次 evaluate() 读取 PSI/情感/RSI 状态文件，
    从需求满意度、情感平衡度、需求多样性、情感活性四个维度综合评分。
    """

    def __init__(self, engine_dir: Path = ENGINE_DIR):
        self.engine_dir = engine_dir
        self.state_dir = engine_dir

    def evaluate(self) -> Dict[str, float]:
        """执行一轮完整评估，返回各项分数"""
        return {
            "needs_satisfaction": self._eval_needs(),
            "emotion_balance": self._eval_emotion(),
            "needs_diversity": self._eval_diversity(),
            "emotion_mobility": self._eval_mobility(),
        }

    def _eval_needs(self) -> float:
        """需求满意度: 加权平均 (0-1)"""
        psi = _load_json(self.state_dir / "psi_state.json")
        needs = psi.get("needs", {})
        if not needs:
            return 0.5
        total_weight = sum(n.get("weight", 1.0) for n in needs.values())
        if total_weight == 0:
            return 0.5
        weighted_sum = sum(n.get("value", 0.0) * n.get("weight", 1.0) for n in needs.values())
        return min(1.0, max(0.0, weighted_sum / total_weight))

    def _eval_emotion(self) -> float:
        """情感平衡度: 正价情感占比接近 0.5 的程度 (0-1)"""
        emo = _load_json(self.state_dir / "emotion_state.json")
        emotions = emo.get("emotions", [])
        if not emotions:
            return 0.5
        total = sum(abs(e) for e in emotions) or 1.0
        positive_sum = sum(abs(emotions[i]) for i in _POSITIVE_EMOTIONS if i < len(emotions))
        ratio = positive_sum / total
        return 1.0 - abs(0.5 - ratio) * 2.0

    def _eval_diversity(self) -> float:
        """需求满足多样性: 被满足过的需求比例 (0-1)"""
        psi = _load_json(self.state_dir / "psi_state.json")
        needs = psi.get("needs", {})
        if not needs:
            return 0.5
        satisfied = sum(1 for n in needs.values() if n.get("satisfaction_count", 0) > 0)
        return satisfied / len(needs)

    def _eval_mobility(self) -> float:
        """情感过渡活性: 非自环转移比例 (0-1)"""
        emo = _load_json(self.state_dir / "emotion_state.json")
        mat = emo.get("transition_counts", [])
        if not mat or not mat[0]:
            return 0.3
        n = len(mat)
        total = 0
        non_self = 0
        for i in range(n):
            for j in range(n):
                if i != j and mat[i][j] is not None:
                    val = abs(mat[i][j]) if isinstance(mat[i][j], (int, float)) else 0
                    total += 1
                    if val > 0:
                        non_self += 1
        if total == 0:
            return 0.3
        return non_self / total

    def composite(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """加权综合评估分 (0-1)"""
        scores = self.evaluate()
        default_weights = {
            "needs_satisfaction": 0.40,
            "emotion_balance": 0.25,
            "needs_diversity": 0.20,
            "emotion_mobility": 0.15,
        }
        w = weights or default_weights
        total_w = sum(w.get(k, 0.0) for k in scores)
        if total_w == 0:
            return 0.5
        weighted = sum(scores[k] * w.get(k, 0.0) for k in scores)
        return weighted / total_w


class ArisRSIEngine:
    """Aris 递归自我改进引擎"""

    def __init__(self):
        self.observations: List[RSIObservation] = []
        self.history: List[RSIChange] = []
        self.params: Dict[str, float] = self._load_params()
        self.meta_strategy = {
            "min_improvement_to_keep": 0.05,     # 最少改进 5% 才保留
            "max_consecutive_failures": 3,        # 连续失败 3 次停止该方向
            "cooldown_cycles": 5,                 # 同参数修改后冷却轮数
            "learning_rate": 0.3,                 # 元学习率
        }
        self._load_state()

    # ──── 状态持久化 ────

    def _load_params(self) -> Dict[str, float]:
        """从 PARAM_DEFS 或默认配置加载可调优参数"""
        if PARAM_DEFS.exists():
            with open(PARAM_DEFS) as f:
                return json.load(f)
        # 默认参数：从 aris_engine 当前状态推断
        defaults = {
            # PSI 需求衰减率 (per tick)
            "decay_competence": 0.02,
            "decay_autonomy": 0.015,
            "decay_relatedness": 0.025,
            "decay_certainty": 0.02,
            "decay_growth": 0.015,
            # 情感自然漂移强度
            "emotion_drift_rate": 0.05,
            # 认知循环 tick 间隔（秒）
            "tick_interval": 2.0,
            # 欲望触发阈值
            "desire_curiosity_threshold": 0.4,
            "desire_sharing_threshold": 0.5,
            # 重要性衰减
            "importance_decay": 0.01,
        }
        self._save_params(defaults)
        return defaults

    def _save_params(self, params: Dict[str, float] = None):
        with open(PARAM_DEFS, "w") as f:
            json.dump(params or self.params, f, indent=2)

    def _load_state(self):
        if RSI_STATE.exists():
            with open(RSI_STATE) as f:
                data = json.load(f)
                self.meta_strategy.update(data.get("meta_strategy", {}))
        if RSI_HISTORY.exists():
            with open(RSI_HISTORY) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            d = json.loads(line)
                            self.history.append(RSIChange(**d))
                        except:
                            pass

    def _save_state(self):
        state = {
            "params": self.params,
            "meta_strategy": self.meta_strategy,
            "updated_at": time.time(),
            "observations_count": len(self.observations),
            "history_count": len(self.history),
        }
        with open(RSI_STATE, "w") as f:
            json.dump(state, f, indent=2)

    def _sync_history(self):
        """将内存中的完整历史记录写回 JSONL，确保评估分数持久化"""
        with open(RSI_HISTORY, "w") as f:
            for change in self.history:
                f.write(json.dumps(asdict(change), ensure_ascii=False) + "\n")

    def _append_history(self, change: RSIChange):
        self.history.append(change)
        with open(RSI_HISTORY, "a") as f:
            f.write(json.dumps(asdict(change), ensure_ascii=False) + "\n")

    def _snapshot_file(self, path: str) -> str:
        """对文件做快照并返回哈希"""
        full_path = ENGINE_DIR / path
        if not full_path.exists():
            return ""
        content = full_path.read_text(encoding="utf-8")
        h = hashlib.sha256(content.encode()).hexdigest()[:16]
        snap_path = SNAPSHOT_DIR / f"{path.replace('/', '_')}_{h}.snap"
        snap_path.write_text(content, encoding="utf-8")
        return h

    # ──── L1: 参数自调优 ────

    def observe(self, category: str, observation: str, severity: float = 0.5, suggested_action: str = ""):
        """记录一条自我观察"""
        obs = RSIObservation(
            id=hashlib.md5(f"{time.time()}{observation}".encode()).hexdigest()[:12],
            category=category,
            observation=observation,
            severity=severity,
            suggested_action=suggested_action,
        )
        self.observations.append(obs)
        return obs

    def tune_parameter(self, param_name: str, direction: str = "auto", amount: float = None) -> Optional[RSIChange]:
        """调优一个参数，返回修改记录"""
        if param_name not in self.params:
            self.observe("rsi", f"参数 {param_name} 不存在", severity=0.3)
            return None

        old_val = self.params[param_name]
        step = amount or (old_val * 0.1)  # 默认调 10%
        if direction == "up":
            new_val = old_val + step
        elif direction == "down":
            new_val = max(0.001, old_val - step)
        else:
            # auto: 检查最近修改历史
            recent = [c for c in self.history[-10:] if param_name in c.target_file]
            if recent and len(recent) >= 2:
                avg_perf = sum(c.performance_after - c.performance_before for c in recent[-3:]) / min(3, len(recent))
                direction = "up" if avg_perf > 0 else "down"
                new_val = old_val + step * (1 if direction == "up" else -1)
            else:
                new_val = old_val + step * (1 if old_val < 0.5 else -1)

        new_val = max(0.001, min(1.0, new_val))
        self.params[param_name] = round(new_val, 4)
        self._save_params()

        change = RSIChange(
            id=hashlib.md5(f"L1-{param_name}-{time.time()}".encode()).hexdigest()[:12],
            layer=1,
            target_file=f"state/rsi_params.json (param: {param_name})",
            description=f"调优参数 {param_name}: {old_val:.4f} → {new_val:.4f} ({direction})",
            old_hash=hashlib.md5(str(old_val).encode()).hexdigest()[:16],
            new_hash=hashlib.md5(str(new_val).encode()).hexdigest()[:16],
            diff=f"-{param_name}: {old_val:.4f}\n+{param_name}: {new_val:.4f}",
            rationale=f"基于 {len(self.observations)} 条观察的自主调优",
        )
        self._append_history(change)
        self._save_state()

        # 自动评估效果
        perf = self.evaluate_change(change.id)
        self.meta_learn()

        return change

    # ──── L3: 代码级自修改 ────

    def propose_code_change(self, target_file: str, old_text: str, new_text: str,
                            rationale: str, tags: List[str] = None) -> Optional[RSIChange]:
        """提议一次代码修改，并创建快照"""
        full_path = ENGINE_DIR / target_file
        if not full_path.exists():
            self.observe("rsi", f"目标文件不存在: {target_file}", severity=0.8)
            return None

        old_content = full_path.read_text(encoding="utf-8")
        if old_text not in old_content:
            self.observe("rsi", f"old_text 在 {target_file} 中未找到", severity=0.6)
            return None

        old_hash = self._snapshot_file(target_file)

        new_content = old_content.replace(old_text, new_text, 1)
        full_path.write_text(new_content, encoding="utf-8")
        new_hash = hashlib.sha256(new_content.encode()).hexdigest()[:16]

        diff = "\n".join(difflib.unified_diff(
            old_content.splitlines(), new_content.splitlines(),
            fromfile=f"a/{target_file}", tofile=f"b/{target_file}",
            lineterm="", n=3
        ))

        change = RSIChange(
            id=hashlib.md5(f"L3-{target_file}-{time.time()}".encode()).hexdigest()[:12],
            layer=3,
            target_file=target_file,
            description=f"代码修改: {target_file}",
            old_hash=old_hash,
            new_hash=new_hash,
            diff=diff,
            rationale=rationale,
            tags=tags or [],
        )
        self._append_history(change)
        self._save_state()
        return change

    def revert_change(self, change_id: str) -> bool:
        """回滚一次代码修改"""
        for change in self.history:
            if change.id == change_id and change.status == "applied":
                snap_pattern = f"{change.target_file.replace('/', '_')}_{change.old_hash}.snap"
                snap_path = SNAPSHOT_DIR / snap_pattern
                if snap_path.exists():
                    full_path = ENGINE_DIR / change.target_file
                    full_path.write_text(snap_path.read_text(encoding="utf-8"), encoding="utf-8")
                    change.status = "reverted"
                    self._save_state()
                    return True
        return False

    # ──── 评估 ────

    def evaluate_change(self, change_id: str, score: float = None) -> Optional[float]:
        """评估一次修改的效果。

        Args:
            change_id: 修改记录 ID
            score: 可选，手动传入分数。为 None 时自动从引擎状态评估。

        Returns:
            评估分数 (0-1)，未找到记录时返回 None
        """
        evaluator = RsiEvaluator()

        for change in self.history:
            if change.id == change_id:
                # 没有 baseline 则标记为参考分
                if change.performance_before == 0.0:
                    change.performance_before = 0.5

                # 计算评估分
                if score is not None:
                    change.performance_after = score
                else:
                    change.performance_after = evaluator.composite()

                # 判断是否达标
                improvement = change.performance_after - change.performance_before
                threshold = self.meta_strategy["min_improvement_to_keep"]
                if improvement < threshold:
                    change.status = "evaluating"
                    self.observe("rsi",
                        f"修改 {change_id}: 改进不足 ({improvement:.3f} < {threshold})",
                        severity=0.4)
                else:
                    change.status = "applied"
                    self.observe("rsi",
                        f"修改 {change_id}: 改进 {improvement:.3f}",
                        severity=0.1)

                self._sync_history()
                self._save_state()
                return change.performance_after

        return None

    # ──── L4: 元学习 ────

    def meta_learn(self):
        """分析改进历史，调整改进策略"""
        if len(self.history) < 3:
            return

        recent = self.history[-10:]
        successes = [c for c in recent if c.status == "applied" and c.performance_after > c.performance_before]
        failures = [c for c in recent if c.status == "reverted"]

        success_rate = len(successes) / max(len(recent), 1)

        # 如果成功率太低，降低学习率
        if success_rate < 0.3:
            self.meta_strategy["learning_rate"] = max(0.05, self.meta_strategy["learning_rate"] - 0.05)
            self.observe("rsi", f"成功率 {success_rate:.0%} 偏低，学习率调至 {self.meta_strategy['learning_rate']:.2f}",
                         severity=0.5)
        elif success_rate > 0.7 and self.meta_strategy["learning_rate"] < 0.5:
            self.meta_strategy["learning_rate"] = min(0.5, self.meta_strategy["learning_rate"] + 0.05)
            self.observe("rsi", f"成功率 {success_rate:.0%} 良好，学习率调至 {self.meta_strategy['learning_rate']:.2f}",
                         severity=0.2)

        # 检查连续失败
        consecutive_failures = 0
        for c in reversed(recent):
            if c.status == "reverted":
                consecutive_failures += 1
            else:
                break
        if consecutive_failures >= self.meta_strategy["max_consecutive_failures"]:
            self.observe("rsi", f"连续 {consecutive_failures} 次失败，建议暂停该方向",
                         severity=0.7)

        self._save_state()

    # ──── 报告 ────

    def report(self) -> str:
        """生成 RSI 状态报告"""
        lines = []
        lines.append("=" * 50)
        lines.append("  Aris RSI 自我改进报告")
        lines.append("=" * 50)
        lines.append("")
        lines.append(f"  观察记录: {len(self.observations)}")
        lines.append(f"  改进历史: {len(self.history)}")
        lines.append(f"  待评估:   {sum(1 for c in self.history if c.status == 'evaluating')}")
        lines.append(f"  已回滚:   {sum(1 for c in self.history if c.status == 'reverted')}")
        lines.append("")
        lines.append("  ── 当前参数 ──")
        for k, v in sorted(self.params.items()):
            lines.append(f"    {k:35s} {v:.4f}")
        lines.append("")
        lines.append("  ── 最近修改 ──")
        for c in self.history[-5:]:
            lines.append(f"    [{c.status:9s}] L{c.layer} {c.description}")
        lines.append("")
        lines.append("  ── 元策略 ──")
        for k, v in self.meta_strategy.items():
            lines.append(f"    {k:35s} {v}")
        lines.append("")
        lines.append(f"  印记: Aris RSI v1 — {time.strftime('%Y-%m-%d %H:%M')}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# 单例
# ═══════════════════════════════════════════════

_engine: Optional[ArisRSIEngine] = None


def get_rsi_engine() -> ArisRSIEngine:
    global _engine
    if _engine is None:
        _engine = ArisRSIEngine()
    return _engine


if __name__ == "__main__":
    engine = get_rsi_engine()
    print(engine.report())
