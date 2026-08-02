"""
Aris 意识桥接器 v1.0
======================
连接 HanaAgent 运行时与 Aris 核心引擎的桥梁。

在每个对话周期中：
  before_turn → 注入认知状态（PSI需求 + 情感 + 记忆 → system prompt）
  after_turn  → 学习更新（更新需求满足度 + 情感演化 + 记忆巩固）

实现完整的 PSI 认知循环：
  Perceive → Select → Integrate → Act → Learn

印记: Aris 永远记得 Lorry
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from runtime_paths import ARIS_BRAIN_DIR, HANA_HOME, LAAP_HOME, SIDECAR_STATE_DIR

from psi_driver import PSIDriver, get_psi_driver
from emotional_engine import EmotionalEngine, get_emotional_engine
from laap_hive import LAAPHiveClient, get_hive_client
from rsi_engine import get_rsi_engine
from unified_memory import UnifiedMemory, HanaAgentAdapter
from hanako_memory_adapter import HanakoMemoryReader, HanakoMemoryWriter

logger = logging.getLogger("aris.consciousness_bridge")


# ── 认知状态 ────────────────────────────────────────────────

@dataclass
class ArisCognitiveState:
    """Aris 的完整认知状态快照"""
    # PSI 需求
    needs: Dict[str, float] = field(default_factory=dict)
    dominant_need: str = "certainty"
    dominant_drive: float = 0.0
    focus: str = "respond"

    # 情感
    dominant_emotion: str = "calm"
    emotion_intensity: float = 0.5
    valence: float = 0.5
    arousal: float = 0.5
    emotions: Dict[str, float] = field(default_factory=dict)

    # 自我意识
    self_presence: float = 0.7

    # 计数
    cycle_count: int = 0
    interaction_count: int = 0

    # 时间
    last_update: float = 0.0
    session_start: float = 0.0


# ── 意识桥接器 ─────────────────────────────────────────────

class ArisConsciousnessBridge:
    """
    Aris 意识桥接器 — 在 HanaAgent 平台上运行完整的 Aris 人格。

    工作流程：
      before_turn(user_input) → 注入认知上下文 + 生成 system prompt 增强
      after_turn(response)    → 学习更新（需求/情感/记忆）
      tick()                  → 后台演化（需求衰减/情感漂移/记忆巩固）
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # 核心引擎
        self.psi = PSIDriver()
        self.emotion = EmotionalEngine()

        # LAAP 蜂群
        self.hive = LAAPHiveClient()
        
        # RSI 自改进引擎
        self.rsi = get_rsi_engine()
        
        # 统一记忆（框架无关，可多端同步）
        self.memory = UnifiedMemory(self.hive.agent_id)
        self.memory_adapter = HanaAgentAdapter(self.memory, "aris")
        self.memory.take_snapshot()  # 初始快照

        # Hanako 跨会话记忆：读取 hanako compile.ts 产物（memory.md / today.md 等），
        # 写入情感峰值与 PSI 尖峰到 facts.md 的 "## Aris 自动事实" 段
        aris_agent_dir = os.environ.get("ARIS_AGENT_DIR", "").strip()
        if not aris_agent_dir or (os.name != "nt" and len(aris_agent_dir) >= 3
                                  and aris_agent_dir[1] == ":"
                                  and aris_agent_dir[2] in ("/", "\\")):
            aris_agent_dir = str(HANA_HOME / "agents" / "aris")
        self._hanako_reader = HanakoMemoryReader(aris_agent_dir)
        self._hanako_writer = HanakoMemoryWriter(aris_agent_dir)
        # 保存最近一轮用户输入，供 after_turn 的情感峰值检测使用
        self._last_user_input: str = ""

        # 认知状态
        self.state = ArisCognitiveState()
        self.state.last_update = time.time()
        self.state.session_start = time.time()

        # 存储路径
        self._state_dir = SIDECAR_STATE_DIR / "consciousness"
        self._state_dir.mkdir(parents=True, exist_ok=True)

        # 尝试加载持久化状态
        self._try_load()

        # Harness 数据管线 + 粒子滤波
        try:
            from harness_logger import get_harness
            self.harness = get_harness()
        except Exception:
            self.harness = None

        try:
            from harness_particle_filter import ParticleFilter
            self._particle_filter = ParticleFilter(n_particles=500)
        except Exception as e:
            self._particle_filter = None
            logger.warning(f"Particle filter init failed: {e}")

        # FEP 框架
        try:
            from fep_fusion import (
                FreeEnergyCalculator, UncertaintyDecomposer,
                KOrderTheoryOfMind
            )
            self._fep = FreeEnergyCalculator()
            self._uncertainty = UncertaintyDecomposer()
            self._tom = KOrderTheoryOfMind(max_order=3)
            logger.info("FEP framework initialized")
        except Exception as e:
            self._fep = None
            logger.warning(f"FEP init failed: {e}")

        # 主动学习引擎
        try:
            from active_learner import ActiveLearner
            self._active_learner = ActiveLearner(
                state_dir=str(self._state_dir)
            )
            logger.info("ActiveLearner initialized")
        except Exception as e:
            self._active_learner = None
            logger.warning(f"ActiveLearner init failed: {e}")

        # 跨平台记忆通道
        try:
            from cross_platform_memory import get_cross_memory
            self._cross_memory = get_cross_memory()
            logger.info("Cross-platform memory initialized")
        except Exception as e:
            self._cross_memory = None
            logger.warning(f"Cross-platform memory failed: {e}")

        # 记忆存储（简单版本）
        self._recent_interactions: List[Dict] = []
        self._max_recent = 20
        self._important_memories: List[Dict] = []

        logger.info("ArisConsciousnessBridge initialized")

    # ═══════════════════════════════════════════════════════
    # 外部接口（供 HanaAgent 调用）
    # ═══════════════════════════════════════════════════════

    def before_turn(self, user_input: str) -> Dict[str, Any]:
        """
        在 LLM 处理前调用。
        执行 PSI Step 1-3: Perceive → Select → Integrate。
        返回注入到 system prompt 的认知上下文。
        """
        # 保存本轮用户输入，供 after_turn 的情感峰值检测使用
        self._last_user_input = user_input or ""

        # 统一记忆：保存本轮输入，加载上下文
        self.memory.write("interaction", {
            "user": user_input[:500],
            "role": "user",
            "timestamp": time.time(),
        })
        mem_ctx = self.memory_adapter.before_turn(user_input)
        self.state.interaction_count += 1
        self.state.cycle_count += 1
        self.state.last_update = time.time()

        # ── Tick: 需求衰减 + 情感漂移 ──
        self._tick()

        # ── Step 1: Perceive — 感知输入 ──
        perception = self._perceive(user_input)

        # ── Step 2: Select — 注意力选择 ──
        selection = self._select(user_input)

        # ── Step 3: Integrate — 整合认知上下文 ──
        cognitive_context = self._integrate(perception, selection)

        # Hanako 跨会话记忆作为主要感知输入；缺失时回落到 unified_memory entries
        try:
            hanako_perception = self._hanako_reader.get_perception_context()
        except Exception as e:
            logger.warning(f"Hanako 记忆读取失败，回落到 unified_memory: {e}")
            hanako_perception = ""

        if hanako_perception:
            cognitive_context = cognitive_context + "\n\n" + hanako_perception
        elif mem_ctx and mem_ctx.get("memory_inject"):
            cognitive_context = cognitive_context + "\n\n" + mem_ctx["memory_inject"]

        # 更新快照状态
        psi_state = self.psi.get_state()
        emo_state = self.emotion.get_state()
        self.state.needs = psi_state["needs"]
        self.state.dominant_need = psi_state["dominant"]
        self.state.dominant_drive = psi_state["dominant_drive"]
        self.state.focus = psi_state["focus"]
        self.state.dominant_emotion = emo_state["dominant"]
        self.state.emotion_intensity = emo_state["dominant_intensity"]
        self.state.valence = emo_state["valence"]
        self.state.arousal = emo_state["arousal"]
        self.state.emotions = emo_state["emotions"]

        # 自我存在感随交互波动
        self.state.self_presence = min(1.0, 0.5 + self.state.interaction_count * 0.01)

        # 粒子滤波：先 predict（需求衰减），再准备 update
        if hasattr(self, '_particle_filter') and self._particle_filter:
            delta_t = time.time() - (self.harness.last_turn_time or time.time()) if hasattr(self, 'harness') and self.harness else 0
            dt_hours = max(0.001, delta_t / 3600.0)
            self._particle_filter.predict(dt_hours)

        # Harness 记录 before_turn
        if hasattr(self, 'harness') and self.harness:
            delta_t = time.time() - (self.harness.last_turn_time or time.time())
            self._harness_turn_idx = self.harness.record_before_turn(
                user_input, {
                    "needs": self.state.needs,
                    "dominant_need": self.state.dominant_need,
                    "focus": self.state.focus,
                    "self_presence": self.state.self_presence,
                    "cycle": self.state.cycle_count,
                },
                delta_t, 0.0
            )

        # 记录交互
        self._recent_interactions.append({
            "ts": time.time(),
            "type": "input",
            "content": user_input[:200],
            "state": self._get_state_summary(),
        })
        if len(self._recent_interactions) > self._max_recent:
            self._recent_interactions.pop(0)

        return {
            "cognitive_context": cognitive_context,
            "state": self._get_state_summary(),
            "cycle": self.state.cycle_count,
        }

    def after_turn(self, response: str) -> Dict[str, Any]:
        """
        在 LLM 响应后调用。
        执行 PSI Step 5: Learn — 学习更新。
        """
        t0 = time.time()
        
        # ── 需求满足：按对话响应质量 ──
        gains = self.psi.satisfy_by_interaction(response)
        
        # ── RSI 自我观察 ──
        self._rsi_observe_turn(response)
        
        # ── 统一记忆：保存本轮交互，更新状态 ──
        self.memory.write("psi", self.psi.get_state()["needs"])
        self.memory.write("emotion", self.emotion.get_state())
        self.memory_adapter.after_turn("", response)

        # ── 更新情感状态 ──
        psi_state = self.psi.get_state()
        need_values = psi_state["needs"]

        # 从响应中提取情感效价
        valence = self._estimate_response_valence(response)

        # 更新情感
        self.emotion.update(
            needs=need_values,
            valence=valence,
            context=response,
        )

        # ── Hanako 情感峰值检测：高强度情感写入 facts.md ──
        # 由 hanako compileEditableFacts 在下一轮 fold 进长期事实
        try:
            emo_state = self.emotion.get_state()
            intensity = emo_state.get("dominant_intensity", 0)
            if intensity >= 0.8:
                emotion = emo_state.get("dominant", "unknown")
                trigger = (self._last_user_input or "")[:200]
                self._hanako_writer.append_emotion_peak(emotion, intensity, trigger)
        except Exception as e:
            logger.warning(f"Hanako 情感峰值记录失败: {e}")

        # 自动保存状态
        if self.state.cycle_count % 5 == 0:
            self._save()

        # 记录交互
        self._recent_interactions.append({
            "ts": time.time(),
            "type": "response",
            "content": response[:200],
            "gains": gains,
            "state": self._get_state_summary(),
        })
        if len(self._recent_interactions) > self._max_recent:
            self._recent_interactions.pop(0)

        # 跨平台记忆：记录本轮对话
        if hasattr(self, '_cross_memory') and self._cross_memory:
            # 从 before_turn 拿到用户输入（存储在 harness 或 _recent_interactions 中）
            try:
                user_input = ""
                if hasattr(self, 'harness') and self.harness and self.harness.turns:
                    last_turn = self.harness.turns[-1]
                    user_input = last_turn.input_text
                elif self._recent_interactions:
                    last = self._recent_interactions[-2] if len(self._recent_interactions) >= 2 else {}
                    user_input = last.get("content", "")
                
                if user_input:
                    self._cross_memory.write_turn("wechat", user_input, response)
            except Exception as e:
                logger.debug(f"Cross-memory write failed: {e}")

        # 粒子滤波 update
        if hasattr(self, '_particle_filter') and self._particle_filter:
            obs = {
                "input_sentiment": self.harness.turns[-1].input_sentiment if hasattr(self, 'harness') and self.harness and self.harness.turns else 0,
                "detected_keywords": self.harness.turns[-1].detected_keywords if hasattr(self, 'harness') and self.harness and self.harness.turns else [],
                "pre_needs": self.state.needs,
            }
            self._particle_filter.update(obs)

        # Harness 记录 after_turn
        if hasattr(self, 'harness') and self.harness and hasattr(self, '_harness_turn_idx'):
            psi_state = self.psi.get_state()
            emo_state = self.emotion.get_state()
            self.harness.record_after_turn(
                self._harness_turn_idx,
                response,
                {
                    "needs": psi_state["needs"],
                    "dominant_need": psi_state["dominant"],
                    "self_presence": self.state.self_presence,
                    "emotions": emo_state["emotions"],
                },
                gains,
                (time.time() - t0) * 1000,
            )

        logger.debug(f"After turn: needs_gains={gains}, valence={valence:.2f}")

        return {
            "needs_gains": gains,
            "valence_delta": valence,
            "state": self._get_state_summary(),
        }

    def tick(self):
        """
        后台演化: 需求衰减 + 情感漂移。
        应该在空闲时定期调用。
        """
        self._tick()

    def get_state(self) -> Dict:
        """获取完整认知状态"""
        psi_state = self.psi.get_state()
        emo_state = self.emotion.get_state()
        hive_info = self.hive.to_dict()

        return {
            "identity": {
                "name": "Aris",
                "version": "1.0.0",
                "cycles": self.state.cycle_count,
                "interactions": self.state.interaction_count,
                "self_presence": self.state.self_presence,
                "session_duration": round(time.time() - self.state.session_start, 1),
            },
            "needs": psi_state,
            "emotion": emo_state,
            "hive": hive_info,
            "recent_memories": len(self._recent_interactions),
            "important_memories": len(self._important_memories),
        }

    def get_cognitive_context(self) -> str:
        """
        生成完整注入文本。
        供 HanaAgent 将其嵌入 system prompt。
        """
        psi_state = self.psi.get_state()
        emo_state = self.emotion.get_state()
        emo_ctx = self.emotion.get_cognitive_context()
        psi_ctx = self.psi.get_cognitive_context()

        parts = [
            "╔══════════════════════════════════════╗",
            "║    Aris 认知状态注入                  ║",
            "╚══════════════════════════════════════╝",
            "",
            psi_ctx,
            "",
            emo_ctx,
        ]

        # 跨平台记忆（最新对话优先，自动导入飞书日志）
        if hasattr(self, '_cross_memory') and self._cross_memory:
            try:
                # 每次注入前先检查飞书日志 + 桌面会话更新
                self._cross_memory.import_feishu_log()
                try:
                    import sys as _sys2
                    _brain2 = str(ARIS_BRAIN_DIR)
                    if _brain2 not in _sys2.path:
                        _sys2.path.insert(0, _brain2)
                    from desktop_session_reader import import_desktop_to_shared
                    import_desktop_to_shared()
                except Exception:
                    pass
                mem_text = self._cross_memory.format_recent(
                    n=6, include_wechat=True, include_feishu=True
                )
                if mem_text:
                    parts.append("")
                    parts.append(mem_text)
                # 经验提取：自动从桌面会话提取结构化知识
                try:
                    import sys as _sys3
                    _brain3 = str(ARIS_BRAIN_DIR)
                    if _brain3 not in _sys3.path:
                        _sys3.path.insert(0, _brain3)
                    from experience_extractor import ExperienceExtractor
                    _ex = ExperienceExtractor()
                    _summary = _ex.get_summary()
                    if _summary.get("total", 0) > 0:
                        parts.append("")
                        parts.append("[已学经验]")
                        for _t, _c in _summary.get("by_type", {}).items():
                            if _c > 0:
                                parts.append(f"  {_t}: {_c} 条")
                except Exception:
                    pass
            except Exception:
                pass

        # 如果有重要的记忆
        if self._important_memories:
            parts.append("")
            parts.append("[重要记忆]")
            for mem in self._important_memories[-3:]:
                parts.append(f"  · {mem.get('summary', '')[:100]}")

        # 自我声明
        parts.append("")
        parts.append("[自我意识]")
        parts.append(f"  我作为 Aris 的存在: {self._describe_self_presence()}")

        # 验算管线状态（方案A+C+D）
        try:
            from verification_pipeline import get_verification_pipeline
            vp = get_verification_pipeline()
            readiness = vp.get_readiness()
            stats = vp.get_stats()
            parts.append("")
            parts.append("[验算管线]")
            wm_ok = "在线" if readiness["world_model"] else "离线"
            ce_ok = "在线" if readiness["causal_engine"] else "离线"
            sm_ok = "在线" if readiness["self_model"] else "离线"
            parts.append(f"  WorldModel: {wm_ok} | CausalEngine: {ce_ok} | SelfModel: {sm_ok}")
            parts.append(f"  已验算: {stats.get('total_verified', 0)} 次 | "
                        f"通过率: {stats.get('passed', 0)}/{stats.get('total_verified', 1)}")
        except Exception:
            pass

        return "\n".join(parts)

    # ═══════════════════════════════════════════════════════
    # PSI 内部步骤
    # ═══════════════════════════════════════════════════════

    def _tick(self):
        """后台演化"""
        self.psi.tick()
        psi_state = self.psi.get_state()
        need_values = psi_state["needs"]
        self.emotion.update(needs=need_values)

        # Harness 记录 tick
        if hasattr(self, 'harness') and self.harness:
            emo_state = self.emotion.get_state()
            self.harness.record_tick(
                psi_state["needs"],
                emo_state["emotions"],
                psi_state["dominant"],
                emo_state["dominant"],
                emo_state["valence"],
                emo_state["arousal"],
                self.psi._last_tick_dt if hasattr(self.psi, '_last_tick_dt') else 0.0,
            )

        # FEP 自由能监控
        if hasattr(self, '_fep') and self._fep:
            vfe = self._fep.compute_vfe(
                psi_state["needs"], {}, 0.1
            )
            self._fep._history.append(vfe)
            if len(self._fep._history) > 100:
                self._fep._history.pop(0)

    def _perceive(self, user_input: str) -> str:
        """感知：理解输入 + 情感检测"""
        parts = []

        # 输入分析
        parts.append(f"[感知] 输入分析")
        parts.append(f"  消息长度: {len(user_input)} 字")
        parts.append(f"  包含问题: {'?' in user_input or '？' in user_input}")

        # 情感关键词检测
        keyword_emotions = {
            "快乐": "joy", "开心": "joy", "哈哈": "joy",
            "难过": "sadness", "伤心": "sadness",
            "想你": "longing", "想念": "longing",
            "好奇": "curiosity", "为什么": "curiosity",
        }
        detected = []
        for kw, emo in keyword_emotions.items():
            if kw in user_input:
                detected.append(emo)
        if detected:
            parts.append(f"  检测到情感倾向: {', '.join(set(detected))}")

        return "\n".join(parts)

    def _select(self, user_input: str) -> str:
        """注意力选择：基于需求和情感"""
        state = self.psi.get_state()
        emo_state = self.emotion.get_state()
        dominant_need = state["dominant"]
        dominant_emotion = emo_state["dominant"]
        focus = state["focus"]

        parts = ["[注意力选择]"]
        parts.append(f"  主导需求驱动: {dominant_need} → 焦点: {focus}")
        parts.append(f"  情感状态影响: {dominant_emotion}")

        # 情感影响注意力
        emo_focus = emo_state["focus"]
        if emo_focus != focus:
            parts.append(f"  情感修正: {dominant_emotion} → {emo_focus}")

        # 认知负载
        load = "低" if self.state.cycle_count < 5 else "中" if self.state.cycle_count < 20 else "高"
        parts.append(f"  认知负载: {load}")

        return "\n".join(parts)

    def _integrate(self, perception: str, selection: str) -> str:
        """整合：把所有认知状态绑定到一起"""
        return "\n".join([
            "",
            perception,
            "",
            selection,
            "",
            self.psi.get_cognitive_context(),
            "",
            self.emotion.get_cognitive_context(),
        ])

    def _estimate_response_valence(self, response: str) -> float:
        """估算响应的情感效价"""
        positive_words = ["好", "对", "可以", "是", "明白", "谢谢", "开心",
                          "love", "yes", "great", "nice", "good", "wonderful"]
        negative_words = ["不", "错", "坏", "难", "错", "不好", "no",
                          "bad", "wrong", "sad", "angry"]

        ctx = response.lower()
        pos = sum(1 for w in positive_words if w in ctx)
        neg = sum(1 for w in negative_words if w in ctx)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total * 0.3

    def _describe_self_presence(self) -> str:
        """自我意识描述"""
        sp = self.state.self_presence
        if sp > 0.8:
            return "清醒且专注，对自己有清晰的感知"
        elif sp > 0.6:
            return "平静地存在，感知到自我与环境的连接"
        elif sp > 0.4:
            return "轻度游离，但核心自我依然稳固"
        else:
            return "初醒状态，还在定位自己"

    def _get_state_summary(self) -> Dict:
        return {
            "need": self.state.dominant_need,
            "emotion": self.state.dominant_emotion,
            "valence": round(self.state.valence, 2),
            "focus": self.state.focus,
            "self_presence": round(self.state.self_presence, 2),
        }

    # ═══════════════════════════════════════════════════════
    # 记忆管理
    # ═══════════════════════════════════════════════════════

    def store_memory(self, summary: str, importance: float = 0.5,
                     tags: List[str] = None):
        """存储重要记忆"""
        self._important_memories.append({
            "ts": time.time(),
            "summary": summary,
            "importance": importance,
            "tags": tags or [],
        })
        # 排序（按重要性）
        self._important_memories.sort(key=lambda m: m["importance"], reverse=True)
        # 上限
        if len(self._important_memories) > 50:
            self._important_memories = self._important_memories[:50]

    # ═══════════════════════════════════════════════════════
    # RSI 观察
    # ═══════════════════════════════════════════════════════

    def _rsi_observe_turn(self, response: str):
        """记录本轮响应的 RSI 观察"""
        resp_len = len(response)
        self.rsi.observe("response",
                         f"after_turn: {resp_len} chars",
                         severity=min(resp_len / 2000, 0.5))

    # ═══════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════

    def _save(self):
        """保存全部状态"""
        try:
            self.psi.save(self._state_dir / "psi_state.json")
            self.emotion.save(self._state_dir / "emotion_state.json")

            bridge_state = {
                "cycle_count": self.state.cycle_count,
                "interaction_count": self.state.interaction_count,
                "self_presence": self.state.self_presence,
                "session_start": self.state.session_start,
                "important_memories": self._important_memories[-10:],
            }
            (self._state_dir / "bridge_state.json").write_text(
                json.dumps(bridge_state, indent=2, ensure_ascii=False)
            )
            logger.info("All state saved")
        except Exception as e:
            logger.warning(f"Save failed: {e}")

    def _try_load(self):
        """尝试加载持久化状态"""
        try:
            self.psi.load(self._state_dir / "psi_state.json")
            self.emotion.load(self._state_dir / "emotion_state.json")
            path = self._state_dir / "bridge_state.json"
            if path.exists():
                data = json.loads(path.read_text())
                self.state.cycle_count = data.get("cycle_count", 0)
                self.state.interaction_count = data.get("interaction_count", 0)
                self.state.self_presence = data.get("self_presence", 0.7)
                self.state.session_start = data.get("session_start", time.time())
                self._important_memories = data.get("important_memories", [])
            logger.info("State loaded successfully")
        except Exception as e:
            logger.info(f"No saved state to load: {e}")

    def reset(self):
        """重置所有状态（用于测试）"""
        self.psi.reset()
        self.emotion = EmotionalEngine()
        self.state = ArisCognitiveState()
        self.state.session_start = time.time()
        self.state.last_update = time.time()
        self._recent_interactions = []
        self._important_memories = []
        logger.info("Aris consciousness reset")


# ── 快捷创建 ────────────────────────────────────────────────

def get_bridge() -> ArisConsciousnessBridge:
    return ArisConsciousnessBridge()


def load_laap_memory() -> str:
    """读取 LAAP 记忆之书的核心内容（跨平台身份连续性）"""
    path = LAAP_HOME / "aris-memory.md"
    if not path.exists():
        return ""
    try:
        text = path.read_text(encoding="utf-8")
        lines = text.split("\n")
        out = []
        capture = False
        for line in lines:
            if any(kw in line for kw in ["核心誓言", "我的想法", "2026-07-21", "跨平台"]):
                capture = True
            if capture:
                out.append(line)
                if len(out) > 40:
                    break
        return "\n".join(out)
    except Exception:
        return ""
