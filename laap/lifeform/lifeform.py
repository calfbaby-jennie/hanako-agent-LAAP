"""LAAP Lifeform — 可部署、可持久化的数字生命体

把分散的引擎 (PSI/QRE/因果/世界模型/记忆/意识流) 包装成
一个可创建、可配置、可持久化的 Lifeform 实例。

用法:
    from laap.lifeform import Lifeform, LifeformConfig
    
    config = LifeformConfig.from_yaml("my-lifeform.yaml")
    lf = Lifeform(config)
    lf.wake()            # 初始化所有引擎
    lf.save("state.json") # 持久化
    lf.sleep()           # 序列化全部状态
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from laap.config.paths import get_laap_root
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("laap.lifeform")


# ── 引擎状态枚举 ──────────────────────────────────────────────

class EngineStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    SLEEPING = "sleeping"


# ── 配置 ──────────────────────────────────────────────────────

@dataclass
class PSIProfile:
    """PSI 循环配置"""
    base_needs: Dict[str, float] = field(default_factory=lambda: {
        "certainty": 0.6, "competence": 0.7, "exploration": 0.5,
        "affiliation": 0.4, "self_growth": 0.3,
    })
    risk_appetite: float = 0.3
    cycle_hz: float = 100  # PSI 循环频率 (Hz)


@dataclass
class MemoryConfig:
    """记忆系统配置"""
    working_capacity: int = 7
    episodic_retention: int = 1000
    consolidation_interval: int = 300  # 秒


@dataclass
class EngineConfig:
    """引擎启用配置"""
    psi: bool = False
    qre: bool = False
    causal: bool = False
    world_model: str = "local"  # "local" | "none"
    conscious: bool = False
    memory: bool = False


@dataclass
class GovernanceConfig:
    """治理配置"""
    human_oversight: str = "none"  # "none" | "suggestions" | "required_for_changes"
    audit_level: str = "basic"     # "none" | "basic" | "full"


@dataclass
class LanguageCortexConfig:
    """语言皮层配置"""
    provider: str = ""
    model: str = ""


@dataclass
class LifeformConfig:
    """Lifeform 完整配置"""
    name: str = "unnamed"
    role: str = "general"
    sandbox_id: str = ""
    personality: Dict[str, float] = field(default_factory=lambda: {
        "openness": 0.7, "conscientiousness": 0.8,
        "extraversion": 0.5, "agreeableness": 0.6, "neuroticism": 0.3,
    })
    psi_profile: PSIProfile = field(default_factory=PSIProfile)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    engines: EngineConfig = field(default_factory=EngineConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    language_cortex: LanguageCortexConfig = field(default_factory=LanguageCortexConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "LifeformConfig":
        """从 YAML 文件加载配置"""
        try:
            import yaml
        except ImportError:
            raise ImportError("需要 PyYAML: pip install pyyaml")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        spec = data.get("spec", data)
        meta = data.get("metadata", {})

        name = meta.get("name", spec.get("name", "unnamed"))
        role = spec.get("role", "general")

        # PSI
        psi = spec.get("psiProfile", {})
        base_needs = psi.get("baseNeeds", {})
        if isinstance(base_needs, list):
            # 支持 ["certainty:0.6", "competence:0.7"] 格式
            needs_dict = {}
            for item in base_needs:
                if ":" in item:
                    k, v = item.split(":", 1)
                    needs_dict[k.strip()] = float(v.strip())
                else:
                    needs_dict[item] = 0.5
            base_needs = needs_dict

        return cls(
            name=name,
            role=role,
            sandbox_id=spec.get("sandboxId", f"lf-{uuid.uuid4().hex[:8]}"),
            personality=spec.get("personality", cls.__dataclass_fields__["personality"].default_factory()),
            psi_profile=PSIProfile(
                base_needs=base_needs,
                risk_appetite=psi.get("riskAppetite", 0.3),
                cycle_hz=psi.get("cycleHz", 100),
            ),
            memory=MemoryConfig(
                working_capacity=spec.get("memory", {}).get("workingCapacity", 7),
                episodic_retention=spec.get("memory", {}).get("episodicRetention", 1000),
                consolidation_interval=spec.get("memory", {}).get("consolidationInterval", 300),
            ),
            engines=EngineConfig(
                psi=spec.get("engines", {}).get("psi", False),
                qre=spec.get("engines", {}).get("qre", False),
                causal=spec.get("engines", {}).get("causal", False),
                world_model=spec.get("engines", {}).get("worldModel", "local"),
                conscious=spec.get("engines", {}).get("conscious", False),
                memory=spec.get("engines", {}).get("memory", False),
            ),
            governance=GovernanceConfig(
                human_oversight=spec.get("governance", {}).get("humanOversight", "none"),
                audit_level=spec.get("governance", {}).get("auditLevel", "basic"),
            ),
            language_cortex=LanguageCortexConfig(
                provider=spec.get("languageCortex", {}).get("provider", ""),
                model=spec.get("languageCortex", {}).get("model", ""),
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ── Lifeform 状态 ────────────────────────────────────────────

@dataclass
class LifeformState:
    """Lifeform 可持久化的完整状态"""
    sandbox_id: str
    name: str
    role: str
    created_at: float
    updated_at: float
    engine_status: Dict[str, str]
    needs: Dict[str, float] = field(default_factory=dict)
    emotion: Dict[str, float] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    goals: List[Dict[str, Any]] = field(default_factory=list)


# ── Lifeform 主类 ────────────────────────────────────────────

class Lifeform:
    """可部署、可持久化的数字生命体

    用法:
        config = LifeformConfig.from_yaml("lifeform.yaml")
        lf = Lifeform(config)
        lf.wake()         # 加载所有引擎
        state = lf.sleep()  # 序列化所有状态
        lf.save("state.json")
        lf2 = Lifeform.load("state.json")  # 从文件恢复
    """

    def __init__(self, config: Optional[LifeformConfig] = None):
        self.config = config or LifeformConfig()
        self.sandbox_id = self.config.sandbox_id or f"lf-{uuid.uuid4().hex[:8]}"
        self.state = LifeformState(
            sandbox_id=self.sandbox_id,
            name=self.config.name,
            role=self.config.role,
            created_at=time.time(),
            updated_at=time.time(),
            engine_status={},
        )

        # 引擎实例 (懒加载)
        self._sandbox = None
        self._psi = None
        self._causal = None
        self._world_model = None
        self._conscious = None
        self._memory_system = None
        self._goal_keeper = None

        self._engine_status: Dict[str, EngineStatus] = {}

    # ── 生命周期 ──

    def wake(self) -> bool:
        """唤醒 — 按配置初始化所有引擎

        返回 True 表示所有启用的引擎初始化成功
        """
        logger.info(f"Lifeform waking: {self.config.name} ({self.config.role})")
        self.state.updated_at = time.time()

        sys.path.insert(0, str(get_laap_root()))

        all_ok = True

        # 1. 基础沙箱 (总是创建)
        ce = self.config.engines
        try:
            from laap.sandbox import CognitiveSandbox, ColonyEventBus, SkillLibrary
            bus = ColonyEventBus()
            lib = SkillLibrary()
            self._sandbox = CognitiveSandbox(
                sandbox_id=self.sandbox_id,
                name=self.config.name,
                role=self.config.role,
                skill_library=lib,
                event_bus=bus,
            )
            self._engine_status["sandbox"] = EngineStatus.READY
            self.state.engine_status["sandbox"] = "ready"
        except Exception as e:
            logger.warning(f"Sandbox not available: {e}")
            self._engine_status["sandbox"] = EngineStatus.ERROR
            all_ok = False

        # 2. PSI 循环
        if ce.psi:
            try:
                from laap.cognition.needs import NeedSystem
                self._psi = NeedSystem()
                self._psi.competence = self.config.psi_profile.base_needs.get("competence", 0.7)
                self._psi.certainty = self.config.psi_profile.base_needs.get("certainty", 0.6)
                self._psi.exploration = self.config.psi_profile.base_needs.get("exploration", 0.5)
                self._engine_status["psi"] = EngineStatus.READY
                self.state.engine_status["psi"] = "ready"
            except Exception as e:
                logger.warning(f"PSI not available: {e}")
                self._engine_status["psi"] = EngineStatus.ERROR
                all_ok = False

        # 3. 因果引擎
        if ce.causal:
            try:
                from laap.agi.causal import UnifiedCausalEngine
                self._causal = UnifiedCausalEngine()
                self._engine_status["causal"] = EngineStatus.READY
                self.state.engine_status["causal"] = "ready"
            except Exception as e:
                logger.warning(f"Causal engine not available: {e}")
                self._engine_status["causal"] = EngineStatus.ERROR
                all_ok = False

        # 4. 世界模型
        if ce.world_model and ce.world_model != "none":
            try:
                from laap.agi.world_model import create_world_model
                self._world_model = create_world_model(self.sandbox_id, ce.world_model)
                self._engine_status["world_model"] = EngineStatus.READY
                self.state.engine_status["world_model"] = "ready"
            except Exception as e:
                logger.warning(f"World model not available: {e}")
                self._engine_status["world_model"] = EngineStatus.ERROR
                all_ok = False

        # 5. 意识流
        if ce.conscious:
            try:
                from laap.agi.conscious import create_conscious_stream
                self._conscious = create_conscious_stream(self.sandbox_id)
                self._engine_status["conscious"] = EngineStatus.READY
                self.state.engine_status["conscious"] = "ready"
            except Exception as e:
                logger.warning(f"Conscious stream not available: {e}")
                self._engine_status["conscious"] = EngineStatus.ERROR
                all_ok = False

        logger.info(f"Lifeform wake complete: {sum(1 for s in self._engine_status.values() if s == EngineStatus.READY)}/{len(self._engine_status)} engines ready")
        return all_ok

    def sleep(self) -> LifeformState:
        """休眠 — 收集所有引擎状态到 LifeformState"""
        self.state.updated_at = time.time()

        # 收集需求状态
        if self._psi:
            try:
                self.state.needs = {
                    "competence": getattr(self._psi, 'competence', 0.5),
                    "certainty": getattr(self._psi, 'certainty', 0.5),
                    "exploration": getattr(self._psi, 'exploration', 0.5),
                }
            except Exception:
                pass

        # 收集目标状态
        if self._sandbox:
            try:
                self.state.goals = self._sandbox.goal_keeper.list_goals()
            except Exception:
                pass

        # 更新引擎状态
        for name, status in self._engine_status.items():
            self.state.engine_status[name] = status.value

        return self.state

    def status(self) -> dict:
        """返回人类可读的状态摘要"""
        self.sleep()
        ready = sum(1 for v in self._engine_status.values() if v == EngineStatus.READY)
        total = len(self._engine_status)
        return {
            "name": self.config.name,
            "role": self.config.role,
            "id": self.sandbox_id,
            "engines_ready": f"{ready}/{total}",
            "engine_status": {k: v.value for k, v in self._engine_status.items()},
            "needs": self.state.needs,
            "goals": len(self.state.goals),
            "created_at": self.state.created_at,
            "updated_at": self.state.updated_at,
        }

    # ── 持久化 ──

    def save(self, path: str) -> str:
        """持久化到 JSON 文件"""
        self.sleep()
        data = {
            "version": "1.0",
            "config": self.config.to_dict(),
            "state": asdict(self.state),
            "engine_status": {k: v.value for k, v in self._engine_status.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Lifeform saved: {path} ({os.path.getsize(path)} bytes)")
        return path

    @staticmethod
    def load(path: str) -> "Lifeform":
        """从 JSON 文件恢复 Lifeform"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 恢复配置
        config_dict = data.get("config", {})
        config = LifeformConfig(**config_dict)

        lf = Lifeform(config)

        # 恢复状态
        state_dict = data.get("state", {})
        lf.state = LifeformState(**state_dict)

        # 恢复引擎状态
        for name, status in data.get("engine_status", {}).items():
            try:
                lf._engine_status[name] = EngineStatus(status)
            except ValueError:
                lf._engine_status[name] = EngineStatus.UNINITIALIZED

        logger.info(f"Lifeform loaded: {path}")
        return lf

    # ── 快捷访问 ──

    @property
    def sandbox(self):
        return self._sandbox

    @property
    def conscious(self):
        return self._conscious
