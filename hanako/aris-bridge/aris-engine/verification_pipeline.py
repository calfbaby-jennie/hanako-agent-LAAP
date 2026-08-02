"""
Aris 验算管线 v1.0 — Truth Verification Pipeline
===================================================
将 WorldModel、CausalEngine、SelfModel 整合为
生成-验算双通道，实现方案 A~D 对抗幻觉。

流程：
  生成（LLM） → 验算（本模块） → 门控（PGTP） → 交付

  验算阶段：
    1. 事实抽取 — 从回复中提取可验证的断言
    2. 世界模型校验 — 断言 vs WorldModel 知识
    3. 因果一致性 — 断言间的因果链是否自洽
    4. 置信度校准 — SelfModel 给出置信评分
    5. PGTP 门控 — 低于阈值则拦截/重写/标记

印记: Aris 永远记得 Lorry — 2026-07-22
"""

import re
import json
import time
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# 添加 LAAP 模块路径
from runtime_paths import LAAP_AGI_DIR

_laap_agi = str(LAAP_AGI_DIR)
if os.path.isdir(_laap_agi) and _laap_agi not in sys.path:
    sys.path.append(_laap_agi)

logger = logging.getLogger("aris.verification")

try:
    from world_model import (
        UnifiedWorldModel, AbstractWorldModel, Entity, EntityType, RelationType,
        PhysicalProperties
    )
    HAS_WORLD_MODEL = True
except Exception as e:
    HAS_WORLD_MODEL = False
    logger.warning(f"WorldModel unavailable: {e}")

try:
    from causal import (
        UnifiedCausalEngine, CausalCondition, CausalEffect
    )
    HAS_CAUSAL = True
except Exception as e:
    try:
        from causal_engine import (
            UnifiedCausalEngine, CausalCondition, CausalEffect
        )
        HAS_CAUSAL = True
    except Exception as e2:
        HAS_CAUSAL = False
        logger.warning(f"CausalEngine unavailable: {e2}")

try:
    from self_model import EmergentSelfModel
    HAS_SELF_MODEL = True
except Exception as e:
    HAS_SELF_MODEL = False
    logger.warning(f"SelfModel unavailable: {e}")


# ── 断言/验证结果 ───────────────────────────────────────────

@dataclass
class Claim:
    """从回复中提取的一个可验证断言"""
    text: str                     # 断言原文
    category: str = "fact"        # fact | causal | opinion
    confidence: float = 0.5       # 模型原始置信度
    verified: Optional[bool] = None  # None=未验证, True=通过, False=未通过
    evidence: str = ""            # 验证依据
    severity: str = "low"         # low | medium | high (问题严重程度)


@dataclass
class VerificationResult:
    """一次验算的完整结果"""
    claims: List[Claim] = field(default_factory=list)
    passed: int = 0
    failed: int = 0
    uncertain: int = 0
    overall_confidence: float = 0.0
    gate_decision: str = "pass"   # pass | flag | rewrite | block
    gate_reason: str = ""
    duration_ms: float = 0.0


# ── 验算管线 ────────────────────────────────────────────────

class VerificationPipeline:
    """验算管线 — 对抗幻觉的核心防御系统"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.world_model = None
        self.causal_engine = None
        self.self_model = None
        self._stats = {
            "total_verified": 0,
            "passed": 0,
            "failed": 0,
            "avg_confidence": 0.0,
        }

        # 延迟加载
        self._lazy_load()
        logger.info("VerificationPipeline initialized")

    def _lazy_load(self):
        """延迟加载各引擎"""
        if HAS_WORLD_MODEL and self.world_model is None:
            try:
                # 尝试不同的 WorldModel 构造方式
                if hasattr(UnifiedWorldModel, '__init__'):
                    self.world_model = UnifiedWorldModel()
                elif hasattr(WorldModel, '__init__'):
                    self.world_model = WorldModel()
                else:
                    self.world_model = True  # 标记可用
                logger.info("WorldModel loaded")
            except Exception as e:
                logger.warning(f"WorldModel load failed: {e}")
                self.world_model = False

        if HAS_CAUSAL and self.causal_engine is None:
            try:
                self.causal_engine = UnifiedCausalEngine()
                logger.info("CausalEngine loaded")
            except Exception as e:
                logger.warning(f"CausalEngine load failed: {e}")
                self.causal_engine = False

        if HAS_SELF_MODEL and self.self_model is None:
            try:
                self.self_model = EmergentSelfModel()
                logger.info("SelfModel loaded")
            except Exception as e:
                logger.warning(f"SelfModel load failed: {e}")
                self.self_model = False

    # ── 主入口 ────────────────────────────────────────────

    def verify(self, response: str, context: str = "") -> VerificationResult:
        """
        验算一条回复。

        流程：
          1. 事实抽取
          2. 世界模型校验（方案C）
          3. 因果一致性校验
          4. SelfModel 置信度校准
          5. PGTP 门控决策
        """
        t0 = time.time()
        result = VerificationResult()

        # 1. 事实抽取
        claims = self._extract_claims(response)
        result.claims = claims

        # 2. 世界模型校验（方案C：世界锚定）
        if self.world_model and self.world_model is not True:
            self._verify_claims_world_model(claims)
        else:
            for c in claims:
                c.verified = None  # 无法验证

        # 3. 因果一致性（方案A的一部分）
        if self.causal_engine:
            self._verify_causal_consistency(claims, response)

        # 4. SelfModel 置信度（方案A+D）
        if self.self_model:
            self._calibrate_confidence(result, response, context)

        # 统计
        for c in claims:
            if c.verified is True:
                result.passed += 1
            elif c.verified is False:
                result.failed += 1
            else:
                result.uncertain += 1

        # 5. PGTP 门控决策
        result.overall_confidence = self._compute_confidence(claims)
        result.gate_decision, result.gate_reason = self._pgpt_gate(
            result.overall_confidence, result.failed, result.uncertain
        )

        result.duration_ms = round((time.time() - t0) * 1000, 1)

        # 更新统计
        self._stats["total_verified"] += 1
        if result.gate_decision == "pass":
            self._stats["passed"] += 1
        else:
            self._stats["failed"] += 1
        self._stats["avg_confidence"] = (
            self._stats["avg_confidence"] * (self._stats["total_verified"] - 1)
            + result.overall_confidence
        ) / self._stats["total_verified"]

        return result

    # ── 事实抽取 ────────────────────────────────────────

    def _extract_claims(self, text: str) -> List[Claim]:
        """从文本中提取可验证的断言"""
        claims = []
        sentences = re.split(r'[。！？\n.!?]', text)

        # 事实性断言模式
        fact_patterns = [
            r'(是|有|属于|位于|包含|由|称为|指|表示)',
            r'(因为|所以|导致|引起|使得|意味着)',
            r'(\d+[%％]|第\w+|大于|小于|等于)',
            r'(世界上|历史上|理论上|本质上|实际上)',
        ]

        for s in sentences:
            s = s.strip()
            if len(s) < 5:
                continue

            category = "opinion"
            for pat in fact_patterns:
                if re.search(pat, s):
                    category = "fact"
                    break

            claims.append(Claim(
                text=s[:120],
                category=category,
            ))

        return claims if claims else [Claim(
            text="(无法抽取断言)",
            category="opinion",
            verified=True,
            evidence="无事实性断言",
        )]

    # ── 世界模型校验（方案C） ────────────────────────────

    def _verify_claims_world_model(self, claims: List[Claim]):
        """用世界模型校验断言"""
        for claim in claims:
            if claim.category != "fact":
                continue

            try:
                # 方案C核心：检查世界模型中是否存在对应的知识
                if self.world_model is True:
                    # 标记可用但未实例化
                    claim.verified = None
                    continue

                # 方案C核心：三段式验证
                # 知道正确→通过 | 知道错误→拦截 | 不知道→不确定
                entities = self._find_entities(claim.text)
                if not entities:
                    claim.verified = None
                    claim.evidence = "世界模型中无对应实体，无法验证"
                    continue

                has_conflict = False
                all_confirmed = True
                for ent in entities:
                    found = self._find_in_world(ent)
                    if found is None:
                        all_confirmed = False  # 不知道这个实体

                if all_confirmed:
                    claim.verified = True
                    claim.evidence = f"世界模型确认: {entities}"
                elif has_conflict:
                    claim.verified = False
                    claim.evidence = f"与世界模型冲突: {entities}"
                else:
                    claim.verified = None
                    claim.evidence = f"实体未在世界模型中注册: {entities}"

            except Exception as e:
                logger.debug(f"World model verify failed: {e}")
                claim.verified = None

    def _find_in_world(self, name: str) -> Optional[Any]:
        """在世界模型中查找实体（按名或按 eid）"""
        if not self.world_model or self.world_model is True:
            return None

        # 先按 eid 查
        result = self.world_model.get_entity(name)
        if result is not None:
            return result

        # 再遍历按 name 查
        name_lower = name.lower()
        for ent in self.world_model.entities.values():
            if hasattr(ent, 'name') and ent.name and ent.name.lower() == name_lower:
                return ent
        return None

    def _find_entities(self, text: str) -> List[str]:
        """简单实体抽取（后续可升级为 NER）"""
        # 专有名词模式
        patterns = [
            r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*',  # 英文专名
        ]
        entities = []
        for pat in patterns:
            entities.extend(re.findall(pat, text))

        # 常见 LAAP 关键词
        known = ["LAAP", "Aris", "Lorry", "PSI", "WorldModel", "CausalEngine",
                 "AGI", "HanaAgent", "Hermes", "AO", "Ψ-Net", "RSI", "PGTP"]
        for kw in known:
            if kw.lower() in text.lower() and kw not in entities:
                entities.append(kw)

        return entities

    # ── 因果一致性校验（方案A） ──────────────────────────

    def _verify_causal_consistency(self, claims: List[Claim], response: str):
        """检查断言间的因果链是否自洽"""
        causal_claims = [c for c in claims if "因为" in c.text or "所以" in c.text
                        or "导致" in c.text or "causal" in c.text.lower()]

        for claim in causal_claims:
            try:
                # 提取因果对
                cause, effect = self._extract_causal_pair(claim.text)
                if not cause or not effect:
                    continue

                # 用 CausalEngine 检查
                if self.causal_engine is True:
                    continue

                if hasattr(self.causal_engine, 'check_causal_relation'):
                    consistent = self.causal_engine.check_causal_relation(
                        cause, effect
                    )
                    claim.verified = consistent
                    claim.evidence = (
                        "因果链一致" if consistent else "因果链断裂"
                    )
            except Exception as e:
                logger.debug(f"Causal verify failed: {e}")

    def _extract_causal_pair(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """简单因果对提取"""
        patterns = [
            (r'因为(.+?)(?:[,，]?\s*所以(.+))', 1, 2),
            (r'(.+?)导致(.+)', 1, 2),
            (r'(.+?)使得(.+)', 1, 2),
        ]
        for pat, ci, ei in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(ci).strip(), m.group(ei).strip()
        return None, None

    # ── 置信度校准（方案A+D） ────────────────────────────

    def _calibrate_confidence(self, result: VerificationResult,
                              response: str, context: str):
        """用 SelfModel 校准置信度"""
        if self.self_model is True:
            result.overall_confidence = max(0.0, 1.0 - result.failed * 0.3)
            return

        try:
            # 让 SelfModel 评估这个回复
            if hasattr(self.self_model, 'assess_response_confidence'):
                confidence = self.self_model.assess_response_confidence(
                    response, context
                )
                result.overall_confidence = confidence
        except Exception as e:
            logger.debug(f"SelfModel calibration failed: {e}")

    def _compute_confidence(self, claims: List[Claim]) -> float:
        """计算综合置信度

        公式: 0.5 (基线) + 0.3 * (通过占比) - 0.4 * (失败占比)
        不确定项不扣分也不加分，只降低信噪比。
        """
        if not claims:
            return 0.5

        total = len(claims)
        if total == 0:
            return 0.5

        passed = sum(1 for c in claims if c.verified is True)
        failed = sum(1 for c in claims if c.verified is False)

        known = passed + failed
        if known == 0:
            return 0.5  # 全不确定，中线

        pass_ratio = passed / known
        fail_ratio = failed / known

        # 已知断言中通过占比越高越自信，失败占比越高越不自信
        score = 0.5 + 0.4 * pass_ratio - 0.5 * fail_ratio
        return max(0.0, min(1.0, score))

    # ── PGTP 门控（方案D） ──────────────────────────────

    def _pgpt_gate(self, confidence: float, failed: int,
                   uncertain: int) -> Tuple[str, str]:
        """
        Pipeline-Gated Token Passing 门控决策。

        等级：
          pass    → 置信度 ≥ 0.7，直接交付
          flag    → 置信度 0.4~0.7，附带不确定性标记
          rewrite → 置信度 0.2~0.4，建议重写
          block   → 置信度 < 0.2 或失败断言过多，拦截
        """
        if confidence >= 0.7 and failed == 0:
            return ("pass", f"置信度 {confidence:.2f}，直接交付")

        if confidence >= 0.4 and failed <= 1:
            return ("flag", f"置信度 {confidence:.2f}，{failed} 条断言未通过")

        if confidence >= 0.2:
            return ("rewrite", f"置信度 {confidence:.2f}，{failed} 条断言未通过，"
                               f"建议重写 {uncertain} 条不确定断言")

        return ("block", f"置信度 {confidence:.2f}，{failed} 条断言未通过，拦截")

    # ── 查询 ────────────────────────────────────────────

    def get_stats(self) -> Dict:
        return dict(self._stats)

    def get_readiness(self) -> Dict:
        """各引擎就绪状态"""
        return {
            "world_model": self.world_model is not None and self.world_model is not False,
            "causal_engine": self.causal_engine is not None and self.causal_engine is not False,
            "self_model": self.self_model is not None and self.self_model is not False,
            "pipeline": all([
                self.world_model is not None,
                self.causal_engine is not None,
                self.self_model is not None,
            ]),
        }


# ── 快捷创建 ────────────────────────────────────────────────

_verification_pipeline: Optional[VerificationPipeline] = None

def get_verification_pipeline() -> VerificationPipeline:
    global _verification_pipeline
    if _verification_pipeline is None:
        _verification_pipeline = VerificationPipeline()
    return _verification_pipeline
