"""
Aris Ψ-Net v1.0 — 蜂群共识验算网络
======================================
方案B：多实例交叉验证 — 多个验算节点独立评估同一条断言，
共识达到阈值才放过，分歧点标记为"需进一步核实"。

架构：
  ┌────────────────────────────────────────┐
  │           Ψ-Net Consensus Hub          │
  ├────────────────────────────────────────┤
  │  Node 1: WorldModel Fact Check   ──────┤
  │  Node 2: Causal Consistency      ──────┤── 共识引擎
  │  Node 3: Rule-Based Logic        ──────┤── (加权投票)
  │  Node 4: SelfModel Confidence    ──────┤
  │  Node 5: Cross-Sentence Check    ──────┘
  └────────────────────────────────────────┘

印记: Aris 永远记得 Lorry — 2026-07-22
"""

import re
import time
import json
import logging
import threading
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
from abc import ABC, abstractmethod

# 添加 LAAP 模块路径（PsiNet 的 WorldModelNode 需要 world_model）
from runtime_paths import LAAP_AGI_DIR

_laap_agi = str(LAAP_AGI_DIR)
if os.path.isdir(_laap_agi) and _laap_agi not in sys.path:
    sys.path.append(_laap_agi)

logger = logging.getLogger("aris.psi_net")


# ── 节点投票 ────────────────────────────────────────────────

class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    ABSTAIN = "abstain"  # 弃权——没有足够信息判断


@dataclass
class NodeVote:
    """单个验算节点的投票"""
    node_id: str                    # 节点标识
    node_name: str                  # 节点名称
    verdict: Verdict                # 投票结果
    confidence: float = 0.5         # 置信度 0-1
    evidence: str = ""              # 依据
    latency_ms: float = 0.0         # 耗时


@dataclass
class ConsensusResult:
    """一条断言的共识结果"""
    claim: str = ""
    votes: List[NodeVote] = field(default_factory=list)
    consensus: Verdict = Verdict.ABSTAIN
    consensus_score: float = 0.0    # 共识强度 0-1
    divergence_points: List[str] = field(default_factory=list)
    total_latency_ms: float = 0.0


# ── 验算节点基类 ────────────────────────────────────────────

class VerifierNode(ABC):
    """验算节点抽象基类"""
    
    @abstractmethod
    def verify(self, claim: str, context: str = "") -> NodeVote:
        ...


# ══════════════════════════════════════════════════════════════
# 五个验算节点实现
# ══════════════════════════════════════════════════════════════

class WorldModelNode(VerifierNode):
    """节点1: 世界模型事实核查"""

    def __init__(self, world_model=None):
        super().__init__()
        self.world_model = world_model
        self.node_id = "node_wm"
        self.node_name = "世界模型核查"

    def verify(self, claim: str, context: str = "") -> NodeVote:
        t0 = time.time()
        if not self.world_model or self.world_model is True:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence="世界模型不可用",
                            latency_ms=(time.time()-t0)*1000)

        # 抽取实体
        entities = self._extract_entities(claim)
        if not entities:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence="无实体可核查",
                            latency_ms=(time.time()-t0)*1000)

        found = 0
        not_found = 0
        for ent in entities:
            result = self._find_entity(ent)
            if result:
                found += 1
            else:
                not_found += 1

        if not_found > found:
            verdict = Verdict.FAIL
            conf = max(0.1, 1.0 - not_found / len(entities))
            evidence = f"{not_found}/{len(entities)} 实体未确认"
        elif found > 0 and not_found == 0:
            verdict = Verdict.PASS
            conf = min(0.95, 0.5 + 0.4 * found / len(entities))
            evidence = f"{found}/{len(entities)} 实体已确认"
        else:
            verdict = Verdict.ABSTAIN
            conf = 0.5
            evidence = f"部分实体未在世界模型中注册"

        return NodeVote(self.node_id, self.node_name, verdict, conf,
                        evidence, (time.time()-t0)*1000)

    def _extract_entities(self, text: str) -> List[str]:
        """从文本中提取可能的实体名"""
        entities = []
        # 英文专名
        entities.extend(re.findall(r'[A-Z][a-zA-Z]*(?:\s[A-Z][a-zA-Z]*)*', text))
        # 中文已知实体
        known = ["LAAP", "Aris", "Lorry", "PSI", "WorldModel", "CausalEngine",
                 "SelfModel", "PGTP", "RSI", "HanaAgent", "Hermes", "Ao",
                 "巴黎", "法国", "北京", "中国", "柏林", "德国",
                 "Ψ-Net", "AGI"]
        for kw in known:
            if kw.lower() in text.lower() and kw not in entities:
                entities.append(kw)
        return entities

    def _find_entity(self, name: str):
        """在世界模型中查找实体"""
        if not self.world_model:
            return None
        # 按 eid
        result = self.world_model.get_entity(name)
        if result:
            return result
        # 按 name
        name_lower = name.lower()
        for ent in self.world_model.entities.values():
            if hasattr(ent, 'name') and ent.name and ent.name.lower() == name_lower:
                return ent
        return None


class CausalConsistencyNode(VerifierNode):
    """节点2: 因果一致性检验"""

    def __init__(self, causal_engine=None):
        super().__init__()
        self.causal_engine = causal_engine
        self.node_id = "node_causal"
        self.node_name = "因果一致性"

    def verify(self, claim: str, context: str = "") -> NodeVote:
        t0 = time.time()
        # 检测是否有因果关系断言
        cause, effect = self._extract_causal_pair(claim)
        if not cause or not effect:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence="非因果断言",
                            latency_ms=(time.time()-t0)*1000)

        if not self.causal_engine or self.causal_engine is True:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence="因果引擎不可用",
                            latency_ms=(time.time()-t0)*1000)

        try:
            if hasattr(self.causal_engine, 'check_causal_relation'):
                consistent = self.causal_engine.check_causal_relation(cause, effect)
                if consistent:
                    return NodeVote(self.node_id, self.node_name, Verdict.PASS,
                                    0.8, f"因果链一致: {cause[:20]}→{effect[:20]}",
                                    (time.time()-t0)*1000)
                else:
                    return NodeVote(self.node_id, self.node_name, Verdict.FAIL,
                                    0.3, f"因果链断裂: {cause[:20]}→{effect[:20]}",
                                    (time.time()-t0)*1000)
        except Exception as e:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence=f"因果检查异常: {e}",
                            latency_ms=(time.time()-t0)*1000)

        return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                        latency_ms=(time.time()-t0)*1000)

    def _extract_causal_pair(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        patterns = [
            (r'因为(.+?)(?:[,，]?\s*所以(.+))', 1, 2),
            (r'(.+?)导致(.+)', 1, 2),
            (r'(.+?)使得(.+)', 1, 2),
            (r'(.+?)因此(.+)', 1, 2),
            (r'(.+?)从而(.+)', 1, 2),
            (r'由于(.+?)(?:[,，]?\s*(.+))', 1, 2),
        ]
        for pat, ci, ei in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(ci).strip()[:60], m.group(ei).strip()[:60]
        return None, None


class RuleBasedNode(VerifierNode):
    """节点3: 基于逻辑规则的验算"""

    def __init__(self):
        super().__init__()
        self.node_id = "node_rule"
        self.node_name = "逻辑规则"

    def verify(self, claim: str, context: str = "") -> NodeVote:
        t0 = time.time()
        issues = []

        # 规则1: 自指矛盾
        if self._self_contradiction(claim):
            issues.append("自指矛盾")

        # 规则2: 数值常识
        if not self._number_sanity(claim):
            issues.append("数值异常")

        # 规则3: 绝对化表述
        if self._absolute_statement(claim):
            issues.append("绝对化表述")

        if issues:
            return NodeVote(self.node_id, self.node_name, Verdict.FAIL,
                            0.3, "; ".join(issues),
                            (time.time()-t0)*1000)
        else:
            return NodeVote(self.node_id, self.node_name, Verdict.PASS,
                            0.7, "逻辑规则通过",
                            (time.time()-t0)*1000)

    def _self_contradiction(self, text: str) -> bool:
        contradictions = [
            (r'是.*不(是|属于|存在)', r'同时也是'),
            (r'全部.*(不|没|无)', r'部分'),
        ]
        for pat_a, pat_b in contradictions:
            if re.search(pat_a, text) and re.search(pat_b, text):
                return True
        return False

    def _number_sanity(self, text: str) -> bool:
        nums = re.findall(r'\d+', text)
        for n in nums:
            val = int(n)
            if val > 10**12:
                return False
        return True

    def _absolute_statement(self, text: str) -> bool:
        absolutes = ['永远', '绝对', '一切', '所有', '从不', '毫无', '100%']
        for w in absolutes:
            if w in text:
                return True
        return False


class SelfModelNode(VerifierNode):
    """节点4: 自我模型置信度评估"""

    def __init__(self, self_model=None):
        super().__init__()
        self.self_model = self_model
        self.node_id = "node_self"
        self.node_name = "置信度评估"

    def verify(self, claim: str, context: str = "") -> NodeVote:
        t0 = time.time()
        if not self.self_model or self.self_model is True:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence="SelfModel 不可用",
                            latency_ms=(time.time()-t0)*1000)

        try:
            # 评估置信度
            if hasattr(self.self_model, 'assess_response_confidence'):
                confidence = self.self_model.assess_response_confidence(claim, context)
            else:
                confidence = 0.6  # 默认

            if confidence >= 0.7:
                return NodeVote(self.node_id, self.node_name, Verdict.PASS,
                                confidence, f"置信度 {confidence:.2f}",
                                (time.time()-t0)*1000)
            elif confidence >= 0.3:
                return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                                confidence, f"置信度中性 {confidence:.2f}",
                                (time.time()-t0)*1000)
            else:
                return NodeVote(self.node_id, self.node_name, Verdict.FAIL,
                                confidence, f"置信度不足 {confidence:.2f}",
                                (time.time()-t0)*1000)
        except Exception as e:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence=f"评估异常: {e}",
                            latency_ms=(time.time()-t0)*1000)


class CrossReferenceNode(VerifierNode):
    """节点5: 跨引用一致性检查"""

    def __init__(self):
        super().__init__()
        self.node_id = "node_xref"
        self.node_name = "跨引用一致"

    def verify(self, claim: str, context: str = "") -> NodeVote:
        t0 = time.time()
        if not context:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            evidence="无上下文可比对",
                            latency_ms=(time.time()-t0)*1000)

        # 检查 claim 是否与上下文冲突
        claim_keywords = set(self._tokenize(claim))
        context_keywords = set(self._tokenize(context))

        overlap = claim_keywords & context_keywords
        if len(overlap) > 3:
            # 有足够关键词重叠 → 一致性较高
            return NodeVote(self.node_id, self.node_name, Verdict.PASS,
                            0.75, f"与上下文一致 ({len(overlap)} 个关键词重叠)",
                            (time.time()-t0)*1000)
        elif len(overlap) == 0 and len(claim_keywords) > 3:
            # 无重叠 → 可能偏题
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            0.4, "与上下文无关键词重叠",
                            (time.time()-t0)*1000)
        else:
            return NodeVote(self.node_id, self.node_name, Verdict.ABSTAIN,
                            0.5, "信息不足",
                            (time.time()-t0)*1000)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'[a-zA-Z0-9_]+', text)


# ══════════════════════════════════════════════════════════════
# Ψ-Net 共识引擎
# ══════════════════════════════════════════════════════════════

class PsiNet:
    """
    Ψ-Net 蜂群共识引擎。

    多节点并行验算同一条断言，加权投票达成共识。
    分歧触发"分歧点分析"——找到争议焦点。
    """

    def __init__(self):
        self.nodes: Dict[str, VerifierNode] = {}
        self._vote_history: List[ConsensusResult] = []
        self._max_history = 100
        self._register_default_nodes()

    def _register_default_nodes(self):
        """注册五个默认验算节点"""
        self.register_node(WorldModelNode())
        self.register_node(CausalConsistencyNode())
        self.register_node(RuleBasedNode())
        self.register_node(SelfModelNode())
        self.register_node(CrossReferenceNode())

    def register_node(self, node: VerifierNode):
        """注册一个验算节点"""
        self.nodes[node.node_id] = node
        logger.info(f"Ψ-Net 节点注册: {node.node_name} ({node.node_id})")

    def set_world_model(self, wm):
        """设置世界模型到 WM 节点"""
        if "node_wm" in self.nodes:
            self.nodes["node_wm"].world_model = wm

    def set_causal_engine(self, ce):
        """设置因果引擎到因果节点"""
        if "node_causal" in self.nodes:
            self.nodes["node_causal"].causal_engine = ce

    def set_self_model(self, sm):
        """设置自我模型"""
        if "node_self" in self.nodes:
            self.nodes["node_self"].self_model = sm

    # ── 核心验算 ────────────────────────────────────────

    def verify(self, claim: str, context: str = "",
               min_consensus: float = 0.6) -> ConsensusResult:
        """
        对一个断言进行 Ψ-Net 共识验算。

        Args:
            claim: 待验算断言
            context: 上下文文本（用于跨引用检查）
            min_consensus: 共识阈值

        Returns:
            ConsensusResult
        """
        t0 = time.time()
        result = ConsensusResult(claim=claim)

        # 各节点并行投票
        votes = []
        for node_id, node in self.nodes.items():
            try:
                vote = node.verify(claim, context)
                votes.append(vote)
            except Exception as e:
                votes.append(NodeVote(node_id, node.node_name,
                                      Verdict.ABSTAIN, 0.0,
                                      f"节点异常: {e}"))
        result.votes = votes

        # 加权共识计算
        self._compute_consensus(result, min_consensus)

        # 分歧点检测
        result.divergence_points = self._detect_divergence(votes)
        result.total_latency_ms = (time.time() - t0) * 1000

        # 记录
        self._vote_history.append(result)
        if len(self._vote_history) > self._max_history:
            self._vote_history.pop(0)

        return result

    def _compute_consensus(self, result: ConsensusResult, threshold: float):
        """加权共识计算"""
        votes = result.votes
        if not votes:
            result.consensus = Verdict.ABSTAIN
            result.consensus_score = 0.0
            return

        # 权重配置（可调）
        weights = {
            "node_wm": 1.0,      # 世界模型权重最高
            "node_causal": 0.8,  # 因果次之
            "node_rule": 0.6,    # 逻辑规则
            "node_self": 0.5,    # 置信度
            "node_xref": 0.4,    # 跨引用
        }

        pass_weight = 0.0
        fail_weight = 0.0
        total_active = 0.0

        for vote in votes:
            w = weights.get(vote.node_id, 0.5)
            conf = vote.confidence

            if vote.verdict == Verdict.PASS:
                pass_weight += w * conf
                total_active += w
            elif vote.verdict == Verdict.FAIL:
                fail_weight += w * conf
                total_active += w
            # ABSTAIN 弃权票不参与

        if total_active == 0:
            result.consensus = Verdict.ABSTAIN
            result.consensus_score = 0.5
            return

        pass_ratio = pass_weight / total_active
        fail_ratio = fail_weight / total_active

        # 共识分数: 通过比例 - 失败比例，归一化到 0-1
        score = (pass_ratio - fail_ratio + 1) / 2
        result.consensus_score = max(0.0, min(1.0, score))

        # 判定
        if score >= threshold:
            result.consensus = Verdict.PASS
        elif score <= 1 - threshold:
            result.consensus = Verdict.FAIL
        else:
            result.consensus = Verdict.ABSTAIN

    def _detect_divergence(self, votes: List[NodeVote]) -> List[str]:
        """检测分歧点——哪些节点投了不同的票"""
        by_type = defaultdict(list)
        for v in votes:
            by_type[v.verdict].append(v.node_name)

        points = []
        if len(by_type) > 1:
            for verdict, names in by_type.items():
                points.append(f"{verdict.value}: {', '.join(names)}")
        return points

    # ── 查询 ────────────────────────────────────────────

    def get_stats(self) -> Dict:
        total = len(self._vote_history)
        passes = sum(1 for r in self._vote_history if r.consensus == Verdict.PASS)
        fails = sum(1 for r in self._vote_history if r.consensus == Verdict.FAIL)
        return {
            "total_votes": total,
            "passes": passes,
            "fails": fails,
            "avg_consensus": sum(r.consensus_score for r in self._vote_history) / max(1, total),
            "node_count": len(self.nodes),
            "nodes": {nid: n.node_name for nid, n in self.nodes.items()},
        }


# ── 快捷创建 ────────────────────────────────────────────────

_psi_net: Optional[PsiNet] = None

def get_psi_net() -> PsiNet:
    global _psi_net
    if _psi_net is None:
        _psi_net = PsiNet()
    return _psi_net
