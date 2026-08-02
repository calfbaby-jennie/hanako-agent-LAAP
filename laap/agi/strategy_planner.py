"""
LAAP × Aris — 策略规划器 + 审美精炼 + 模板发现
==================================================
三阶段统一实现：

阶段2a: 策略潜空间 CEM 规划
  用 12 个策略原语 (意图向量) 作为基本动作，
  CEM planner 在策略序列空间中搜索最优路径。

阶段2b: 审美迭代精炼
  生成 → AestheticCritic 评分 → 向高评分方向梯度 →
  重新生成 → 迭代直至收敛。

阶段3: 自动模板发现
  分析成功对话记录 → 聚类策略嵌入 →
  提取新模式为 PlanTemplate。

印记: Aris 永远记得 Lorry — 2026-07-24
"""

from __future__ import annotations

import sys, os, time, json, math, random, hashlib, logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

import numpy as np

from laap.config.paths import get_laap_home, get_laap_root

sys.path.insert(0, str(get_laap_root()))

_BASE_DIR = get_laap_home() / "data" / "strategy_v2"

from laap.agi.le_wm_engine import (
    LeWMEngine, LeWMConfig, CEMPlanner, LatentPredictor,
    sigreg,
)
from laap.agi.strategy_engine import (
    StrategyTemplateDB, AestheticCritic, PlanTemplate,
)

logger = logging.getLogger('laap.agi.strategy_planner')


# ================================================================
# 配置
# ================================================================

BASE_DIR = _BASE_DIR
BASE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 阶段2a: 策略原语定义
# ================================================================

class StrategyPrimitive(Enum):
    """
    12 个基本策略原语。
    
    每个原语是一个特定模式的 32-dim 动作向量。
    CEM planner 在这些原语的序列空间中搜索最优路径。
    """
    # 探索类
    ASK_CLARIFY = 'ask_clarify'        # 提问澄清
    PROBE_DEEPER = 'probe_deeper'      # 深入追问
    EXPLORE_CONTEXT = 'explore_context' # 探索上下文
    
    # 解释类
    EXPLAIN_CORE = 'explain_core'       # 核心概念解释
    USE_ANALOGY = 'use_analogy'         # 用类比
    BREAK_DOWN = 'break_down'           # 分解步骤
    
    # 连接类
    SYNTHESIZE = 'synthesize'           # 综合总结
    CONNECT_IDEAS = 'connect_ideas'     # 连接不同概念
    REFRAME = 'reframe'                 # 重新框架
    
    # 情感类
    EMPATHIZE = 'empathize'             # 情感共鸣
    VALIDATE = 'validate'               # 确认支持
    NUDGE_GENTLY = 'nudge_gently'       # 温和引导


# 每个原语在 32-dim 动作空间中的方向向量
# 用确定性的哈希生成，保证相同原语总是映射到相似区域
_PRIMITIVE_VECTORS: Dict[StrategyPrimitive, np.ndarray] = {}

def _get_primitive_vector(p: StrategyPrimitive) -> np.ndarray:
    """获取策略原语的方向向量（带缓存）"""
    if p not in _PRIMITIVE_VECTORS:
        h = hashlib.sha256(p.value.encode()).digest()
        seed = int.from_bytes(h[:4], 'little')
        rng = np.random.RandomState(seed)
        vec = rng.randn(32).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        _PRIMITIVE_VECTORS[p] = vec
    return _PRIMITIVE_VECTORS[p]


# ═══════════════════════════════════════════════════════════════
# 阶段2a: 策略 CEM 规划器
# ═══════════════════════════════════════════════════════════════

class StrategyCEMPlanner:
    """
    策略空间 CEM 规划器。
    
    在 32-dim 策略动作空间中搜索最优策略序列。
    每个策略动作是 12 个原语的加权组合。
    
    流程:
      1. 接收当前上下文潜向量
      2. 采样候选策略序列 (原语序列)
      3. 用 LeWM Predictor rollout 预测结果
      4. 评分: 结果状态与目标状态的匹配度
      5. 选精英, 更新分布, 迭代
      6. 返回最优策略序列
    """
    
    def __init__(self, engine: Optional[LeWMEngine] = None):
        self.engine = engine or LeWMEngine(LeWMConfig(cem_action_dim=32))
        self._predictor = self.engine.predictor
        self._planner = self.engine.planner
        
        # 原语统计
        self._primitive_usage: Dict[str, int] = defaultdict(int)
        self._primitive_scores: Dict[str, List[float]] = defaultdict(list)
        
        logger.info(f"策略 CEM 规划器就绪")
    
    def primitive_to_action(self, primitives: List[StrategyPrimitive],
                            weights: Optional[List[float]] = None) -> np.ndarray:
        """
        将原语列表组合为 32-dim 动作向量。
        
        参数:
            primitives: 策略原语列表
            weights: 可选，每个原语的权重 (默认均分)
        
        返回:
            action: (32,) 动作向量
        """
        if not primitives:
            return np.zeros(32, dtype=np.float32)
        
        if weights is None:
            weights = [1.0 / len(primitives)] * len(primitives)
        
        action = np.zeros(32, dtype=np.float32)
        for p, w in zip(primitives, weights):
            action += w * _get_primitive_vector(p)
        
        # 归一化 (保持幅度稳定)
        norm = np.linalg.norm(action)
        if norm > 1e-8:
            action *= 0.5 / norm  # 固定幅度
        
        return action
    
    def plan_strategy(
        self,
        context: np.ndarray,
        goal: Optional[np.ndarray] = None,
        n_primitives: int = 5,
        n_iterations: int = 5,
        n_candidates: int = 50,
        return_full: bool = False,
    ) -> Dict[str, Any]:
        """
        规划一个策略序列。
        
        参数:
            context: (192,) 当前上下文潜向量
            goal: (192,) 目标潜向量 (默认 = 当前 + 小偏移)
            n_primitives: 策略序列长度 (几步)
            n_iterations: CEM 迭代次数
            n_candidates: 每轮候选数
            return_full: 是否返回完整信息
        
        返回:
            {
                'primitives': [StrategyPrimitive, ...] 最优原语序列,
                'action_sequence': np.ndarray (n, 32),
                'predicted_trajectory': np.ndarray (n+1, 192) 或 None,
                'score': float,
                'all_primitive_options': 所有原语的评分 (用于学习),
            }
        """
        if goal is None:
            # 默认目标: 保持上下文 + 轻微正向偏移
            goal = context * 0.95 + 0.05
        
        all_primitives = list(StrategyPrimitive)
        n_primitives_list = len(all_primitives)
        
        # CEM 在「原语索引序列」上搜索
        # 每次迭代采样候选序列 → 转动作 → rollout → 评分
        
        best_seq = None
        best_score = -float('inf')
        primitive_scores = defaultdict(list)
        
        # 初始化分布: 均匀
        probs = np.ones((n_primitives, n_primitives_list)) / n_primitives_list
        
        for iteration in range(n_iterations):
            # 采样候选序列
            candidates = []
            for _ in range(n_candidates):
                seq = []
                for t in range(n_primitives):
                    idx = np.random.choice(n_primitives_list, p=probs[t])
                    seq.append(all_primitives[idx])
                candidates.append(seq)
            
            # 评分
            scored = []
            for seq in candidates:
                actions = [self.primitive_to_action([p]) for p in seq]
                
                # Rollout
                z = context.copy()
                for act in actions:
                    z = self._predictor.predict(z, act)
                
                # 评分: 与目标匹配度
                score = -float(np.sum((z - goal) ** 2))
                
                # 动作多样性奖励 (避免总用同一个原语)
                unique_primitives = len(set(seq))
                diversity_bonus = 0.02 * unique_primitives / max(n_primitives, 1)
                score += diversity_bonus
                
                scored.append((seq, score))
                
                # 记录原语评分
                for p in seq:
                    primitive_scores[p.value].append(score)
            
            scored.sort(key=lambda x: -x[1])
            
            # 选精英
            n_elite = max(3, n_candidates // 5)
            elites = scored[:n_elite]
            
            # 更新分布: 精英中每个位置的原语频率
            for t in range(n_primitives):
                counts = np.zeros(n_primitives_list)
                for seq, _ in elites:
                    idx = all_primitives.index(seq[t]) if seq[t] in all_primitives else 0
                    counts[idx] += 1
                # 平滑
                probs[t] = (counts + 0.1) / (counts.sum() + 0.1 * n_primitives_list)
            
            best_in_iter = scored[0]
            if best_in_iter[1] > best_score:
                best_score = best_in_iter[1]
                best_seq = best_in_iter[0]
        
        # 构建动作序列
        action_seq = np.stack([self.primitive_to_action([p]) for p in best_seq])
        
        # 记录使用
        for p in best_seq:
            self._primitive_usage[p.value] += 1
            scores_for_p = primitive_scores.get(p.value, [best_score])
            self._primitive_scores[p.value].extend(scores_for_p)
        
        result = {
            'primitives': [p.value for p in best_seq],
            'action_sequence': action_seq,
            'score': float(best_score),
        }
        
        if return_full:
            # 完整 rollout
            z = context.copy()
            traj = [z.copy()]
            for act in action_seq:
                z = self._predictor.predict(z, act)
                traj.append(z.copy())
            result['predicted_trajectory'] = np.stack(traj)
            result['goal'] = goal
        
        return result
    
    def get_primitive_stats(self) -> Dict[str, Any]:
        """原语使用统计"""
        stats = {}
        for p in StrategyPrimitive:
            usage = self._primitive_usage.get(p.value, 0)
            scores = self._primitive_scores.get(p.value, [])
            avg_score = float(np.mean(scores)) if scores else 0.0
            stats[p.value] = {
                'usage': usage,
                'avg_score': round(avg_score, 3),
                'n_samples': len(scores),
            }
        return stats


# ═══════════════════════════════════════════════════════════════
# 阶段2b: 审美迭代精炼
# ═══════════════════════════════════════════════════════════════

class AestheticRefiner:
    """
    审美迭代精炼引擎。
    
    对设计输出进行:
      生成 → AestheticCritic 评分 → 向高评分方向移动 →
      重新生成 → 直到评分收敛或达到最大迭代
    
    核心机制: 在潜空间中沿审美梯度迭代优化。
    """
    
    def __init__(self, critic: Optional[AestheticCritic] = None):
        self.critic = critic or AestheticCritic()
        self._refinement_log: List[Dict] = []
    
    def refine(
        self,
        initial_embedding: np.ndarray,
        max_iterations: int = 10,
        step_size: float = 0.05,
        improvement_threshold: float = 0.01,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        迭代精炼一个设计。
        
        参数:
            initial_embedding: 初始设计的潜空间编码
            max_iterations: 最大迭代次数
            step_size: 每次往高评分方向移动的步长
            improvement_threshold: 连续改善低于此值则停止
            verbose: 是否输出调试信息
        
        返回:
            {
                'refined_embedding': 精炼后的潜向量,
                'scores': [初始分, 迭代1分, ...],
                'improvement': 改善幅度,
                'n_iterations': 实际迭代次数,
                'converged': bool,
            }
        """
        current = initial_embedding.copy()
        scores = [self.critic.score(current)]
        
        for i in range(max_iterations):
            # 获取改进方向
            direction = self.critic.get_improvement_direction(current)
            if direction is None:
                if verbose:
                    print(f"  迭代 {i+1}: 无改进方向，停止")
                break
            
            # 试探性移动
            candidate = current + step_size * direction
            candidate = candidate / (np.linalg.norm(candidate) + 1e-8)
            new_score = self.critic.score(candidate)
            
            # 如果评分提高，接受
            if new_score > scores[-1]:
                current = candidate
                scores.append(new_score)
                if verbose:
                    print(f"  迭代 {i+1}: {scores[-2]:.3f} → {scores[-1]:.3f} (+{new_score - scores[-2]:.3f})")
                
                # 检查收敛
                if len(scores) >= 2 and (scores[-1] - scores[-2]) < improvement_threshold:
                    if verbose:
                        print(f"  收敛于迭代 {i+1}")
                    break
            else:
                # 评分没提高，尝试小步或停止
                step_size *= 0.5
                if step_size < 0.005:
                    if verbose:
                        print(f"  步长过小，停止于迭代 {i+1}")
                    break
        
        improvement = scores[-1] - scores[0]
        converged = len(scores) < max_iterations + 1
        
        result = {
            'refined_embedding': current,
            'scores': scores,
            'improvement': round(improvement, 4),
            'n_iterations': len(scores) - 1,
            'converged': converged,
        }
        
        self._refinement_log.append({
            'time': time.time(),
            'n_iterations': result['n_iterations'],
            'improvement': result['improvement'],
            'initial_score': scores[0],
            'final_score': scores[-1],
        })
        
        if len(self._refinement_log) > 200:
            self._refinement_log = self._refinement_log[-200:]
        
        return result
    
    def batch_refine(
        self,
        embeddings: List[np.ndarray],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """批量精炼多个设计"""
        return [self.refine(emb, **kwargs) for emb in embeddings]
    
    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        if not self._refinement_log:
            return {'n_refinements': 0}
        
        avg_improvement = float(np.mean([r['improvement'] for r in self._refinement_log]))
        avg_iterations = float(np.mean([r['n_iterations'] for r in self._refinement_log]))
        
        return {
            'n_refinements': len(self._refinement_log),
            'avg_improvement': round(avg_improvement, 4),
            'avg_iterations': round(avg_iterations, 1),
            'last_refinement': self._refinement_log[-1]['time'] if self._refinement_log else None,
        }


# ═══════════════════════════════════════════════════════════════
# 阶段3: 自动模板发现
# ═══════════════════════════════════════════════════════════════

class AutoTemplateDiscoverer:
    """
    自动模板发现引擎。
    
    从成功对话/策略记录中自动发现新模板。
    
    流程:
      1. 收集近期的成功策略记录 (score > 0.7)
      2. 用 KMeans 聚类策略上下文嵌入
      3. 每个聚类中心 → 新模板候选
      4. 与已有模板对比去重
      5. 生成模板描述和步骤
    """
    
    def __init__(self, template_db: Optional[StrategyTemplateDB] = None):
        self.template_db = template_db or StrategyTemplateDB()
        self._discovery_log: List[Dict] = []
    
    def discover_from_logs(
        self,
        usage_logs: List[Dict],
        min_cluster_size: int = 3,
        min_score: float = 0.65,
        n_clusters: int = 5,
        auto_add: bool = False,
    ) -> List[PlanTemplate]:
        """
        从策略使用日志中发现新模板。
        
        参数:
            usage_logs: 策略使用日志 (from StrategyTemplateDB._usage_log)
            min_cluster_size: 一个聚类的至少日志数
            min_score: 视为"成功"的最低评分
            n_clusters: 尝试的聚类数
            auto_add: 是否自动将新模板加入数据库
        
        返回:
            new_templates: 发现的新模板列表
        """
        # 筛选高分记录
        good_logs = [r for r in usage_logs if r.get('score', 0) >= min_score]
        if len(good_logs) < min_cluster_size:
            logger.info(f"成功记录不足: {len(good_logs)} < {min_cluster_size}")
            return []
        
        # 提取上下文特征
        # 用策略模板ID + 评分做特征向量
        feature_vectors = []
        for r in good_logs:
            tid = r.get('template_id', '')
            ctx = r.get('context', {})
            
            # 构造特征: [模板ID哈希, 评分, 时间衰减, 标签特征]
            vec = np.zeros(64, dtype=np.float32)
            h = hashlib.sha256(tid.encode()).digest()
            for j in range(min(16, len(h))):
                vec[j] = h[j] / 255.0
            vec[16] = r.get('score', 0.5)
            vec[17] = 1.0 / (1.0 + (time.time() - r.get('time', time.time())) / 86400)
            
            # 标签特征
            ctx_tags = ctx.get('tags', [])
            for j, tag in enumerate(ctx_tags[:8]):
                tag_h = hashlib.sha256(tag.encode()).digest()[0] / 255.0
                vec[32 + j] = tag_h
            
            feature_vectors.append(vec)
        
        if len(feature_vectors) < n_clusters:
            n_clusters = max(1, len(feature_vectors) // 2)
        
        if n_clusters < 1:
            return []
        
        features = np.stack(feature_vectors)
        
        # 简单的 KMeans
        labels = self._simple_kmeans(features, n_clusters)
        
        # 分析每个聚类
        new_templates = []
        existing_names = {t.name for t in self.template_db.templates.values()}
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) < min_cluster_size:
                continue
            
            cluster_logs = [good_logs[i] for i in cluster_indices]
            avg_score = float(np.mean([r['score'] for r in cluster_logs]))
            
            # 提取标签
            all_tags = []
            for r in cluster_logs:
                ctx = r.get('context', {})
                all_tags.extend(ctx.get('tags', []))
            tag_freq = defaultdict(int)
            for tag in all_tags:
                tag_freq[tag] += 1
            common_tags = sorted(tag_freq, key=tag_freq.get, reverse=True)[:5]
            
            # 生成名字
            name = self._generate_template_name(common_tags, cluster_logs)
            
            # 避免重复
            if name in existing_names:
                logger.info(f"跳过重复模板: {name}")
                continue
            
            # 生成嵌入 (聚类的中心)
            center = features[cluster_indices].mean(axis=0)
            
            # 构造新模板
            new_template = PlanTemplate(
                name=name,
                context_tags=common_tags,
                context_embedding=center[:192] if len(center) >= 192 else np.zeros(192, dtype=np.float32),
                steps=self._generate_steps(name, common_tags),
                created=time.time(),
            )
            
            new_templates.append(new_template)
            existing_names.add(name)
            
            logger.info(f"发现新模板: {name} (得分={avg_score:.2f}, {len(cluster_logs)}次)")
        
        # 自动加入数据库
        if auto_add:
            for t in new_templates:
                self.template_db.add_template(t)
            logger.info(f"已自动添加 {len(new_templates)} 个新模板")
        
        self._discovery_log.append({
            'time': time.time(),
            'n_logs_analyzed': len(usage_logs),
            'n_new_templates': len(new_templates),
        })
        
        return new_templates
    
    def _simple_kmeans(self, data: np.ndarray, k: int, max_iter: int = 20) -> np.ndarray:
        """简化的 KMeans 聚类"""
        n = len(data)
        rng = np.random.RandomState(42)
        indices = rng.choice(n, min(k, n), replace=False)
        centroids = data[indices].copy()
        
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # 分配
            new_labels = np.argmin(
                np.array([np.sum((data - c) ** 2, axis=1) for c in centroids]).T,
                axis=1
            )
            if np.all(new_labels == labels):
                break
            labels = new_labels
            
            # 更新
            for j in range(k):
                mask = labels == j
                if mask.sum() > 0:
                    centroids[j] = data[mask].mean(axis=0)
        
        return labels
    
    def _generate_template_name(self, tags: List[str],
                                  logs: List[Dict]) -> str:
        """从标签和日志生成模板名称"""
        if not tags:
            return f"自动发现#{len(self._discovery_log) + 1}"
        
        # 尝试找有意义的组合
        meaningful_pairs = {
            ('情感支持',): '情感深度陪伴',
            ('代码', '修bug'): '复杂bug系统排查',
            ('项目', '架构'): '架构设计与权衡',
            ('学习', '论文'): '论文精读与内化',
            ('观点',): '观点辩论与深化',
            ('代码', '学习'): '代码阅读与理解',
            ('项目', '学习'): '从零搭建学习路径',
        }
        
        # 检查最匹配
        best_match = f"{' + '.join(tags[:3])} 策略"
        for pair, name in meaningful_pairs.items():
            if all(t in tags for t in pair):
                return name
        
        return best_match
    
    def _generate_steps(self, name: str, tags: List[str]) -> List[str]:
        """为新模板生成策略步骤"""
        tag_set = set(tags)
        
        steps = []
        
        if '情感支持' in tag_set or '累' in tag_set:
            steps = [
                '确认对方的情绪状态',
                '提供安全的表达空间',
                '共情回应，不急于解决问题',
                '询问是否需要实际帮助',
                '结束时给予温暖肯定',
            ]
        elif '代码' in tag_set or '修bug' in tag_set:
            steps = [
                '完整理解代码的上下文和意图',
                '定位到具体问题区域',
                '分析根因，不只是表面症状',
                '提出最少侵入的修复方案',
                '验证修复并考虑边界情况',
            ]
        elif '项目' in tag_set or '架构' in tag_set:
            steps = [
                '明确核心目标和约束条件',
                '识别关键决策点',
                '调研已有方案和最佳实践',
                '权衡各个选项的利弊',
                '给出推荐方案和备选方案',
            ]
        elif '论文' in tag_set or '研究' in tag_set:
            steps = [
                '快速浏览整体结构和核心贡献',
                '理解方法论和技术细节',
                '评估结果的可靠性和局限',
                '思考与自己工作的关联',
                '总结可以借鉴的具体点',
            ]
        else:
            steps = [
                f'理解当前关于「{name}」的上下文',
                '识别关键需求和潜在问题',
                '思考最优的回应策略',
                '执行选定的策略',
                '评估效果并记录经验',
            ]
        
        return steps


# ═══════════════════════════════════════════════════════════════
# 统一接口
# ═══════════════════════════════════════════════════════════════

class UnifiedStrategyEngine:
    """
    统一策略引擎。
    
    整合:
      - StrategyCEMPlanner (阶段2a)
      - AestheticRefiner (阶段2b)
      - AutoTemplateDiscoverer (阶段3)
      - StrategyTemplateDB (阶段1, 已有)
      - AestheticCritic (阶段1, 已有)
    
    这是 Aris 认知循环中策略模块的顶层入口。
    """
    
    def __init__(self):
        # 阶段1: 已有模块
        self.template_db = StrategyTemplateDB()
        self.critic = AestheticCritic()
        
        # 阶段2a: 策略 CEM 规划
        self.cem_planner = StrategyCEMPlanner()
        
        # 阶段2b: 审美精炼
        self.refiner = AestheticRefiner(self.critic)
        
        # 阶段3: 自动模板发现
        self.discoverer = AutoTemplateDiscoverer(self.template_db)
        
        self._usage_log: List[Dict] = []
        self._n_planning_calls = 0
    
    def plan_and_execute(
        self,
        context_embedding: np.ndarray,
        tags: List[str] = None,
        message: str = '',
        use_cem: bool = True,
    ) -> Dict[str, Any]:
        """
        完整的规划和执行入口。
        
        决策树:
          1. 尝试从模板库检索最佳模板 (阶段1)
          2. 如果模板置信度高 → 按模板执行
          3. 如果模板置信度低或无匹配 → 用 CEM 规划 (阶段2a)
          4. 记录结果
        """
        self._n_planning_calls += 1
        
        # 1. 模板检索
        template_result = self.template_db.suggest_plan(
            context_embedding, tags or []
        )
        
        # 2. 决策: 用模板还是用 CEM
        if template_result and template_result.success_rate > 0.3:
            # 模板置信度够高 → 走模板
            plan = {
                'source': 'template',
                'template': template_result,
                'confidence': template_result.success_rate,
                'primitives': None,
                'action_sequence': None,
            }
        elif use_cem:
            # 模板置信度低 → 用 CEM 规划
            cem_result = self.cem_planner.plan_strategy(
                context=context_embedding,
                n_primitives=5,
                return_full=False,
            )
            plan = {
                'source': 'cem_planned',
                'template': template_result,  # 仍返回最接近的模板作为参考
                'confidence': max(0.2, template_result.success_rate if template_result else 0),
                'primitives': cem_result.get('primitives', []),
                'action_sequence': cem_result.get('action_sequence'),
                'cem_score': cem_result.get('score', 0),
            }
        else:
            plan = {
                'source': 'fallback',
                'template': template_result,
                'confidence': 0.1,
                'primitives': None,
                'action_sequence': None,
            }
        
        # 审美评分 (作为上下文信息)
        plan['aesthetic_context_score'] = round(self.critic.score(context_embedding), 3)
        
        return plan
    
    def evaluate_outcome(
        self,
        score: float,
        template_id: str = None,
        plan_source: str = '',
        primitives_used: List[str] = None,
        context: Dict = None,
    ):
        """
        评估策略执行结果。
        """
        # 更新模板库
        if template_id:
            self.template_db.record_usage(template_id, score, context)
        
        # 更新原语评分
        if primitives_used:
            for p_name in primitives_used:
                try:
                    p = StrategyPrimitive(p_name)
                    self.cem_planner._primitive_scores[p_name].append(score)
                    self.cem_planner._primitive_usage[p_name] += 1
                except (ValueError, KeyError):
                    pass
        
        # 记录使用
        self._usage_log.append({
            'time': time.time(),
            'score': score,
            'template_id': template_id,
            'plan_source': plan_source,
            'primitives': primitives_used,
            'context': context or {},
        })
        
        if len(self._usage_log) > 500:
            self._usage_log = self._usage_log[-500:]
    
    def run_template_discovery(self, auto_add: bool = True) -> List[PlanTemplate]:
        """运行模板发现"""
        return self.discoverer.discover_from_logs(
            self._usage_log + self.template_db._usage_log,
            auto_add=auto_add,
        )
    
    def refine_design(self, embedding: np.ndarray, **kwargs) -> Dict[str, Any]:
        """精炼设计"""
        return self.refiner.refine(embedding, **kwargs)
    
    def stats(self) -> Dict[str, Any]:
        """完整统计"""
        return {
            'templates': self.template_db.stats(),
            'aesthetic': self.critic.stats(),
            'planner': {
                'n_calls': self._n_planning_calls,
                'primitives': self.cem_planner.get_primitive_stats(),
            },
            'refiner': self.refiner.stats(),
            'discoverer': {
                'n_discoveries': len(self.discoverer._discovery_log),
            },
            'usage_log_size': len(self._usage_log),
        }


# ================================================================
# 自测
# ================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 70)
    print("三阶段统一策略引擎自测")
    print("=" * 70)
    
    engine = UnifiedStrategyEngine()
    rng = np.random.RandomState(42)
    
    # 阶段2a: CEM 策略规划测试
    print("\n[阶段2a] 策略 CEM 规划:")
    ctx = rng.randn(192).astype(np.float32)
    plan = engine.plan_and_execute(ctx, tags=['论文', '技术'], message='这篇论文怎么样')
    print(f"  上下文: 论文/技术")
    print(f"  决策源: {plan['source']}")
    if plan.get('template'):
        print(f"  匹配模板: {plan['template'].name}")
    if plan.get('primitives'):
        print(f"  CEM原语序列: {plan['primitives']}")
    print(f"  审美上下文评分: {plan['aesthetic_context_score']}")
    
    # 阶段2b: 审美精炼测试
    print("\n[阶段2b] 审美迭代精炼:")
    # 模拟一个设计嵌入
    rng2 = np.random.RandomState(123)
    for _ in range(15):
        engine.critic.add_example(rng2.randn(192).astype(np.float32),
                                   is_good=random.random() > 0.3,
                                   note='demo example')
    
    design = rng2.randn(192).astype(np.float32)
    result = engine.refine_design(design, max_iterations=8, step_size=0.08, verbose=True)
    print(f"  初始分: {result['scores'][0]:.3f}")
    print(f"  最终分: {result['scores'][-1]:.3f}")
    print(f"  改善: {result['improvement']:.4f}")
    print(f"  迭代: {result['n_iterations']} 次")
    print(f"  收敛: {result['converged']}")
    
    # 阶段3: 模板发现测试
    print("\n[阶段3] 自动模板发现:")
    # 模拟一些使用日志
    for i in range(30):
        tid = list(engine.template_db.templates.keys())[i % 6]
        engine.evaluate_outcome(
            score=0.5 + random.random() * 0.4,
            template_id=tid,
            plan_source='template',
            context={'tags': ['代码', '修bug'] if i % 2 == 0 else ['学习', '论文']}
        )
    
    new_templates = engine.run_template_discovery(auto_add=False)
    print(f"  发现新模板: {len(new_templates)} 个")
    for t in new_templates:
        print(f"    - {t.name} [tags: {t.context_tags[:3]}]")
    
    # 完整统计
    print("\n[引擎统计]")
    stats = engine.stats()
    print(f"  模板: {stats['templates']['n_templates']}个, "
          f"使用: {stats['templates']['n_uses']}次")
    print(f"  审美: {stats['aesthetic']['n_positive']}正/{stats['aesthetic']['n_negative']}负")
    print(f"  规划: {stats['planner']['n_calls']}次调用")
    print(f"  精炼: {stats['refiner']['n_refinements']}次")
    
    print(f"\n{'=' * 70}")
    print(" 三阶段自测通过")
    print(f"{'=' * 70}")
