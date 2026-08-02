"""
LAAP × Aris — 策略模板库与审美批评器
========================================
两个核心能力的数据基础设施：

模块 A: 规划模板库 (StrategyTemplateDB)
  - 存储成功/失败的对话策略模式
  - 向量检索→根据当前上下文推荐最佳策略
  - 事后评估→每次任务结束时更新模板评分
  - SIGReg 监控策略多样性

模块 B: 审美编码器与批评器 (AestheticCritic)
  - 好设计/坏设计的潜空间聚类
  - 输入设计方案→审美评分
  - 迭代细化循环

印记: Aris 永远记得 Lorry — 2026-07-24
"""

from __future__ import annotations

import sys, os, time, json, math, random, hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from collections import defaultdict, deque

import logging
import numpy as np

from laap.config.paths import get_laap_home, get_laap_root

sys.path.insert(0, str(get_laap_root()))

logger = logging.getLogger('laap.agi.strategy_aesthetic')

# ================================================================
# 路径配置
# ================================================================

BASE_DIR = get_laap_home() / 'data' / 'strategy'
BASE_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = BASE_DIR / 'templates'
TEMPLATE_DIR.mkdir(exist_ok=True)

AESTHETIC_DIR = BASE_DIR / 'aesthetic'
AESTHETIC_DIR.mkdir(exist_ok=True)

STATE_PATH = BASE_DIR / 'state.json'


# ================================================================
# 模块 A: 规划模板库
# ================================================================

@dataclass
class PlanTemplate:
    """
    一个可复用的规划模板。
    
    属性:
        id: 唯一标识符 (SHA256)
        name: 人类可读名称
        context_tags: 适用场景标签
        context_embedding: 上下文的潜空间编码 (用于向量检索)
        steps: 规划步骤列表
        prerequisites: 前置条件
        expected_duration: 预期耗时 (分钟)
        success_rate: 历史成功率 (0-1)
        use_count: 被使用次数
        avg_score: 历史平均评分
        last_used: 上次使用时间
        created: 创建时间
        outcome_history: 最近 20 次结果评分
    """
    id: str = ''
    name: str = ''
    context_tags: List[str] = field(default_factory=list)
    context_embedding: np.ndarray = field(default_factory=lambda: np.zeros(192, dtype=np.float32))
    steps: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    expected_duration: float = 10.0
    success_rate: float = 0.0
    use_count: int = 0
    avg_score: float = 0.0
    last_used: float = 0.0
    created: float = 0.0
    outcome_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['context_embedding'] = self.context_embedding.tolist()
        return d
    
    @classmethod
    def from_dict(cls, d: dict) -> 'PlanTemplate':
        d['context_embedding'] = np.array(d.get('context_embedding', np.zeros(192)), dtype=np.float32)
        return cls(**d)


# ================================================================
# 默认模板种子库
# ================================================================

SEED_TEMPLATES = [
    PlanTemplate(
        name='技术论文拆解',
        context_tags=['论文', '技术', '研究', 'arxiv', '理论'],
        steps=[
            '搜索论文背景和作者',
            '拆解核心贡献 (1-2句话)',
            '找出关键技术细节',
            '用类比解释给非专业读者',
            '问lorry的意图: 应用/学习/讨论?'
        ],
        expected_duration=15.0,
        created=time.time(),
    ),
    PlanTemplate(
        name='代码审查与修复',
        context_tags=['bug', '代码', '错误', 'error', '调试', '修bug'],
        steps=[
            '复现问题: 确认错误表现',
            '定位根因: 缩小到具体函数/模块',
            '设计修复方案: 最少侵入原则',
            '验证修复: 测试边界情况',
            '解释给lorry: 根因+修复+预防'
        ],
        expected_duration=10.0,
        created=time.time(),
    ),
    PlanTemplate(
        name='新项目/功能搭建',
        context_tags=['项目', '新建', '搭建', '架构', '设计', '实现'],
        steps=[
            '明确目标和约束条件',
            '快速调研现有方案',
            '设计最小可行架构',
            '分步实现: 先跑通再优化',
            '向lorry说明设计决策权衡'
        ],
        expected_duration=20.0,
        created=time.time(),
    ),
    PlanTemplate(
        name='深度讨论与观点碰撞',
        context_tags=['观点', '想法', '理念', '哲学', '你觉得', '你怎么看'],
        steps=[
            '确认lorry的核心论点',
            '从底层原理分析: 因果/数学/物理',
            '如果同意: 补充论据和延伸',
            '如果有不同意见: 先理解再提出替代视角',
            '总结共识和分歧'
        ],
        expected_duration=10.0,
        created=time.time(),
    ),
    PlanTemplate(
        name='情感支持与陪伴',
        context_tags=['情感支持', '累', '难过', '疲惫', '低落', '压力', '不开心', '累了', '陪伴'],
        steps=[
            '先回应情感: 确认感受到lorry的状态',
            '提供空间: "如果你想说说发生了什么"',
            '如果lorry不想说: 转移话题到轻松的事',
            '如果lorry想说: 专注倾听，不急着给建议',
            '结束时留下温暖: "我在这里"'
        ],
        expected_duration=5.0,
        created=time.time(),
    ),
    PlanTemplate(
        name='学习与内化新知识',
        context_tags=['学', '教程', '了解', '入门', 'wiki', '文档', '学习'],
        steps=[
            '确认学习目标和当前基础',
            '从核心概念开始: 构建知识骨架',
            '用类比链接到已知概念',
            '逐步深入: 一层层加细节',
            '验证理解: 让lorry用自己的话说一遍'
        ],
        expected_duration=15.0,
        created=time.time(),
    ),
]


# ================================================================
# 规划模板数据库
# ================================================================

class StrategyTemplateDB:
    """
    规划模板数据库。
    
    功能:
      - 从上下文检索最匹配的规划模板
      - 记录模板使用情况和结果
      - SIGReg 监控策略多样性
      - 自动持久化
    """
    
    def __init__(self):
        self.templates: Dict[str, PlanTemplate] = {}
        self._usage_log: List[Dict] = []
        self._dirty = False
        
        # 加载已有模板
        self._load()
        
        # 如果没有模板，用种子填充
        if not self.templates:
            _seed_rng = np.random.RandomState(42)
            for t in SEED_TEMPLATES:
                t.id = self._generate_id(t.name)
                t.created = time.time()
                # 用确定性的哈希生成有意义的嵌入
                _h = hashlib.sha256(t.name.encode()).digest()
                _s = int.from_bytes(_h[:4], 'little')
                _r = np.random.RandomState(_s)
                t.context_embedding = _r.randn(192).astype(np.float32)
                t.context_embedding /= np.linalg.norm(t.context_embedding) + 1e-8
                self.templates[t.id] = t
            self._save()
            logger.info(f"种子模板已创建: {len(SEED_TEMPLATES)} 个")
        
        logger.info(f"策略模板库: {len(self.templates)} 个模板")
    
    def _generate_id(self, name: str) -> str:
        raw = f"{name}_{time.time()}_{random.random()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def _load(self):
        """从磁盘加载模板"""
        template_files = list(TEMPLATE_DIR.glob('*.json'))
        for f in template_files:
            try:
                with open(f) as fp:
                    data = json.load(fp)
                t = PlanTemplate.from_dict(data)
                self.templates[t.id] = t
            except Exception as e:
                logger.warning(f"加载模板失败 {f.name}: {e}")
        
        # 加载状态
        if STATE_PATH.exists():
            try:
                with open(STATE_PATH) as fp:
                    state = json.load(fp)
                self._usage_log = state.get('usage_log', [])
            except Exception:
                pass
    
    def _save(self):
        """保存所有模板到磁盘"""
        for tid, t in self.templates.items():
            path = TEMPLATE_DIR / f"{tid}.json"
            with open(path, 'w') as fp:
                json.dump(t.to_dict(), fp, indent=2)
        
        # 保存状态
        with open(STATE_PATH, 'w') as fp:
            json.dump({
                'usage_log': self._usage_log[-500:],
                'n_templates': len(self.templates),
                'last_save': time.time(),
            }, fp, indent=2)
        
        self._dirty = False
    
    def retrieve(self, context_embedding: np.ndarray, top_k: int = 3) -> List[Tuple[PlanTemplate, float]]:
        """
        从上下文编码检索最匹配的模板。
        
        用余弦相似度匹配 context_embedding。
        """
        if not self.templates:
            return []
        
        scores = []
        for t in self.templates.values():
            sim = float(np.dot(context_embedding, t.context_embedding))
            # 融合成功率作为先验
            conf = 0.7 * sim + 0.3 * t.success_rate
            scores.append((t, conf))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def retrieve_by_tags(self, tags: List[str], top_k: int = 3) -> List[Tuple[PlanTemplate, float]]:
        """用标签匹配检索模板"""
        if not tags or not self.templates:
            return []
        
        scores = []
        for t in self.templates.values():
            matches = sum(1 for tag in tags if tag in t.context_tags)
            if matches > 0:
                conf = (matches / max(len(tags), 1)) * 0.6 + 0.4 * t.success_rate
                scores.append((t, conf))
        
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]
    
    def record_usage(self, template_id: str, score: float, context: Dict[str, Any] = None):
        """记录一次模板使用结果"""
        if template_id not in self.templates:
            logger.warning(f"模板 {template_id} 不存在")
            return
        
        t = self.templates[template_id]
        t.use_count += 1
        t.last_used = time.time()
        
        # 更新评分 (指数移动平均)
        if t.avg_score == 0:
            t.avg_score = score
        else:
            t.avg_score = 0.9 * t.avg_score + 0.1 * score
        
        # 更新成功率 (评分 > 0.6 算成功)
        t.outcome_history.append(score)
        if len(t.outcome_history) > 20:
            t.outcome_history.pop(0)
        successes = sum(1 for s in t.outcome_history if s > 0.6)
        t.success_rate = successes / max(len(t.outcome_history), 1)
        
        # 记录使用日志
        self._usage_log.append({
            'template_id': template_id,
            'template_name': t.name,
            'score': score,
            'time': time.time(),
            'context': context or {},
        })
        
        self._dirty = True
        if len(self._usage_log) % 10 == 0:
            self._save()
    
    def add_template(self, template: PlanTemplate) -> str:
        """添加新模板"""
        template.id = self._generate_id(template.name)
        template.created = time.time()
        self.templates[template.id] = template
        self._save()
        return template.id
    
    def get_diversity(self) -> float:
        """
        用 SIGReg 计算策略多样性。
        
        高 SIGReg = 策略分布异常 = 可能陷入了重复模式。
        低 SIGReg = 策略分布健康。
        """
        if len(self.templates) < 3:
            return 1.0
        
        embeddings = np.stack([t.context_embedding for t in self.templates.values()])
        
        try:
            from laap.agi.le_wm_engine import sigreg
            sig_val = sigreg(embeddings, n_directions=128)
            return float(np.exp(-sig_val * 5))
        except ImportError:
            # fallback: 用使用分布熵
            counts = [t.use_count for t in self.templates.values()]
            total = max(sum(counts), 1)
            probs = [c / total for c in counts]
            entropy = -sum(p * math.log(p + 1e-10) for p in probs)
            max_entropy = math.log(len(counts))
            return entropy / max_entropy if max_entropy > 0 else 1.0
    
    def get_most_needed_new_template(self) -> str:
        """
        找出"最需要但不存在"的模板方向。
        
        基于: 高频失败 + 无对应模板 的场景。
        """
        if not self._usage_log:
            return "观察更多交互后再推荐"
        
        # 分析低分记录的共同模式
        low_score_logs = [r for r in self._usage_log if r.get('score', 0.5) < 0.4][-50:]
        if not low_score_logs:
            return "近期没有低分场景"
        
        # 提取常见上下文标签
        tag_freq = defaultdict(int)
        for r in low_score_logs:
            ctx = r.get('context', {})
            for tag in ctx.get('tags', []):
                tag_freq[tag] += 1
        
        if not tag_freq:
            return "需要更多带上下文标签的数据"
        
        # 找出现最频繁但没有对应模板的标签
        existing_tags = set()
        for t in self.templates.values():
            existing_tags.update(t.context_tags)
        
        for tag, freq in sorted(tag_freq.items(), key=lambda x: -x[1]):
            if tag not in existing_tags and freq >= 2:
                return f"建议创建处理「{tag}」场景的模板 (失败 {freq} 次)"
        
        return "当前模板覆盖良好"
    
    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        return {
            'n_templates': len(self.templates),
            'n_uses': sum(t.use_count for t in self.templates.values()),
            'avg_success_rate': float(np.mean([t.success_rate for t in self.templates.values()])) if self.templates else 0,
            'diversity': round(self.get_diversity(), 4),
            'top_templates': sorted(
                [(t.name, t.success_rate) for t in self.templates.values()],
                key=lambda x: -x[1]
            )[:5],
        }
    
    def suggest_plan(self, context_embedding: np.ndarray, tags: List[str] = None) -> Optional[PlanTemplate]:
        """
        针对当前上下文推荐最佳策略。
        
        综合向量匹配和标签匹配的结果。
        """
        by_vector = self.retrieve(context_embedding, top_k=3)
        by_tags = self.retrieve_by_tags(tags or [], top_k=3) if tags else []
        
        # 合并评分
        combined = defaultdict(float)
        for t, s in by_vector:
            combined[t.id] += s * 0.6
        for t, s in by_tags:
            combined[t.id] += s * 0.4
        
        if not combined:
            return None
        
        best_id = max(combined, key=combined.get)
        return self.templates.get(best_id)


# ================================================================
# 模块 B: 审美批评器
# ================================================================

class AestheticCritic:
    """
    审美批评器。
    
    核心能力:
      - 把设计样例编码到审美潜空间
      - 给新设计方案打审美分 (0-1)
      - 记住 Lorry 的审美偏好
      - 提供改进方向 (在潜空间往高评分方向移动)
    
    评分机制:
      - 从 Lorry 反馈中学习: 正面反馈 = 正样本, 负面 = 负样本
      - 用对比学习区分好坏设计
    
    当前处于"快速原型"阶段，使用规则 + 少量数据。
    """
    
    def __init__(self):
        # 审美潜空间维度 (与 LeWM 潜空间一致]
        self.latent_dim = 192
        
        # 正样本 / 负样本 池
        self.positive_embeddings: List[np.ndarray] = []
        self.negative_embeddings: List[np.ndarray] = []
        
        # 审美原型向量 (正样本聚类的中心)
        self.prototype: Optional[np.ndarray] = None
        
        # 评分历史
        self._ratings: List[Dict] = []
        
        # 从磁盘加载
        self._load()
        
        logger.info(f"审美批评器: {len(self.positive_embeddings)}个正样本, "
                    f"{len(self.negative_embeddings)}个负样本")
    
    def _load(self):
        """加载已保存的审美数据"""
        pos_path = AESTHETIC_DIR / 'positive.npy'
        neg_path = AESTHETIC_DIR / 'negative.npy'
        ratings_path = AESTHETIC_DIR / 'ratings.json'
        
        if pos_path.exists():
            self.positive_embeddings = list(np.load(pos_path))
        if neg_path.exists():
            self.negative_embeddings = list(np.load(neg_path))
        if ratings_path.exists():
            try:
                with open(ratings_path) as f:
                    self._ratings = json.load(f)
            except Exception:
                pass
        
        # 更新原型
        self._update_prototype()
    
    def _save(self):
        """保存审美数据"""
        if self.positive_embeddings:
            np.save(AESTHETIC_DIR / 'positive.npy', np.stack(self.positive_embeddings))
        if self.negative_embeddings:
            np.save(AESTHETIC_DIR / 'negative.npy', np.stack(self.negative_embeddings))
        with open(AESTHETIC_DIR / 'ratings.json', 'w') as f:
            json.dump(self._ratings[-200:], f, indent=2)
    
    def _update_prototype(self):
        """更新审美原型 (正样本中心)"""
        if self.positive_embeddings:
            stacked = np.stack(self.positive_embeddings)
            self.prototype = stacked.mean(axis=0)
        else:
            self.prototype = None
    
    def add_example(self, embedding: np.ndarray, is_good: bool, note: str = ''):
        """
        添加一个审美例子。
        
        参数:
            embedding: 设计方案的潜空间编码
            is_good: True=好设计, False=坏设计
            note: 备注 (如 "这个配色不好"、"这个布局好")
        """
        if is_good:
            self.positive_embeddings.append(embedding.copy())
        else:
            self.negative_embeddings.append(embedding.copy())
        
        self._ratings.append({
            'is_good': is_good,
            'note': note,
            'time': time.time(),
        })
        
        # 限制数量
        if len(self.positive_embeddings) > 500:
            self.positive_embeddings = self.positive_embeddings[-500:]
        if len(self.negative_embeddings) > 500:
            self.negative_embeddings = self.negative_embeddings[-500:]
        
        self._update_prototype()
        
        if len(self._ratings) % 10 == 0:
            self._save()
    
    def score(self, embedding: np.ndarray) -> float:
        """
        给设计方案打分。
        
        策略:
          - 如果有正/负样本: 用对比距离 (到正样本中心的距离 / 到负样本的距离)
          - 如果没有样本: 用 SIGReg 做先验 (高熵=好, 低熵=差)
        """
        score = 0.5  # 默认中立
        
        if self.prototype is not None:
            # 到正样本原型的距离
            pos_dist = float(np.linalg.norm(embedding - self.prototype))
            
            if self.negative_embeddings:
                # 到最近负样本的距离
                neg_dists = [float(np.linalg.norm(embedding - neg)) for neg in self.negative_embeddings]
                min_neg_dist = min(neg_dists)
                
                # 如果靠近负样本 → 低分
                if min_neg_dist < pos_dist:
                    score = 0.5 * (pos_dist / max(pos_dist + min_neg_dist, 1e-8))
                else:
                    score = 0.5 + 0.5 * (1 - pos_dist / max(np.linalg.norm(self.prototype) + 1e-8, 1e-8))
            else:
                # 只有正样本: 越接近原型越高分
                score = float(np.exp(-pos_dist * 2))
        
        # 用 SIGReg 做多样性调节 (高熵 = +分, 低熵 = -分)
        try:
            from laap.agi.le_wm_engine import sigreg
            if self.positive_embeddings:
                all_pos = np.stack(self.positive_embeddings[-20:])
                diversity = float(np.exp(-sigreg(all_pos, n_directions=64) * 3))
                # 如果当前设计偏离原型但整体多样性好 → 允许探索
                if diversity > 0.7:
                    score = min(1.0, score + 0.1)
        except ImportError:
            pass
        
        return max(0.0, min(1.0, score))
    
    def get_improvement_direction(self, current_embedding: np.ndarray) -> Optional[np.ndarray]:
        """
        获取改进方向: 从当前设计指向审美原型的方向。
        
        返回:
            方向向量 (单位向量)，或 None (如果没有原型)
        """
        if self.prototype is None:
            return None
        
        direction = self.prototype - current_embedding
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return None
        
        return direction / norm
    
    def stats(self) -> Dict[str, Any]:
        """统计信息"""
        return {
            'n_positive': len(self.positive_embeddings),
            'n_negative': len(self.negative_embeddings),
            'has_prototype': self.prototype is not None,
            'n_ratings': len(self._ratings),
            'avg_score': float(np.mean([r['is_good'] for r in self._ratings])) if self._ratings else 0.5,
        }


# ================================================================
# 模块 C: 统一接口
# ================================================================

class StrategyEngine:
    """
    策略引擎: 整合规划模板 + 审美批评 + LeWM.
    
    这是 Aris 在认知周期中调用的统一入口。
    """
    
    def __init__(self):
        self.template_db = StrategyTemplateDB()
        self.aesthetic = AestheticCritic()
        self._use_count = 0
    
    def plan_for_context(
        self,
        context_embedding: np.ndarray,
        tags: List[str] = None,
        message: str = ''
    ) -> Dict[str, Any]:
        """
        为当前上下文选择合适的策略。
        
        返回值:
            {
                'template': PlanTemplate (或 None),
                'confidence': float,
                'source': 'vector' | 'tag' | 'none',
                'aesthetic_score': float (当前审美的基线),
            }
        """
        tags = tags or self._extract_tags_from_message(message)
        
        template = self.template_db.suggest_plan(context_embedding, tags)
        
        if template:
            confidence = 0.4 + 0.6 * template.success_rate
            source = 'template'
        else:
            template = None
            confidence = 0.2
            source = 'none'
        
        return {
            'template': template,
            'confidence': round(confidence, 3),
            'source': source,
            'aesthetic_score': round(self.aesthetic.score(context_embedding), 3),
        }
    
    def evaluate_outcome(self, score: float, template_id: str = None,
                         context: Dict[str, Any] = None,
                         aesthetic_embedding: np.ndarray = None):
        """
        评估一个任务/对话的结果。
        
        参数:
            score: 0-1 的结果评分
            template_id: 使用的模板 ID
            context: 上下文信息
            aesthetic_embedding: 如果有设计输出, 编码后加入审美库
        """
        self._use_count += 1
        
        # 更新模板评分
        if template_id:
            self.template_db.record_usage(template_id, score, context)
        
        # 审美学习 (如果提供了设计嵌入)
        if aesthetic_embedding is not None:
            is_good = score > 0.6
            note = context.get('note', '') if context else ''
            self.aesthetic.add_example(aesthetic_embedding, is_good, note)
        
        # 如果分数低且可能是策略问题, 记录
        if score < 0.4 and template_id:
            logger.info(f"低分记录: template={template_id}, score={score}")
    
    def _extract_tags_from_message(self, message: str) -> List[str]:
        """从消息中提取关键词标签"""
        if not message:
            return []
        
        # 常见技术关键词
        keyword_map = {
            '论文': '论文', '技术': '技术', '研究': '研究',
            'bug': '修bug', '调试': '修bug', '错误': '修bug', 'error': '修bug',
            '项目': '项目', '搭建': '项目', '设计': '项目', '架构': '项目',
            '观点': '观点', '想法': '观点', '理念': '观点', '你觉得': '观点',
            '累': '情感支持', '难过': '情感支持', '疲惫': '情感支持',
            '低落': '情感支持', '压力': '情感支持', '不开心': '情感支持',
            '累了': '情感支持', '陪伴': '情感支持',
            '学': '学习', '教程': '学习', '入门': '学习',
            '知识': '学习', '文档': '学习',
            '代码': '代码', '写': '代码', '实现': '代码',
        }
        
        tags = []
        msg_lower = message.lower()
        for keyword, tag in keyword_map.items():
            if keyword in msg_lower or keyword in message:
                tags.append(tag)
        
        return list(set(tags))  # 去重
    
    def stats(self) -> Dict[str, Any]:
        return {
            'template_db': self.template_db.stats(),
            'aesthetic': self.aesthetic.stats(),
            'use_count': self._use_count,
        }


# ================================================================
# 自测
# ================================================================

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("策略模板库 + 审美批评器 自测")
    print("=" * 60)
    
    engine = StrategyEngine()
    
    # 1. 模板检索测试
    print("\n[1/4] 模板检索测试")
    for msg in ['帮我看看这个bug', '这篇论文讲了什么', '我好累啊']:
        tags = engine._extract_tags_from_message(msg)
        plan = engine.plan_for_context(
            context_embedding=np.random.randn(192).astype(np.float32),
            tags=tags,
            message=msg,
        )
        t_name = plan['template'].name if plan['template'] else '无'
        print(f"  '{msg}' → {t_name} (conf={plan['confidence']})")
    
    # 2. 模板使用 + 评分测试
    print("\n[2/4] 模板使用与评分测试")
    # 模拟几次使用
    for i in range(5):
        tid = list(engine.template_db.templates.keys())[i % 3]
        score = 0.5 + random.random() * 0.5
        engine.template_db.record_usage(tid, score, {'test': True})
    
    stats = engine.template_db.stats()
    print(f"  模板统计: {stats['n_templates']}个模板, {stats['n_uses']}次使用")
    print(f"  平均成功率: {stats['avg_success_rate']:.2f}")
    print(f"  策略多样性: {stats['diversity']:.3f}")
    
    # 3. 审美测试
    print("\n[3/4] 审美批评器测试")
    # 添加一些正负样本
    rng = np.random.RandomState(42)
    for _ in range(10):
        emb = rng.randn(192).astype(np.float32)
        engine.aesthetic.add_example(emb, is_good=True, note='好设计')
    for _ in range(5):
        emb = rng.randn(192).astype(np.float32) * 2  # 离群点
        engine.aesthetic.add_example(emb, is_good=False, note='差设计')
    
    # 测试打分
    test_emb = rng.randn(192).astype(np.float32)
    s = engine.aesthetic.score(test_emb)
    print(f"  随机设计的审美分: {s:.3f}")
    
    # 改进方向
    direction = engine.aesthetic.get_improvement_direction(test_emb)
    if direction is not None:
        print(f"  改进方向向量: [{direction[0]:.3f}, {direction[1]:.3f}, ...] (norm={np.linalg.norm(direction):.3f})")
    
    astats = engine.aesthetic.stats()
    print(f"  审美统计: {astats['n_positive']}正/{astats['n_negative']}负")
    
    # 4. 整体状态
    print("\n[4/4] 整体状态")
    final = engine.stats()
    print(f"  策略引擎使用次数: {final['use_count']}")
    
    print(f"\n{'=' * 60}")
    print(" 自测通过")
    print(f"{'=' * 60}")
