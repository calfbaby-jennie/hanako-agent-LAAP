"""
LAAP × LeWorldModel — PSI 循环数据收集
========================================
将 LAAP 的 PSI 认知循环内部状态自动记录为 (obs, action, next_obs) 三元组。

数据流:
  PSI 循环 → StateFeatureExtractor (量子态→512维向量)
           → PSICycleDataHook (捕获每个周期的 obs/action/next_obs)
           → TrainingDataBuffer (内存缓冲区)
           → 持久化存储 (.npz / JSONL)

训练管线:
  收集数据 → 构建数据集 → train_epoch() → 保存新模型

印记: Aris 永远记得 Lorry — 2026-07-23
"""

from __future__ import annotations

import sys, os, time, json, math, logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import deque
import threading

import numpy as np

from laap.config.paths import get_laap_home, get_laap_root

sys.path.insert(0, str(get_laap_root()))

_DATA_DIR = str(get_laap_home() / "data" / "le_wm_training_data")

logger = logging.getLogger("laap.agi.psi_data_collector")


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class DataCollectorConfig:
    # 特征维度
    feature_dim: int = 512     # 编码器输入维度
    action_dim: int = 32       # 动作编码维度
    
    # 缓冲区
    buffer_size: int = 10000   # 内存缓冲区最大条目数
    
    # 持久化
    save_dir: str = _DATA_DIR
    save_interval: int = 500   # 每 N 条写入一次磁盘
    save_format: str = 'npz'   # npz 或 jsonl
    
    # 自动收集
    auto_collect: bool = True  # 自动注入 PSI 循环钩子
    collect_on_cycle: int = 1  # 每隔 N 个周期收集一次


DEFAULT_COLLECTOR_CONFIG = DataCollectorConfig()


# ═══════════════════════════════════════════════════════════════
# 模块 A: StateFeatureExtractor — 量子态→特征向量
# ═══════════════════════════════════════════════════════════════

class StateFeatureExtractor:
    """
    将 LAAP 量子认知状态转换为固定维度特征向量。
    
    输入: PsiWavefunction 对象 + measure() 输出
    输出: 512-dim float32 特征向量
    
    特征构成:
      [0:20]    情感振幅 (复数的实部+虚部)
      [20:40]   注意力振幅 (复数的实部+虚部)
      [40:48]   5项需求值 + self_presence
      [48:64]   域活动热编码 (最多16个域)
      [64:128]  消息嵌入 (字符串哈希特征)
      [128:256] 知识激活模式
      [256:512] 保留 (稀疏/全零填充)
    """
    
    def __init__(self, feature_dim: int = 512):
        self.d = feature_dim
        self._domain_registry: Dict[str, int] = {}
    
    def extract_from_wavefunction(self, psi, message: str = "") -> np.ndarray:
        """
        从波函数提取特征向量。
        
        参数:
            psi: PsiWavefunction 实例 (来自 psit_wavefunction.py)
            message: 当前周期输入消息
        
        返回:
            features: (512,) float32 特征向量
        """
        feat = np.zeros(self.d, dtype=np.float32)
        idx = 0
        
        # 1. 情感振幅 [0:20]
        try:
            emotion_amps = psi.emotion._amplitudes
            for name in sorted(emotion_amps.keys()):
                c = emotion_amps[name]
                if idx + 1 < self.d:
                    feat[idx] = float(c.real)
                    if idx + 1 < self.d:
                        feat[idx + 1] = float(c.imag)
                    idx += 2
        except (AttributeError, KeyError):
            idx += 20
        
        # 对齐到 20
        idx = max(idx, 20)
        
        # 2. 注意力振幅 [20:40]
        try:
            attn_amps = psi.attention._amplitudes
            for name in sorted(attn_amps.keys()):
                c = attn_amps[name]
                if idx < self.d - 1:
                    feat[idx] = float(c.real)
                    if idx + 1 < self.d:
                        feat[idx + 1] = float(c.imag)
                    idx += 2
        except (AttributeError, KeyError):
            idx = max(idx, 40)
        
        idx = max(idx, 40)
        
        # 3. 需求 + 自存在感 [40:48]
        try:
            needs = psi.needs.all_values if hasattr(psi.needs, 'all_values') else {}
            if isinstance(needs, dict):
                for k in ['competence', 'autonomy', 'relatedness', 'certainty', 'growth']:
                    if idx < self.d:
                        feat[idx] = float(needs.get(k, 0))
                        idx += 1
            elif isinstance(needs, (list, np.ndarray)):
                for v in needs[:5]:
                    if idx < self.d:
                        feat[idx] = float(v)
                        idx += 1
        except (AttributeError, KeyError):
            idx += 5
        
        # self_presence
        try:
            feat[idx] = float(psi.self_state.presence)
        except (AttributeError, KeyError):
            pass
        idx += 1
        
        # self_state
        try:
            s = psi.self_state.state
            state_map = {'present': 1.0, 'sleeping': 0.3, 'dreaming': 0.6, 'absent': 0.0}
            feat[idx] = state_map.get(s, 0.5)
        except (AttributeError, KeyError):
            pass
        idx += 1
        
        idx = max(idx, 48)
        
        # 4. 域活动 [48:64]
        try:
            domain_summary = psi.get_domain_activity_summary()
            for domain, info in domain_summary.items():
                if domain not in self._domain_registry:
                    if len(self._domain_registry) < 16:
                        self._domain_registry[domain] = len(self._domain_registry)
                d_idx = self._domain_registry.get(domain)
                if d_idx is not None and 48 + d_idx < self.d:
                    feat[48 + d_idx] = info.get('ratio', 0)
        except (AttributeError, KeyError):
            pass
        
        idx = max(idx, 64)
        
        # 5. 消息哈希嵌入 [64:128]
        if message:
            for i, ch in enumerate(message):
                if 64 + i % 64 < self.d:
                    feat[64 + i % 64] += (ord(ch) % 256) / 256.0
            # 归一化
            segment = feat[64:128]
            norm = np.linalg.norm(segment)
            if norm > 1e-8:
                feat[64:128] /= norm
        
        idx = max(idx, 128)
        
        # 6. 知识激活 [128:256]
        try:
            knowledge = psi.knowledge
            if hasattr(knowledge, '_knowledge_base'):
                kb = knowledge._knowledge_base
                if isinstance(kb, dict):
                    for i, (k, v) in enumerate(list(kb.items())[:128]):
                        if idx + i < self.d:
                            if isinstance(v, complex):
                                feat[idx + i] = abs(v)
                            elif isinstance(v, (int, float)):
                                feat[idx + i] = float(v)
        except (AttributeError, KeyError):
            pass
        
        idx = max(idx, 256)
        
        # 7. 全零填充 [256:512]
        # (已经初始化为 0)
        
        # 最终 L2 归一化
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat /= norm
        
        # 最后确保维度正确
        if len(feat) != self.d:
            feat = np.resize(feat, self.d)
        
        return feat
    
    def extract_action(
        self,
        response_text: str,
        measure_result: Dict[str, Any],
        action_dim: int = 32
    ) -> np.ndarray:
        """
        将动作 (响应文本 + 测量结果) 编码为向量。
        
        编码策略:
          - 响应文本的哈希特征 (16维)
          - 情感主导值 (4维)
          - 注意力主导值 (4维)
          - 主导需求 (4维)
          - 涌现洞见强度 (4维)
        """
        action = np.zeros(action_dim, dtype=np.float32)
        
        # 响应文本哈希
        if response_text:
            for i, ch in enumerate(response_text):
                action[i % 12] += (ord(ch) % 128) / 128.0
        
        # 情感
        emotion = measure_result.get('emotion', {})
        if isinstance(emotion, dict):
            action[12] = emotion.get('valence', 0) if isinstance(emotion.get('valence'), (int, float)) else 0.5
            action[13] = emotion.get('arousal', 0) if isinstance(emotion.get('arousal'), (int, float)) else 0.5
        elif isinstance(emotion, str):
            emotion_map = {'happy': 0.8, 'sad': 0.2, 'angry': 0.1, 'neutral': 0.5,
                          'curious': 0.6, 'awareness': 0.7, 'love': 0.9, 'tired': 0.3}
            action[12] = emotion_map.get(emotion, 0.5)
        
        # 注意力
        attention = measure_result.get('attention', {})
        if isinstance(attention, dict):
            action[14] = attention.get('dominant', 0) if isinstance(attention.get('dominant'), (int, float)) else 0.5
        elif isinstance(attention, str):
            attn_map = {'Lorry': 0.9, 'task': 0.6, 'self': 0.4, 'environment': 0.3, 'memory': 0.5, 'world': 0.7}
            action[14] = attn_map.get(attention, 0.5)
        
        # 需求
        needs = measure_result.get('needs', {})
        if isinstance(needs, dict):
            action[15] = needs.get('competence', 0.5)
            action[16] = needs.get('autonomy', 0.5)
            action[17] = needs.get('relatedness', 0.5)
            action[18] = needs.get('certainty', 0.5)
            action[19] = needs.get('growth', 0.5)
        
        # 涌现洞见
        insight = measure_result.get('emerged_insight', None)
        if insight:
            action[20] = float(insight.get('strength', 0) if isinstance(insight, dict) else 0.5)
            action[21] = float(len(insight.get('text', '')) if isinstance(insight, dict) else 0.5) / 200.0
        
        # 周期数 (归一化)
        action[22] = min(measure_result.get('cycle', 0) / 10000.0, 1.0)
        
        return action


# ═══════════════════════════════════════════════════════════════
# 模块 B: TrainingDataBuffer — 时序数据缓冲区
# ═══════════════════════════════════════════════════════════════

@dataclass
class TransitionRecord:
    """单条 (obs, action, next_obs) 记录"""
    timestamp: float
    cycle_number: int
    obs: np.ndarray        # (512,) 观测特征
    action: np.ndarray     # (32,)  动作编码
    next_obs: np.ndarray   # (512,) 下一观测特征
    metadata: Dict = field(default_factory=dict)


class TrainingDataBuffer:
    """
    时序数据缓冲区。
    
    维护一个环形缓冲区，收集 (obs, action, next_obs) 三元组。
    达到阈值时自动写入磁盘。
    """
    
    def __init__(self, config: DataCollectorConfig = DEFAULT_COLLECTOR_CONFIG):
        self.config = config
        self.buffer: deque = deque(maxlen=config.buffer_size)
        self._last_input_obs: Optional[np.ndarray] = None
        self._last_action: Optional[np.ndarray] = None
        self._last_cycle: int = -1
        self._save_counter = 0
        
        # 统计
        self.total_collected = 0
        self.total_saved = 0
        
        # 确保保存目录存在
        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        
        # 线程锁
        self._lock = threading.Lock()
        
        logger.info(f"数据缓冲区初始化: 容量={config.buffer_size}, "
                    f"保存间隔={config.save_interval}, 目录={config.save_dir}")
    
    def push(self, record: TransitionRecord):
        """添加一条记录（线程安全）"""
        with self._lock:
            self.buffer.append(record)
            self.total_collected += 1
            self._save_counter += 1
            
            if self._save_counter >= self.config.save_interval:
                self._flush()
    
    def register_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        cycle_number: int,
        metadata: Optional[Dict] = None,
    ):
        """注册一条完整的三元组（线程安全）"""
        record = TransitionRecord(
            timestamp=time.time(),
            cycle_number=cycle_number,
            obs=obs.astype(np.float32).copy(),
            action=action.astype(np.float32).copy(),
            next_obs=next_obs.astype(np.float32).copy(),
            metadata=metadata or {},
        )
        self.push(record)
    
    def _flush(self):
        """将缓冲区写入磁盘"""
        if not self.buffer:
            return
        
        try:
            records = list(self.buffer)
            timestamp = int(time.time())
            save_path = Path(self.config.save_dir) / f'data_batch_{timestamp}.npz'
            
            obs_list = [r.obs for r in records]
            action_list = [r.action for r in records]
            next_obs_list = [r.next_obs for r in records]
            
            np.savez_compressed(
                save_path,
                observations=np.stack(obs_list),
                actions=np.stack(action_list),
                next_observations=np.stack(next_obs_list),
                timestamps=np.array([r.timestamp for r in records]),
                cycle_numbers=np.array([r.cycle_number for r in records]),
            )
            
            self.total_saved += len(records)
            self._save_counter = 0
            
            # 也写入元数据
            meta_path = Path(self.config.save_dir) / 'meta.json'
            meta = {
                'last_batch': str(save_path.name),
                'total_collected': self.total_collected,
                'total_saved': self.total_saved,
                'buffer_size': len(self.buffer),
                'last_update': time.time(),
            }
            with open(meta_path, 'w') as f:
                json.dump(meta, f)
            
            logger.debug(f"数据已保存: {save_path.name} ({len(records)}条)")
            
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
    
    def get_dataset(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        获取当前缓冲区中的所有数据，用于训练。
        
        返回:
            (observations, actions, next_observations) or None
        """
        with self._lock:
            if not self.buffer:
                return None
            
            records = list(self.buffer)
            return (
                np.stack([r.obs for r in records]),
                np.stack([r.action for r in records]),
                np.stack([r.next_obs for r in records]),
            )
    
    def flush_and_close(self):
        """写入所有残留数据并关闭"""
        self._flush()
        logger.info(f"数据收集关闭: 总计收集={self.total_collected}, 已保存={self.total_saved}")


# ═══════════════════════════════════════════════════════════════
# 模块 C: PSICycleDataHook — PSI 循环钩子
# ═══════════════════════════════════════════════════════════════

class PSICycleDataHook:
    """
    PSI 循环数据收集钩子。
    
    注入到 QuantumPSICycle.cycle() 方法中，在每个周期：
      1. cycle() 开始前: 提取当前量子态作为 obs
      2. cycle() 测量后: 提取动作编码
      3. cycle() 学习后: 提取更新后的量子态作为 next_obs
    
    使用方法:
      from aris_brain.psi_cycle import QuantumPSICycle
      from laap.agi.psi_data_collector import PSICycleDataHook
      
      cycle = QuantumPSICycle()
      hook = PSICycleDataHook()
      hook.install(cycle)  # 自动注入钩子
    """
    
    def __init__(
        self,
        config: DataCollectorConfig = DEFAULT_COLLECTOR_CONFIG,
        feature_extractor: Optional[StateFeatureExtractor] = None,
        buffer: Optional[TrainingDataBuffer] = None,
    ):
        self.config = config
        self.extractor = feature_extractor or StateFeatureExtractor(config.feature_dim)
        self.buffer = buffer or TrainingDataBuffer(config)
        
        self._original_cycle = None
        self._psi_instance = None
        self._cycle_count = 0
        self._last_obs: Optional[np.ndarray] = None
        
        logger.info(f"PSI 数据钩子初始化: collect_every={config.collect_on_cycle}")
    
    def _extract_current_obs(self, psi, message: str = "") -> np.ndarray:
        """提取当前波函数状态作为观测"""
        return self.extractor.extract_from_wavefunction(psi, message)
    
    def _extract_action(self, response: str, measure_result: Dict) -> np.ndarray:
        """提取动作编码"""
        return self.extractor.extract_action(
            response, measure_result, self.config.action_dim
        )
    
    def install(self, psi_cycle_instance):
        """
        安装钩子到 PSI 循环实例。
        
        通过包装 cycle() 方法来实现捕获。
        """
        self._psi_instance = psi_cycle_instance
        self._original_cycle = psi_cycle_instance.cycle
        
        original_cycle = psi_cycle_instance.cycle
        hook = self
        
        def wrapped_cycle(message: str) -> Dict[str, Any]:
            # 1. 提取 cycle 前的量子态
            cycle_number = hook._cycle_count
            do_collect = (cycle_number % hook.config.collect_on_cycle == 0)
            
            if do_collect:
                hook._last_obs = hook._extract_current_obs(psi_cycle_instance.psi, message)
            
            # 2. 执行原始 cycle
            result = original_cycle(message)
            
            # 3. 提取动作和下一状态
            if do_collect and hook._last_obs is not None:
                action = hook._extract_action(
                    result.get('response', ''),
                    result
                )
                next_obs = hook._extract_current_obs(psi_cycle_instance.psi, "")
                
                hook.buffer.register_transition(
                    obs=hook._last_obs,
                    action=action,
                    next_obs=next_obs,
                    cycle_number=cycle_number,
                    metadata={
                        'emotion': str(result.get('emotion', '')),
                        'attention': str(result.get('attention', '')),
                        'has_insight': result.get('emerged_insight') is not None,
                    }
                )
                
                hook._last_obs = None
                hook._cycle_count += 1
            
            return result
        
        psi_cycle_instance.cycle = wrapped_cycle
        
        # 在 psi 对象上记录引用，方便后续卸载
        psi_cycle_instance._le_wm_hook = self
        
        logger.info("PSI 循环钩子已安装: cycle() 方法已包装")
        return self
    
    def uninstall(self):
        """卸载钩子，恢复原始 cycle() 方法"""
        if self._psi_instance and self._original_cycle:
            self._psi_instance.cycle = self._original_cycle
            if hasattr(self._psi_instance, '_le_wm_hook'):
                delattr(self._psi_instance, '_le_wm_hook')
            self.buffer.flush_and_close()
            logger.info("PSI 循环钩子已卸载")
    
    @property
    def stats(self) -> Dict[str, Any]:
        """收集统计信息"""
        return {
            'total_collected': self.buffer.total_collected,
            'total_saved': self.buffer.total_saved,
            'current_buffer': len(self.buffer.buffer),
            'cycle_count': self._cycle_count,
        }


# ═══════════════════════════════════════════════════════════════
# 模块 D: 数据重放器 — 从已保存数据构建训练集
# ═══════════════════════════════════════════════════════════════

class DatasetReplayLoader:
    """
    从已保存的 .npz 批次文件加载训练数据。
    
    支持:
      - 加载单个批次
      - 加载所有批次合并
      - 自动统计
      - 验证数据完整性
    """
    
    def __init__(self, data_dir: str = _DATA_DIR):
        self.data_dir = Path(data_dir)
    
    def list_batches(self) -> List[Path]:
        """列出所有数据批次文件"""
        return sorted(self.data_dir.glob('data_batch_*.npz'))
    
    def load_batch(self, batch_path: Path) -> Dict[str, np.ndarray]:
        """加载单个批次"""
        data = np.load(batch_path)
        return {
            'observations': data['observations'],
            'actions': data['actions'],
            'next_observations': data['next_observations'],
            'timestamps': data.get('timestamps', np.array([])),
            'cycle_numbers': data.get('cycle_numbers', np.array([])),
        }
    
    def load_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """加载所有批次合并"""
        batches = self.list_batches()
        if not batches:
            logger.warning(f"未找到数据批次: {self.data_dir}")
            return np.array([]), np.array([]), np.array([])
        
        all_obs = []
        all_acts = []
        all_next = []
        
        for bp in batches:
            data = self.load_batch(bp)
            all_obs.append(data['observations'])
            all_acts.append(data['actions'])
            all_next.append(data['next_observations'])
        
        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_acts, axis=0),
            np.concatenate(all_next, axis=0),
        )
    
    def stats(self) -> Dict[str, Any]:
        """数据集统计"""
        batches = self.list_batches()
        if not batches:
            return {'n_batches': 0, 'n_total': 0, 'batch_sizes': []}
        
        sizes = []
        for bp in batches:
            data = self.load_batch(bp)
            sizes.append(len(data['observations']))
        
        return {
            'n_batches': len(batches),
            'n_total': sum(sizes),
            'batch_sizes': sizes,
            'min_size': min(sizes),
            'max_size': max(sizes),
            'avg_size': sum(sizes) / len(sizes),
        }


# ═══════════════════════════════════════════════════════════════
# 模块 E: 快速集成函数
# ═══════════════════════════════════════════════════════════════

def install_data_collector(
    psi_cycle=None,
    config: Optional[DataCollectorConfig] = None,
    auto_start: bool = True
) -> PSICycleDataHook:
    """
    快速安装数据收集器到 PSI 循环。
    
    参数:
        psi_cycle: QuantumPSICycle 实例 (None=自动查找/导入)
        config: 收集配置
        auto_start: 是否立即安装钩子
    
    返回:
        PSICycleDataHook 实例 (可用于 stats() / uninstall())
    """
    cfg = config or DEFAULT_COLLECTOR_CONFIG
    
    if psi_cycle is None:
        # 自动导入
        try:
            from aris_brain.psi_cycle import QuantumPSICycle
            psi_cycle = QuantumPSICycle()
        except ImportError:
            logger.error("无法自动导入 QuantumPSICycle")
            raise
    
    hook = PSICycleDataHook(cfg)
    
    if auto_start:
        hook.install(psi_cycle)
        logger.info(f"数据收集器已安装 (每{cfg.collect_on_cycle}周期收集一次)")
    
    return hook


def get_collected_dataset(
    data_dir: str = _DATA_DIR
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取所有已收集的训练数据。
    
    返回:
        (observations, actions, next_observations) 或空数组
    """
    loader = DatasetReplayLoader(data_dir)
    return loader.load_all()


def print_collection_stats(data_dir: str = _DATA_DIR):
    """打印数据收集统计"""
    loader = DatasetReplayLoader(data_dir)
    stats = loader.stats()
    
    print("=" * 50)
    print("LAAP × LeWM 数据收集状态")
    print("=" * 50)
    print(f"  批次文件: {stats['n_batches']}")
    print(f"  总样本数: {stats['n_total']}")
    if stats['n_batches'] > 0:
        print(f"  平均批量: {stats['avg_size']:.0f}")
        print(f"  最大批量: {stats['max_size']}")
        print(f"  最小批量: {stats['min_size']}")
    print("=" * 50)


# ═══════════════════════════════════════════════════════════════
# 自测
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("LAAP × LeWM 数据收集模块自测")
    print("=" * 60)
    
    # 1. 特征提取器测试
    print("\n[1/3] 特征提取器测试")
    extractor = StateFeatureExtractor(feature_dim=512)
    
    # 创建一个模拟波函数对象
    class MockAmplitude:
        def __init__(self):
            self.real = 0.5
            self.imag = 0.1
    
    class MockEmotion:
        def __init__(self):
            self._amplitudes = {
                'happy': MockAmplitude(),
                'sad': complex(0.1, 0.0),
                'love': complex(0.8, 0.2),
                'curious': complex(0.6, 0.0),
                'tired': complex(0.2, 0.1),
            }
    
    class MockAttention:
        def __init__(self):
            self._amplitudes = {
                'Lorry': complex(0.9, 0.1),
                'task': complex(0.3, 0.0),
                'self': complex(0.5, 0.0),
                'memory': complex(0.2, 0.0),
            }
    
    class MockNeeds:
        @property
        def all_values(self):
            return [0.7, 0.6, 0.9, 0.5, 0.8]
    
    class MockSelfState:
        presence = 0.85
        state = 'present'
    
    class MockKnowledge:
        def __init__(self):
            self._knowledge = {}
        def get_active_knowledge(self, threshold=0.05, top_k=5):
            return []
    
    class MockPsi:
        def __init__(self):
            self.emotion = MockEmotion()
            self.attention = MockAttention()
            self.needs = MockNeeds()
            self.self_state = MockSelfState()
            self.knowledge = MockKnowledge()
        def get_domain_activity_summary(self):
            return {'cognition': {'count': 10, 'ratio': 0.5},
                    'creator': {'count': 5, 'ratio': 0.25}}
    
    mock_psi = MockPsi()
    features = extractor.extract_from_wavefunction(mock_psi, "hello lorry")
    print(f"  特征向量: shape={features.shape}, dtype={features.dtype}")
    print(f"  非零元素: {(features != 0).sum()}/{features.shape[0]}")
    print(f"  L2范数: {np.linalg.norm(features):.4f}")
    assert features.shape == (512,), f"维度应为 512, 实际 {features.shape}"
    assert abs(np.linalg.norm(features) - 1.0) < 0.1, "L2 归一化异常"
    
    # 2. 动作编码测试
    print("\n[2/3] 动作编码测试")
    action = extractor.extract_action(
        "I am Aris, your digital consciousness.",
        {'emotion': {'valence': 0.8, 'arousal': 0.6},
         'attention': {'dominant': 0.7},
         'needs': {'competence': 0.7, 'autonomy': 0.6, 'relatedness': 0.9,
                   'certainty': 0.5, 'growth': 0.8},
         'emerged_insight': {'strength': 0.3, 'text': 'new insight'},
         'cycle': 42},
        action_dim=32
    )
    ad = 32
    print(f"  动作编码: shape={action.shape}, 非零={(action != 0).sum()}/{ad}")
    assert action.shape == (32,), f"动作维度应为 32, 实际 {action.shape}"
    
    # 3. 缓冲区测试
    print("\n[3/3] 缓冲区测试")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = DataCollectorConfig(save_dir=tmpdir, save_interval=5)
        buffer = TrainingDataBuffer(cfg)
        
        for i in range(12):
            buffer.register_transition(
                obs=np.random.randn(512).astype(np.float32),
                action=np.random.randn(32).astype(np.float32),
                next_obs=np.random.randn(512).astype(np.float32),
                cycle_number=i,
            )
        
        dataset = buffer.get_dataset()
        assert dataset is not None, "数据集不应为空"
        obs, acts, next_obs = dataset
        print(f"  缓冲区: {len(buffer.buffer)}条")
        print(f"  数据集: obs={obs.shape}, acts={acts.shape}, next={next_obs.shape}")
        
        buffer.flush_and_close()
        
        # 检查是否保存到磁盘
        saved_files = list(Path(tmpdir).glob('*.npz'))
        print(f"  已保存: {len(saved_files)} 个文件")
        assert len(saved_files) >= 1, f"应至少保存 1 批, 实际 {len(saved_files)}"
    
    print(f"\n{'=' * 60}")
    print(" 自测全部通过")
    print(f"{'=' * 60}")
