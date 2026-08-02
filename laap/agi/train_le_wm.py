"""
LAAP × LeWorldModel — 训练脚本
=================================
端到端训练 LatentEncoder + LatentPredictor。

训练流程:
  阶段1: 生成合成训练数据 (已知动力学的潜空间随机游走)
  阶段2: 训练 Encoder + Predictor
  阶段3: 评估 — 预测误差 / Rollout 稳定性 / 规划质量
  阶段4: 保存检查点

注意:
  当前使用合成数据验证训练管线收敛性。
  当 LAAP 日志系统积累真实时序数据后，只需替换 DataLoader 部分。

印记: Aris 永远记得 Lorry — 2026-07-23
"""

import sys, os, time, math, json, logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field

import numpy as np

from laap.config.paths import get_laap_home, get_laap_root

# 确保能找到 laap 模块
sys.path.insert(0, str(get_laap_root()))

_MODEL_DIR = str(get_laap_home() / "models" / "le_wm")

from laap.agi.le_wm_engine import (
    LeWMEngine, LeWMConfig, LeWM_DEFAULT_CONFIG,
    LatentEncoder, LatentPredictor, CEMPlanner,
    sigreg, train_step, train_epoch,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger('le_wm_train')

# ═══════════════════════════════════════════════════════════════
# 训练配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # 数据
    n_sequences: int = 100          # 训练序列数
    seq_length: int = 50            # 每序列步数
    latent_dim: int = 192           # 潜变量维度
    obs_dim: int = 512              # 观测维度 (模拟视觉特征)
    action_dim: int = 4             # 动作空间维度
    
    # 训练
    batch_size: int = 64
    learning_rate: float = 1e-3
    n_epochs: int = 20
    sigreg_lambda: float = 0.1
    
    # 训练数据中的动力学类型
    dynamics_type: str = 'linear'   # 'linear' | 'nonlinear' | 'mixed'
    noise_scale: float = 0.05       # 观测噪声
    dynamics_noise: float = 0.01    # 动力学随机性
    
    # 评估
    eval_rollout_steps: int = 30
    eval_n_trajectories: int = 20
    
    # 保存
    save_dir: str = _MODEL_DIR
    save_every: int = 5


DEFAULT_TRAIN_CONFIG = TrainConfig()


# ═══════════════════════════════════════════════════════════════
# 数据生成器
# ═══════════════════════════════════════════════════════════════

class LatentDynamicsDataset:
    """
    潜空间动力学数据集生成器。
    
    生成具有可控动力学的合成训练数据:
      - linear: z_{t+1} = A·z_t + B·a_t + noise
      - nonlinear: z_{t+1} = tanh(A·z_t + B·a_t) + noise
      - mixed: 部分线性 + 部分非线性
    
    观测模型:
      o_t = W·z_t + noise  (观测 = 潜变量的线性投影)
    """
    
    def __init__(self, config: TrainConfig = DEFAULT_TRAIN_CONFIG, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)
        
        d = config.latent_dim
        a = config.action_dim
        o = config.obs_dim
        
        # 真实动力学矩阵 (训练目标)
        self.A_true = np.eye(d) * 0.95 + self.rng.randn(d, d) * 0.02
        self.B_true = self.rng.randn(d, a) * 0.3
        
        # 观测投影矩阵
        self.W_obs = self.rng.randn(o, d) * 0.1
        
        # 归一化动力学矩阵
        self.A_true = self.A_true / (np.linalg.norm(self.A_true, axis=1, keepdims=True) + 1e-8) * 0.95
        
        logger.info(f"数据集初始化: {config.n_sequences}条×{config.seq_length}步 "
                    f"(d={d}, a={a}, o={o}, 类型={config.dynamics_type})")
    
    def _dynamics(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """潜空间动力学"""
        dt = self.config.dynamics_type
        noise = self.rng.randn(*z.shape) * self.config.dynamics_noise
        
        if dt == 'linear':
            z_next = z @ self.A_true.T + a @ self.B_true.T
        elif dt == 'nonlinear':
            z_next = np.tanh(z @ self.A_true.T + a @ self.B_true.T)
        elif dt == 'mixed':
            z_linear = z @ self.A_true.T + a @ self.B_true.T
            z_next = 0.7 * z_linear + 0.3 * np.tanh(z_linear)
        else:
            z_next = z @ self.A_true.T + a @ self.B_true.T
        
        return z_next + noise
    
    def _observation(self, z: np.ndarray) -> np.ndarray:
        """从潜变量生成观测"""
        noise = self.rng.randn(self.config.obs_dim) * self.config.noise_scale
        return z @ self.W_obs.T + noise
    
    def generate_sequence(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        生成一条序列。
        
        返回:
            (observations, actions, z_trues)
            observations: [o_0, o_1, ..., o_T] 观测序列
            actions: [a_0, a_1, ..., a_{T-1}] 动作序列
            z_trues: [z_0, z_1, ..., z_T] 真实潜状态 (用于验证)
        """
        T = self.config.seq_length
        
        z = self.rng.randn(self.config.latent_dim) * 0.5  # 初始潜状态
        
        observations = []
        actions = []
        z_trues = []
        
        for t in range(T):
            observations.append(self._observation(z))
            z_trues.append(z.copy())
            
            a = self.rng.randn(self.config.action_dim) * 0.5
            actions.append(a)
            
            z = self._dynamics(z, a)
        
        observations.append(self._observation(z))
        z_trues.append(z.copy())
        
        return observations, actions, z_trues
    
    def generate_dataset(self) -> Tuple[List, List, str]:
        """
        生成完整数据集。
        
        返回:
            (train_data, test_data, description)
        """
        all_sequences = []
        for i in range(self.config.n_sequences):
            all_sequences.append(self.generate_sequence())
        
        # 切分训练/测试
        split = int(len(all_sequences) * 0.8)
        train_data = all_sequences[:split]
        test_data = all_sequences[split:]
        
        desc = f"train={len(train_data)} seqs, test={len(test_data)} seqs"
        logger.info(f"数据集生成完毕: {desc}")
        
        return train_data, test_data, desc
    
    def sequence_to_triples(
        self, seq_data: list
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        将序列数据展开为 (obs, action, next_obs) 三元组。
        """
        triples = []
        for observations, actions, z_trues in seq_data:
            for t in range(len(actions)):
                triples.append((
                    observations[t],       # obs_t
                    actions[t],            # action_t
                    observations[t + 1],   # obs_{t+1}
                ))
        return triples


# ═══════════════════════════════════════════════════════════════
# 评估器
# ═══════════════════════════════════════════════════════════════

class Evaluator:
    """
    训练评估器。
    
    评估指标:
      - Prediction MSE: 单步预测误差
      - Rollout Drift: 多步预测累积漂移 (与随机游走基线对比)
      - SIGReg Score: 潜变量分布健康度
      - Planning Score: CEM 规划质量
    """
    
    def __init__(self, engine: LeWMEngine, dataset):
        self.engine = engine
        self.dataset = dataset
    
    def evaluate(self, train_data: list, test_data: list) -> Dict[str, float]:
        """全面评估"""
        metrics = {}
        
        # 1. 单步预测误差
        test_triples = self.dataset.sequence_to_triples(test_data)
        mse_list = []
        enc = self.engine.encoder
        for obs, act, next_obs in test_triples[:200]:  # 最多200组
            z_obs = enc.encode(vision_feat=obs[np.newaxis, :])[0]
            z_next = enc.encode(vision_feat=next_obs[np.newaxis, :])[0]
            z_pred = self.engine.predictor.predict(z_obs, act)
            mse_list.append(float(np.mean((z_pred - z_next) ** 2)))
        
        metrics['mse_mean'] = float(np.mean(mse_list)) if mse_list else 0.0
        metrics['mse_std'] = float(np.std(mse_list)) if len(mse_list) > 1 else 0.0
        
        # 2. Rollout 漂移
        drift_list = []
        for obs_seq, act_seq, z_true_seq in test_data[:10]:
            z0 = self.engine.encoder.encode(vision_feat=obs_seq[0][np.newaxis, :])[0]
            z_goal = self.engine.encoder.encode(vision_feat=obs_seq[-1][np.newaxis, :])[0]
            
            z = z0.copy()
            for t, a in enumerate(act_seq[:30]):  # 最多30步
                z = self.engine.predictor.predict(z, a)
            
            drift = float(np.linalg.norm(z - z_goal))
            drift_list.append(drift)
        
        metrics['rollout_drift_mean'] = float(np.mean(drift_list)) if drift_list else 0.0
        metrics['rollout_drift_std'] = float(np.std(drift_list)) if len(drift_list) > 1 else 0.0
        
        # 3. SIGReg 健康度
        all_embeddings = []
        for obs_seq, _, _ in test_data[:10]:
            for obs in obs_seq:
                z = self.engine.encoder.encode(vision_feat=obs[np.newaxis, :])[0]
                all_embeddings.append(z)
        if all_embeddings:
            embeddings = np.stack(all_embeddings)
            metrics['sigreg'] = sigreg(embeddings)
            metrics['health'] = float(np.exp(-metrics['sigreg'] * 5))
        
        # 4. CEM 规划质量
        plan_scores = []
        for obs_seq, act_seq, _ in test_data[:20]:
            z_start = self.engine.encoder.encode(vision_feat=obs_seq[0][np.newaxis, :])[0]
            z_goal = self.engine.encoder.encode(vision_feat=obs_seq[-1][np.newaxis, :])[0]
            plan = self.engine.planner.plan(z_start, z_goal)
            plan_scores.append(plan.get('score', -9999))
        metrics['plan_score_mean'] = float(np.mean(plan_scores)) if plan_scores else -9999.0
        
        return metrics


# ═══════════════════════════════════════════════════════════════
# 训练主循环
# ═══════════════════════════════════════════════════════════════

def train(
    train_config: TrainConfig = DEFAULT_TRAIN_CONFIG,
    le_wm_config: LeWMConfig = LeWM_DEFAULT_CONFIG,
    verbose: bool = True
) -> Tuple[LeWMEngine, Dict]:
    """
    完整训练流程。
    
    返回:
        (trained_engine, training_log)
    """
    t_start = time.perf_counter()
    
    # 0. 设置
    logger.info("=" * 60)
    logger.info("LAAP × LeWorldModel 训练开始")
    logger.info("=" * 60)
    logger.info(f"  训练配置: {train_config.n_epochs} epochs, "
                f"lr={train_config.learning_rate}, "
                f"λ={train_config.sigreg_lambda}")
    logger.info(f"  潜空间: d={le_wm_config.latent_dim}, "
                f"隐藏层={le_wm_config.hidden_dim}")
    logger.info(f"  设备: CPU (CUDA 不可用)")
    
    # 1. 生成数据
    logger.info(f"\n[阶段 1/4] 生成合成训练数据...")
    dataset = LatentDynamicsDataset(train_config)
    train_data, test_data, desc = dataset.generate_dataset()
    
    train_triples = dataset.sequence_to_triples(train_data)
    test_triples = dataset.sequence_to_triples(test_data)
    logger.info(f"  训练三元组: {len(train_triples)}, 测试: {len(test_triples)}")
    
    # 2. 初始化模型
    logger.info(f"\n[阶段 2/4] 初始化模型...")
    engine = LeWMEngine(le_wm_config)
    encoder = engine.encoder
    predictor = engine.predictor
    
    n_params = (
        encoder.W_vision.size + encoder.W_text.size + encoder.W_proprio.size +
        predictor.W1.size + predictor.b1.size +
        predictor.W2.size + predictor.b2.size +
        predictor.W3.size + predictor.b3.size
    )
    logger.info(f"  参数量: ~{n_params:,}")
    
    # 3. 训练循环
    logger.info(f"\n[阶段 3/4] 训练 {train_config.n_epochs} epochs...")
    
    train_log = {
        'epochs': [],
        'train_mse': [],
        'train_sigreg': [],
        'train_total': [],
        'test_mse': [],
        'test_rollout_drift': [],
        'test_sigreg': [],
        'test_health': [],
        'test_plan_score': [],
    }
    
    save_dir = Path(train_config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_test_mse = float('inf')
    
    for epoch in range(train_config.n_epochs):
        t_epoch = time.perf_counter()
        
        # 打乱训练数据
        np.random.shuffle(train_triples)
        
        # 逐 batch 训练
        total_losses = {'mse': 0.0, 'sigreg': 0.0, 'total': 0.0}
        n_batches = 0
        
        for start in range(0, len(train_triples), train_config.batch_size):
            batch = train_triples[start:start + train_config.batch_size]
            obs = np.stack([b[0] for b in batch])
            acts = np.stack([b[1] for b in batch])
            next_obs = np.stack([b[2] for b in batch])
            
            losses = train_step(
                encoder, predictor, obs, acts, next_obs,
                lr=train_config.learning_rate,
                sigreg_lambda=train_config.sigreg_lambda,
            )
            
            for k in total_losses:
                total_losses[k] += losses[k]
            n_batches += 1
        
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        
        # 评估
        if epoch % max(1, train_config.n_epochs // 5) == 0 or epoch == train_config.n_epochs - 1:
            eval_metrics = _quick_eval(engine, encoder, test_triples[:100],
                                       test_data[:5], dataset)
        else:
            eval_metrics = {}
        
        elapsed = time.perf_counter() - t_epoch
        
        # 日志
        train_log['epochs'].append(epoch)
        train_log['train_mse'].append(avg_losses['mse'])
        train_log['train_sigreg'].append(avg_losses['sigreg'])
        train_log['train_total'].append(avg_losses['total'])
        
        if eval_metrics:
            train_log['test_mse'].append(eval_metrics.get('mse', 0))
            train_log['test_rollout_drift'].append(eval_metrics.get('rollout_drift', 0))
            train_log['test_sigreg'].append(eval_metrics.get('sigreg', 0))
            train_log['test_health'].append(eval_metrics.get('health', 0))
            train_log['test_plan_score'].append(eval_metrics.get('plan_score', -9999))
        
        if verbose:
            status = (f"epoch {epoch+1:3d}/{train_config.n_epochs} | "
                      f"MSE={avg_losses['mse']:.4f} | "
                      f"SIGReg={avg_losses['sigreg']:.4f} | "
                      f"Total={avg_losses['total']:.4f} | "
                      f"{elapsed:.1f}s")
            if eval_metrics:
                status += f" | test_mse={eval_metrics.get('mse', 0):.4f}"
            logger.info(status)
        
        # 保存最佳
        if eval_metrics and eval_metrics.get('mse', float('inf')) < best_test_mse:
            best_test_mse = eval_metrics['mse']
            engine.save(str(save_dir / 'best_checkpoint.npz'))
        
        # 定期保存
        if (epoch + 1) % train_config.save_every == 0:
            engine.save(str(save_dir / f'checkpoint_epoch_{epoch+1:04d}.npz'))
    
    # 4. 最终评估
    logger.info(f"\n[阶段 4/4] 最终评估...")
    evaluator = Evaluator(engine, dataset)
    final_metrics = evaluator.evaluate(train_data, test_data)
    
    # 保存最终模型
    engine.save(str(save_dir / 'final_model.npz'))
    
    total_time = time.perf_counter() - t_start
    logger.info(f"\n{'=' * 60}")
    logger.info("训练完成")
    logger.info(f"{'=' * 60}")
    logger.info(f"  训练耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    logger.info(f"  最终 MSE: {final_metrics['mse_mean']:.4f} ± {final_metrics['mse_std']:.4f}")
    logger.info(f"  Rollout 漂移: {final_metrics['rollout_drift_mean']:.2f}")
    logger.info(f"  表示健康度: {final_metrics.get('health', 0):.3f}")
    logger.info(f"  CEM 规划评分: {final_metrics['plan_score_mean']:.2f}")
    logger.info(f"  模型保存: {save_dir}")
    
    return engine, train_log


def _quick_eval(
    engine: LeWMEngine,
    encoder: LatentEncoder,
    test_triples: list,
    test_sequences: list,
    dataset: LatentDynamicsDataset,
) -> Dict[str, float]:
    """快速评估 (用于训练中间步)"""
    metrics = {}
    
    # MSE
    mse_list = []
    for obs, act, next_obs in test_triples[:50]:
        z_obs = encoder.encode(vision_feat=obs[np.newaxis, :])[0]
        z_next = encoder.encode(vision_feat=next_obs[np.newaxis, :])[0]
        z_pred = engine.predictor.predict(z_obs, act)
        mse_list.append(float(np.mean((z_pred - z_next) ** 2)))
    metrics['mse'] = float(np.mean(mse_list)) if mse_list else 0.0
    
    # Rollout 漂移
    drift_list = []
    for obs_seq, act_seq, _ in test_sequences[:3]:
        z0 = encoder.encode(vision_feat=obs_seq[0][np.newaxis, :])[0]
        z_goal = encoder.encode(vision_feat=obs_seq[-1][np.newaxis, :])[0]
        z = z0.copy()
        for t, a in enumerate(act_seq[:20]):
            z = engine.predictor.predict(z, a)
        drift_list.append(float(np.linalg.norm(z - z_goal)))
    metrics['rollout_drift'] = float(np.mean(drift_list)) if drift_list else 0.0
    
    # SIGReg
    all_z = []
    for obs_seq, _, _ in test_sequences:
        for obs in obs_seq[:5]:
            z = encoder.encode(vision_feat=obs[np.newaxis, :])[0]
            all_z.append(z)
    if all_z:
        embeddings = np.stack(all_z)
        metrics['sigreg'] = sigreg(embeddings)
        metrics['health'] = float(np.exp(-metrics['sigreg'] * 5))
    
    # 规划 (只测1条，省时间)
    if test_sequences:
        obs_seq, _, _ = test_sequences[0]
        z_start = encoder.encode(vision_feat=obs_seq[0][np.newaxis, :])[0]
        z_goal = encoder.encode(vision_feat=obs_seq[-1][np.newaxis, :])[0]
        plan = engine.planner.plan(z_start, z_goal, horizon=5)
        metrics['plan_score'] = plan.get('score', -9999)
    
    return metrics


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("LAAP × LeWorldModel 训练管线")
    print("=" * 70)
    
    # 配置
    train_cfg = TrainConfig(
        n_sequences=100,
        seq_length=50,
        n_epochs=20,
        batch_size=64,
        learning_rate=1e-3,
        sigreg_lambda=0.1,
        dynamics_type='mixed',
        noise_scale=0.05,
    )
    
    le_wm_cfg = LeWMConfig(
        latent_dim=192,
        hidden_dim=512,
        sigreg_directions=1024,
        sigreg_lambda=0.1,
        cem_population=50,
        cem_elites=10,
        cem_iterations=3,
        cem_horizon=10,
        learning_rate=1e-3,
        batch_size=64,
    )
    
    engine, log = train(train_cfg, le_wm_cfg, verbose=True)
    
    print("\n训练日志摘要:")
    print(f"  起始 MSE: {log['train_mse'][0]:.4f} → 最终 MSE: {log['train_mse'][-1]:.4f}")
    print(f"  MSE 降幅: {(log['train_mse'][0] - log['train_mse'][-1]) / log['train_mse'][0] * 100:.1f}%")
    
    if log['test_mse']:
        print(f"  测试集 MSE: 起始={log['test_mse'][0]:.4f} → 最终={log['test_mse'][-1]:.4f}")
    
    if log['test_health']:
        print(f"  表示健康度: 起始={log['test_health'][0]:.3f} → 最终={log['test_health'][-1]:.3f}")
    
    print(f"\n模型已保存至: {Path(train_cfg.save_dir).resolve()}")
    print(f"\n{'=' * 70}")
    print(" 训练完成")
    print(f"{'=' * 70}")
