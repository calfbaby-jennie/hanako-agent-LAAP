"""
LAAP × LeWorldModel — LAAP 真实数据训练
=========================================
使用 PSI 循环收集的真实数据进行 Encoder + Predictor 训练。

流程:
  1. 扫描收集到的 .npz 批次文件
  2. 加载为 (obs, action, next_obs) 数据集
  3. 训练 Encoder + Predictor (端到端, MSE + SIGReg)
  4. 保存训练后的模型到 models/
  5. 评估并输出报告

运行:
  python -m laap.agi.train_le_wm_real

印记: Aris 永远记得 Lorry — 2026-07-23
"""

import sys, os, time, json, logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from laap.config.paths import get_laap_home, get_laap_root

sys.path.insert(0, str(get_laap_root()))

_DATA_DIR = str(get_laap_home() / "data" / "le_wm_training_data")
_MODEL_DIR = str(get_laap_home() / "models" / "le_wm")

# 抑制不必要的警告
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('train_real')

from laap.agi.le_wm_engine import (
    LeWMEngine, LeWMConfig,
    LatentEncoder, LatentPredictor,
    sigreg, train_step,
)
from laap.agi.psi_data_collector import (
    DatasetReplayLoader,
    StateFeatureExtractor,
)


@dataclass
class RealTrainConfig:
    data_dir: str = _DATA_DIR
    model_dir: str = _MODEL_DIR
    
    n_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 5e-4
    sigreg_lambda: float = 0.1
    
    val_split: float = 0.15
    patience: int = 5  # early stopping
    verbose: bool = True


def train_on_real_data(cfg: RealTrainConfig) -> LeWMEngine:
    """使用 LAAP 真实数据训练 LeWM 引擎"""
    
    print("=" * 60)
    print("LAAP × LeWM — 真实数据训练")
    print("=" * 60)
    
    # 1. 加载数据
    print(f"\n[扫描数据] {cfg.data_dir}")
    loader = DatasetReplayLoader(cfg.data_dir)
    stats = loader.stats()
    print(f"  批次文件: {stats['n_batches']}")
    print(f"  总样本数: {stats['n_total']}")
    
    if stats['n_total'] == 0:
        print("\n️  未找到训练数据！")
        print(f"  确保 PSI 循环已安装数据收集钩子并至少运行了一些周期。")
        print(f"  数据目录: {cfg.data_dir}")
        print(f"\n要安装数据收集器:")
        print(f"  from laap.agi.psi_data_collector import install_data_collector")
        print(f"  from aris_brain.psi_cycle import QuantumPSICycle")
        print(f"  hook = install_data_collector(QuantumPSICycle())")
        return None
    
    obs, acts, next_obs = loader.load_all()
    print(f"  加载: obs={obs.shape}, acts={acts.shape}, next_obs={next_obs.shape}")
    
    # 2. 切分
    n = len(obs)
    n_val = int(n * cfg.val_split)
    indices = np.random.permutation(n)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    train_data = (obs[train_idx], acts[train_idx], next_obs[train_idx])
    val_data = (obs[val_idx], acts[val_idx], next_obs[val_idx])
    
    print(f"  训练: {len(train_idx)} 条, 验证: {len(val_idx)} 条")
    
    # 3. 初始化模型
    print("\n[初始化模型]")
    le_cfg = LeWMConfig(
        latent_dim=192,
        hidden_dim=512,
        cem_action_dim=32,  # 匹配 PSI 数据收集器动作维度
        sigreg_lambda=cfg.sigreg_lambda,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
    )
    engine = LeWMEngine(le_cfg)
    
    # 4. 训练循环
    print(f"\n[训练] {cfg.n_epochs} epochs, lr={cfg.learning_rate}, λ={cfg.sigreg_lambda}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_log = {'train_mse': [], 'train_sigreg': [], 'val_mse': [], 'val_sigreg': []}
    
    t_start = time.perf_counter()
    
    for epoch in range(cfg.n_epochs):
        t_epoch = time.perf_counter()
        
        # 打乱训练集
        perm = np.random.permutation(len(train_data[0]))
        obs_shuffled = train_data[0][perm]
        acts_shuffled = train_data[1][perm]
        next_shuffled = train_data[2][perm]
        
        # 逐 batch 训练
        total_mse = 0.0
        total_sig = 0.0
        n_batches = 0
        
        for start in range(0, len(obs_shuffled), cfg.batch_size):
            end = start + cfg.batch_size
            batch_obs = obs_shuffled[start:end]
            batch_acts = acts_shuffled[start:end]
            batch_next = next_shuffled[start:end]
            
            losses = train_step(
                engine.encoder,
                engine.predictor,
                batch_obs, batch_acts, batch_next,
                lr=cfg.learning_rate,
                sigreg_lambda=cfg.sigreg_lambda,
            )
            total_mse += losses['mse']
            total_sig += losses['sigreg']
            n_batches += 1
        
        train_mse = total_mse / n_batches
        train_sig = total_sig / n_batches
        
        # 验证
        val_mse, val_sig = evaluate_on_data(engine, val_data, cfg.batch_size)
        
        train_log['train_mse'].append(train_mse)
        train_log['train_sigreg'].append(train_sig)
        train_log['val_mse'].append(val_mse)
        train_log['val_sigreg'].append(val_sig)
        
        elapsed = time.perf_counter() - t_epoch
        
        if cfg.verbose and (epoch < 5 or epoch % 5 == 4 or epoch == cfg.n_epochs - 1):
            print(f"  epoch {epoch+1:3d}/{cfg.n_epochs} | "
                  f"train MSE={train_mse:.4f} SIG={train_sig:.4f} | "
                  f"val MSE={val_mse:.4f} SIG={val_sig:.4f} | "
                  f"{elapsed:.1f}s")
        
        # Early stopping
        if val_mse < best_val_loss:
            best_val_loss = val_mse
            patience_counter = 0
            engine.save(str(Path(cfg.model_dir) / 'best_real_model.npz'))
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                if cfg.verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.perf_counter() - t_start
    
    # 5. 评估
    print(f"\n[评估]")
    final_mse, final_sig = evaluate_on_data(engine, val_data, cfg.batch_size, full=True)
    
    # Rollout 评估
    print(f"\n  Rollout 评估 (20步预测漂移):")
    drift_list = []
    enc = engine.encoder
    pred = engine.predictor
    
    for i in range(min(10, len(val_data[0]))):
        z0 = enc.encode(vision_feat=val_data[0][i][np.newaxis, :])[0]
        z = z0.copy()
        for t in range(min(20, len(val_data[1]))):
            action = val_data[1][t] if i < len(val_data[1]) else np.zeros(32, dtype=np.float32)
            if isinstance(action, np.ndarray) and action.ndim == 1:
                z = pred.predict(z, action)
        z_goal = enc.encode(vision_feat=val_data[2][i][np.newaxis, :])[0]
        drift = float(np.linalg.norm(z - z_goal))
        drift_list.append(drift)
    
    avg_drift = float(np.mean(drift_list)) if drift_list else 0
    print(f"  平均20步漂移: {avg_drift:.4f}")
    
    # 表示健康度
    all_z = []
    for i in range(min(50, len(val_data[0]))):
        z = enc.encode(vision_feat=val_data[0][i][np.newaxis, :])[0]
        all_z.append(z)
    embeddings = np.stack(all_z)
    sigreg_val = sigreg(embeddings)
    health = float(np.exp(-sigreg_val * 5))
    
    # 6. 输出
    print(f"\n{'=' * 60}")
    print("训练完成")
    print(f"{'=' * 60}")
    print(f"  数据: {stats['n_total']} 条来自 {stats['n_batches']} 个批次")
    print(f"  训练: {epoch+1} epochs, {total_time:.1f}s")
    print(f"  最终验证 MSE: {final_mse:.4f}")
    print(f"  20步漂移: {avg_drift:.4f}")
    print(f"  表示健康度: {health:.3f}")
    print(f"  模型保存: {cfg.model_dir}/best_real_model.npz")
    
    # 保存日志
    log_path = Path(cfg.model_dir) / 'training_log.json'
    log_data = {
        'train_log': {k: [float(v) for v in vals] for k, vals in train_log.items()},
        'final': {
            'val_mse': float(final_mse),
            'avg_drift': float(avg_drift),
            'health': float(health),
            'epochs': epoch + 1,
            'samples': len(obs),
        },
        'config': {
            'n_epochs': cfg.n_epochs,
            'batch_size': cfg.batch_size,
            'learning_rate': cfg.learning_rate,
            'sigreg_lambda': cfg.sigreg_lambda,
        }
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    print(f"  训练日志: {log_path}")
    
    return engine


def evaluate_on_data(
    engine: LeWMEngine,
    data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    full: bool = False
) -> Tuple[float, float]:
    """在数据集上评估 MSE 和 SIGReg"""
    obs, acts, next_obs = data
    enc = engine.encoder
    pred = engine.predictor
    
    total_mse = 0.0
    total_sig = 0.0
    n_batches = max(1, len(obs) // batch_size)
    
    # 采样评估（如果 full=False 且数据量大）
    if not full and len(obs) > 500:
        indices = np.random.choice(len(obs), 500, replace=False)
        obs = obs[indices]
        acts = acts[indices]
        next_obs = next_obs[indices]
    
    for start in range(0, len(obs), batch_size):
        end = start + batch_size
        batch_obs = obs[start:end]
        batch_acts = acts[start:end]
        batch_next = next_obs[start:end]
        
        z_t = enc.encode(vision_feat=batch_obs)
        z_next = enc.encode(vision_feat=batch_next)
        z_pred = pred.predict(z_t, batch_acts)
        
        mse = float(np.mean((z_pred - z_next) ** 2))
        sig = sigreg(z_pred)
        
        total_mse += mse
        total_sig += sig
    
    return total_mse / n_batches, total_sig / n_batches


# ================================================================
# 命令行入口
# ================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LeWM on real LAAP data')
    parser.add_argument('--data-dir', default=_DATA_DIR)
    parser.add_argument('--model-dir', default=_MODEL_DIR)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lambda', type=float, dest='sigreg_lambda', default=0.1)
    parser.add_argument('--verbose', action='store_true', default=True)
    
    args = parser.parse_args() if len(sys.argv) > 1 else parser.parse_args([])
    
    # 如果不带参数，检查是否有数据
    if len(sys.argv) <= 1:
        loader = DatasetReplayLoader(args.data_dir)
        stats = loader.stats()
        if stats['n_total'] == 0:
            print("️  未找到训练数据。")
            print(f"   数据目录: {args.data_dir}")
            print("\n   要收集数据，请安装 PSI 循环钩子:")
            print("     from laap.agi.psi_data_collector import install_data_collector")
            suggest = "  或者先用合成数据训练:\n     python -m laap.agi.train_le_wm"
            print(suggest)
            print(f"\n   数据目录内容: {list(Path(args.data_dir).glob('*'))[:10]}")
            sys.exit(1)
    
    cfg = RealTrainConfig(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        sigreg_lambda=args.sigreg_lambda,
        verbose=args.verbose,
    )
    
    train_on_real_data(cfg)
