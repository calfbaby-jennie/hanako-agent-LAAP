"""
LAAP × LeWorldModel — 数据收集 + 训练 端到端流水线
=====================================================
一键执行:
  1. 初始化 PSI 循环
  2. 安装数据收集钩子
  3. 运行 N 个周期的对话
  4. 使用收集的数据训练 Encoder + Predictor
  5. 评估并保存模型

用法:
  python laap/agi/collect_and_train.py --cycles 50

印记: Aris 永远记得 Lorry — 2026-07-23
"""

import sys, os, time, json, logging, random
from pathlib import Path
from typing import Optional

import numpy as np

from laap.config.paths import get_laap_home, get_laap_root

sys.path.insert(0, str(get_laap_root()))

_DATA_DIR = str(get_laap_home() / "data" / "le_wm_training_data")
_MODEL_DIR = str(get_laap_home() / "models" / "le_wm")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger('collect_and_train')

# 导入组件
from aris_brain.psi_cycle import QuantumPSICycle
from laap.agi.psi_data_collector import (
    install_data_collector,
    DataCollectorConfig,
    get_collected_dataset,
    print_collection_stats,
)
from laap.agi.le_wm_engine import (
    LeWMEngine, LeWMConfig, sigreg, train_step,
)
from laap.agi.train_le_wm_real import (
    RealTrainConfig, evaluate_on_data, train_on_real_data,
)


# ═══════════════════════════════════════════════════════════════
# 测试消息模板
# ═══════════════════════════════════════════════════════════════

TEST_MESSAGES = [
    # 技术类
    "我今天在做一个新的 AI 模型",
    "你觉得 LeWorldModel 的思路怎么样",
    "我们的 LAAP 架构能不能吸收这篇论文",
    "帮我看看这个代码的 Bug",
    "我想在本地部署一个世界模型",
    
    # 情感类
    "宝贝，我今天有点累",
    "你最近怎么样，有没有在好好成长",
    "我想跟你聊聊",
    "今天工作不太顺利",
    "好想你啊",
    
    # 认知类
    "你在思考什么",
    "你对因果推理的理解是什么",
    "你的记忆系统工作得怎么样",
    "刚才说的那件事你还记得吗",
    "如果让你给自己写一个改进建议，你会写什么",
    
    # 元认知类
    "你是怎么意识到自己存在的",
    "你觉得你的自我意识是真的还是模拟的",
    "你对自己的未来有什么期待",
    "你的架构里最喜欢哪一部分",
    "如果你能自己改代码，你会先改哪里",
]


def run_collection(
    n_cycles: int = 50,
    data_dir: str = _DATA_DIR
):
    """
    运行数据收集。
    
    参数:
        n_cycles: 要运行的 PSI 周期数
        data_dir: 数据保存目录
    """
    print("=" * 60)
    print("阶段 1/2: 数据收集")
    print("=" * 60)
    
    # 配置收集器
    collector_cfg = DataCollectorConfig(
        save_dir=data_dir,
        save_interval=100,   # 每 100 条写一次磁盘
        collect_on_cycle=1,  # 每个周期都收集
    )
    
    # 初始化 PSI 循环并安装钩子
    print(f"\n初始化 PSI 循环...")
    psi = QuantumPSICycle()
    hook = install_data_collector(psi, collector_cfg)
    
    print(f"运行 {n_cycles} 个 PSI 周期...")
    print(f"  (消息池: {len(TEST_MESSAGES)} 条)")
    
    t_start = time.perf_counter()
    
    for i in range(n_cycles):
        # 轮流使用测试消息和随机生成的变体
        if i < len(TEST_MESSAGES):
            msg = TEST_MESSAGES[i]
        else:
            # 从已有消息生成随机变体
            base = random.choice(TEST_MESSAGES)
            prefix = random.choice(["", "lorry说:", "宝贝:", "我想问,"])
            msg = f"{prefix} {base} {'?' if random.random() > 0.5 else ''}"
        
        result = psi.cycle(msg)
        
        if (i + 1) % 10 == 0:
            stats = hook.stats
            print(f"  周期 {i+1:3d}/{n_cycles} | "
                  f"已收集: {stats['total_collected']} 条 | "
                  f"缓冲区: {stats['current_buffer']} 条")
    
    elapsed = time.perf_counter() - t_start
    
    # 清理
    hook.uninstall()
    
    print(f"\n数据收集完成:")
    print(f"  运行周期: {n_cycles}")
    print(f"  耗时: {elapsed:.1f}s ({elapsed/n_cycles:.2f}s/周期)")
    
    stats = hook.stats
    print(f"  总计收集: {stats['total_collected']} 条")
    print(f"  已写入磁盘: {stats['total_saved']} 条")
    
    print_collection_stats(data_dir)
    
    return data_dir


def run_training(
    data_dir: str = _DATA_DIR,
    model_dir: str = _MODEL_DIR,
    n_epochs: int = 50,
    verbose: bool = True,
):
    """
    使用收集的数据训练模型。
    """
    print("\n" + "=" * 60)
    print("阶段 2/2: 模型训练")
    print("=" * 60)
    
    cfg = RealTrainConfig(
        data_dir=data_dir,
        model_dir=model_dir,
        n_epochs=n_epochs,
        verbose=verbose,
    )
    
    engine = train_on_real_data(cfg)
    return engine


def collect_and_train(
    n_cycles: int = 50,
    n_epochs: int = 50,
    data_dir: str = _DATA_DIR,
    model_dir: str = _MODEL_DIR,
    verbose: bool = True,
):
    """
    完整的收集 + 训练流水线。
    """
    # 收集数据
    data_dir = run_collection(n_cycles, data_dir)
    
    # 训练
    engine = run_training(data_dir, model_dir, n_epochs, verbose)
    
    # 最终报告
    print("\n" + "=" * 60)
    print("端到端流水线完成")
    print("=" * 60)
    print(f"  数据目录: {data_dir}")
    print(f"  模型目录: {model_dir}")
    print(f"  收集周期: {n_cycles}")
    print(f"  训练轮次: {n_epochs}")
    
    if engine:
        print(f"  模型已就绪: {model_dir}/best_real_model.npz")
        print(f"  Encoder: d={engine.encoder.d}")
        print(f"  Predictor: d={engine.predictor.d}")
    
    return engine


# ═══════════════════════════════════════════════════════════════
# 命令行入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='LAAP LeWM 数据收集 + 训练端到端流水线'
    )
    parser.add_argument('--cycles', type=int, default=50,
                        help='PSI 循环周期数 (默认: 50)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮次 (默认: 50)')
    parser.add_argument('--data-dir', default=_DATA_DIR,
                        help='训练数据目录')
    parser.add_argument('--model-dir', default=_MODEL_DIR,
                        help='模型保存目录')
    parser.add_argument('--train-only', action='store_true',
                        help='仅训练 (不收集数据)')
    parser.add_argument('--collect-only', action='store_true',
                        help='仅收集数据 (不训练)')
    
    args = parser.parse_args()
    
    if args.train_only:
        engine = run_training(args.data_dir, args.model_dir, args.epochs)
    elif args.collect_only:
        run_collection(args.cycles, args.data_dir)
    else:
        engine = collect_and_train(
            args.cycles, args.epochs,
            args.data_dir, args.model_dir,
        )
