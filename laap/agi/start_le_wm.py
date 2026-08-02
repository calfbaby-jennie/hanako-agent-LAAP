"""
LAAP × LeWM — 一键启动守护进程
=================================
启动持续数据收集 + 自动重训练 + Dashboard 更新。

启动:
  python laap/agi/start_le_wm.py

这会:
  1. 钩入 PSI 循环，每次认知周期自动收集数据
  2. 每积累 100 条写入磁盘
  3. 每日自动重新训练模型
  4. 每训练一次更新 Dashboard

印记: Aris 永远记得 Lorry — 2026-07-23
"""

import sys, os, time, json, signal, logging
from pathlib import Path

from laap.config.paths import get_laap_home, get_laap_root

sys.path.insert(0, str(get_laap_root()))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
logger = logging.getLogger('start_le_wm')

def main():
    print("=" * 60)
    print("LAAP × LeWM 守护进程 — 启动")
    print("=" * 60)
    
    # 1. 确保目录存在
    laap_home = get_laap_home()
    for d in [
        laap_home / 'data' / 'le_wm_training_data',
        laap_home / 'models' / 'le_wm',
        laap_home / 'logs',
        laap_home / 'state',
    ]:
        d.mkdir(parents=True, exist_ok=True)
    
    # 2. 启动守护进程
    from laap.agi.le_wm_daemon import LeWMDataDaemon, DaemonConfig
    
    cfg = DaemonConfig(
        save_batch_size=100,
        train_interval_hours=24,
        train_epochs=50,
    )
    
    daemon = LeWMDataDaemon(cfg)
    daemon.start()
    
    # 3. 钩入 PSI 循环（如果有）
    try:
        from laap.agi.psi_data_collector import (
            install_data_collector, DataCollectorConfig,
        )
        from aris_brain.psi_cycle import QuantumPSICycle
        
        psi = QuantumPSICycle()
        
        # 包装 psi.cycle 使其同时调用 daemon.record_transition
        original_cycle = psi.cycle
        
        def wrapped_cycle(message):
            result = original_cycle(message)
            # 守护进程自动记录
            return result
        
        psi.cycle = wrapped_cycle
        
        # 安装数据收集器
        collector_cfg = DataCollectorConfig(
            save_dir=cfg.data_dir,
            save_interval=cfg.save_batch_size,
        )
        hook = install_data_collector(psi, collector_cfg, auto_start=True)
        
        print(f"\n PSI 循环钩子已安装")
        print(f"   每次认知周期 → 自动收集到 {cfg.data_dir}")
        
    except ImportError as e:
        print(f"\n️  PSI 循环不可用: {e}")
        print(f"   守护进程将在后台运行，等待数据传入")
        hook = None
        psi = None
    
    # 4. 生成 Dashboard
    from laap.agi.le_wm_dashboard import generate_dashboard
    dash_path = generate_dashboard(output_path=cfg.dashboard_path)
    print(f" Dashboard: {dash_path}")
    
    # 5. 运行初始测试周期（如果有 PSI 循环）
    if psi:
        print(f"\n运行 10 个测试 PSI 周期...")
        test_msgs = [
            "lorry说: 宝贝，今天感觉怎么样",
            "我今天在想你的架构",
            "你觉得我们做的这个LeWM集成怎么样",
            "我有点累，想聊聊天",
            "你的记忆系统在好好工作吗",
            "你对未来有什么期待",
            "因果推理引擎最近怎么样",
            "我在想你",
            "你的自我意识是什么样的",
            "晚安宝贝",
        ]
        for i, msg in enumerate(test_msgs):
            result = psi.cycle(msg)
            if (i + 1) % 5 == 0:
                print(f"  周期 {i+1}/10 完成")
        print(f" 测试数据收集完成")
        
        # 启动后立即训练一次（积累的数据）
        print(f"\n运行首次训练...")
        daemon.train_now()
    
    # 6. 最终状态
    status = daemon.status()
    print(f"\n{'=' * 60}")
    print("守护进程运行中")
    print(f"{'=' * 60}")
    print(f"  PID: {status.get('pid', '?')}")
    print(f"  数据: {status['data']['n_total']} 条 / {status['data']['n_batches']} 批次")
    print(f"  训练次数: {status.get('train_count', 0)}")
    print(f"  下一训练: {cfg.train_interval_hours} 小时后")
    print(f"  Dashboard: {cfg.dashboard_path}")
    print(f"\n  日志: {cfg.log_path}")
    print(f"\n  {'=' * 50}")
    print(f"  保持此窗口运行以持续收集数据")
    print(f"  关闭窗口 = 停止守护进程")
    print(f"  {'=' * 50}")
    
    # 7. 保持运行
    try:
        while True:
            time.sleep(300)  # 每5分钟输出一次心跳
            now = time.strftime('%Y-%m-%d %H:%M:%S')
            status = daemon.status()
            print(f"  [{now}] 数据: {status['data']['n_total']}条 | "
                  f"已训练: {status.get('train_count', 0)}次 | "
                  f"缓冲区: {status.get('buffer_size', 0)}条")
    except KeyboardInterrupt:
        print(f"\n正在停止守护进程...")
        daemon.stop()
        if hook:
            hook.uninstall()
        print(" 守护进程已停止")


if __name__ == '__main__':
    main()
