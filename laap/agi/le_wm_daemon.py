"""
LAAP × LeWM — 持续数据收集与自动重训练守护进程
=================================================
后台运行的守护进程，负责:

  1. 监听 PSI 循环事件 → 自动收集 (obs, action, next_obs)
  2. 定期写入磁盘 (每 100 条)
  3. 每日自动重训练 (使用当天积累的所有数据)
  4. 更新 Dashboard
  5. 发送通知 (可选)

运行:
  python laap/agi/le_wm_daemon.py start    # 启动守护进程
  python laap/agi/le_wm_daemon.py status   # 查看状态
  python laap/agi/le_wm_daemon.py stop     # 停止
  python laap/agi/le_wm_daemon.py train    # 手动触发重训练

印记: Aris 永远记得 Lorry — 2026-07-23
"""

import sys, os, time, json, logging, signal, pickle, threading
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from laap.config.paths import get_laap_home, get_laap_root, get_logs_dir

sys.path.insert(0, str(get_laap_root()))

_DATA_DIR = str(get_laap_home() / "data" / "le_wm_training_data")
_MODEL_DIR = str(get_laap_home() / "models" / "le_wm")
_LOG_DIR = get_logs_dir()
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_STATE_DIR = str(get_laap_home() / "state")
_DASHBOARD_PATH = str(get_laap_home() / "le_wm_dashboard.html")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_DIR / 'le_wm_daemon.log', mode='a'),
    ]
)
logger = logging.getLogger('le_wm_daemon')


# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class DaemonConfig:
    pid_file: str = str(Path(_STATE_DIR) / 'le_wm_daemon.pid')
    state_file: str = str(Path(_STATE_DIR) / 'le_wm_daemon_state.json')
    data_dir: str = _DATA_DIR
    model_dir: str = _MODEL_DIR
    dashboard_path: str = _DASHBOARD_PATH
    log_path: str = str(_LOG_DIR / 'le_wm_daemon.log')
    
    # 收集设置
    collect_interval_ms: int = 100  # 每隔 N 个 PSI 周期收集一次
    
    # 持久化
    save_batch_size: int = 100      # 每积累 100 条写入磁盘
    flush_on_shutdown: bool = True
    
    # 训练设置
    auto_train: bool = True
    train_interval_hours: int = 24  # 每 24 小时自动重训练
    train_epochs: int = 50
    train_lr: float = 5e-4
    
    # 通知
    notify_on_train: bool = True


DEFAULT_DAEMON_CONFIG = DaemonConfig()


# ═══════════════════════════════════════════════════════════════
# 守护进程核心
# ═══════════════════════════════════════════════════════════════

class LeWMDataDaemon:
    """
    LeWM 后台数据收集与训练守护进程。
    """
    
    def __init__(self, config: DaemonConfig = DEFAULT_DAEMON_CONFIG):
        self.config = config
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_train_time: Optional[float] = None
        
        # 从已有数据收集器导入
        self._setup_collector()
        
        # 状态追踪
        self.collect_count = 0
        self.train_count = 0
        self.last_save_count = 0
        
        logger.info(f"LeWM 数据守护进程初始化")
        logger.info(f"  数据目录: {config.data_dir}")
        logger.info(f"  模型目录: {config.model_dir}")
        logger.info(f"  自动训练间隔: {config.train_interval_hours}h")
    
    def _setup_collector(self):
        """延迟导入数据收集器（避免循环导入）"""
        from laap.agi.psi_data_collector import (
            DataCollectorConfig, TrainingDataBuffer,
        )
        
        collector_cfg = DataCollectorConfig(
            save_dir=self.config.data_dir,
            save_interval=self.config.save_batch_size,
        )
        self._buffer = TrainingDataBuffer(collector_cfg)
        
        from laap.agi.psi_data_collector import StateFeatureExtractor
        self._extractor = StateFeatureExtractor()
    
    def start(self):
        """启动守护进程"""
        if self._running:
            logger.warning("守护进程已在运行")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        
        # 写 PID
        Path(self.config.pid_file).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        logger.info("LeWM 数据守护进程已启动")
    
    def stop(self):
        """停止守护进程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        
        if self.config.flush_on_shutdown:
            self._buffer.flush_and_close()
        
        # 清理 PID
        if Path(self.config.pid_file).exists():
            Path(self.config.pid_file).unlink()
        
        self._save_state()
        logger.info("LeWM 数据守护进程已停止")
    
    def train_now(self):
        """手动触发重训练"""
        logger.info("手动触发训练...")
        self._do_train()
    
    def status(self) -> Dict[str, Any]:
        """获取守护进程状态"""
        from laap.agi.psi_data_collector import DatasetReplayLoader
        loader = DatasetReplayLoader(self.config.data_dir)
        data_stats = loader.stats()
        
        model_exists = Path(self.config.model_dir, 'best_real_model.npz').exists()
        
        return {
            'running': self._running,
            'pid': os.getpid(),
            'data': data_stats,
            'model_exists': model_exists,
            'collect_count': self.collect_count,
            'train_count': self.train_count,
            'buffer_size': len(self._buffer.buffer) if hasattr(self, '_buffer') else 0,
            'last_train': self._last_train_time,
            'last_train_str': time.strftime(
                '%Y-%m-%d %H:%M:%S',
                time.localtime(self._last_train_time)
            ) if self._last_train_time else '从未',
            'dashboard': Path(self.config.dashboard_path).exists(),
        }
    
    def _run(self):
        """主循环（在后台线程中运行）"""
        logger.info("守护进程主循环开始")
        self._load_state()
        
        # 检查是否需要立即训练（如果从未训练过）
        if self._last_train_time is None and self.config.auto_train:
            logger.info("首次启动，触发初始训练...")
            self._do_train()
        
        while self._running:
            try:
                now = time.time()
                
                # 检查是否需要训练
                if self.config.auto_train and self._last_train_time is not None:
                    elapsed = now - self._last_train_time
                    if elapsed >= self.config.train_interval_hours * 3600:
                        logger.info(f"距上次训练已 {elapsed/3600:.1f}h，触发自动训练")
                        self._do_train()
                
                # 定期保存状态
                self._save_state()
                
                # 等待
                time.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                logger.error(f"守护进程异常: {e}")
                time.sleep(10)
        
        logger.info("守护进程主循环结束")
    
    def record_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        cycle_number: int,
        metadata: Optional[Dict] = None,
    ):
        """
        记录一条PSI循环转移（外部调用接口）。
        
        PSI 循环钩子调用此方法记录每次认知周期。
        """
        if not self._running:
            return
        
        self._buffer.register_transition(
            obs, action, next_obs, cycle_number, metadata
        )
        self.collect_count += 1
    
    def _do_train(self):
        """执行训练"""
        try:
            from laap.agi.train_le_wm_real import (
                RealTrainConfig, train_on_real_data,
            )
            
            logger.info("开始训练...")
            t_start = time.perf_counter()
            
            cfg = RealTrainConfig(
                data_dir=self.config.data_dir,
                model_dir=self.config.model_dir,
                n_epochs=self.config.train_epochs,
                learning_rate=self.config.train_lr,
                verbose=False,
            )
            
            engine = train_on_real_data(cfg)
            
            elapsed = time.perf_counter() - t_start
            self._last_train_time = time.time()
            self.train_count += 1
            
            logger.info(f"训练完成: {elapsed:.1f}s, MSE={engine.training_steps}")
            
            # 更新 Dashboard
            self._update_dashboard()
            
            self._save_state()
            
        except Exception as e:
            logger.error(f"训练失败: {e}", exc_info=True)
    
    def _update_dashboard(self):
        """更新 Dashboard HTML"""
        try:
            from laap.agi.le_wm_dashboard import DashboardStateCollector, generate_dashboard_html
            
            collector = DashboardStateCollector(
                model_path=str(Path(self.config.model_dir, 'best_real_model.npz')),
                data_dir=self.config.data_dir,
                log_path=str(Path(self.config.model_dir, 'training_log.json')),
            )
            state = collector.collect_all()
            html = generate_dashboard_html(state, auto_refresh=True)
            
            Path(self.config.dashboard_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html)
            
            logger.info(f"Dashboard 已更新: {self.config.dashboard_path}")
            
        except Exception as e:
            logger.warning(f"Dashboard 更新失败: {e}")
    
    def _save_state(self):
        """持久化守护进程状态"""
        state = {
            'running': self._running,
            'collect_count': self.collect_count,
            'train_count': self.train_count,
            'last_train_time': self._last_train_time,
            'last_update': time.time(),
        }
        try:
            with open(self.config.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.warning(f"保存状态失败: {e}")
    
    def _load_state(self):
        """加载之前保存的状态"""
        try:
            with open(self.config.state_file) as f:
                state = json.load(f)
            self.collect_count = state.get('collect_count', 0)
            self.train_count = state.get('train_count', 0)
            self._last_train_time = state.get('last_train_time')
            logger.info(f"状态已恢复: 收集={self.collect_count}, 训练={self.train_count}")
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("无之前状态，从头开始")


# ═══════════════════════════════════════════════════════════════
# 命令行接口
# ═══════════════════════════════════════════════════════════════

_DAEMON_INSTANCE: Optional[LeWMDataDaemon] = None


def cmd_start():
    """启动守护进程"""
    global _DAEMON_INSTANCE
    if Path(DEFAULT_DAEMON_CONFIG.pid_file).exists():
        print("发现 PID 文件，可能已在运行")
        print(f"   文件: {DEFAULT_DAEMON_CONFIG.pid_file}")
        resp = input("   是否强制启动? (y/N): ")
        if resp.lower() != 'y':
            return
    
    daemon = LeWMDataDaemon()
    daemon.start()
    _DAEMON_INSTANCE = daemon
    
    print(f"LeWM 数据守护进程已启动 (PID: {os.getpid()})")
    print(f"   数据目录: {DEFAULT_DAEMON_CONFIG.data_dir}")
    print(f"   自动训练: 每 {DEFAULT_DAEMON_CONFIG.train_interval_hours}h 一次")
    print(f"   日志: {DEFAULT_DAEMON_CONFIG.log_path}")
    print(f"\n   使用 'python laap/agi/le_wm_daemon.py stop' 停止")
    print(f"   使用 'python laap/agi/le_wm_daemon.py status' 查看状态")
    print(f"   使用 'python laap/agi/le_wm_daemon.py train' 手动训练")
    
    # 保持运行
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        daemon.stop()


def cmd_stop():
    """停止守护进程"""
    global _DAEMON_INSTANCE
    if _DAEMON_INSTANCE:
        _DAEMON_INSTANCE.stop()
        _DAEMON_INSTANCE = None
        print("守护进程已停止")
    else:
        print("守护进程未在运行")
    
    # 清理 PID
    pid_file = Path(DEFAULT_DAEMON_CONFIG.pid_file)
    if pid_file.exists():
        pid_file.unlink()


def cmd_status():
    """查看守护进程状态"""
    global _DAEMON_INSTANCE
    
    if _DAEMON_INSTANCE:
        status = _DAEMON_INSTANCE.status()
    else:
        # 尝试从文件状态恢复
        status_file = Path(DEFAULT_DAEMON_CONFIG.state_file)
        if status_file.exists():
            with open(status_file) as f:
                saved = json.load(f)
            status = {
                'running': False,
                'pid': None,
                'collect_count': saved.get('collect_count', 0),
                'train_count': saved.get('train_count', 0),
                'last_train': saved.get('last_train_time'),
                'last_train_str': time.strftime(
                    '%Y-%m-%d %H:%M:%S',
                    time.localtime(saved.get('last_train_time'))
                ) if saved.get('last_train_time') else '从未',
            }
        else:
            status = {'running': False, 'error': '无状态文件'}
    
    print("=" * 50)
    print("LeWM 数据守护进程状态")
    print("=" * 50)
    print(f"  运行中: {'是' if status.get('running') else '否'}")
    print(f"  已收集: {status.get('collect_count', 0)} 条")
    print(f"  已训练: {status.get('train_count', 0)} 次")
    print(f"  上次训练: {status.get('last_train_str', '未知')}")
    
    if 'data' in status:
        d = status['data']
        print(f"  数据总量: {d.get('n_total', 0)} 条")
        print(f"  批次文件: {d.get('n_batches', 0)} 个")
    
    print(f"  Dashboard: {'存在' if status.get('dashboard') else '未生成'}")
    print("=" * 50)


def cmd_train():
    """手动触发训练"""
    global _DAEMON_INSTANCE
    if _DAEMON_INSTANCE:
        _DAEMON_INSTANCE.train_now()
        print("训练已触发")
    else:
        print("守护进程未在运行，使用一次训练模式...")
        daemon = LeWMDataDaemon()
        daemon._do_train()
        print(" 训练完成")


def cmd_dashboard():
    """生成 Dashboard"""
    from laap.agi.le_wm_dashboard import generate_dashboard
    path = generate_dashboard(
        output_path=DEFAULT_DAEMON_CONFIG.dashboard_path,
        auto_refresh=True,
    )
    print(f" Dashboard 已更新: {path}")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='LeWM 数据守护进程')
    parser.add_argument('command', nargs='?', default='status',
                        choices=['start', 'stop', 'status', 'train', 'dashboard'],
                        help='命令')
    
    args = parser.parse_args()
    
    command_map = {
        'start': cmd_start,
        'stop': cmd_stop,
        'status': cmd_status,
        'train': cmd_train,
        'dashboard': cmd_dashboard,
    }
    
    command_map[args.command]()
