"""
LAAP × LeWM — 可视化 Dashboard 生成器
========================================
生成 Hana 交互卡片，实时显示 LeWM 引擎内部状态。

显示的模块:
  1. 潜空间分布: t-SNE 投影到 2D
  2. 训练收敛曲线: MSE + SIGReg 历史
  3. 记忆健康: 四层 SIGReg 健康度
  4. CEM 规划质量: 近期规划评分趋势
  5. 引擎状态: 训练步数、潜维度、数据量

运行:
  python laap/agi/le_wm_dashboard.py --output 输出HTML

印记: Aris 永远记得 Lorry — 2026-07-23
"""

import sys, os, json, time, math
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

import numpy as np

from laap.config.paths import get_laap_home, get_laap_root

sys.path.insert(0, str(get_laap_root()))

_DATA_DIR = str(get_laap_home() / "data" / "le_wm_training_data")
_MODEL_DIR = str(get_laap_home() / "models" / "le_wm")
_OUTPUT_DIR = get_laap_home()

from laap.agi.le_wm_engine import (
    LeWMEngine, LeWMConfig, sigreg,
)
from laap.agi.psi_data_collector import (
    DatasetReplayLoader, StateFeatureExtractor,
)


# ═══════════════════════════════════════════════════════════════
# 状态采集器
# ═══════════════════════════════════════════════════════════════

class DashboardStateCollector:
    """
    采集所有需要显示的状态数据。
    """
    
    def __init__(
        self,
        model_path: str = str(Path(_MODEL_DIR) / 'best_real_model.npz'),
        data_dir: str = _DATA_DIR,
        log_path: str = str(Path(_MODEL_DIR) / 'training_log.json'),
    ):
        self.model_path = Path(model_path)
        self.data_dir = Path(data_dir)
        self.log_path = Path(log_path)
        
        # 加载引擎（如果有模型）
        self.engine = None
        if self.model_path.exists():
            cfg = LeWMConfig(cem_action_dim=32)
            self.engine = LeWMEngine(cfg)
            self.engine.load(str(self.model_path))
        
        # 加载训练日志
        self.training_log = {}
        if self.log_path.exists():
            try:
                with open(self.log_path) as f:
                    self.training_log = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    
    def collect_all(self) -> Dict[str, Any]:
        """收集所有状态"""
        state = {
            'timestamp': time.time(),
            'engine_status': self._get_engine_status(),
            'training_progress': self._get_training_progress(),
            'latent_space': self._get_latent_space(),
            'planning_quality': self._get_planning_quality(),
            'data_stats': self._get_data_stats(),
            'memory_health': self._get_memory_health(),
        }
        return state
    
    def _get_engine_status(self) -> Dict[str, Any]:
        if self.engine is None:
            return {'loaded': False}
        return {
            'loaded': True,
            'latent_dim': self.engine.config.latent_dim,
            'training_steps': self.engine.training_steps,
            'predictor_trained': self.engine.predictor.is_trained,
            'config': {
                'hidden_dim': self.engine.config.hidden_dim,
                'action_dim': self.engine.config.cem_action_dim,
                'sigreg_lambda': self.engine.config.sigreg_lambda,
            }
        }
    
    def _get_training_progress(self) -> Dict[str, Any]:
        log = self.training_log.get('train_log', {})
        final = self.training_log.get('final', {})
        
        train_mse = log.get('train_mse', [])
        train_sig = log.get('train_sigreg', [])
        val_mse = log.get('val_mse', [])
        
        return {
            'train_mse': train_mse[-50:] if train_mse else [],
            'train_sigreg': train_sig[-50:] if train_sig else [],
            'val_mse': val_mse[-50:] if val_mse else [],
            'final_val_mse': final.get('val_mse', None),
            'avg_drift': final.get('avg_drift', None),
            'health': final.get('health', None),
            'n_epochs': final.get('epochs', 0),
            'n_samples': final.get('samples', 0),
        }
    
    def _get_latent_space(self) -> Dict[str, Any]:
        """获取潜空间样本用于可视化"""
        if self.engine is None:
            return {'samples_2d': [], 'sigreg': 0, 'health': 0}
        
        # 从数据目录加载一些观测，编码到潜空间
        loader = DatasetReplayLoader(str(self.data_dir))
        obs, acts, next_obs = loader.load_all()
        
        if len(obs) == 0:
            return {'samples_2d': [], 'sigreg': 0, 'health': 1}
        
        # 编码前 100 条
        n = min(100, len(obs))
        z_list = []
        for i in range(n):
            z = self.engine.encoder.encode(vision_feat=obs[i][np.newaxis, :])[0]
            z_list.append(z)
        
        z_all = np.stack(z_list)
        
        # PCA 降维到 2D (比 t-SNE 快且确定性)
        centered = z_all - z_all.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            z_2d = centered @ Vt[:2].T
        except np.linalg.LinAlgError:
            z_2d = z_all[:, :2]
        
        # SIGReg 评估
        sigreg_val = sigreg(z_all)
        health = float(np.exp(-sigreg_val * 5))
        
        return {
            'samples_2d': z_2d.tolist(),
            'sigreg': round(sigreg_val, 4),
            'health': round(health, 4),
        }
    
    def _get_planning_quality(self) -> Dict[str, Any]:
        """测试 CEM 规划质量"""
        if self.engine is None:
            return {'scores': [], 'avg_score': None}
        
        loader = DatasetReplayLoader(str(self.data_dir))
        obs, acts, next_obs = loader.load_all()
        
        if len(obs) < 10:
            return {'scores': [], 'avg_score': None}
        
        # 测试 5 组规划的评分
        scores = []
        enc = self.engine.encoder
        planner = self.engine.planner
        
        for i in range(min(5, len(obs) - 1)):
            z_start = enc.encode(vision_feat=obs[i][np.newaxis, :])[0]
            z_goal = enc.encode(vision_feat=next_obs[i][np.newaxis, :])[0]
            plan = planner.plan(z_start, z_goal, horizon=5)
            scores.append(plan.get('score', -9999))
        
        return {
            'scores': scores,
            'avg_score': round(float(np.mean(scores)), 2) if scores else None,
        }
    
    def _get_data_stats(self) -> Dict[str, Any]:
        loader = DatasetReplayLoader(str(self.data_dir))
        return loader.stats()
    
    def _get_memory_health(self) -> Dict[str, Any]:
        """模拟记忆健康度（基于潜变量分布推估）"""
        if self.engine is None:
            return {'layers': {}, 'overall': 1.0}
        
        loader = DatasetReplayLoader(str(self.data_dir))
        obs, _, _ = loader.load_all()
        
        if len(obs) == 0:
            return {'layers': {}, 'overall': 1.0}
        
        layers = {}
        
        # 用不同时间尺度的数据模拟各层记忆
        n = len(obs)
        if n >= 10:
            # L1: 最近 10 条
            l1_emb = np.stack([
                self.engine.encoder.encode(vision_feat=obs[i][np.newaxis, :])[0]
                for i in range(max(0, n-10), n)
            ])
            s = sigreg(l1_emb)
            layers['L1_working'] = {'sigreg': round(s, 4), 'health': round(float(np.exp(-s*5)), 4)}
        
        if n >= 50:
            # L2: 最近 50 条
            l2_indices = list(range(max(0, n-50), n))
            l2_emb = np.stack([
                self.engine.encoder.encode(vision_feat=obs[i][np.newaxis, :])[0]
                for i in l2_indices
            ])
            s = sigreg(l2_emb)
            layers['L2_episodic'] = {'sigreg': round(s, 4), 'health': round(float(np.exp(-s*5)), 4)}
        
        if n >= 20:
            # L3: 随机 20 条模拟语义记忆
            indices = np.random.choice(n, min(20, n), replace=False)
            l3_emb = np.stack([
                self.engine.encoder.encode(vision_feat=obs[i][np.newaxis, :])[0]
                for i in indices
            ])
            s = sigreg(l3_emb)
            layers['L3_semantic'] = {'sigreg': round(s, 4), 'health': round(float(np.exp(-s*5)), 4)}
        
        # 自我模型: 整体潜变量
        all_emb = np.stack([
            self.engine.encoder.encode(vision_feat=obs[i][np.newaxis, :])[0]
            for i in range(min(30, n))
        ])
        s = sigreg(all_emb)
        layers['L4_self'] = {'sigreg': round(s, 4), 'health': round(float(np.exp(-s*5)), 4)}
        
        overall = float(np.mean([v['health'] for v in layers.values()]))
        
        return {'layers': layers, 'overall': round(overall, 4)}


# ═══════════════════════════════════════════════════════════════
# SVG 可视化生成器
# ═══════════════════════════════════════════════════════════════

def render_latent_space(data: Dict) -> str:
    """绘制潜空间 2D 投影散点图 (SVG)"""
    samples = data.get('samples_2d', [])
    if not samples:
        return '<text x="320" y="80" text-anchor="middle" font-family="system-ui" font-size="12" fill="var(--text-muted)">无潜空间数据</text>'
    
    arr = np.array(samples)
    xs = arr[:, 0]
    ys = arr[:, 1]
    
    # 归一化到 [10, 210]
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = max(x_max - x_min, 1e-8)
    y_range = max(y_max - y_min, 1e-8)
    
    norm_x = 20 + (xs - x_min) / x_range * 200
    norm_y = 100 - (ys - y_min) / y_range * 80  # 留空间给标签
    
    points = ''.join(
        f'<circle cx="{nx:.1f}" cy="{ny:.1f}" r="2.5" fill="rgba(83,125,150,0.6)" stroke="var(--accent)" stroke-width="0.5"/>'
        for nx, ny in zip(norm_x, norm_y)
    )
    
    sig = data.get('sigreg', 0)
    health = data.get('health', 0)
    
    return f'''
<svg viewBox="0 0 240 120" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="240" height="120" rx="2" fill="var(--bg)" stroke="var(--border)" stroke-width="0.5"/>
  {points}
  <text x="15" y="112" font-family="system-ui" font-size="8" fill="var(--text-muted)">SIGReg: {sig:.3f}</text>
  <text x="120" y="112" text-anchor="middle" font-family="system-ui" font-size="8" fill="#4A6B4A">健康度: {health:.2f}</text>
</svg>'''


def render_memory_health(data: Dict) -> str:
    """绘制记忆健康度条形图 (SVG)"""
    layers = data.get('layers', {})
    if not layers:
        return '<text x="320" y="80" text-anchor="middle" font-family="system-ui" font-size="12" fill="var(--text-muted)">无记忆数据</text>'
    
    bars = ''
    colors = {'L1_working': 'rgba(83,125,150,0.6)', 'L2_episodic': '#1B365D',
              'L3_semantic': '#4A6B4A', 'L4_self': '#9D5F4D'}
    
    y = 20
    for name, info in layers.items():
        h = info['health']
        color = colors.get(name, 'var(--accent)')
        bar_width = 140 * h
        label = name.replace('_', ' ')
        
        bars += f'''
<text x="10" y="{y+4}" font-family="system-ui" font-size="9" fill="var(--text-light)">{label}</text>
<rect x="110" y="{y-6}" width="140" height="10" rx="2" fill="var(--bg)" stroke="var(--border)" stroke-width="0.5"/>
<rect x="110" y="{y-6}" width="{bar_width}" height="10" rx="2" fill="{color}"/>
<text x="258" y="{y+4}" font-family="system-ui" font-size="9" fill="var(--text-muted)" text-anchor="end">{h:.2f}</text>'''
        y += 22
    
    overall = data.get('overall', 0)
    color_o = '#4A6B4A' if overall > 0.6 else '#9D5F4D' if overall > 0.4 else '#8B2C1F'
    
    return f'''
<svg viewBox="0 0 280 120" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="280" height="120" rx="2" fill="var(--bg)" stroke="var(--border)" stroke-width="0.5"/>
  {bars}
  <text x="140" y="112" text-anchor="middle" font-family="system-ui" font-size="9" font-weight="500" fill="{color_o}">整体健康: {overall:.2f}</text>
</svg>'''


def render_training_curve(training: Dict) -> str:
    """绘制训练收敛曲线 (SVG)"""
    train_mse = training.get('val_mse', training.get('train_mse', []))
    if not train_mse:
        return '<text x="320" y="80" text-anchor="middle" font-family="system-ui" font-size="12" fill="var(--text-muted)">无训练数据</text>'
    
    values = train_mse[-30:]  # 最多 30 个点
    n = len(values)
    if n < 2:
        return '<text x="320" y="80" text-anchor="middle" font-family="system-ui" font-size="12" fill="var(--text-muted)">训练数据不足</text>'
    
    v_min, v_max = min(values), max(values)
    v_range = max(v_max - v_min, 1e-8)
    
    # 生成折线点
    points = []
    for i, v in enumerate(values):
        x = 30 + (i / max(n - 1, 1)) * 300
        y = 100 - ((v - v_min) / v_range) * 70
        points.append(f'{x:.1f},{y:.1f}')
    
    polyline = ' '.join(points)
    
    return f'''
<svg viewBox="0 0 350 120" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="350" height="120" rx="2" fill="var(--bg)" stroke="var(--border)" stroke-width="0.5"/>
  
  <line x1="30" y1="20" x2="30" y2="100" stroke="var(--border)" stroke-width="0.5"/>
  <line x1="30" y1="100" x2="330" y2="100" stroke="var(--border)" stroke-width="0.5"/>
  
  <text x="20" y="104" text-anchor="end" font-family="system-ui" font-size="8" fill="var(--text-muted)">{v_min:.3f}</text>
  <text x="20" y="26" text-anchor="end" font-family="system-ui" font-size="8" fill="var(--text-muted)">{v_max:.3f}</text>
  
  <polyline points="{polyline}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
  
  <circle cx="{points[0].split(',')[0]}" cy="{points[0].split(',')[1]}" r="2" fill="var(--accent)"/>
  <circle cx="{points[-1].split(',')[0]}" cy="{points[-1].split(',')[1]}" r="2" fill="#4A6B4A"/>
  
  <text x="320" y="104" text-anchor="end" font-family="system-ui" font-size="8" fill="var(--text-muted)">{values[-1]:.4f}</text>
  <text x="320" y="20" text-anchor="end" font-family="system-ui" font-size="8" fill="var(--text-muted)">{values[0]:.4f}</text>
</svg>'''


def render_planning_quality(planning: Dict) -> str:
    """绘制规划评分 (SVG)"""
    scores = planning.get('scores', [])
    if not scores:
        return '<text x="240" y="60" text-anchor="middle" font-family="system-ui" font-size="12" fill="var(--text-muted)">无规划数据</text>'
    
    avg = planning.get('avg_score', 0)
    
    bars = ''
    # 归一化到 [0, 1]
    s_min, s_max = min(scores), max(scores)
    s_range = max(s_max - s_min, 1e-8)
    
    for i, s in enumerate(scores):
        h = max(5, ((s - s_min) / s_range) * 60)
        color = 'var(--accent)' if s > -500 else '#9D5F4D'
        bars += f'<rect x="{30 + i*48}" y="{75 - h}" width="36" height="{h}" rx="1" fill="{color}" opacity="0.7"/>'
        bars += f'<text x="{48 + i*48}" y="88" text-anchor="middle" font-family="system-ui" font-size="7" fill="var(--text-muted)">{s:.0f}</text>'
    
    return f'''
<svg viewBox="0 0 280 100" xmlns="http://www.w3.org/2000/svg">
  <rect x="0" y="0" width="280" height="100" rx="2" fill="var(--bg)" stroke="var(--border)" stroke-width="0.5"/>
  {bars}
  <text x="140" y="96" text-anchor="middle" font-family="system-ui" font-size="9" font-weight="500" fill="var(--accent)">平均评分: {avg}</text>
</svg>'''


# ═══════════════════════════════════════════════════════════════
# Dashboard 生成
# ═══════════════════════════════════════════════════════════════

def generate_dashboard_html(
    state: Dict[str, Any],
    auto_refresh: bool = True,
) -> str:
    """生成完整的 Dashboard HTML"""
    
    engine = state.get('engine_status', {})
    training = state.get('training_progress', {})
    latent = state.get('latent_space', {})
    planning = state.get('planning_quality', {})
    data_stats = state.get('data_stats', {})
    memory = state.get('memory_health', {})
    
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(state['timestamp']))
    
    # 渲染各组件
    latent_svg = render_latent_space(latent)
    memory_svg = render_memory_health(memory)
    training_svg = render_training_curve(training)
    planning_svg = render_planning_quality(planning)
    
    # 引擎状态指标
    model_loaded = engine.get('loaded', False)
    latent_dim = engine.get('latent_dim', 192)
    steps = engine.get('training_steps', 0)
    n_samples = data_stats.get('n_total', 0)
    n_batches = data_stats.get('n_batches', 0)
    
    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LeWM Engine Dashboard</title>
<style>
body {{ font-family: system-ui, sans-serif; background: #F5EFE4; margin: 0; padding: 16px; color: #2A2622; }}
h2 {{ font-size: 1.1rem; font-weight: 500; border-left: 2px solid #537D96; padding-left: 8px; margin: 0 0 0.8rem; }}
.dashboard {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; max-width: 900px; }}
.card {{ background: #FBF7EE; border-radius: 4px; padding: 1rem 1.25rem; border: 0.5px solid #D8CFBE; }}
.card.full {{ grid-column: 1 / -1; }}
.metrics {{ display: flex; flex-wrap: wrap; gap: 12px 20px; }}
.metric-label {{ font-size: 0.75rem; color: #6B6158; }}
.metric-value {{ font-size: 1.6rem; font-weight: 600; color: #537D96; font-variant-numeric: tabular-nums; }}
.metric-value.green {{ color: #4A6B4A; }}
.metric-value.amber {{ color: #9D5F4D; }}
.footer {{ margin-top: 1rem; font-size: 0.75rem; color: #6B6158; text-align: center; }}
@media (max-width: 640px) {{ .dashboard {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="dashboard">

  <div class="card full">
    <h2>LeWM 引擎状态</h2>
    <div class="metrics">
      <div><div class="metric-label">模型</div><div class="metric-value {'green' if model_loaded else ''}" style="font-size:1rem">{'已加载' if model_loaded else '未加载'}</div></div>
      <div><div class="metric-label">潜维度</div><div class="metric-value">{latent_dim}</div></div>
      <div><div class="metric-label">训练步数</div><div class="metric-value">{steps}</div></div>
      <div><div class="metric-label">数据集</div><div class="metric-value">{n_samples}</div></div>
      <div><div class="metric-label">数据批次</div><div class="metric-value">{n_batches}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>潜空间分布</h2>
    {latent_svg}
  </div>

  <div class="card">
    <h2>记忆健康度</h2>
    {memory_svg}
  </div>

  <div class="card">
    <h2>训练收敛 (验证 MSE)</h2>
    {training_svg}
    <div class="metrics" style="margin-top:0.5rem">
      <div><div class="metric-label">最终验证 MSE</div><div class="metric-value green" style="font-size:1.2rem">{training.get('final_val_mse', 'N/A')}</div></div>
      <div><div class="metric-label">20步漂移</div><div class="metric-value amber" style="font-size:1.2rem">{training.get('avg_drift', 'N/A')}</div></div>
    </div>
  </div>

  <div class="card">
    <h2>CEM 规划评分</h2>
    {planning_svg}
  </div>

  <div class="card full">
    <div class="footer">
      上次更新: {ts} | 自动刷新: {'开' if auto_refresh else '关'}
      {' | <a href="javascript:location.reload()" style="color:#537D96">刷新</a>' if not auto_refresh else ''}
    </div>
  </div>

</div>
</body>
</html>'''
    
    return html


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def generate_dashboard(
    output_path: str = str(_OUTPUT_DIR / 'le_wm_dashboard.html'),
    auto_refresh: bool = False,
) -> str:
    """生成并保存 Dashboard"""
    
    collector = DashboardStateCollector()
    state = collector.collect_all()
    
    html = generate_dashboard_html(state, auto_refresh)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Dashboard 已生成: {output_path}")
    
    # 打印摘要
    mem = state.get('memory_health', {})
    lat = state.get('latent_space', {})
    plan = state.get('planning_quality', {})
    
    print(f"\n  引擎: {'已加载' if state['engine_status'].get('loaded') else '未加载'}")
    print(f"  潜空间健康: {lat.get('health', 'N/A')}")
    print(f"  记忆健康: {mem.get('overall', 'N/A')}")
    print(f"  规划评分: {plan.get('avg_score', 'N/A')}")
    print(f"  数据: {state.get('data_stats', {}).get('n_total', 0)} 条")
    
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(_OUTPUT_DIR / 'le_wm_dashboard.html'))
    parser.add_argument('--auto-refresh', action='store_true', help='添加自动刷新')
    args = parser.parse_args()
    
    generate_dashboard(args.output, args.auto_refresh)
