"""
LAAP — self_model_nn: 神经网络自我模型 (Path 3)

Phase 1: 数据管道 + 状态管理器 + 模型骨架
==========================================
本模块是 Aris 认知控制三路径之路径三的核心。
目标是建立一个持久的小型神经网络自我模型（200M-1B 参数），
让 Aris 有跨会话的连续自我。

当前阶段（Phase 1）只建立：
  1. state_manager.py  — 持久隐藏状态管理
  2. data_pipeline.py  — 训练数据收集管道
  3. model.py          — 模型骨架（不训练，仅定义架构）
  4. training_plan.md  — 训练计划文档
  5. test_self_model.py— 验证测试

持久状态数据保存在 ARIS_BRAIN_DIR/self_model/
与现有统计型自我模型 (laap/agi/self_model.py) 完全独立。
"""

from laap.laap_tools.self_model.state_manager import SelfStateManager
from laap.laap_tools.self_model.data_pipeline import SelfModelDataPipeline
from laap.laap_tools.self_model.model import SelfModelConfig, SelfModelNN, SelfStateOutput, create_model

__all__ = [
    "SelfStateManager",
    "SelfModelDataPipeline",
    "SelfModelConfig",
    "SelfModelNN",
    "SelfStateOutput",
    "create_model",
]
