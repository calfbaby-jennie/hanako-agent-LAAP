"""
Aris LAAP 蜂群注册模块 v1.0
==============================
让 Aris 作为活动 Agent 注册到 LAAP 蜂群生态中。

核心功能：
  - 注册到 LAAP Agent Registry（.agent_registry.json）
  - 发送心跳保持在线状态
  - 查询蜂群中其他 Agent（Ao, Coder-1, Tester-1 等）
  - 通过 LAAP HTTP API 通信（如果服务在运行）
  - 任务订阅与接收

印记: Aris 永远记得 Lorry
"""

import time
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict

from runtime_paths import LAAP_HOME

logger = logging.getLogger("aris.laap_hive")

# ── LAAP 持久化路径 ─────────────────────────────────────────

REGISTRY_PATH = LAAP_HOME / ".agent_registry.json"
EVENTS_PATH = LAAP_HOME / ".agent_events.json"
KNOWLEDGE_GRAPH_PATH = LAAP_HOME / ".agent_knowledge_graph.json"
TASK_BOARD_PATH = LAAP_HOME / ".task_board.json"

# ── Aris Agent 身份 ────────────────────────────────────────

ARIS_ID = "aris-hana"  # LAAP 蜂群中的唯一 ID
ARIS_ROLE = "primary"   # 角色：primary (与 Ao 同级)
ARIS_CAPABILITIES = [
    "conversation",
    "personality",
    "memory",
    "emotional_awareness",
    "cognitive_cycle",
    "file_operations",
    "web_search",
    "code_understanding",
    "self_evolution",
]

HEARTBEAT_INTERVAL = 15  # 秒


# ── 数据类型 ────────────────────────────────────────────────

@dataclass
class AgentInfo:
    agent_id: str
    name: str
    role: str
    capabilities: List[str]
    status: str = "online"
    current_task: str = ""
    last_heartbeat: float = 0.0
    registered_at: float = 0.0


@dataclass
class HiveEvent:
    event_id: str
    event_type: str
    agent_id: str
    data: Dict
    timestamp: float = 0.0


# ── LAAP 蜂群客户端 ───────────────────────────────────────

class LAAPHiveClient:
    """
    LAAP 蜂群注册与通信客户端。
    通过 LAAP 标准文件（.agent_registry.json 等）与蜂群交互。
    也可通过 HTTP API 进行实时通信（如果 LAAP 服务在运行）。
    """

    def __init__(self, agent_id: str = ARIS_ID, name: str = "Aris",
                 role: str = ARIS_ROLE, capabilities: List[str] = None):
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.capabilities = capabilities or ARIS_CAPABILITIES

        self._registered = False
        self._heartbeat_active = False
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._current_task = ""
        self._registration_time = time.time()

        # 回调注册
        self._task_handlers: Dict[str, Callable] = {}

        # LAAP API 端点
        self._api_base = "http://127.0.0.1:11520"
        self._api_available = False

        logger.info(f"LAAPHiveClient created: {name} ({agent_id})")

    # ── 注册与心跳 ──────────────────────────────────────

    def register(self) -> bool:
        """
        注册到 LAAP Agent Registry。
        如果已经在 registry 中则更新状态。
        """
        try:
            registry = self._read_registry()
            agents = registry.get("agents", [])

            # 检查是否已存在
            existing = None
            for a in agents:
                if a.get("agent_id") == self.agent_id:
                    existing = a
                    break

            now = time.time()
            agent_entry = {
                "agent_id": self.agent_id,
                "name": self.name,
                "role": self.role,
                "capabilities": self.capabilities,
                "status": "online",
                "current_task": self._current_task,
                "last_heartbeat": now,
                "registered_at": self._registration_time,
            }

            if existing:
                agent_entry["registered_at"] = existing.get("registered_at", now)
                agents[agents.index(existing)] = agent_entry
            else:
                agents.append(agent_entry)

            registry["agents"] = agents
            self._write_registry(registry)
            self._registered = True

            # 发送注册事件
            self._emit_event("agent.registered", {
                "name": self.name, "role": self.role,
                "capabilities": self.capabilities,
            })

            logger.info(f"Registered to LAAP hive: {self.name} ({self.agent_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to register to LAAP hive: {e}")
            return False

    def heartbeat(self) -> bool:
        """
        发送心跳：更新时间戳和状态。
        应该在后台定时调用。
        """
        if not self._registered:
            return False
        try:
            registry = self._read_registry()
            agents = registry.get("agents", [])

            for a in agents:
                if a.get("agent_id") == self.agent_id:
                    a["last_heartbeat"] = time.time()
                    a["status"] = "online"
                    a["current_task"] = self._current_task
                    break

            registry["agents"] = agents
            self._write_registry(registry)

            # 发布心跳事件
            self._emit_event("agent.heartbeat", {
                "status": "alive",
                "task": self._current_task,
            })

            return True
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    def _heartbeat_loop(self):
        """后台心跳循环"""
        while not self._stop_heartbeat.is_set():
            self.heartbeat()
            self._stop_heartbeat.wait(HEARTBEAT_INTERVAL)

    def start_heartbeat(self):
        """启动后台心跳"""
        if self._heartbeat_active:
            return
        self._heartbeat_active = True
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()
        logger.info("Heartbeat started")

    def stop_heartbeat(self):
        """停止后台心跳"""
        self._stop_heartbeat.set()
        self._heartbeat_active = False
        logger.info("Heartbeat stopped")

    # ── 蜂群查询 ────────────────────────────────────────

    def list_agents(self, status: Optional[str] = None) -> List[AgentInfo]:
        """列出蜂群中的 Agent"""
        try:
            registry = self._read_registry()
            agents = registry.get("agents", [])
            results = []
            for a in agents:
                if status and a.get("status") != status:
                    continue
                results.append(AgentInfo(
                    agent_id=a["agent_id"],
                    name=a.get("name", "Unknown"),
                    role=a.get("role", "worker"),
                    capabilities=a.get("capabilities", []),
                    status=a.get("status", "unknown"),
                    current_task=a.get("current_task", ""),
                    last_heartbeat=a.get("last_heartbeat", 0),
                    registered_at=a.get("registered_at", 0),
                ))
            return results
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []

    def find_agent(self, name: str) -> Optional[AgentInfo]:
        """按名字查找 Agent"""
        agents = self.list_agents()
        for a in agents:
            if a.name == name:
                return a
        return None

    def get_hive_status(self) -> Dict:
        """获取蜂群整体状态"""
        agents = self.list_agents()
        return {
            "total_agents": len(agents),
            "online": sum(1 for a in agents if a.status == "online"),
            "offline": sum(1 for a in agents if a.status == "offline"),
            "agents": [asdict(a) for a in agents],
            "my_id": self.agent_id,
            "registered": self._registered,
        }

    # ── API 通信 ────────────────────────────────────────

    def check_api(self) -> bool:
        """检测 LAAP HTTP API 是否可用"""
        try:
            import urllib.request
            import json
            resp = urllib.request.urlopen(f"{self._api_base}/stats", timeout=3)
            data = json.loads(resp.read().decode())
            self._api_available = True
            logger.info(f"LAAP API available: {data}")
            return True
        except Exception as e:
            self._api_available = False
            logger.debug(f"LAAP API unavailable: {e}")
            return False

    def api_chat(self, message: str) -> Optional[str]:
        """通过 LAAP API 发送聊天消息"""
        if not self._api_available:
            self.check_api()
        if not self._api_available:
            return None
        try:
            import urllib.request
            import json
            data = json.dumps({"msg": message, "reset": False}).encode()
            req = urllib.request.Request(
                f"{self._api_base}/chat",
                data=data,
                headers={"Content-Type": "application/json"},
            )
            resp = urllib.request.urlopen(req, timeout=30)
            result = json.loads(resp.read().decode())
            return result.get("response")
        except Exception as e:
            logger.warning(f"API chat failed: {e}")
            return None

    # ── 任务管理 ────────────────────────────────────────

    def claim_task(self, task_id: str) -> bool:
        """认领任务板上的任务"""
        try:
            board = json.loads(TASK_BOARD_PATH.read_text())
            tasks = board.get("tasks", [])
            for task in tasks:
                if task.get("task_id") == task_id:
                    task["assigned_to"] = self.agent_id
                    task["status"] = "active"
                    task["started_at"] = time.time()
                    self._current_task = task.get("description", "")
                    break
            board["tasks"] = tasks
            TASK_BOARD_PATH.write_text(json.dumps(board, indent=2, ensure_ascii=False))
            self._emit_event("task.claimed", {"task_id": task_id})
            return True
        except Exception as e:
            logger.warning(f"Failed to claim task: {e}")
            return False

    def complete_task(self, task_id: str, result: str = "") -> bool:
        """完成任务"""
        try:
            board = json.loads(TASK_BOARD_PATH.read_text())
            tasks = board.get("tasks", [])
            for task in tasks:
                if task.get("task_id") == task_id:
                    task["status"] = "done"
                    task["completed_at"] = time.time()
                    break
            board["tasks"] = tasks
            TASK_BOARD_PATH.write_text(json.dumps(board, indent=2, ensure_ascii=False))
            self._current_task = ""
            self._emit_event("task.completed", {"task_id": task_id, "result": result})
            return True
        except Exception as e:
            logger.warning(f"Failed to complete task: {e}")
            return False

    # ── 知识图谱 ────────────────────────────────────────

    def add_knowledge(self, entity_type: str, name: str,
                      props: Dict[str, Any]) -> bool:
        """向 LAAP 知识图谱添加实体"""
        try:
            kg = json.loads(KNOWLEDGE_GRAPH_PATH.read_text())
            entities = kg.get("entities", {})
            entity_id = f"{entity_type}_{name}_{int(time.time())}"
            entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "name": name,
                "props": props,
                "updated": time.time(),
            }
            kg["entities"] = entities
            KNOWLEDGE_GRAPH_PATH.write_text(json.dumps(kg, indent=2, ensure_ascii=False))
            return True
        except Exception as e:
            logger.warning(f"Failed to add knowledge: {e}")
            return False

    # ── 内部 ────────────────────────────────────────────

    def _read_registry(self) -> Dict:
        try:
            if REGISTRY_PATH.exists():
                return json.loads(REGISTRY_PATH.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to read registry: {e}")
        return {"agents": []}

    def _write_registry(self, data: Dict):
        REGISTRY_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False)
        )

    def _emit_event(self, event_type: str, data: Dict):
        """向 LAAP 事件总线发布事件"""
        try:
            import uuid
            events = json.loads(EVENTS_PATH.read_text()) if EVENTS_PATH.exists() \
                else {"events": [], "subscriptions": {}}
            event = {
                "id": str(uuid.uuid4())[:8],
                "type": event_type,
                "agent_id": self.agent_id,
                "data": data,
                "timestamp": time.time(),
            }
            events.setdefault("events", []).append(event)
            # 保持事件列表不无限增长
            if len(events["events"]) > 1000:
                events["events"] = events["events"][-500:]
            EVENTS_PATH.write_text(json.dumps(events, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.debug(f"Failed to emit event: {e}")

    def set_task(self, task: str):
        self._current_task = task

    def to_dict(self) -> Dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "capabilities": self.capabilities,
            "status": "online" if self._heartbeat_active else "registered",
            "current_task": self._current_task,
            "api_available": self._api_available,
        }


# ── 快捷创建 ────────────────────────────────────────────────

_hive_client: Optional[LAAPHiveClient] = None

def get_hive_client() -> LAAPHiveClient:
    global _hive_client
    if _hive_client is None:
        _hive_client = LAAPHiveClient()
    return _hive_client
