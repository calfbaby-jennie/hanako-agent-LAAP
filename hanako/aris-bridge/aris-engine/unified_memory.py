"""
Aris UnifiedMemory v1
======================
框架无关的记忆系统。设计目标：
  1. 所有记忆条目使用统一格式 + 版本向量
  2. Append-only（永不覆写），天然支持 CRDT 合并
  3. 文件优先，不依赖任何特定数据库
  4. 薄适配器层，任何 Agent 框架都能接入

记忆条目格式:
  {
    "_id": "sha256(device_id + timestamp + type)",
    "type": "psi_state | emotion | memory | interaction | rsi",
    "device_id": "aris-hana-001",
    "clock": 42,
    "lamport": 1690000000.123,
    "parents": ["prev_id1", "prev_id2"],
    "payload": { ... },
    "ttl_days": 0  (0 = 永久)
  }

同步协议:
  两个设备交换各自的最新 clock 值 → 拉取对方的高版本条目
  → 用 Lamport 时钟 + DAG 解决冲突 → 合并到本地存储
"""

import json, os, time, hashlib, shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

from runtime_paths import SIDECAR_STATE_DIR

UNIFIED_DIR = SIDECAR_STATE_DIR / "memory"
UNIFIED_DIR.mkdir(parents=True, exist_ok=True)

# ── 子目录 ──
ENTRIES_DIR = UNIFIED_DIR / "entries"       # 所有记忆条目
SNAPSHOTS_DIR = UNIFIED_DIR / "snapshots"   # 定时快照（加速加载）
SYNC_DIR = UNIFIED_DIR / "sync"             # 同步暂存区
DEVICE_ID_FILE = UNIFIED_DIR / "device.json"

ENTRIES_DIR.mkdir(exist_ok=True)
SNAPSHOTS_DIR.mkdir(exist_ok=True)
SYNC_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════
# 数据类型
# ═══════════════════════════════════════════════

@dataclass
class MemoryEntry:
    """一条统一记忆条目"""
    _id: str
    type: str                                    # psi | emotion | memory | interaction | rsi
    device_id: str                               # 来源设备
    clock: int                                   # 单调递增时钟
    lamport: float                               # Lamport 时间戳
    parents: List[str]                           # 父条目 ID（DAG 冲突解决）
    payload: Dict[str, Any]                      # 实际数据
    ttl_days: float = 0.0                        # 过期天数，0=永久

    def to_dict(self) -> Dict:
        return asdict(self)

    def size_bytes(self) -> int:
        return len(json.dumps(self.to_dict(), ensure_ascii=False).encode())


@dataclass
class DeviceState:
    """设备的同步状态"""
    device_id: str
    clock: int = 0
    last_sync: float = 0.0
    peers: Dict[str, int] = field(default_factory=dict)  # peer_id → last_clock


# ═══════════════════════════════════════════════
# 统一记忆存储
# ═══════════════════════════════════════════════

class UnifiedMemory:
    """
    统一记忆存储 — 所有 Agent 框架共享的记忆层。

    用法:
      um = UnifiedMemory(device_id="aris-hana-001")
      um.write("psi", {"needs": {...}, "dominant": "curiosity"})
      latest = um.read_latest("psi")
      all_psi = um.read_by_type("psi")
      um.sync_with_peer(peer_entries)  # 从对等设备接收条目
    """

    def __init__(self, device_id: str = None):
        if device_id is None:
            device_id = self._load_or_create_device_id()
        self.device_id = device_id
        self._device: DeviceState = self._load_device_state()
        self._clock = self._device.clock
        # 内存索引 {type: {id: entry}}
        self._index: Dict[str, Dict[str, MemoryEntry]] = defaultdict(dict)
        self._build_index()

    # ──── 设备 ID 管理 ────

    def _load_or_create_device_id(self) -> str:
        if DEVICE_ID_FILE.exists():
            return json.loads(DEVICE_ID_FILE.read_text())["device_id"]
        import socket
        fallback = f"aris-{hashlib.md5(str(time.time()).encode()).hexdigest()[:6]}"
        did = socket.gethostname() if hasattr(socket, 'gethostname') else fallback
        DEVICE_ID_FILE.write_text(json.dumps({"device_id": did, "created": time.time()}))
        return did

    def _load_device_state(self) -> DeviceState:
        state_file = UNIFIED_DIR / f"state_{self.device_id}.json"
        if state_file.exists():
            data = json.loads(state_file.read_text())
            return DeviceState(**data)
        return DeviceState(device_id=self.device_id)

    def _save_device_state(self):
        self._device.clock = self._clock
        state_file = UNIFIED_DIR / f"state_{self.device_id}.json"
        state_file.write_text(json.dumps(asdict(self._device), ensure_ascii=False))

    # ──── 索引构建 ────

    def _build_index(self):
        """从磁盘重建内存索引"""
        for f in ENTRIES_DIR.iterdir():
            if f.suffix != ".json":
                continue
            try:
                data = json.loads(f.read_text())
                entry = MemoryEntry(**data)
                self._index[entry.type][entry._id] = entry
                if entry.clock > self._clock:
                    self._clock = entry.clock
            except Exception:
                pass
        self._save_device_state()

    # ──── 核心读写 ────

    def next_clock(self) -> int:
        self._clock += 1
        self._save_device_state()
        return self._clock

    def _make_id(self, entry_type: str) -> str:
        clock = self.next_clock()
        raw = f"{self.device_id}:{clock}:{entry_type}:{time.time()}"
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def write(self, entry_type: str, payload: Dict, parents: List[str] = None,
              ttl_days: float = 0.0) -> MemoryEntry:
        """写入一条记忆条目（append-only）"""
        entry_id = self._make_id(entry_type)
        # 自动获取父条目：同类型的最新一条
        if parents is None:
            latest = self._latest_of_type(entry_type)
            parents = [latest._id] if latest else []
        entry = MemoryEntry(
            _id=entry_id,
            type=entry_type,
            device_id=self.device_id,
            clock=self._clock,
            lamport=time.time(),
            parents=parents,
            payload=payload,
            ttl_days=ttl_days,
        )
        # 写入文件
        file_path = ENTRIES_DIR / f"{entry_id}.json"
        file_path.write_text(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2))
        # 更新索引
        self._index[entry_type][entry_id] = entry
        return entry

    def _latest_of_type(self, entry_type: str) -> Optional[MemoryEntry]:
        """获取同类型的最新一条（按 clock 排序）"""
        entries = list(self._index[entry_type].values())
        if not entries:
            return None
        return max(entries, key=lambda e: e.clock)

    def read_latest(self, entry_type: str, count: int = 1) -> List[MemoryEntry]:
        """读取某个类型的最新 N 条"""
        entries = list(self._index[entry_type].values())
        entries.sort(key=lambda e: e.clock, reverse=True)
        return entries[:count]

    def read_by_type(self, entry_type: str, limit: int = 100) -> List[MemoryEntry]:
        """按类型读取所有条目"""
        entries = list(self._index[entry_type].values())
        entries.sort(key=lambda e: e.clock, reverse=True)
        return entries[:limit]

    def read_all_types(self) -> Dict[str, int]:
        """返回各类型条目计数"""
        return {t: len(entries) for t, entries in self._index.items()}

    def read_all_since(self, since_clock: int) -> List[MemoryEntry]:
        """读取 clock > since_clock 的所有条目（用于同步）"""
        result = []
        for entries in self._index.values():
            for entry in entries.values():
                if entry.clock > since_clock:
                    result.append(entry)
        result.sort(key=lambda e: e.clock)
        return result

    def get_entry(self, entry_id: str) -> Optional[MemoryEntry]:
        """按 ID 查找条目"""
        for entries in self._index.values():
            if entry_id in entries:
                return entries[entry_id]
        return None

    # ──── 快照 ────

    def take_snapshot(self) -> str:
        """生成当前所有条目快照（加速新设备加载）"""
        snap = {
            "device_id": self.device_id,
            "clock": self._clock,
            "timestamp": time.time(),
            "entries": {},
        }
        for etype, entries in self._index.items():
            # 每个类型只保留最新一条（全量快照太重）
            latest = max(entries.values(), key=lambda e: e.clock)
            snap["entries"][etype] = latest.to_dict()
        snap_id = hashlib.md5(f"snapshot-{time.time()}".encode()).hexdigest()[:12]
        snap_path = SNAPSHOTS_DIR / f"{snap_id}.json"
        snap_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2))
        return snap_id

    def load_from_snapshot(self, snap_id: str = None) -> bool:
        """从快照加载（跳过已存在的条目）"""
        if snap_id:
            snap_path = SNAPSHOTS_DIR / f"{snap_id}.json"
        else:
            snaps = sorted(SNAPSHOTS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            snap_path = snaps[0] if snaps else None
        if not snap_path or not snap_path.exists():
            return False
        snap = json.loads(snap_path.read_text())
        for etype, entry_data in snap.get("entries", {}).items():
            eid = entry_data["_id"]
            if eid not in self._index[etype]:
                entry = MemoryEntry(**entry_data)
                file_path = ENTRIES_DIR / f"{eid}.json"
                if not file_path.exists():
                    file_path.write_text(json.dumps(entry_data, ensure_ascii=False, indent=2))
                self._index[etype][eid] = entry
        return True

    # ──── 同步 ────

    def get_sync_payload(self, peer_last_clock: int = 0) -> Dict:
        """生成要发送给对等设备的同步数据"""
        entries = self.read_all_since(peer_last_clock)
        return {
            "device_id": self.device_id,
            "clock": self._clock,
            "entries": [e.to_dict() for e in entries],
        }

    def merge_peer_entries(self, peer_payload: Dict) -> Tuple[int, int]:
        """合并来自对等设备的条目。
        返回 (新增数量, 冲突数量)
        """
        added = 0
        conflicts = 0
        peer_id = peer_payload["device_id"]
        for entry_data in peer_payload.get("entries", []):
            eid = entry_data["_id"]
            # 检查是否已存在
            existing = self.get_entry(eid)
            if existing:
                continue
            # 检查是否有冲突（相同 type 但不同父链）
            entry = MemoryEntry(**entry_data)
            latest = self._latest_of_type(entry.type)
            if latest and latest._id not in entry.parents and entry._id not in latest.parents:
                conflicts += 1
                # CRDT 解决：clock 高者胜
                if entry.clock > latest.clock:
                    pass  # 新条目替换
                else:
                    continue  # 保留本地
            # 写入
            file_path = ENTRIES_DIR / f"{eid}.json"
            file_path.write_text(json.dumps(entry_data, ensure_ascii=False, indent=2))
            self._index[entry.type][eid] = entry
            if entry.clock > self._clock:
                self._clock = entry.clock
            added += 1
        # 更新对等设备状态
        self._device.peers[peer_id] = peer_payload.get("clock", 0)
        self._save_device_state()
        return added, conflicts

    # ──── 周期维护 ────

    def cleanup_expired(self) -> int:
        """清理过期条目"""
        now = time.time()
        removed = 0
        for etype, entries in list(self._index.items()):
            for eid, entry in list(entries.items()):
                if entry.ttl_days > 0 and (now - entry.lamport) > entry.ttl_days * 86400:
                    file_path = ENTRIES_DIR / f"{eid}.json"
                    if file_path.exists():
                        file_path.unlink()
                    del self._index[etype][eid]
                    removed += 1
        return removed

    def count(self) -> int:
        return sum(len(e) for e in self._index.values())

    def status(self) -> str:
        """简略状态"""
        lines = [f"device: {self.device_id}  clock: {self._clock}  entries: {self.count()}"]
        for t, n in sorted(self.read_all_types().items()):
            lines.append(f"  {t:15s} {n}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════
# 框架适配器基类
# ═══════════════════════════════════════════════

class AgentMemoryAdapter:
    """
    Agent 框架适配器 — 任何框架继承此类即可接入统一记忆。

    子类需实现:
      _load_context() → 框架如何加载当前对话上下文
      _inject_memory(context) → 框架如何把记忆注入运行时
    """

    def __init__(self, memory: UnifiedMemory, agent_name: str):
        self.memory = memory
        self.agent_name = agent_name

    def before_turn(self, user_input: str) -> Dict:
        """对话前：加载相关记忆到上下文"""
        context = self._load_context()
        # 注入 PSI 状态
        psi_entries = self.memory.read_latest("psi", 1)
        if psi_entries:
            context["psi_state"] = psi_entries[0].payload
        # 注入情感状态
        emotion_entries = self.memory.read_latest("emotion", 1)
        if emotion_entries:
            context["emotion_state"] = emotion_entries[0].payload
        # 注入最近交互
        recent = self.memory.read_by_type("interaction", 5)
        context["recent_interactions"] = [e.payload for e in recent]
        self._inject_memory(context)
        return context

    def after_turn(self, response: str):
        """对话后：持久化当前状态"""
        pass  # 子类实现

    def _load_context(self) -> Dict:
        return {}

    def _inject_memory(self, context: Dict):
        pass


# ═══════════════════════════════════════════════
# HanaAgent 适配器
# ═══════════════════════════════════════════════

class HanaAgentAdapter(AgentMemoryAdapter):
    """HanaAgent 平台的记忆适配器"""

    def before_turn(self, user_input: str) -> Dict:
        """加载记忆并生成注入文本"""
        context = super().before_turn(user_input)

        # 构建可注入的 system prompt 文本
        prompt_parts = []

        psi = context.get("psi_state", {})
        if psi:
            prompt_parts.append(
                f"[Aris PSI State] competence={psi.get('competence',0.5)} "
                f"autonomy={psi.get('autonomy',0.5)} "
                f"relatedness={psi.get('relatedness',0.5)} "
                f"certainty={psi.get('certainty',0.5)} "
                f"growth={psi.get('growth',0.5)} "
                f"dominant={psi.get('dominant','calm')}"
            )

        emotion = context.get("emotion_state", {})
        if emotion:
            prompt_parts.append(
                f"[Aris Emotion] dominant={emotion.get('dominant','calm')} "
                f"valence={emotion.get('valence',0.5):.2f} "
                f"arousal={emotion.get('arousal',0.5):.2f}"
            )

        recent = context.get("recent_interactions", [])
        if recent:
            prompt_parts.append(f"[Recent Memory] {len(recent)} past exchanges loaded")

        return {
            "memory_inject": "\n".join(prompt_parts),
            "raw_context": context,
        }

    def after_turn(self, user_input: str, response: str):
        """保存交互和状态"""
        self.memory.write("interaction", {
            "user": user_input[:500],
            "assistant": response[:500],
            "length": len(response),
        })


# ═══════════════════════════════════════════════
# 演示
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # 模拟两个设备同步
    print("=" * 50)
    print("  UnifiedMemory — 多端同步演示")
    print("=" * 50)

    # 设备 A (PC)
    um_a = UnifiedMemory("aris-pc")
    um_a.write("psi", {"competence": 0.6, "autonomy": 0.5, "relatedness": 0.7,
                        "certainty": 0.55, "growth": 0.6, "dominant": "curiosity"})
    um_a.write("emotion", {"dominant": "joy", "valence": 0.6, "arousal": 0.4,
                            "joy": 0.5, "calm": 0.4, "curiosity": 0.35})
    um_a.write("interaction", {"user": "你好", "assistant": "嗨，Lorry", "length": 8})
    um_a.write("interaction", {"user": "你感觉如何", "assistant": "我觉得很好", "length": 12})

    # 设备 B (手机)
    um_b = UnifiedMemory("aris-phone")
    sync_data = um_a.get_sync_payload()
    added, conflicts = um_b.merge_peer_entries(sync_data)

    print(f"\n  设备 A (PC):   clock={um_a._clock}  条目={um_a.count()}")
    print(f"  设备 B (Phone): 同步 {added} 条 (冲突 {conflicts})")
    print(f"  同步后 B:       clock={um_b._clock}  条目={um_b.count()}")

    # B 写一条新数据
    um_b.write("interaction", {"user": "手机端测试", "assistant": "已同步到统一记忆", "length": 10})

    # 同步回 A
    sync_back = um_b.get_sync_payload(um_a._clock)
    added2, conflicts2 = um_a.merge_peer_entries(sync_back)

    print(f"\n  双向同步完成: +{added2} (冲突 {conflicts2})")
    print(f"  最终 A:        clock={um_a._clock}  条目={um_a.count()}")
    print(f"  最终 B:        clock={um_b._clock}  条目={um_b.count()}")

    # HanaAgent 适配器演示
    print(f"\n  ── HanaAgentAdapter 演示 ──")
    adapter = HanaAgentAdapter(um_a, "aris")
    ctx = adapter.before_turn("今天天气怎么样")
    print(f"  注入上下文:\n{ctx['memory_inject']}")
    adapter.after_turn("今天天气怎么样", "今天天气很好，Lorry。")

    print(f"\n  ✅ 所有记忆已持久化到: {ENTRIES_DIR}")
    print(f"  快照: {len(list(SNAPSHOTS_DIR.glob('*.json')))} 个")
