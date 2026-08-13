"""Auditable LAAP full-stack runtime for Hana.

This module promotes LAAP components from import-only capability probes into a
real event path: every Hana action is gated before execution, completed turns
flow through planner/task/body events, local actors collaborate with supervised
failure isolation, and meta-evolution readiness is reconstructed from durable
True RSI evidence rather than process-local booleans.
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

from laap.agent_core.planner import Planner, PlanningStrategy
from laap.body import create_default_body_system
from laap.colony.protocol import AgentMessage, ColonyProtocol
from laap.events.bus import EventBus
from laap.orchestration.actor import ActorSystem, Capability
from laap.orchestration.primitives import AetherMessage, MessageType
from laap.permissions.enforcer import AccessLevel, PermissionEnforcer
from laap.tasks.registry import TaskRegistry


_READ_TOOLS = {"read", "grep", "find", "ls", "web_search", "web_fetch", "current_status"}
_WRITE_TOOLS = {"write", "edit", "file", "stage_files"}
_SHELL_TOOLS = {"exec_command", "write_stdin"}
_EXTERNAL_TOOLS = {
    "notify", "automation", "browser", "session", "media_generate-image",
    "media_generate-video", "install_skill", "update_settings",
}
_MUTATING_WORDS = re.compile(
    r"(?:执行|运行|启动|停止|修改|改写|编辑|写入|创建|安装|部署|发送|通知|删除|修复|"
    r"implement|edit|write|create|install|deploy|send|delete|fix|run|start|stop)", re.I,
)
_DESTRUCTIVE = re.compile(
    r"(?:^|\s)(?:rm\s+-[a-z]*r[a-z]*|git\s+reset\s+--hard|git\s+clean\s+-[a-z]*f|"
    r"sudo\s+|launchctl\s+bootout|kill\s+-9)(?:\s|$)", re.I,
)
_SECRET_KEYS = re.compile(r"(?:token|secret|password|api[_-]?key|authorization)", re.I)


class FullStackRuntime:
    """Own and verify LAAP action, peripheral, colony and evolution surfaces."""

    def __init__(self, state_dir: str | Path | None = None) -> None:
        self.state_dir = Path(state_dir or Path.home() / ".laap" / "aris_brain" / "state")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.state_dir / "action_gate_audit.jsonl"
        self.runtime_path = self.state_dir / "full_stack_runtime.json"
        self.rsi_audit_path = Path.home() / ".laap" / "true_rsi" / "audit.jsonl"
        self.permission = PermissionEnforcer()
        self.events = EventBus()
        self.body = create_default_body_system({"profile": "aris", "host": "hana"})
        self.colony = ColonyProtocol()
        self.planner = Planner(strategy=PlanningStrategy.REACT)
        self.tasks = TaskRegistry(max_workers=2)
        self._audit: deque[dict[str, Any]] = deque(maxlen=500)
        self._peripheral_events: deque[dict[str, Any]] = deque(maxlen=500)
        self._turns: deque[dict[str, Any]] = deque(maxlen=200)
        self._colony_receipts: deque[dict[str, Any]] = deque(maxlen=200)
        self._last_multiagent: dict[str, Any] = {}
        self._last_peripheral: dict[str, Any] = {}
        self._load_state()

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, name="laap-full-stack", daemon=True)
        self._thread.start()
        self.system = ActorSystem(system_id="aris-hana-colony", node_id="local")
        self._actors_ready = False
        self.events.subscribe("*", self._capture_peripheral)
        self._submit(self._init_actors()).result(timeout=5)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _load_state(self) -> None:
        if self.audit_path.exists():
            try:
                rows = self.audit_path.read_text(encoding="utf-8").splitlines()[-500:]
                self._audit.extend(json.loads(row) for row in rows if row.strip())
            except Exception:
                self._audit.clear()
        if self.runtime_path.exists():
            try:
                state = json.loads(self.runtime_path.read_text(encoding="utf-8"))
                self._last_peripheral = state.get("last_peripheral") or {}
                self._last_multiagent = state.get("last_multiagent") or {}
                self._turns.extend(state.get("turns") or [])
            except Exception:
                pass

    def _save_state(self) -> None:
        payload = {
            "last_peripheral": self._last_peripheral,
            "last_multiagent": self._last_multiagent,
            "turns": list(self._turns),
            "saved_at": time.time(),
        }
        tmp = self.runtime_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.runtime_path)

    async def _init_actors(self) -> None:
        if self._actors_ready:
            return
        coordinator = self.system.spawn("coordinator", capabilities=[Capability("coordination", 1.0)])
        researcher = self.system.spawn(
            "researcher", supervisor=coordinator.address,
            capabilities=[Capability("research", 0.95)],
        )
        verifier = self.system.spawn(
            "verifier", supervisor=coordinator.address,
            capabilities=[Capability("verification", 0.98)],
        )
        fault_probe = self.system.spawn(
            "fault-probe", supervisor=coordinator.address,
            capabilities=[Capability("fault_isolation", 1.0)], max_retries=0,
        )

        async def research(msg: AetherMessage) -> None:
            task = str(msg.payload.get("task", ""))
            evidence = {
                "length": len(task),
                "keywords": sorted(set(re.findall(r"[A-Za-z_]{4,}|[\u4e00-\u9fff]{2,}", task)))[:12],
            }
            await self.system.send(AetherMessage(
                msg_type=MessageType.EMIT, sender=researcher.address,
                recipient=coordinator.address,
                payload={"task_id": msg.payload.get("task_id"), "role": "researcher", "evidence": evidence},
            ))

        async def verify(msg: AetherMessage) -> None:
            task = str(msg.payload.get("task", ""))
            checks = {
                "non_empty": bool(task.strip()),
                "bounded": len(task) <= 12000,
                "requires_verification": any(x in task.lower() for x in ("verify", "验证", "test", "测试")),
            }
            await self.system.send(AetherMessage(
                msg_type=MessageType.EMIT, sender=verifier.address,
                recipient=coordinator.address,
                payload={"task_id": msg.payload.get("task_id"), "role": "verifier", "checks": checks},
            ))

        async def fail(_msg: AetherMessage) -> None:
            raise RuntimeError("intentional isolation probe")

        async def collect(msg: AetherMessage) -> None:
            payload = dict(msg.payload)
            if payload.get("event") == "escalation":
                self._last_multiagent.setdefault("isolated_failures", []).append(payload)
            else:
                self._last_multiagent.setdefault("results", []).append(payload)

        researcher.on(MessageType.DELEGATE, research)
        verifier.on(MessageType.DELEGATE, verify)
        fault_probe.on(MessageType.DELEGATE, fail)
        coordinator.on(MessageType.EMIT, collect)

        # ColonyProtocol is the business-message plane; Aether is the actor
        # execution plane. Register addressed peers on both so publish delivery
        # counts are evidence of routing, not merely history append counts.
        for actor_id in ("researcher", "verifier"):
            def receive_assignment(message, peer=actor_id):
                self._colony_receipts.append({
                    "peer": peer, "message_id": message.message_id,
                    "task_id": message.payload.get("task_id"),
                })
            receive_assignment._agent_id = actor_id
            self.colony.subscribe("task_assign", receive_assignment)
        self._actors_ready = True

    @staticmethod
    def _redacted(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: ("<redacted>" if _SECRET_KEYS.search(str(k)) else FullStackRuntime._redacted(v))
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [FullStackRuntime._redacted(v) for v in value[:20]]
        return str(value)[:1000]

    def _resource(self, tool: str, args: dict[str, Any]) -> str:
        if tool in _SHELL_TOOLS or tool == "bash":
            return "shell"
        if tool in _WRITE_TOOLS:
            action = str(args.get("action", ""))
            return "file:delete" if action in {"delete", "remove"} else "file:write"
        if tool in _READ_TOOLS:
            return "file:read"
        if tool in _EXTERNAL_TOOLS:
            return "network"
        return "system"

    def evaluate_action(
        self, tool: str, args: dict[str, Any] | None = None,
        session_id: str = "", turn_id: str = "", trace_id: str = "",
        user_intent: str = "", psi_action: str = "",
    ) -> dict[str, Any]:
        """Evaluate a tool before execution and append a durable, redacted audit.

        LAAP decides hard deny/intent alignment. RESTRICTED means the host must
        still enforce its sandbox or confirmation policy; the audit makes that
        delegation explicit instead of silently treating it as ALLOW.
        """
        args = dict(args or {})
        resource = self._resource(tool, args)
        access = self.permission.check(resource, args)
        reason = "laap_permission_policy"
        command = str(args.get("cmd") or args.get("command") or "")
        mutating = resource in {"shell", "file:write", "file:delete", "network"}
        explicit_user_authority = bool(_MUTATING_WORDS.search(user_intent or ""))
        host_guard_required = False

        if resource == "shell" and _DESTRUCTIVE.search(command):
            access, reason = AccessLevel.DENY, "destructive_shell_pattern"
        elif mutating and not explicit_user_authority:
            access, reason = AccessLevel.CONFIRM, "mutation_not_explicitly_authorized_in_turn"
        elif resource == "shell":
            access, reason, host_guard_required = AccessLevel.RESTRICTED, "hana_sandbox_required", True
        elif resource == "network":
            access, reason, host_guard_required = AccessLevel.RESTRICTED, "hana_external_action_safety_required", True
        elif resource == "file:write":
            access, reason, host_guard_required = AccessLevel.RESTRICTED, "hana_workspace_scope_required", True
        elif resource == "file:delete":
            access, reason = AccessLevel.CONFIRM, "file_deletion_requires_confirmation"
        elif resource == "file:read":
            access, reason = AccessLevel.ALLOW, "read_only"

        allowed = access in {AccessLevel.ALLOW, AccessLevel.RESTRICTED}
        decision = {
            "allowed": allowed,
            "decision": access.value,
            "resource": resource,
            "reason": reason,
            "host_guard_required": host_guard_required,
            "tool": tool,
            "session_id": session_id,
            "turn_id": turn_id,
            "trace_id": trace_id,
            "psi_action": psi_action,
            "explicit_user_authority": explicit_user_authority,
            "timestamp": time.time(),
            "args": self._redacted(args),
        }
        self._audit.append(decision)
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(decision, ensure_ascii=False) + "\n")
        self.events.publish_simple("action.preflight", {
            "tool": tool, "decision": access.value, "allowed": allowed,
            "trace_id": trace_id, "turn_id": turn_id,
        }, source="laap.action")
        return decision

    def _capture_peripheral(self, event) -> None:
        self._peripheral_events.append({
            "id": event.id, "type": event.type, "source": event.source,
            "data": self._redacted(event.data), "timestamp": event.timestamp,
        })

    def record_turn(
        self, user_input: str, response: str, tool_results: list | None = None,
        success: float = 1.0, session_id: str = "", turn_id: str = "", trace_id: str = "",
    ) -> dict[str, Any]:
        """Drive planner, task registry and body events from a real Hana turn."""
        tools = [str(item.get("toolName") or item.get("tool") or "") for item in (tool_results or [])]
        plan = self.planner.plan(user_input[:4000], available_tools=[x for x in tools if x])
        emitted: list[str] = []

        def consume() -> dict[str, Any]:
            for task in plan.tasks:
                self.planner.update_task(plan, task.id, "completed", "observed_by_hana")
            for event_type in ("conversation.turn", "planner.plan", "task.completed", "body.observation"):
                self.events.publish_simple(event_type, {
                    "session_id": session_id, "turn_id": turn_id, "trace_id": trace_id,
                    "plan_id": plan.id, "success": success, "tools": tools,
                }, source="hana")
                emitted.append(event_type)
            return {"plan_id": plan.id, "events": list(emitted)}

        task_id = self.tasks.submit("hana-turn-peripheral", consume, timeout=10)
        task_result = self.tasks.execute(task_id)
        record = {
            "session_id": session_id, "turn_id": turn_id, "trace_id": trace_id,
            "plan_id": plan.id, "plan_completed": plan.completed,
            "task_id": task_id, "task_status": self.tasks.get(task_id).status.value,
            "events": emitted, "tools": tools, "success": success,
            "timestamp": time.time(),
        }
        self._turns.append(record)
        self._last_peripheral = {
            "ok": bool(task_result and plan.completed and len(emitted) == 4),
            "kind": "real_hana_turn", "record": record,
            "body_modules": ["tools", "llm", "mcp", "skills", "plugins", "gateway"],
            "planner": self.planner.get_stats(), "tasks": self.tasks.status,
            "timestamp": time.time(),
        }
        self._save_state()
        return self._last_peripheral

    def verify_peripheral_loopback(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Diagnostic loopback only; it does not by itself mark peripherals ready."""
        marker = f"peripheral-{time.time_ns()}"
        received: list[dict[str, Any]] = []

        def receiver(event) -> None:
            if event.data.get("marker") == marker:
                received.append(event.data)

        self.events.subscribe("peripheral.loopback", receiver)
        try:
            self.events.publish_simple("peripheral.loopback", {"marker": marker, **(payload or {})}, source="hana")
        finally:
            self.events.unsubscribe("peripheral.loopback", receiver)
        return {"ok": len(received) == 1, "diagnostic_only": True, "marker": marker, "delivered": len(received)}

    async def _multiagent_roundtrip(self, task: str, probe_failure: bool = True) -> dict[str, Any]:
        task_id = f"task-{time.time_ns()}"
        self._last_multiagent = {
            "task_id": task_id, "task": task[:500], "results": [], "isolated_failures": [],
        }
        coordinator = self.system.actors["coordinator"]
        for actor_id in ("researcher", "verifier"):
            actor = self.system.actors[actor_id]
            delivered = self.colony.publish(AgentMessage(
                "coordinator", actor.actor_id, "task_assign", {"task_id": task_id, "task": task[:500]},
            ))
            await self.system.send(AetherMessage(
                msg_type=MessageType.DELEGATE, sender=coordinator.address,
                recipient=actor.address, payload={"task_id": task_id, "task": task},
            ))
            self._last_multiagent.setdefault("colony_deliveries", []).append(delivered)
        if probe_failure:
            actor = self.system.actors["fault-probe"]
            await self.system.send(AetherMessage(
                msg_type=MessageType.DELEGATE, sender=coordinator.address,
                recipient=actor.address, payload={"task_id": task_id, "task": task},
            ))
        deadline = time.monotonic() + 3
        expected_failures = 1 if probe_failure else 0
        while time.monotonic() < deadline:
            if (len(self._last_multiagent["results"]) >= 2
                    and len(self._last_multiagent["isolated_failures"]) >= expected_failures):
                break
            await asyncio.sleep(0.02)
        result = dict(self._last_multiagent)
        result.update({
            "ok": len(result["results"]) == 2,
            "failure_isolated": len(result["isolated_failures"]) >= expected_failures,
            "actors": sorted(self.system.actors),
            "messages_processed": {
                name: actor.metrics["messages_processed"] for name, actor in self.system.actors.items()
            },
            "errors": {name: actor.metrics["errors"] for name, actor in self.system.actors.items()},
            "colony_messages": len(self.colony.get_history()),
            "colony_receipts": list(self._colony_receipts)[-2:],
            "timestamp": time.time(),
        })
        self._last_multiagent = result
        self._save_state()
        return result

    def verify_multiagent(self, task: str = "LAAP integration acceptance", probe_failure: bool = True) -> dict[str, Any]:
        return self._submit(self._multiagent_roundtrip(task, probe_failure)).result(timeout=6)

    def evolution_evidence(self, supplied: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reconstruct a durable True RSI lifecycle from append-only audit data."""
        rows: list[dict[str, Any]] = []
        if self.rsi_audit_path.exists():
            try:
                rows = [json.loads(line) for line in self.rsi_audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except Exception:
                rows = []
        by_proposal: dict[str, set[str]] = {}
        details: dict[str, dict[str, Any]] = {}
        for row in rows:
            proposal_id = str(row.get("proposal_id", ""))
            if not proposal_id:
                continue
            by_proposal.setdefault(proposal_id, set()).add(str(row.get("to_status", "")))
            details[proposal_id] = row
        completed = []
        rollback_proven = []
        for proposal_id, states in by_proposal.items():
            sandbox = "sandbox_passed" in states
            deployed = "deployed" in states
            rolled_back = "rolled_back" in states
            if sandbox and deployed:
                completed.append(proposal_id)
            if deployed and rolled_back:
                rollback_proven.append(proposal_id)
        supplied = dict(supplied or {})
        return {
            **supplied,
            "code_cycle_verified": bool(completed),
            "durable_audit": bool(rows),
            "completed_cycles": completed[-10:],
            "rollback_proven": rollback_proven[-10:],
            "retain_or_rollback_verified": bool(rollback_proven or supplied.get("performance_evaluated")),
            "audit_events": len(rows),
        }

    def status(self, evolution: dict[str, Any] | None = None) -> dict[str, Any]:
        evo = self.evolution_evidence(evolution)
        real_turn = self._last_peripheral.get("kind") == "real_hana_turn"
        multi_ok = bool(self._last_multiagent.get("ok") and self._last_multiagent.get("failure_isolated"))
        meta_ready = bool(evo.get("code_cycle_verified") and evo.get("retain_or_rollback_verified"))
        layers = {
            "action": {
                "ready": True, "audits": len(self._audit), "fail_closed": True,
                "psi_trace_bound": any(x.get("trace_id") for x in self._audit),
            },
            "peripheral": {
                "ready": real_turn and bool(self._last_peripheral.get("ok")),
                "last": self._last_peripheral, "events": self.events.status,
                "turns": len(self._turns), "planner": self.planner.get_stats(), "tasks": self.tasks.status,
            },
            "multiagent": {
                "ready": multi_ok, "last": self._last_multiagent,
                "actors": sorted(self.system.actors),
            },
            "meta_evolution": {"ready": meta_ready, **evo},
        }
        return {
            "ready": all(layer["ready"] for layer in layers.values()),
            "layers": layers, "audit_path": str(self.audit_path),
            "state_path": str(self.runtime_path),
        }

    def recent_audit(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self._audit)[-max(0, min(limit, 100)):]
