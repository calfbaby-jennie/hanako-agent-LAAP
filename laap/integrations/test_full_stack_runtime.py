from laap.integrations.full_stack_runtime import FullStackRuntime


def runtime(tmp_path):
    return FullStackRuntime(tmp_path)


def test_action_gate_allows_read(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action("read", {"path": "/tmp/a"}, trace_id="trace-1")
    assert result["allowed"] is True
    assert result["decision"] == "allow"
    assert result["trace_id"] == "trace-1"


def test_action_gate_requires_turn_authority_for_mutation(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action("edit", {"path": "/workspace/a"}, user_intent="看看这个文件")
    assert result["allowed"] is False
    assert result["decision"] == "confirm"
    assert result["reason"] == "mutation_not_authorized_for_task"


def test_action_gate_delegates_explicit_workspace_write_to_hana_scope(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action("edit", {"path": "/workspace/a"}, user_intent="修改这个文件")
    assert result["allowed"] is True
    assert result["decision"] == "restricted"
    assert result["host_guard_required"] is True


def test_action_gate_recognizes_explicit_resolution_authority(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action(
        "edit", {"path": "/workspace/a"},
        user_intent="递归修复直到真实回复可见并解决问题",
    )
    assert result["allowed"] is True
    assert result["decision"] == "restricted"
    assert result["explicit_user_authority"] is True


def test_action_gate_does_not_treat_resolution_discussion_as_authority(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action(
        "edit", {"path": "/workspace/a"},
        user_intent="分析这个问题为什么没有解决",
    )
    assert result["allowed"] is False
    assert result["decision"] == "confirm"


def test_action_gate_allows_conservative_read_only_shell_without_authority(tmp_path):
    rt = runtime(tmp_path)
    for command in (
        "git status --short --branch",
        "docker info --format '{{.ServerVersion}}'",
        "lsof -nP -iTCP -sTCP:LISTEN | grep 11521",
        "curl -fsS http://127.0.0.1:11521/health",
        "launchctl print gui/501/com.aris.laap-bridge",
    ):
        result = rt.evaluate_action("exec_command", {"cmd": command}, user_intent="检查状态")
        assert result["allowed"] is True, command
        assert result["decision"] == "allow", command
        assert result["reason"] == "read_only_shell", command


def test_action_gate_does_not_misclassify_mutating_shell_as_read_only(tmp_path):
    rt = runtime(tmp_path)
    for command in (
        "git branch new-branch",
        "curl -X POST http://127.0.0.1/action",
        "sed -i '' 's/a/b/' file.txt",
        "cat source > target",
        "find . -delete",
        "find . -exec rm {} ;",
        "env python3 repair.py",
        "git branch -D old-branch",
        "plutil -replace Key -string value file.plist",
    ):
        result = rt.evaluate_action("exec_command", {"cmd": command}, user_intent="检查状态")
        assert result["allowed"] is False, command
        assert result["decision"] == "confirm", command


def test_action_gate_persists_scoped_task_authority(tmp_path):
    first = runtime(tmp_path)
    granted = first.evaluate_action(
        "edit", {"path": "/workspace/a"}, session_id="session-1",
        user_intent="授权你自主完成这四项的修复",
    )
    assert granted["allowed"] is True
    restored = FullStackRuntime(tmp_path)
    continued = restored.evaluate_action(
        "exec_command", {"cmd": "python3 repair.py"}, session_id="session-1",
        user_intent="继续处理",
    )
    assert continued["allowed"] is True
    assert continued["continuing_authority"] is True
    unrelated = restored.evaluate_action(
        "exec_command", {"cmd": "python3 repair.py"}, session_id="session-2",
        user_intent="看看情况",
    )
    assert unrelated["allowed"] is False


def test_action_gate_revokes_persisted_authority(tmp_path):
    rt = runtime(tmp_path)
    rt.evaluate_action(
        "edit", {"path": "/workspace/a"}, session_id="session-1",
        user_intent="授权你自主完成修复",
    )
    revoked = rt.evaluate_action(
        "exec_command", {"cmd": "python3 repair.py"}, session_id="session-1",
        user_intent="暂停自主执行并撤销授权",
    )
    assert revoked["allowed"] is False


def test_action_gate_blocks_destructive_shell(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action("exec_command", {"cmd": "rm -rf /"}, user_intent="运行测试")
    assert result["allowed"] is False
    assert result["decision"] == "deny"
    assert result["reason"] == "destructive_shell_pattern"


def test_action_gate_external_side_effect_keeps_host_guard(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action("notify", {"body": "hello"}, user_intent="发送通知")
    assert result["allowed"] is True
    assert result["decision"] == "restricted"
    assert result["host_guard_required"] is True
    assert result["reason"] == "hana_external_action_safety_required"


def test_action_audit_redacts_secrets_and_restores(tmp_path):
    rt = runtime(tmp_path)
    result = rt.evaluate_action(
        "browser", {"api_key": "secret", "url": "https://example.com"},
        user_intent="打开网页", trace_id="trace-redact",
    )
    assert result["args"]["api_key"] == "<redacted>"
    restored = FullStackRuntime(tmp_path)
    assert restored.recent_audit(1)[0]["trace_id"] == "trace-redact"


def test_diagnostic_loopback_does_not_claim_real_peripheral_readiness(tmp_path):
    rt = runtime(tmp_path)
    result = rt.verify_peripheral_loopback({"probe": 1})
    assert result["ok"] is True
    assert result["diagnostic_only"] is True
    assert rt.status({})["layers"]["peripheral"]["ready"] is False


def test_real_hana_turn_drives_planner_task_body_events(tmp_path):
    rt = runtime(tmp_path)
    result = rt.record_turn(
        "分析项目并测试", "已完成", [{"toolName": "read"}],
        session_id="s1", turn_id="t1", trace_id="trace-turn",
    )
    assert result["ok"] is True
    assert result["kind"] == "real_hana_turn"
    assert result["record"]["plan_completed"] is True
    assert result["record"]["task_status"] == "completed"
    assert set(result["record"]["events"]) == {
        "conversation.turn", "planner.plan", "task.completed", "body.observation"
    }
    status = rt.status({})
    assert status["layers"]["peripheral"]["ready"] is True
    assert status["layers"]["peripheral"]["planner"]["total_plans"] == 1


def test_two_worker_multiagent_roundtrip_and_failure_isolation(tmp_path):
    rt = runtime(tmp_path)
    result = rt.verify_multiagent("verify integration", probe_failure=True)
    assert result["ok"] is True
    assert result["failure_isolated"] is True
    assert {item["role"] for item in result["results"]} == {"researcher", "verifier"}
    assert result["messages_processed"]["researcher"] >= 1
    assert result["messages_processed"]["verifier"] >= 1
    assert result["messages_processed"]["coordinator"] >= 3
    assert result["errors"]["fault-probe"] >= 1
    assert result["colony_deliveries"] == [1, 1]
    assert {item["peer"] for item in result["colony_receipts"]} == {"researcher", "verifier"}


def test_evolution_evidence_requires_full_lifecycle_and_retention_decision(tmp_path):
    rt = runtime(tmp_path)
    rt.rsi_audit_path = tmp_path / "rsi_audit.jsonl"
    proposal = "true_rsi_test"
    states = [
        ("proposed", "patched"),
        ("patched", "sandbox_passed"),
        ("sandbox_passed", "approved"),
        ("approved", "deployed"),
        ("deployed", "rolled_back"),
    ]
    rt.rsi_audit_path.write_text("".join(
        __import__("json").dumps({"proposal_id": proposal, "from_status": a, "to_status": b}) + "\n"
        for a, b in states
    ))
    evidence = rt.evolution_evidence()
    assert evidence["code_cycle_verified"] is True
    assert evidence["retain_or_rollback_verified"] is True
    assert proposal in evidence["rollback_proven"]


def test_status_needs_all_four_real_layers(tmp_path):
    rt = runtime(tmp_path)
    rt.rsi_audit_path = tmp_path / "rsi_audit.jsonl"
    rt.rsi_audit_path.write_text("\n".join([
        '{"proposal_id":"p","to_status":"sandbox_passed"}',
        '{"proposal_id":"p","to_status":"deployed"}',
        '{"proposal_id":"p","to_status":"rolled_back"}',
    ]) + "\n")
    rt.evaluate_action("read", {"path": "/tmp/a"}, trace_id="trace")
    rt.record_turn("分析", "完成", [], trace_id="trace")
    rt.verify_multiagent(probe_failure=True)
    status = rt.status()
    assert status["ready"] is True
    assert all(layer["ready"] for layer in status["layers"].values())
