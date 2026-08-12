from laap.integrations.hana_psi_contract import (
    REDACTION_MARKER,
    deterministic_grounding_redaction,
    evaluate_preflight,
    evaluate_review,
    integration_fitness,
)


def healthy_preflight():
    layers = {name: True for name in (
        "perception", "decision", "memory", "self", "motivation",
        "symbolic", "grounding", "transport", "evolution",
    )}
    return {
        "ok": True,
        "context": (
            "pipeline.complete=true\nao.intent=analyze\nao.trace=t1\n"
            "organs=all:on\nhealth.layers=all:ok\ndirective=safe"
        ),
        "health": {"layers": layers, "degraded": []},
        "degraded": [],
        "perception": {"blocked": False},
        "ao_decision": {"trace_id": "t1"},
    }


def healthy_review():
    return {
        "approved": True,
        "trace_id": "t1",
        "reviewed_by": [
            "assertion_extractor", "laap_grounding", "tool_evidence",
            "psilang", "ao_health",
        ],
        "issues": [],
        "grounding": {"grounded": True},
    }


def test_contract_accepts_complete_end_to_end_evidence():
    result = integration_fitness(healthy_preflight(), healthy_review())
    assert result.passed is True
    assert result.score == 1.0


def test_contract_names_missing_layer():
    payload = healthy_preflight()
    payload["health"]["layers"]["memory"] = False
    result = evaluate_preflight(payload)
    assert result.passed is False
    assert "layer:memory" in result.failures


def test_review_fails_closed_when_ao_is_unreachable():
    payload = healthy_review()
    payload["issues"] = ["ao_unreachable"]
    result = evaluate_review(payload)
    assert result.passed is False
    assert "review:cognition_unreachable" in result.failures


def test_deterministic_redaction_preserves_verified_content():
    response = "操作建议：保留备份。\n系统已于2099年完成部署。"
    rewritten = deterministic_grounding_redaction(
        response, ["系统已于2099年完成部署。"]
    )
    assert "操作建议" in rewritten
    assert "2099年" not in rewritten
    assert REDACTION_MARKER in rewritten


def test_redaction_fails_closed_when_claim_cannot_be_located():
    assert deterministic_grounding_redaction("安全建议", ["不存在的原句"]) == ""
