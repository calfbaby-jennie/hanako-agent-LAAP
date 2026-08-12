"""Hana ↔ LAAP/PSI integration acceptance contract.

This module is intentionally pure and side-effect free so True RSI can mutate it
inside Zone2 while acceptance tests remain the fitness oracle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hana_psi_runtime import REDACTION_MARKER, deterministic_grounding_redaction

REQUIRED_LAYERS = (
    "perception", "decision", "memory", "self", "motivation",
    "symbolic", "grounding", "transport", "evolution",
)
REQUIRED_CONTEXT_MARKERS = (
    "pipeline.complete=true", "ao.intent=", "ao.trace=",
    "organs=", "health.layers=", "directive=",
)
@dataclass(frozen=True)
class ContractResult:
    passed: bool
    score: float
    failures: tuple[str, ...] = field(default_factory=tuple)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "score": round(self.score, 4),
            "failures": list(self.failures),
            "evidence": self.evidence,
        }


def evaluate_preflight(payload: dict[str, Any]) -> ContractResult:
    """Verify that every required brain layer participated before the LLM."""
    health = payload.get("health") or {}
    layers = health.get("layers") or {}
    context = str(payload.get("context") or "")
    failures: list[str] = []
    for name in REQUIRED_LAYERS:
        if layers.get(name) is not True:
            failures.append(f"layer:{name}")
    for marker in REQUIRED_CONTEXT_MARKERS:
        if marker not in context:
            failures.append(f"context:{marker}")
    if payload.get("ok") is not True:
        failures.append("preflight:not_ok")
    if (payload.get("perception") or {}).get("blocked") is True:
        failures.append("perception:blocked")
    denominator = len(REQUIRED_LAYERS) + len(REQUIRED_CONTEXT_MARKERS) + 2
    score = max(0.0, 1.0 - len(set(failures)) / denominator)
    return ContractResult(not failures, score, tuple(dict.fromkeys(failures)), {
        "layers": layers,
        "degraded": payload.get("degraded") or health.get("degraded") or [],
        "trace_id": (payload.get("ao_decision") or {}).get("trace_id"),
    })


def evaluate_review(payload: dict[str, Any]) -> ContractResult:
    """Verify the post-LLM PSI release decision and its audit evidence."""
    failures: list[str] = []
    if payload.get("approved") is not True:
        failures.append("review:not_approved")
    reviewers = set(payload.get("reviewed_by") or [])
    for reviewer in ("laap_grounding", "tool_evidence", "psilang", "ao_health"):
        if reviewer not in reviewers:
            failures.append(f"reviewer:{reviewer}")
    if not payload.get("trace_id"):
        failures.append("review:missing_trace")
    issues = payload.get("issues") or []
    if any(str(issue).startswith(("psi_unreachable", "ao_unreachable")) for issue in issues):
        failures.append("review:cognition_unreachable")
    denominator = 7
    score = max(0.0, 1.0 - len(set(failures)) / denominator)
    return ContractResult(not failures, score, tuple(dict.fromkeys(failures)), {
        "issues": issues,
        "grounding": payload.get("grounding") or {},
        "regenerated": bool(payload.get("regenerated")),
    })


def integration_fitness(preflight: dict[str, Any], review: dict[str, Any]) -> ContractResult:
    """Weighted end-to-end fitness used as the recursive evolution oracle."""
    before = evaluate_preflight(preflight)
    after = evaluate_review(review)
    score = before.score * 0.6 + after.score * 0.4
    failures = before.failures + after.failures
    return ContractResult(before.passed and after.passed, score, failures, {
        "preflight": before.to_dict(), "review": after.to_dict(),
    })
