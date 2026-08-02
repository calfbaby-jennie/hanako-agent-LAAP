"""LAAP — Living Computation Paradigm / 生命计算范式

PSI 认知架构数字生命体入口包。
运行 `python -m laap` 启动 LAAP 数字生命体。

SDK 入口：
    from laap import AetherClient, LAAPRuntime, ActorSystem, PetriNet
"""

__version__ = "5.0.0"

# ── SDK 核心 API (轻量导入) ──────────────────────────────────────────
# ── 框架核心 (按需导入，惰性加载) ─────────────────────────────────────
def __getattr__(name):
    """Lazy import for framework core to keep SDK import lightning-fast."""
    _lazy = {
        "AetherClient":     "laap.sdk.client",
        "LAAPRuntime":      "laap.sdk.runtime",
        "ActorSystem":       "laap.orchestration.actor",
        "AgentCell":         "laap.orchestration.actor",
        "PetriNet":          "laap.orchestration.petri",
        "PetriPlace":        "laap.orchestration.petri",
        "PetriTransition":   "laap.orchestration.petri",
        "ColoredToken":      "laap.orchestration.petri",
        "TokenColor":        "laap.orchestration.petri",
        "AetherAddress":     "laap.orchestration.primitives",
        "AetherMessage":     "laap.orchestration.primitives",
        "MessageType":       "laap.orchestration.primitives",
        "MessageRouter":     "laap.orchestration.primitives",
        "OrchestrationKernel": "laap.orchestration.kernel",
        "MetaAgent":         "laap.orchestration.meta_agent",
        "Capability":         "laap.orchestration.actor",
        "PSIAgent":          "laap.orchestration.psi",
        "SelfInspectionEngine": "laap.self_inspection",
        "ArisCognitiveBus":  "laap.orchestration.cognitive_bus",
        "seq":               "laap.orchestration.dsl",
        "par":               "laap.orchestration.dsl",
        "act":               "laap.orchestration.dsl",
        "guard":             "laap.orchestration.dsl",
        "loop":              "laap.orchestration.dsl",
        "infer":             "laap.orchestration.dsl",
        "skill":             "laap.orchestration.dsl",
        "compile_workflow":  "laap.orchestration.dsl",
        "LAAPBuilder":       "laap.orchestration.dsl",
        # ── Domain SDK framework ──
        "DomainSDKBase":           "laap.domain_sdk",
        "DomainManifest":          "laap.domain_sdk",
        "HarnessFunction":         "laap.domain_sdk",
        "HarnessFunctionRegistry": "laap.domain_sdk",
        "harness_function":        "laap.domain_sdk",
        "DomainSafetyPolicy":      "laap.domain_sdk",
        "SafetyBreachError":       "laap.domain_sdk",
        "SafetyCheckResult":       "laap.domain_sdk",
        "SafetyViolationType":     "laap.domain_sdk",
        "SpeciesTemplate":         "laap.domain_sdk",
        "SpeciesInstance":         "laap.domain_sdk",
        "SpeciesLibrary":          "laap.domain_sdk",
        "DomainSDKRegistry":       "laap.domain_sdk",
    }
    if name in _lazy:
        import importlib
        mod = importlib.import_module(_lazy[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # SDK 核心
    "AetherClient",
    "LAAPRuntime",
    # 框架核心
    "ActorSystem",
    "AgentCell",
    "PetriNet",
    "PetriPlace",
    "PetriTransition",
    "ColoredToken",
    "TokenColor",
    "AetherAddress",
    "AetherMessage",
    "MessageType",
    "MessageRouter",
    "OrchestrationKernel",
    "MetaAgent",
    "Capability",
    "PSIAgent",
    "ArisCognitiveBus",
    # DSL
    "seq", "par", "act", "guard", "loop", "infer", "skill",
    "compile_workflow",
    "LAAPBuilder",
    # Domain SDK framework
    "DomainSDKBase",
    "DomainManifest",
    "HarnessFunction",
    "HarnessFunctionRegistry",
    "harness_function",
    "DomainSafetyPolicy",
    "SafetyBreachError",
    "SafetyCheckResult",
    "SafetyViolationType",
    "SpeciesTemplate",
    "SpeciesInstance",
    "SpeciesLibrary",
    "DomainSDKRegistry",
]
