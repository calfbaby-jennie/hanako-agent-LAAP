"""LAAP SDK — 生命计算范式的大脑接入层。

两种使用模式：

Client 模式（挂载到外部 Agent）::

    from laap import AetherClient

    client = AetherClient()
    async with client:
        response = await client.query("你好")
        print(response)

Framework 模式（独立运行 LAAP 框架）::

    from laap import LAAPRuntime, Capability, MessageType

    runtime = LAAPRuntime(enable_psi=True)
    actor = runtime.spawn("worker", capabilities=[
        Capability(name="code", confidence=0.9),
    ])
    actor.on(MessageType.INVOKE, my_handler)

    # 或者运行认知循环
    result = await runtime.cognize("Hello")
    await runtime.shutdown()
"""

from __future__ import annotations

def __getattr__(name: str):
    """Load SDK entry points only when requested."""
    import importlib

    lazy = {
        "AetherClient": ("laap.sdk.client", "AetherClient"),
        "LAAPRuntime": ("laap.sdk.runtime", "LAAPRuntime"),
        "AgentAdapter": ("laap.sdk.adapter", "AgentAdapter"),
        "mount_brain_to_agent": ("laap.sdk.mount", "mount_brain_to_agent"),
    }
    if name not in lazy:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = lazy[name]
    return getattr(importlib.import_module(module_name), attribute)

__all__ = [
    "AetherClient",
    "LAAPRuntime",
    "AgentAdapter",
    "mount_brain_to_agent",
]
