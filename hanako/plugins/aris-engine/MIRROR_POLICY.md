# Aris Engine source-of-truth policy

Canonical implementation:

`hanako-agent-LAAP/hanako/plugins/aris-engine`

Deployment mirror:

`OH-WorkSpace/hanako/plugins/aris-engine`

Synchronization is one-way: **canonical active source → OH-WorkSpace mirror**.
Never copy an older OH-WorkSpace mirror over the active source without reviewing
Git diff and running `tests/aris-engine-cache-contract.test.ts`.

Invariant:

- session-stable identity/instructions may live in the system prompt;
- PSI, emotion, focus, memory counters, Ao intent and other volatile cognition
  must be injected through the Pi `context` hook into the current user turn;
- the extension must preserve `/perceive`, `tool_result`, `/after_turn`, explicit
  `Content-Length`, and the `arisInjectContext` runtime resolver.
