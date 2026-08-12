"""Pure runtime helpers for the Hana ↔ PSI transport boundary.

Kept separate from the acceptance oracle so recursive improvement cannot rewrite
its own success criteria.
"""
from __future__ import annotations

import re
from typing import Iterable

REDACTION_MARKER = "该项尚未验证，已由 PSI 暂缓发送。"


def deterministic_grounding_redaction(response: str, failed_claims: Iterable[str]) -> str:
    """Remove every ungrounded claim without generating replacement facts."""
    redacted = response
    claims = [str(claim).strip() for claim in failed_claims if str(claim).strip()]
    removed = 0
    for claim in claims:
        if claim in redacted:
            redacted = redacted.replace(claim, REDACTION_MARKER)
            removed += 1
    if removed != len(claims):
        return ""
    redacted = re.sub(
        rf"(?:{re.escape(REDACTION_MARKER)}\s*){{2,}}",
        REDACTION_MARKER + "\n",
        redacted,
    )
    return redacted.strip()
