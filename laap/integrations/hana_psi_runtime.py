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
    if not response:
        return ""
    claims = [
        claim
        for claim in dict.fromkeys(str(c).strip() for c in failed_claims)
        if claim
    ]
    if not claims:
        return response.strip()
    pattern = "|".join(
        re.escape(claim) for claim in sorted(claims, key=len, reverse=True)
    )
    redacted = re.sub(pattern, REDACTION_MARKER, response)
    redacted = re.sub(
        rf"(?:{re.escape(REDACTION_MARKER)}\s*){{2,}}",
        REDACTION_MARKER + "\n",
        redacted,
    )
    return redacted.strip()
