"""Pure runtime helpers for the Hana ↔ PSI transport boundary.

Kept separate from the acceptance oracle so recursive improvement cannot rewrite
its own success criteria.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

REDACTION_MARKER = "该项尚未验证，已由 PSI 暂缓发送。"


def tool_evidence_boost(
    claims: List[str], tool_results_text: str,
) -> Tuple[List[str], List[str]]:
    """对每条断言，检查是否能在 tool_results 文本里找到数字证据。

    本轮工具调用产出的数字（端口号、PID、测试结果、文件大小等）
    不在 grounding 记忆库里，但可以直接从 tool_results 里验证。
    返回 (grounded_claims, ungrounded_claims)。
    """
    if not tool_results_text:
        return [], list(claims)
    tool_lower = tool_results_text.lower()
    grounded: List[str] = []
    ungrounded: List[str] = []
    for claim in claims:
        # 提取断言中的数字串（含小数、百分比、ms、KB 等）
        numbers = re.findall(r'\d+\.?\d*\s*(?:ms|kb|mb|gb|hz|byte|秒|分|步|个|项|条|份)?', str(claim), re.I)
        hit = any(num.strip() and num.strip() in tool_lower for num in numbers[:8])
        if hit:
            grounded.append(claim)
        else:
            ungrounded.append(claim)
    return grounded, ungrounded


def deterministic_grounding_redaction(response: str, failed_claims: Iterable[str]) -> str:
    """Remove every ungrounded claim without generating replacement facts."""
    claims = [str(claim).strip() for claim in failed_claims if str(claim).strip()]
    if not claims:
        return response.strip()

    unique_claims = list(dict.fromkeys(claims))
    if not all(claim in response for claim in unique_claims):
        return ""

    pattern = "|".join(re.escape(claim) for claim in sorted(unique_claims, key=len, reverse=True))
    redacted = re.sub(pattern, REDACTION_MARKER, response)

    redacted = re.sub(
        rf"(?:{re.escape(REDACTION_MARKER)}\s*){{2,}}",
        REDACTION_MARKER + "\n",
        redacted,
    )
    return redacted.strip()
