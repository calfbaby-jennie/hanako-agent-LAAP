const ARIS_AGENT_IDS = new Set(["hanako", "aris"]);

function normalized(value: unknown): string {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

/** Restrict PSI transport gating to the Aris agent runtime. */
export function isArisPsiGatedAgent(agentOrSession: any): boolean {
  const candidates = [
    agentOrSession?.id,
    agentOrSession?.agentId,
    agentOrSession?.name,
    agentOrSession?.agentName,
    agentOrSession?.agent?.id,
    agentOrSession?.agent?.name,
    agentOrSession?.agent?.agentName,
  ].map(normalized).filter(Boolean);
  return candidates.some((value) => ARIS_AGENT_IDS.has(value));
}

/** Text after extension message_end replacement; this is the only releasable body. */
export function finalizedAssistantText(message: any): string {
  const content = message?.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter((block) => block?.type === "text" && typeof block.text === "string")
    .map((block) => block.text)
    .join("");
}

export function psiReviewResult(message: any): {
  present: boolean;
  approved: boolean;
  issues: string[];
} {
  const review = message?.psiReview;
  if (!review || typeof review !== "object") {
    return { present: false, approved: false, issues: ["psi_review_missing"] };
  }
  return {
    present: true,
    approved: review.approved === true,
    issues: Array.isArray(review.issues) ? review.issues.map(String) : [],
  };
}

export function isIntermediateToolMessage(message: any): boolean {
  const reason = normalized(message?.stopReason);
  return reason === "tooluse" || reason === "tool_use";
}
