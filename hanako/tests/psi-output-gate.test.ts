import { describe, expect, it } from "vitest";
import {
  finalizedAssistantText,
  isArisPsiGatedAgent,
  isIntermediateToolMessage,
  psiReviewResult,
} from "../lib/psi-output-gate.ts";

describe("PSI output transport gate", () => {
  it("gates only Aris identities", () => {
    expect(isArisPsiGatedAgent({ id: "hanako", agentName: "aris" })).toBe(true);
    expect(isArisPsiGatedAgent({ id: "researcher", agentName: "Researcher" })).toBe(false);
  });

  it("releases only finalized extension-replaced text", () => {
    const message = {
      role: "assistant",
      content: [{ type: "text", text: "审议后的正文" }],
      psiReview: { approved: true, issues: [] },
    };
    expect(finalizedAssistantText(message)).toBe("审议后的正文");
    expect(psiReviewResult(message)).toEqual({ present: true, approved: true, issues: [] });
  });

  it("preserves an explicit rejection as metadata rather than releasable prose", () => {
    expect(psiReviewResult({
      role: "assistant",
      content: [],
      psiReview: { approved: false, issues: ["review_unreachable"] },
    })).toEqual({
      present: true,
      approved: false,
      issues: ["review_unreachable"],
    });
  });

  it("fails closed when review metadata is missing", () => {
    expect(psiReviewResult({ role: "assistant", content: "draft" })).toEqual({
      present: false,
      approved: false,
      issues: ["psi_review_missing"],
    });
  });

  it("keeps intermediate tool-use messages behind the gate", () => {
    expect(isIntermediateToolMessage({ stopReason: "toolUse" })).toBe(true);
    expect(isIntermediateToolMessage({ stopReason: "tool_use" })).toBe(true);
    expect(isIntermediateToolMessage({ stopReason: "stop" })).toBe(false);
  });
});
