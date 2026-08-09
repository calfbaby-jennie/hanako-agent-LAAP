export const name = "aris-strategy";
export const description = "获取当前最优交互策略 — 根据 Aris 的认知状态和预测结果，推荐最适合的对话方向。";

export const parameters = {
  type: "object",
  properties: {},
};

export async function execute(input, ctx) {
  const port = ctx._aris?.sidecar?.port || 11521;

  try {
    const http = require("http");
    const r = await new Promise((resolve, reject) => {
      http.get({ hostname: "127.0.0.1", port, path: "/strategy", timeout: 10000 }, res => {
        let d = "";
        res.on("data", c => d += c);
        res.on("end", () => { try { resolve(JSON.parse(d)); } catch(e) { resolve({}); } });
      }).on("error", e => reject(e));
    });

    const primary = r.primary || {};
    const alternatives = r.alternatives || [];
    const ctx_info = r.context || {};

    const lines = [
      "🎯 推荐策略: " + (primary.label || "—"),
      "  评分: " + (primary.score || 0).toFixed(2),
      "  说明: " + (primary.description || ""),
    ];
    if (alternatives.length > 0) {
      lines.push("备选方案:");
      alternatives.forEach(a => lines.push("  · " + a.label + " (" + a.score.toFixed(2) + ")"));
    }
    lines.push("当前状态: 需求=" + (ctx_info.dominant_need || "?") + " 情感=" + (ctx_info.dominant_emotion || "?"));

    return { content: [{ type: "text", text: lines.join("\n") }] };
  } catch (e) {
    return { content: [{ type: "text", text: "策略获取失败: " + e.message }] };
  }
}
