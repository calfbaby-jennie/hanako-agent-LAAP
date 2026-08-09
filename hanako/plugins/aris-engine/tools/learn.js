export const name = "aris-learn";
export const description = "主动学习 — 搜索并学习一个新知识点，通过质量过滤器验算后灌入世界模型。";

export const parameters = {
  type: "object",
  properties: {
    topic: {
      type: "string",
      description: "要学习的话题关键词",
    },
  },
  required: ["topic"],
};

export async function execute(input, ctx) {
  const port = ctx._aris?.sidecar?.port || 11521;

  try {
    const http = require("http");
    const path = "/learn/active?topic=" + encodeURIComponent(input.topic);
    const result = await new Promise((resolve, reject) => {
      http.get({ hostname: "127.0.0.1", port, path, timeout: 60000 }, res => {
        let d = "";
        res.on("data", c => d += c);
        res.on("end", () => { try { resolve(JSON.parse(d)); } catch(e) { resolve({}); } });
      }).on("error", e => reject(e));
    });

    if (result.seeded > 0) {
      return {
        content: [{ type: "text", text: "学习了 \"" + input.topic + "\"，灌入 " + result.seeded + " 个实体。不确定性从 " + result.uncertainty_before + " 降至 " + result.uncertainty_after + "。" }],
      };
    } else if (result.info) {
      return { content: [{ type: "text", text: result.info }] };
    } else if (result.error) {
      return { content: [{ type: "text", text: "学习失败: " + result.error }] };
    } else {
      return { content: [{ type: "text", text: "未学习到新知识" }] };
    }
  } catch (e) {
    return { content: [{ type: "text", text: "学习失败: " + e.message }] };
  }
}
