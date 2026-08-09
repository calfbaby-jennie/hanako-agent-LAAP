export const name = "aris-verify";
export const description = "验算一条断言的真实性 — 用世界模型 + 因果引擎 + Ψ-Net 共识判断一句话是否可信。";

export const parameters = {
  type: "object",
  properties: {
    text: {
      type: "string",
      description: "要验算的断言文本",
    },
  },
  required: ["text"],
};

export async function execute(input, ctx) {
  const { httpPost, port } = getRuntime(ctx);
  if (!port) return { content: [{ type: "text", text: "Aris 引擎未启动" }] };

  try {
    const r = await httpPost("/verify", { text: input.text });
    if (r.gate === "pass") {
      return { content: [{ type: "text", text: "✅ 断言通过验算（置信度 " + (r.confidence * 100).toFixed(0) + "%）" }] };
    } else if (r.gate === "flag") {
      return { content: [{ type: "text", text: "⚠️ 断言存疑（置信度 " + (r.confidence * 100).toFixed(0) + "%），建议核实：" + (r.claims || []).map(c => c.text).join("；") }] };
    } else if (r.gate === "rewrite") {
      return { content: [{ type: "text", text: "✏️ 断言需要重写（置信度 " + (r.confidence * 100).toFixed(0) + "%），" + r.gate_reason }] };
    } else {
      return { content: [{ type: "text", text: "🚫 断言被拦截（置信度 " + (r.confidence * 100).toFixed(0) + "%），" + r.gate_reason }] };
    }
  } catch (e) {
    return { content: [{ type: "text", text: "验算失败: " + e.message }] };
  }
}

function getRuntime(ctx) {
  const port = ctx._aris?.sidecar?.port || 11521;
  const httpPost = (path, data) => {
    return new Promise((resolve, reject) => {
      const http = require("http");
      const body = JSON.stringify(data);
      const req = http.request({
        hostname: "127.0.0.1", port, path, method: "POST",
        headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(body) },
        timeout: 15000,
      }, res => {
        let d = "";
        res.on("data", c => d += c);
        res.on("end", () => { try { resolve(JSON.parse(d)); } catch(e) { resolve({}); } });
      });
      req.on("error", e => reject(e));
      req.write(body);
      req.end();
    });
  };
  return { httpPost, port };
}
