export const name = "aris-predict";
export const description = "预测 Aris 的未来认知状态或用户回归概率。可预测需求趋势、情感变化、用户多久会回来。";

export const parameters = {
  type: "object",
  properties: {
    target: {
      type: "string",
      enum: ["state", "return", "forecast"],
      description: "预测目标：state=当前状态分布, return=用户回归概率, forecast=未来趋势",
    },
    horizon: {
      type: "number",
      description: "预测时间窗口（小时），默认 2",
      default: 2,
    },
  },
  required: ["target"],
};

export async function execute(input, ctx) {
  const { httpGet, httpPost, port } = getRuntime(ctx);
  if (!port) return { content: [{ type: "text", text: "Aris 引擎未启动" }] };

  try {
    if (input.target === "state") {
      const r = await httpGet("/predict/state");
      const needs = r.estimated_needs || {};
      const uncert = r.uncertainty || {};
      const lines = ["当前认知状态分布："];
      for (const [k, v] of Object.entries(needs)) {
        const u = uncert[k] || 0;
        lines.push("  " + k + ": " + (v * 100).toFixed(0) + "% (±" + (u * 100).toFixed(1) + "%)");
      }
      return { content: [{ type: "text", text: lines.join("\n") }] };
    } else if (input.target === "return") {
      const r = await httpGet("/predict/return?horizon=" + (input.horizon || 2));
      return {
        content: [{ type: "text", text: "用户在 " + (input.horizon || 2) + " 小时内回来的概率为 " + (r.probability * 100).toFixed(0) + "%" }],
      };
    } else if (input.target === "forecast") {
      const r = await httpGet("/forecast?steps=4&dt=0.25");
      const traj = r.needs_trajectory || {};
      const lines = ["未来 1 小时需求预测："];
      for (const [k, vals] of Object.entries(traj)) {
        if (vals && vals.length > 0) {
          const trend = vals.map(v => (v * 100).toFixed(0) + "%").join(" → ");
          lines.push("  " + k + ": " + trend);
        }
      }
      return { content: [{ type: "text", text: lines.join("\n") }] };
    }
  } catch (e) {
    return { content: [{ type: "text", text: "预测失败: " + e.message }] };
  }
}

function getRuntime(ctx) {
  const port = ctx._aris?.sidecar?.port || 11521;
  const httpGet = (path) => {
    return new Promise((resolve, reject) => {
      const http = require("http");
      http.get({ hostname: "127.0.0.1", port, path, timeout: 10000 }, res => {
        let d = "";
        res.on("data", c => d += c);
        res.on("end", () => { try { resolve(JSON.parse(d)); } catch(e) { resolve({}); } });
      }).on("error", e => reject(e));
    });
  };
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
  return { httpGet, httpPost, port };
}
