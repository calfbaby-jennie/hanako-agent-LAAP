export const name = "aris-experience";
export const description = "查看或更新 Aris 从桌面会话中积累的经验知识。包括行为规则、偏好和技能。";

export const parameters = {
  type: "object",
  properties: {
    action: {
      type: "string",
      enum: ["summary", "list", "refresh"],
      description: "操作：summary=查看统计, list=列出所有, refresh=重新从桌面会话提取",
      default: "summary",
    },
  },
};

export async function execute(input, ctx) {
  const port = ctx._aris?.sidecar?.port || 11521;
  const action = input.action || "summary";

  try {
    const http = require("http");

    if (action === "summary") {
      const r = await httpGet(port, "/learn/stats");
      return { content: [{ type: "text", text: "主动学习: " + (r.learn_count || 0) + " 次" }] };
    } else if (action === "list") {
      const r = await httpGet(port, "/harness/turns?n=3");
      return { content: [{ type: "text", text: "最近 3 次交互已记录" }] };
    } else if (action === "refresh") {
      // 触发从桌面会话重新提取
      const http = require("http");
      await new Promise((resolve, reject) => {
        http.get({ hostname: "127.0.0.1", port, path: "/tick", timeout: 5000 }, res => {
          let d = "";
          res.on("data", c => d += c);
          res.on("end", () => resolve(d));
        }).on("error", e => reject(e));
      });
      return { content: [{ type: "text", text: "已触发经验刷新" }] };
    }
  } catch (e) {
    return { content: [{ type: "text", text: "操作失败: " + e.message }] };
  }
}

function httpGet(port, path) {
  return new Promise((resolve, reject) => {
    const http = require("http");
    http.get({ hostname: "127.0.0.1", port, path, timeout: 10000 }, res => {
      let d = "";
      res.on("data", c => d += c);
      res.on("end", () => { try { resolve(JSON.parse(d)); } catch(e) { resolve({}); } });
    }).on("error", e => reject(e));
  });
}
