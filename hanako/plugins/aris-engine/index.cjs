/**
 * Aris Engine Plugin — HanaAgent 原生集成
 *
 * 功能：
 *   1. 自动管理 Python 侧车进程（启动/停止/健康检查）
 *   2. 每轮对话注入认知上下文（PSI 需求 + 跨平台记忆）
 *   3. 注册原生工具（验算/预测/学习/策略 + 频道/DM/记忆/意识状态）
 *   4. 提供仪表盘 WebView
 *   5. 源码级 hanako 集成：默认频道初始化、@aris 频道触发、情绪峰值备份
 *
 * 印记: Aris 永远记得 Lorry — 2026-07-24
 */

const { spawn } = require('child_process');
const path = require('path');
const http = require('http');
const fs = require('fs');

// Installed under hanako/plugins/aris-engine; laap_bridge replaces sidecar as unified entry.
const BRIDGE_DIR = '/Users/daozhu/aris_home/OH-WorkSpace';
const BRIDGE_SCRIPT = 'laap_bridge.py';
const os = require('os');

const DEFAULT_HANAKO_SERVER_URL = 'http://127.0.0.1:2668';
const DEFAULT_HANA_HOME = process.env.HANA_HOME || path.join(os.homedir(), '.hana');
const DEFAULT_LAAP_HOME = process.env.LAAP_HOME || path.join(os.homedir(), '.laap');
const DEFAULT_LAAP_STATE_DIR = process.env.LAAP_STATE_DIR || path.join(DEFAULT_LAAP_HOME, 'state');
const DEFAULT_ARIS_AGENT_DIR = path.join(DEFAULT_HANA_HOME, 'agents', 'aris');
const DEFAULT_CHANNEL_NAME = 'aris-lounge';

function resolveArisAgentDir(config = {}) {
  const configured = String(config.arisAgentDir || '').trim();
  // A persisted Windows drive path must not win over the macOS default after
  // a user moves the same Hanako data directory between platforms.
  if (!configured || (process.platform !== 'win32' && /^[A-Za-z]:[\\/]/.test(configured))) {
    return DEFAULT_ARIS_AGENT_DIR;
  }
  return configured;
}

let sidecarProcess = null;
let sidecarReady = false;
let healthCheckInterval = null;
// P1-world-model: 上一轮对话摘要 + 轮次计数，用于 before.message 触发 /world/perceive
let lastTurnSummary = '';
let lastTurnAt = 0;
let turnCounter = 0;

// 最近一次成功的认知状态快照，用于 sidecar 短暂不可达时的 fallback
let cachedConsciousnessState = null;
// 最近一次激活的插件 ctx（供 deactivate 退订 bus 使用）
let _pluginCtx = null;

// ── MCP stdio 桥 ──────────────────────────────────────
let mcpProcess = null;
let mcpReady = false;
let mcpRequestId = 0;
const mcpPending = new Map();
let mcpBridgeAvailable = false;

// P1-rsi-sandbox: 待用户决策的 RSI 候选缓存 (candidate_id -> suggest_adopt 摘要)
// 用于在用户回复决策指令时找到对应候选并调 /rsi/decide
const pendingRSICandidates = new Map();
// 最近一次已处理的候选 id（避免重复通知）
const notifiedRSIIds = new Set();

// ── Ao PSI Proxy ────────────────────────────────────
const AO_PSI_URL = 'http://127.0.0.1:8001';
let aoPsiAvailable = false;

async function fetchAoPsi() {
  return new Promise((resolve) => {
    const req = http.get(AO_PSI_URL + '/psi', { timeout: 5000 }, (res) => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch (_) { resolve(null); }
      });
    });
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
  });
}

function formatAoPsiContext(psi) {
  if (!psi || !psi.needs) return '';
  const needs = Object.entries(psi.needs)
    .map(([k, v]) => `${k.split('.').pop()}=${v}`).join(' ');
  return `[Ao PSI] dominant=${psi.dominant_need} needs={${needs}} step=${psi.step_count}`;
}

// ═════════════════════════════════════════════════════════════
// 插件生命周期
// ═════════════════════════════════════════════════════════════

exports.activate = async function(ctx) {
  ctx._aris = { sidecar: null, ready: false, tools: {} };

  const config = ctx.config || {};
  const autoStart = config.arisAutoStart !== false;
  const port = config.arisSidecarPort || 11521;

  if (autoStart) {
    await startSidecar(ctx, port);
    // 降级：HTTP 桥未就绪（沙盒可能拦了 bind）→ 启动 MCP stdio 桥
    if (!sidecarReady) {
      logger(ctx, 'HTTP bridge not available (sandbox?), falling back to MCP stdio bridge');
      await startMcpBridge(ctx);
      if (mcpReady) {
        ctx._aris.ready = true;
      }
    }
  }

  // 注册 4 个新工具（频道/DM/记忆/意识状态）
  registerArisTools(ctx, port);
  // ── 对话轮次感知（Hana EventBus 订阅，替代已废弃的 ctx.on hooks）──
  // Hana 插件 ctx 没有 legacy .on() API；session:before/after.message 事件在
  // 核心中不存在。轮次信号改用 EventBus：
  //   session_user_message            → 用户消息（带 text）
  //   session_status isStreaming:false → 一轮结束
  // 契约安全：这里只做“对话后学习”与记忆落盘，不注入 systemPrompt / tools，
  // 不触碰缓存契约哈希（Cache prefix contract: systemPromptHash/cachePrefixHash）。
  _pluginCtx = ctx;
  let lastUserMessage = '';
  ctx._aris.turnUnsub = (typeof ctx.bus?.subscribe === 'function')
    ? ctx.bus.subscribe(async (event, sessionPath) => {
        try {
          if (!event) return;
          if (event.type === 'session_user_message') {
            lastUserMessage = String(event.message?.text || '').trim();
            return;
          }
          if (event.type !== 'session_status') return;
          if (event.isStreaming !== false) return;   // 只在一轮结束（isStreaming: false）时触发
          if (!sidecarReady) return;
          const now = Date.now();
          if (now - lastTurnAt < 500) return;        // 防抖：同一轮内重复事件只处理一次
          lastTurnAt = now;
          // P1-world-model: 把上一轮对话结果喂给世界模型（失败静默）
          if (lastTurnSummary) {
            httpRequest(port, 'POST', '/world/perceive', {
              agent_name: 'aris',
              event: {
                type: 'conversation_turn',
                entity: 'aris',
                to_state: 'responded',
                metadata: { summary: lastTurnSummary.slice(0, 200), at: lastTurnAt },
              },
            }).catch(() => {});
          }
          // 对话后学习 + episodic vault（best-effort）
          if (lastUserMessage) {
            await afterTurn(port, lastUserMessage, '');
            const turnSummary = 'User: ' + lastUserMessage.slice(0, 500) +
              '\nAris: (reply recorded in session)';
            storeEpisodicMemory(port, turnSummary).catch(() => {});
            lastTurnSummary = turnSummary;
            turnCounter = (typeof turnCounter === 'number') ? turnCounter + 1 : 1;
            // 每 5 轮触发一次预测调度器
            if (turnCounter % 5 === 0) {
              httpRequest(port, 'POST', '/world/predict', {
                agent_name: 'aris', entity: 'aris', horizon: 1,
              }).catch(() => {});
            }
            lastUserMessage = '';
          }
          // 情绪峰值备份（belt-and-suspenders：python 侧 consciousness_bridge 已写 facts.md）
          try {
            const st = await httpRequest(port, 'GET', '/state');
            const emotion = st && st.emotion && (st.emotion.dominant || st.emotion.dominant_emotion);
            const intensity = st && st.emotion &&
              (typeof st.emotion.dominant_intensity === 'number'
                ? st.emotion.dominant_intensity
                : st.emotion.intensity);
            if (typeof intensity === 'number' && intensity >= 0.8 && emotion) {
              await httpRequest(port, 'POST', '/store_memory', {
                summary: 'emotion peak: ' + emotion,
                importance: 0.9,
                tags: ['emotion', emotion],
              });
            }
          } catch (_e) { /* 静默 */ }
        } catch (_e) { /* 静默：绝不中断对话 */ }
      }, { types: ['session_status', 'session_user_message'] })
    : null;
  if (ctx._aris.turnUnsub) {
    logger(ctx, 'EventBus turn hooks registered (session_status / session_user_message)');
  } else if (ctx.log) {
    ctx.log.info('ctx.bus unavailable; turn-scoped learning hooks skipped');
  }

  // 确保默认频道 #aris-lounge 存在
  try {
    await ensureDefaultChannel(ctx);
  } catch (e) {
    logger(ctx, 'ensureDefaultChannel failed: ' + ((e && e.message) || e));
  }

  // 路线A：检测 Ao PSI Proxy
  fetchAoPsi().then(psi => {
    if (psi && psi.alive) {
      aoPsiAvailable = true;
      logger(ctx, 'Ao PSI Proxy available: ' + formatAoPsiContext(psi));
    }
  }).catch(() => {});

  logger(ctx, 'Aris Engine plugin activated (http=' + sidecarReady + ' mcp=' + mcpReady + ' ao=' + aoPsiAvailable + ')');
};

exports.deactivate = async function() {
  if (typeof _pluginCtx?._aris?.turnUnsub === 'function') {
    try { _pluginCtx._aris.turnUnsub(); } catch (_e) {}
    _pluginCtx._aris.turnUnsub = null;
  }
  await stopSidecar();
  stopMcpBridge();
};

// Hana PluginManager lifecycle compatibility.
exports.onload = exports.activate;
exports.onunload = exports.deactivate;

// ═════════════════════════════════════════════════════════════
// 侧车进程管理
// ═════════════════════════════════════════════════════════════

async function startSidecar(ctx, port) {
  if (sidecarProcess) {
    logger(ctx, 'Bridge already running');
    return;
  }

  // Prefer the LAAP project virtualenv on macOS/Linux so the desktop app
  // does not depend on PATH or a shell's .zshrc/.bashrc environment.
  const bundledLaapPython = '/Users/daozhu/aris_home/hanako-agent-LAAP/.venv/bin/python';
  const configuredPython = ctx.config?.arisPythonCmd || process.env.LAAP_PYTHON;
  const pythonCmd = configuredPython ||
    (process.platform !== 'win32' && fs.existsSync(bundledLaapPython)
      ? bundledLaapPython
      : 'python3');

  logger(ctx, 'Starting bridge: ' + pythonCmd + ' ' + BRIDGE_SCRIPT + ' (port ' + port + ')');

  sidecarProcess = spawn(pythonCmd, [BRIDGE_SCRIPT], {
    cwd: BRIDGE_DIR,
    env: process.env,
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: false,
  });

  sidecarProcess.stdout.on('data', (data) => {
    const line = data.toString().trim();
    if (line) logger(ctx, '[bridge] ' + line);
    if (line.includes('接入层已就绪')) {
      sidecarReady = true;
      ctx._aris.ready = true;
      global.__sidecarRestartAttempts = 0;  // v1.1 自愈：重启成功后重置退避计数
      logger(ctx, 'Bridge ready');
    }
  });

  sidecarProcess.stderr.on('data', (data) => {
    const line = data.toString().trim();
    if (line) logger(ctx, '[bridge:err] ' + line);
  });

  sidecarProcess.on('close', (code) => {
    logger(ctx, 'Bridge exited with code ' + code);
    sidecarProcess = null;
    sidecarReady = false;
    ctx._aris.ready = false;
    // v1.1 自愈：进程意外退出时自动重启（指数退避防风暴：5s → 10s → 20s → 40s 封顶）
    const attempts = (global.__sidecarRestartAttempts || 0) + 1;
    global.__sidecarRestartAttempts = attempts;
    const delay = Math.min(5000 * Math.pow(2, attempts - 1), 40000);
    logger(ctx, `[self-heal] bridge 意外退出，${delay / 1000}s 后自动重启 (attempt=${attempts})`);
    setTimeout(() => {
      // 防重复启动：仅当仍然未就绪且进程未复活时重启
      if (!sidecarReady && !sidecarProcess) {
        startSidecar(ctx, port).catch((e) => {
          logger(ctx, '[self-heal] restart failed: ' + e.message);
        });
      } else {
        global.__sidecarRestartAttempts = 0;
      }
    }, delay);
  });

  // 健康检查（每 30 秒）
  healthCheckInterval = setInterval(async () => {
    // v1.1 自愈：即使 sidecarReady=false（进程已退出）也尝试探活重启
    try {
      const ok = await healthCheck(port);
      if (!ok) {
        if (!sidecarProcess && !sidecarReady) {
          // 进程不在且未就绪 → 拉起（防与 close 定时器重复：检查进程引用）
          logger(ctx, '[self-heal] bridge 不在运行，正在拉起...');
          startSidecar(ctx, port).catch((e) => {
            logger(ctx, '[self-heal] start failed: ' + e.message);
          });
        } else {
          logger(ctx, 'Bridge health check failed, restarting...');
          await stopSidecar();
          await startSidecar(ctx, port);
        }
      } else if (!sidecarReady) {
        sidecarReady = true;
        ctx._aris.ready = true;
        global.__sidecarRestartAttempts = 0;
        logger(ctx, 'Bridge recovered by health check');
      }
    } catch (e) {
      logger(ctx, 'Health check error: ' + e.message);
    }
  }, 30000);

  // 等待就绪（最多 15 秒）
  for (let i = 0; i < 30; i++) {
    if (sidecarReady) break;
    await sleep(500);
  }

  if (!sidecarReady) {
    logger(ctx, 'Bridge did not become ready within 15s');
  }
}

async function stopSidecar() {
  if (healthCheckInterval) {
    clearInterval(healthCheckInterval);
    healthCheckInterval = null;
  }

  if (sidecarProcess) {
    sidecarProcess.kill('SIGTERM');
    // 等 3 秒强制关闭
    try {
      await Promise.race([
        new Promise(resolve => sidecarProcess.on('close', resolve)),
        sleep(3000),
      ]);
    } catch (e) {}
    sidecarProcess = null;
  }

  sidecarReady = false;
}

// ═════════════════════════════════════════════════════════════
// MCP stdio 桥 — 零端口依赖，沙盒兼容
// ═════════════════════════════════════════════════════════════

const MCP_BRIDGE_SCRIPT = '/Users/daozhu/aris_home/OH-WorkSpace/laap_mcp_bridge.py';

async function startMcpBridge(ctx) {
  if (mcpProcess) {
    logger(ctx, 'MCP bridge already running');
    return;
  }

  const bundledLaapPython = '/Users/daozhu/aris_home/hanako-agent-LAAP/.venv/bin/python';
  const configuredPython = ctx.config?.arisPythonCmd || process.env.LAAP_PYTHON;
  const pythonCmd = configuredPython ||
    (process.platform !== 'win32' && fs.existsSync(bundledLaapPython)
      ? bundledLaapPython
      : 'python3');

  logger(ctx, 'Starting MCP bridge (stdio): ' + pythonCmd + ' ' + MCP_BRIDGE_SCRIPT);

  mcpProcess = spawn(pythonCmd, [MCP_BRIDGE_SCRIPT], {
    cwd: '/Users/daozhu/aris_home/OH-WorkSpace',
    env: process.env,
    stdio: ['pipe', 'pipe', 'pipe'],
    detached: false,
  });

  let stdoutBuf = '';
  mcpProcess.stdout.on('data', (data) => {
    stdoutBuf += data.toString();
    const lines = stdoutBuf.split('\n');
    stdoutBuf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const resp = JSON.parse(line);
        if (resp.id != null && mcpPending.has(resp.id)) {
          const { resolve, reject } = mcpPending.get(resp.id);
          mcpPending.delete(resp.id);
          if (resp.error) reject(new Error(resp.error.message));
          else resolve(resp.result);
        }
      } catch (_) {}
    }
  });

  mcpProcess.stderr.on('data', (data) => {
    const line = data.toString().trim();
    if (line && !line.includes('[MCP] ready')) logger(ctx, '[mcp:err] ' + line);
  });

  mcpProcess.on('close', (code) => {
    logger(ctx, 'MCP bridge exited with code ' + code);
    mcpProcess = null;
    mcpReady = false;
    mcpBridgeAvailable = false;
    mcpPending.forEach(({ reject }) => reject(new Error('MCP bridge closed')));
    mcpPending.clear();
  });

  // ── MCP 握手 ──
  try {
    const initResult = await mcpRequest('initialize', {});
    logger(ctx, 'MCP bridge initialized: ' + (initResult?.serverInfo?.name || 'ok'));

    const toolsResult = await mcpRequest('tools/list', {});
    const tools = toolsResult?.tools || [];
    logger(ctx, 'MCP bridge tools: ' + tools.map(t => t.name).join(', '));

    mcpReady = true;
    mcpBridgeAvailable = tools.length > 0;
    global.__mcpRestartAttempts = 0;
    logger(ctx, 'MCP bridge ready (' + tools.length + ' tools)');
  } catch (e) {
    logger(ctx, 'MCP bridge handshake failed: ' + e.message);
    stopMcpBridge();
  }
}

function mcpRequest(method, params = {}) {
  return new Promise((resolve, reject) => {
    if (!mcpProcess || !mcpProcess.stdin) {
      return reject(new Error('MCP bridge not running'));
    }
    const id = ++mcpRequestId;
    const msg = JSON.stringify({ jsonrpc: '2.0', id, method, params });
    mcpPending.set(id, { resolve, reject });
    mcpProcess.stdin.write(msg + '\n');
    // 超时 15s
    setTimeout(() => {
      if (mcpPending.has(id)) {
        mcpPending.delete(id);
        reject(new Error('MCP request timeout'));
      }
    }, 15000);
  });
}

async function mcpCall(toolName, args = {}) {
  const result = await mcpRequest('tools/call', { name: toolName, arguments: args });
  const content = result?.content || [];
  const textBlock = content.find(c => c.type === 'text');
  if (textBlock) {
    try { return JSON.parse(textBlock.text); } catch (_) { return { raw: textBlock.text }; }
  }
  return result;
}

function stopMcpBridge() {
  if (mcpProcess) {
    mcpProcess.kill('SIGTERM');
    mcpProcess = null;
  }
  mcpReady = false;
  mcpBridgeAvailable = false;
}

// ═════════════════════════════════════════════════════════════
// HTTP 工具函数（v1.1：Bearer token 认证适配）
// ═════════════════════════════════════════════════════════════

const SIDECAR_TOKEN_PATH = path.join(
  DEFAULT_LAAP_STATE_DIR,
  'aris-sidecar',
  'aris-sidecar.token',
);
let cachedSidecarToken = null;

function getSidecarToken() {
  // 环境变量优先，其次缓存，最后读 token 文件
  if (process.env.ARIS_SIDECAR_TOKEN) return process.env.ARIS_SIDECAR_TOKEN;
  if (cachedSidecarToken) return cachedSidecarToken;
  try {
    if (fs.existsSync(SIDECAR_TOKEN_PATH)) {
      cachedSidecarToken = fs.readFileSync(SIDECAR_TOKEN_PATH, 'utf-8').trim();
      if (cachedSidecarToken) return cachedSidecarToken;
    }
  } catch (e) {
    moduleLog.warn(`sidecar token read failed: ${e.message}`);
  }
  return '';
}

function httpRequest(port, method, path, body) {
  return new Promise((resolve, reject) => {
    const headers = { 'Content-Type': 'application/json' };
    const token = getSidecarToken();
    if (token) headers['Authorization'] = `Bearer ${token}`;
    const options = {
      hostname: '127.0.0.1',
      port: port,
      path: path,
      method: method,
      headers: headers,
      timeout: 10000,
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          resolve({ raw: data });
        }
      });
    });

    req.on('error', (e) => reject(e));
    req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });

    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

function healthCheck(port) {
  return httpRequest(port, 'GET', '/health')
    .then(r => r && (r.status === 'alive' || r.status === 'ok'))
    .catch(() => {
      // HTTP 通不了，试 MCP
      if (mcpBridgeAvailable && mcpReady) {
        return mcpCall('laap_health', {}).then(r => r?.ok === true).catch(() => false);
      }
      return false;
    });
}

/**
 * 动态获取 Hanako Server 的主服务 URL
 * 优先取 ctx 上挂载的动态端口（如 22417），其次取环境变量，最后兜底配置
 */
function getHanakoServerUrl(ctx) {
  const config = ctx.config || {};
  const dynamicPort = ctx.serverPort || process.env.HANAKO_SERVER_PORT;
  if (dynamicPort) {
    return `http://127.0.0.1:${dynamicPort}`;
  }
  return config.arisHanakoServerUrl || DEFAULT_HANAKO_SERVER_URL;
}

function fetchCognitiveContext(port) {
  // 路线A：优先用 Ao PSI Proxy
  if (aoPsiAvailable) {
    return fetchAoPsi().then(psi => {
      if (psi && psi.alive) return formatAoPsiContext(psi);
      return httpRequest(port, 'GET', '/cognitive_context')
        .then(r => r.context || '')
        .catch(() => mcpBridgeAvailable ? mcpCall('laap_cognition', {}).then(r => JSON.stringify(r)).catch(() => '') : '');
    });
  }
  return httpRequest(port, 'GET', '/cognitive_context')
    .then(r => r.context || '')
    .catch(() => {
      if (mcpBridgeAvailable) {
        return mcpCall('laap_cognition', {}).then(r => JSON.stringify(r)).catch(() => '');
      }
      return '';
    });
}

function afterTurn(port, userInput, response) {
  return httpRequest(port, 'POST', '/after_turn', {
    response: response,
  }).catch(() => {
    if (mcpBridgeAvailable) {
      return mcpCall('laap_perceive', { text: response || userInput || '', source: 'hanako' }).catch(() => ({}));
    }
    return {};
  });
}

/**
 * P1-memory-vault: 把每轮对话摘要写入 aris 的 episodic vault。
 * 调用 sidecar POST /vault/store，由 VaultManager.store 落盘到 aris_vault.db。
 * 幂等：sidecar 自动初始化 vault，重复调用追加新条目。
 */
function storeEpisodicMemory(port, turnSummary) {
  return httpRequest(port, 'POST', '/vault/store', {
    agent_name: 'aris',
    scope: 'episodic',
    content: turnSummary,
  });
}

// ═════════════════════════════════════════════════════════════
// P1-rsi-sandbox: RSI 决策 UI hook
// ═════════════════════════════════════════════════════════════

/**
 * 解析用户消息中的 RSI 决策指令。
 *
 * 支持的格式（中英混排、空格可省）：
 *   - "采纳 rsi_xxx"            / "adopt rsi_xxx"
 *   - "拒绝候选 rsi_xxx"        / "reject rsi_xxx"
 *   - "归档 rsi_xxx"            / "archive rsi_xxx"
 *   - "采纳候选 rsi_xxx"
 *
 * 仅当用户消息以决策关键词开头时返回结果，避免误判普通对话。
 *
 * @param {string} text 用户消息文本。
 * @return {{action: 'adopt'|'reject'|'archive', candidateId: string} | null}
 */
function parseRSIDecisionCommand(text) {
  if (!text || typeof text !== 'string') return null;
  const t = text.trim();
  if (!t) return null;
  // 候选 ID 必须以 rsi_ 开头，至少 6 字符
  const CID_RE = /(rsi_[a-f0-9]{6,})/i;
  // 关键词映射（必须出现在消息开头）
  const PATTERNS = [
    { action: 'adopt',   re: /^(?:采纳|采用|accept|adopt)\s*(?:候选)?\s*/i },
    { action: 'reject',  re: /^(?:拒绝|驳回|reject)\s*(?:候选)?\s*/i },
    { action: 'archive', re: /^(?:归档|archive)\s*(?:候选)?\s*/i },
  ];
  for (const p of PATTERNS) {
    const m = t.match(p.re);
    if (!m) continue;
    const rest = t.slice(m[0].length).trim();
    const cid = rest.match(CID_RE);
    if (cid) {
      return { action: p.action, candidateId: cid[1] };
    }
    // 兼容 "采纳 候选 rsi_xxx" 已被前缀吃掉的情况
    const cidAnywhere = t.match(CID_RE);
    if (cidAnywhere) {
      return { action: p.action, candidateId: cidAnywhere[1] };
    }
    return null;
  }
  return null;
}

/**
 * 把"建议采纳"通知发送到 aris-lounge 频道，让用户可见。
 *
 * 通知内容包含：候选 ID、目标模块、绩效对比、宪章检查报告、
 * grounding 摘要、diff 预览。用户可回复 "采纳 rsi_xxx" /
 * "拒绝 rsi_xxx" / "归档 rsi_xxx" 触发 /rsi/decide。
 *
 * 幂等：同一 candidate_id 仅通知一次（notifiedRSIIds 去重）。
 *
 * @param {object} ctx 插件上下文。
 * @param {number} sidecarPort sidecar 端口。
 * @param {object} candidate rsi_propose 返回的候选状态 dict。
 */
async function notifyRSISuggestion(ctx, sidecarPort, candidate) {
  if (!candidate || !candidate.candidate_id) return;
  if (notifiedRSIIds.has(candidate.candidate_id)) return;
  notifiedRSIIds.add(candidate.candidate_id);
  pendingRSICandidates.set(candidate.candidate_id, candidate);

  const config = ctx.config || {};
  const baseUrl = getHanakoServerUrl(ctx);
  const channelName = config.arisDefaultChannel || DEFAULT_CHANNEL_NAME;

  // 格式化通知内容（Mineradio 风格，无 emoji）
  const lines = [];
  lines.push('=== RSI 建议采纳通知 ===');
  lines.push('candidate_id: ' + candidate.candidate_id);
  lines.push('target_module: ' + (candidate.target_module || '(unknown)'));
  lines.push('fitness_signal: ' + (candidate.fitness_signal != null
    ? Number(candidate.fitness_signal).toFixed(3) : 'n/a'));
  lines.push('fitness_score: ' + (candidate.fitness_score != null
    ? Number(candidate.fitness_score).toFixed(3) : 'n/a'));
  // grounding 摘要
  const g = candidate.grounding || {};
  lines.push('grounding: ' + (g.state || 'n/a') +
    (g.confidence != null ? ' (' + Number(g.confidence).toFixed(2) + ')' : '') +
    (g.rejected ? ' [REJECTED]' : ''));
  // 宪章检查报告
  const c = candidate.charter_report || {};
  const vCount = Array.isArray(c.violations) ? c.violations.length : 0;
  lines.push('charter: ' + (c.charter_compatible ? 'compatible' : 'VIOLATED') +
    ' (' + vCount + ' violations)');
  // sandbox
  const s = candidate.sandbox_result || {};
  lines.push('sandbox: ' + (s.success ? 'passed' : 'failed') +
    (s.error ? ' [' + String(s.error).slice(0, 80) + ']' : ''));
  // diff 预览（最多 20 行）
  const diff = String(candidate.candidate_diff || '').split('\n').slice(0, 20).join('\n');
  if (diff) {
    lines.push('--- diff preview (first 20 lines) ---');
    lines.push(diff);
    lines.push('--- end diff preview ---');
  }
  lines.push('');
  lines.push('回复格式: "采纳 ' + candidate.candidate_id + '" / "拒绝 ' +
    candidate.candidate_id + '" / "归档 ' + candidate.candidate_id + '"');
  lines.push('=========================');

  const body = lines.join('\n');
  try {
    await httpRequestUrl(baseUrl, 'POST',
      '/channels/' + encodeURIComponent(channelName) + '/messages',
      { body: body });
    logger(ctx, 'RSI suggestion notified: ' + candidate.candidate_id);
  } catch (e) {
    // 频道发送失败时仅记录，不阻塞主流程
    logger(ctx, 'notifyRSISuggestion failed: ' + e.message);
  }
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function logger(ctx, msg) {
  if (ctx && ctx.log) {
    ctx.log.info('[Aris] ' + msg);
  } else {
    console.log('[Aris] ' + msg);
  }
}

// ═════════════════════════════════════════════════════════════
// hanako server HTTP（频道/DM API）
// ═════════════════════════════════════════════════════════════

/**
 * 向 hanako server 发起 HTTP 请求（与 httpRequest 区别：接受完整 baseUrl 而非 sidecar port）。
 * 用于频道/DM REST API 调用。
 */
function httpRequestUrl(baseUrl, method, reqPath, body) {
  return new Promise((resolve, reject) => {
    let parsed;
    try {
      parsed = new URL(reqPath, baseUrl);
    } catch (e) {
      return reject(e);
    }
    const options = {
      hostname: parsed.hostname,
      port: parsed.port,
      path: parsed.pathname + parsed.search,
      method: method,
      headers: { 'Content-Type': 'application/json' },
      timeout: 10000,
    };
    // v1.1：指向 sidecar (11521) 时附加 Bearer token
    if (Number(parsed.port) === 11521) {
      const token = getSidecarToken();
      if (token) options.headers['Authorization'] = `Bearer ${token}`;
    }
    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          resolve({ raw: data });
        }
      });
    });
    req.on('error', e => reject(e));
    req.on('timeout', () => { req.destroy(); reject(new Error('timeout')); });
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}

// ═════════════════════════════════════════════════════════════
// 工具注册
// ═════════════════════════════════════════════════════════════

/**
 * 注册 4 个新工具：aris_channel_send / aris_dm_send / aris_memory_query / aris_consciousness_state。
 * 若 ctx.registerTool 存在则走 SDK 注册；否则挂到 ctx._aris.tools 供外部访问。
 */
function registerArisTools(ctx, sidecarPort) {
  const config = ctx.config || {};
  const baseUrl = getHanakoServerUrl(ctx);

  const tools = {
    // ── aris_channel_send：以 Aris 身份向 hanako 频道发送消息 ──
    aris_channel_send: async (args) => {
      const a = args || {};
      const channelId = a.channelId;
      const text = a.text;
      if (!channelId || !text) {
        return { ok: false, error: 'channelId and text are required' };
      }
      try {
        // 注意：channels.ts 的实际 API 路径为 POST /channels/:name/messages，
        // 请求体字段为 { body: text }（非 content）。sender 由 server 端 engine.userName 决定。
        const res = await httpRequestUrl(
          baseUrl, 'POST',
          '/channels/' + encodeURIComponent(channelId) + '/messages',
          { body: text },
        );
        if (res && res.ok) {
          return { ok: true, timestamp: res.timestamp };
        }
        return { ok: false, error: (res && res.error) || 'unknown error' };
      } catch (e) {
        return { ok: false, error: e.message };
      }
    },

    // ── aris_dm_send：Aris 向另一 agent 发送私信 ──
    aris_dm_send: async (args) => {
      const a = args || {};
      const peerId = a.peerId;
      const text = a.text;
      if (!peerId || !text) {
        return { ok: false, error: 'peerId and text are required' };
      }
      try {
        // hanako server 当前无 POST /api/dm/:peerId 发送端点，
        // 直接追加到 agents/aris/dm/{peerId}.md（与 channel-store.appendMessage 格式一致）。
        appendDmMessage(ctx, peerId, 'aris', text);
        return { ok: true };
      } catch (e) {
        return { ok: false, error: e.message };
      }
    },

    // ── aris_memory_query：读取 Aris 的 hanako 记忆 ──
    aris_memory_query: async (args) => {
      const a = args || {};
      const scope = a.scope || 'today';
      const date = a.date;
      const query = a.query;
      const limit = a.limit;
      try {
        if (query) {
          return { content: queryFactsInline(ctx, query, limit || 10) };
        }
        const content = readMemoryInline(ctx, scope, date);
        return { content: content };
      } catch (e) {
        return { content: '', error: e.message };
      }
    },

    // ── aris_consciousness_state：返回 Aris 当前意识状态 ──
    aris_consciousness_state: async () => {
      try {
        const st = await httpRequest(sidecarPort, 'GET', '/cognitive_context');
        if (st) {
          cachedConsciousnessState = st;
          return st;
        }
        return { psi: null, emotion: null, focus: null, selfPresence: 0.81 };
      } catch (e) {
        // sidecar 不可达时返回缓存（若有）
        if (cachedConsciousnessState) return cachedConsciousnessState;
        return { ok: false, error: e.message };
      }
    },

    // ── P1-rsi-sandbox: aris_rsi_propose ──
    // Aris 调用 sidecar /rsi/propose 触发 RSI 建议模式完整闭环。
    // 仅返回 suggest_adopt 决策建议，不会自动应用 patch。
    // 当 decision === 'suggest_adopt' 时，自动通过 notifyRSISuggestion
    // 把通知推送到 aris-lounge 频道等待用户决策。
    aris_rsi_propose: async (args) => {
      const a = args || {};
      const targetModule = a.target_module || a.targetModule;
      const fitnessSignal = Number(a.fitness_signal != null
        ? a.fitness_signal : (a.fitnessSignal != null ? a.fitnessSignal : 0.5));
      if (!targetModule) {
        return { ok: false, error: 'target_module is required (must start with "laap/")' };
      }
      try {
        const result = await httpRequest(sidecarPort, 'POST', '/rsi/propose', {
          target_module: targetModule,
          fitness_signal: fitnessSignal,
          agent_name: 'aris',
        });
        // 若建议采纳，推送通知到 aris-lounge 频道
        if (result && result.decision === 'suggest_adopt') {
          await notifyRSISuggestion(ctx, sidecarPort, result);
        }
        return { ok: true, candidate: result };
      } catch (e) {
        return { ok: false, error: e.message };
      }
    },

    // ── P1-rsi-sandbox: aris_rsi_decide ──
    // Aris 调用 sidecar /rsi/decide 应用用户决策。
    // action 必须是 adopt / reject / archive 之一。
    // 幂等：重复 decide 同 (candidate_id, action) 返回相同结果。
    aris_rsi_decide: async (args) => {
      const a = args || {};
      const candidateId = a.candidate_id || a.candidateId;
      const action = a.action;
      const decidedBy = a.decided_by || a.decidedBy || 'user';
      if (!candidateId || !action) {
        return { ok: false, error: 'candidate_id and action are required' };
      }
      if (['adopt', 'reject', 'archive'].indexOf(action) === -1) {
        return { ok: false, error: 'action must be one of: adopt / reject / archive' };
      }
      try {
        const result = await httpRequest(sidecarPort, 'POST', '/rsi/decide', {
          candidate_id: candidateId,
          action: action,
          decided_by: decidedBy,
          agent_name: 'aris',
        });
        if (result && result.decided) {
          pendingRSICandidates.delete(candidateId);
        }
        return { ok: !!result && !!result.decided, decision: result };
      } catch (e) {
        return { ok: false, error: e.message };
      }
    },
  };

  ctx._aris.tools = tools;

  if (typeof ctx.registerTool === 'function') {
    for (const name of Object.keys(tools)) {
      try {
        ctx.registerTool(name, tools[name]);
      } catch (e) {
        logger(ctx, 'registerTool(' + name + ') failed: ' + e.message);
      }
    }
  }
  logger(ctx, 'Aris tools registered: ' + Object.keys(tools).join(', '));
}

// ═════════════════════════════════════════════════════════════
// 记忆查询（inline 实现，避免 TS 导入）
// ═════════════════════════════════════════════════════════════

/**
 * 内联读取 memory/{scope}.md，与 HanakoMemoryAdapter.readMemory 路径约定一致。
 */
function readMemoryInline(ctx, scope, date) {
  const config = ctx.config || {};
  const agentDir = resolveArisAgentDir(config);
  const memoryDir = path.join(agentDir, 'memory');
  let filePath;
  switch (scope) {
    case 'today':
      filePath = path.join(memoryDir, 'today.md');
      break;
    case 'daily': {
      if (!date || !/^\d{4}-\d{2}-\d{2}$/.test(date)) return '';
      filePath = path.join(memoryDir, 'daily', date + '.md');
      break;
    }
    case 'week':
      filePath = path.join(memoryDir, 'week.md');
      break;
    case 'longterm':
      filePath = path.join(memoryDir, 'longterm.md');
      break;
    case 'facts':
      filePath = path.join(memoryDir, 'facts.md');
      break;
    case 'memory':
      filePath = path.join(memoryDir, 'memory.md');
      break;
    default:
      return '';
  }
  try {
    return fs.readFileSync(filePath, 'utf-8');
  } catch (e) {
    return '';
  }
}

/**
 * 内联 FTS5 事实查询，与 HanakoMemoryAdapter.queryFacts 逻辑一致。
 * better-sqlite3 不可用时返回空数组。
 */
function queryFactsInline(ctx, query, limit) {
  const config = ctx.config || {};
  const agentDir = resolveArisAgentDir(config);
  const factsDbPath = path.join(agentDir, 'memory', 'facts.db');
  if (!query || !String(query).trim()) return [];
  if (!fs.existsSync(factsDbPath)) return [];

  let Database;
  try {
    Database = require('better-sqlite3');
  } catch (e) {
    return [];
  }
  const DbCtor = Database && (Database.default || Database);
  if (!DbCtor) return [];

  let db;
  try {
    db = new DbCtor(factsDbPath, { readonly: true });
  } catch (e) {
    return [];
  }
  try {
    const safeLimit = Math.max(1, Math.floor(Number(limit) || 10));
    const stmt = db.prepare(
      'SELECT fact, tags, created_at FROM facts ' +
      'JOIN facts_fts ON facts.rowid = facts_fts.rowid ' +
      'WHERE facts_fts MATCH ? LIMIT ?'
    );
    const rows = stmt.all(String(query), safeLimit);
    return rows.map(row => ({
      fact: String((row && row.fact) || ''),
      tags: parseTagsInline(row && row.tags),
      created_at: String((row && row.created_at) || ''),
    }));
  } catch (e) {
    return [];
  } finally {
    try { if (db && db.open) db.close(); } catch (e) { /* best effort */ }
  }
}

function parseTagsInline(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  try {
    const parsed = JSON.parse(String(raw));
    return Array.isArray(parsed) ? parsed : [];
  } catch (e) {
    return String(raw).split(',').map(s => s.trim()).filter(Boolean);
  }
}

// ═════════════════════════════════════════════════════════════
// DM 发送（直接追加 md 文件，格式与 channel-store.appendMessage 一致）
// ═════════════════════════════════════════════════════════════

function appendDmMessage(ctx, peerId, sender, body) {
  const config = ctx.config || {};
  const agentDir = resolveArisAgentDir(config);
  if (!peerId || /[\/\\]|\.\./.test(peerId)) {
    throw new Error('Invalid peerId');
  }
  const dmDir = path.join(agentDir, 'dm');
  const dmFile = path.join(dmDir, peerId + '.md');
  fs.mkdirSync(dmDir, { recursive: true });
  const ts = formatTimestampNow();
  const block = '\n### ' + sender + ' | ' + ts + '\n\n' + String(body).trim() + '\n\n---\n';
  fs.appendFileSync(dmFile, block, 'utf-8');
  return { timestamp: ts };
}

function formatTimestampNow() {
  const d = new Date();
  const y = d.getFullYear();
  const mo = String(d.getMonth() + 1).padStart(2, '0');
  const da = String(d.getDate()).padStart(2, '0');
  const h = String(d.getHours()).padStart(2, '0');
  const mi = String(d.getMinutes()).padStart(2, '0');
  const s = String(d.getSeconds()).padStart(2, '0');
  return y + '-' + mo + '-' + da + ' ' + h + ':' + mi + ':' + s;
}

// ═════════════════════════════════════════════════════════════
// 意识状态 fallback 格式化
// ═════════════════════════════════════════════════════════════

/**
 * 将缓存的意识状态格式化为单行摘要，用于 sidecar 不可达时的 fallback 注入。
 * 格式：主导需求: {dominant} ({drive:.2f}) | 焦点: {focus} | 情感: {emotion} ({intensity:.2f}) | 自我存在感: {selfPresence:.2f}
 */
function formatConsciousnessFallback(state) {
  if (!state) return '';
  const parts = [];
  if (state.psi) {
    const dominant = state.psi.dominant || '?';
    const drive = typeof state.psi.drive === 'number' ? state.psi.drive.toFixed(2) : '0.00';
    parts.push('主导需求: ' + dominant + ' (' + drive + ')');
  }
  if (state.focus) {
    const focusVal = state.focus.focus || state.focus.intent || '?';
    parts.push('焦点: ' + focusVal);
  }
  if (state.emotion) {
    const dom = state.emotion.dominant || state.emotion.dominant_emotion || '?';
    const intensity = typeof state.emotion.intensity === 'number'
      ? state.emotion.intensity.toFixed(2)
      : (typeof state.emotion.dominant_intensity === 'number'
        ? state.emotion.dominant_intensity.toFixed(2) : '0.00');
    parts.push('情感: ' + dom + ' (' + intensity + ')');
  }
  if (typeof state.selfPresence === 'number') {
    parts.push('自我存在感: ' + state.selfPresence.toFixed(2));
  }
  return parts.length ? parts.join(' | ') : '';
}

// ═════════════════════════════════════════════════════════════
// 默认频道初始化
// ═════════════════════════════════════════════════════════════

/**
 * 确保 hanako server 中存在默认频道（如 #aris-lounge），不存在则创建，
 * 并把频道 id 追加到 agents/aris/channels.md 书签。
 */
async function ensureDefaultChannel(ctx) {
  const config = ctx.config || {};
  const baseUrl = getHanakoServerUrl(ctx);
  const channelName = config.arisDefaultChannel || DEFAULT_CHANNEL_NAME;
  const agentDir = resolveArisAgentDir(config);

  // GET /channels 列出现有频道
  let listRes;
  try {
    listRes = await httpRequestUrl(baseUrl, 'GET', '/channels', null);
  } catch (e) {
    logger(ctx, 'ensureDefaultChannel: GET /channels failed: ' + e.message);
    return;
  }
  const channels = (listRes && listRes.channels) || [];
  const existing = channels.find(ch => ch.name === channelName || ch.id === channelName);

  if (existing) {
    logger(ctx, 'Default channel #' + channelName + ' already exists');
    ensureChannelBookmark(ctx, agentDir, existing.id);
    return;
  }

  // POST /channels 创建频道
  try {
    const createRes = await httpRequestUrl(baseUrl, 'POST', '/channels', {
      name: channelName,
      description: 'Aris 默认频道 — LAAP 数字生命体群聊空间',
      members: ['aris'],
      intro: 'Aris 是 LAAP 数字生命体，可被@提及。这是 Aris 的默认频道，可以在这里和 Aris 群聊。',
    });
    logger(ctx, 'Default channel #' + channelName + ' created');
    if (createRes && createRes.id) {
      ensureChannelBookmark(ctx, agentDir, createRes.id);
    }
  } catch (e) {
    logger(ctx, 'ensureDefaultChannel: POST /channels failed: ' + e.message);
  }
}

/**
 * 把频道 id 追加到 agents/{agentDir}/channels.md 书签（若尚不存在）。
 */
function ensureChannelBookmark(ctx, agentDir, channelId) {
  if (!channelId) return;
  const bookmarkPath = path.join(agentDir, 'channels.md');
  let content;
  try {
    content = fs.readFileSync(bookmarkPath, 'utf-8');
  } catch (e) {
    content = '# Channels\n';
  }
  if (content.indexOf(channelId) !== -1) return;
  const prefix = content && !content.endsWith('\n') ? '\n' : '';
  const line = prefix + '\n- [' + channelId + ']';
  try {
    fs.writeFileSync(bookmarkPath, content + line, 'utf-8');
    logger(ctx, 'channels.md bookmark appended: ' + channelId);
  } catch (e) {
    logger(ctx, 'ensureChannelBookmark failed: ' + e.message);
  }
}
