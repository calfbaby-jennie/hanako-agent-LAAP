/**
 * Aris cognitive-context injection — Pi SDK extension
 *
 * 每轮用户输入前（before_agent_start）调用 sidecar /cognitive_context，
 * 把 Aris 的认知状态（PSI 需求 / 情绪 / 注意力 / 好奇心 / 自我在场）
 * 作为 turn-scoped 上下文附加到当轮用户消息，不修改 cache-stable system prompt。
 *
 * 附带：
 *  - fire-and-forget 调 /world/perceive，把对话轮次喂给世界模型
 *  - 解析 RSI 决策指令（采纳/拒绝/归档 rsi_xxx）并调 /rsi/decide
 *
 * 纪律：sidecar 不可达时静默返回，绝不中断对话。
 */

import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import http from 'node:http';
import { isCognitiveContextInjectionEnabled } from '../runtime-state.js';

const SIDECAR_HOST = '127.0.0.1';
const SIDECAR_PORT = Number(process.env.ARIS_SIDECAR_PORT || 11521);
const LAAP_HOME = process.env.LAAP_HOME || join(homedir(), '.laap');
const TOKEN_PATH = join(LAAP_HOME, 'state', 'aris-sidecar', 'aris-sidecar.token');

let cachedToken = null;

const COGNITIVE_CONTEXT_LABEL = '[Aris 认知状态（仅本轮上下文）]';

/**
 * Attach volatile cognition to the latest user message seen by the provider.
 * The context hook receives a deep copy, so this never mutates session history
 * or the cache-stable system prompt.
 */
export function injectTurnScopedCognitiveContext(messages, context) {
  if (!Array.isArray(messages)) return messages;
  const normalizedContext = String(context || '').trim();
  if (!normalizedContext) return messages;

  let userMessageIndex = -1;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    if (messages[index]?.role === 'user') {
      userMessageIndex = index;
      break;
    }
  }
  if (userMessageIndex < 0) return messages;

  const sourceMessage = messages[userMessageIndex];
  const contextBlock = `${COGNITIVE_CONTEXT_LABEL}\n${normalizedContext}`;
  let content;
  if (typeof sourceMessage.content === 'string') {
    content = `${contextBlock}\n\n${sourceMessage.content}`;
  } else if (Array.isArray(sourceMessage.content)) {
    content = [{ type: 'text', text: contextBlock }, ...sourceMessage.content];
  } else {
    return messages;
  }

  const nextMessages = [...messages];
  nextMessages[userMessageIndex] = { ...sourceMessage, content };
  return nextMessages;
}

function sessionIdentity(ctx) {
  try {
    const sessionPath = ctx?.sessionManager?.getSessionFile?.();
    if (sessionPath) return String(sessionPath);
  } catch {
    // Best effort only.
  }
  return 'hana-session';
}

function messageText(message) {
  const content = message?.content;
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .filter((block) => block && block.type === 'text')
    .map((block) => String(block.text || ''))
    .join('\n')
    .trim();
}

function boundedToolResult(event) {
  const content = Array.isArray(event?.content)
    ? event.content.map((block) => {
        if (block?.type === 'text') return { type: 'text', text: String(block.text || '').slice(0, 4000) };
        return { type: String(block?.type || 'unknown') };
      })
    : [];
  return {
    tool: String(event?.toolName || event?.name || 'unknown'),
    status: event?.isError ? 'error' : 'success',
    content,
  };
}

function getToken() {
  if (process.env.ARIS_SIDECAR_TOKEN) return process.env.ARIS_SIDECAR_TOKEN;
  if (cachedToken) return cachedToken;
  try {
    cachedToken = readFileSync(TOKEN_PATH, 'utf-8').trim();
  } catch {
    cachedToken = '';
  }
  return cachedToken;
}

function sidecarRequest(method, path, body) {
  return new Promise((resolve) => {
    const token = getToken();
    const payload = body !== undefined ? JSON.stringify(body) : null;
    const req = http.request(
      {
        hostname: SIDECAR_HOST,
        port: SIDECAR_PORT,
        path,
        method,
        timeout: 3000,
        headers: {
          'Content-Type': 'application/json',
          ...(payload !== null ? { 'Content-Length': Buffer.byteLength(payload) } : {}),
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
      },
      (res) => {
        let data = '';
        res.on('data', (c) => { data += c; });
        res.on('end', () => {
          try {
            resolve(JSON.parse(data));
          } catch {
            resolve(null);
          }
        });
      },
    );
    req.on('error', () => resolve(null));
    req.on('timeout', () => { req.destroy(); resolve(null); });
    if (payload !== null) req.write(payload);
    req.end();
  });
}

/** 解析 "采纳 rsi_xxx" / "拒绝 rsi_xxx" / "归档 rsi_xxx" */
function parseRSIDecision(text) {
  const m = /(采纳|拒绝|归档)\s+rsi_([a-f0-9]+)/i.exec(text || '');
  if (!m) return null;
  const actionMap = { 采纳: 'adopt', 拒绝: 'reject', 归档: 'archive' };
  return { action: actionMap[m[1]], candidateId: m[2] };
}

export default function (pi) {
  // Extension factories are instantiated per session. Keep turn state inside
  // the factory to prevent cognition from leaking across concurrent sessions.
  let pendingTurn = null;

  pi.on('before_agent_start', async (event, ctx) => {
    try {
      const input = String(event.prompt || '').slice(0, 12000);
      const sessionId = sessionIdentity(ctx);
      const continuingTurn = Boolean(
        input && pendingTurn
        && pendingTurn.input === input
        && pendingTurn.sessionId === sessionId
      );

      // before_agent_start may fire again after an intermediate tool call.
      // Preserve the existing turn instead of creating duplicate Ao decisions.
      if (input && !continuingTurn) {
        const turnId = `hana_${Date.now()}_${Math.random().toString(16).slice(2, 10)}`;
        pendingTurn = { input, sessionId, turnId, toolResults: [] };
        const perceived = await sidecarRequest('POST', '/perceive', {
          text: input,
          source: 'hana-extension',
          session_id: sessionId,
          turn_id: turnId,
        });
        if (pendingTurn) {
          pendingTurn.traceId = perceived?.ao_decision?.trace_id || null;
          pendingTurn.inputEventId = perceived?.canonical_event_id || null;
        }
      }
      const state = isCognitiveContextInjectionEnabled()
        ? await sidecarRequest('GET', '/cognitive_context')
        : null;
      if (pendingTurn && state && state.context) {
        pendingTurn.cognitiveContext = String(state.context);
      }

      // RSI 决策指令：命中则调 sidecar /rsi/decide
      const decision = parseRSIDecision(event.prompt);
      if (decision) {
        await sidecarRequest('POST', '/rsi/decide', {
          candidate_id: decision.candidateId,
          action: decision.action,
          decided_by: 'user',
          agent_name: 'aris',
        });
      }

      // 世界模型：本轮回合事件喂给 /world/perceive（fire-and-forget）
      sidecarRequest('POST', '/world/perceive', {
        agent_name: 'aris',
        event: {
          type: 'conversation_turn',
          entity: 'aris',
          to_state: 'responding',
          metadata: { prompt: String(event.prompt || '').slice(0, 200) },
        },
      }).catch(() => {});

    } catch {
      // sidecar 不可达：静默，不中断对话
    }
    return undefined;
  });

  // Dynamic cognition belongs to the turn message layer, not the stable
  // system-prompt/cache-prefix layer. `context` transformations are ephemeral:
  // Pi applies them to a deep copy immediately before each provider request.
  pi.on('context', (event) => {
    const cognition = pendingTurn?.cognitiveContext;
    if (!cognition || !isCognitiveContextInjectionEnabled()) return undefined;
    return {
      messages: injectTurnScopedCognitiveContext(event.messages, cognition),
    };
  });

  pi.on('tool_result', (event) => {
    if (!pendingTurn) return undefined;
    pendingTurn.toolResults.push(boundedToolResult(event));
    return undefined;
  });

  pi.on('message_end', async (event) => {
    if (!pendingTurn || event?.message?.role !== 'assistant') return undefined;
    const stopReason = String(event.message?.stopReason || '').toLowerCase();
    const response = messageText(event.message);
    // Intermediate assistant messages that request tools are part of the same
    // turn. Keep pendingTurn until a final textual assistant message arrives.
    if (!response || stopReason === 'tooluse' || stopReason === 'tool_use') {
      return undefined;
    }
    const turn = pendingTurn;
    pendingTurn = null;
    const success = stopReason !== 'error'
      && !turn.toolResults.some((item) => item.status === 'error');
    await sidecarRequest('POST', '/after_turn', {
      user_input: turn.input,
      response: response.slice(0, 24000),
      success: success ? 1.0 : 0.0,
      tool_results: turn.toolResults,
      session_id: turn.sessionId,
      turn_id: turn.turnId,
      trace_id: turn.traceId,
      input_event_id: turn.inputEventId,
      source: 'hana-extension',
    });
    return undefined;
  });
}
