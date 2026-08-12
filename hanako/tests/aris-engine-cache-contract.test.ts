import fs from 'node:fs';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { injectTurnScopedCognitiveContext } from '../plugins/aris-engine/extensions/cognitive-context.js';
import {
  buildLlmContextCachePrefixContract,
  diffCachePrefixContracts,
} from '../lib/llm/cache-prefix-contract.ts';
import {
  assertSessionSnapshotRequest,
  buildSessionCacheSnapshot,
  buildSessionSnapshotRequestContract,
} from '../core/session-cache-snapshot.ts';
import {
  isCognitiveContextInjectionEnabled,
  setCognitiveContextInjectionResolver,
} from '../plugins/aris-engine/runtime-state.js';

describe('Aris cognitive context cache contract', () => {
  it('keeps the extension source out of the cache-stable system-prompt layer', () => {
    const extensionSource = fs.readFileSync(
      path.join(process.cwd(), 'plugins/aris-engine/extensions/cognitive-context.js'),
      'utf8',
    );

    expect(extensionSource).toContain("pi.on('context'");
    expect(extensionSource).not.toMatch(/\bsystemPrompt\s*:/);
  });

  it('preserves the complete Hana↔Ao turn lifecycle in the extension source', () => {
    const extensionSource = fs.readFileSync(
      path.join(process.cwd(), 'plugins/aris-engine/extensions/cognitive-context.js'),
      'utf8',
    );

    expect(extensionSource).toContain("pi.on('before_agent_start'");
    expect(extensionSource).toContain("pi.on('context'");
    expect(extensionSource).toContain("pi.on('tool_result'");
    expect(extensionSource).toContain("pi.on('message_end'");
    expect(extensionSource).toContain("'/cognitive/preflight'");
    expect(extensionSource).toContain("'/cognitive/review'");
    expect(extensionSource).toContain("'/after_turn'");
    expect(extensionSource).toContain('PSI_STRICT');
    expect(extensionSource).toContain('visibleResponse');
    expect(extensionSource).toContain('psiReview');
    expect(extensionSource).toContain('}, 75000);');
    expect(extensionSource).toContain("'Content-Length'");
  });

  it('injects volatile cognition into the turn message without changing the stable prefix', () => {
    const messages = [
      {
        role: 'user',
        content: [{ type: 'text', text: '继续升级项目' }],
        timestamp: 1,
      },
    ];
    const originalMessages = structuredClone(messages);
    const stablePrefix = buildLlmContextCachePrefixContract({
      model: { id: 'gpt-5.6-sol', provider: 'openai-codex' },
      systemPrompt: 'frozen session prompt',
      tools: [{ name: 'read', description: 'Read a file', parameters: {} }],
    });

    const providerMessages = injectTurnScopedCognitiveContext(
      messages,
      'dominant=curiosity focus=integration',
    );
    const requestPrefix = buildLlmContextCachePrefixContract({
      model: { id: 'gpt-5.6-sol', provider: 'openai-codex' },
      systemPrompt: 'frozen session prompt',
      tools: [{ name: 'read', description: 'Read a file', parameters: {} }],
    });

    expect(diffCachePrefixContracts(stablePrefix, requestPrefix)).toEqual([]);
    expect(messages).toEqual(originalMessages);
    expect(providerMessages).not.toBe(messages);
    expect(providerMessages[0].content[0].text).toContain('[Aris 认知状态（仅本轮上下文）]');
    expect(providerMessages[0].content[1].text).toBe('继续升级项目');
  });

  it('keeps session_restore → first request cache prefix frozen', () => {
    const model = { id: 'gpt-5.6-sol', provider: 'openai-codex' };
    const tools = [{ name: 'read', description: 'Read a file', parameters: {} }];
    const restoredHistory = [
      { role: 'user', content: '旧问题', timestamp: 1 },
      { role: 'assistant', content: '旧回答', timestamp: 2 },
    ];
    const snapshot = buildSessionCacheSnapshot({
      sessionPath: '/sessions/restored.jsonl',
      model,
      systemPrompt: 'frozen session prompt',
      tools,
      messages: restoredHistory,
      reason: 'session_restore',
    });
    const firstRequestMessages = injectTurnScopedCognitiveContext(
      [...restoredHistory, { role: 'user', content: '恢复后的第一条消息', timestamp: 3 }],
      'dominant=safety focus=integration',
    );
    const requestContract = buildSessionSnapshotRequestContract({
      snapshot,
      model,
      systemPrompt: 'frozen session prompt',
      tools,
      messages: firstRequestMessages,
      prefixMessageCount: snapshot.messageCount,
    });

    expect(assertSessionSnapshotRequest(snapshot, requestContract).diffs).toEqual([]);
    expect(requestContract.systemPromptHash).toBe(snapshot.systemPromptHash);
    expect(requestContract.cachePrefixHash).toBe(snapshot.cachePrefixHash);
    expect(firstRequestMessages.at(-1)?.content).toContain('[Aris 认知状态（仅本轮上下文）]');
  });

  it('honors the runtime arisInjectContext resolver', () => {
    setCognitiveContextInjectionResolver(() => false);
    expect(isCognitiveContextInjectionEnabled()).toBe(false);
    setCognitiveContextInjectionResolver(() => true);
    expect(isCognitiveContextInjectionEnabled()).toBe(true);
    setCognitiveContextInjectionResolver(null);
  });

  it('leaves messages untouched when there is no usable cognition or user message', () => {
    const assistantOnly = [{ role: 'assistant', content: [], timestamp: 1 }];

    expect(injectTurnScopedCognitiveContext(assistantOnly, 'state')).toBe(assistantOnly);
    expect(injectTurnScopedCognitiveContext(assistantOnly, '')).toBe(assistantOnly);
  });
});
