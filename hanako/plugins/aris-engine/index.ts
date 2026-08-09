// Aris Engine lifecycle entry for HanaAgent's ESM plugin runtime.
// Conversation cognition is implemented by extensions/cognitive-context.js;
// tools, routes, and the page are loaded as declarative contributions.
import { setCognitiveContextInjectionResolver } from './runtime-state.js';

export default class ArisEnginePlugin {
  declare ctx: any;

  async onload() {
    // Resolve on every turn so a settings change takes effect without keeping a
    // stale copy inside the independently loaded Pi extension factory.
    setCognitiveContextInjectionResolver(
      () => this.ctx.config?.get?.('arisInjectContext') !== false,
    );
    this.ctx.log.info('Aris Engine lifecycle loaded; cognitive extension active');
  }

  async onunload() {
    setCognitiveContextInjectionResolver(null);
    this.ctx.log.info('Aris Engine lifecycle unloaded');
  }
}
