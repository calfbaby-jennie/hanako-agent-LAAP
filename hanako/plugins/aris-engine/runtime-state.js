let cognitiveContextInjectionEnabled = () => true;

export function setCognitiveContextInjectionResolver(resolver) {
  cognitiveContextInjectionEnabled = typeof resolver === 'function'
    ? resolver
    : () => true;
}

export function isCognitiveContextInjectionEnabled() {
  try {
    return cognitiveContextInjectionEnabled() !== false;
  } catch {
    // A malformed or temporarily unavailable config store must not interrupt a
    // conversation. Preserve the manifest default (enabled).
    return true;
  }
}
