// ES module compatibility bridge for the legacy CommonJS implementation.
// Hana resolves index.ts first; this file remains a safe fallback if the
// TypeScript lifecycle entry is unavailable during packaging or migration.
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const legacy = require('./index.cjs');

export const activate = legacy.activate;
export const deactivate = legacy.deactivate;
export const onload = legacy.onload;
export const onunload = legacy.onunload;
