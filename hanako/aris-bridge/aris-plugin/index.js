// Compatibility entry only. The single Aris plugin implementation lives at
// hanako/plugins/aris-engine. Keep this historical path as a thin re-export so
// migration tooling cannot resurrect an independent system-prompt injector.
export { default } from '../../plugins/aris-engine/index.ts';
