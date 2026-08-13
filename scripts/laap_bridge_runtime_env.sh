#!/bin/bash
# Runtime environment shared by the launchd guardian and manual bridge starts.
# Secrets are retrieved from macOS Keychain and never printed or stored here.

export HERMES_ROOT="${HERMES_ROOT:-$HOME/.hermes/hermes-agent}"
export LAAP_RUST_LIB_DIR="${LAAP_RUST_LIB_DIR:-/Users/daozhu/aris_home/OH-WorkSpace/laap_core_src/target/release}"
export PATH="$HOME/.local/bin:$HERMES_ROOT:/usr/local/bin:/opt/homebrew/bin:$PATH"
export PYTHONPATH="$LAAP_RUST_LIB_DIR:$HERMES_ROOT:${PYTHONPATH:-}"

_LAAP_BUS_KEY="$(security find-generic-password -a "$USER" -s com.aris.cognitive-bus -w 2>/dev/null || true)"
if [ -n "$_LAAP_BUS_KEY" ]; then
  export LAAP_BUS_PSK="$_LAAP_BUS_KEY"
fi
unset _LAAP_BUS_KEY
