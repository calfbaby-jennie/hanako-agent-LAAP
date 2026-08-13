#!/bin/sh
set -eu
IMAGE="${LAAP_ZONE2_IMAGE:-python:3.11-laap-zone2-full}"
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
exec docker run --rm --network none \
  -e PYTHONPATH=/workspace \
  -e LAAP_VAULT_DIR=/tmp/vault \
  -e LAAP_DISABLE_RUST=1 \
  -v "$ROOT:/workspace:ro" \
  -w /workspace \
  "$IMAGE" python -m pytest -q -p no:cacheprovider \
  laap/integrations/test_full_stack_runtime.py \
  laap/tests/test_hana_psi_contract.py \
  laap/tests/test_hana_psi_rsi_safety.py \
  laap/protocol/test_1v1_protocol.py \
  laap/protocol/test_p2p_relay.py \
  laap/protocol/test_trio_chatroom.py \
  laap/agi/test_world_model_mcp.py \
  laap/agi/test_causal_mcp.py \
  laap/skills/test_pack_manager.py \
  laap/skills/test_sync.py
