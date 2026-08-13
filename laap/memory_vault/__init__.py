"""LAAP per-agent Memory Vault public API."""
from .vault_manager import (
    VAULT_DIR,
    VaultManager,
    _SQLCIPHER_AVAILABLE,
    _open_vault_connection,
    vault_manager,
)

__all__ = [
    "VAULT_DIR", "VaultManager", "_SQLCIPHER_AVAILABLE",
    "_open_vault_connection", "vault_manager",
]
