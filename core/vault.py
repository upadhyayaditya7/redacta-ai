"""Local encrypted vault: reversible redaction with a master key.

Design notes
------------
* Pure standard library — no third-party crypto dependency, so the demo
  runs anywhere (and the supply chain stays auditable).
* Confidentiality: SHA-256-CTR style stream cipher (keystream blocks are
  ``SHA256(nonce || counter || key)``) with a random 128-bit nonce.
* Integrity: HMAC-SHA256 over (nonce || ciphertext).
* Key: PBKDF2-HMAC-SHA256, 200k iterations, per-vault random salt.
* Vault file lives in ``~/.redacta/vault.json``; nothing ever leaves the
  machine (no cloud, no telemetry).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path

from .config import VAULT_PATH

_PBKDF2_ITERS = 200_000
_NONCE_LEN = 16
_SALT_LEN = 16


class VaultError(Exception):
    pass


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        block = hashlib.sha256(nonce + counter.to_bytes(8, "big") + key).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _hint(value: str) -> str:
    if len(value) <= 4:
        return "•" * len(value)
    return value[:2] + "•" * (len(value) - 4) + value[-2:]


class Vault:
    """Encrypted store mapping placeholders back to original secrets."""

    def __init__(self, path: Path = VAULT_PATH) -> None:
        self.path = path
        self._salt = b""
        self._key = b""
        self._entries: dict[str, dict] = {}

    # -- lifecycle ---------------------------------------------------

    def exists(self) -> bool:
        return self.path.exists()

    def create(self, passphrase: str) -> None:
        """Create a brand-new vault (caller decides about overwriting)."""
        self._salt = secrets.token_bytes(_SALT_LEN)
        self._key = hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode(), self._salt, _PBKDF2_ITERS
        )
        self._entries = {}
        self._save()

    def load(self, passphrase: str) -> None:
        if not self.path.exists():
            raise VaultError(f"No vault found at {self.path}")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._salt = bytes.fromhex(raw["salt"])
        self._key = hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode(), self._salt, _PBKDF2_ITERS
        )
        if not hmac.compare_digest(self._hmac(self._salt), bytes.fromhex(raw["verifier"])):
            raise VaultError("Wrong passphrase")
        self._entries = raw["entries"]

    # -- entries ------------------------------------------------------

    def store(self, value: str) -> str:
        """Encrypt *value* and return its placeholder token."""
        token = "«RDCT-" + secrets.token_hex(4).upper() + "»"
        nonce = secrets.token_bytes(_NONCE_LEN)
        ct = self._crypt(nonce, value.encode("utf-8"))
        self._entries[token] = {
            "nonce": nonce.hex(),
            "ct": ct.hex(),
            "mac": self._hmac(nonce + ct).hex(),
            "hint": _hint(value),
        }
        self._save()
        return token

    def reveal(self, token: str) -> str:
        """Decrypt a placeholder back to the original value."""
        if token not in self._entries:
            raise VaultError(f"Unknown token: {token}")
        e = self._entries[token]
        nonce, ct = bytes.fromhex(e["nonce"]), bytes.fromhex(e["ct"])
        if not hmac.compare_digest(self._hmac(nonce + ct), bytes.fromhex(e["mac"])):
            raise VaultError("Vault integrity check failed (corrupted or wrong key)")
        return self._crypt(nonce, ct).decode("utf-8")

    def tokens(self) -> list[str]:
        return sorted(self._entries)

    def entry_hint(self, token: str) -> str:
        return self._entries.get(token, {}).get("hint", "?")

    # -- internals ------------------------------------------------------

    def _crypt(self, nonce: bytes, data: bytes) -> bytes:
        return _xor_bytes(data, _keystream(self._key, nonce, len(data)))

    def _hmac(self, data: bytes) -> bytes:
        return hmac.new(self._key, data, hashlib.sha256).digest()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "salt": self._salt.hex(),
            "verifier": self._hmac(self._salt).hex(),
            "entries": self._entries,
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
