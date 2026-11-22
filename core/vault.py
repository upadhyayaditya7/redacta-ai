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
* Writes are atomic (``tmp`` + ``os.replace``) and the file is chmod 0600,
  so an interrupted save cannot truncate the vault and a crash cannot leave
  a half-written one behind.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
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
        # value-id (keyed HMAC of the plaintext) -> token, so the same value
        # always gets the same placeholder instead of a new one per occurrence.
        self._index: dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------

    def exists(self) -> bool:
        return self.path.exists()

    @property
    def unlocked(self) -> bool:
        return bool(self._key)

    def create(self, passphrase: str) -> None:
        """Create a brand-new vault (caller decides about overwriting).

        An existing vault is copied to ``<name>.bak`` first, so creating over a
        vault can never silently destroy the only copy of recoverable data.
        """
        if self.path.exists():
            self._backup()
        self._salt = secrets.token_bytes(_SALT_LEN)
        self._key = hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode(), self._salt, _PBKDF2_ITERS
        )
        self._entries = {}
        self._index = {}
        self._save()

    def load(self, passphrase: str) -> None:
        if not self.path.exists():
            raise VaultError(f"No vault found at {self.path}")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        salt = bytes.fromhex(raw["salt"])
        # Derive into a local first: a wrong passphrase must leave the object
        # exactly as it was, never half-initialised with the wrong key.
        key = hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode(), salt, _PBKDF2_ITERS
        )
        if not hmac.compare_digest(
            hmac.new(key, salt, hashlib.sha256).digest(),
            bytes.fromhex(raw["verifier"]),
        ):
            raise VaultError("Wrong passphrase")

        self._salt = salt
        self._key = key
        self._entries = raw["entries"]
        index = raw.get("index")
        self._index = dict(index) if isinstance(index, dict) else {}
        if not self._index:
            self._rebuild_index()

    # -- entries ------------------------------------------------------

    def store(self, value: str) -> str:
        """Encrypt *value* and return its placeholder token.

        Repeated values reuse their existing token, which keeps the shared
        document internally consistent (the same Aadhaar is always the same
        placeholder) and stops the vault growing one entry per occurrence.
        """
        if not self._key:
            raise VaultError("Vault is locked — call create() or load() first")
        vid = self._value_id(value)
        existing = self._index.get(vid)
        if existing is not None and existing in self._entries:
            return existing

        token = "«RDCT-" + secrets.token_hex(4).upper() + "»"
        nonce = secrets.token_bytes(_NONCE_LEN)
        ct = self._crypt(nonce, value.encode("utf-8"))
        self._entries[token] = {
            "nonce": nonce.hex(),
            "ct": ct.hex(),
            "mac": self._hmac(nonce + ct).hex(),
            "hint": _hint(value),
        }
        self._index[vid] = token
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

    def _value_id(self, value: str) -> str:
        """Keyed fingerprint of a plaintext, safe to persist (needs the key)."""
        return hmac.new(
            self._key, b"vid:" + value.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _rebuild_index(self) -> None:
        """Derive the dedupe index for a vault written before it existed."""
        for token, entry in self._entries.items():
            try:
                value = self.reveal(token)
            except VaultError:
                continue
            self._index[self._value_id(value)] = token

    def _crypt(self, nonce: bytes, data: bytes) -> bytes:
        return _xor_bytes(data, _keystream(self._key, nonce, len(data)))

    def _hmac(self, data: bytes) -> bytes:
        return hmac.new(self._key, data, hashlib.sha256).digest()

    def _backup(self) -> None:
        """Keep a copy of the vault we are about to replace (best effort)."""
        try:
            self.path.with_name(self.path.name + ".bak").write_bytes(
                self.path.read_bytes()
            )
        except OSError:
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "salt": self._salt.hex(),
            "verifier": self._hmac(self._salt).hex(),
            "entries": self._entries,
            "index": self._index,
        }
        tmp = self.path.with_name(self.path.name + ".tmp")
        # ensure_ascii keeps the file pure ASCII: the «RDCT-» tokens then survive
        # any reader that opens the vault without declaring UTF-8 (Windows
        # defaults to cp1252), at a negligible size cost.
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic: never a truncated vault
        try:
            os.chmod(self.path, 0o600)
        except OSError:  # Windows uses ACLs; POSIX perms are best effort
            pass
