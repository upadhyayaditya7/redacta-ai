"""Tests for the encrypted vault and text redaction roundtrip."""

import pytest

from core.entities import Action, EntityType
from core.pipeline import RedactionPipeline
from core.redact import redact_text
from core.vault import Vault, VaultError
from utils.samples import document


@pytest.fixture()
def vault(tmp_path):
    v = Vault(tmp_path / "vault.json")
    v.create("correct horse battery staple")
    return v


def test_store_reveal_roundtrip(vault):
    token = vault.store("2345 6789 1234")
    assert token.startswith("«RDCT-")
    assert vault.reveal(token) == "2345 6789 1234"


def test_wrong_passphrase_rejected(tmp_path, vault):
    token = vault.store("secret")
    v2 = Vault(tmp_path / "vault.json")
    with pytest.raises(VaultError):
        v2.load("wrong-passphrase")


def test_tamper_detection(tmp_path, vault):
    import json

    token = vault.store("sensitive-value")
    raw = json.loads((tmp_path / "vault.json").read_text())
    raw["entries"][token]["ct"] = "ff" * (len(raw["entries"][token]["ct"]) // 2)
    (tmp_path / "vault.json").write_text(json.dumps(raw))
    v2 = Vault(tmp_path / "vault.json")
    v2.load("correct horse battery staple")
    with pytest.raises(VaultError):
        v2.reveal(token)


def test_redact_then_reveal(vault):
    text = document()
    report = RedactionPipeline(use_ner=False).analyze(text)
    result = redact_text(text, report, vault)
    assert result.redacted_count > 0
    assert result.vaulted_count > 0
    # No raw PII remains for vaulted types:
    for token, original in result.token_map.items():
        assert original not in result.redacted_text
        assert vault.reveal(token) == original


def test_irreversible_when_no_vault():
    text = "Email me at a@b.com"
    report = RedactionPipeline(use_ner=False).analyze(text)
    result = redact_text(text, report, vault=None)
    assert "a@b.com" not in result.redacted_text
    assert "█" in result.redacted_text


def test_vault_action_defaults():
    from core.entities import spec

    assert spec(EntityType.AADHAAR).action is Action.VAULT
    assert spec(EntityType.PASSWORD).action is Action.IRREVERSIBLE


# --- regression: a bad passphrase must never destroy the vault ------------


def test_wrong_passphrase_never_recreates_the_vault(tmp_path):
    """The UI used to fall back to create() on any unlock error. A typo then
    replaced the real vault, orphaning every token it had issued."""
    path = tmp_path / "vault.json"
    vault = Vault(path)
    vault.create("real-pass")
    token = vault.store("2345 6789 1234")

    with pytest.raises(VaultError):
        Vault(path).load("typo-pass")

    reopened = Vault(path)
    reopened.load("real-pass")
    assert reopened.reveal(token) == "2345 6789 1234"


def test_failed_load_leaves_a_fresh_instance_locked(tmp_path):
    """A bad passphrase must not leave the object half-initialised with a wrong
    key, which would silently write unreadable entries on the next store()."""
    path = tmp_path / "vault.json"
    Vault(path).create("real-pass")

    fresh = Vault(path)
    with pytest.raises(VaultError):
        fresh.load("typo-pass")
    assert not fresh.unlocked
    with pytest.raises(VaultError):
        fresh.store("anything")


def test_create_backs_up_the_previous_vault(tmp_path):
    path = tmp_path / "vault.json"
    vault = Vault(path)
    vault.create("first")
    token = vault.store("kept-value")

    vault.create("second")  # deliberate overwrite

    backup = tmp_path / "vault.json.bak"
    assert backup.exists()
    old = Vault(backup)
    old.load("first")
    assert old.reveal(token) == "kept-value"


def test_store_dedupes_identical_values(vault):
    """One value must map to one placeholder, not one per occurrence."""
    first = vault.store("same-value")
    second = vault.store("same-value")
    assert first == second
    assert vault.tokens() == [first]

