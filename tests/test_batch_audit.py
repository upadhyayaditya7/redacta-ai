"""Tests for the audit trail and batch mode (M1 'upscale' features)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.audit import build_record
from core.batch import process_folder
from core.pipeline import RedactionPipeline
from core.redact import redact_text
from core.vault import Vault
from utils.samples import document


def _fresh_vault(tmp_path: Path) -> Vault:
    vault = Vault(path=tmp_path / "vault.json")
    vault.create("test-pass-123")
    vault.load("test-pass-123")
    return vault


def test_audit_record_hides_values(tmp_path: Path) -> None:
    doc = document()
    pipeline = RedactionPipeline(use_ner=False)
    report = pipeline.analyze(doc)
    vault = _fresh_vault(tmp_path)
    result = redact_text(doc, report, vault)

    rec = build_record(
        source="demo.txt", report=report, result=result,
        vault_hints={t: vault.entry_hint(t) for t in result.token_map},
    )
    payload = json.loads(rec.to_json())

    assert payload["redacted_count"] == len(report.spans)
    assert payload["vaulted_count"] == result.vaulted_count
    assert payload["engine"] == "regex"
    # No raw PII may leak into the audit file.
    flat = json.dumps(payload)
    for span in report.spans:
        assert span.text not in flat, f"raw value leaked: {span.text}"
    # Vaulted entries carry a token + hint; irreversible ones don't.
    vaulted = [e for e in payload["entities"] if e["action"] == "vaulted"]
    assert vaulted, "expected vaulted entries"
    assert all(e["token"] and "•" in e["hint"] for e in vaulted)
    irreversible = [e for e in payload["entities"] if e["action"] == "irreversible"]
    assert irreversible and all(e["token"] is None for e in irreversible)


def test_batch_folder(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.txt").write_text(document(), encoding="utf-8")
    (inbox / "b.md").write_text(document(), encoding="utf-8")
    (inbox / "ignore.exe").write_bytes(b"MZ...")  # unsupported -> reported as error

    pipeline = RedactionPipeline(use_ner=False)
    items = process_folder(inbox, pipeline, None)

    out_dir = tmp_path / "inbox_redacted"
    assert out_dir.is_dir()
    done = {it.src.name: it for it in items}
    assert set(done) == {"a.txt", "b.md", "ignore.exe"}
    assert done["a.txt"].error is None
    assert done["b.md"].error is None
    assert done["ignore.exe"].error  # unsupported type reported, not crashed
    redacted = (out_dir / "a.redacted.txt").read_text(encoding="utf-8")
    assert "«RDCT-" not in redacted  # no vault passed -> irreversible masking
    assert "█" in redacted
